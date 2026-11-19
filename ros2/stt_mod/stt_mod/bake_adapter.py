#!/usr/bin/env python3
"""Bake a NeMo encoder adapter into a sherpa-onnx Nemotron export.

The dorai fine-tuned models (e.g. ``mc-ref``, ``mc-das`` from the stt repo) are
NeMo ``LinearAdapter`` modules trained on top of a *frozen*
nvidia/nemotron-speech-streaming-en-0.6b. Each of the 24 Conformer blocks gets

    y <- y + up(silu(down(LayerNorm(y))))          (dim 32, 1.62 M params)

right after the block's final LayerNorm (``norm_out``). Because the base is
frozen, the adapted model is the stock export plus 24 small residual MLPs. This
tool splices those MLPs into ``encoder*.onnx`` after every
``/layers.<i>/norm_out/LayerNormalization`` node, in fp32 (the adapter costs
~0.1 % of the encoder's FLOPs). Decoder, joiner and tokens are unchanged and
are copied over, so the output directory is a drop-in sherpa-onnx model dir.

Only valid if the export and the adapter share a base checkpoint. The tool
checks this: the export's fp32 LayerNorm weights must equal the base
``model.safetensors`` weights when ``--base`` is given (bit-identical for the
2026-04-25 k2-fsa int8 exports and the HF checkpoint the adapters were trained
on).

Run it once on a dev machine (needs ``onnx``; no torch/NeMo/safetensors):

    python3 -m stt_mod.bake_adapter \\
        --export ~/dorai/models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25 \\
        --adapter ~/stt/models/mc-ref  --name mc-ref

-> ~/dorai/models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25-dorai-mc-ref/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys

import numpy as np

_ST_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": None, "F64": np.float64}


def read_safetensors(path):
    """Minimal numpy reader for .safetensors (header len + JSON + raw)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, meta in header.items():
            if k == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            dt = meta["dtype"]
            if dt == "BF16":
                u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                arr = u.view(np.float32)
            else:
                arr = np.frombuffer(raw, dtype=_ST_DTYPES[dt])
            out[k] = arr.reshape(meta["shape"]).astype(np.float32)
    return out


def load_adapter(adapter_dir):
    """-> (list of per-layer dicts {norm.weight, norm.bias, down.weight, up.weight}, config)."""
    cfg_p = os.path.join(adapter_dir, "adapter_config.json")
    cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
    w = read_safetensors(os.path.join(adapter_dir, "adapter.safetensors"))
    layers = []
    i = 0
    while True:
        p = f"encoder.layers.{i}.adapter."
        own = {k[len(p):]: v for k, v in w.items() if k.startswith(p)}
        if not own:
            break
        need = {"norm.weight", "norm.bias", "down.weight", "up.weight"}
        if set(own) != need:
            raise ValueError(f"layer {i}: adapter tensors {sorted(own)} != {sorted(need)}")
        layers.append(own)
        i += 1
    if sum(len(l) for l in layers) != len(w):
        raise ValueError("adapter file has tensors outside encoder.layers.<i>.adapter.*")
    return layers, cfg


def _encoder_path(export_dir):
    names = sorted(n for n in os.listdir(export_dir)
                   if n.startswith("encoder") and n.endswith(".onnx"))
    if not names:
        raise FileNotFoundError(f"no encoder*.onnx in {export_dir}")
    int8 = [n for n in names if "int8" in n]
    return os.path.join(export_dir, (int8 or names)[0])


def bake(export_dir, adapter_dir, out_dir, name, base_safetensors=None, eps=1e-5,
         log=print):
    import onnx
    from onnx import helper, numpy_helper

    layers, cfg = load_adapter(adapter_dir)
    enc_p = _encoder_path(export_dir)
    log(f"loading {enc_p}")
    model = onnx.load(enc_p, load_external_data=True)
    g = model.graph
    init = {t.name: t for t in g.initializer}

    norm_out = {}
    for n in g.node:
        if n.op_type == "LayerNormalization" and n.name.endswith("/norm_out/LayerNormalization"):
            i = int(n.name.split("/layers.")[1].split("/")[0])
            norm_out[i] = n
    if len(norm_out) != len(layers):
        raise ValueError(f"export has {len(norm_out)} norm_out layers, adapter has {len(layers)}")

    # Same-base check: the fp32 LayerNorm weights of the export vs the HF base.
    if base_safetensors:
        base = read_safetensors(base_safetensors)
        worst = 0.0
        for i in range(len(layers)):
            for suf in ("weight", "bias"):
                k = f"layers.{i}.norm_out.{suf}"
                a = numpy_helper.to_array(init[k]).astype(np.float32)
                b = base[f"encoder.{k}"]
                worst = max(worst, float(np.abs(a - b).max()))
        log(f"base check: max |export - base| over norm_out weights = {worst:.3g}")
        if worst > 1e-4:
            raise ValueError("export and adapter base are different checkpoints "
                             f"(max diff {worst}); refusing to bake")

    consumers = {}
    for n in g.node:
        for j, x in enumerate(n.input):
            consumers.setdefault(x, []).append((n, j))

    new_nodes_after = {}
    for i, lw in enumerate(layers):
        ln = norm_out[i]
        y = ln.output[0]
        pre = f"/dorai_adapter/layers.{i}"
        t = {
            f"{pre}/norm.weight": lw["norm.weight"],
            f"{pre}/norm.bias": lw["norm.bias"],
            f"{pre}/down.wT": np.ascontiguousarray(lw["down.weight"].T),   # [H, d]
            f"{pre}/up.wT": np.ascontiguousarray(lw["up.weight"].T),       # [d, H]
        }
        for k, v in t.items():
            g.initializer.append(numpy_helper.from_array(v.astype(np.float32), k))
        z = f"{pre}/out"
        nodes = [
            helper.make_node("LayerNormalization", [y, f"{pre}/norm.weight", f"{pre}/norm.bias"],
                             [f"{pre}/ln"], name=f"{pre}/ln", axis=-1, epsilon=float(eps)),
            helper.make_node("MatMul", [f"{pre}/ln", f"{pre}/down.wT"], [f"{pre}/down"],
                             name=f"{pre}/down"),
            helper.make_node("Sigmoid", [f"{pre}/down"], [f"{pre}/sig"], name=f"{pre}/sig"),
            helper.make_node("Mul", [f"{pre}/down", f"{pre}/sig"], [f"{pre}/silu"],
                             name=f"{pre}/silu"),
            helper.make_node("MatMul", [f"{pre}/silu", f"{pre}/up.wT"], [f"{pre}/up"],
                             name=f"{pre}/up"),
            helper.make_node("Add", [y, f"{pre}/up"], [z], name=f"{pre}/add"),
        ]
        # Re-wire every original consumer of y to the adapted output z.
        for n, j in consumers.get(y, []):
            n.input[j] = z
        for o in g.output:
            if o.name == y:
                raise ValueError("norm_out output is a graph output; unexpected export")
        new_nodes_after[ln.name] = nodes

    # Rebuild the node list keeping topological order (insert right after norm_out).
    old = list(g.node)
    del g.node[:]
    for n in old:
        g.node.append(n)
        for m in new_nodes_after.get(n.name, []):
            g.node.append(m)

    meta = {p.key: p for p in model.metadata_props}
    for k, v in (("dorai_adapter", name),
                 # names only: no local absolute paths in published models
                 ("dorai_adapter_source", os.path.basename(os.path.abspath(adapter_dir))),
                 ("dorai_base_export", os.path.basename(os.path.abspath(export_dir))),
                 ("dorai_adapter_checkpoint", str(cfg.get("checkpoint", "")))):
        if k in meta:
            meta[k].value = v
        else:
            model.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))

    onnx.checker.check_model(model, full_check=False)
    os.makedirs(out_dir, exist_ok=True)
    out_enc = os.path.join(out_dir, os.path.basename(enc_p))
    log(f"writing {out_enc}")
    onnx.save(model, out_enc)   # < 2 GB: single file, like the source export
    for f in os.listdir(export_dir):
        src = os.path.join(export_dir, f)
        if f.startswith("encoder"):
            continue
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, f))
        elif os.path.isdir(src):          # test_wavs/
            shutil.copytree(src, os.path.join(out_dir, f), dirs_exist_ok=True)
    info = {
        "adapter": name,
        "adapter_dir": os.path.basename(os.path.abspath(adapter_dir)),
        "adapter_config": {k: v for k, v in cfg.items() if k != "base_model"},
        "base_model": "nvidia/nemotron-speech-streaming-en-0.6b",
        "base_export": os.path.basename(os.path.abspath(export_dir)),
        "layers": len(layers),
        "adapter_dim": int(layers[0]["down.weight"].shape[0]),
        "encoder_sha256": hashlib.sha256(open(out_enc, "rb").read()).hexdigest(),
    }
    json.dump(info, open(os.path.join(out_dir, "dorai_adapter.json"), "w"), indent=2)
    log(f"done: {out_dir}")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", required=True, help="sherpa-onnx Nemotron export dir")
    ap.add_argument("--adapter", required=True, help="dir with adapter.safetensors")
    ap.add_argument("--name", default=None, help="adapter tag (default: adapter dir name)")
    ap.add_argument("--out", default=None, help="output dir (default: <export>-dorai-<name>)")
    ap.add_argument("--base", default=None,
                    help="base model.safetensors for the same-checkpoint check "
                         "(default: <adapter>/<adapter_config.base_model>/model.safetensors if present)")
    a = ap.parse_args(argv)
    export = os.path.expanduser(a.export).rstrip("/")
    adapter = os.path.expanduser(a.adapter).rstrip("/")
    name = a.name or os.path.basename(adapter)
    out = os.path.expanduser(a.out) if a.out else f"{export}-dorai-{name}"
    base = a.base
    if base is None:
        cfg_p = os.path.join(adapter, "adapter_config.json")
        if os.path.exists(cfg_p):
            rel = json.load(open(cfg_p)).get("base_model")
            cand = os.path.join(adapter, rel or "", "model.safetensors")
            base = cand if os.path.exists(cand) else None
    if base is None:
        print("warning: no base checkpoint found, skipping same-base check", file=sys.stderr)
    bake(export, adapter, out, name, base_safetensors=base)


if __name__ == "__main__":
    main()
