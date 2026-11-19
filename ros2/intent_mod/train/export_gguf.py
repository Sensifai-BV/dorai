#!/usr/bin/env python3
"""Merge an mlx_lm LoRA adapter into the ORIGINAL Hugging Face Qwen3.5 weights, then GGUF + quantise.

Why not `mlx_lm.fuse`: mlx_lm's Qwen3.5 loader rewrites tensors on load (RMSNorm weights +1,
conv1d axes moved, vision tower dropped), so its fused output is no longer the HF layout
llama.cpp's converter expects. Here every LoRA delta is added to the matching HF tensor and
everything else is copied byte-for-byte:

    W_hf[out, in] += scale * (lora_a[in, r] @ lora_b[r, out]).T
    mlx 'language_model.model.layers.N.<mod>'  <->  HF 'model.language_model.layers.N.<mod>.weight'

Steps (each skipped if its output exists):
  1. merge   -> <work>/merged-hf/            (bf16 safetensors + config/tokenizer)
  2. convert -> <work>/<name>-f16.gguf       (llama.cpp convert_hf_to_gguf.py, text model only)
  3. quant   -> <out>/<name>-<QTYPE>.gguf    (llama_cpp.llama_model_quantize)

    python export_gguf.py --base Qwen/Qwen3.5-4B --adapter ~/dorai/models/lora/intent-qwen35-4b-mlx \
        --converter /path/to/llama.cpp/convert_hf_to_gguf.py --name intent-qwen35-4b --qtype Q4_0 \
        --out ~/dorai/models/qwen35-4b-intent
Steps 1-2 need torch + safetensors + transformers; step 3 needs llama_cpp (--python-quant picks the interpreter).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys


def merge(base_dir, adapter_dir, out_dir):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    lp = cfg.get("lora_parameters", {})
    scale = float(lp.get("scale", 20.0))
    ad = {}
    with safe_open(os.path.join(adapter_dir, "adapters.safetensors"), "pt") as f:
        for k in f.keys():
            ad[k] = f.get_tensor(k).float()
    mods = sorted({k.rsplit(".", 1)[0] for k in ad if k.endswith(".lora_a")})
    print(f"adapter: {len(mods)} LoRA modules, rank {lp.get('rank')}, scale {scale}")

    def hf_name(m):
        assert m.startswith("language_model.model."), m
        return "model.language_model." + m[len("language_model.model."):] + ".weight"

    want = {hf_name(m): m for m in mods}
    os.makedirs(out_dir, exist_ok=True)
    idx = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))
    done = set()
    for shard in sorted(set(idx["weight_map"].values())):
        tensors = {}
        with safe_open(os.path.join(base_dir, shard), "pt") as f:
            meta = f.metadata()
            for k in f.keys():
                t = f.get_tensor(k)
                if k in want:
                    m = want[k]
                    delta = scale * (ad[m + ".lora_a"] @ ad[m + ".lora_b"]).T
                    assert delta.shape == t.shape, (k, delta.shape, t.shape)
                    t = (t.float() + delta).to(t.dtype)
                    done.add(k)
                tensors[k] = t.contiguous()
        save_file(tensors, os.path.join(out_dir, shard), metadata=meta or {"format": "pt"})
        print(f"  wrote {shard}: {len(tensors)} tensors")
    missing = set(want) - done
    assert not missing, f"LoRA modules with no HF tensor: {sorted(missing)[:5]}"
    for fn in os.listdir(base_dir):
        if not fn.endswith(".safetensors") and os.path.isfile(os.path.join(base_dir, fn)):
            shutil.copy(os.path.join(base_dir, fn), out_dir)
    print(f"merged {len(done)} tensors -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B", help="HF repo id or local dir")
    ap.add_argument("--adapter", help="mlx_lm adapter dir; omit to export the base model unchanged")
    ap.add_argument("--converter", required=True, help="llama.cpp convert_hf_to_gguf.py")
    ap.add_argument("--name", default="intent-qwen35-4b")
    ap.add_argument("--qtype", default="Q4_0")
    ap.add_argument("--work", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--python-quant", default=sys.executable)
    a = ap.parse_args()

    base = a.base
    if not os.path.isdir(base):
        from huggingface_hub import snapshot_download
        base = snapshot_download(base)
    work = a.work or os.path.join(a.out, "work")
    os.makedirs(work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)

    src = base
    if a.adapter:
        src = os.path.join(work, "merged-hf")
        if not os.path.exists(os.path.join(src, "config.json")):
            merge(base, os.path.expanduser(a.adapter), src)
    f16 = os.path.join(work, f"{a.name}-f16.gguf")
    if not os.path.exists(f16):
        subprocess.check_call([sys.executable, a.converter, src, "--outtype", "f16", "--outfile", f16])
    q = os.path.join(a.out, f"{a.name}-{a.qtype}.gguf")
    if not os.path.exists(q):
        code = ("import sys, llama_cpp as L\n"
                "p = L.llama_model_quantize_default_params()\n"
                f"p.ftype = L.LLAMA_FTYPE_MOSTLY_{a.qtype}\n"
                "p.nthread = 8\n"
                "rc = L.llama_model_quantize(sys.argv[1].encode(), sys.argv[2].encode(), p)\n"
                "sys.exit(rc)\n")
        subprocess.check_call([a.python_quant, "-c", code, f16, q])
    print(f"{q}  {os.path.getsize(q) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
