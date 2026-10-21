"""stt_mod tests. Pure-Python tests always run; model tests run when a
Nemotron export is installed (see README) and sherpa-onnx is importable.

  python3 -m pytest stt_mod/test -v
"""
import json
import os
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from stt_mod import models  # noqa: E402


# ------------------------------------------------------------------ models
def test_parse_dir():
    P = models.PREFIX
    assert models.parse_dir(f"{P}-560ms-int8-2026-04-25") == (560, "none")
    assert models.parse_dir(f"{P}-1120ms-int8-2026-04-25-dorai-mc-ref") == (1120, "mc-ref")
    assert models.parse_dir("something-else") is None


def test_find_model_reports_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "default_roots", lambda extra="": [extra])
    d = tmp_path / f"{models.PREFIX}-560ms-int8-2026-04-25"
    d.mkdir(); (d / "tokens.txt").write_text("x 0\n")
    p, lat, ad = models.find_model(560, "none", models_dir=str(tmp_path))
    assert (p, lat, ad) == (str(d), 560, "none")
    with pytest.raises(FileNotFoundError, match="560ms/none"):
        models.find_model(1120, "mc-ref", models_dir=str(tmp_path))


def test_root_order_wins_over_mtime(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    for r in (a, b):
        d = r / f"{models.PREFIX}-1120ms-int8-2026-04-25-dorai-mc-ref"
        d.mkdir(parents=True); (d / "tokens.txt").write_text("x 0\n")
    os.utime(a / f"{models.PREFIX}-1120ms-int8-2026-04-25-dorai-mc-ref", (0, 0))  # older
    monkeypatch.setattr(models, "default_roots", lambda extra="": [str(a), str(b)])
    assert models.find_model(1120, "mc-ref")[0].startswith(str(a))


def test_default_model_bundled_in_package(monkeypatch):
    """The default (1120 ms, mc-ref) ships in stt_mod/models and is found with no env."""
    monkeypatch.delenv("DORAI_MODELS_DIR", raising=False)
    pkg = os.path.normpath(os.path.join(HERE, "..", "models"))
    p, lat, ad = models.find_model(1120, "mc-ref")
    assert (lat, ad) == (1120, "mc-ref") and p.startswith(pkg), p
    for f in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        assert os.path.getsize(os.path.join(p, f)) > 0
    info = json.load(open(os.path.join(p, "dorai_adapter.json")))
    assert info["adapter"] == "mc-ref" and info["layers"] == 24


# ---------------------------------------------------------------- metrics
def test_level_meter_snr_estimate():
    """Speech-like bursts over stationary noise at a known SNR."""
    from stt_mod.metrics import LevelMeter
    rng = np.random.default_rng(0)
    fs = 16000
    noise = rng.normal(0, 0.01, 6 * fs).astype(np.float32)
    sig = np.zeros_like(noise)
    t = np.arange(2 * fs) / fs
    sig[2 * fs:4 * fs] = 0.1 * np.sin(2 * np.pi * 300 * t)      # 2 s burst, power 5e-3
    m = LevelMeter()
    m.push(noise + sig)
    w = m.window(0, 6)
    true = 10 * np.log10(np.mean(sig[2 * fs:4 * fs] ** 2) / 1e-4)    # ~17 dB
    assert abs(w["snr_est_db"] - true) < 1.5, (w, true)
    assert 0.25 < w["speech_frac"] < 0.4


def test_level_meter_ignores_digital_silence():
    from stt_mod.metrics import LevelMeter
    rng = np.random.default_rng(0)
    fs = 16000
    x = np.zeros(6 * fs, np.float32)                       # gated pauses: exact zeros
    t = np.arange(2 * fs) / fs
    x[2 * fs:4 * fs] = 0.1 * np.sin(2 * np.pi * 300 * t) + rng.normal(0, 0.01, 2 * fs)
    m = LevelMeter(); m.push(x)
    assert m.window(0, 6)["snr_est_db"] is None          # no measurable noise: no estimate (was ~100 dB)
    x[:2 * fs] += rng.normal(0, 0.003, 2 * fs).astype(np.float32)   # residual noise in one pause
    m = LevelMeter(); m.push(x)
    true = 10 * np.log10(np.mean((0.1 * np.sin(2 * np.pi * 300 * t)) ** 2) / 0.003 ** 2)
    assert abs(m.window(0, 6)["snr_est_db"] - true) < 1.5


def test_score_wer_cer_and_name():
    from stt_mod.metrics import score
    s = score("artaban turn left", "Arthur ban turn left.", name="artaban")
    assert s["n_ref"] == 3 and s["err"] == 2 and s["err_noname"] == 0
    assert score("abc", "")["empty"] == 1


# ------------------------------------------------------------- bake tool
def test_bake_adapter_graph(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import helper, numpy_helper, TensorProto
    from stt_mod import bake_adapter as B
    H, d, L = 8, 2, 2
    rng = np.random.default_rng(0)
    nodes, inits, x = [], [], "x"
    for i in range(L):
        inits += [numpy_helper.from_array(np.ones(H, np.float32), f"layers.{i}.norm_out.weight"),
                  numpy_helper.from_array(np.zeros(H, np.float32), f"layers.{i}.norm_out.bias")]
        y = f"/layers.{i}/norm_out/out"
        nodes.append(helper.make_node("LayerNormalization",
                     [x, f"layers.{i}.norm_out.weight", f"layers.{i}.norm_out.bias"], [y],
                     name=f"/layers.{i}/norm_out/LayerNormalization", axis=-1))
        x2 = f"/layers.{i}/relu"
        nodes.append(helper.make_node("Relu", [y], [x2], name=x2)); x = x2
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, H])],
                          [helper.make_tensor_value_info(x, TensorProto.FLOAT, [1, 3, H])], inits)
    exp = tmp_path / f"{models.PREFIX}-560ms-int8-x"; exp.mkdir()
    mp = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    mp.ir_version = 8
    onnx.save(mp, exp / "encoder.int8.onnx")
    (exp / "tokens.txt").write_text("x 0\n")
    W = {}
    for i in range(L):
        W[f"encoder.layers.{i}.adapter.norm.weight"] = rng.normal(size=H).astype(np.float32)
        W[f"encoder.layers.{i}.adapter.norm.bias"] = rng.normal(size=H).astype(np.float32)
        W[f"encoder.layers.{i}.adapter.down.weight"] = rng.normal(size=(d, H)).astype(np.float32)
        W[f"encoder.layers.{i}.adapter.up.weight"] = rng.normal(size=(H, d)).astype(np.float32)
    ad = tmp_path / "mc-test"; ad.mkdir()
    hdr, off, blobs = {}, 0, []
    for k, v in W.items():
        b = v.tobytes(); hdr[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        off += len(b); blobs.append(b)
    h = json.dumps(hdr).encode()
    (ad / "adapter.safetensors").write_bytes(len(h).to_bytes(8, "little") + h + b"".join(blobs))
    out = tmp_path / "baked"
    B.bake(str(exp), str(ad), str(out), "mc-test", log=lambda s: None)

    def ln(v, w, b):
        m, s = v.mean(-1, keepdims=True), v.var(-1, keepdims=True)
        return (v - m) / np.sqrt(s + 1e-5) * w + b
    inp = rng.normal(size=(1, 3, H)).astype(np.float32)
    ref = inp
    for i in range(L):
        y = ln(ref, 1, 0)
        a = ln(y, W[f"encoder.layers.{i}.adapter.norm.weight"], W[f"encoder.layers.{i}.adapter.norm.bias"])
        dn = a @ W[f"encoder.layers.{i}.adapter.down.weight"].T
        y = y + (dn / (1 + np.exp(-dn))) @ W[f"encoder.layers.{i}.adapter.up.weight"].T
        ref = np.maximum(y, 0)
    got = ort.InferenceSession(str(out / "encoder.int8.onnx")).run(None, {"x": inp})[0]
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)
    assert json.load(open(out / "dorai_adapter.json"))["layers"] == L


# ------------------------------------------------------- real model tests
def _installed():
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return None
    try:
        return models.find_model(1120, "mc-ref")[0]      # the bundled default
    except FileNotFoundError:
        return None


MODEL = _installed()
needs_model = pytest.mark.skipif(MODEL is None, reason="no Nemotron export installed")


def _test_wav():
    d = os.path.join(MODEL, "test_wavs")
    from stt_mod.transcribe import read_wav_16k
    return read_wav_16k(os.path.join(d, sorted(f for f in os.listdir(d) if f.endswith(".wav"))[0]))


@needs_model
def test_streamer_chunking_invariant():
    """Same text whether the audio arrives as one frame or as 0.7 s frames."""
    from stt_mod.streamer import NemotronStreamer, transcribe
    x = _test_wav()
    s = NemotronStreamer(MODEL, num_threads=4)
    whole = transcribe(s, x)
    finals = []
    for i in range(0, len(x), 11200):
        s.push(x[i:i + 11200], on_final=lambda t, p: finals.append(t))
    t = s.finalize(on_final=lambda t, p: None)
    assert whole and " ".join(finals + [t]).split() == whole.split()


@needs_model
def test_evaluate_cli(tmp_path):
    from stt_mod import evaluate
    d = os.path.join(MODEL, "test_wavs")
    trans = dict(l.split(maxsplit=1) for l in open(os.path.join(d, "trans.txt")) if l.strip())
    man = tmp_path / "m.jsonl"
    man.write_text(json.dumps({"audio_filepath": os.path.join(d, "0.wav"),
                               "text": trans["0.wav"] if "0.wav" in trans else trans[sorted(trans)[0]]}) + "\n")
    noise = tmp_path / "n.wav"
    import wave
    with wave.open(str(noise), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.random.default_rng(1).normal(0, 3000, 16000 * 8)).astype(np.int16).tobytes())
    out = tmp_path / "out"
    evaluate.main(["--manifest", str(man), "--out", str(out), "--threads", "4",
                   "--noise", str(noise), "--snr", "10", "--keep-clean"])
    summ = json.load(open(out / "summary.json"))
    clean, noisy = summ["1120ms/mc-ref|all"], summ["1120ms/mc-ref|all+noise+10dB"]
    assert clean["wer"] < 20 and clean["rtf"] > 0 and clean["lag_med_s"] is not None
    assert noisy["snr_true_db"] == 10 and noisy["snr_est_db"] is not None


@needs_model
def test_node_metrics_topic():
    import rosmock
    rosmock.install()
    import importlib
    import stt_mod.node as N
    importlib.reload(N)
    rosmock.Node.overrides = {"num_threads": 4, "metrics": True}
    node = N.SttNode()
    x = np.concatenate([_test_wav(), np.zeros(32000, np.float32)])
    cb = node.subs["/dorai_clean_audio"]
    for k, i in enumerate(range(0, len(x), 16000)):
        cb(rosmock.Float32MultiArray([0.0, 16000, 1, k] + x[i:i + 16000].tolist()))
    t0 = time.time()
    while node._q or node._busy:
        time.sleep(0.05)
        assert time.time() - t0 < 120
    node.shutdown()
    ms = [json.loads(m) for m in node.pubs["/dorai_stt/metrics"].msgs]
    assert ms and all(m["snr_est_db"] is not None and m["rtf_window"] for m in ms[:1])
    assert {"lag_s", "backlog_s", "speech_dbfs", "noise_dbfs", "proc_cpu_pct"} <= set(ms[0])


@needs_model
def test_node_end_to_end_with_rosmock():
    import rosmock
    rosmock.install()
    import importlib
    import stt_mod.node as N
    importlib.reload(N)
    rosmock.Node.overrides = {"num_threads": 4}      # all defaults: bundled 1120 ms + mc-ref
    node = N.SttNode()
    x = np.concatenate([_test_wav(), np.zeros(32000, np.float32)])
    cb = node.subs["/dorai_clean_audio"]
    fr = 16000 * 2
    for k, i in enumerate(range(0, len(x), fr)):
        cb(rosmock.Float32MultiArray([0.0, 16000, 1, k] + x[i:i + fr].tolist()))
    t0 = time.time()
    while node.backlog_s() > 0 or node._q:
        time.sleep(0.05)
        assert time.time() - t0 < 120
    time.sleep(0.5)
    node.shutdown()
    finals = node.pubs["/dorai_transcript"].msgs
    assert finals and "/dorai_intent" not in node.pubs
    assert node.model_tag == "nemotron-0.6b/1120ms/mc-ref"
    node._status()
    st = json.loads(node.pubs["/dorai_stt/status"].msgs[-1])
    assert st["finals"] >= 1 and st["rtf"] > 0
