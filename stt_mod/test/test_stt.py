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


def test_default_nemotron_model_is_complete_when_installed(monkeypatch):
    """The default Nemotron export (1120 ms, mc-ref), wherever it is installed, has all its files."""
    try:
        p, lat, ad = models.find_model(1120, "mc-ref")
    except FileNotFoundError:
        pytest.skip("no Nemotron export installed (scripts/download_models.sh)")
    assert (lat, ad) == (1120, "mc-ref")
    for f in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        assert os.path.getsize(os.path.join(p, f)) > 0
    info = json.load(open(os.path.join(p, "dorai_adapter.json")))
    assert info["adapter"] == "mc-ref" and info["layers"] == 24


def test_models_are_found_in_the_hub_repo_layout(tmp_path, monkeypatch):
    """`hf download sensifai/dorai --local-dir <root>` leaves the models where the repo has them:
    stt-model/whisper-finetuned/<sherpa dir>/ and stt-model/nemo-finetuned/ (no sherpa-onnx name)."""
    monkeypatch.setattr(models, "default_roots", lambda extra="": [extra])
    root = tmp_path / "models"
    w = root / "stt-model" / "whisper-finetuned" / "sherpa-onnx-whisper-large-v3-turbo-dorai-artaban"
    w.mkdir(parents=True)
    for f in ("turbo-encoder.int8.onnx", "turbo-decoder.int8.onnx", "turbo-tokens.txt"):
        (w / f).write_text("x")
    n = root / "stt-model" / "nemo-finetuned"
    n.mkdir(parents=True)
    (n / "tokens.txt").write_text("x 0\n")
    with pytest.raises(FileNotFoundError):                       # not identified without its adapter info
        models.find_model(1120, "mc-ref", models_dir=str(root))
    (n / "dorai_adapter.json").write_text(json.dumps({
        "adapter": "mc-ref", "base_export": "sherpa-onnx-nemotron-speech-streaming-en-0.6b-1120ms-int8-2026-04-25"}))
    assert models.find_model(1120, "mc-ref", models_dir=str(root)) == (str(n), 1120, "mc-ref")
    assert models.parse_dir(str(n)) == (1120, "mc-ref") and models.chunk_ms_of(str(n)) == 1120
    with pytest.raises(FileNotFoundError, match="1120ms/mc-ref"):
        models.find_model(560, "mc-ref", models_dir=str(root))
    assert models.find_whisper_model(models_dir=str(root)) == (str(w), w.name)
    assert models.find_whisper_model("dorai-artaban", models_dir=str(root))[0] == str(w)
    # a model placed directly in the root still wins over the repo layout below it
    d = root / f"{models.PREFIX}-1120ms-int8-2026-04-25-dorai-mc-ref"
    d.mkdir(); (d / "tokens.txt").write_text("x 0\n")
    assert models.find_model(1120, "mc-ref", models_dir=str(root))[0] == str(d)
    (n / "dorai_adapter.json").write_text("not json")
    assert models.parse_dir(str(n)) is None


# ---------------------------------------------------------------- whisper
def _whisper_dir(root, name):
    d = root / name
    d.mkdir(parents=True)
    for f in ("turbo-encoder.onnx", "turbo-encoder.int8.onnx", "turbo-decoder.int8.onnx", "turbo-tokens.txt"):
        (d / f).write_text("x")
    return d


def test_find_whisper_model(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "default_roots", lambda extra="": [extra])
    with pytest.raises(FileNotFoundError, match="Installed: none"):
        models.find_whisper_model(models_dir=str(tmp_path))
    stock = _whisper_dir(tmp_path, "sherpa-onnx-whisper-turbo")
    assert models.find_whisper_model(models_dir=str(tmp_path)) == (str(stock), stock.name)
    tuned = _whisper_dir(tmp_path, "sherpa-onnx-whisper-large-v3-turbo-dorai-artaban")
    os.utime(tuned, (0, 0))                                     # older than the stock export
    assert models.find_whisper_model(models_dir=str(tmp_path))[0] == str(tuned)       # a dorai fine-tune wins
    assert models.find_whisper_model("turbo", models_dir=str(tmp_path))[0] == str(stock)
    assert models.find_whisper_model("dorai-artaban", models_dir=str(tmp_path))[0] == str(tuned)
    enc, dec, tok = models.pick_whisper_files(str(tuned))
    assert enc.endswith("turbo-encoder.int8.onnx") and dec.endswith("decoder.int8.onnx") and tok.endswith("tokens.txt")
    (tmp_path / "sherpa-onnx-whisper-broken").mkdir()            # no model files: not listed
    assert len(models.list_whisper_models([str(tmp_path)])) == 2
    with pytest.raises(FileNotFoundError, match="dorai-artaban"):
        models.find_whisper_model("missing", models_dir=str(tmp_path))


def test_tame_cuts_loops_and_overlong_text():
    from stt_mod.whisper import tame
    assert tame("Artaban come here.", 1.2) == "Artaban come here."
    assert tame("no no no", 1.0) == "no"                                        # a word three times in a row
    loop = "I have " + "a little bit of " * 12
    assert tame(loop, 4.0) == "I have a little bit of"
    assert tame("Artaban come here. I have been here. I am here. I am here. I am here.", 6.0) \
        == "Artaban come here. I have been here."                              # back to the last whole sentence
    assert tame("Go to welding station two. Go to welding station two.", 4.0).count("station") == 2   # twice is speech
    long = " ".join(f"w{i}" for i in range(60))
    assert len(tame(long, 2.0).split()) == 13                                   # 3 + 5 words per second
    assert tame("", 1.0) == ""


def _burst(fs, seconds, amp, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    return (amp * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))
            + rng.normal(0, amp / 20, len(t))).astype(np.float32)


def test_energy_gate_finds_utterances_over_noise():
    from stt_mod.whisper import EnergyGate
    fs = 16000
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.004, 12 * fs).astype(np.float32)        # steady noise, -48 dBFS
    for start, dur in ((2.0, 1.5), (6.0, 0.8)):
        x[int(start * fs):int((start + dur) * fs)] += _burst(fs, dur, 0.08, 1)
    x[int(9.5 * fs):int(9.5 * fs) + 800] += 0.3                  # a 50 ms click: too short to be speech
    g = EnergyGate(fs)
    segs = []
    for i in range(0, len(x), 1600):                             # 0.1 s frames, as from voice_mod
        segs += g.accept(x[i:i + 1600])
    segs += g.flush()
    assert len(segs) == 2
    for (start, seg), (t0, dur) in zip(segs, ((2.0, 1.5), (6.0, 0.8))):
        assert t0 - 0.35 <= start / fs <= t0 - 0.2               # 0.3 s pre-roll
        assert dur + 0.8 <= len(seg) / fs <= dur + 1.1           # pre-roll + speech + 0.6 s hangover
    # the floor adapts: the same recording 20 dB louder gives the same two utterances
    g = EnergyGate(fs)
    assert len(g.accept(10 * x) + g.flush()) == 2
    # steady noise alone: nothing
    g = EnergyGate(fs)
    assert g.accept(rng.normal(0, 0.05, 10 * fs).astype(np.float32)) + g.flush() == []


class _FakeStream:
    def __init__(self): self.n = 0; self.result = type("R", (), {"text": ""})()
    def accept_waveform(self, fs, x): self.n = len(x)


class _FakeRecognizer:
    """Stands in for sherpa_onnx.OfflineRecognizer: 'heard N' with N = tenths of a second decoded."""
    def __init__(self, text=None): self.text, self.calls = text, []
    def create_stream(self): return _FakeStream()

    def decode_stream(self, st):
        self.calls.append(st.n)
        st.result.text = self.text if self.text is not None else f" heard {round(st.n / 1600)} "


def test_whisper_streamer_decodes_only_gated_speech():
    from stt_mod.whisper import WhisperStreamer
    fs = 16000
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.004, 14 * fs).astype(np.float32)
    x[2 * fs:int(3.5 * fs)] += _burst(fs, 1.5, 0.08, 1)
    x[8 * fs:int(8.1 * fs)] += _burst(fs, 0.1, 0.08, 2)          # 0.1 s: under min_speech_s, never decoded
    rec = _FakeRecognizer()
    s = WhisperStreamer("unused", vad="energy", recognizer=rec)
    finals = []
    for i in range(0, len(x), 8000):
        s.push(x[i:i + 8000], on_final=lambda t, p: finals.append((t, p)))
    assert s.finalize(on_final=lambda t, p: finals.append((t, p))) == ""
    assert len(finals) == 1 and len(rec.calls) == 1
    text, pos = finals[0]
    assert text.startswith("heard") and 3.5 < pos <= 4.6          # reported once the hangover has passed
    assert s.stats["finals"] == 1 and abs(s.stats["audio_s"] - 14.0) < 1e-6 and s.pos_s == 14.0
    # a loop from the recogniser is cut; an empty result gives no final
    s = WhisperStreamer("unused", vad="energy", recognizer=_FakeRecognizer("Artaban come here. " + "I am here. " * 9))
    out = []
    s.push(x, on_final=lambda t, p: out.append(t)); s.finalize(on_final=lambda t, p: out.append(t))
    assert out == ["Artaban come here."]
    s = WhisperStreamer("unused", vad="energy", recognizer=_FakeRecognizer(""))
    s.push(x, on_final=lambda t, p: out.append(t))
    assert s.stats["finals"] == 0 and s.stats["dropped"] >= 1
    # vad none / frame_is_utterance: every frame is one utterance
    rec = _FakeRecognizer()
    s = WhisperStreamer("unused", frame_is_utterance=True, recognizer=rec)
    s.push(x[:fs]); s.push(x[fs:3 * fs])
    assert rec.calls == [fs, 2 * fs] and s.vad_kind == "none"
    with pytest.raises(ValueError):
        WhisperStreamer("unused", vad="magic", recognizer=rec)


def test_node_whisper_engine_with_stub(monkeypatch, tmp_path):
    """engine=whisper end to end through the node with a stub recogniser; fallback when the model is missing."""
    import rosmock
    rosmock.install()
    import importlib
    import stt_mod.node as N
    importlib.reload(N)
    made = {}

    class Stub(N.WhisperStreamer):
        def __init__(self, model_dir, **kw):
            made.update(kw, model_dir=model_dir)
            super().__init__(model_dir, recognizer=_FakeRecognizer("Artaban turn left."), **kw)

    monkeypatch.setattr(N, "WhisperStreamer", Stub)
    d = _whisper_dir(tmp_path, "sherpa-onnx-whisper-large-v3-turbo-dorai-artaban")
    rosmock.Node.overrides = {"models_dir": str(tmp_path), "vad": "energy"}       # engine: the default
    node = N.SttNode()
    assert node.engine == "whisper" and node.model_tag == "whisper/large-v3-turbo-dorai-artaban"
    assert made["model_dir"] == str(d) and made["vad"] == "energy"
    fs = 16000
    x = np.random.default_rng(0).normal(0, 0.004, 6 * fs).astype(np.float32)
    x[fs:int(2.5 * fs)] += _burst(fs, 1.5, 0.08, 1)
    cb = node.subs["/dorai_clean_audio"]
    for k, i in enumerate(range(0, len(x), fs)):
        cb(rosmock.Float32MultiArray([0.0, 16000, 1, k] + x[i:i + fs].tolist()))
    t0 = time.time()
    while node._q or node._busy:
        time.sleep(0.02)
        assert time.time() - t0 < 30
    node.shutdown()
    assert node.pubs["/dorai_transcript"].msgs == ["Artaban turn left."]
    assert node.pubs["/dorai_partial_transcript"].msgs == []
    node._status()
    assert json.loads(node.pubs["/dorai_stt/status"].msgs[-1])["model"].startswith("whisper/")
    # no Whisper export and no fallback: the node refuses to start
    rosmock.Node.overrides = {"models_dir": str(tmp_path / "empty"), "engine_fallback": False}
    monkeypatch.setattr(N, "find_whisper_model", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no Whisper model")))
    with pytest.raises(FileNotFoundError):
        N.SttNode()
    rosmock.Node.overrides = {"engine": "parrot"}
    with pytest.raises(ValueError):
        N.SttNode()


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
    rosmock.Node.overrides = {"engine": "nemotron", "num_threads": 4, "metrics": True}
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
    rosmock.Node.overrides = {"engine": "nemotron", "num_threads": 4}   # bundled 1120 ms + mc-ref
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


@needs_model
def test_node_falls_back_to_nemotron_without_whisper(monkeypatch):
    import rosmock
    rosmock.install()
    import importlib
    import stt_mod.node as N
    importlib.reload(N)
    monkeypatch.setattr(N, "find_whisper_model",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no Whisper model in []")))
    rosmock.Node.overrides = {"num_threads": 4}                     # engine: whisper (default), fallback on
    node = N.SttNode()
    assert node.engine == "nemotron" and node.model_tag == "nemotron-0.6b/1120ms/mc-ref"
    assert any(lvl == "error" and "Falling back" in msg for lvl, msg in node.get_logger().lines)
    node.shutdown()


def _whisper_installed():
    try:
        import sherpa_onnx  # noqa: F401
        return models.find_whisper_model()[0]
    except (ImportError, FileNotFoundError):
        return None


WHISPER = _whisper_installed()
needs_whisper = pytest.mark.skipif(WHISPER is None, reason="no Whisper export installed")


@needs_whisper
def test_whisper_real_model_transcribes_and_stays_silent_on_noise():
    from stt_mod.whisper import WhisperStreamer, find_vad_model
    vad_model = find_vad_model("", WHISPER, models.default_roots())
    s = WhisperStreamer(WHISPER, num_threads=4, vad="neural" if vad_model else "energy", vad_model=vad_model)
    x = np.concatenate([np.zeros(8000, np.float32), _test_wav() if MODEL else np.zeros(0, np.float32),
                        np.zeros(24000, np.float32)])
    finals = []
    for i in range(0, len(x), 16000):
        s.push(x[i:i + 16000], on_final=lambda t, p: finals.append(t))
    s.finalize(on_final=lambda t, p: finals.append(t))
    assert (len(" ".join(finals).split()) >= 3) == bool(MODEL)
    noise = np.random.default_rng(0).normal(0, 0.02, 16000 * 10).astype(np.float32)
    s.push(noise, on_final=lambda t, p: finals.append("NOISE:" + t))
    s.finalize(on_final=lambda t, p: finals.append("NOISE:" + t))
    assert not [f for f in finals if f.startswith("NOISE:")]


def _vad_file():
    from stt_mod.whisper import find_vad_model
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return ""
    return find_vad_model("", WHISPER or "", models.default_roots())


@pytest.mark.skipif(MODEL is None or not _vad_file(), reason="needs sherpa-onnx, vad_gated.onnx and a test wav")
def test_neural_gate_keeps_the_start_of_the_utterance():
    """The detector fires after the first sounds; the gate hands the utterance out with a pre-roll, without
    reaching back into the previous utterance, and passes nothing for noise."""
    from stt_mod.whisper import NeuralGate
    fs = 16000
    speech = _test_wav()
    onset = int(np.argmax(np.abs(speech) > 0.02 * np.abs(speech).max()))
    lead = 2 * fs
    x = np.concatenate([np.zeros(lead, np.float32), speech, np.zeros(2 * fs, np.float32)])

    def run(pre):
        g = NeuralGate(_vad_file(), fs, pre_roll_s=pre)
        segs = []
        for i in range(0, len(x), 1600):
            segs += g.accept(x[i:i + 1600])
        return segs + g.flush()

    plain, rolled = run(0.0), run(0.3)
    assert plain and len(plain) == len(rolled)
    assert rolled[0][0] <= lead + onset                            # the first sound is inside the utterance
    assert plain[0][0] - int(0.3 * fs) <= rolled[0][0] < plain[0][0]
    for (s0, a), (s1, b) in zip(rolled, rolled[1:]):
        assert s0 + len(a) <= s1                                   # utterances do not overlap
    assert sum(len(a) for _, a in rolled) <= len(x)
    g = NeuralGate(_vad_file(), fs)
    noise = np.random.default_rng(0).normal(0, 0.02, 10 * fs).astype(np.float32)
    assert g.accept(noise) + g.flush() == []
