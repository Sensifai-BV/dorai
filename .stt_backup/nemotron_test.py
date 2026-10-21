#!/usr/bin/env python3
"""
Offline test for the Nemotron streaming engine — runs the SAME NemotronStreamer
class the live stt_mod node uses, on one or more WAV files, exactly as the node
would see them (16 kHz mono float, fed in 200 ms blocks, final at frame end),
and prints partials, finals, wall time and RTF.

Use it to (a) verify the sherpa-onnx model files load on a machine, (b) measure
real-time factor on the Pi, (c) A/B raw vs dorai_beamformer-clean recordings.

Usage:
  python3 stt_mod/test/nemotron_test.py clean.wav [more.wav ...]
  python3 stt_mod/test/nemotron_test.py --threads 4 --frame-s 0 clean.wav   # whole file as one frame
  python3 stt_mod/test/nemotron_test.py --frame-s 10 --continuous long.wav   # voice_mod `fixed` mode
  python3 stt_mod/test/nemotron_test.py --model-dir ~/dorai/models/sherpa-onnx-nemotron-... x.wav

Any sample rate / channel count is accepted (downmixed + resampled to 16 kHz).
"""
import os
import sys
import time
import wave
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# stt.py imports rclpy at module level; stub it when ROS is not sourced so the
# engine class can still be exercised offline.
try:
    import rclpy  # noqa: F401
except ImportError:
    import types
    for name in ("rclpy", "rclpy.node", "std_msgs", "std_msgs.msg"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["rclpy.node"].Node = object
    sys.modules["std_msgs.msg"].Float32MultiArray = object
    sys.modules["std_msgs.msg"].String = object
from stt import NemotronStreamer, find_nemotron_model_dir  # noqa: E402


def read_wav_16k(path):
    w = wave.open(path, "rb")
    sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
    raw = w.readframes(w.getnframes())
    w.close()
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        n = int(round(len(x) * 16000 / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--chunk-ms", type=int, default=0,
                    help="prefer the export with this encoder chunk (e.g. 560, 1120)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frame-s", type=float, default=0.0,
                    help="split the file into frames of this length (0 = whole "
                         "file is one frame, like voice_mod vad segmentation)")
    ap.add_argument("--continuous", action="store_true",
                    help="frame_is_utterance=False: finals only at endpoints")
    ap.add_argument("--rule2", type=float, default=0.8)
    ap.add_argument("--quiet-partials", action="store_true")
    a = ap.parse_args()

    md = find_nemotron_model_dir(a.model_dir, a.chunk_ms)
    if md is None:
        sys.exit("Nemotron model dir not found (see README 'Nemotron').")
    print(f"model: {md}")
    t0 = time.perf_counter()
    eng = NemotronStreamer(md, num_threads=a.threads, rule2_sil=a.rule2,
                           frame_is_utterance=not a.continuous)
    print(f"load: {time.perf_counter() - t0:.2f}s")

    def on_partial(t):
        if not a.quiet_partials:
            print(f"  partial: {t}")

    finals = []
    for path in a.wavs:
        x = read_wav_16k(path)
        dur = len(x) / 16000
        print(f"\n== {path}  ({dur:.1f}s)")
        finals.clear()
        eng.stats = {"audio_s": 0.0, "compute_s": 0.0, "finals": 0}
        t1 = time.perf_counter()
        step = int(a.frame_s * 16000) if a.frame_s > 0 else len(x)
        for i in range(0, len(x), step):
            eng.push(x[i:i + step], on_partial, lambda t: finals.append(t))
        if a.continuous:
            t = eng.finalize(on_partial, lambda t: finals.append(t))
        wall = time.perf_counter() - t1
        for f in finals:
            print(f"  FINAL: {f}")
        print(f"  wall {wall:.2f}s for {dur:.1f}s audio -> RTF {wall / dur:.3f}"
              f"  (engine RTF {eng.rtf():.3f}, {len(finals)} finals)")


if __name__ == "__main__":
    main()
