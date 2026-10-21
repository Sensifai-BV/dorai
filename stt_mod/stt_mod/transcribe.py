#!/usr/bin/env python3
"""Offline check of the stt_mod engine on WAV files (no ROS needed).

Runs the same NemotronStreamer as the node, feeding each file the way
voice_mod frames arrive, and prints finals and the real-time factor. Use it to check models load on a machine, measure RTF on the Pi, and
A/B the latency profiles and adapters.

  ros2 run stt_mod transcribe clean.wav
  python3 -m stt_mod.transcribe --latency-ms 560 --adapter mc-ref a.wav b.wav
  python3 -m stt_mod.transcribe --frame-s 5 --threads 2 long.wav   # voice_mod framing
"""
from __future__ import annotations

import argparse
import json
import time
import wave

import numpy as np

from .models import auto_rule2, default_roots, find_model, list_models
from .streamer import NemotronStreamer


def read_wav_16k(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dt = {2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dt).astype(np.float32) / float(np.iinfo(dt).max + 1)
    x = x.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        try:   # polyphase with anti-alias filter (USB mics on the robot capture 48 kHz)
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), 16000)
            x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
        except ImportError:   # no scipy: linear interpolation (aliases above 8 kHz)
            n = int(round(len(x) * 16000 / sr))
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("wavs", nargs="*")
    ap.add_argument("--latency-ms", type=int, default=1120)
    ap.add_argument("--adapter", default="mc-ref")
    ap.add_argument("--rule2", type=float, default=0.0, help="0 = auto per latency profile")
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--models-dir", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--frame-s", type=float, default=0.0,
                    help="split into frames of this length (0 = whole file)")
    ap.add_argument("--list", action="store_true", help="list installed models and exit")
    ap.add_argument("--json", action="store_true", help="one JSON line per final")
    a = ap.parse_args(argv)
    if a.list or not a.wavs:
        for lat, ad, p in list_models(default_roots(a.models_dir)):
            print(f"{lat:5d} ms  {ad:7s}  {p}")
        return
    d, lat, ad = find_model(a.latency_ms, a.adapter, a.model_dir, a.models_dir)
    t0 = time.time()
    s = NemotronStreamer(d, num_threads=a.threads, rule2_sil=a.rule2 or auto_rule2(lat))
    print(f"model {lat} ms / {ad}: {d} (load {time.time() - t0:.1f}s)")
    for path in a.wavs:
        x = read_wav_16k(path)
        fr = int(a.frame_s * 16000) or len(x)
        finals = []
        cb = lambda t, p: finals.append((t, p))  # noqa: E731
        c0 = time.perf_counter()
        for i in range(0, len(x), fr):
            s.push(x[i:i + fr], on_final=cb)
        s.finalize(on_final=cb)
        comp = time.perf_counter() - c0
        for t, p in finals:
            if a.json:
                print(json.dumps({"file": path, "pos_s": round(p, 2), "text": t}))
            else:
                print(f"  {p:6.2f}s  {t}")
        print(f"{path}: {len(x) / 16000:.1f}s audio, RTF {comp / (len(x) / 16000):.3f}")


if __name__ == "__main__":
    main()
