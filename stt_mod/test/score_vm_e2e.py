#!/usr/bin/env python3
"""Score a voice_mod -> stt_mod e2e run (from vm_e2e_harness.py + the stt log).

The stimulus is utterances concatenated with pauses; its schedule JSON lists
start_s / dur_s / text / condition per utterance. The recorded clean stream
(clean.wav = what stt_mod received) is aligned to stimulus channel 0 by
cross-correlating band-passed log envelopes. Each final in the stt log
(`Transcript: "..." (lag Xs, stream Ys)`) is then mapped to the utterance
whose [start, next start) contains stream - offset.

Also reports, per utterance, the blind SNR estimate (metrics.LevelMeter) of
the stimulus channel 0 (stt input without voice_mod) and of voice_mod's output,
so the change in input SNR can be compared.

  python3 score_vm_e2e.py --run out/A3 --stimulus A_ch0.wav --schedule A_schedule.json --name artaban
"""
import argparse
import json
import os
import re
import sys
import wave

import numpy as np
from scipy.signal import butter, correlate, sosfiltfilt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stt_mod.metrics import LevelMeter, score  # noqa: E402

FS = 16000
PAT = re.compile(r'Transcript: "(.*)" \(lag ([-\d.na]+)s, stream ([\d.]+)s\)')


def rd(p):
    with wave.open(p) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        assert w.getframerate() == FS
        return x.reshape(-1, w.getnchannels()).mean(1)


def env(x, hop=160):
    y = sosfiltfilt(butter(4, [300, 3400], btype="band", fs=FS, output="sos"), x)
    n = len(y) // hop
    e = np.log(np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-10))
    return (e - e.mean()) / (e.std() + 1e-9)


def offset(clean, stim, max_s=180):
    ec, es = env(clean), env(stim)
    c = correlate(ec, es, mode="full", method="fft")
    lags = np.arange(-len(es) + 1, len(ec))
    ok = (lags >= 0) & (lags <= max_s * 100)
    return lags[ok][np.argmax(c[ok])] / 100.0


def local_offsets(clean, stim, sched, off, win=1.5):
    """Per-utterance offset, TRACKED: each search is centred on the previous
    utterance's offset, so a slow clock drift between the stimulus and the
    clean stream (seen: +3 s over 13 min through snd-aloop + voice_mod's
    drift lock) is followed instead of clipping at the search edge."""
    ec, es = env(clean), env(stim)
    out, cur = [], off
    for u in sched:
        a, b = int(u["start_s"] * 100), int((u["start_s"] + u["dur_s"]) * 100)
        lo, hi = int((u["start_s"] + cur - win) * 100), int((u["start_s"] + u["dur_s"] + cur + win) * 100)
        if lo < 0 or hi > len(ec) or b - a < 20:
            out.append(None); continue
        c = correlate(ec[lo:hi], es[a:b], mode="valid")
        o = (lo + int(np.argmax(c))) / 100.0 - u["start_s"]
        out.append(round(o, 2))
        cur = o
    return out


def snr_est(x, t0, t1):
    m = LevelMeter(); m.push(x[max(0, int(t0 * FS)):int(t1 * FS)])
    return m.window(0, t1 - t0)["snr_est_db"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stimulus", required=True)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sched = json.load(open(a.schedule))
    clean, stim = rd(os.path.join(a.run, "clean.wav")), rd(a.stimulus)
    off = offset(clean, stim)
    loc = local_offsets(clean, stim, sched, off)
    finals = [(m.group(1), float(m.group(3))) for m in
              (PAT.search(l) for l in open(os.path.join(a.run, "stt.log"), errors="replace")) if m]
    starts = np.array([u["start_s"] for u in sched])
    # utterance start times on the CLEAN timeline, from the tracked offsets
    # (missing ones interpolated); a final belongs to the utterance whose
    # [clean start, next clean start) contains its stream position.
    lo_arr = np.array([np.nan if v is None else v for v in loc], float)
    ok = ~np.isnan(lo_arr)
    lo_arr = np.interp(np.arange(len(sched)), np.nonzero(ok)[0], lo_arr[ok]) if ok.any() else np.full(len(sched), off)
    cstarts = starts + lo_arr
    hyp, stray = [[] for _ in sched], []
    for text, pos in finals:
        i = int(np.searchsorted(cstarts, pos, side="right")) - 1
        (hyp[i] if i >= 0 else stray).append(text)
    rows = []
    for k, (u, h, lo) in enumerate(zip(sched, hyp, loc)):
        t1 = sched[k + 1]["start_s"] if k + 1 < len(sched) else u["start_s"] + u["dur_s"] + 6
        H = " ".join(h)
        rows.append({"id": u["id"], "condition": u["condition"], "ref": u["text"], "hyp": H,
                     "n_finals": len(h), "local_offset_s": lo, "snr_db": u.get("snr_db"),
                     "snr_est_in_db": snr_est(stim, u["start_s"] - 1.5, t1 - 1.5),
                     "snr_est_out_db": snr_est(clean, cstarts[k] - 1.5, t1 + lo_arr[k] - 1.5),
                     **score(u["text"], H, a.name)})
    out = a.out or os.path.join(a.run, "scored.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lo = [r["local_offset_s"] for r in rows if r["local_offset_s"] is not None]
    wer = 100 * sum(r["err"] for r in rows) / sum(r["n_ref"] for r in rows)
    print(json.dumps({"run": a.run, "offset_s": off, "local_offset_range_s": [min(lo), max(lo)] if lo else None,
                      "finals": len(finals), "stray_finals": len(stray), "clean_s": len(clean) / FS,
                      "wer": round(wer, 1), "scored": out}))


if __name__ == "__main__":
    main()
