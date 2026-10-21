#!/usr/bin/env python3
"""Offline evaluation of the stt_mod engine: WER, speed, latency and SNR.

Decodes a manifest with exactly the streaming path the node uses
(NemotronStreamer fed in voice_mod-like frames, endpoint finals), for one or
more latency profiles / adapters, optionally with noise mixed in at given
SNRs, and reports per condition x configuration:

  WER, CER, WER without an out-of-vocabulary name (--name), empty-transcript
  rate, end-of-speech -> final lag (audio time), decode RTF (compute time /
  audio time on this machine), true mixing SNR (when --noise is used), blind
  SNR estimate (metrics.LevelMeter), CPU temperature / clock.

Manifest: one JSON per line with audio_filepath (relative to the manifest or
absolute), text and optional condition. No ROS needed.

  ros2 run stt_mod evaluate --manifest data/manifest.jsonl --latency-ms 80 160 560 1120 --out eval_out
  python3 -m stt_mod.evaluate --manifest m.jsonl --adapter mc-ref none --noise welding/*.wav --snr 0 5 10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from .metrics import LevelMeter, SystemProbe, score, speech_end_s
from .models import auto_rule2, find_model
from .streamer import NemotronStreamer
from .transcribe import read_wav_16k

FS = 16000


def active_rms(x, frame=320):
    """RMS of the active (speech) frames: frames within 30 dB of the loudest."""
    n = len(x) // frame
    p = (x[:n * frame].reshape(n, frame) ** 2).mean(1) if n else np.array([np.mean(x ** 2)])
    p = p[p > p.max() * 1e-3] if p.max() > 0 else p
    return float(np.sqrt(p.mean() + 1e-20))


def mix(speech, noise, snr_db, rng):
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    o = int(rng.integers(0, len(noise) - len(speech) + 1))
    nz = noise[o:o + len(speech)]
    g = active_rms(speech) / (np.sqrt(np.mean(nz ** 2)) + 1e-12) / (10 ** (snr_db / 20))
    y = speech + g * nz
    pk = float(np.abs(y).max())
    return (y * (0.99 / pk) if pk > 0.99 else y).astype(np.float32)


def decode(streamer, x, frame_s, gap_s):
    """Feed x + gap of silence in frames; return finals [(text, pos_s)] relative to x start."""
    y = np.concatenate([x, np.zeros(int(gap_s * FS), np.float32)])
    p0 = streamer.pos_s
    finals = []
    cb = lambda t, p: finals.append((t, p - p0))  # noqa: E731
    fr = max(1, int(frame_s * FS))
    t0 = time.perf_counter()
    for i in range(0, len(y), fr):
        streamer.push(y[i:i + fr], on_final=cb)
    streamer.finalize(on_final=lambda t, p: finals.append((t, float("inf"))))
    return finals, time.perf_counter() - t0, len(y) / FS


def summarize(rows):
    def agg(rs):
        lag = np.array([r["lag_s"] for r in rs if r["lag_s"] is not None])
        est = [r["snr_est_db"] for r in rs if r["snr_est_db"] is not None]
        tru = [r["snr_db"] for r in rs if r.get("snr_db") is not None]
        tmp = [r["cpu_temp_c"] for r in rs if r.get("cpu_temp_c") is not None]
        frq = [r["cpu_freq_mhz"] for r in rs if r.get("cpu_freq_mhz") is not None]
        return {"n": len(rs),
                "wer": 100 * sum(r["err"] for r in rs) / max(1, sum(r["n_ref"] for r in rs)),
                "wer_noname": 100 * sum(r["err_noname"] for r in rs) / max(1, sum(r["n_ref_noname"] for r in rs)),
                "cer": 100 * sum(r["err_chr"] for r in rs) / max(1, sum(r["n_chr"] for r in rs)),
                "empty": 100 * float(np.mean([r["empty"] for r in rs])),
                "lag_med_s": float(np.median(lag)) if lag.size else None,
                "lag_p90_s": float(np.percentile(lag, 90)) if lag.size else None,
                "rtf": sum(r["compute_s"] for r in rs) / max(1e-9, sum(r["audio_s"] for r in rs)),
                "snr_true_db": float(np.mean(tru)) if tru else None,
                "snr_est_db": float(np.median(est)) if est else None,
                "temp_max_c": max(tmp) if tmp else None, "freq_min_mhz": min(frq) if frq else None}
    groups = {}
    for r in rows:
        groups.setdefault((r["config"], r["condition"]), []).append(r)
        groups.setdefault((r["config"], "ALL"), []).append(r)
    return {f"{k[0]}|{k[1]}": agg(v) for k, v in sorted(groups.items())}


def fmt(v, f="{:.1f}"):
    return "—" if v is None else f.format(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--latency-ms", type=int, nargs="+", default=[1120])
    ap.add_argument("--adapter", nargs="+", default=["mc-ref"])
    ap.add_argument("--models-dir", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--frame-s", type=float, default=1.0, help="feed frame (voice_mod publish frame)")
    ap.add_argument("--gap-s", type=float, default=3.0, help="silence appended after each utterance")
    ap.add_argument("--noise", nargs="*", default=[], help="noise wav(s) (glob ok) to mix in")
    ap.add_argument("--snr", type=float, nargs="*", default=[], help="mixing SNRs in dB (needs --noise)")
    ap.add_argument("--keep-clean", action="store_true", help="also score the unmixed input when mixing")
    ap.add_argument("--name", default="", help="out-of-vocabulary name to exclude for wer_noname")
    ap.add_argument("--conditions", nargs="*", default=[], help="only these manifest conditions")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    base = os.path.dirname(os.path.abspath(a.manifest))
    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    if a.conditions:
        rows = [r for r in rows if r.get("condition", "all") in a.conditions]
    if a.limit:
        rows = rows[:a.limit]
    noises = [read_wav_16k(p) for g in a.noise for p in sorted(glob.glob(g))]
    if a.snr and not noises:
        raise SystemExit("--snr needs --noise")
    rng = np.random.default_rng(a.seed)
    items = []           # (id, condition, text, audio, true_snr)
    for r in rows:
        p = r["audio_filepath"] if os.path.isabs(r["audio_filepath"]) else os.path.join(base, r["audio_filepath"])
        x = read_wav_16k(p)
        cond = r.get("condition", "all")
        if not a.snr or a.keep_clean:
            items.append((r["audio_filepath"], cond, r["text"], x, r.get("snr_db")))
        for snr in a.snr:
            nz = noises[int(rng.integers(len(noises)))]
            items.append((r["audio_filepath"], f"{cond}+noise{snr:+g}dB", r["text"], mix(x, nz, snr, rng), snr))
    os.makedirs(a.out, exist_ok=True)
    probe = SystemProbe()
    out_rows = []
    fh = open(os.path.join(a.out, "utterances.jsonl"), "w")
    for lat in a.latency_ms:
        for ad in a.adapter:
            d, lat_, ad_ = find_model(lat, ad, models_dir=a.models_dir)
            cfg = f"{lat_}ms/{ad_}"
            t0 = time.time()
            s = NemotronStreamer(d, num_threads=a.threads, rule2_sil=auto_rule2(lat_))
            print(f"== {cfg}: {len(items)} utterances (load {time.time() - t0:.1f}s, rule2 {s.rule2_sil:.2f}s)", flush=True)
            for k, (uid, cond, text, x, snr) in enumerate(items):
                finals, comp, dur = decode(s, x, a.frame_s, a.gap_s)
                hyp = " ".join(t for t, _ in finals).strip()
                end = speech_end_s(x)
                # last endpoint final = when the utterance was complete (an earlier
                # final can come from a pause inside the utterance)
                last = next((p for _, p in reversed(finals) if np.isfinite(p)), None)
                m = LevelMeter(); m.push(x)
                row = {"config": cfg, "latency_ms": lat_, "adapter": ad_, "id": uid, "condition": cond,
                       "ref": text, "hyp": hyp, "n_finals": len(finals),
                       "lag_s": round(last - end, 3) if last is not None else None,
                       "compute_s": round(comp, 3), "audio_s": round(dur, 3), "rtf": round(comp / dur, 3),
                       "snr_db": snr, **m.window(0, len(x) / FS), **probe.sample(),
                       **score(text, hyp, a.name)}
                out_rows.append(row)
                fh.write(json.dumps(row) + "\n"); fh.flush()
                if (k + 1) % 20 == 0 or k + 1 == len(items):
                    print(f"  {k + 1}/{len(items)}  RTF {row['rtf']:.2f}  temp {row['cpu_temp_c']}", flush=True)
            del s
    summ = summarize(out_rows)
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    lines = ["| config | condition | n | WER % | WER w/o name % | CER % | empty % | lag med / p90 s | RTF | SNR true / est dB | temp max °C |",
             "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- | ---: |"]
    for k, v in summ.items():
        cfg, cond = k.split("|")
        lines.append(f"| {cfg} | {cond} | {v['n']} | {v['wer']:.1f} | {v['wer_noname']:.1f} | {v['cer']:.1f} | "
                     f"{v['empty']:.1f} | {fmt(v['lag_med_s'], '{:.2f}')} / {fmt(v['lag_p90_s'], '{:.2f}')} | "
                     f"{v['rtf']:.2f} | {fmt(v['snr_true_db'])} / {fmt(v['snr_est_db'])} | {fmt(v['temp_max_c'])} |")
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
