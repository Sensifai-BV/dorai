#!/usr/bin/env python3
"""replay_session.py — score doa_mod's estimator against a ground-truth session.

Drives the NODE'S OWN classes (SpeechGate + BearingEstimator, exactly as
doa.py wires them) over the recorded takes of a Panza bundle, so what is
measured here is the shipping code path rather than a re-implementation of it.
The only thing not exercised is ROS transport and the audio device.

Usage:
    python3 test/replay_session.py [session_dir] [--modes neural energy off]

Writes doa_mod_scores.csv next to the session and prints a summary.

PASS BAR (the round-2 protocol): over the eight marked Task-2 positions,
mean absolute error <= 15 deg and max <= 25 deg, on the RAW reported bearing
with no rotation fitted. The established figure for this pipeline is 6.1 deg
mean / 21 deg max; the 21 deg outlier is the 315 deg mark, which has a known
reflector. A material departure from that means the node's wiring differs
from the validated offline path -- investigate before shipping.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import wave

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PKG)

import rosmock                                                     # noqa: E402
rosmock.install()          # must precede `import doa`

import doa as D                                                    # noqa: E402
import doa_core as dc                                              # noqa: E402

# Ground truth for the marked takes. Unmarked takes (calibration walks, the
# walk-through, the two-talker takes) have no single true bearing and are
# reported but not scored.
TRUTH = {
    't2_000': 0.0, 't2_045': 45.0, 't2_090': 90.0, 't2_135': 135.0,
    't2_180': 180.0, 't2_225': 225.0, 't2_270': 270.0, 't2_315': 315.0,
    't3_045': 45.0, 't3_315': 315.0,
}
TASK2 = [k for k in TRUTH if k.startswith('t2_')]


def read_take(take_dir):
    """[4, N] float64 in [-1,1], plus fs."""
    chans, fs = [], None
    for i in range(1, 5):
        path = os.path.join(take_dir, f"ch{i}_mic{i}.wav")
        with wave.open(path, "rb") as w:
            assert w.getsampwidth() == 2, path
            fs = w.getframerate() if fs is None else fs
            assert w.getframerate() == fs, f"fs mismatch in {path}"
            raw = w.readframes(w.getnframes())
        chans.append(np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0)
    n = min(len(c) for c in chans)
    return np.stack([c[:n] for c in chans]), fs


def wrap180(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def make_estimator(mode, fs, avg_tc, subtract_noise=False):
    gate = D.SpeechGate(
        mode, model_path=os.path.join(_PKG, 'vad.onnx') if mode == 'neural' else None)
    return D.BearingEstimator(dc.array_positions(), fs=fs, gate=gate,
                              avg_tc=avg_tc, subtract_noise=subtract_noise)


def score_take(x, fs, mode, avg_tc=None):
    """Pool over the whole take (avg_tc=None => plain sum, the offline mode)."""
    est = make_estimator(mode, fs, avg_tc)
    t0 = time.perf_counter()
    for s in range(0, x.shape[1] - D.BLOCK + 1, D.HOP):
        est.push(x[:, s:s + D.BLOCK])
    wall = time.perf_counter() - t0
    az, sharp = est.bearing()
    audio_s = x.shape[1] / fs
    return dict(az=az, sharpness=sharp, n_frames=est.n_frames,
                n_speech=est.n_speech, consistent_frac=est.consistent_frac,
                wall_s=wall, realtime_x=audio_s / wall if wall > 0 else float('inf'),
                duration_s=audio_s)


def tracking_latency(x, fs, mode, avg_tc=0.5, settle_deg=15.0):
    """Time for a live-mode estimator to settle within settle_deg of the end bearing.

    t4_walk is a talker moving between marks, so the interesting number is not
    accuracy but how fast an exponential-forget pool follows them. Measured as:
    run the whole take, take the final bearing as the destination, then find
    the last time the running bearing was still outside settle_deg of it.
    """
    est = make_estimator(mode, fs, avg_tc)
    track = []
    for s in range(0, x.shape[1] - D.BLOCK + 1, D.HOP):
        est.push(x[:, s:s + D.BLOCK])
        az, _ = est.bearing()
        track.append(((s + D.BLOCK) / fs, az))
    if not track:
        return float('nan'), []
    final = track[-1][1]
    if not np.isfinite(final):
        return float('nan'), track
    outside = [t for t, a in track
               if np.isfinite(a) and abs(wrap180(a - final)) > settle_deg]
    return (max(outside) if outside else 0.0), track


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('session', nargs='?',
                    default=os.path.join(_PKG, '..', 'doa',
                                         'session_2026-08-12-10-04-37'))
    ap.add_argument('--modes', nargs='+', default=['neural', 'energy', 'off'],
                    help="VAD modes to score (default: all three)")
    ap.add_argument('--out', default=None, help="CSV path")
    args = ap.parse_args(argv)

    session = os.path.abspath(args.session)
    if not os.path.isdir(session):
        print(f"no such session: {session}", file=sys.stderr)
        return 1
    out_csv = args.out or os.path.join(os.getcwd(), 'doa_mod_scores.csv')

    takes = sorted(d for d in os.listdir(session)
                   if os.path.isdir(os.path.join(session, d))
                   and os.path.isfile(os.path.join(session, d, 'ch1_mic1.wav')))

    rows = []
    print(f"session: {session}\n{len(takes)} takes, modes: {args.modes}\n")
    hdr = f"{'take':16s} {'truth':>6s} " + " ".join(
        f"{m:>10s}" for m in args.modes)
    print(hdr)
    print("-" * len(hdr))
    for t in takes:
        x, fs = read_take(os.path.join(session, t))
        truth = TRUTH.get(t)
        line = f"{t:16s} " + (f"{truth:6.0f} " if truth is not None else f"{'--':>6s} ")
        row = dict(take=t, truth_deg=truth if truth is not None else '',
                   duration_s=round(x.shape[1] / fs, 2))
        for mode in args.modes:
            r = score_take(x, fs, mode)
            err = wrap180(r['az'] - truth) if truth is not None else None
            line += f"{r['az']:7.1f}" + (f"{err:+4.0f}" if err is not None else "    ")
            row[f'{mode}_deg'] = round(r['az'], 1)
            row[f'{mode}_err_deg'] = round(err, 1) if err is not None else ''
            row[f'{mode}_speech_frac'] = round(r['n_speech'] / max(r['n_frames'], 1), 3)
            row[f'{mode}_sharpness'] = round(r['sharpness'], 3)
            row[f'{mode}_realtime_x'] = round(r['realtime_x'], 1)
        rows.append(row)
        print(line)

    # -- the pass bar --------------------------------------------------------
    print()
    summary = {}
    for mode in args.modes:
        errs = np.array([abs(float(r[f'{mode}_err_deg'])) for r in rows
                         if r['take'] in TASK2 and r[f'{mode}_err_deg'] != ''])
        no315 = np.array([abs(float(r[f'{mode}_err_deg'])) for r in rows
                          if r['take'] in TASK2 and r['take'] != 't2_315'
                          and r[f'{mode}_err_deg'] != ''])
        ok = errs.mean() <= 15.0 and errs.max() <= 25.0
        summary[mode] = dict(mean=errs.mean(), max=errs.max(),
                             mean_no315=no315.mean(), max_no315=no315.max(),
                             passed=ok)
        print(f"  {mode:7s} Task-2 marks (n={len(errs)}): "
              f"mean {errs.mean():5.2f} deg  max {errs.max():4.0f} deg   "
              f"[excl 315: mean {no315.mean():4.2f} max {no315.max():3.0f}]   "
              f"{'PASS' if ok else 'FAIL'} (bar: mean<=15, max<=25)")

    # -- tracking on the walk take ------------------------------------------
    walk = os.path.join(session, 't4_walk')
    if os.path.isdir(walk):
        print("\n  t4_walk settling time (live mode, avg_tc=0.5 s):")
        x, fs = read_take(walk)
        for mode in args.modes:
            lat, _ = tracking_latency(x, fs, mode)
            print(f"    {mode:7s} last outside +-15 deg of final at {lat:5.2f} s "
                  f"of {x.shape[1]/fs:.1f} s")

    if rows:
        keys = list(rows[0].keys())
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_csv}")

    return 0 if all(s['passed'] for s in summary.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
