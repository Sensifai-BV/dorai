#!/usr/bin/env python3
"""capture_debug — record the raw microphone streams and report what is wrong.

Run this INSTEAD of guessing. It captures each device to its own wav, then
measures the things that actually break DoA, in the order they break it:

    python capture_debug.py --devices 1 3 4 --seconds 20 --linear 140 140

Speak normally, from a fixed position, for the whole recording. Then send the
printed report (and optionally the wavs) back for analysis.

It writes  capture_debug/ch1.wav … chN.wav  plus  capture_debug/report.json,
so the exact audio can be replayed offline:

    python live_doa.py --wav-dir capture_debug --linear 140 140
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doa_core as dc                                    # noqa: E402


def write_wav(path, x, fs):
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes((x * 32767).astype('<i2').tobytes())


def capture(devices, seconds, fs):
    """Record each device to its own array. Returns (data [N,T], fs)."""
    import sounddevice as sd

    n = len(devices)
    rates = []
    for d in devices:
        info = sd.query_devices(d, 'input')
        if info['max_input_channels'] < 1:
            raise SystemExit(f"device {d} has no input channels")
        rates.append(int(info['default_samplerate']))
    if fs <= 0:
        fs = max(set(rates), key=rates.count)
    print(f"[capture] devices {devices}, native rates {rates}, using {fs} Hz")

    cap = int(seconds * fs) + fs
    bufs = [np.zeros(cap) for _ in range(n)]
    counts = [0] * n
    streams = []

    def make_cb(k):
        def cb(indata, frames, tinfo, status):
            if status:
                print(f"[capture] dev{devices[k]}: {status}", file=sys.stderr)
            x = np.asarray(indata, dtype=np.float64)[:, 0]
            c = counts[k]
            m = min(len(x), cap - c)
            if m > 0:
                bufs[k][c:c + m] = x[:m]
                counts[k] = c + m
        return cb

    for k, d in enumerate(devices):
        info = sd.query_devices(d, 'input')
        streams.append(sd.InputStream(
            device=d, channels=int(info['max_input_channels']),
            samplerate=fs, dtype='float32', blocksize=1024,
            callback=make_cb(k)))
    for s in streams:
        s.start()
    print(f"[capture] recording {seconds:.0f} s — SPEAK NOW, from one fixed spot")
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(0.1)
        done = int((time.time() - t0) / seconds * 40)
        print(f"\r  [{'#'*done}{'.'*(40-done)}]", end='', flush=True)
    print()
    for s in streams:
        s.stop(); s.close()
    T = min(counts)
    return np.stack([b[:T] for b in bufs]), fs


def analyse(data, fs, gaps_mm, report):
    n, T = data.shape
    print(f"\n=== 1. levels ({T/fs:.1f} s captured) ===")
    rms = np.sqrt((data ** 2).mean(axis=1))
    for k in range(n):
        print(f"  ch{k+1}: rms {rms[k]:.5f}  peak {np.abs(data[k]).max():.3f}")
    report['rms'] = rms.tolist()
    if rms.min() < 1e-5:
        print("  [!] a channel is SILENT — its delays are meaningless.")
        report['problems'].append('silent channel')
    elif rms.max() / max(rms.min(), 1e-12) > 20:
        print(f"  [!] level spread {rms.max()/rms.min():.0f}x — check placement/gain.")
        report['problems'].append('level imbalance')

    print("\n=== 2. start skew between streams ===")
    lags, _ = _align(data, fs)
    skew_ms = (lags.max() - lags.min()) / fs * 1000
    print(f"  lags {lags} samples = {np.round(lags/fs*1000,1)} ms, spread {skew_ms:.1f} ms")
    report['start_skew_ms'] = float(skew_ms)
    gaps = np.atleast_1d(np.asarray(gaps_mm, float)) / 1000.0
    if gaps.size == 1:
        gaps = np.repeat(gaps, n - 1)
    aperture = float(gaps.sum())
    geo_us = aperture / dc.C_SOUND * 1e6
    print(f"  geometric max delay for a {aperture*1000:.0f} mm array: {geo_us:.0f} us")
    if skew_ms * 1000 > geo_us:
        print(f"  [!] skew is {skew_ms*1000/geo_us:.0f}x the whole geometric range —")
        print("      streams MUST be aligned before any bearing means anything.")
        report['problems'].append('start skew')

    aligned = _apply(data, lags)
    print("\n=== 3. clock drift (after alignment) ===")
    seg = int(2.0 * fs)
    nseg = max(2, aligned.shape[1] // seg)
    taus = []
    for i in range(nseg):
        s = aligned[:, i*seg:(i+1)*seg]
        if s.shape[1] < seg // 2:
            break
        r = dc.doa_linear(s, gaps, fs=fs)
        taus.append(r['taus'])
    taus = np.array(taus)
    ts = np.arange(len(taus)) * 2.0
    ppm = []
    for c in range(taus.shape[1]):
        A = np.vstack([ts, np.ones_like(ts)]).T
        sl, _ = np.linalg.lstsq(A, taus[:, c], rcond=None)[0]
        ppm.append(sl * 1e6)
    print(f"  per-pair drift rate: {np.round(ppm,1)} ppm")
    report['drift_ppm'] = [float(v) for v in ppm]
    worst = float(np.max(np.abs(ppm)))
    if worst > 5:
        win = (aperture/dc.C_SOUND) * abs(np.cos(np.deg2rad(85))-np.cos(np.deg2rad(90))) * 1e6 / worst
        print(f"  [!] {worst:.0f} ppm — bearing self-rotates 5 deg every {win:.2f} s.")
        print("      Independent clocks. Use a macOS Aggregate Device with Drift")
        print("      Correction, or accept short-window bearings only.")
        report['problems'].append('clock drift')

    print("\n=== 4. bearing consistency ===")
    F, H = 2048, 512
    ok = tot = 0
    th = []
    resid = []
    for s0 in range(0, aligned.shape[1] - F, H):
        blk = aligned[:, s0:s0+F]
        if dc.frame_rms(blk) < 1e-4:
            continue
        r = dc.doa_linear(blk, gaps, fs=fs)
        tot += 1
        resid.append(r['resid_us'])
        if r['consistent']:
            ok += 1
            th.append(r['theta_deg'])
    if tot == 0:
        print("  no frames above the noise floor — was anyone speaking?")
        report['problems'].append('no speech')
        return report
    print(f"  {ok}/{tot} frames physically consistent ({100*ok/tot:.0f}%)")
    print(f"  median residual {np.median(resid):.0f} us of {geo_us:.0f} us possible")
    report['consistent_frac'] = ok / tot
    report['median_resid_us'] = float(np.median(resid))
    if th:
        m, R, sd = dc.circular_mean(th)
        print(f"  bearing {m:.1f} deg (or {(-m)%360:.0f} deg mirrored), sd {sd:.1f}, R {R:.3f}")
        report['bearing_deg'] = float(m)
        report['bearing_sd'] = float(sd)
        report['bearing_R'] = float(R)
    if ok / tot < 0.5:
        print("  [!] most frames rejected — fix the problems above first.")
    return report


def _align(data, fs, max_lag_s=0.25, f_lo=100.0, f_hi=4000.0):
    n, T = data.shape
    max_lag = int(max_lag_s * fs)
    nfft = 1 << int(np.ceil(np.log2(2 * T)))
    fr = np.fft.rfftfreq(nfft, 1.0 / fs)
    keep = (fr >= f_lo) & (fr <= f_hi)

    def spec(x):
        X = np.fft.rfft(x - x.mean(), nfft)
        X[~keep] = 0.0
        return X

    R = spec(data[0])
    lags = np.zeros(n, dtype=int)
    for k in range(1, n):
        cc = np.fft.irfft(spec(data[k]) * np.conj(R), nfft)
        cc = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])
        lags[k] = int(np.argmax(cc)) - max_lag
    return lags - lags.min(), None


def _apply(data, lags):
    n, T = data.shape
    span = int(lags.max())
    out = np.empty((n, T - span))
    for k in range(n):
        s = span - int(lags[k])
        out[k] = data[k, s:s + T - span]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--devices', type=int, nargs='+', required=True)
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--fs', type=int, default=0, help='0 = device native rate')
    ap.add_argument('--linear', type=float, nargs='+', default=[140, 140],
                    help='gaps between consecutive mics, mm')
    ap.add_argument('--out', default='capture_debug')
    args = ap.parse_args(argv)

    data, fs = capture(args.devices, args.seconds, args.fs)
    os.makedirs(args.out, exist_ok=True)
    for k in range(data.shape[0]):
        write_wav(os.path.join(args.out, f'ch{k+1}.wav'), data[k], fs)
    print(f"[capture] wrote {data.shape[0]} wavs to {args.out}/")

    report = dict(devices=args.devices, fs=fs, seconds=data.shape[1] / fs,
                  gaps_mm=list(np.atleast_1d(args.linear)), problems=[])
    report = analyse(data, fs, args.linear, report)
    with open(os.path.join(args.out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 62)
    if report['problems']:
        print("PROBLEMS FOUND: " + ", ".join(dict.fromkeys(report['problems'])))
    else:
        print("No blocking problems found.")
    print(f"Full report: {args.out}/report.json")
    print(f"Replay offline:  python live_doa.py --wav-dir {args.out} "
          f"--linear {' '.join(str(int(g)) for g in np.atleast_1d(args.linear))}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
