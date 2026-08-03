#!/usr/bin/env python3
"""doa_protocol.py — guided positional DoA test for a 2-mic linear array.

Run this ON THE PI, in your own terminal, so you can see the prompts live:

    cd ~/dorai/doa
    sudo python3 doa_protocol.py --spacing 140

It walks you through a fixed set of speaking positions, records both mics
continuously, then reports the measured bearing for each position against the
angle the geometry predicts. That comparison is the actual test: a bearing on
its own proves nothing, a bearing that tracks a KNOWN position proves the
pipeline works.

Positions use the ARRAY AXIS as the reference:
    right  -> 0deg    (endfire, past mic 2)
    front  -> 90deg   (broadside, facing the middle)
    left   -> 180deg  (endfire, past mic 1)

Note left and right are the two ENDS of the axis and ARE distinguishable.
Above/below the line are NOT (the mirror ambiguity), which is why "front"
is the only non-axis position in the list.
"""
import argparse, json, os, subprocess, sys, time, wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doa_core as dc

SEGMENTS = [
    ("right", 0.30, 0.0),
    ("right", 1.00, 0.0),
    ("front", 0.30, 90.0),
    ("front", 1.00, 90.0),
    ("left",  0.30, 180.0),
    ("left",  1.00, 180.0),
]
SEG_SECONDS = 10.0
GAP_SECONDS = 6.0


def cards():
    out = subprocess.run(['arecord', '-l'], capture_output=True, text=True).stdout
    return [int(l.split()[1].rstrip(':')) for l in out.splitlines() if l.startswith('card')]



def preflight(a, b, fs, seconds=2.0):
    """Prove BOTH mics stream together before asking anyone to move.

    Without this the script records the whole routine, then discovers at
    analysis time that one device never opened -- which wastes the take and,
    worse, the user's time walking through positions for nothing. On a
    bandwidth-limited USB bus that failure is the NORMAL case, not an edge one.
    """
    import tempfile
    d = tempfile.mkdtemp()
    p = [os.path.join(d, 'pf1.wav'), os.path.join(d, 'pf2.wav')]
    procs = [subprocess.Popen(
        ['arecord', '-D', f'hw:{c},0', '-f', 'S16_LE', '-r', str(fs),
         '-c', '1', '-d', str(int(seconds)), p[i]],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for i, c in enumerate((a, b))]
    errs = [pr.communicate()[1].decode(errors='replace') for pr in procs]
    ok, msgs = True, []
    for i, (c, path, err) in enumerate(zip((a, b), p, errs)):
        size = os.path.getsize(path) if os.path.exists(path) else 0
        if size < 1000:
            ok = False
            first = err.strip().splitlines()[0] if err.strip() else 'no data'
            msgs.append(f"  card {c}: FAILED — {first}")
        else:
            x = np.frombuffer(wave.open(path).readframes(10 ** 9), np.int16) / 32768.
            r = float(np.sqrt((x ** 2).mean()))
            msgs.append(f"  card {c}: ok, {size} bytes, rms {r:.5f}"
                        + ("   [!] SILENT — check the mic" if r < 1e-5 else ""))
    print("\n".join(msgs))
    if not ok:
        print()
        print("  Both microphones must stream SIMULTANEOUSLY. The usual cause on")
        print("  a Raspberry Pi is USB isochronous bandwidth: full-speed audio")
        print("  dongles on separate ports of the Pi's internal hub each need")
        print("  their own split-transaction budget from ONE transaction")
        print("  translator, and the second one is refused (-28 ENOSPC).")
        print()
        print("  Fix: put the mics behind a single EXTERNAL powered USB hub.")
        print("  Counterintuitive but measured — behind one downstream hub they")
        print("  share a single 12 Mbit/s bus with room for both, instead of")
        print("  competing for the internal hub's translator.")
        print()
        print("  Check with:  dmesg | grep -i bandwidth")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spacing', type=float, required=True,
                    help='distance between the two mics, MILLIMETRES (measure it)')
    ap.add_argument('--fs', type=int, default=48000)
    ap.add_argument('--out', default='doa_protocol_out')
    ap.add_argument('--analyse-only', action='store_true',
                    help='skip recording; re-analyse an existing --out directory. '
                         'Use this to salvage a take whose analysis crashed.')
    args = ap.parse_args()

    if args.analyse_only:
        p1 = os.path.join(args.out, 'mic1.wav')
        p2 = os.path.join(args.out, 'mic2.wav')
        marks = json.load(open(os.path.join(args.out, 'marks.json')))
        missing = [p for p in (p1, p2) if not os.path.exists(p)]
        if missing:
            sys.exit(f"cannot analyse: missing {missing}. Both mics must have "
                     f"recorded; re-run the capture.")
        analyse(p1, p2, marks, args.spacing / 1000.0, args.fs, args.out)
        return

    cc = cards()
    if len(cc) < 2:
        sys.exit(f"need 2 capture cards, found {cc}. Reconnect the mics.")
    a, b = cc[0], cc[1]
    os.makedirs(args.out, exist_ok=True)
    gap_m = args.spacing / 1000.0
    total = len(SEGMENTS) * (SEG_SECONDS + GAP_SECONDS) + 4

    print("=" * 66)
    print(f" 2-MIC DoA POSITIONAL TEST   spacing {args.spacing:.0f} mm")
    print(f" recording cards {a} and {b} for {total:.0f} s total")
    print(f" aliasing limit {dc.C_SOUND/(2*gap_m):.0f} Hz")
    print("=" * 66)
    print(" mic 1 = card %d (LEFT end)      mic 2 = card %d (RIGHT end)" % (a, b))
    print(" If that is backwards, swap the cables or pass --swap later.")
    print("=" * 66)
    print("\n checking both microphones stream together…")
    if not preflight(a, b, args.fs):
        sys.exit(1)
    print("\n both microphones OK.")
    input(" press Enter to start… ")

    p1 = os.path.join(args.out, 'mic1.wav')
    p2 = os.path.join(args.out, 'mic2.wav')
    dur = int(total)
    r1 = subprocess.Popen(['arecord', '-D', f'hw:{a},0', '-f', 'S16_LE',
                           '-r', str(args.fs), '-c', '1', '-d', str(dur), p1],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r2 = subprocess.Popen(['arecord', '-D', f'hw:{b},0', '-f', 'S16_LE',
                           '-r', str(args.fs), '-c', '1', '-d', str(dur), p2],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    time.sleep(2.0)                      # let both streams settle

    marks = []
    for name, dist, truth in SEGMENTS:
        print()
        print("-" * 66)
        print(f"  MOVE: {name.upper()}   {dist*100:.0f} cm from the array"
              f"   (expected {truth:.0f}deg)")
        for k in range(int(GAP_SECONDS), 0, -1):
            print(f"    starting in {k}…", end='\r', flush=True)
            time.sleep(1.0)
        s = time.time() - t0
        print(f"    >>> SPEAK NOW for {SEG_SECONDS:.0f} s <<<        ")
        time.sleep(SEG_SECONDS)
        e = time.time() - t0
        marks.append(dict(position=name, distance_m=dist, truth_deg=truth,
                          start_s=s, end_s=e))
        print("    (stop)                                   ")

    r1.wait(); r2.wait()
    json.dump(marks, open(os.path.join(args.out, 'marks.json'), 'w'), indent=2)
    print("\nrecording finished — analysing\n")
    analyse(p1, p2, marks, gap_m, args.fs, args.out)


def analyse(p1, p2, marks, gap_m, fs, out):
    def rd(p):
        w = wave.open(p)
        return np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768.
    A, B = rd(p1), rd(p2)
    n = min(len(A), len(B)); A, B = A[:n], B[:n]
    X = np.stack([A, B])
    gaps = np.array([gap_m])
    f_hi = min(dc.F_HI, dc.C_SOUND / (2 * gap_m))
    F, H = 2048, 512

    print(f"{'position':>8} {'dist':>6} {'expected':>9} {'measured':>9} "
          f"{'mirror':>8} {'sd':>6} {'kept':>6} {'rms':>8}")
    rows = []
    for m in marks:
        s0, s1 = int(m['start_s'] * fs), int(m['end_s'] * fs)
        seg = X[:, s0:min(s1, n)]
        if seg.shape[1] < F:
            continue
        th, kept, tot = [], 0, 0
        for i in range(0, seg.shape[1] - F, H):
            blk = seg[:, i:i + F]
            if dc.frame_rms(blk) < 1e-4:
                continue
            tot += 1
            r = dc.doa_linear(blk, gaps, fs=fs, f_hi=f_hi)
            if r['consistent']:
                kept += 1; th.append(r['theta_deg'])
        if th:
            mu, R, sd = dc.circular_mean(th)
            err = abs(((mu - m['truth_deg'] + 180) % 360) - 180)
            rows.append(dict(**m, measured_deg=float(mu), sd_deg=float(sd),
                             kept_frac=kept / max(tot, 1), error_deg=float(err)))
            print(f"{m['position']:>8} {m['distance_m']*100:5.0f}cm "
                  f"{m['truth_deg']:8.0f}° {mu:8.1f}° {(-mu)%360:7.1f}° "
                  f"{sd:5.1f}° {100*kept/max(tot,1):5.0f}% "
                  f"{np.sqrt((seg[0]**2).mean()):8.5f}")
        else:
            rows.append(dict(**m, measured_deg=None, error_deg=None,
                             kept_frac=0.0))
            print(f"{m['position']:>8} {m['distance_m']*100:5.0f}cm "
                  f"{m['truth_deg']:8.0f}° {'REFUSED':>9} "
                  f"{'':>8} {'':>6} {0:5.0f}%")
    json.dump(rows, open(os.path.join(out, 'results.json'), 'w'), indent=2)

    ok = [r for r in rows if r.get('error_deg') is not None]
    print()
    if ok:
        errs = [r['error_deg'] for r in ok]
        print(f"median absolute error: {np.median(errs):.1f}°   "
              f"worst {max(errs):.1f}°   ({len(ok)}/{len(rows)} positions resolved)")
        lr = [r for r in ok if r['position'] in ('left', 'right')]
        if len(lr) >= 2:
            L = np.mean([r['measured_deg'] for r in lr if r['position'] == 'left'])
            R = np.mean([r['measured_deg'] for r in lr if r['position'] == 'right'])
            print(f"left/right separation: {abs(L-R):.0f}°  "
                  f"(should be ~180° — this is the test the array CAN pass)")
    print(f"\nwritten to {out}/  — send results.json back for analysis")
    print(f"replay offline: python3 live_doa.py --wav-dir {out} --linear "
          f"{gap_m*1000:.0f}")


if __name__ == '__main__':
    main()
