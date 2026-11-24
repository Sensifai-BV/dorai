"""DoA benchmark on simulated scenes: the DorAI node against SRP-PHAT and MUSIC.

    python -m experiments.doa_benchmark --speech <folder of commands> --out reports/doa_benchmark

Three matrices, following the design document:

  A  acceptance   free field, quiet, source at 0/45/90/135/180/225/270/315 deg, 2 m.
                  Checks the coordinate convention and the channel map; target < 5 deg.
  B  rooms        image-source rooms with RT60 0.2 .. 1.0 s, source at 1 and 3 m.
  C  noise        RT60 0.4 s, welding from one point, SNR -10 .. 20 dB at the array.

Every scene is rendered by the virtual lab (simulator/acoustic), written to disk
with its truth, and scored three ways on the same four channels:

  dorai   the doa_mod node itself (gate, pooling, publish rule), last bearing of each command
  srp     pyroomacoustics SRP-PHAT over the frames of each command (told when the command is)
  music   pyroomacoustics MUSIC, the same frames

The two baselines are given the command's true start and end, the node is not:
it has to detect the speech. A command the node gives no bearing for counts as
not detected, it is not averaged in.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_REPO]

from evaluation import doa_metrics as M                                    # noqa: E402
from evaluation import doa_node as score_doa                               # noqa: E402
from simulator.acoustic import audio_generator as render, config, room_engine, scene_mixer   # noqa: E402

WELDING = "simulator/lab/wavs/welding.wav"
AZ8 = [0, 45, 90, 135, 180, 225, 270, 315]
AZ4 = [20, 110, 200, 290]


def scene(name, room, az, dist, seed, noise=None, speech=None):
    """One talker at a fixed bearing and distance, array in the middle of the room, facing +x."""
    size = room_engine.load_room(room)["size_m"]
    centre = [size[0] / 2 + 0.31, size[1] / 2 - 0.23]                      # off the exact centre: no symmetric reflections
    return dict(name=name, room=room, seed=seed, arrays=["respeaker_4mic"],
                robot=dict(model="artaban", place=centre, heading_deg=0.0, height=1.0),
                talker=dict(commands=2, distance_m=[dist, dist], bearing_deg=float(az), mouth_height=1.5, moves=False,
                            lead_in_s=2.0, gap_s=2.0),
                speech=speech, noise=noise or [],
                hardware=dict(gain_db_sd=0.0, position_error_mm=0.0, self_noise_snr_db=50.0))


def matrices(speech, quick=False):
    out = []
    for az in AZ8:
        out.append(("A", dict(az=az, dist=2.0), scene(f"A_az{az:03d}", "anechoic", az, 2.0, az, speech=speech)))
    for rt in ([0.4] if quick else [0.2, 0.4, 0.6, 0.8, 1.0]):
        for dist in (1.0, 3.0):
            for az in (AZ4[:2] if quick else AZ4):
                out.append(("B", dict(rt60=rt, dist=dist, az=az),
                            scene(f"B_rt{int(rt * 10):02d}_d{int(dist)}_az{az:03d}", dict(name="rt60", rt60_s=rt, size_m=[9.0, 8.0, 3.5]), az, dist,
                                  1000 + az + int(rt * 100), speech=speech)))
    for snr in ([0] if quick else [-10, -5, 0, 5, 10, 15, 20]):
        for az in (AZ4[:2] if quick else AZ4):
            for rep in ((0,) if quick else (0, 1)):
                out.append(("C", dict(snr=snr, az=az, rep=rep),
                            scene(f"C_snr{snr:+03d}_az{az:03d}_{rep}", dict(name="rt60", rt60_s=0.4, size_m=[9.0, 8.0, 3.5]), az, 2.0,
                                  2000 + az + 10 * rep + snr, noise=[dict(file=WELDING, snr_db=snr)], speech=speech)))
    return out


def baseline(x, pos, commands, algo, fs=config.FS, nfft=512, f_lo=200.0, f_hi=2600.0):
    """pyroomacoustics DoA over each command's own frames: one bearing per command."""
    import pyroomacoustics as pra
    out = []
    for c in commands:
        seg = x[:, int(c["t0"] * fs): int(c["t1"] * fs)]
        X = np.array([pra.transform.stft.analysis(ch, nfft, nfft // 2, win=np.hanning(nfft)).T for ch in seg])   # [M, F, S]
        d = pra.doa.algorithms[algo](pos.T, fs, nfft, c=config.C_SOUND, num_src=1, azimuth=np.deg2rad(np.arange(360.0)))
        d.locate_sources(X, freq_range=[f_lo, f_hi])
        out.append(float(np.rad2deg(d.azimuth_recon[0]) % 360.0))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speech", required=True, help="folder of command recordings (16 kHz wav, text in .txt or the name)")
    ap.add_argument("--speech-pattern", default="*.wav")
    ap.add_argument("--out", default=os.path.join(_REPO, "reports", "doa_benchmark"))
    ap.add_argument("--quick", action="store_true", help="a small subset, to check that everything runs")
    ap.add_argument("--keep-audio", action="store_true", help="keep every rendered scene (WAV + truth.json)")
    ap.add_argument("params", nargs="*", help="doa_mod parameters as name=value")
    a = ap.parse_args(argv)
    params = {}
    for kv in a.params:
        k, v = kv.split("=", 1)
        params[k] = {"true": True, "false": False}.get(v.lower(), v)
        if params[k] is v:
            try:
                params[k] = float(v)
            except ValueError:
                pass
    os.makedirs(a.out, exist_ok=True)
    doa, rosmock = score_doa.import_doa(os.path.join(_REPO, "ros2", "doa_mod"))
    speech = dict(dir=os.path.abspath(os.path.expanduser(a.speech)), pattern=a.speech_pattern)
    rows, t_start = [], time.time()
    for matrix, cond, sc in matrices(speech, a.quick):
        signals, truth = scene_mixer.render(sc, root=_REPO)
        if a.keep_audio:
            render.write_scene(os.path.join(a.out, "scenes", sc["name"]), signals, truth)
        x = signals["respeaker_4mic"].astype(np.float64)
        pos = np.asarray(truth["arrays"]["respeaker_4mic"])
        pubs, node = score_doa.run_node(doa, rosmock, x, **params)
        cmds = truth["commands"]
        est = {"dorai": [], "srp": baseline(x, pos, cmds, "SRP"), "music": baseline(x, pos, cmds, "MUSIC")}
        owned = np.zeros(len(pubs), bool)
        for c in cmds:
            m = (pubs[:, 0] >= c["t0"] + 0.2) & (pubs[:, 0] <= c["t1"] + 1.1) if len(pubs) else np.zeros(0, bool)
            owned |= m
            est["dorai"].append(float(pubs[m][-1, 1]) if m.any() else float("nan"))
        node.shutdown()
        for i, c in enumerate(cmds):
            rows.append(dict(matrix=matrix, scene=sc["name"], **cond, truth=c["bearing_deg"], rt60_measured=truth["room"]["rt60_s"],
                             dorai=est["dorai"][i], srp=est["srp"][i], music=est["music"][i],
                             stray=int((~owned).sum()) if i == 0 else 0, seconds=truth["seconds"] if i == 0 else 0.0))
        print(f"{sc['name']:<26} truth {cmds[0]['bearing_deg']:5.1f} | dorai {est['dorai']} | srp {est['srp']} | music {est['music']}", flush=True)

    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)

    def table(title, keyname, keys, sel):
        lines = [f"### {title}", "", f"| {keyname} | commands | algorithm | detected | MAE (deg) | RMSE | median | max | within 15 deg |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
        for k in keys:
            rr = [r for r in rows if sel(r, k)]
            for algo in ("dorai", "srp", "music"):
                s = M.summary([r[algo] for r in rr], [r["truth"] for r in rr])
                if "mae" in s:
                    lines.append(f"| {k} | {s['n']} | {algo} | {s['detected']} | {s['mae']:.1f} | {s['rmse']:.1f} | {s['median']:.1f} | {s['max']:.1f} | {100 * s['within']:.0f} % |")
                else:
                    lines.append(f"| {k} | {s['n']} | {algo} | 0 | | | | | |")
        return lines + [""]

    md = [f"# DoA benchmark", "", f"doa_mod parameters: {params or 'defaults'}. {len(rows)} commands in {len({r['scene'] for r in rows})} scenes, "
          f"{time.time() - t_start:.0f} s. Simulation results (pyroomacoustics), not hardware.", ""]
    md += table("A. Free field, quiet, 2 m", "azimuth", sorted({r["az"] for r in rows if r["matrix"] == "A"}), lambda r, k: r["matrix"] == "A" and r["az"] == k)
    md += table("B. Reverberation (target RT60, s), quiet", "RT60", sorted({r["rt60"] for r in rows if r["matrix"] == "B"}), lambda r, k: r["matrix"] == "B" and r["rt60"] == k)
    md += table("B. By distance (m), all RT60", "distance", sorted({r["dist"] for r in rows if r["matrix"] == "B"}), lambda r, k: r["matrix"] == "B" and r["dist"] == k)
    md += table("C. Welding noise, RT60 0.4 s, 2 m (SNR at the array, dB)", "SNR", sorted({r["snr"] for r in rows if r["matrix"] == "C"}), lambda r, k: r["matrix"] == "C" and r["snr"] == k)
    minutes = sum(r["seconds"] for r in rows) / 60.0
    md += [f"Bearings the node gave while nobody spoke: {M.false_detection_rate(sum(r['stray'] for r in rows), minutes):.2f} per minute "
           f"({sum(r['stray'] for r in rows)} in {minutes:.1f} min).", ""]
    rts = sorted({(r["rt60"], r["rt60_measured"]) for r in rows if r["matrix"] == "B"})
    md += ["RT60 reached, measured from the impulse responses (target: measured values): "
           + "; ".join(f"{t}: {sorted({m for tt, m in rts if tt == t})}" for t in sorted({t for t, _ in rts})), ""]
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(md))
    json.dump(dict(params=params, rows=rows), open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
