"""doa_mod on real recordings: the LOCATA corpus, single-source tasks.

    python -m experiments.locata_doa --corpus /path/to/data-corpus --out reports/locata_doa

Tasks 1 (a loudspeaker that stands still), 3 (a person walking) and 5 (a person
walking, the array moving too). From the Eigenmike, a 42 mm sphere, the
microphones on its equator are used as a small planar array (4 of them by
default: the nearest thing in the corpus to the ReSpeaker; --mics 8 takes all
eight). The node gets their real positions as positions_mm. Each bearing the
node publishes is compared with where the source really was at that moment.

What this is not: the ReSpeaker. The capsules sit on a rigid sphere, which
bends the sound around it, so the delays are not the free-field ones the
estimator assumes. It tests the estimator on real rooms and real talkers, not
the DorAI hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_REPO]

from evaluation import doa_metrics as M, locata                            # noqa: E402
from evaluation import doa_node as score_doa                               # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default=os.path.join(_REPO, "reports", "locata_doa"))
    ap.add_argument("--array", default="eigenmike")
    ap.add_argument("--mics", type=int, default=4)
    ap.add_argument("--tasks", type=int, nargs="*", default=[1, 3, 5])
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
    rows, lines = [], []
    for folder in locata.recordings(a.corpus, a.tasks, a.array):
        rec = locata.load(folder)
        if len(rec["sources"]) != 1:
            continue
        name = next(iter(rec["sources"]))
        idx, xy = locata.planar_subset(rec["mics_local"], a.mics)
        x = rec["audio"][idx]
        x = x / max(np.abs(x).max(), 1e-9) * 0.5
        p = dict(positions_mm=" ".join(f"{v * 1000:.2f}" for v in xy.reshape(-1)), channels=list(range(len(idx))))
        p.update(params)
        pubs, node = score_doa.run_node(doa, rosmock, x, **p)
        src = rec["sources"][name]
        speech_s = float(src["vad"].sum() / rec["fs"]) if src["vad"] is not None else float("nan")
        label = "/".join(folder.split(os.sep)[-3:-1])
        if len(pubs):
            truth = locata.truth_at(rec, name, pubs[:, 0])
            err = np.abs(M.angular_error(pubs[:, 1], truth))
            in_speech = src["vad"][np.minimum((pubs[:, 0] * rec["fs"]).astype(int), len(src["vad"]) - 1)] if src["vad"] is not None else np.ones(len(pubs), bool)
            s = M.summary(pubs[:, 1], truth)
            row = dict(recording=label, source=name, seconds=round(x.shape[1] / rec["fs"], 1), speech_s=round(speech_s, 1),
                       array_moves=rec["array_moves"], bearings=len(pubs), bearings_outside_speech=int((~in_speech).sum()),
                       distance_m=round(float(np.median(src["distance"])), 2), elevation_deg=round(float(np.median(src["elevation"])), 1),
                       azimuth_span_deg=round(float(np.ptp(np.unwrap(np.deg2rad(src["azimuth"])))) * 180 / np.pi, 0),
                       mae=s["mae"], median=s["median"], p90=s["p90"], within15=s["within"], misfit_us=node.timing.misfit_us,
                       errors=err.round(1).tolist(), times=pubs[:, 0].round(2).tolist())
        else:
            row = dict(recording=label, source=name, seconds=round(x.shape[1] / rec["fs"], 1), speech_s=round(speech_s, 1),
                       array_moves=rec["array_moves"], bearings=0)
        node.shutdown()
        rows.append(row)
        lines.append(f"| {label} | {name} | {row['seconds']} | {row['speech_s']} | {'yes' if row['array_moves'] else 'no'} | {row['bearings']} | "
                     + (f"{row['mae']:.1f} | {row['median']:.1f} | {row['p90']:.1f} | {100 * row['within15']:.0f} % | {row['distance_m']} | {row['azimuth_span_deg']:.0f} |"
                        if row["bearings"] else "| | | | | |"))
        print(lines[-1], flush=True)
    allerr = np.array(sum((r.get("errors", []) for r in rows), []))
    md = ["# doa_mod on LOCATA (real recordings)", "",
          f"Array: {a.array}, {a.mics} equatorial microphones as a planar array. doa_mod parameters: {params or 'defaults'}.", "",
          "| recording | source | length (s) | speech (s) | array moves | bearings | MAE (deg) | median | 90th pct | within 15 deg | distance (m) | azimuth covered (deg) |",
          "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"] + lines + [""]
    for task in sorted({r["recording"].split("/")[0] for r in rows}):
        e = np.array(sum((r.get("errors", []) for r in rows if r["recording"].startswith(task)), []))
        if len(e):
            md.append(f"{task}: {len(e)} bearings, MAE {e.mean():.1f} deg, median {np.median(e):.1f}, within 15 deg {100 * np.mean(e <= 15):.0f} %.")
    if len(allerr):
        md += ["", f"All: {len(allerr)} bearings, MAE {allerr.mean():.1f} deg, median {np.median(allerr):.1f}, 90th pct {np.percentile(allerr, 90):.1f}, "
               f"within 15 deg {100 * np.mean(allerr <= 15):.0f} %.", ""]
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(md))
    json.dump(dict(params=params, rows=rows), open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print("\n".join(md[-8:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
