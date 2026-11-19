"""Score doa_mod on rendered scenes, offline: the real node, no ROS.

    python -m sim.score_doa ~/dorai/sim [--doa-mod ros2/doa_mod] [name=value ...]

Every scene's array recording is fed to the DoaMod node as /dorai_raw_audio
frames (the node runs on doa_mod/test/rosmock.py, so its own worker, gate,
publish rule and hangover decide what comes out), and each published bearing
is compared with where the talker of that command really stood. `name=value`
pairs are node parameters (subtract_noise=true, avg_tc=1.0, ...).

What is scored is what a consumer of /dorai_doa holds when a command has been
spoken: the LAST bearing published for it. "stray" bearings are those
published while nobody spoke.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def load_scene(folder, array="respeaker_4mic"):
    import soundfile as sf
    truth = json.load(open(os.path.join(folder, "truth.json"), encoding="utf-8"))
    x, fs = sf.read(os.path.join(folder, array + ".wav"), dtype="float64", always_2d=True)
    assert fs == truth["fs"]
    return x.T, truth


def import_doa(doa_mod_dir):
    """doa_mod's node module, on the test mock of rclpy (no ROS needed)."""
    sys.path[:0] = [os.path.join(doa_mod_dir, "test"), doa_mod_dir]
    import rosmock
    rosmock.install()
    import doa
    return doa, rosmock


def run_node(doa, rosmock, x, frame_s=0.1, **params):
    """Play [M, n] through a DoaMod node. Returns ([(audio time, bearing)], node)."""
    p = dict(source="topic", publish_human=False)
    p.update(params)
    rosmock.set_overrides(p)
    node = doa.DoaMod()
    node._running = False
    if node._worker.is_alive():
        node._worker.join(timeout=2.0)
    n = int(round(frame_s * doa.FS))
    for k, a in enumerate(range(0, x.shape[1], n)):
        msg = type("Msg", (), {})()
        msg.data = np.concatenate(([0.0, float(doa.FS), float(x.shape[0]), float(k)], x[:, a:a + n].T.reshape(-1)))
        node.src._on_msg(msg)
        while node._step():
            pass
    state = node.publishers_by_topic["/dorai_doa/state"].sent
    # state[0] counts hops from the first block; the audio time of a bearing is the end of its block
    pubs = np.array([(m.data[0] + (doa.BLOCK - doa.HOP) / doa.FS, m.data[1]) for m in state]).reshape(-1, 2)
    return pubs, node


def angle_error(a, b):
    return np.abs((np.asarray(a) - b + 180.0) % 360.0 - 180.0)


def score(pubs, commands, hangover_s=0.8):
    """Per command the error of the last bearing published for it (NaN: none),
    and the number of bearings that belong to no command."""
    errs, owned = [], np.zeros(len(pubs), bool)
    for c in commands:
        m = (pubs[:, 0] >= c["t0"] + 0.2) & (pubs[:, 0] <= c["t1"] + hangover_s + 0.3)
        owned |= m
        errs.append(float(angle_error(pubs[m][-1, 1], c["bearing_deg"])) if m.any() else float("nan"))
    return errs, int((~owned).sum())


def summarise(errs):
    e = np.asarray(errs, dtype=float)
    ok = e[np.isfinite(e)]
    if not len(ok):
        return dict(n=len(e), none=len(e))
    return dict(n=len(e), none=int(np.isnan(e).sum()), median=float(np.median(ok)), p90=float(np.percentile(ok, 90)),
                within15=float(np.mean(ok <= 15)), within30=float(np.mean(ok <= 30)))


def fmt(s):
    if "median" not in s:
        return f"{s['n']} commands, no bearing for any"
    return (f"{s['n']:>3} commands | no bearing {s['none']:>2} | median error {s['median']:5.1f} deg | 90th pct {s['p90']:5.1f} | "
            f"within 15 deg {100 * s['within15']:3.0f} % | within 30 deg {100 * s['within30']:3.0f} %")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", help="folder of rendered scenes")
    ap.add_argument("params", nargs="*", help="doa_mod parameters as name=value")
    ap.add_argument("--doa-mod", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "ros2", "doa_mod"))
    ap.add_argument("--array", default="respeaker_4mic")
    ap.add_argument("--json", default=None, help="also write the per-command results here")
    a = ap.parse_args(argv)
    params = {}
    for kv in a.params:
        k, v = kv.split("=", 1)
        params[k] = {"true": True, "false": False}.get(v.lower(), v)
        try:
            params[k] = float(v) if params[k] is v else params[k]
        except ValueError:
            pass
    doa, rosmock = import_doa(os.path.abspath(a.doa_mod))
    groups, rows = {}, []
    for folder in sorted(glob.glob(os.path.join(os.path.expanduser(a.scenes), "*", ""))):
        if not os.path.isfile(os.path.join(folder, "truth.json")):
            continue
        x, truth = load_scene(folder, a.array)
        pubs, node = run_node(doa, rosmock, x, **params)
        errs, stray = score(pubs, truth["commands"], float(params.get("vad_hangover_ms", 800.0)) / 1000.0)
        base = truth["name"].rsplit("_s", 1)[0] if truth["name"].rsplit("_s", 1)[-1].isdigit() else truth["name"]
        g = groups.setdefault(base, dict(errs=[], stray=0, minutes=0.0, misfit=[]))
        g["errs"] += errs; g["stray"] += stray; g["minutes"] += truth["seconds"] / 60.0
        g["misfit"].append(node.timing.misfit_us)
        rows.append(dict(scene=truth["name"], errors=errs, stray=stray, misfit_us=node.timing.misfit_us,
                         commands=truth["commands"], bearings=pubs.tolist()))
        node.shutdown()
    print(f"doa_mod parameters: {params or 'defaults'}")
    for name, g in groups.items():
        m = np.asarray(g["misfit"], dtype=float)
        print(f"{name:<28} {fmt(summarise(g['errs']))} | stray {g['stray'] / g['minutes']:4.1f}/min | "
              f"timing misfit {'n/a' if not np.isfinite(m).any() else f'{np.nanmax(m):.0f} us'}")
    print(f"{'ALL':<28} {fmt(summarise(sum((g['errs'] for g in groups.values()), [])))}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
