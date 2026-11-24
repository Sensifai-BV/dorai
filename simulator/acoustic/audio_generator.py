"""Render scenario files to audio: one WAV per microphone array plus truth.json.

    python -m simulator.acoustic.audio_generator config/scenarios/factory_welding_0db.yaml --out ~/dorai/sim
    python -m simulator.acoustic.audio_generator config/scenarios/*.yaml --out ~/dorai/sim --speech-dir my_commands --seeds 0 1 2

Each scenario becomes <out>/<name>[_s<seed>]/ with artaban_line3.wav, respeaker_4mic.wav
(float32, 16 kHz, one channel per microphone) and truth.json. ros2/dorai_simulator plays such a folder
into the ROS graph; evaluation/ scores what came out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def load_scenario(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        sc = yaml.safe_load(f) or {}
    sc.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return sc


def write_scene(out_dir, signals, truth):
    import soundfile as sf
    os.makedirs(out_dir, exist_ok=True)
    for name, x in signals.items():
        sf.write(os.path.join(out_dir, name + ".wav"), x.T, truth["fs"], subtype="FLOAT")
    with open(os.path.join(out_dir, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=1)


def main(argv=None):
    from . import scene_mixer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", nargs="+", help="scenario YAML files")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--root", default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")),
                    help="folder that relative audio paths in the scenarios refer to (default: the repo root)")
    ap.add_argument("--speech-dir", default=None, help="folder of command recordings, overrides the scenarios' speech.dir")
    ap.add_argument("--speech-pattern", default=None, help="file pattern inside the speech folder (default *.wav)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="render each scenario once per seed")
    ap.add_argument("--keep-parts", action="store_true",
                    help="also write <array>.speech.wav and <array>.noise.wav: each part of the mix on its own (clean reference)")
    ap.add_argument("--force", action="store_true", help="render again even if the scene exists")
    a = ap.parse_args(argv)
    for path in a.scenarios:
        base = load_scenario(path)
        for seed in (a.seeds if a.seeds else [base.get("seed", 0)]):
            sc = dict(base, seed=seed)
            if a.speech_dir:
                sc["speech"] = dict(sc.get("speech") or {}, dir=a.speech_dir, files=None)
            if a.speech_pattern:
                sc["speech"] = dict(sc.get("speech") or {}, pattern=a.speech_pattern)
            name = base["name"] + (f"_s{seed}" if a.seeds else "")
            sc["name"] = name
            out_dir = os.path.join(os.path.expanduser(a.out), name)
            if os.path.isfile(os.path.join(out_dir, "truth.json")) and not a.force:
                print(f"{name}: exists, skipped")
                continue
            t0 = time.time()
            signals, truth = scene_mixer.render(sc, root=a.root, keep_parts=a.keep_parts)
            write_scene(out_dir, signals, truth)
            print(f"{name}: {truth['seconds']:.0f} s, room {truth['room']['name']} (RT60 {truth['room']['rt60_s']:.2f} s), "
                  f"{len(truth['commands'])} commands, {len(truth['noise'])} noise source(s), rendered in {time.time() - t0:.0f} s",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
