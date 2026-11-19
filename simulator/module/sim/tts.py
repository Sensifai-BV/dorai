"""Generated speech for the simulator: commands from text, so that no recording
of a person is needed and the text of every utterance is known.

    python -m sim.tts --out ~/dorai/sim/speech            # the library in config/commands.yaml

Uses the macOS `say` command (offline, on the development machine). Each file
is 16 kHz mono with a `.txt` next to it holding the text, which is how
acoustics.text_of() finds what was said. On Linux there is no `say`; generate
on the Mac and copy the folder, or put recorded commands with `.txt` files in a
folder of the same shape. (Piper, named in the design document, is not wired
in: it needs a voice model to be downloaded first.)

Synthetic voices are cleaner and more uniform than people. Results on them are
simulation results; recorded commands remain the reference.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def available_voices():
    if not shutil.which("say"):
        return []
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    return [line.split("  ")[0].strip() for line in out.splitlines() if line.strip()]


def synthesize(text, path, voice=None, rate_wpm=None):
    """Write `text` to `path` (16 kHz mono WAV) and the text to `<path>.txt`."""
    if not shutil.which("say"):
        raise RuntimeError("the macOS `say` command is not available on this machine")
    cmd = ["say", "-o", path, "--data-format=LEI16@16000"]
    if voice:
        cmd += ["-v", voice]
    if rate_wpm:
        cmd += ["-r", str(int(rate_wpm))]
    subprocess.run(cmd + [text], check=True)
    with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write(" ".join(text.replace(",", " ").replace(".", " ").split()) + "\n")
    return path


def build_library(out_dir, library=None):
    """Every command of the library in every voice and rate it lists."""
    import yaml
    from .acoustics import config_dir
    lib = yaml.safe_load(open(library or os.path.join(config_dir(), "commands.yaml"), encoding="utf-8"))
    have = set(available_voices())
    voices = [v for v in lib.get("voices", []) if v in have] or [None]
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for c in lib["commands"]:
        for v in voices:
            for r in lib.get("rates_wpm", [None]):
                name = f"{c['id']}_{(v or 'default').lower()}_{r or 'std'}.wav"
                path = os.path.join(out_dir, name)
                if not os.path.isfile(path):
                    synthesize(c["text"], path, v, r)
                made.append(path)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--library", default=None, help="command library YAML (default config/commands.yaml)")
    a = ap.parse_args(argv)
    files = build_library(os.path.expanduser(a.out), a.library)
    print(f"{len(files)} files in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
