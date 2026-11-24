"""Sound sources: command recordings with their text, and noise.

A noise source is a recording (`file:`) or a generated signal (`type:`
welding_noise | machinery_noise | ventilation_noise, see generators/). Either
way it is one point in the room with its own impulse responses; the same noise
is never simply added to all channels.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from .config import FS


def text_of(path):
    """What is said in a speech file: `<file>.txt` next to it, else the file
    name ('07_Artaban_come_here_raw16k.wav' -> 'Artaban come here'), else None
    for names that are not words (corpus ids such as 481-123719-0000)."""
    side = os.path.splitext(path)[0] + ".txt"
    if os.path.isfile(side):
        return " ".join(open(side, encoding="utf-8").read().split())
    words = [w for w in os.path.splitext(os.path.basename(path))[0].replace("-", "_").split("_")
             if w and not w.isdigit() and w.lower() not in ("raw16k", "raw", "16k", "clean")]
    if not words or not all(w.isalpha() for w in words):
        return None
    return " ".join(words)


def read_mono(path, fs=FS):
    import soundfile as sf
    x, rate = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if rate != fs:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(rate), fs)
        x = resample_poly(x, fs // g, int(rate) // g)
    return x


def speech_files(spec, root="."):
    if spec.get("files"):
        files = [f if os.path.isabs(f) else os.path.join(root, f) for f in spec["files"]]
    else:
        d = spec.get("dir") or os.environ.get("DORAI_SIM_SPEECH") or os.path.join(root, "simulator", "lab", "samples", "speech")
        d = d if os.path.isabs(d) else os.path.join(root, d)
        files = sorted(glob.glob(os.path.join(d, spec.get("pattern", "*.wav"))))
    if not files:
        raise FileNotFoundError(f"no speech files for {spec}")
    return files


def trim(x, pad=0.05, fs=FS):
    """Cut the silence before and after the speech, keeping `pad` seconds.
    Speech is where the level of a 20 ms window is within 30 dB of the loudest
    one (single samples are no guide: a recording's noise floor has peaks)."""
    w = int(0.02 * fs)
    n = len(x) // w
    if n < 3:
        return x
    level = np.sqrt(np.mean(x[:n * w].reshape(n, w) ** 2, axis=1))
    act = np.flatnonzero(level > level.max() * 10 ** (-30 / 20))
    return x[max(0, act[0] * w - int(pad * fs)): (act[-1] + 1) * w + int(pad * fs)]


def noise_signal(spec, n, rng, root="."):
    """`n` samples of a noise source's own signal, and a label for the truth file."""
    kind = spec.get("type")
    if kind:
        from . import generators
        gen = generators.NOISE[kind]
        return gen(n / FS, FS, seed=int(rng.integers(0, 2 ** 31)), **(spec.get("params") or {}))[:n].astype(np.float64), kind
    src = spec["file"] if os.path.isabs(spec["file"]) else os.path.join(root, spec["file"])
    x = read_mono(src)
    x = np.tile(x, int(np.ceil((n + FS) / len(x))) + 1)
    a = int(rng.integers(0, len(x) - n - 1))
    return x[a:a + n], os.path.basename(src)
