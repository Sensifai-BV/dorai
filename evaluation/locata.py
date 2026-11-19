"""Reader for the LOCATA corpus (real recordings with measured source and array
positions): https://www.locata.lms.tf.fau.de/corpus/

A recording folder holds one sub-folder per array (eigenmike, benchmark2,
dicit, dummy) with the audio (48 kHz), the array's position, orientation and
microphone positions over time (OptiTrack, about 120 Hz), and the same for each
source. Nothing here assumes a DorAI geometry: positions are taken from the
recording itself, and a subset of microphones is chosen by where they are.

Frames. LOCATA gives everything in the room's frame; the array's own frame is
local = R^T (global - array position), with R the rotation in the position
file. That was checked on the Eigenmike: its 32 capsules then land exactly on
the published design angles (elevations 0 and +-21 deg, radius 42 mm).
A bearing is the azimuth of the source in that local frame, counter-clockwise
from local +x.
"""
from __future__ import annotations

import glob
import os

import numpy as np


def _table(path):
    return np.loadtxt(path, skiprows=1, ndmin=2)


def _seconds(t):
    """Seconds of the day from the hour/minute/second columns."""
    return t[:, 3] * 3600.0 + t[:, 4] * 60.0 + t[:, 5]


def recordings(root, tasks=(1, 3, 5), array="eigenmike"):
    """Folders <root>/task<k>/recording<n>/<array> that exist."""
    out = []
    for k in tasks:
        out += sorted(glob.glob(os.path.join(root, f"task{k}", "recording*", array)))
    return out


def load(folder, fs_out=16000):
    """One array of one recording.

    Returns a dict: audio [C, n] at fs_out, fs, mics_local [C, 3] (metres, the
    array's own frame), t [T] seconds from the first audio sample, sources
    {name: dict(azimuth [T] deg, elevation [T] deg, distance [T] m, vad [n] bool)},
    valid [T] bool (the corpus' own flag for usable position samples).
    """
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    array = os.path.basename(os.path.normpath(folder))
    x, fs = sf.read(os.path.join(folder, f"audio_array_{array}.wav"), dtype="float64", always_2d=True)
    g = gcd(int(fs), fs_out)
    audio = resample_poly(x, fs_out // g, int(fs) // g, axis=0).T if fs != fs_out else x.T
    t_audio0 = _seconds(_table_head(os.path.join(folder, f"audio_array_timestamps_{array}.txt")))[0]
    a = _table(os.path.join(folder, f"position_array_{array}.txt"))
    t = _seconds(a) - t_audio0
    pos, R = a[:, 6:9], a[:, 12:21].reshape(-1, 3, 3)
    mics = a[:, 21:].reshape(len(a), -1, 3)
    mics_local = np.einsum("tmi,tij->tmj", mics - pos[:, None, :], R).mean(axis=0)
    valid = _table(os.path.join(folder, "required_time.txt"))[:, 6] > 0.5
    sources = {}
    for f in sorted(glob.glob(os.path.join(folder, "position_source_*.txt"))):
        name = os.path.basename(f)[len("position_source_"):-4]
        s = _table(f)
        loc = np.einsum("ti,tij->tj", s[:, 6:9] - pos, R)
        vad_file = os.path.join(folder, f"VAD_{array}_{name}.txt")
        vad = np.loadtxt(vad_file, skiprows=1) > 0.5 if os.path.isfile(vad_file) else None
        if vad is not None and fs != fs_out:
            vad = vad[np.minimum((np.arange(audio.shape[1]) * fs / fs_out).astype(int), len(vad) - 1)]
        sources[name] = dict(azimuth=np.rad2deg(np.arctan2(loc[:, 1], loc[:, 0])) % 360.0,
                             elevation=np.rad2deg(np.arcsin(loc[:, 2] / np.linalg.norm(loc, axis=1))),
                             distance=np.linalg.norm(loc, axis=1), vad=vad)
    return dict(folder=folder, array=array, audio=audio, fs=fs_out, mics_local=mics_local, t=t, valid=valid,
                sources=sources, array_moves=bool(np.ptp(pos, axis=0).max() > 0.01))


def _table_head(path, rows=2):
    with open(path) as f:
        f.readline()
        return np.array([[float(v) for v in f.readline().split()] for _ in range(rows)])


def planar_subset(mics_local, n=4, max_elevation_deg=5.0):
    """Indices of `n` microphones near the array's own horizontal plane, spread
    in azimuth as evenly as the array allows, and their (x, y) positions.
    This is how a small planar array is cut out of a spherical one."""
    r = np.linalg.norm(mics_local, axis=1)
    el = np.rad2deg(np.arcsin(mics_local[:, 2] / np.maximum(r, 1e-9)))
    cand = np.flatnonzero(np.abs(el) <= max_elevation_deg)
    if len(cand) < n:
        raise ValueError(f"only {len(cand)} microphones within {max_elevation_deg} deg of the array's plane")
    az = np.rad2deg(np.arctan2(mics_local[cand, 1], mics_local[cand, 0])) % 360.0
    order = cand[np.argsort(az)]
    az = np.sort(az)
    best, best_cost = None, np.inf
    from itertools import combinations
    for combo in combinations(range(len(order)), n):
        gaps = np.diff(np.r_[az[list(combo)], az[combo[0]] + 360.0])
        cost = float(np.ptp(gaps))
        if cost < best_cost:
            best, best_cost = combo, cost
    idx = order[list(best)]
    return idx, mics_local[idx, :2]


def truth_at(rec, source, times):
    """The source's bearing at `times` (seconds of audio), interpolated on the circle."""
    z = np.exp(1j * np.deg2rad(rec["sources"][source]["azimuth"]))
    zi = np.interp(times, rec["t"], z.real) + 1j * np.interp(times, rec["t"], z.imag)
    return np.rad2deg(np.angle(zi)) % 360.0
