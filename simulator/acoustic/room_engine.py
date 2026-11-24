"""Rooms: a description from config/rooms -> a pyroomacoustics room
(pra.ShoeBox with pra.Material walls; image sources, optional ray tracing).

Walls scatter (SCATTER) in the material rooms: a shoebox of perfect mirrors
traps sound between its parallel walls and rings for seconds, which a room
with furniture or machines in it does not.
"""
from __future__ import annotations

import numpy as np
import pyroomacoustics as pra

from .config import FS, load_yaml

SCATTER = 0.25


def load_room(room):
    """A room description: a name in config/rooms, or a dict; a dict with a
    `name` starts from that file and overrides it ({name: rt60, rt60_s: 0.8})."""
    if isinstance(room, str):
        spec = dict(load_yaml("rooms", room), name=room)
    else:
        spec = dict(room)
        if "name" in spec and "size_m" not in spec and "size" not in spec:
            spec = dict(load_yaml("rooms", spec["name"]), **spec)
    spec.setdefault("name", "custom")
    spec["size_m"] = spec.pop("size", None) or spec["size_m"]
    if "rt60" in spec:
        spec["rt60_s"] = spec.pop("rt60")
    return spec


def make_room(spec, mics_xyz):
    """The pyroomacoustics room for a description from load_room().

    Three levels, by what the description holds:
      anechoic: true         free field, the direct sound only
      rt60_s: 0.4            image sources only, one absorption for all walls from Sabine's formula
      materials: {...}       image sources to order 6 plus ray tracing, a material per wall, air absorption
    """
    size = np.asarray(spec["size_m"], dtype=float)
    scatter = float(spec.get("scatter", SCATTER))
    if spec.get("anechoic"):
        r = pra.ShoeBox(size, fs=FS, materials=pra.Material(1.0), max_order=0)
    elif "materials" in spec:
        mats = {k: pra.Material(v, scatter) for k, v in spec["materials"].items()}
        r = pra.ShoeBox(size, fs=FS, materials=mats, max_order=int(spec.get("max_order", 6)),
                        ray_tracing=True, air_absorption=True)
        r.set_ray_tracing(receiver_radius=0.3, n_rays=int(spec.get("n_rays", 20000)))
    else:
        absorption, order = pra.inverse_sabine(float(spec["rt60_s"]), size)
        trace = bool(spec.get("ray_tracing", False))
        r = pra.ShoeBox(size, fs=FS, materials=pra.Material(absorption, scatter if trace else None),
                        max_order=int(spec.get("max_order", order)), ray_tracing=trace,
                        air_absorption=bool(spec.get("air_absorption", False)))
        if trace:
            r.set_ray_tracing(receiver_radius=0.3, n_rays=int(spec.get("n_rays", 20000)))
    r.add_microphone_array(np.asarray(mics_xyz).T)
    return r, size


def rt60_of(ir, fs=FS):
    """Reverberation time from one impulse response (Schroeder, -5 to -25 dB, times 3)."""
    e = np.cumsum((ir ** 2)[::-1])[::-1]
    db = 10 * np.log10(e / e[0] + 1e-30)
    a, b = np.argmax(db <= -5), np.argmax(db <= -25)
    return float(3.0 * (b - a) / fs) if b > a else float("nan")
