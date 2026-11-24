"""Microphone arrays: geometry and channel order from config/arrays/<name>.yaml.

Positions are in the robot's frame, metres: x is the robot's heading (0 deg),
y its left (90 deg); bearings are counter-clockwise from x, as doa_mod measures
them. The channel order of the file is the channel order of the audio.
"""
from __future__ import annotations

import numpy as np

from .config import load_yaml


def load_array(name):
    """[M, 2] microphone positions (x, y in metres) in channel order, from
    config/arrays/<name>.yaml. A list of positions is passed through."""
    if not isinstance(name, str):
        return np.asarray(name, dtype=float)[:, :2]
    mics = sorted(load_yaml("arrays", name)["array"]["microphones"], key=lambda m: m["channel"])
    return np.asarray([m["position_m"][:2] for m in mics], dtype=float)
