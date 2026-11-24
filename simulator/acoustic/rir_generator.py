"""Room impulse responses for every (source, microphone) pair, computed by
pyroomacoustics (room.compute_rir), and their use."""
from __future__ import annotations

import numpy as np

from .room_engine import make_room


def impulse_responses(room_spec, mics_xyz, sources_xyz):
    """rirs[s][m]: the response from source s to microphone m (arrays of
    different lengths). One room computation for all of them."""
    room, _ = make_room(room_spec, mics_xyz)
    for p in sources_xyz:
        room.add_source(np.asarray(p, dtype=float))
    room.compute_rir()
    return [[np.asarray(room.rir[m][s]) for m in range(len(mics_xyz))] for s in range(len(sources_xyz))]


def propagate(x, irs):
    """The source signal at every microphone, [M, len(x)]. The travel time
    common to all microphones is dropped (so events keep their time in the
    scene); the differences between the microphones are kept exactly."""
    from scipy.signal import fftconvolve
    lag = min(int(np.argmax(np.abs(ir))) for ir in irs)
    out = np.stack([fftconvolve(x, ir)[lag:lag + len(x)] for ir in irs])
    return out
