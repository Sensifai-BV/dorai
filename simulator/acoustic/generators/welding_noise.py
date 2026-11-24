"""A welding-like noise: broadband hiss with crackle bursts, in arcs with pauses.

A model for controlled tests, not a measurement of a welding process. The
recorded welding noise (simulator/lab/wavs/welding.wav, used as `file:`) is the
reference; results on this generator are results on a synthetic noise.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def generate_welding_noise(duration, sample_rate=16000, seed=42, arc_s=(1.5, 5.0), pause_s=(0.3, 1.5), bursts_per_s=8.0):
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    hiss = sosfilt(butter(4, [300, min(6500, 0.45 * sample_rate)], btype="bandpass", fs=sample_rate, output="sos"),
                   rng.normal(0, 1, n))
    rumble = sosfilt(butter(2, 200, btype="lowpass", fs=sample_rate, output="sos"), rng.normal(0, 1, n)) * 2.0
    env = np.ones(n)
    burst = int(0.015 * sample_rate)                                # crackle: short, strong
    for start in rng.integers(0, max(1, n - burst), int(bursts_per_s * duration)):
        env[start:start + burst] *= rng.uniform(1.5, 4.0)
    wobble = 1.0 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.3, 1.5) * np.arange(n) / sample_rate + rng.uniform(0, 6.28))
    arc = np.zeros(n)                                               # the arc is on, then off
    t = int(rng.uniform(0, 0.5) * sample_rate)
    ramp = int(0.02 * sample_rate)
    while t < n:
        on = int(rng.uniform(*arc_s) * sample_rate)
        seg = np.ones(min(on, n - t))
        k = min(ramp, len(seg) // 2)
        if k:
            seg[:k] = np.linspace(0, 1, k); seg[-k:] = np.linspace(1, 0, k)
        arc[t:t + len(seg)] = seg
        t += on + int(rng.uniform(*pause_s) * sample_rate)
    x = (hiss * env * wobble + 0.3 * rumble) * (0.03 + 0.97 * arc)
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)
