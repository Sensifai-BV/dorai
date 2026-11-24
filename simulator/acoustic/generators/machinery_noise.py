"""A machine: a rotating part (harmonics of its rotation rate, the speed
wandering a little), broadband mechanical noise, and low-frequency vibration.
A model for controlled tests; recordings (simulator/lab/wavs/factory.wav, or
the MIMII set) are the reference."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def generate_machinery_noise(duration, sample_rate=16000, seed=42, rotation_hz=48.0, harmonics=12, speed_wander=0.02):
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    wander = np.cumsum(rng.normal(0, 1, n))
    wander = sosfilt(butter(2, 0.5, btype="lowpass", fs=sample_rate, output="sos"), wander - wander.mean())
    wander = wander / (np.max(np.abs(wander)) + 1e-9) * speed_wander
    phase = 2 * np.pi * np.cumsum(rotation_hz * (1.0 + wander)) / sample_rate
    tone = sum(rng.uniform(0.3, 1.0) / (k ** 0.7) * np.sin(k * phase + rng.uniform(0, 6.28))
               for k in range(1, harmonics + 1) if k * rotation_hz < 0.45 * sample_rate)
    broad = sosfilt(butter(2, [150, 3000], btype="bandpass", fs=sample_rate, output="sos"), rng.normal(0, 1, n))
    broad *= 1.0 + 0.4 * np.sin(phase)                               # the noise breathes with the rotation
    vib = sosfilt(butter(2, 60, btype="lowpass", fs=sample_rate, output="sos"), rng.normal(0, 1, n)) * 3.0
    x = tone / (np.std(tone) + 1e-9) + 0.6 * broad / (np.std(broad) + 1e-9) + 0.4 * vib / (np.std(vib) + 1e-9)
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)
