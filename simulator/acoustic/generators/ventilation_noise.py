"""Ventilation or a compressor running steadily: noise that falls off towards
high frequencies, a weak blade tone, slow level changes. A model for tests."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def generate_ventilation_noise(duration, sample_rate=16000, seed=42, blade_hz=120.0, tone_level=0.15):
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    spec = np.fft.rfft(rng.normal(0, 1, n))
    f = np.fft.rfftfreq(n, 1.0 / sample_rate)
    spec /= np.maximum(f, 20.0) ** 0.8                               # about -5 dB per octave
    air = np.fft.irfft(spec, n)
    air = sosfilt(butter(2, 40, btype="highpass", fs=sample_rate, output="sos"), air)
    t = np.arange(n) / sample_rate
    slow = 1.0 + 0.15 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 6.28))
    x = air / (np.std(air) + 1e-9) * slow + tone_level * np.sin(2 * np.pi * blade_hz * t)
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)
