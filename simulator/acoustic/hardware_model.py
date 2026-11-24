"""Microphone hardware imperfections, each one switched on by its own setting.

    hardware:
      gain_db_sd: 0.5            # channel gain mismatch, standard deviation
      position_error_mm: 1.0     # capsule placement error, standard deviation per axis
      self_noise_snr_db: 40.0    # independent noise per channel, below the speech level
      channel_delay_samples: null   # e.g. [0, 0.25, 0, 0]: a fixed timing offset per channel
      quantization_bits: null    # e.g. 16: ADC resolution
      clip: false                # true: samples beyond full scale are clipped, not rescaled

These are simulation parameters, not measured properties of the ReSpeaker or
of the robot's USB microphones. Measured values (calibration) override them by
being written into the scenario or the robot file. Not modelled: per-channel
frequency response and frequency-dependent phase. Clock drift between separate
capture devices is not here either: it belongs to the capture path, see
ros2/dorai_simulator (ppm per device).
"""
from __future__ import annotations

import numpy as np

DEFAULTS = dict(gain_db_sd=0.5, position_error_mm=1.0, self_noise_snr_db=40.0,
                channel_delay_samples=None, quantization_bits=None, clip=False)


def perturb_positions(pos, cfg, rng):
    """Where the capsules really are: the nominal positions plus placement error."""
    return pos + rng.normal(0.0, float(cfg["position_error_mm"]) / 1000.0, pos.shape)


def delay(x, samples):
    """x delayed by a (fractional) number of samples, exact for band-limited x."""
    n = len(x)
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * samples), n)


def apply(mix, cfg, rng, speech_power):
    """The signals as the hardware delivers them. mix: [M, n] at the capsules.
    Returns (signals, report): report says what was applied, per channel."""
    m = mix.shape[0]
    gains = rng.normal(0.0, float(cfg["gain_db_sd"]), m)
    out = mix * 10 ** (gains[:, None] / 20)
    delays = cfg.get("channel_delay_samples")
    if delays is not None:
        delays = [float(d) for d in delays][:m] + [0.0] * max(0, m - len(delays))
        out = np.stack([delay(ch, d) if d else ch for ch, d in zip(out, delays)])
    out = out + rng.standard_normal(out.shape) * np.sqrt(speech_power) * 10 ** (-float(cfg["self_noise_snr_db"]) / 20)
    peak = float(np.abs(out).max())
    scale = 1.0
    if cfg.get("clip"):
        clipped = float(np.mean(np.abs(out) > 1.0))
        out = np.clip(out, -1.0, 1.0)
    else:
        clipped = 0.0
        scale = 0.95 / peak if peak > 0.95 else 1.0
        out = out * scale
    bits = cfg.get("quantization_bits")
    if bits:
        q = 2.0 ** (int(bits) - 1)
        out = np.round(out * q) / q
    return out, dict(gain_db=[float(g) for g in gains], delay_samples=delays, scale=scale,
                     clipped_fraction=clipped, quantization_bits=bits)
