"""Direction-of-arrival metrics. Angles in degrees; every difference is taken
on the circle, so 359 and 1 are 2 degrees apart."""
from __future__ import annotations

import numpy as np


def wrap(delta):
    """An angle difference in [-180, 180)."""
    return (np.asarray(delta, dtype=float) + 180.0) % 360.0 - 180.0


def angular_error(estimate, truth):
    """Signed error estimate - truth, wrapped; NaN where there is no estimate."""
    return wrap(np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float))


def summary(estimates, truths, threshold_deg=15.0):
    """The metrics of one condition. `estimates` may hold NaN: no bearing was
    given for that utterance (a missed detection, counted, not averaged)."""
    e = np.abs(angular_error(estimates, truths))
    ok = e[np.isfinite(e)]
    out = dict(n=int(e.size), detected=int(ok.size), detection_rate=float(ok.size / e.size) if e.size else float("nan"))
    if ok.size:
        out.update(mae=float(ok.mean()), rmse=float(np.sqrt((ok ** 2).mean())), median=float(np.median(ok)),
                   p90=float(np.percentile(ok, 90)), max=float(ok.max()),
                   within=float((ok <= threshold_deg).mean()), threshold_deg=float(threshold_deg))
    return out


def false_detection_rate(n_stray, minutes):
    """Bearings given while nobody spoke, per minute."""
    return float(n_stray / minutes) if minutes > 0 else float("nan")


def tracking_error(times, estimates, truth_times, truth_angles):
    """Mean absolute error against a moving truth (linear in time between its
    samples, interpolated on the circle)."""
    z = np.exp(1j * np.deg2rad(np.asarray(truth_angles, dtype=float)))
    zi = np.interp(times, truth_times, z.real) + 1j * np.interp(times, truth_times, z.imag)
    return float(np.abs(angular_error(estimates, np.rad2deg(np.angle(zi)))).mean())
