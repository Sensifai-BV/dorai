"""Metrics for stt_mod: blind input SNR, system load, and WER/CER scoring.

Used two ways:

* online, by the node with ``-p metrics:=true``: one JSON record per final on
  /dorai_stt/metrics (and optionally appended to ``metrics_file``) carrying the
  estimated input SNR of that utterance, decode speed, lag, backlog and CPU
  temperature/clock/load;
* offline, by ``stt_mod.evaluate`` (WER/CER against reference transcripts).

The SNR here is a *blind estimate*: no clean reference exists on the robot.
The stream is cut into 20 ms frames; the noise floor is a low percentile of
the frame powers in the utterance window (which includes the pause before
and after the speech), and speech frames are those above floor + 6 dB.
SNR = 10 log10((P_speech - P_noise) / P_noise). It tracks true SNR well for
stationary noise and reads high for impulsive noise (welding crackle and
knocks raise frames that then count as "speech"). Treat it as an indicator of
how hard the input was, not a measurement. Frames below -80 dBFS (digital
silence from a gated front-end) are excluded from the noise floor.
"""
from __future__ import annotations

import collections
import os
import re
import time

import numpy as np

FS = 16000


class LevelMeter:
    """Running 20 ms frame powers of the stream, indexed by stream position."""

    def __init__(self, fs=FS, frame_ms=20, keep_s=60.0):
        self.fs = fs
        self.n = int(fs * frame_ms / 1000)
        self._rem = np.zeros(0, np.float32)
        self._pow = collections.deque(maxlen=int(keep_s * 1000 / frame_ms))
        self._frames = 0                     # frames seen in total

    def push(self, x):
        x = np.concatenate([self._rem, np.asarray(x, np.float32)])
        k = len(x) // self.n
        if k:
            p = (x[:k * self.n].reshape(k, self.n) ** 2).mean(1)
            for v in p:
                self._pow.append((self._frames, float(v)))
                self._frames += 1
        self._rem = x[k * self.n:]

    def window(self, t0_s, t1_s, floor_pct=10, speech_db=6.0):
        """Estimate levels for stream time [t0_s, t1_s)."""
        f0, f1 = int(t0_s * 1000 / 20), int(t1_s * 1000 / 20)
        p = np.array([v for i, v in self._pow if f0 <= i < f1])
        if p.size < 10:
            return {"snr_est_db": None, "speech_dbfs": None, "noise_dbfs": None, "speech_frac": None}
        p = np.maximum(p, 1e-12)
        # Ignore digital silence (< -80 dBFS) when locating the noise floor:
        # a gated front-end (voice_mod zeroes the pauses) would otherwise put
        # the floor at ~-120 dBFS and report absurd SNRs (~100 dB).
        live = p[p > 1e-8]
        if live.size < 10:
            return {"snr_est_db": None, "speech_dbfs": None, "noise_dbfs": None,
                    "speech_frac": 0.0, "gated": True}
        noise = float(np.percentile(live, floor_pct))
        sp = p[10 * np.log10(p) > 10 * np.log10(noise) + speech_db]
        if sp.size == 0:
            return {"snr_est_db": None, "speech_dbfs": None,
                    "noise_dbfs": round(10 * np.log10(noise), 1), "speech_frac": 0.0}
        ps = float(sp.mean())
        snr = 10 * np.log10(max(ps - noise, 1e-12) / noise)
        return {"snr_est_db": round(snr, 1), "speech_dbfs": round(10 * np.log10(ps), 1),
                "noise_dbfs": round(10 * np.log10(noise), 1),
                "speech_frac": round(sp.size / p.size, 3)}


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


class SystemProbe:
    """CPU temperature/clock/load and this process's CPU share (Linux; None elsewhere)."""

    def __init__(self):
        self._t, self._c = time.monotonic(), self._cpu()

    @staticmethod
    def _cpu():
        t = os.times()
        return t.user + t.system

    def sample(self):
        now, c = time.monotonic(), self._cpu()
        dt = now - self._t
        pct = 100.0 * (c - self._c) / dt if dt > 0 else None
        self._t, self._c = now, c
        temp = _read("/sys/class/thermal/thermal_zone0/temp")
        freq = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        load = _read("/proc/loadavg")
        return {"cpu_temp_c": round(int(temp) / 1000, 1) if temp else None,
                "cpu_freq_mhz": int(freq) // 1000 if freq else None,
                "load1": float(load.split()[0]) if load else None,
                "proc_cpu_pct": round(pct, 1) if pct is not None else None}


# ---------------------------------------------------------------- scoring
def norm_words(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] if x == y else 1 + min(prev[j - 1], prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def strip_name(ref, hyp, name):
    """Remove 'hey'/<name> from the reference and the best-matching leading
    0-3 hypothesis words (for an out-of-vocabulary name). No-op if absent."""
    if not name:
        return ref, hyp
    rc = [w for w in ref if w not in ("hey", name)]
    if len(rc) == len(ref):
        return ref, hyp
    lead = 1 + (ref[:1] == ["hey"])
    k = min(range(0, lead + 3), key=lambda k: (edit_distance(rc, hyp[k:]), k))
    return rc, hyp[k:]


def score(ref_text, hyp_text, name=""):
    r, h = norm_words(ref_text), norm_words(hyp_text)
    rc, hc = strip_name(r, h, name)
    rch, hch = "".join(r), "".join(h)
    return {"n_ref": len(r), "err": edit_distance(r, h),
            "n_ref_noname": len(rc), "err_noname": edit_distance(rc, hc),
            "n_chr": len(rch), "err_chr": edit_distance(list(rch), list(hch)),
            "empty": int(not h)}


def speech_end_s(x, fs=FS, win=480):
    n = len(x) // win
    if n == 0:
        return len(x) / fs
    r = np.sqrt((x[:n * win].reshape(n, win) ** 2).mean(1))
    thr = max(0.1 * r.max(), 3 * np.median(r))
    idx = np.nonzero(r > thr)[0]
    return (idx[-1] + 1) * win / fs if idx.size else len(x) / fs
