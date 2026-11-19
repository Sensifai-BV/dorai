"""Turn a one-clock render into what separate capture devices deliver.

The render is the sound at every microphone on one ideal clock. A real
microphone is its own USB device: it starts when its stream is opened (not
together with the others), samples at its own crystal's rate (nominal 48 kHz,
off by some tens of ppm), and hands its audio to the host in blocks with a
capture time stamp that jitters. `device_blocks` produces exactly that, as a
time-ordered list the virtual microphone node plays back in real time.

No ROS in this file, so the clock model can be tested on its own.
"""
from __future__ import annotations

import numpy as np

HEADER = 4       # [seq, rate, stamp // 1000, stamp % 1000] -- voice_mod.parse_device_msg reads this


def device_signal(x, fs, rate, ppm=0.0, start_s=0.0):
    """One microphone's samples as its device takes them: from `start_s` on
    (the stream was opened then), at rate * (1 + ppm * 1e-6) samples per second
    of real time -- which the device itself calls `rate`."""
    from scipy.signal import resample
    x = np.asarray(x, dtype=np.float64)[int(round(start_s * fs)):]
    n = int(round(len(x) / fs * rate * (1.0 + ppm * 1e-6)))
    return resample(x, n).astype(np.float32)          # band-limited, exact length


def device_blocks(signals, fs, rate=48000, block_ms=100.0, ppm=None, start_ms=None, stamp_jitter_ms=0.0, seed=0):
    """All blocks of all devices, ordered by the time they reach the host.

    signals: [M, n] one-clock render. Returns a list of dicts
        device   index of the microphone
        seq      block number of that device
        due_s    when the block is complete and delivered, seconds from scene start
        stamp_s  capture time of its first sample as the host is told (true time + jitter)
        samples  float32 block
    """
    m = len(signals)
    ppm = list(ppm) if ppm else [0.0] * m
    start = [v / 1000.0 for v in (start_ms if start_ms else [0.0] * m)]
    if len(ppm) != m or len(start) != m:
        raise ValueError(f"ppm and start_ms need one value per microphone ({m})")
    rng = np.random.default_rng(seed)
    blk = int(round(rate * block_ms / 1000.0))
    out = []
    for i in range(m):
        y = device_signal(signals[i], fs, rate, ppm[i], start[i])
        true_rate = rate * (1.0 + ppm[i] * 1e-6)
        for k in range(len(y) // blk):
            t_first = start[i] + k * blk / true_rate
            out.append(dict(device=i, seq=k, due_s=start[i] + (k + 1) * blk / true_rate,
                            stamp_s=t_first + float(rng.normal(0.0, stamp_jitter_ms / 1000.0)),
                            samples=y[k * blk:(k + 1) * blk]))
    out.sort(key=lambda b: (b["due_s"], b["device"]))
    return out


def device_message(block, rate, t0_mono):
    """The Float32MultiArray payload for one block (see HEADER)."""
    stamp = t0_mono + block["stamp_s"]
    head = np.array([block["seq"], rate, stamp // 1000.0, stamp % 1000.0], dtype=np.float32)
    return np.concatenate((head, block["samples"]))


def array_frames(signals, fs, frame_ms=100.0):
    """The same render as ONE multichannel device delivers it (for doa_mod):
    (due_s, payload) pairs in voice_mod's raw-tap layout
    [capture_s, rate, n_channels, seq, interleaved samples...]."""
    x = np.asarray(signals, dtype=np.float32)
    n = int(round(fs * frame_ms / 1000.0))
    out = []
    for k, a in enumerate(range(0, x.shape[1] - n + 1, n)):
        head = np.array([a / fs, fs, x.shape[0], k], dtype=np.float32)
        out.append(((a + n) / fs, np.concatenate((head, x[:, a:a + n].T.reshape(-1)))))
    return out
