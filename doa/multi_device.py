"""multi_device — capture from SEVERAL independent USB microphones as one array.

Read this before trusting a bearing from it
-------------------------------------------
One multichannel device (the ReSpeaker) samples every mic from a single crystal,
so the only inter-channel delay is acoustic and DoA is well posed. N separate
USB microphones are N independent clocks, and that breaks two assumptions that
single-device DoA silently relies on:

1. NO COMMON TIME ORIGIN. Each stream starts when its callback first fires.
   The resulting fixed offset is tens of milliseconds -- hundreds of times the
   entire acoustic range -- so raw cross-correlation is meaningless until it is
   removed. `align_streams()` removes the bulk of it; DriftTracker handles the
   remainder.

2. THE OFFSET DOES NOT STAY PUT. At 1 ppm relative error the offset moves 1 us
   per second. Cheap USB crystals run +-50 to +-100 ppm, so a pair drifts
   50-100 us/s; a 100 mm baseline spans only 583 us end to end. The bearing
   therefore rotates on its own within seconds. This is physics of the
   hardware, not a software defect, and no amount of code makes independent
   clocks sample-synchronous.

What you get, honestly stated
-----------------------------
  * relative bearing CHANGES, tracked well over seconds
  * absolute bearing ONLY after calibrate_offset() against a known source
    position, and only until the clocks walk away again
  * a drift-rate readout, so the display can tell you how fast it is happening

If you need dependable absolute DoA, use one multichannel interface (the
ReSpeaker, or any multi-input audio interface where all mics share a clock).
That is a hardware fix, and it is the right one.
"""
from __future__ import annotations

import sys
import threading
import time

import numpy as np


class DeviceSpec:
    """One capture device: its PortAudio index and which of its channels to take."""

    def __init__(self, index, channel=0, n_channels=None, label=None):
        self.index = int(index)
        self.channel = int(channel)
        self.n_channels = n_channels        # None => query the device
        self.label = label or f"dev{index}"

    def __repr__(self):
        return f"DeviceSpec(index={self.index}, channel={self.channel}, label={self.label!r})"


def query_inputs(verbose=True):
    """List input-capable devices with their true channel counts.

    The channel count is the thing to get right: opening a 1-channel USB mic
    with a 6-channel request is what produces PortAudio's
    'Invalid number of channels'.
    """
    import sounddevice as sd
    rows = []
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0:
            rows.append(dict(index=i, name=d['name'],
                             in_ch=d['max_input_channels'],
                             sr=d['default_samplerate']))
    if verbose:
        print(f"{'idx':>4} {'in':>3} {'default sr':>11}  name")
        for r in rows:
            print(f"{r['index']:4d} {r['in_ch']:3d} {r['sr']:11.0f}  {r['name']}")
    return rows


class MultiDeviceSource:
    """Aggregate N independent input devices into one [N, block] source.

    Each device gets its own PortAudio stream opened at ITS OWN channel count,
    writing into a per-device ring buffer. read() returns the most recent block
    across all of them.

    The streams are not synchronous and cannot be made so; see the module
    docstring. `start_skew_s` reports the spread in first-callback times, which
    is the initial offset you are asking DriftTracker to absorb.
    """

    def __init__(self, specs, fs=16000, block=2048, hop=512, capacity_s=30.0):
        import sounddevice as sd
        self.sd = sd
        self.specs = list(specs)
        self.fs = int(fs)
        self.block = int(block)
        self.hop = int(hop)
        self.n = len(self.specs)
        cap = int(capacity_s * fs)
        self.bufs = [np.zeros(cap) for _ in range(self.n)]
        self.written = [0] * self.n
        self.first_cb = [None] * self.n
        self.overflows = [0] * self.n
        self.lock = threading.Lock()
        self.streams = []
        self._open()

    def _open(self):
        for k, sp in enumerate(self.specs):
            n_ch = sp.n_channels
            if n_ch is None:
                info = self.sd.query_devices(sp.index)
                n_ch = int(info['max_input_channels'])
                sp.n_channels = n_ch
            if n_ch < 1:
                raise ValueError(f"{sp.label}: device {sp.index} has no input channels")
            if sp.channel >= n_ch:
                raise ValueError(
                    f"{sp.label}: channel {sp.channel} requested but device "
                    f"{sp.index} exposes only {n_ch} ({0}..{n_ch-1})")
            self.streams.append(self.sd.InputStream(
                device=sp.index, channels=n_ch, samplerate=self.fs,
                dtype='float32', blocksize=self.hop,
                callback=self._make_cb(k, sp.channel)))

    def _make_cb(self, k, ch):
        def cb(indata, frames, time_info, status):
            if status:
                if getattr(status, 'input_overflow', False):
                    self.overflows[k] += 1
                else:
                    print(f"[multi_device] {self.specs[k].label}: {status}",
                          file=sys.stderr)
            x = np.asarray(indata, dtype=np.float64)[:, ch]
            with self.lock:
                if self.first_cb[k] is None:
                    self.first_cb[k] = time_info.inputBufferAdcTime
                b = self.bufs[k]
                m = len(x)
                if m >= len(b):
                    b[:] = x[-len(b):]
                else:
                    b[:-m] = b[m:]
                    b[-m:] = x
                self.written[k] += m
        return cb

    def start(self):
        # Idempotent: the alignment step starts the streams before the display
        # does, and re-starting a running PortAudio stream is an error.
        if getattr(self, '_started', False):
            return
        for s in self.streams:
            s.start()
        self._started = True

    def stop(self):
        for s in self.streams:
            try:
                s.stop(); s.close()
            except Exception:
                pass

    def start_skew_s(self):
        """Spread in first-callback ADC times, seconds. The initial offset."""
        t = [v for v in self.first_cb if v is not None]
        if len(t) < 2:
            return float('nan')
        return float(max(t) - min(t))

    def calibrate_alignment(self, seconds=3.0, verbose=True, positions=None,
                            cal_az_deg=90.0):
        """Lock each stream's start offset. STAND AT A KNOWN POSITION for this.

        Three PortAudio streams begin when their own callback first fires, so
        they start tens of milliseconds apart. On a 280 mm array the whole
        geometric delay range is 816 us (39 samples at 48 kHz), so even a 5 ms
        skew is 6x that range and every frame is physically impossible until it
        is removed.

        THE TRAP: a blind cross-correlation aligner cannot tell start skew from
        acoustic delay, because both are just delay. Measured on a CLEAN
        simulated capture with zero skew and a source at 15 deg, blind
        alignment found lags of [0, 19, 38] samples -- those ARE the direction
        information -- and zeroing them moved the reported bearing from a
        correct 15.5 deg to 90.3 deg, i.e. exactly broadside, the
        "no delay" answer.

        So calibration needs a source whose true delay you KNOW. Broadside is
        trivial to arrange physically: stand in front of the MIDDLE of the array,
        facing it, and speak. There the true acoustic delay is zero, so whatever
        lag is measured is purely instrumental and can be removed safely.

        Parameters
        ----------
        positions : [M,2] array or None
            Mic positions in metres. With `cal_az_deg` this predicts the true
            acoustic delay during calibration and subtracts only the remainder.
            None assumes broadside (zero acoustic delay).
        cal_az_deg : float
            Where the calibration source actually is, in degrees. 90 = broadside.

        Returns the per-stream lag in samples.
        """
        need = int(seconds * self.fs)
        t0 = time.time()
        while min(self.written) < need:
            if time.time() - t0 > seconds + 10.0:
                raise RuntimeError(
                    "timed out waiting for audio on every stream; check that "
                    "each device is actually delivering (see query_inputs()).")
            time.sleep(0.05)
        with self.lock:
            blk = np.stack([b[-need:].copy() for b in self.bufs])
        lags, _ = align_streams(blk, fs=self.fs)
        lags = np.asarray(lags, dtype=float)

        # Subtract the acoustic delay the calibration geometry PREDICTS, so only
        # the instrumental part is removed. At broadside this term is zero.
        if positions is not None:
            pos = np.asarray(positions, dtype=np.float64)
            u = np.array([np.cos(np.deg2rad(cal_az_deg)),
                          np.sin(np.deg2rad(cal_az_deg))])
            expected = (pos @ u) / 343.0 * self.fs        # samples, per mic
            expected = expected - expected[0]
            lags = lags - expected

        self.lags = np.rint(lags - lags.min()).astype(int)
        if verbose:
            span = (self.lags.max() - self.lags.min()) / self.fs * 1000
            print(f"[multi_device] instrumental offsets: {self.lags} samples "
                  f"= {np.round(self.lags / self.fs * 1000, 1)} ms")
            print(f"[multi_device] start skew {span:.1f} ms removed "
                  f"(calibrated at {cal_az_deg:.0f}°). Clock DRIFT is NOT "
                  f"removed and still applies.")
            if span * self.fs / 1000 < 2:
                print("[multi_device] [!] offsets are near zero — if the display "
                      "still rejects frames, the cause is drift or geometry, "
                      "not skew.")
        return self.lags

    def read(self, n=None):
        """Most recent [N, n] frame, or None until every stream has filled.

        `n` defaults to the configured block size. It is accepted so this
        matches the single-device sources' read(n) signature -- the display
        calls read(BLOCK) and must not care which source it holds.

        Applies the startup alignment measured by calibrate_alignment(). Without
        that call the streams are offset by tens of milliseconds and every frame
        is geometrically impossible.
        """
        n = self.block if n is None else int(n)
        lags = getattr(self, 'lags', None)
        extra = int(lags.max()) if lags is not None else 0
        with self.lock:
            if min(self.written) < n + extra:
                return None
            if lags is None:
                return np.stack([b[-n:].copy() for b in self.bufs])
            out = np.empty((self.n, n))
            for k, b in enumerate(self.bufs):
                end = len(b) - (extra - int(lags[k]))
                out[k] = b[end - n:end]
            return out


def align_streams(block, ref=0, max_lag_s=0.25, fs=16000, f_lo=100.0, f_hi=4000.0):
    """Coarse integer-sample alignment of independently-started streams.

    Removes the tens-of-milliseconds start skew so that the residual offset is
    small enough for DriftTracker to work on. This is a BULK alignment on a
    broad band: it finds the delay that best lines the streams up, which
    presumes one dominant common source during the alignment window.

    Returns (lags_samples, aligned_block). The aligned block is shorter by the
    lag span. Run this once at startup on a few seconds of speech, then keep the
    lags fixed and let DriftTracker follow the remainder.
    """
    block = np.asarray(block, dtype=np.float64)
    n_ch, n = block.shape
    max_lag = int(max_lag_s * fs)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    keep = (freqs >= f_lo) & (freqs <= f_hi)

    def spec(x):
        X = np.fft.rfft(x - x.mean(), nfft)
        X[~keep] = 0.0
        return X

    R = spec(block[ref])
    lags = np.zeros(n_ch, dtype=int)
    for k in range(n_ch):
        if k == ref:
            continue
        cc = np.fft.irfft(spec(block[k]) * np.conj(R), nfft)
        cc = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])
        lags[k] = int(np.argmax(cc)) - max_lag
    lo, hi = lags.min(), lags.max()
    span = hi - lo
    out = np.empty((n_ch, n - span))
    for k in range(n_ch):
        s = lags[k] - lo
        out[k] = block[k, s:s + n - span]
    return lags, out
