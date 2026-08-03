#!/usr/bin/env python3
"""live_doa — live DoA display for the ReSpeaker 4-Mic array.

Three panels, matching the offline analysis figure:
  (a) array geometry with the current bearing, plus a decaying history fan
  (b) SRP-PHAT spatial spectrum, instantaneous and time-averaged
  (c) azimuth vs time, speech-active frames highlighted

Usage
-----
    python live_doa.py                       # live from the ReSpeaker
    python live_doa.py --list-devices        # find the device index
    python live_doa.py --device 7
    python live_doa.py --wav-dir <dir>       # replay ch1..ch4 wavs (no hardware)
    python live_doa.py --wav-dir <dir> --realtime   # replay at wall-clock speed

The device presents 6 channels; the beamformer/DoA input is channels 1-4
(0-based), i.e. ch0 is the device's own processed output and ch5 is playback.
See --channels to override.

Estimates are gated on voice activity: a frame below the adaptive RMS floor is
plotted grey and excluded from the running average, because a DoA estimate from
a silent frame is a direction to the room's noise field, not to the talker.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time
import wave
from collections import deque

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doa_core as dc

FS = 16000
BLOCK = 2048            # 128 ms — long enough for a stable GCC-PHAT peak
HOP = 512               # 32 ms display cadence
HISTORY_S = 12.0
TALKER_COLOR = '#c1440e'
QUIET_COLOR = '#b8b8b8'
SECOND_COLOR = '#55a868'
MIC_COLOR = '#4c72b0'


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
class RingBuffer:
    """Lock-protected multichannel ring buffer written by the audio callback."""

    def __init__(self, n_ch, capacity):
        self.buf = np.zeros((n_ch, capacity))
        self.capacity = capacity
        self.written = 0
        self.lock = threading.Lock()

    def push(self, block):
        n = block.shape[1]
        with self.lock:
            if n >= self.capacity:
                self.buf[:] = block[:, -self.capacity:]
            else:
                self.buf[:] = np.roll(self.buf, -n, axis=1)
                self.buf[:, -n:] = block
            self.written += n

    def latest(self, n):
        with self.lock:
            if self.written < n:
                return None
            return self.buf[:, -n:].copy()


class MicSource:
    """Live capture from the ReSpeaker via sounddevice."""

    def __init__(self, device=None, channels=(1, 2, 3, 4), n_dev_ch=6, fs=FS):
        import sounddevice as sd
        self.sd = sd
        info = sd.query_devices(device, 'input')
        avail = int(info['max_input_channels'])
        if fs is None or fs <= 0:
            # Most USB mics are 48 kHz native and refuse a 16 kHz open. Nothing
            # downstream assumes a rate, and higher is better for TDOA.
            fs = int(info['default_samplerate'])
            print(f"[live_doa] sample rate auto-set to {fs} Hz")
        if n_dev_ch is None or n_dev_ch <= 0:
            # Auto: open at the device's own width and take the first N.
            n_dev_ch = avail
            channels = tuple(range(min(len(channels), avail)))
        if n_dev_ch > avail:
            raise ValueError(
                f"device exposes {avail} input channel(s) but {n_dev_ch} were "
                f"requested — this is what PortAudio reports as 'Invalid number "
                f"of channels'.\n"
                f"The defaults describe the ReSpeaker 4-Mic Array (6 ch). For a "
                f"mono USB mic they cannot work.\n"
                f"DoA needs >=3 non-collinear mics on ONE clock: on macOS build "
                f"an Aggregate Device (Audio MIDI Setup -> + -> Create Aggregate "
                f"Device), tick each mic, ENABLE DRIFT CORRECTION on all but the "
                f"master, then pass --device <its index> --device-channels 0 "
                f"--positions <x y per mic, mm>.")
        if max(channels) >= avail:
            raise ValueError(
                f"channel {max(channels)} requested but the device exposes "
                f"{avail} ({0}..{avail-1}). Pass --channels within that range.")
        self.sel = list(channels)
        self.fs = fs
        self.ring = RingBuffer(len(self.sel), fs * 30)
        self.stream = sd.InputStream(
            device=device, channels=n_dev_ch, samplerate=fs,
            dtype='float32', blocksize=HOP, callback=self._cb)
        print(f"[live_doa] opened device with {n_dev_ch} channel(s), "
              f"using {self.sel}")

    def _cb(self, indata, frames, time_info, status):
        if status:
            print(f"[live_doa] stream status: {status}", file=sys.stderr)
        self.ring.push(np.asarray(indata, dtype=np.float64).T[self.sel])

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def read(self, n):
        return self.ring.latest(n)


class WavSource:
    """Replay N mono wavs as if they were arriving live.

    Lets the display be exercised, and its numbers checked against the offline
    analysis, without the hardware present.

    The mic count is DISCOVERED, not assumed: it walks ch1/mic1, ch2/mic2, …
    until a number is missing. Hardcoding four broke every two- and three-mic
    replay -- including the captures doa_protocol.py writes, which is exactly
    when replay is most useful (no hardware attached, real audio in hand).
    """

    def __init__(self, wav_dir, fs=None, realtime=False):
        # fs=None means ADOPT the files' own rate. Hardcoding 16 kHz here made
        # every 48 kHz capture unreplayable -- and 48 kHz is what the USB
        # dongles actually record, so that covered every real capture the
        # protocol script writes.
        paths = []
        i = 1
        while True:
            hits = sorted(glob.glob(os.path.join(wav_dir, f'*ch{i}*.wav'))) or \
                   sorted(glob.glob(os.path.join(wav_dir, f'*mic{i}*.wav')))
            if not hits:
                break
            paths.append(hits[0])
            i += 1
        if len(paths) < 2:
            raise FileNotFoundError(
                f"need at least 2 wavs named *ch1*/*mic1*, *ch2*/*mic2*, … in "
                f"{wav_dir}; found {len(paths)}")
        with wave.open(paths[0], 'rb') as w0:
            file_fs = w0.getframerate()
        if fs is None:
            fs = file_fs
        chans = []
        for p in paths:
            with wave.open(p, 'rb') as w:
                if w.getframerate() != fs:
                    raise ValueError(
                        f"{p}: {w.getframerate()} Hz but {os.path.basename(paths[0])} "
                        f"is {fs} Hz — all channels must share one rate")
                chans.append(np.frombuffer(w.readframes(w.getnframes()),
                                          np.int16).astype(np.float64) / 32767.0)
        n = min(len(c) for c in chans)
        self.data = np.stack([c[:n] for c in chans])
        self.pos = 0
        self.fs = fs
        self.realtime = realtime
        self._t0 = None
        print(f"[live_doa] replaying {n/fs:.2f}s from {os.path.basename(wav_dir)}")
        for p in paths:
            print(f"           {os.path.basename(p)}")

    def start(self):
        self._t0 = time.time()

    def stop(self):
        pass

    def read(self, n):
        if self.realtime:
            want = int((time.time() - self._t0) * self.fs)
            if want < self.pos + HOP:
                return None
        self.pos += HOP
        if self.pos < n:
            return None
        if self.pos > self.data.shape[1]:
            self.pos = n                       # loop
        return self.data[:, self.pos - n:self.pos]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
class LiveDoADisplay:
    def __init__(self, source, radius=dc.R_NOMINAL, vad_percentile=70.0,
                 avg_tc=2.0, pos=None, f_hi=dc.F_HI, aperture_m=None,
                 linear_gaps=None, side=None):
        self.src = source
        # pos overrides the built-in ReSpeaker geometry: with separate mics on
        # stands only the operator knows where they are.
        self.pos = dc.array_positions(radius) if pos is None else np.asarray(pos, float)
        self.n_mics = self.pos.shape[0]
        self.pairs = dc.pairs_for(self.n_mics)
        # Linear mode: the estimator solves for one angle to the array axis and
        # reports both mirror branches, instead of a full-plane direction.
        self.linear_gaps = linear_gaps
        self.side = side
        self.f_hi = float(f_hi)
        self.aperture = aperture_m if aperture_m is not None else max(
            np.linalg.norm(self.pos[i] - self.pos[j]) for i, j in self.pairs)
        # A linear array's spatial spectrum is symmetric about its axis, so a
        # full 360 deg scan shows every peak twice. Scan the half-plane and let
        # the mirror arrow carry the ambiguity instead of a duplicate lobe the
        # reader might mistake for a second source.
        self.grid = (np.arange(0.0, 180.5, 1.0) if linear_gaps is not None
                     else np.arange(0.0, 360.0, 1.0))
        self.vad_pct = vad_percentile
        # Take the rate from the SOURCE, not the module constant: a USB mic
        # opened at its native 48 kHz would otherwise get a 16 kHz time axis
        # and a mis-scaled analysis band.
        self.fs = int(getattr(source, 'fs', FS))
        self.alpha = float(np.exp(-HOP / (avg_tc * self.fs)))
        self.srp_avg = None
        self.rms_hist = deque(maxlen=400)
        self.hist = deque(maxlen=int(HISTORY_S * self.fs / HOP))
        self.t = 0.0
        # Only meaningful for independent-clock captures; harmless otherwise.
        self.drift = dc.DriftMonitor(len(self.pairs), hop=HOP, fs=self.fs)
        self.multi = hasattr(source, 'start_skew_s')
        self._build()

    # -- figure ---------------------------------------------------------
    def _build(self):
        self.fig = plt.figure(figsize=(12.4, 4.1))
        self.fig.canvas.manager.set_window_title('Dorai — live DoA')
        gs = self.fig.add_gridspec(1, 3, width_ratios=[0.95, 1.25, 1.05],
                                   wspace=0.34, left=0.045, right=0.988,
                                   top=0.84, bottom=0.155)
        self.axA = self.fig.add_subplot(gs[0, 0])
        self.axB = self.fig.add_subplot(gs[0, 1])
        self.axC = self.fig.add_subplot(gs[0, 2])
        for ax, letter in ((self.axA, 'a'), (self.axB, 'b'), (self.axC, 'c')):
            ax.text(-0.02, 1.13, letter, transform=ax.transAxes,
                    fontsize=11, fontweight='bold', va='top', ha='right')
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.tick_params(direction='out', labelsize=7)
            ax.xaxis.label.set_size(8)
            ax.yaxis.label.set_size(8)
            ax.title.set_size(8.5)

        # (a) geometry
        pm = self.pos * 1000
        span = max(np.abs(pm).max(), 1.0)
        r_mm = float(np.linalg.norm(pm, axis=1).mean())
        self.axA.add_patch(plt.Circle((0, 0), r_mm, fill=False, ec='0.78',
                                      ls='--', lw=0.9))
        for k, (px, py) in enumerate(pm):
            self.axA.plot(px, py, 'o', ms=11, color=MIC_COLOR, zorder=3)
            self.axA.annotate(f'ch{k+1}', (px, py), ha='center', va='center',
                              fontsize=6.5, color='white', zorder=4,
                              fontweight='bold')
        self.fan = [self.axA.plot([0, 0], [0, 0], lw=1.0, color=TALKER_COLOR,
                                  alpha=0.0, zorder=2)[0] for _ in range(14)]
        # Arrow/label lengths scale with the array so this works for a 64 mm
        # ReSpeaker and a 300 mm stand-mounted triangle alike.
        self.arrow_len = 1.6 * span
        self.fan_len = 1.4 * span
        self.arrow = self.axA.annotate(
            '', xy=(0, self.arrow_len), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', lw=2.4, color=TALKER_COLOR))
        self.az_label = self.axA.text(0, 2.1 * span, '', color=TALKER_COLOR,
                                      fontsize=9, ha='center', fontweight='bold')
        # Mirror branch, drawn only in linear mode.
        self.mirror_arrow, = self.axA.plot([0, 0], [0, 0], lw=2.0, ls='--',
                                           color=TALKER_COLOR, alpha=0.55,
                                           zorder=3)
        self.mirror_label = self.axA.text(0, 0, '', color=TALKER_COLOR,
                                          fontsize=8, alpha=0.8, ha='center')
        self.axA.set_xlim(-2.3 * span, 2.3 * span)
        self.axA.set_ylim(-1.9 * span, 2.6 * span)
        self.axA.set_aspect('equal')
        self.axA.set_xlabel('x (mm)')
        self.axA.set_ylabel('y (mm)', labelpad=8)
        self.axA.set_title('Array and current bearing', loc='left')

        # (b) SRP-PHAT
        self.l_inst, = self.axB.plot(self.grid, np.zeros_like(self.grid),
                                     lw=1.0, color='0.62', label='instantaneous')
        self.l_avg, = self.axB.plot(self.grid, np.zeros_like(self.grid),
                                    lw=1.9, color=MIC_COLOR,
                                    label='time-averaged')
        self.v_pk = self.axB.axvline(np.nan, color=TALKER_COLOR, lw=1.6)
        self.v_sec = self.axB.axvline(np.nan, color=SECOND_COLOR, lw=1.2, ls='--')
        self.sec_label = self.axB.text(0, 0, '', color=SECOND_COLOR, fontsize=7)
        self.axB.set_xlim(float(self.grid[0]), float(self.grid[-1]))
        self.axB.set_ylim(-0.04, 1.14)
        self.axB.set_xticks([0, 45, 90, 135, 180] if self.linear_gaps is not None
                            else [0, 90, 180, 270, 359])
        self.axB.set_xlabel('azimuth (deg)')
        self.axB.set_ylabel('normalized SRP-PHAT power')
        self.axB.set_title(f'Spatial spectrum, {dc.F_LO:.0f}–{self.f_hi:.0f} Hz',
                           loc='left')
        self.axB.legend(frameon=False, fontsize=7, loc='upper right',
                        handlelength=1.4)

        # (c) azimuth track
        self.sc_quiet = self.axC.scatter([], [], s=7, color=QUIET_COLOR,
                                         label='low-energy')
        self.sc_talk = self.axC.scatter([], [], s=15, color=TALKER_COLOR,
                                        label='speech-active')
        self.h_mean = self.axC.axhline(np.nan, color=TALKER_COLOR, lw=1.2, ls='--')
        self.axC.set_ylim(0, 360)          # replaced by auto-range once data arrives
        self.axC.set_xlabel('time (s)')
        self.axC.set_ylabel('instantaneous azimuth (deg)')
        self.axC.set_title('Azimuth history', loc='left')
        self.axC.legend(frameon=False, fontsize=7, loc='lower right')

        self.banner = self.fig.text(
            0.045, 0.965, 'waiting for audio…', fontsize=9.5, ha='left')
        self.drift_label = self.fig.text(
            0.045, 0.925, '', fontsize=8, ha='left', color='0.45')
        # Report THIS array's numbers, not the ReSpeaker constants.
        self.fig.text(0.988, 0.955,
                      f'band-limited to {self.f_hi:.0f} Hz '
                      f'(aliasing limit {dc.C_SOUND / (2 * self.aperture):.0f} Hz, '
                      f'aperture {self.aperture * 1000:.0f} mm)',
                      fontsize=6.8, color='0.45', ha='right')

    # -- per-frame update ----------------------------------------------
    def update(self, _frame):
        block = self.src.read(BLOCK)
        if block is None:
            return self._artists()
        self.t += HOP / self.fs

        rms = dc.frame_rms(block)               # windowed — matches the threshold
        self.rms_hist.append(rms)
        thr = (np.percentile(self.rms_hist, self.vad_pct)
               if len(self.rms_hist) >= 20 else np.inf)
        active = rms > thr

        if self.linear_gaps is not None:
            est = dc.doa_linear(block, self.linear_gaps, fs=self.fs,
                                f_hi=self.f_hi, side=self.side)
            est.setdefault('u_norm', 1.0)
            est['resid_mm'] = est['resid_us'] * 1e-6 * dc.C_SOUND * 1000.0
        else:
            est = dc.doa_least_squares(block, pos=self.pos, f_hi=self.f_hi,
                                       fs=self.fs)
        az = est['az_deg'] % 360.0
        # A frame whose pairwise delays cannot come from one plane wave carries
        # no direction. Averaging it in does not cancel -- it drags the pooled
        # bearing toward broadside and makes noise look like a measurement.
        ok = est.get('consistent', True)
        self.n_seen = getattr(self, 'n_seen', 0) + (1 if active else 0)
        self.n_bad = getattr(self, 'n_bad', 0) + (1 if active and not ok else 0)
        self.hist.append((self.t, az, active and ok))
        self.drift.update(est['taus'], active=active)

        _, raw = dc.srp_phat(block, pos=self.pos, grid_deg=self.grid,
                             f_hi=self.f_hi, fs=self.fs)
        if active:
            # accumulate RAW power; normalize only for display. Averaging
            # already-normalized frames biases the pooled peak.
            self.srp_avg = raw if self.srp_avg is None else \
                self.alpha * self.srp_avg + (1 - self.alpha) * raw

        self.l_inst.set_ydata(dc.normalize_power(raw))
        if self.srp_avg is not None:
            avg_n = dc.normalize_power(self.srp_avg)
            self.l_avg.set_ydata(avg_n)
            pk = float(self.grid[int(np.argmax(avg_n))])
            sec, sec_p = dc.secondary_peak(self.grid, avg_n, pk)
            self.v_pk.set_xdata([pk, pk])
            # A wide array with a thin usable band has a BROAD main lobe, and
            # its shoulder trips a naive "second peak" test. Require the
            # candidate to be a genuinely separate lobe: far enough away AND
            # with a real dip between it and the main peak.
            if np.isfinite(sec):
                lo, hi = sorted((pk, sec))
                between = avg_n[(self.grid > lo) & (self.grid < hi)]
                dip = float(between.min()) if between.size else 1.0
                separate = abs(sec - pk) >= 25.0 and dip < 0.6 * sec_p
            else:
                separate = False
            if separate and sec_p > 0.55:
                self.v_sec.set_xdata([sec, sec])
                self.sec_label.set_position((sec + 5, sec_p))
                self.sec_label.set_text(f'2nd source {sec:.0f}°')
            else:
                self.v_sec.set_xdata([np.nan, np.nan])
                self.sec_label.set_text('')

        talk = [(t, a) for t, a, ac in self.hist if ac]
        quiet = [(t, a) for t, a, ac in self.hist if not ac]
        self.sc_talk.set_offsets(np.array(talk) if talk else np.empty((0, 2)))
        self.sc_quiet.set_offsets(np.array(quiet) if quiet else np.empty((0, 2)))
        if self.hist:
            self.axC.set_xlim(max(0.0, self.t - HISTORY_S), self.t + 0.4)
            # Auto-range y to the data: a fixed 0-360 axis leaves a stable
            # bearing occupying a few percent of the panel height, which hides
            # exactly the frame-to-frame spread the panel exists to show.
            ang = np.array([a for _, a, _ in self.hist])
            lo, hi = np.percentile(ang, 1), np.percentile(ang, 99)
            pad = max(12.0, 0.25 * (hi - lo))
            self.axC.set_ylim(max(0.0, lo - pad), min(360.0, hi + pad))

        if talk:
            mean_az, R, sd = dc.circular_mean([a for _, a in talk])
            mean_az %= 360.0
            self.h_mean.set_ydata([mean_az, mean_az])
            rad = np.deg2rad(mean_az)
            self.arrow.xy = (self.arrow_len * np.cos(rad),
                             self.arrow_len * np.sin(rad))
            self.az_label.set_text(f'{mean_az:.0f}° ± {sd:.0f}°')
            if self.linear_gaps is not None:
                # Draw the mirror as an equal partner, not a footnote: the data
                # supports both equally and showing one arrow implies certainty
                # the recording does not contain.
                mrad = np.deg2rad(-mean_az)
                self.mirror_arrow.set_data(
                    [0, self.arrow_len * np.cos(mrad)],
                    [0, self.arrow_len * np.sin(mrad)])
                self.mirror_label.set_position(
                    (1.15 * self.arrow_len * np.cos(mrad),
                     1.15 * self.arrow_len * np.sin(mrad)))
                self.mirror_label.set_text(f'or {(-mean_az) % 360:.0f}°')
            # decaying fan of recent bearings
            recent = [a for _, a in talk][-len(self.fan):]
            for ln, a in zip(self.fan, recent):
                rr = np.deg2rad(a)
                ln.set_data([0, self.fan_len * np.cos(rr)],
                            [0, self.fan_len * np.sin(rr)])
                ln.set_alpha(0.16)
            for ln in self.fan[len(recent):]:
                ln.set_alpha(0.0)
            if self.linear_gaps is not None:
                sat = '   [!] delay exceeds geometry: check spacing' \
                      if est.get('saturated') else ''
                seen, bad = getattr(self, 'n_seen', 0), getattr(self, 'n_bad', 0)
                # When most frames fail the physical-consistency test, the few
                # that pass are survivors of a broken process, not evidence.
                # Refuse rather than report a bearing built from the residue.
                if seen >= 20 and bad > 0.5 * seen:
                    self.arrow.xy = (0, 0)
                    self.mirror_arrow.set_data([0, 0], [0, 0])
                    self.az_label.set_text('')
                    self.mirror_label.set_text('')
                    self.banner.set_color(TALKER_COLOR)
                    self.banner.set_text(
                        f'NO TRUSTWORTHY BEARING — {bad}/{seen} frames '
                        f'({100*bad/seen:.0f}%) rejected: delays no single '
                        f'source could produce')
                    if self.multi:
                        self._update_drift_readout()
                    else:
                        self.drift_label.set_text(
                            'check the --linear spacing matches the real mic '
                            'gaps, or the mics share one clock')
                        self.drift_label.set_color(TALKER_COLOR)
                    return self._artists()
                rej = (f'   {bad}/{seen} frames rejected as inconsistent'
                       if bad else '')
                self.banner.set_text(
                    f'{mean_az:5.1f}° from the array axis  —  or {(-mean_az) % 360:.0f}°, '
                    f'mirrored (a line cannot tell the two apart)   '
                    f'sd {sd:4.1f}°   R {R:.3f}   '
                    f'residual {est["resid_us"]:.0f}/{est["max_delay_us"]:.0f} µs'
                    f'{rej}{sat}')
            else:
                flag = '' if est['u_norm'] <= 1.02 else '   [!] |u|>1: check geometry'
                self.banner.set_text(
                    f'talker {mean_az:5.1f}°   sd {sd:4.1f}°   R {R:.3f}   '
                    f'residual {est["resid_mm"]:.1f} mm   '
                    f'{len(talk)} active / {len(self.hist)} frames{flag}')
            if self.multi:
                self._update_drift_readout()
        else:
            seen, bad = getattr(self, 'n_seen', 0), getattr(self, 'n_bad', 0)
            if seen and bad > 0.5 * seen:
                self.arrow.xy = (0, 0)
                self.mirror_arrow.set_data([0, 0], [0, 0])
                self.az_label.set_text('')
                self.mirror_label.set_text('')
                self.banner.set_text(
                    f'NO VALID BEARING — {bad}/{seen} speech frames rejected: '
                    f'correlation peak too weak or delays beyond the geometry. '
                    f'Clocks drifting, spacing wrong, or the room too reverberant.')
                self.banner.set_color(TALKER_COLOR)
            else:
                self.banner.set_text(
                    f'no speech-active frames yet   ({len(self.hist)} seen)')
        return self._artists()

    def _update_drift_readout(self):
        """Independent-clock health line: drift rate and how long a bearing lasts.

        Shown only for multi-device captures. The rate is a LOWER bound (the
        bounded GCC-PHAT search saturates once drift approaches the aperture),
        so `saturating` is the more important signal of the two.
        """
        ppm = self.drift.drift_rate_ppm()
        if not np.any(np.isfinite(ppm)):
            self.drift_label.set_text('independent clocks — measuring drift…')
            self.drift_label.set_color('0.45')
            return
        worst = float(np.nanmax(np.abs(ppm)))
        win = self.drift.usable_window_s(self.aperture)
        if self.drift.saturating(self.aperture):
            self.drift_label.set_text(
                f'DRIFT SATURATED (≥{worst:.0f} ppm measured, true rate higher) — '
                f'bearing is not trustworthy; restart or use one shared clock')
            self.drift_label.set_color(TALKER_COLOR)
        else:
            self.drift_label.set_text(
                f'clock drift ≥{worst:.0f} ppm — bearing good for ~{win:.1f} s '
                f'before 5° of self-rotation')
            self.drift_label.set_color('0.35' if win > 2.0 else TALKER_COLOR)

    def _artists(self):
        return (self.l_inst, self.l_avg, self.sc_talk, self.sc_quiet,
                self.arrow, self.az_label, self.banner, self.drift_label,
                self.mirror_arrow, self.mirror_label,
                self.h_mean, self.v_pk, self.v_sec, self.sec_label, *self.fan)

    def run(self):
        from matplotlib.animation import FuncAnimation
        self.src.start()
        # cache_frame_data=False: this is an unbounded live stream, not a
        # finite sequence matplotlib should try to cache.
        self.anim = FuncAnimation(self.fig, self.update, interval=40,
                                  blit=False, cache_frame_data=False)
        try:
            plt.show()
        finally:
            self.src.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', type=int, default=None,
                    help='sounddevice input index (see --list-devices)')
    ap.add_argument('--list-devices', action='store_true')
    ap.add_argument('--channels', type=int, nargs='+', default=(1, 2, 3, 4),
                    help='0-based device channels for mic1..micN, in array '
                         'order (default 1 2 3 4 for the ReSpeaker, whose ch0 '
                         'is the processed mix, not a raw mic)')
    ap.add_argument('--circle', type=float, nargs='+', default=None,
                    metavar='RADIUS_MM',
                    help='mics on a CIRCLE: radius in mm, optionally followed '
                         'by the angle of mic1 in degrees (default 90 = +y). '
                         'Count comes from --channels/--devices. Add '
                         '--centre-mic if one mic sits at the centre.')
    ap.add_argument('--centre-mic', action='store_true', dest='centre_mic',
                    help='with --circle: the LAST channel is at the centre, '
                         'the rest on the rim (e.g. 4 rim + 1 centre = 5 mics)')
    ap.add_argument('--clockwise', action='store_true',
                    help='with --circle: channels run clockwise round the rim')
    ap.add_argument('--device-channels', type=int, default=6,
                    dest='device_channels',
                    help='total channels to open on the device')
    ap.add_argument('--wav-dir', default=None,
                    help='replay ch1..ch4 wavs from this directory')
    ap.add_argument('--realtime', action='store_true',
                    help='with --wav-dir, replay at wall-clock speed')
    ap.add_argument('--radius', type=float, default=dc.R_NOMINAL * 1000,
                    help='array radius in mm (azimuth is invariant to this)')
    ap.add_argument('--vad-percentile', type=float, default=70.0)
    ap.add_argument('--fs', type=int, default=FS,
                    help='sample rate; 0 = auto-detect from the device. Use 0 '
                         'for USB mics, which are usually 48 kHz native and '
                         'refuse a 16 kHz open.')
    ap.add_argument('--devices', type=int, nargs='+', default=None,
                    help='SEVERAL independent input device indices, one mono mic '
                         'each (e.g. --devices 1 3 4). Requires --positions. '
                         'Read the drift warning this prints before trusting it.')
    ap.add_argument('--linear', type=float, nargs='+', default=None,
                    help='mics in a straight LINE: gaps between consecutive '
                         'mics in mm, in channel order (e.g. --linear 140 140 '
                         'for 3 mics, or --linear 140 for equal gaps). Reports '
                         'the angle to the array axis plus its mirror — a line '
                         'cannot tell which side you are on.')
    ap.add_argument('--side', choices=['+', '-'], default=None,
                    help="which side of the line you are on, if you know it. "
                         "This is YOUR prior, not a measurement; it only picks "
                         "which mirror branch is displayed.")
    ap.add_argument('--positions', type=float, nargs='+', default=None,
                    help='mic xy positions in mm, flat: x1 y1 x2 y2 ... '
                         'Must be non-collinear and match --devices in count.')
    args = ap.parse_args(argv)

    if args.list_devices:
        from multi_device import query_inputs
        query_inputs()
        return 0

    if args.devices:
        return _run_multi(args)

    # Explicit geometry (mm) overrides the built-in ReSpeaker circle. Needed
    # for an Aggregate Device whose mics sit wherever you put them.
    pos = f_hi = aperture = None
    linear_gaps = None
    if args.linear:
        gaps = np.asarray(args.linear, float) / 1000.0
        pos = dc.linear_positions(gaps if gaps.size > 1 else gaps[0])
        linear_gaps = gaps if gaps.size > 1 else float(gaps[0])
        aperture = float(pos[:, 0].max() - pos[:, 0].min())
        f_hi = min(dc.F_HI, dc.C_SOUND / (2.0 * aperture))
        print(f"[live_doa] LINEAR array: {pos.shape[0]} mics at "
              f"{np.round(pos[:,0]*1000,1)} mm, aperture {aperture*1000:.0f} mm")
        print(f"[live_doa] band {dc.F_LO:.0f}-{f_hi:.0f} Hz "
              f"(spatial aliasing limit {dc.C_SOUND/(2*aperture):.0f} Hz)")
        print("[live_doa] A line measures the angle to its OWN AXIS: 0° = past the")
        print("[live_doa] first mic, 90° = broadside, 180° = past the last. It")
        print("[live_doa] CANNOT tell which side of the line you are on — +40° and")
        print("[live_doa] −40° give identical delays. Both branches are shown.")
        if f_hi < 800.0:
            best = dc.C_SOUND / (2.0 * 1200.0)      # aperture for a 1200 Hz ceiling
            print(f"[live_doa]")
            print(f"[live_doa] [!] The band stops at {f_hi:.0f} Hz because the array is")
            print(f"[live_doa]     {aperture*1000:.0f} mm wide — above that it aliases "
                  f"spatially and")
            print(f"[live_doa]     bearings become ambiguous. Speech DoA wants ~300-1500 Hz,")
            print(f"[live_doa]     so this leaves a thin band and a noisy estimate.")
            print(f"[live_doa]     Closing the mics to ~{best*1000/2:.0f} mm gaps "
                  f"(≈{best*1000:.0f} mm total) would")
            print(f"[live_doa]     raise the ceiling to ~1200 Hz. Wider = finer resolution")
            print(f"[live_doa]     but a lower ceiling; this is a real trade-off, not a bug.")
    elif args.circle:
        radius_mm = float(args.circle[0])
        start_deg = float(args.circle[1]) if len(args.circle) > 1 else 90.0
        n_total = len(args.devices) if args.devices else len(args.channels)
        n_rim = n_total - 1 if args.centre_mic else n_total
        if n_rim < 3:
            print(f"[live_doa] a circle needs 3+ RIM mics; --channels gives "
                  f"{n_total}" + (" with one at the centre" if args.centre_mic
                                  else "") + ".", file=sys.stderr)
            return 1
        pos = dc.circle_positions(n_rim, radius_mm / 1000.0,
                                  start_deg=start_deg,
                                  clockwise=args.clockwise,
                                  centre_mic=args.centre_mic)
        aperture = max(np.linalg.norm(pos[i] - pos[j])
                       for i, j in dc.pairs_for(pos.shape[0]))
        f_hi = min(dc.F_HI, dc.aliasing_limit_hz(pos))
        layout = (f"{n_rim} on the rim + 1 at the centre" if args.centre_mic
                  else f"{n_rim} on the rim")
        print(f"[live_doa] CIRCULAR array: {layout}, radius {radius_mm:.1f} mm, "
              f"{360.0/n_rim:.0f}° apart")
        print(f"[live_doa] mic1 at {start_deg:.0f}°, numbering "
              f"{'clockwise' if args.clockwise else 'counter-clockwise'}")
        print(f"[live_doa] aperture {aperture*1000:.0f} mm, band "
              f"{dc.F_LO:.0f}-{f_hi:.0f} Hz (aliasing limit "
              f"{dc.aliasing_limit_hz(pos):.0f} Hz)")
        print("[live_doa] A circle resolves the FULL 360° — no mirror ambiguity.")
        print("[live_doa] But --circle only says the mics form a circle; it "
              "cannot know")
        print("[live_doa] which physical mic is channel 1 or which way the "
              "numbering runs.")
        print("[live_doa] If bearings are consistently rotated or mirrored, fix "
              "that with")
        print("[live_doa] the mic1 angle argument and --clockwise, not by "
              "adjusting the result.")
    elif args.positions:
        if len(args.positions) % 2:
            print("[live_doa] --positions needs an even count: x y per mic, mm.",
                  file=sys.stderr)
            return 1
        pos = np.asarray(args.positions, float).reshape(-1, 2) / 1000.0
        if pos.shape[0] < 3:
            print("[live_doa] --positions needs 3+ non-collinear mics: a pair "
                  "resolves only the angle to its own axis.", file=sys.stderr)
            return 1
        D = dc.pair_baselines(pos, dc.pairs_for(pos.shape[0]))
        if np.linalg.matrix_rank(D, tol=1e-9) < 2:
            print("[live_doa] those positions are collinear; offset one mic "
                  "perpendicular to the line.", file=sys.stderr)
            return 1
        aperture = max(np.linalg.norm(pos[i] - pos[j])
                       for i, j in dc.pairs_for(pos.shape[0]))
        f_hi = min(dc.F_HI, dc.aliasing_limit_hz(pos))
        print(f"[live_doa] geometry: {pos.shape[0]} mics, aperture "
              f"{aperture*1000:.0f} mm, band {dc.F_LO:.0f}-{f_hi:.0f} Hz")

    if args.wav_dir:
        # fs=None (the default, and what --fs 0 means) adopts the files' rate.
        src = WavSource(args.wav_dir, fs=(args.fs or None),
                        realtime=args.realtime)
    else:
        n_ch = len(pos) if pos is not None else len(args.channels)
        try:
            src = MicSource(device=args.device,
                            channels=tuple(args.channels[:n_ch]),
                            n_dev_ch=args.device_channels, fs=args.fs)
        except Exception as exc:
            print(f"[live_doa] could not open the input device:\n{exc}\n"
                  f"\nRun --list-devices to see channel counts, or --wav-dir "
                  f"to replay files.", file=sys.stderr)
            return 1

    kw = {} if pos is None else dict(pos=pos, f_hi=f_hi, aperture_m=aperture)
    if linear_gaps is not None:
        kw.update(linear_gaps=linear_gaps, side=args.side)
    LiveDoADisplay(src, radius=args.radius / 1000.0,
                   vad_percentile=args.vad_percentile, **kw).run()
    return 0


def _run_multi(args):
    """Live display across several independent USB microphones."""
    from multi_device import DeviceSpec, MultiDeviceSource

    n = len(args.devices)
    if n < 3:
        print(f"[live_doa] {n} device(s) given. A 2-element array cannot resolve a\n"
              f"           full 360 deg bearing -- it measures only the angle to its\n"
              f"           own axis, so every estimate has a mirror twin. Give 3+\n"
              f"           non-collinear mics, or use doa_core.doa_pair() directly\n"
              f"           and accept the two-branch answer.", file=sys.stderr)
        return 1
    linear_gaps = None
    if args.linear:
        gaps = np.asarray(args.linear, float) / 1000.0
        if gaps.size == 1:
            gaps = np.repeat(gaps, n - 1)
        if gaps.size != n - 1:
            print(f"[live_doa] --linear needs {n-1} gap(s) for {n} devices "
                  f"(or one value for equal gaps), got {gaps.size}.",
                  file=sys.stderr)
            return 1
        pos = dc.linear_positions(gaps)
        linear_gaps = gaps
    elif args.positions and len(args.positions) == 2 * n:
        pos = np.asarray(args.positions, dtype=np.float64).reshape(n, 2) / 1000.0
    else:
        print(f"[live_doa] give the geometry for your {n} devices:\n"
              f"           --linear <{n-1} gaps in mm>      mics in a straight line\n"
              f"           --positions <{2*n} numbers>      x y per mic, mm\n"
              f"           There is no default: with separate mics on stands, only\n"
              f"           you know where they are, and a wrong geometry produces a\n"
              f"           confident wrong bearing. Measure them.", file=sys.stderr)
        return 1
    if linear_gaps is None:
        D = dc.pair_baselines(pos, dc.pairs_for(n))
        if np.linalg.matrix_rank(D, tol=1e-9) < 2:
            print("[live_doa] those positions are collinear. Use --linear "
                  f"<{n-1} gaps in mm>\n"
                  "           instead, which solves the axis angle properly and "
                  "reports both\n"
                  "           mirror branches. Or offset one mic perpendicular "
                  "to the line\n"
                  "           to get an unambiguous 360° bearing.", file=sys.stderr)
            return 1

    aperture = max(np.linalg.norm(pos[i] - pos[j]) for i, j in dc.pairs_for(n))
    f_hi = min(dc.F_HI, dc.aliasing_limit_hz(pos))
    print(f"[live_doa] {n} independent devices, aperture {aperture*1000:.0f} mm, "
          f"band {dc.F_LO:.0f}-{f_hi:.0f} Hz")
    print("[live_doa] ---- INDEPENDENT CLOCKS: READ THIS ----")
    print("[live_doa] Each USB mic has its own crystal. Nothing locks them, so the")
    print("[live_doa] inter-stream delay walks at roughly 1 us/s per ppm of error;")
    print("[live_doa] cheap USB audio runs 50-100 ppm. Measured in simulation on a")
    print("[live_doa] 120 mm triangle with a STATIONARY source, independent clocks")
    print("[live_doa] swung the bearing through 359 deg in 12 s, where one shared")
    print("[live_doa] clock held it to 1.19 deg. Drift cannot be removed from TDOA")
    print("[live_doa] alone (see DriftMonitor's docstring for what was tried).")
    print("[live_doa] Trust SHORT-WINDOW bearings and watch the drift readout; for")
    print("[live_doa] dependable absolute DoA use one multichannel interface.")

    specs = [DeviceSpec(idx, channel=0, label=f"mic{k+1}")
             for k, idx in enumerate(args.devices)]
    try:
        src = MultiDeviceSource(specs, fs=FS, block=BLOCK, hop=HOP)
    except Exception as exc:
        print(f"[live_doa] could not open the devices: {exc}", file=sys.stderr)
        return 1

    # Independent streams start tens of milliseconds apart -- on a 280 mm array
    # even 5 ms is 6x the entire 816 us geometric range, so EVERY frame is
    # physically impossible until this offset is removed.
    src.start()
    print("[live_doa] ---- CALIBRATION ----")
    print("[live_doa] Stand in FRONT OF THE MIDDLE of the array (broadside),")
    print("[live_doa] facing it, and speak steadily for 4 seconds.")
    print("[live_doa] Position matters: calibration cannot tell start skew from")
    print("[live_doa] acoustic delay, so it assumes you are at broadside, where")
    print("[live_doa] the true delay is zero. Calibrating from the wrong spot")
    print("[live_doa] bakes that angle in as an error on every later bearing.")
    input("[live_doa] press Enter when you are in position… ")
    try:
        src.calibrate_alignment(seconds=4.0, positions=pos, cal_az_deg=90.0)
    except Exception as exc:
        print(f"[live_doa] alignment failed: {exc}", file=sys.stderr)
        src.stop()
        return 1

    disp = LiveDoADisplay(src, vad_percentile=args.vad_percentile,
                          pos=pos, f_hi=f_hi, aperture_m=aperture,
                          linear_gaps=linear_gaps, side=args.side)
    disp.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
