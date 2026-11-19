#!/usr/bin/env python3
"""
voice_mod / voice.py  —  dorai real-time mic-array front-end.

A single ROS 2 node that runs the full capture-to-enhancement path:

    mic array (N mics, USB/other)
        -> per-mic clock-drift-corrected async resample to 16 kHz
        -> dorai_beamformer.ort multichannel beamformer
        -> one slow AGC for a steady STT level
        -> clean single-channel speech
        -> published on /dorai_clean_audio in N-second frames (default 10 s)

The model file decides how the beamformer is driven:

  * A streaming model (the shipped dorai_beamformer.ort: one 512-sample block
    per microphone plus a state tensor in, the enhanced block and the next
    state out) is fed block by block by StreamingEnhancer. Everything the model
    knows about the scene travels in the state, so the output is continuous by
    construction and lags the input by the model's 24 ms. A streaming model is
    built for one microphone count; the node refuses a mismatch.
  * A whole-signal model (the previous dorai_beamformer.ort, kept working for
    `-p model_path:=...`) goes through ContinuousEnhancer: an `enh_window`-second
    window slides by `enh_hop` seconds over the continuous stream, each window
    is enhanced and Hann-overlap-added. Added latency: one window (2 s).

Publishing is batched into `publish_interval`-second frames only so that
stt_mod receives larger, cheaper messages; it introduces no boundary artifact.

A beamformer needs the delays between the channels to stand still. Two parts
of this file exist for that: at start-up the FIFOs are aligned by the capture
time of their newest samples (prime_offsets), and the clock-drift loop keeps
the capture time at the head of every FIFO where that alignment left it
(lock_channels) instead of comparing FIFO lengths, which jump by a whole block
whenever a device's callback lands. The loop corrects clock DRIFT only. It does
not re-align: start-up alignment discards stamp differences below 10 ms (stream
start offsets, stamp jitter), and a loop that steered those to zero moved the
channels by up to 30 samples in the first 30 s of every run and kept them
wandering by 2-3 samples afterwards. Measured in the ROS 2 lab test, three
passes, 501 words: 11.0 % word errors with that loop against 4.6 % without it.
A constant offset of 1-2 ms cost nothing measurable (5.8 %); movement did.

Design properties:
  * Audio callbacks only append to bounded, per-channel locked ring buffers,
    keeping the PortAudio thread responsive.
  * A dedicated worker thread resamples and assembles aligned blocks; inference
    and publishing run on a second thread, so neither the ROS executor nor the
    capture path is ever blocked by ONNX.
  * Per-mic asynchronous sample-rate conversion with a closed-loop ratio trim
    keeps the independent USB ADC clocks locked to one 16 kHz master timeline.
  * All buffers are bounded (drop-oldest); a stalled mic degrades gracefully to
    zeros instead of blocking the array.
"""

import os
import math
import time
import queue
import threading

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

import onnxruntime as ort

# Quiet ORT's C++ logger. On a headless Pi (no GPU) the provider bring-up
# prints harmless "GetGpuDevices ... Failed to open .../device/vendor" warnings
# at startup; raise the severity floor to errors so they don't clutter the log.
# 0=VERBOSE 1=INFO 2=WARNING 3=ERROR 4=FATAL.
try:
    ort.set_default_logger_severity(3)
except Exception:
    pass

import signal

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

import sounddevice as sd


# ---------------------------------------------------------------------------
# Constants — the topic contract (must match stt_mod) and DSP framing.
# ---------------------------------------------------------------------------
OUTPUT_RATE = 16000               # master timeline rate (Hz)
BLOCK_MS = 100                    # worker tick period (ms)
BLOCK_S = BLOCK_MS / 1000.0
L_OUT = OUTPUT_RATE * BLOCK_MS // 1000          # output samples/block/ch (1600)

# For a whole-signal model driven by the overlap-add sliding window (see
# ContinuousEnhancer), WINDOW_SIZE is the dummy length used to warm the ONNX
# session up before real audio arrives.
WINDOW_SIZE = 3200

# Asynchronous-resampler drift control.
TARGET_BLOCKS = 2                 # steady-state out-FIFO depth (blocks)
TARGET_OUT = TARGET_BLOCKS * L_OUT
RATIO_TRIM_MAX = 0.02            # clamp |ratio correction| to +/-2 %
# The drift loop works on capture time, not on FIFO length. Each device hands
# over its audio in 100 ms callbacks that land at arbitrary moments relative to
# the worker tick, so at any tick the FIFO lengths differ by up to a whole
# block for reasons that have nothing to do with clock drift; a loop that
# reads lengths (as this one used to) answers that with ratio changes of up to
# 2 % and moves the channels against each other by tens of samples per tick.
# That is harmless for a whole-signal enhancer and ruinous for a beamformer,
# which needs the delays between channels to stand still. What must be equal
# across channels is the capture time of the sample at the head of each FIFO,
# and that does not jump when a callback lands.
DRIFT_SMOOTH_S = 5.0             # smoothing of the head-time error (capture stamps jitter by ~1 ms)
DRIFT_TAU_S = 15.0               # proportional part: an error decays with this time constant
DRIFT_INT_TAU_S = 45.0           # integral part: removes the standing error of a constant clock offset
                                 #   (9 x DRIFT_TAU_S: well damped; 2 x gave a slow oscillation)
DRIFT_REF_S = 5.0                # after start-up alignment the head-time differences are averaged this long ...
DRIFT_DEAD_S = 0.0004            # ... and held to within this (6 samples); smaller errors move nothing
DRIFT_FAST_S = 0.002             # errors beyond this (a dropped block, start-up) ...
DRIFT_FAST_TAU_S = 3.0           # ... are pulled in with this time constant

# Buffer bounds (overflow protection). Normal FIFO depth sits at TARGET_OUT;
# this ceiling is only a safety net that absorbs bursts of stalled worker
# ticks before dropping oldest audio to keep latency bounded.
MAX_IN_SECONDS = 2.0              # max input backlog per mic (s)
MAX_OUT_BLOCKS = 16              # max resampled FIFO depth (blocks, ~1.6 s)

DIAG_PERIOD_S = 5.0

# AGC (see ContinuousEnhancer._agc)
AGC_ABOVE_FLOOR = 10.0 ** (8.0 / 20.0)     # a segment counts as speech when it is 8 dB above the background level
AGC_FLOOR_RISE_DB_S = 1.0                  # how fast the background estimate (a running minimum) creeps up
AGC_BELOW_PEAK = 10.0 ** (-20.0 / 20.0)    # ... and within 20 dB of the loudest recent segment
AGC_PEAK_TC_S = 5.0                        # release of that peak hold
AGC_LIMIT_RMS = 0.2                        # no segment is published above this RMS (-14 dBFS)
AGC_LIMIT_RAMP_S = 0.005                   # the limiter's gain moves over this long at a segment start
AGC_SEGMENT_N = OUTPUT_RATE // 10          # the AGC works on 0.1 s segments (or the driver's hop, if shorter)


def prime_offsets(ends, block=L_OUT, snap=0.010 * OUTPUT_RATE):
    """Per channel, how many 16 kHz samples its FIFO ends later than the
    earliest-ending one, from the capture time of each FIFO's newest sample.

    Devices that deliver their blocks together (one clock, or one device split
    into channels) differ by whole capture blocks only, and a time stamp is
    good to a millisecond or so, not to a sample: an offset within `snap`
    samples of a whole number of blocks is taken as exactly that many blocks,
    which keeps such devices sample-aligned. Any other offset is the measured
    one, to the nearest sample (independent devices: as good as the stamps).
    Without stamps for every channel the offsets are zero (the old behaviour).
    """
    if any(e is None for e in ends):
        return [0] * len(ends)
    first = min(ends)
    out = []
    for e in ends:
        d = (e - first) * OUTPUT_RATE
        k = round(d / block)
        out.append(int(k * block) if abs(d - k * block) <= snap else int(round(d)))
    return out


# ---------------------------------------------------------------------------
# Drift-correcting asynchronous resampler to 16 kHz.
# ---------------------------------------------------------------------------
class Resampler:
    """Variable-ratio resampler (libsamplerate) with a scipy polyphase fallback.

    The variable-ratio path lets the drift loop nudge the conversion ratio a
    fraction of a percent per tick so the resampler — not a growing buffer —
    absorbs the difference between a USB ADC's true clock and its nominal rate.
    The scipy fallback is fixed-ratio (no per-tick trim) but still produces an
    exact-rate stream good enough when libsamplerate is unavailable.
    """

    def __init__(self, input_rate, output_rate=OUTPUT_RATE):
        self.input_rate = float(input_rate)
        self.output_rate = float(output_rate)
        self.base_ratio = output_rate / float(input_rate)
        self.backend = "unknown"
        self._r = None
        self.variable = False

        try:
            import samplerate as _libsr
            # sinc_fastest: best latency/quality trade for a Pi-class CPU.
            self._r = _libsr.Resampler("sinc_fastest", channels=1)
            self.backend = "libsamplerate(sinc_fastest)"
            self.variable = True
        except Exception:
            from scipy.signal import resample_poly  # noqa: F401
            gcd = math.gcd(int(round(input_rate)), int(round(output_rate)))
            self.up = int(round(output_rate)) // gcd
            self.down = int(round(input_rate)) // gcd
            self.backend = f"scipy.resample_poly({self.up}:{self.down})"

    def process(self, x, ratio_scale=1.0):
        """Resample float32 mono `x`; `ratio_scale` trims the ratio (variable
        backend only) for closed-loop drift correction."""
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self._r is not None:
            ratio = self.base_ratio * ratio_scale
            return self._r.process(
                x.astype(np.float32, copy=False), ratio, end_of_input=False
            )
        from scipy.signal import resample_poly
        return resample_poly(
            x.astype(np.float32, copy=False), self.up, self.down
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# One microphone: capture ring buffer + DC block + async resample to 16 kHz.
# ---------------------------------------------------------------------------
class MicChannel:
    """Owns capture, drift-corrected resampling and the 16 kHz output FIFO for
    a single microphone. All cross-thread state is guarded by `self.lock`."""

    def __init__(self, label, device, sample_rate, name="", hpf_hz=150.0,
                 capture_channels=1):
        self.label = label
        self.device = device
        self.sample_rate = int(sample_rate)
        self.name = name
        self.capture_channels = int(capture_channels)

        self.lock = threading.Lock()
        self.in_buf = np.zeros(0, dtype=np.float32)     # raw capture samples
        self.out_fifo = np.zeros(0, dtype=np.float32)   # resampled 16 kHz
        self.last_capture_mono = None
        self.in_end_mono = None        # capture time of the newest sample in in_buf
        self.fifo_end_mono = None      # ... and of the newest sample in out_fifo

        self.resampler = Resampler(self.sample_rate, OUTPUT_RATE)
        self.ratio_scale = 1.0
        self.drift_err = 0.0           # smoothed head-time error against the other channels (s)
        self.drift_ref = 0.0           # that error as start-up alignment left it: what the loop holds
        self.drift_ref_t = 0.0         # seconds of it averaged so far (the loop waits for DRIFT_REF_S)
        self.drift_int = 0.0           # integral part of the ratio correction
        self.max_in = int(self.sample_rate * MAX_IN_SECONDS)
        self.max_out = MAX_OUT_BLOCKS * L_OUT

        # Speech high-pass applied to the 16 kHz stream. Cheap USB capsules can
        # put out huge sub-200 Hz rumble that swamps the speech band and wrecks
        # the beamformer's per-channel statistics; a 2nd-order Butterworth
        # high-pass (default 150 Hz) removes it while preserving intelligibility
        # (ASR only needs the ~300-3400 Hz band). Designed at OUTPUT_RATE for
        # numerical stability (low normalized cutoff is fine at 16 kHz).
        self.hpf_hz = float(hpf_hz)
        if self.hpf_hz and self.hpf_hz > 0:
            self.hpf_b, self.hpf_a = butter(
                2, self.hpf_hz / (OUTPUT_RATE * 0.5), btype="high")
            self.hpf_zi = lfilter_zi(self.hpf_b, self.hpf_a) * 0.0
        else:
            self.hpf_b = self.hpf_a = self.hpf_zi = None

        # Diagnostics
        self.underruns = 0
        self.in_overflows = 0
        self.out_overflows = 0
        self.xruns = 0

    def push(self, samples, capture_mono):
        """Called from the PortAudio callback. Append raw capture, bounding the
        buffer so a stalled worker cannot grow memory without limit. Filtering
        is done once at 16 kHz in resample_in (cheaper and numerically nicer
        than at the 48 kHz input rate)."""
        with self.lock:
            self.in_buf = np.concatenate(
                (self.in_buf, samples.astype(np.float32, copy=False)))
            if self.in_buf.size > self.max_in:
                self.in_buf = self.in_buf[-self.max_in:]
                self.in_overflows += 1
            self.last_capture_mono = capture_mono
            self.in_end_mono = capture_mono + len(samples) / float(self.sample_rate)

    def resample_in(self):
        """Resample everything captured since the last tick into the 16 kHz
        FIFO, high-pass to strip sub-band rumble, bounding it so latency can
        never run away."""
        with self.lock:
            x = self.in_buf
            self.in_buf = np.zeros(0, dtype=np.float32)
            if x.size:
                self.fifo_end_mono = self.in_end_mono

        y = self.resampler.process(x, self.ratio_scale)
        if y.size and self.hpf_b is not None:
            yf, self.hpf_zi = lfilter(
                self.hpf_b, self.hpf_a, y.astype(np.float64), zi=self.hpf_zi)
            y = yf.astype(np.float32)
        if y.size:
            self.out_fifo = np.concatenate((self.out_fifo, y))

        if len(self.out_fifo) > self.max_out:
            self.out_fifo = self.out_fifo[-self.max_out:]
            self.out_overflows += 1

    def head_time(self):
        """Capture time of the sample at the head of the FIFO (the next one to
        be popped), or None before any audio has arrived."""
        if self.fifo_end_mono is None:
            return None
        return self.fifo_end_mono - len(self.out_fifo) / float(OUTPUT_RATE)

    def update_ratio(self, err_s, dt=BLOCK_S):
        """Trim the resample ratio so that this channel's head sample keeps its
        capture time against the other channels' (`err_s` = how far this
        channel's head time has moved from where start-up alignment left it,
        in seconds). A head that is later than it was means the channel has
        too few samples for the time it covers: the ratio goes up. Errors
        inside DRIFT_DEAD_S are stamp noise as far as the loop can tell and
        change nothing, so channels on one clock are never moved. Absolute
        FIFO depth is regulated by the consumer, not here."""
        if not self.resampler.variable:
            return
        a = math.exp(-dt / DRIFT_SMOOTH_S)
        self.drift_err = a * self.drift_err + (1.0 - a) * err_s
        e = math.copysign(max(0.0, abs(self.drift_err) - DRIFT_DEAD_S), self.drift_err)
        # The integral sees at most DRIFT_FAST_S of error, so a transient (a lost
        # block) cannot wind it up.
        self.drift_int += max(-DRIFT_FAST_S, min(DRIFT_FAST_S, e)) * dt / (DRIFT_TAU_S * DRIFT_INT_TAU_S)
        self.drift_int = max(-RATIO_TRIM_MAX, min(RATIO_TRIM_MAX, self.drift_int))
        fast = math.copysign(max(0.0, abs(e) - DRIFT_FAST_S), e) / DRIFT_FAST_TAU_S
        corr = e / DRIFT_TAU_S + fast + self.drift_int
        self.ratio_scale = 1.0 + max(-RATIO_TRIM_MAX, min(RATIO_TRIM_MAX, corr))

    def fifo_len(self):
        return len(self.out_fifo)

    def trim_fifo(self, keep):
        """Drop the oldest samples so the FIFO holds exactly `keep` (used once
        at prime time to phase-align all channels to a common latency)."""
        if len(self.out_fifo) > keep:
            self.out_fifo = self.out_fifo[-keep:]

    def pop_block(self):
        """Return exactly L_OUT samples; pad with zeros on underrun so a slow
        or dead mic degrades gracefully instead of stalling the array."""
        if len(self.out_fifo) >= L_OUT:
            out = self.out_fifo[:L_OUT]
            self.out_fifo = self.out_fifo[L_OUT:]
            return out
        out = np.concatenate(
            (self.out_fifo, np.zeros(L_OUT - len(self.out_fifo), dtype=np.float32))
        )
        self.out_fifo = np.zeros(0, dtype=np.float32)
        self.underruns += 1
        return out


# ---------------------------------------------------------------------------
# Device discovery — prefer external USB inputs, fall back to internal.
# ---------------------------------------------------------------------------
def is_usable_input(name: str, include_internal: bool) -> bool:
    exclude = (
        "blackhole", "loopback", "zoom", "teams", "microsoft",
        "obs", "background music", "monitor", "pulse", "default", "sysdefault",
    )
    low = name.lower()
    if any(t in low for t in exclude):
        return False
    if not include_internal and any(
        t in low for t in ("macbook", "built-in", "internal", "hdmi")
    ):
        return False
    return True


def looks_usb(name: str) -> bool:
    low = name.lower()
    return any(t in low for t in ("usb", "mic", "uac", "card"))


def choose_rate(device: int, requested_rate=None, channels=1) -> int:
    info = sd.query_devices(device)
    default_rate = int(info["default_samplerate"])
    candidates = []
    if requested_rate and requested_rate > 0:
        candidates.append(int(requested_rate))
    candidates += [48000, 44100, 32000, 16000, default_rate]
    seen = set()
    for rate in candidates:
        if rate in seen:
            continue
        seen.add(rate)
        try:
            sd.check_input_settings(
                device=device, channels=channels, dtype="float32", samplerate=rate
            )
            return int(rate)
        except Exception:
            continue
    return default_rate


def lock_channels(mics, dt=BLOCK_S):
    """One step of the drift loop, called once the channels are aligned.

    Each channel's head sample has a capture time against the mean of all
    heads. For the first DRIFT_REF_S that difference is only averaged: it is
    what start-up alignment left (whole blocks are exact there, stamp
    differences below its snap are deliberately ignored). From then on the
    loop holds the channel at that reference, so it removes what clocks do
    over time and nothing else. Returns the head-time differences in seconds
    (None until every channel has delivered audio)."""
    heads = [ch.head_time() for ch in mics]
    if len(mics) < 2 or any(h is None for h in heads):
        return None
    ref = sum(heads) / len(heads)
    errs = [h - ref for h in heads]
    for ch, e in zip(mics, errs):
        if ch.drift_ref_t < DRIFT_REF_S:
            ch.drift_ref_t += dt
            ch.drift_ref += (e - ch.drift_ref) * min(1.0, dt / ch.drift_ref_t)
        else:
            ch.update_ratio(e - ch.drift_ref, dt)
    return errs


SIM_HEADER = 4     # a simulated device's message: [seq, rate, stamp // 1000, stamp % 1000, samples...]


def parse_device_msg(data):
    """(seq, rate, capture stamp or None, samples) of one message from a
    simulated capture device (the sim package's virtual microphones): one block of one
    device's audio at the device's own rate, as a PortAudio callback delivers
    it. The stamp is the monotonic capture time of the block's first sample,
    split in two so that float32 keeps it to 0.1 ms; zeros mean "no stamp"."""
    data = np.asarray(data, dtype=np.float32)
    if data.size <= SIM_HEADER:
        return None
    stamp = float(data[2]) * 1000.0 + float(data[3])
    return int(round(float(data[0]))), int(round(float(data[1]))), (stamp if stamp > 0 else None), data[SIM_HEADER:]


def sync_status(mics, drift_correction, seq=0):
    """What a consumer of the raw tap needs to know about channel timing:

        [seq, n_channels, n_devices, drift_correction (0|1),
         largest rate trim (ppm), largest loop error (us)]

    Each microphone is its own capture device here, so with drift correction
    on the channels are held together by capture time stamps (good to about a
    millisecond) and a per-channel resampler: right for beamforming, not good
    enough for a direction-of-arrival estimate on a small array, which reads
    tens of microseconds. doa_mod turns "several devices, drift correction on"
    into a warning. The loop error is how far a channel's smoothed head time
    stands from the reference the loop holds (inside DRIFT_DEAD_S the loop
    does nothing): it says how well the loop is locked, not how well the
    channels are truly aligned.
    """
    errs = [abs(ch.drift_err) for ch in mics]
    return [float(seq), float(len(mics)),
            float(len({ch.device for ch in mics})),
            1.0 if drift_correction else 0.0,
            max((abs(ch.ratio_scale - 1.0) for ch in mics), default=0.0) * 1e6,
            max(errs, default=0.0) * 1e6]


def detect_microphones(logger, include_internal=False, max_mics=3,
                       requested_rate=None, prefer_usb=True,
                       capture_channels=0, mic_name_filter=""):
    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices)
              if d.get("max_input_channels", 0) > 0]

    usable = [(i, d) for i, d in inputs
              if is_usable_input(d["name"], include_internal)]
    logger.info(f"Found {len(usable)} usable input(s) of {len(inputs)} total.")

    if not usable and not include_internal:
        logger.warning("No external mics; retrying with internal inputs.")
        usable = [(i, d) for i, d in inputs
                  if is_usable_input(d["name"], include_internal=True)]

    if mic_name_filter:
        # An explicit pin is applied to ALL inputs, not just `usable`, so it
        # overrides the virtual-device exclusion above. This is what lets a
        # simulation feed the node through an ALSA loopback (snd-aloop:
        # `aplay` a recording into hw:N,0 and pin "Loopback: PCM (hw:N,1)"),
        # exercising the real capture -> resample -> beamform -> publish path
        # with known audio. See test/sim_pipeline.sh.
        needle = mic_name_filter.lower()
        filtered = [(i, d) for i, d in inputs if needle in d["name"].lower()]
        logger.info(
            f"mic_name_filter='{mic_name_filter}' kept {len(filtered)} of "
            f"{len(inputs)} input(s)."
        )
        if filtered:
            usable = filtered
        else:
            logger.warning(
                f"mic_name_filter='{mic_name_filter}' matched no devices; "
                f"ignoring the filter."
            )

    if prefer_usb:
        # Stable order: USB-looking devices first, then by device index.
        usable.sort(key=lambda it: (not looks_usb(it[1]["name"]), it[0]))

    resolved = []
    for idx, dev in usable:
        if len(resolved) >= max_mics:
            break
        # Decide how many channels to open. 0 = auto: native count capped at 2
        # (stereo-native mics like the PULUZ PU425 need their native format to
        # deliver full level). >0 forces that exact count. Always clamped to
        # [1, 2]; the callback downmixes whatever we open back to one mono
        # stream, so downstream stays single-channel-per-mic either way.
        native = int(dev.get("max_input_channels", 1))
        cap_ch = native if capture_channels == 0 else capture_channels
        cap_ch = max(1, min(cap_ch, 2))
        rate = choose_rate(idx, requested_rate, channels=cap_ch)
        label = f"mic{len(resolved)}"
        logger.info(
            f"Auto-detected {label}: dev #{idx} '{dev['name']}' @ {rate} Hz "
            f"({cap_ch}ch capture -> mono)"
        )
        resolved.append({
            "label": label, "device": idx,
            "sample_rate": rate, "name": dev["name"],
            "capture_channels": cap_ch,
        })

    if not resolved:
        raise RuntimeError("No usable microphone devices could be resolved.")
    return resolved


# ---------------------------------------------------------------------------
# dorai_beamformer.ort beamformer wrapper.
# ---------------------------------------------------------------------------
class Beamformer:
    """Thin ONNX wrapper: run one [M, T] multichannel window through the model
    and return the clean [T] mono.

    dorai_beamformer.ort is a *whole-signal* model: its internal STFT / Wiener
    stage needs surrounding context, and two independent facts govern how it
    must be driven:

      1. Its output is peak-normalised to ~0.9 *regardless of input scale* — the
         model is completely gain-invariant. So there is no point normalising
         the input by its peak and multiplying the output back by it (the old
         path did exactly that): the round-trip only re-injects the input's
         peak jitter and does nothing useful. Output level must be set by a
         separate, slow AGC on the continuous stream (see ContinuousEnhancer).

      2. Its synthesis leaves large *uncompensated* samples at the extreme
         edges of the window. Enhancing disjoint frames and concatenating them
         therefore injects a step at every join — an audible 'knock' at the
         publish-frame rate. The cure is overlap-add across a sliding window,
         which ContinuousEnhancer does; this class stays a pure per-window
         call so it can be composed into that overlap-add machinery.
    """

    def __init__(self, model_path, num_threads, logger):
        so = ort.SessionOptions()
        so.log_severity_level = 3          # errors only (suppress GPU-probe warns)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads and num_threads > 0:
            so.intra_op_num_threads = int(num_threads)
            so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.logger = logger

    def warmup(self, num_channels):
        dummy = np.zeros((num_channels, WINDOW_SIZE), dtype=np.float32)
        for _ in range(2):
            self.sess.run([self.out_name], {self.in_name: dummy})

    def run(self, mc):
        """Enhance one [M, T] window in a single pass; return clean [T] float32.

        No input/output scaling here — the model is gain-invariant, so scaling
        is a no-op that would only distort level. On failure return zeros of the
        input length so a bad window never kills the stream."""
        x = np.ascontiguousarray(mc.astype(np.float32, copy=False))
        L = x.shape[1]
        try:
            out = self.sess.run([self.out_name], {self.in_name: x})[0]
            return np.asarray(out, dtype=np.float32).reshape(-1)[:L]
        except Exception as e:  # never let inference kill the stream
            self.logger.error(
                f"Inference failed: {e}", throttle_duration_sec=2.0)
            return np.zeros(L, dtype=np.float32)


class ContinuousEnhancer:
    """Overlap-add streaming wrapper around the whole-signal enhancement model.

    Slides a ``win``-sample analysis window across the *continuous* mic stream
    in ``hop``-sample steps, enhances each window with full context, applies a
    Hann synthesis window and overlap-adds with running window-sum
    normalisation (correct for any win/hop, not just exact COLA). This removes
    the boundary 'knock' the old disjoint-frame path produced: output is
    bit-identical no matter how the input is chunked, so publishing in
    publish_interval batches introduces no seam.

    Level is set by ONE slow, gated RMS AGC on the reconstructed stream — never
    by per-frame peak grabbing. Because the model already peak-normalises every
    window to ~0.9, per-frame peak scaling only pumps loudness up and down and
    amplifies noise in speech-sparse frames (the old 'bad gain for STT'); a slow
    AGC with a floor gate holds a steady speech level and leaves silence alone.

    Algorithmic latency is ``win`` samples (one window must fill before the
    first hop is emitted). Not thread-safe: drive push()/flush() from a single
    thread (here, the publisher thread), or guard with an external lock.
    """

    def __init__(self, run_model, win, hop, fs=OUTPUT_RATE,
                 agc_target_rms=0.06, agc_tc_s=2.0, agc_max_gain=15.0,
                 agc_gate_rms=3e-3, out_ceiling=0.99):
        assert 0 < hop <= win, "require 0 < hop <= win"
        self.run = run_model
        self.win = int(win)
        self.hop = int(hop)
        self.fs = int(fs)
        self.wsyn = np.hanning(self.win).astype(np.float64)
        self.inbuf = None                          # [M, t] pending input
        self.acc = np.zeros(self.win)              # OLA numerator (len win)
        self.accw = np.zeros(self.win)             # OLA denominator (len win)
        self.g = 1.0                               # smoothed AGC gain
        self.tgt = float(agc_target_rms)
        self.alpha = float(np.exp(-self.hop / (agc_tc_s * self.fs)))
        self.max_gain = float(agc_max_gain)
        self.gate = float(agc_gate_rms)
        self.ceil = float(out_ceiling)

    def _agc(self, ready):
        """Slow AGC toward `tgt`, driven by speech only, with a separate limiter.

        The first version set the gain from every segment above an absolute
        gate and held it constant per segment. An enhancer that cleans the
        pauses leaves residue that is far below the speech but still above that
        gate; following it drove the gain to its maximum in every pause and the
        next word into the ceiling, and constant gain per segment meant steps
        of up to 16 dB at the old driver's 1 s segment edges. Now:
          * A segment moves the slow gain only if it stands out from the
            background (8 dB above a running minimum of the segment level that
            creeps up by 1 dB/s) and is within 20 dB of the loudest recent
            segment (peak hold, 5 s release). Steady noise, cleaned pauses and
            the first seconds of a stream therefore leave the gain alone; a
            bang holds the gate shut for about 2 s, not for half a minute.
          * The slow gain is ramped across the segment, never stepped.
          * A limiter, separate from the slow gain and never feeding back into
            it, keeps every segment at or under `AGC_LIMIT_RMS`; its gain moves
            over the first 5 ms of a segment.
          * All of this runs on 0.1 s segments whatever the driver's hop is.
            With the old driver's 1 s segments running speech has no pauses to
            measure a background against, and the gain would never move.
        """
        out = np.empty(len(ready), dtype=np.float32)
        floor = getattr(self, "agc_floor", None)
        peak = getattr(self, "agc_peak", 0.0)
        lim_prev = getattr(self, "agc_lim", 1.0)
        hop = min(self.hop, AGC_SEGMENT_N)
        alpha = self.alpha ** (hop / float(self.hop))          # the same time constant on the shorter segment
        seg_s = hop / float(self.fs)
        floor_up = 10.0 ** (AGC_FLOOR_RISE_DB_S * seg_s / 20.0)
        peak_decay = float(np.exp(-seg_s / AGC_PEAK_TC_S))
        ramp_n = max(1, int(AGC_LIMIT_RAMP_S * self.fs))
        for s in range(0, len(ready), hop):
            seg = ready[s:s + hop]
            rms = float(np.sqrt((seg ** 2).mean() + 1e-12))
            whole = len(seg) * 4 >= hop                         # a short tail says little about the level
            if whole:
                floor = rms if floor is None else min(rms, floor * floor_up)
                peak = max(rms, peak * peak_decay)
            active = (whole and floor is not None and rms > self.gate
                      and rms >= AGC_ABOVE_FLOOR * floor and rms >= AGC_BELOW_PEAK * peak)
            g_inst = min(self.tgt / rms, self.max_gain) if active else self.g
            g_new = alpha * self.g + (1.0 - alpha) * g_inst
            gain = np.linspace(self.g, g_new, len(seg), endpoint=False)
            lim_new = min(1.0, AGC_LIMIT_RMS / (rms * max(self.g, g_new)))
            lim = np.full(len(seg), lim_new)
            k = min(ramp_n, len(seg))
            lim[:k] = np.linspace(lim_prev, lim_new, k, endpoint=False)
            out[s:s + len(seg)] = np.clip(seg * gain * lim, -self.ceil, self.ceil)
            self.g, lim_prev = g_new, lim_new
        self.agc_floor, self.agc_peak, self.agc_lim = floor, peak, lim_prev
        return out

    def _consume(self):
        ready = []
        while self.inbuf is not None and self.inbuf.shape[1] >= self.win:
            raw = np.asarray(self.run(self.inbuf[:, :self.win]),
                             dtype=np.float64).reshape(-1)
            c = np.zeros(self.win)
            m = min(len(raw), self.win)            # model may trim a few edge samples
            c[:m] = raw[:m]
            c *= self.wsyn
            self.acc += c
            self.accw += self.wsyn
            ready.append((self.acc[:self.hop]
                          / np.maximum(self.accw[:self.hop], 1e-8)).copy())
            self.acc = np.concatenate([self.acc[self.hop:], np.zeros(self.hop)])
            self.accw = np.concatenate([self.accw[self.hop:], np.zeros(self.hop)])
            self.inbuf = self.inbuf[:, self.hop:]
        return np.concatenate(ready) if ready else np.zeros(0)

    def push(self, mc):
        """Feed raw [M, n] 16 kHz audio; return finished clean samples (>= 0).

        Returned length lags the input by up to `win` samples (state held for
        overlap); the held tail is emitted by flush()."""
        mc = np.asarray(mc, dtype=np.float32)
        self.inbuf = mc.copy() if self.inbuf is None \
            else np.concatenate([self.inbuf, mc], axis=1)
        r = self._consume()
        return self._agc(r) if len(r) else np.zeros(0, dtype=np.float32)

    def flush(self):
        """Drain the tail on shutdown by zero-padding to a whole window so the
        last words aren't lost."""
        if self.inbuf is None or self.inbuf.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)
        rem = self.inbuf.shape[1]
        pad = (self.win - rem) if rem < self.win \
            else ((self.hop - rem % self.hop) % self.hop)
        if pad:
            self.inbuf = np.concatenate(
                [self.inbuf, np.zeros((self.inbuf.shape[0], pad), np.float32)],
                axis=1)
        r = self._consume()
        return self._agc(r) if len(r) else np.zeros(0, dtype=np.float32)


class StreamingEnhancer:
    """Driver for a *stateful streaming* enhancement model (QWise 2).

    Such a model takes one block of new samples per microphone together with a
    state tensor and returns the enhanced block and the next state:

        mic   float32 [M, 512]     state float32 [S]   (zeros start a stream)
        clean float32 [512]        state_next float32 [S]

    Everything the model has learned about the scene travels in the state, so
    there is no analysis window to slide and nothing to overlap-add: the audio
    is simply fed through in 512-sample blocks and comes out continuous, with
    the model's fixed algorithmic delay (384 samples, 24 ms) instead of
    `enh_window` seconds. The output is in input units (the model does not
    normalise level), so the same slow gated AGC as ContinuousEnhancer sets the
    level for STT.

    Same interface as ContinuousEnhancer (push / flush), same threading rule:
    drive it from one thread.
    """

    def __init__(self, model_path, num_threads, logger, fs=OUTPUT_RATE,
                 agc_target_rms=0.06, agc_tc_s=2.0, agc_max_gain=15.0,
                 agc_gate_rms=3e-3, out_ceiling=0.99, agc_hop=L_OUT):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads and num_threads > 0:
            so.intra_op_num_threads = int(num_threads)
            so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"])
        ins = self.sess.get_inputs()
        self.in_name, self.state_name = ins[0].name, ins[1].name
        self.n_mics, self.block = int(ins[0].shape[0]), int(ins[0].shape[1])
        self.state_size = int(ins[1].shape[0])
        meta = self.sess.get_modelmeta().custom_metadata_map or {}
        # Algorithmic delay of the model: its STFT overlap (n_fft - hop).
        self.latency = int(meta.get("latency_samples", 384))
        self.algorithm = meta.get("algorithm", "?")
        self.logger = logger
        self.fs = int(fs)
        self.hop = int(agc_hop)                    # AGC update grid only
        self.g = 1.0
        self.tgt = float(agc_target_rms)
        self.alpha = float(np.exp(-self.hop / (agc_tc_s * self.fs)))
        self.max_gain = float(agc_max_gain)
        self.gate = float(agc_gate_rms)
        self.ceil = float(out_ceiling)
        self.failures = 0
        self.reset()

    @staticmethod
    def accepts(model_path):
        """True when the file is a streaming model (audio block + state inputs)."""
        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            ins = ort.InferenceSession(
                model_path, sess_options=so, providers=["CPUExecutionProvider"]).get_inputs()
        except Exception:
            return False
        return len(ins) == 2 and len(ins[0].shape) == 2 and len(ins[1].shape) == 1

    def reset(self):
        """Start a new stream: forget the scene and the pending samples."""
        self.state = np.zeros(self.state_size, dtype=np.float32)
        self.inbuf = np.zeros((self.n_mics, 0), dtype=np.float32)
        self.skip = self.latency                   # drop the model's start-up delay once
        self.owed = 0                              # input samples not yet returned

    def warmup(self):
        dummy = np.zeros((self.n_mics, self.block), dtype=np.float32)
        state = np.zeros(self.state_size, dtype=np.float32)
        for _ in range(2):
            _, state = self.sess.run(None, {self.in_name: dummy, self.state_name: state})

    _agc = ContinuousEnhancer._agc                 # the same slow gated AGC

    def _run_blocks(self):
        out = []
        while self.inbuf.shape[1] >= self.block:
            blk = np.ascontiguousarray(self.inbuf[:, :self.block])
            self.inbuf = self.inbuf[:, self.block:]
            try:
                y, self.state = self.sess.run(
                    None, {self.in_name: blk, self.state_name: self.state})
                y = np.asarray(y, dtype=np.float64).reshape(-1)
                if not np.isfinite(y).all():
                    raise FloatingPointError("non-finite output")
            except Exception as e:     # never let inference kill the stream
                self.failures += 1
                self.logger.error(
                    f"Inference failed ({e}); stream state reset", throttle_duration_sec=2.0)
                self.state = np.zeros(self.state_size, dtype=np.float32)
                y = np.zeros(self.block)
            out.append(y)
        if not out:
            return np.zeros(0)
        y = np.concatenate(out)
        if self.skip:                              # align output with input once
            drop = min(self.skip, len(y))
            y, self.skip = y[drop:], self.skip - drop
        return y

    def push(self, mc):
        """Feed raw [M, n] 16 kHz audio; return the clean samples that are
        finished. Output lags input by less than one block plus the model's
        delay (under 56 ms); flush() returns the rest."""
        mc = np.asarray(mc, dtype=np.float32)
        self.inbuf = np.concatenate([self.inbuf, mc], axis=1)
        self.owed += mc.shape[1]
        y = self._run_blocks()
        self.owed -= len(y)
        return self._agc(y) if len(y) else np.zeros(0, dtype=np.float32)

    def flush(self):
        """Drain the tail on shutdown: feed silence until every input sample
        has come out."""
        if self.owed <= 0:
            return np.zeros(0, dtype=np.float32)
        need = self.owed + self.skip - self.inbuf.shape[1]
        pad = need + (-(self.inbuf.shape[1] + need)) % self.block
        self.inbuf = np.concatenate(
            [self.inbuf, np.zeros((self.n_mics, max(pad, 0)), np.float32)], axis=1)
        y = self._run_blocks()[:self.owed]
        self.owed = 0
        return self._agc(y) if len(y) else np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------------------
# ROS 2 node.
# ---------------------------------------------------------------------------
class VoiceMod(Node):
    def __init__(self):
        super().__init__("voice_mod")

        # Float params use dynamic typing so both `p:=10` (int) and `p:=10.0`
        # (double) are accepted on the command line without a type error.
        from rcl_interfaces.msg import ParameterDescriptor
        dyn = ParameterDescriptor(dynamic_typing=True)

        # Where the audio comes from. 'mic': capture devices (the robot).
        # 'topic': one topic per microphone, published by the sim package's virtual
        # microphones; each message is one capture block of one device, and
        # everything behind the capture callback (resampler, alignment, drift
        # loop, beamformer) runs exactly as with real devices.
        self.declare_parameter("source", "mic")
        self.declare_parameter("input_topics", ["/dorai_sim/mic0", "/dorai_sim/mic1", "/dorai_sim/mic2"])
        self.declare_parameter("input_rate", 48000)
        self.declare_parameter("include_internal", False)
        self.declare_parameter("max_mics", 3)
        self.declare_parameter("requested_rate", 48000)
        self.declare_parameter("prefer_usb", True)
        # Case-insensitive substring; when set, only input devices whose name
        # contains it are used as array mics. Lets you pin the real array (e.g.
        # "earpod") and exclude dead/irrelevant inputs like an empty dock jack
        # that would otherwise be auto-selected and collapse the beamformer.
        self.declare_parameter("mic_name_filter", "")
        # Per-mic capture channel count. 0 = auto (open each device at its native
        # channel count, capped at 2, then downmix to mono); 1 = force the legacy
        # mono-open path; 2 = force stereo. Some USB-C lavaliers (e.g. PULUZ
        # PU425) are stereo-native and deliver a ~3x weaker signal when opened
        # mono, so auto restores full level while staying mono downstream.
        self.declare_parameter("capture_channels", 0)
        self.declare_parameter("publish_interval", 10.0, dyn)   # N-second frame
        self.declare_parameter("output_topic", "/dorai_clean_audio")
        # On a 4-core Pi 4B, letting ORT grab all cores starves the PortAudio
        # capture threads and causes input overflows. 2 leaves headroom.
        self.declare_parameter("num_threads", 2)           # 0 = ORT default
        # PortAudio input latency (s). Larger = bigger OS capture buffer = more
        # tolerance to worker/CPU jitter before an overflow. <=0 => 'high'.
        self.declare_parameter("input_latency", 0.0, dyn)
        # Debug: also publish the pre-beamformer multichannel audio so the
        # recorder can capture raw + clean simultaneously, time-aligned.
        self.declare_parameter("publish_raw", False)
        self.declare_parameter("raw_topic", "/dorai_raw_audio")
        # Speech high-pass cutoff (Hz) per mic. Raise toward ~200 if low-freq
        # rumble from cheap USB mics still dominates; 0 disables.
        self.declare_parameter("hpf_hz", 150.0, dyn)
        # Clock-drift correction between the microphones (each USB device has
        # its own crystal). Leave on for separate devices; switch off only when
        # all channels come from one clock (a simulation through snd-aloop).
        self.declare_parameter("drift_correction", True)
        # Continuous overlap-add enhancement window / hop (seconds). The model
        # is a whole-signal enhancer, so a longer window gives it more context
        # and tracks the offline reference more closely, at the cost of added
        # latency (= enh_window). 2.0/1.0 is the quality/latency balance point;
        # hop must be <= window (50% overlap at the default). Changing these
        # does NOT change the seam behaviour — overlap-add stays seamless.
        self.declare_parameter("enh_window", 2.0, dyn)
        self.declare_parameter("enh_hop", 1.0, dyn)
        # Output loudness: a single SLOW, gated RMS AGC lifts the continuous
        # clean stream toward agc_target_rms so STT sees a steady level. Unlike
        # the old per-frame peak grab, it does not pump between frames and does
        # not amplify silence: agc_gate_rms floors it (segments quieter than the
        # gate hold the last gain) and agc_max_gain caps the boost. agc_tc_s is
        # the gain smoothing time-constant (larger = slower, steadier).
        self.declare_parameter("agc_target_rms", 0.06, dyn)
        self.declare_parameter("agc_tc_s", 2.0, dyn)
        self.declare_parameter("agc_max_gain", 15.0, dyn)
        self.declare_parameter("agc_gate_rms", 0.003, dyn)
        # Debug: per-frame capture -> clean-speech delay metering. Setting a
        # non-empty path auto-enables metering (no separate boolean needed)
        # and appends one CSV row per published frame:
        #   seq,capture_epoch,clean_ready_epoch,delay_ms,samples,frame_s
        # start_time = wall-clock when the frame's first raw sample was
        # captured (mic ADC time); end_time = wall-clock right after dorai_beamformer
        # enhancement finishes for that frame (pre-publish). The delta is
        # dominated by the whole-signal buffering wait (~publish_interval)
        # plus inference time — useful for confirming the pipeline stays
        # within its expected latency budget on the Pi.
        self.declare_parameter("delay_log_path", "")
        self.declare_parameter("model_path", self._default_model_path())

        gp = self.get_parameter
        self.include_internal = gp("include_internal").value
        self.max_mics = int(gp("max_mics").value)
        req_rate = int(gp("requested_rate").value)
        self.requested_rate = req_rate if req_rate > 0 else None
        self.prefer_usb = bool(gp("prefer_usb").value)
        self.mic_name_filter = str(gp("mic_name_filter").value or "").strip()
        self.capture_channels = int(gp("capture_channels").value)
        self.publish_interval = float(gp("publish_interval").value)
        self.output_topic = gp("output_topic").value
        num_threads = int(gp("num_threads").value)
        in_lat = float(gp("input_latency").value)
        self.input_latency = in_lat if in_lat > 0 else "high"
        self.publish_raw = bool(gp("publish_raw").value)
        self.raw_topic = gp("raw_topic").value
        self.hpf_hz = float(gp("hpf_hz").value)
        self.drift_correction = bool(gp("drift_correction").value)
        self.enh_window = float(gp("enh_window").value)
        self.enh_hop = float(gp("enh_hop").value)
        self.agc_target_rms = float(gp("agc_target_rms").value)
        self.agc_tc_s = float(gp("agc_tc_s").value)
        self.agc_max_gain = float(gp("agc_max_gain").value)
        self.agc_gate_rms = float(gp("agc_gate_rms").value)
        self.delay_log_path = str(gp("delay_log_path").value or "").strip()
        model_path = gp("model_path").value

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"dorai_beamformer model not found: {model_path}")

        self.get_logger().info(
            f"voice_mod starting: model={model_path}, "
            f"publish_interval={self.publish_interval:.1f}s"
        )

        self.source = str(gp("source").value).lower().strip()
        if self.source not in ("mic", "topic"):
            raise ValueError(f"source must be 'mic' or 'topic', got {self.source!r}")
        if self.source == "topic":
            rate = int(gp("input_rate").value)
            resolved = [{"label": f"mic{i}", "device": i, "sample_rate": rate, "name": str(t), "capture_channels": 1}
                        for i, t in enumerate(gp("input_topics").value)]
            self.get_logger().info(
                f"source=topic: {len(resolved)} simulated microphone(s) at {rate} Hz on "
                + ", ".join(m["name"] for m in resolved))
        else:
            resolved = detect_microphones(
                self.get_logger(), self.include_internal, self.max_mics,
                self.requested_rate, self.prefer_usb, self.capture_channels,
                self.mic_name_filter,
            )
        self.mics = [
            MicChannel(m["label"], m["device"], m["sample_rate"], m["name"],
                       hpf_hz=self.hpf_hz, capture_channels=m["capture_channels"])
            for m in resolved
        ]
        self.num_channels = len(self.mics)

        # The model file decides the driver. A streaming model (audio block +
        # state, e.g. QWise 2) is fed block by block and needs no analysis
        # window; a whole-signal model (dorai_beamformer.ort) goes through the
        # overlap-add sliding window. Either way self.enhancer has push()/flush()
        # and is driven only from the publisher thread (_pub_loop / _flush).
        self.streaming = StreamingEnhancer.accepts(model_path)
        if self.streaming:
            self.beam = None
            self.enhancer = StreamingEnhancer(
                model_path, num_threads, self.get_logger(), fs=OUTPUT_RATE,
                agc_target_rms=self.agc_target_rms, agc_tc_s=self.agc_tc_s,
                agc_max_gain=self.agc_max_gain, agc_gate_rms=self.agc_gate_rms,
            )
            if self.enhancer.n_mics != self.num_channels:
                raise RuntimeError(
                    f"{model_path} is built for {self.enhancer.n_mics} microphone(s) but "
                    f"{self.num_channels} were opened. A streaming model is exported per "
                    f"microphone count: export one for {self.num_channels}, or set max_mics.")
            self.enhancer.warmup()
            self.get_logger().info(
                f"Streaming enhancer ready ({self.num_channels}ch, algorithm "
                f"{self.enhancer.algorithm}): block={self.enhancer.block} samples, "
                f"state={self.enhancer.state_size}, model delay "
                f"{1000.0 * self.enhancer.latency / OUTPUT_RATE:.0f} ms; "
                f"AGC target_rms={self.agc_target_rms:.3f} tc={self.agc_tc_s:.1f}s "
                f"max_gain={self.agc_max_gain:.1f} gate={self.agc_gate_rms:.4f}"
            )
        else:
            self.beam = Beamformer(model_path, num_threads, self.get_logger())
            self.beam.warmup(self.num_channels)
            self.get_logger().info(
                f"dorai_beamformer.ort ready ({self.num_channels}ch) via {self.beam.in_name}"
                f"->{self.beam.out_name}"
            )

            # Continuous overlap-add enhancer wrapping the per-window model call.
            # enh_hop is clamped to <= enh_window.
            win = max(1, int(round(self.enh_window * OUTPUT_RATE)))
            hop = max(1, int(round(min(self.enh_hop, self.enh_window) * OUTPUT_RATE)))
            self.enhancer = ContinuousEnhancer(
                self.beam.run, win=win, hop=hop, fs=OUTPUT_RATE,
                agc_target_rms=self.agc_target_rms, agc_tc_s=self.agc_tc_s,
                agc_max_gain=self.agc_max_gain, agc_gate_rms=self.agc_gate_rms,
            )
            self.get_logger().info(
                f"Continuous overlap-add enhancer: win={win / OUTPUT_RATE:.2f}s "
                f"hop={hop / OUTPUT_RATE:.2f}s (latency +{win / OUTPUT_RATE:.2f}s), "
                f"AGC target_rms={self.agc_target_rms:.3f} tc={self.agc_tc_s:.1f}s "
                f"max_gain={self.agc_max_gain:.1f} gate={self.agc_gate_rms:.4f}"
            )

        self.pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)
        self.diag_pub = self.create_publisher(String, "/voice_mod/diagnostics", 10)
        self.raw_pub = None
        if self.publish_raw:
            self.raw_pub = self.create_publisher(
                Float32MultiArray, self.raw_topic, 10)
            # One small message per raw frame: how the channels are clocked
            # (see sync_status). doa_mod reads it to judge the raw tap.
            self.sync_pub = self.create_publisher(
                Float32MultiArray, self.raw_topic + "/sync", 10)

        self._t0 = time.monotonic()
        self._t0_wall = time.time()
        self._delay_log_file = self._open_delay_log(self.delay_log_path)
        self._seq = 0
        self._raw_seq = 0
        self._acc_lock = threading.Lock()
        self._mc_acc = []                    # raw [M, L_OUT] blocks pending
        self._mc_caps = []                   # capture time per pending block
        self._frame_blocks = max(
            1, round(self.publish_interval * OUTPUT_RATE / L_OUT))
        self._last_diag = time.monotonic()
        self._primed = False
        self._running = True

        # Building a ~160k-element Python list and serializing it over DDS is
        # slow; doing it inline would stall the resample/inference loop long
        # enough to overflow the mic buffers. Hand finished frames to a separate
        # publisher thread so the worker keeps draining the capture buffers.
        self._pubq = queue.Queue(maxsize=8)
        self._pub_thread = threading.Thread(
            target=self._pub_loop, name="voice-pub", daemon=True)
        self._pub_thread.start()

        self._topic_subs = []
        for ch in self.mics:
            if self.source == "topic":
                ch.stream = None
                self._topic_subs.append(self.create_subscription(
                    Float32MultiArray, ch.name, self._make_topic_callback(ch), 50))
                continue
            blocksize = int(ch.sample_rate * BLOCK_S)
            ch.stream = sd.InputStream(
                samplerate=ch.sample_rate, device=ch.device,
                channels=ch.capture_channels,
                blocksize=blocksize, dtype="float32",
                latency=self.input_latency,
                callback=self._make_callback(ch),
            )
            ch.stream.start()
            self.get_logger().info(
                f"Opened {ch.label} dev#{ch.device} @ {ch.sample_rate} Hz "
                f"(blk {blocksize}) via {ch.resampler.backend}"
            )

        self._worker = threading.Thread(
            target=self._run, name="voice-worker", daemon=True)
        self._worker.start()
        self.get_logger().info(
            f"Publishing {self.output_topic}: 1ch @ {OUTPUT_RATE} Hz, "
            f"batched every {self.publish_interval:.1f}s"
        )

    def _open_delay_log(self, path):
        """Open the delay-metering log if `path` is set (auto-enable; no
        separate boolean flag). Returns the open file handle, or None if
        metering is disabled or the path can't be opened."""
        if not path:
            return None
        try:
            d = os.path.dirname(os.path.abspath(path))
            if d:
                os.makedirs(d, exist_ok=True)
            new_file = not os.path.exists(path) or os.path.getsize(path) == 0
            f = open(path, "a", buffering=1)  # line-buffered
            if new_file:
                f.write(
                    "seq,capture_epoch,frame_ready_epoch,dequeue_epoch,"
                    "clean_ready_epoch,publish_done_epoch,"
                    "buffer_ms,queue_ms,infer_ms,publish_ms,total_ms,"
                    "samples,frame_s\n"
                )
            self.get_logger().info(f"Delay metering ENABLED -> {path}")
            return f
        except Exception as e:
            self.get_logger().error(
                f"Could not open delay_log_path '{path}': {e}. "
                f"Delay metering disabled."
            )
            return None

    def _log_delay(self, cap0, seq, n_samples, frame_ready_mono, dequeue_mono,
                    infer_done_mono, publish_done_mono):
        """Append one CSV row breaking the per-frame delay into stages:

          buffer_ms  - capture -> frame fully accumulated (mandatory wait,
                       ~= publish_interval; whole-signal model needs the
                       full frame before it can run)
          queue_ms   - frame accumulated -> publisher thread picked it up
                       (should be ~0; growth means the pub thread is still
                       busy on the previous frame's inference, i.e. falling
                       behind real time)
          infer_ms   - dequeue -> dorai_beamformer enhancement done (includes the
                       optional publish_raw tap + array concat, but
                       dominated by ONNX inference)
          publish_ms - enhancement done -> ROS publish call returns
                       (float->list serialization + DDS send)
          total_ms   - capture -> publish done (buffer+queue+infer+publish)
        """
        if self._delay_log_file is None:
            return
        capture_mono = self._t0 + cap0
        buffer_ms = (frame_ready_mono - capture_mono) * 1000.0
        queue_ms = (dequeue_mono - frame_ready_mono) * 1000.0
        infer_ms = (infer_done_mono - dequeue_mono) * 1000.0
        publish_ms = (publish_done_mono - infer_done_mono) * 1000.0
        total_ms = (publish_done_mono - capture_mono) * 1000.0

        capture_epoch = self._t0_wall + cap0
        frame_ready_epoch = self._t0_wall + (frame_ready_mono - self._t0)
        dequeue_epoch = self._t0_wall + (dequeue_mono - self._t0)
        clean_ready_epoch = self._t0_wall + (infer_done_mono - self._t0)
        publish_done_epoch = self._t0_wall + (publish_done_mono - self._t0)
        frame_s = n_samples / float(OUTPUT_RATE)

        try:
            self._delay_log_file.write(
                f"{seq},{capture_epoch:.3f},{frame_ready_epoch:.3f},"
                f"{dequeue_epoch:.3f},{clean_ready_epoch:.3f},"
                f"{publish_done_epoch:.3f},{buffer_ms:.1f},{queue_ms:.1f},"
                f"{infer_ms:.1f},{publish_ms:.1f},{total_ms:.1f},"
                f"{n_samples},{frame_s:.3f}\n"
            )
        except Exception as e:
            self.get_logger().error(
                f"delay log write failed: {e}", throttle_duration_sec=5.0)
        self.get_logger().info(
            f"delay seq={seq}: total={total_ms:.0f}ms "
            f"(buffer={buffer_ms:.0f} queue={queue_ms:.0f} "
            f"infer={infer_ms:.0f} publish={publish_ms:.0f})",
            throttle_duration_sec=1.0,
        )

    def _default_model_path(self):
        try:
            from ament_index_python.packages import get_package_share_directory
            p = os.path.join(get_package_share_directory("voice_mod"), "dorai_beamformer.ort")
            if os.path.exists(p):
                return p
        except Exception:
            pass
        local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dorai_beamformer.ort")
        return local if os.path.exists(local) else "voice_mod/dorai_beamformer.ort"

    def _make_topic_callback(self, ch):
        """The capture callback of a simulated device: same hand-over as the
        PortAudio callback (samples and their capture time into ch.push)."""
        state = {"seq": None, "warned": False}

        def on_msg(msg):
            parsed = parse_device_msg(msg.data)
            if parsed is None:
                return
            seq, rate, stamp, samples = parsed
            if rate != ch.sample_rate:
                if not state["warned"]:
                    state["warned"] = True
                    self.get_logger().error(
                        f"{ch.name}: blocks at {rate} Hz but input_rate is {ch.sample_rate} Hz; dropping them.")
                return
            if state["seq"] is not None and seq != state["seq"] + 1:
                ch.xruns += 1                      # a lost message is a lost capture block
            state["seq"] = seq
            ch.push(samples, stamp if stamp is not None else time.monotonic())
        return on_msg

    def _make_callback(self, ch):
        def callback(indata, frames, time_info, status):
            if status:
                ch.xruns += 1
                self.get_logger().warning(
                    f"{ch.label} xrun ({status}); total={ch.xruns}. If frequent, "
                    f"raise input_latency or lower num_threads.",
                    throttle_duration_sec=5.0
                )
            now = time.monotonic()
            try:
                age = time_info.currentTime - time_info.inputBufferAdcTime
                capture_mono = now - max(age, 0.0)
            except Exception:
                capture_mono = now
            # Collapse whatever we captured to one mono stream. For a genuine
            # mono device this is exactly the old indata[:, 0]; for a stereo
            # capture it averages the channels (identical channels -> full
            # level; signal on one channel -> still preserved).
            if indata.shape[1] > 1:
                mono = indata.mean(axis=1)
            else:
                mono = indata[:, 0]
            ch.push(mono.copy(), capture_mono)
        return callback

    def _run(self):
        next_tick = time.monotonic()
        while self._running:
            # Pace to a steady BLOCK_S grid without accumulating drift.
            next_tick += BLOCK_S
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()   # we fell behind; resync

            try:
                self._tick()
            except Exception as e:
                self.get_logger().error(
                    f"worker tick error: {e}", throttle_duration_sec=2.0
                )

    def _tick(self):
        for ch in self.mics:
            ch.resample_in()
        # Lock the independent USB clocks to each other: keep the capture time
        # at the head of every FIFO equal (from the first aligned block on).
        if self.drift_correction and self._primed:
            lock_channels(self.mics)

        # On first fill, trim all FIFOs to a common depth so the channels
        # start phase-aligned before enhancement begins.
        if not self._primed:
            depths = [ch.fifo_len() for ch in self.mics]
            if min(depths) >= max(TARGET_OUT, L_OUT):
                # The channels must start on the same capture instant. Keeping
                # the newest `common` samples of every FIFO does that only if
                # every FIFO ends at the same instant, and it does not when one
                # device's 100 ms callback has landed before this tick and
                # another's has not: that channel then ends one capture block
                # later and would stay 1600 samples off for the whole run. So
                # each channel keeps, beyond the common depth, the samples by
                # which its FIFO ends later than the earliest-ending one.
                extra = prime_offsets([ch.fifo_end_mono for ch in self.mics])
                common = min(d - e for d, e in zip(depths, extra))
                if common < L_OUT:                  # not enough behind the latest block yet
                    return
                for ch, e in zip(self.mics, extra):
                    ch.trim_fifo(common + e)
                self._primed = True
                self.get_logger().info(
                    "FIFOs primed and aligned; enhancing. Samples kept beyond the common "
                    f"depth per channel (later capture end): {extra}")
            else:
                return

        # Consume every aligned block above the alignment cushion, so the
        # output rate tracks the true capture rate even when the worker tick
        # jitters (an inference spike can let two input blocks pile up). Popping
        # the same count from every channel, gated by the slowest, keeps them
        # phase-aligned; the cushion absorbs brief input starvation.
        # Accumulate phase-aligned multichannel blocks at the true capture rate.
        # The worker stays light (resample + pop + append); the heavy dorai_beamformer pass
        # happens on the publisher thread so it never starves the mic buffers.
        min_fifo = min(ch.fifo_len() for ch in self.mics)
        n_blocks = max(0, (min_fifo - TARGET_OUT) // L_OUT)
        for _ in range(int(n_blocks)):
            block = np.stack([ch.pop_block() for ch in self.mics], axis=0)
            with self.mics[0].lock:
                cap = self.mics[0].last_capture_mono
            capture_s = (cap or time.monotonic()) - self._t0
            with self._acc_lock:
                self._mc_acc.append(block)
                self._mc_caps.append(capture_s)

        # Hand each complete publish_interval-second multichannel frame to the
        # publisher thread for whole-frame enhancement + publish.
        while True:
            with self._acc_lock:
                if len(self._mc_acc) < self._frame_blocks:
                    break
                blocks = self._mc_acc[:self._frame_blocks]
                cap0 = self._mc_caps[0]
                del self._mc_acc[:self._frame_blocks]
                del self._mc_caps[:self._frame_blocks]
            self._enqueue((blocks, cap0, time.monotonic()))

        self._maybe_diag(time.monotonic())

    def _enqueue(self, item):
        """Queue a finished frame for the publisher thread; drop oldest on a
        full queue so a slow consumer can never back-pressure the worker."""
        try:
            self._pubq.put_nowait(item)
        except queue.Full:
            try:
                self._pubq.get_nowait()
            except queue.Empty:
                pass
            try:
                self._pubq.put_nowait(item)
            except queue.Full:
                pass

    def _pub_loop(self):
        """Enhance (whole frame) + serialize + publish, off the worker thread."""
        while self._running:
            try:
                blocks, cap0, frame_ready_mono = self._pubq.get(timeout=0.2)
            except queue.Empty:
                continue
            dequeue_mono = time.monotonic()
            try:
                self._process_frame(blocks, cap0, frame_ready_mono, dequeue_mono)
            except Exception as e:
                self.get_logger().error(
                    f"publish error: {e}", throttle_duration_sec=2.0)

    def _process_frame(self, blocks, cap0, frame_ready_mono=None,
                        dequeue_mono=None):
        """Assemble [M, T], optionally publish the raw tap, enhance, publish.

        `frame_ready_mono`/`dequeue_mono` are only used for delay metering
        (stage breakdown); on the shutdown flush path there's no queueing, so
        they default to "now" at call time.
        """
        now = time.monotonic()
        if frame_ready_mono is None:
            frame_ready_mono = now
        if dequeue_mono is None:
            dequeue_mono = now

        mc = np.concatenate(blocks, axis=1)          # [M, T]
        if self.publish_raw:
            self._emit_raw(mc, cap0)
        # Continuous overlap-add: push this frame's audio into the enhancer and
        # publish whatever finished clean samples come back. Output is seamless
        # across frames (the enhancer holds up to one window of overlap state),
        # so no boundary knock. push() may return slightly fewer samples than
        # the frame length (the held tail is emitted on the next frame / flush)
        # and, on the very first frame, could be empty if the frame is shorter
        # than one window.
        clean = self.enhancer.push(mc)
        infer_done_mono = time.monotonic()
        seq = self._seq
        if len(clean):
            self._emit(clean, cap0)
        publish_done_mono = time.monotonic()
        if self._delay_log_file is not None and len(clean):
            self._log_delay(
                cap0, seq, len(clean),
                frame_ready_mono, dequeue_mono, infer_done_mono, publish_done_mono,
            )

    def _flush(self):
        """Publish any remaining partial frame, then drain the enhancer's
        overlap-add tail so the last ~window of speech isn't lost (called on
        shutdown)."""
        with self._acc_lock:
            blocks = self._mc_acc
            cap0 = self._mc_caps[0] if self._mc_caps else 0.0
            self._mc_acc = []
            self._mc_caps = []
        if blocks:
            self._process_frame(blocks, cap0)
        try:
            tail = self.enhancer.flush()
            if len(tail):
                self._emit(tail, cap0)
        except Exception as e:
            self.get_logger().error(f"enhancer flush failed: {e}")

    def _emit_raw(self, mc, capture_s):
        """Publish one frame of pre-beamformer multichannel audio (debug tap).
        `mc` is a [M, T] array; channels are interleaved in `data[4:]` exactly
        like the clean topic so the same reader works."""
        if self.raw_pub is None:
            return
        m = mc.shape[0]
        interleaved = mc.T.reshape(-1).astype(np.float32)   # T*M, ch-interleaved
        msg = Float32MultiArray()
        header = np.array(
            [capture_s, float(OUTPUT_RATE), float(m), float(self._raw_seq)],
            dtype=np.float32,
        )
        msg.data = np.concatenate((header, interleaved)).tolist()
        msg.layout.data_offset = 4
        per_ch = mc.shape[1]
        msg.layout.dim = [
            MultiArrayDimension(label="channels", size=m, stride=len(interleaved)),
            MultiArrayDimension(label="samples", size=per_ch, stride=per_ch),
        ]
        self.raw_pub.publish(msg)
        sync = Float32MultiArray()
        sync.data = sync_status(self.mics, self.drift_correction, self._raw_seq)
        self.sync_pub.publish(sync)
        self.get_logger().info(
            f"Published RAW frame seq={self._raw_seq}: {m}ch x {per_ch} samples "
            f"({per_ch / OUTPUT_RATE:.2f}s)"
        )
        self._raw_seq += 1

    def _emit(self, clean, capture_s):
        # Level is already set by the enhancer's slow gated AGC and hard-limited
        # to +/-0.99, so there is deliberately no per-frame gain here — a second
        # per-frame normalisation is exactly what caused the old level pumping.
        clean = np.ascontiguousarray(np.asarray(clean, dtype=np.float32))
        msg = Float32MultiArray()
        header = np.array(
            [capture_s, float(OUTPUT_RATE), 1.0, float(self._seq)],
            dtype=np.float32,
        )
        msg.data = np.concatenate((header, clean)).tolist()
        msg.layout.data_offset = 4
        msg.layout.dim = [
            MultiArrayDimension(label="channels", size=1, stride=len(clean)),
            MultiArrayDimension(label="samples", size=len(clean), stride=len(clean)),
        ]
        self.pub.publish(msg)
        self.get_logger().info(
            f"Published frame seq={self._seq}: {len(clean)} samples "
            f"({len(clean) / OUTPUT_RATE:.2f}s)"
        )
        self._seq += 1

    def _maybe_diag(self, now):
        if now - self._last_diag < DIAG_PERIOD_S:
            return
        self._last_diag = now
        parts = []
        for ch in self.mics:
            with ch.lock:
                inb = len(ch.in_buf)
            parts.append(
                f"{ch.label}:{ch.sample_rate}Hz fifo={ch.fifo_len():5d} "
                f"in={inb:5d} ur={ch.underruns} xr={ch.xruns} "
                f"ovf={ch.in_overflows + ch.out_overflows} "
                f"trim={(ch.ratio_scale - 1.0) * 1e6:+.0f}ppm "
                f"err={ch.drift_err * 1e3:+.2f}ms"
            )
        with self._acc_lock:
            acc = len(self._mc_acc) * L_OUT
        text = " | ".join(parts) + f" || acc={acc} samples"
        self.get_logger().info(text)
        self.diag_pub.publish(String(data=text))

    def shutdown(self):
        self._running = False
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        for ch in self.mics:
            if getattr(ch, "stream", None) is not None:
                try:
                    ch.stream.stop()
                    ch.stream.close()
                except Exception:
                    pass
        # Join the publisher thread BEFORE flushing. The publisher thread calls
        # self.enhancer.push() and _flush() calls self.enhancer.flush(); the
        # ContinuousEnhancer holds multi-step state (inbuf/acc/accw/g) and is
        # NOT thread-safe. Draining it here while the publisher is still mid-
        # frame would be two threads mutating it at once. Joining first makes
        # the main thread the sole enhancer owner during the tail drain. Any
        # frame still queued-but-unprocessed is dropped, which is harmless at
        # shutdown (the worker has already stopped feeding the queue, so it is
        # normally empty); the flush still emits the enhancer's held overlap
        # tail for everything the publisher did process.
        if getattr(self, "_pub_thread", None) is not None \
                and self._pub_thread.is_alive():
            self._pub_thread.join(timeout=2.0)
        # flush any tail audio so the last words aren't lost
        try:
            self._flush()
        except Exception:
            pass
        if getattr(self, "_delay_log_file", None) is not None:
            try:
                self._delay_log_file.close()
            except Exception:
                pass


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def main(args=None):
    # Handle SIGINT/SIGTERM ourselves: rclpy's own handlers invalidate the
    # context before `finally` runs, so the tail flush in node.shutdown() could
    # never be published ("publisher's context is invalid").
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    node = VoiceMod()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
