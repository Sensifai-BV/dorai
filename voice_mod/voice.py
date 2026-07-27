#!/usr/bin/env python3
"""
voice_mod / voice.py  —  dorai real-time mic-array front-end.

A single ROS 2 node that runs the full capture-to-enhancement path:

    mic array (N mics, USB/other)
        -> per-mic clock-drift-corrected async resample to 16 kHz
        -> dorai_beamformer.ort multichannel beamformer (continuous overlap-add)
        -> single slow AGC for a steady STT level
        -> clean single-channel speech
        -> published on /dorai_clean_audio in N-second frames (default 10 s)

Enhancement runs continuously via overlap-add: a `enh_window`-second analysis
window slides by `enh_hop` seconds across the *continuous* stream, each window
is enhanced with full context and Hann-overlap-added, so the reconstruction is
seamless (bit-identical no matter how the input is chunked). Publishing is
batched into `publish_interval`-second frames only so stt_mod receives larger,
cheaper chunks — it introduces no boundary artifact. Added algorithmic latency
is one analysis window (`enh_window`, default 2 s).

Why this shape (see ContinuousEnhancer): the model is a whole-signal enhancer
that (a) peak-normalises its output regardless of input scale and (b) leaves
uncompensated samples at each window edge. The previous path enhanced disjoint
publish frames and concatenated them, injecting a step ('knock') at every join,
and set level by grabbing each frame's peak — which, given (a), only pumped
loudness and amplified noise between frames. Overlap-add fixes the knock; the
slow gated AGC fixes the level.

Design properties:
  * Audio callbacks only append to bounded, per-channel locked ring buffers,
    keeping the PortAudio thread responsive.
  * A single dedicated worker thread does resample + inference + publish, so
    the ROS executor is never blocked by ONNX.
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

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

import sounddevice as sd


# ---------------------------------------------------------------------------
# Constants — the topic contract (must match stt_mod) and DSP framing.
# ---------------------------------------------------------------------------
OUTPUT_RATE = 16000               # master timeline rate (Hz)
BLOCK_MS = 100                    # worker tick period (ms)
BLOCK_S = BLOCK_MS / 1000.0
L_OUT = OUTPUT_RATE * BLOCK_MS // 1000          # output samples/block/ch (1600)

# dorai_beamformer.ort is a whole-signal model driven by a continuous
# overlap-add sliding window (see ContinuousEnhancer). WINDOW_SIZE is only the
# dummy length used to warm the ONNX session up before real audio arrives.
WINDOW_SIZE = 3200

# Asynchronous-resampler drift control.
TARGET_BLOCKS = 2                 # steady-state out-FIFO depth (blocks)
TARGET_OUT = TARGET_BLOCKS * L_OUT
RATIO_TRIM_KP = 0.05              # proportional gain of the drift loop
RATIO_TRIM_MAX = 0.02            # clamp |ratio correction| to +/-2 %

# Buffer bounds (overflow protection). Normal FIFO depth sits at TARGET_OUT;
# this ceiling is only a safety net that absorbs bursts of stalled worker
# ticks before dropping oldest audio to keep latency bounded.
MAX_IN_SECONDS = 2.0              # max input backlog per mic (s)
MAX_OUT_BLOCKS = 16              # max resampled FIFO depth (blocks, ~1.6 s)

DIAG_PERIOD_S = 5.0


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

    def __init__(self, label, device, sample_rate, name="", hpf_hz=150.0):
        self.label = label
        self.device = device
        self.sample_rate = int(sample_rate)
        self.name = name

        self.lock = threading.Lock()
        self.in_buf = np.zeros(0, dtype=np.float32)     # raw capture samples
        self.out_fifo = np.zeros(0, dtype=np.float32)   # resampled 16 kHz
        self.last_capture_mono = None

        self.resampler = Resampler(self.sample_rate, OUTPUT_RATE)
        self.ratio_scale = 1.0
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

    def resample_in(self):
        """Resample everything captured since the last tick into the 16 kHz
        FIFO, high-pass to strip sub-band rumble, bounding it so latency can
        never run away."""
        with self.lock:
            x = self.in_buf
            self.in_buf = np.zeros(0, dtype=np.float32)

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

    def update_ratio(self, ref_len):
        """Trim the resample ratio to pull this channel's FIFO toward the
        cross-channel reference, locking the independent USB clocks to each
        other. Absolute FIFO depth is regulated by the consumer, not here, so
        a uniform offset shared by all channels leaves the ratio at 1.0."""
        if not self.resampler.variable:
            return
        err = (len(self.out_fifo) - ref_len) / float(TARGET_OUT)
        corr = max(-RATIO_TRIM_MAX, min(RATIO_TRIM_MAX, -RATIO_TRIM_KP * err))
        self.ratio_scale = 1.0 + corr

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


def choose_rate(device: int, requested_rate=None) -> int:
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
                device=device, channels=1, dtype="float32", samplerate=rate
            )
            return int(rate)
        except Exception:
            continue
    return default_rate


def detect_microphones(logger, include_internal=False, max_mics=3,
                       requested_rate=None, prefer_usb=True):
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

    if prefer_usb:
        # Stable order: USB-looking devices first, then by device index.
        usable.sort(key=lambda it: (not looks_usb(it[1]["name"]), it[0]))

    resolved = []
    for idx, dev in usable:
        if len(resolved) >= max_mics:
            break
        rate = choose_rate(idx, requested_rate)
        label = f"mic{len(resolved)}"
        logger.info(
            f"Auto-detected {label}: dev #{idx} '{dev['name']}' @ {rate} Hz"
        )
        resolved.append({
            "label": label, "device": idx,
            "sample_rate": rate, "name": dev["name"],
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
        out = np.empty(len(ready), dtype=np.float32)
        for s in range(0, len(ready), self.hop):
            seg = ready[s:s + self.hop]
            rms = float(np.sqrt((seg ** 2).mean() + 1e-12))
            g_inst = min(self.tgt / rms, self.max_gain) if rms > self.gate else self.g
            self.g = self.alpha * self.g + (1.0 - self.alpha) * g_inst
            out[s:s + len(seg)] = np.clip(seg * self.g, -self.ceil, self.ceil)
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

        self.declare_parameter("include_internal", False)
        self.declare_parameter("max_mics", 3)
        self.declare_parameter("requested_rate", 48000)
        self.declare_parameter("prefer_usb", True)
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
        self.publish_interval = float(gp("publish_interval").value)
        self.output_topic = gp("output_topic").value
        num_threads = int(gp("num_threads").value)
        in_lat = float(gp("input_latency").value)
        self.input_latency = in_lat if in_lat > 0 else "high"
        self.publish_raw = bool(gp("publish_raw").value)
        self.raw_topic = gp("raw_topic").value
        self.hpf_hz = float(gp("hpf_hz").value)
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

        resolved = detect_microphones(
            self.get_logger(), self.include_internal, self.max_mics,
            self.requested_rate, self.prefer_usb,
        )
        self.mics = [
            MicChannel(m["label"], m["device"], m["sample_rate"], m["name"],
                       hpf_hz=self.hpf_hz)
            for m in resolved
        ]
        self.num_channels = len(self.mics)

        self.beam = Beamformer(model_path, num_threads, self.get_logger())
        self.beam.warmup(self.num_channels)
        self.get_logger().info(
            f"dorai_beamformer.ort ready ({self.num_channels}ch) via {self.beam.in_name}"
            f"->{self.beam.out_name}"
        )

        # Continuous overlap-add enhancer wrapping the per-window model call.
        # Driven only from the publisher thread (_pub_loop / _flush), so it
        # needs no lock. enh_hop is clamped to <= enh_window.
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

        for ch in self.mics:
            blocksize = int(ch.sample_rate * BLOCK_S)
            ch.stream = sd.InputStream(
                samplerate=ch.sample_rate, device=ch.device, channels=1,
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
            ch.push(indata[:, 0].copy(), capture_mono)
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
        # Lock the independent USB clocks to each other by pulling every
        # channel's FIFO toward the cross-channel mean.
        if len(self.mics) > 1:
            ref = sum(ch.fifo_len() for ch in self.mics) / len(self.mics)
            for ch in self.mics:
                ch.update_ratio(ref)

        # On first fill, trim all FIFOs to a common depth so the channels
        # start phase-aligned before enhancement begins.
        if not self._primed:
            depths = [ch.fifo_len() for ch in self.mics]
            if min(depths) >= max(TARGET_OUT, L_OUT):
                common = min(depths)
                for ch in self.mics:
                    ch.trim_fifo(common)
                self._primed = True
                self.get_logger().info("FIFOs primed and aligned; enhancing.")
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
                f"rs={ch.ratio_scale:.4f}"
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
        # flush any tail audio so the last words aren't lost
        try:
            self._flush()
        except Exception:
            pass
        if getattr(self, "_pub_thread", None) is not None \
                and self._pub_thread.is_alive():
            self._pub_thread.join(timeout=2.0)
        if getattr(self, "_delay_log_file", None) is not None:
            try:
                self._delay_log_file.close()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = VoiceMod()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
