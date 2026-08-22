#!/usr/bin/env python3
"""
doa_mod / doa.py  —  dorai direction-of-arrival estimator.

A single ROS 2 node that turns raw microphone-array audio into a talker
bearing:

    mic array (M mics, one clock)
        -> ring buffer -> 2048-sample analysis block every 512-sample hop
        -> VAD gate (neural | energy | off)
        -> pooled SRP-PHAT over speech frames  [doa_core.SRPAccumulator]
        -> bearing in degrees, published on /dorai_doa

Published topics
----------------
  /dorai_doa         std_msgs/Float32          azimuth, degrees, 0-360
  /dorai_doa/state   std_msgs/Float32MultiArray  diagnostics, see _emit()
  /doa_mod/diagnostics  std_msgs/String        periodic health line

THE ESTIMATOR IS NOT NEW CODE. doa_core.py is vendored byte-identical from the
validated doa/ research tree, where it scores 6.1 deg mean absolute error (21
deg max) over the eight marked talker positions of the 2026-08-12 hardware
session, with raw bearings and no fitted rotation. Do not "improve" it here:
test/test_doa_mod.py checksums it against doa/ and fails if the copies fork.
Fix it there, re-run doa/score_session.py, then re-copy.

WHY POOLED SRP AND NOT PER-FRAME AVERAGING
------------------------------------------
The bearing comes from accumulating RAW steered-response power across frames
and taking the argmax ONCE, at the end -- not from averaging a per-frame
bearing. A reverberation-dominated frame produces a wild per-frame angle that
drags a circular mean and cannot be averaged back out; pooled power lets that
same frame contribute only a low broad ridge. The per-frame least-squares
estimate is still computed, but purely as a quality diagnostic (closure gate).

GEOMETRY IS THE THING THAT WILL BITE YOU
----------------------------------------
The default is the ReSpeaker 4-Mic Array: ch1..ch4 at 45/135/225/315 deg,
counter-clockwise, 32 mm radius. A MIRRORED channel map is not a rotation and
CANNOT be calibrated away with an offset -- it silently reports plausible
bearings that are wrong everywhere except on the mirror axis. That fault
shipped undetected for months. If bearings look right at 90 deg and 270 deg
but wrong at 0 deg and 180 deg, you have a mirror, not an offset. See
doa_core.array_positions() and README.md.

Auto-calibration is deliberately NOT exposed here. The research tree's
MDS-based `--calibrate` path recovers geometry up to rotation AND reflection,
never resolves the reflection, and was measured at 87 deg mean error on this
same session. Set the geometry explicitly instead.
"""

import math
import os
import sys
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doa_core as dc                                              # noqa: E402


# ---------------------------------------------------------------------------
# Framing constants. These match the validated offline harness
# (doa/score_session.py uses FRAME=1024/HOP=512; live_doa.py uses 2048/512).
# BLOCK is the GCC-PHAT analysis length -- long enough for a stable peak; HOP
# is how often a new estimate is produced.
# ---------------------------------------------------------------------------
FS = 16000
BLOCK = 2048            # 128 ms analysis block
HOP = 512               # 32 ms estimator cadence == the VAD's native chunk


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
class RingBuffer:
    """Lock-protected multichannel ring buffer, written by the audio callback.

    Write-INDEX ring, deliberately. The research tree's version called
    np.roll() on the whole buffer for every 512-sample callback, which copies
    the entire capacity (4 ch x 30 s = 1.92 M samples) each time and allocates
    a fresh array inside a realtime audio callback. Measured at 0.50 ms per
    callback against a 32 ms budget -- tolerable on a laptop, but the cost
    scales with CAPACITY rather than block size, so it degrades as soon as
    anyone adds channels or lengthens the buffer, and allocation in an audio
    callback is the thing you avoid on a Pi.

    This version writes in place at a moving index: cost is proportional to the
    block, and nothing is allocated after construction.
    """

    def __init__(self, n_ch, capacity):
        self.buf = np.zeros((int(n_ch), int(capacity)), dtype=np.float64)
        self.capacity = int(capacity)
        self.widx = 0             # next write position
        self.written = 0          # total samples ever written
        self.lock = threading.Lock()

    def push(self, block):
        """Append [M, n] samples. Oldest are overwritten once full."""
        block = np.asarray(block, dtype=np.float64)
        n = block.shape[1]
        with self.lock:
            if n >= self.capacity:
                # A block bigger than the ring: keep only its tail.
                self.buf[:] = block[:, -self.capacity:]
                self.widx = 0
                self.written += n
                return
            end = self.widx + n
            if end <= self.capacity:
                self.buf[:, self.widx:end] = block
            else:
                split = self.capacity - self.widx
                self.buf[:, self.widx:] = block[:, :split]
                self.buf[:, :end - self.capacity] = block[:, split:]
            self.widx = end % self.capacity
            self.written += n

    def latest(self, n):
        """Most recent n samples in chronological order, or None if unfilled."""
        n = int(n)
        with self.lock:
            if self.written < n or n > self.capacity:
                return None
            start = (self.widx - n) % self.capacity
            if start + n <= self.capacity:
                return self.buf[:, start:start + n].copy()
            split = self.capacity - start
            out = np.empty((self.buf.shape[0], n), dtype=np.float64)
            out[:, :split] = self.buf[:, start:]
            out[:, split:] = self.buf[:, :n - split]
            return out


class MicSource:
    """Direct capture from ONE multichannel device (the ReSpeaker array).

    ONE DEVICE, ONE CLOCK -- this is not a stylistic choice. TDOA measures
    inter-channel delays of tens of microseconds; N separate USB microphones
    are N independent crystals, and 1 ppm of relative clock error is 1 us of
    drift per second, which walks straight through the aperture's entire 187 us
    delay range in a couple of minutes. A single multichannel device samples
    every capsule off one crystal, so the only inter-channel delay is acoustic.

    This is also why doa_mod does NOT reuse voice_mod's capture layer: that one
    is built for separate USB lavaliers and per-mic asynchronous resampling,
    which is right for beamforming and wrong for TDOA.
    """

    def __init__(self, device=None, channels=(1, 2, 3, 4), n_dev_ch=6, fs=FS,
                 capacity_s=8.0, logger=None):
        import sounddevice as sd
        self.sd = sd
        self._log = logger
        info = sd.query_devices(device, 'input')
        avail = int(info['max_input_channels'])
        if fs is None or fs <= 0:
            fs = int(info['default_samplerate'])
        if n_dev_ch is None or n_dev_ch <= 0:
            n_dev_ch = avail
            channels = tuple(range(min(len(channels), avail)))
        if n_dev_ch > avail:
            # Keep the research tree's diagnostic here verbatim in spirit: this
            # is the error everyone hits first, and PortAudio's own message
            # ("Invalid number of channels") explains nothing.
            raise ValueError(
                f"device '{info['name']}' exposes {avail} input channel(s) but "
                f"{n_dev_ch} were requested -- this is what PortAudio reports "
                f"as 'Invalid number of channels'.\n"
                f"The defaults describe the ReSpeaker 4-Mic Array (6 ch: ch0 is "
                f"the board's own processed mix, ch1..ch4 the raw capsules, ch5 "
                f"playback). For a mono USB mic they cannot work.\n"
                f"DoA needs >=3 non-collinear mics on ONE clock. On macOS build "
                f"an Aggregate Device (Audio MIDI Setup -> + -> Create Aggregate "
                f"Device), tick each mic, ENABLE DRIFT CORRECTION on all but the "
                f"master, then set -p device_channels:=0 and give the geometry "
                f"with -p positions_mm.")
        if max(channels) >= avail:
            raise ValueError(
                f"channel {max(channels)} requested but the device exposes "
                f"{avail} ({0}..{avail - 1}). Set -p channels within that range.")
        self.sel = list(channels)
        self.fs = int(fs)
        self.n_ch = len(self.sel)
        self.ring = RingBuffer(self.n_ch, int(self.fs * capacity_s))
        self.xruns = 0
        self.stream = sd.InputStream(
            device=device, channels=n_dev_ch, samplerate=self.fs,
            dtype='float32', blocksize=HOP, callback=self._cb)
        self.name = info['name']

    def _cb(self, indata, frames, time_info, status):
        if status:
            self.xruns += 1
        self.ring.push(np.asarray(indata, dtype=np.float64).T[self.sel])

    def start(self):
        self.stream.start()

    def stop(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    def read(self, n):
        return self.ring.latest(n)


def parse_positions_mm(spec, n_ch):
    """Explicit geometry from a flat 'x1,y1,x2,y2,...' string, in millimetres.

    Returns [M,2] in METRES, or None when the string is empty.
    """
    spec = str(spec or '').replace(',', ' ').split()
    if not spec:
        return None
    vals = [float(v) for v in spec]
    if len(vals) % 2:
        raise ValueError(
            f"positions_mm needs an even count (x y per mic); got {len(vals)}")
    pos = np.array(vals, dtype=np.float64).reshape(-1, 2) / 1000.0
    if pos.shape[0] != n_ch:
        raise ValueError(
            f"positions_mm gives {pos.shape[0]} mic(s) but {n_ch} channel(s) "
            f"are selected -- the geometry must match the channel order "
            f"exactly, or every bearing is wrong.")
    if pos.shape[0] < 3:
        raise ValueError(
            "DoA needs >= 3 non-collinear mics: with 2 the normal equations "
            "are rank-1 and the solution collapses onto the baseline axis.")
    if np.linalg.matrix_rank(dc.pair_baselines(pos, dc.pairs_for(pos.shape[0])),
                             tol=1e-9) < 2:
        raise ValueError(
            "positions_mm describes a COLLINEAR array: a straight line cannot "
            "resolve which side of its axis a source is on. Offset one mic "
            "perpendicular to the line.")
    return pos


class SpeechGate:
    """Decides which frames are allowed to move the bearing. Three modes.

        'neural'  vad.onnx, streaming, one 512-sample chunk per hop.
        'energy'  adaptive rms percentile over a rolling history.
        'off'     every frame is active. No gating at all.

    WHY THE MODE IS EXPLICIT, not a silent fallback chain
    -----------------------------------------------------
    The research tree tries the neural VAD and quietly drops to energy gating
    if the model fails to load. That means a missing vad.onnx changes the
    estimator's behaviour with nothing in the logs to say so. Here the mode is
    a parameter: if you ask for 'neural' and it cannot load, that is an error,
    not a downgrade.

    WHY 'off' IS A SUPPORTED MODE, not a debug hack
    -----------------------------------------------
    Pooled SRP is markedly robust to admitting bad frames -- the audit measured
    6.25 deg vs 6.12 deg when a third of frames were mis-gated, because a
    noise frame contributes a low broad ridge rather than a confident wrong
    angle. So 'off' is a legitimate configuration:

      * it removes the onnxruntime dependency entirely;
      * it is the right choice for a continuously-speaking source, or when an
        upstream stage has already segmented speech;
      * it is the correct control when you suspect the VAD is the problem.

    What you lose: in a room with a stationary interferer, non-speech frames
    carry the interferer's direction and pool into the answer. With gating on,
    those frames go to the separate noise pool instead, and subtract_noise can
    remove them. Do not use 'off' with subtract_noise -- there is nothing in
    the noise pool to subtract.

    THE CALLING CONVENTION IS THE POINT (neural mode)
    -------------------------------------------------
    The model is a STREAMING one: 512 new samples per call, a 64-sample
    context prefix, and an LSTM state carried across calls. It must therefore
    be fed each hop's NEW samples exactly once, in order.

    The research tree instead passed the whole 2048-sample analysis block every
    hop. With 75% block overlap that re-ran the model 4x per hop AND fed every
    chunk to the LSTM four times at four different state positions, driving the
    recurrent state with a signal repeating every sample four times over, never
    reset. Measured consequences: 35% of frames disagreed with correct
    streaming, and time to re-acquire a moved talker to +-15 deg went from
    2.73 s to 5.07 s. Accuracy was barely affected (see above) but TRACKING was
    halved. Feeding block[:, -HOP:] costs 4x less and recovers 2.3 s.
    """

    def __init__(self, mode='neural', threshold=0.45, percentile=70.0,
                 model_path=None, history=400):
        self.mode = str(mode).lower().strip()
        if self.mode not in ('neural', 'energy', 'off'):
            raise ValueError(
                f"vad_mode must be 'neural', 'energy' or 'off', got "
                f"{mode!r}")
        self.threshold = float(threshold)
        self.percentile = float(percentile)
        self._vad = None
        self.last_prob = float('nan')
        self._rms_hist = []
        self._history = int(history)
        if self.mode == 'neural':
            from vad import NeuralVAD
            self._vad = NeuralVAD(model_path)

    def __call__(self, block):
        """True if this hop's new samples are speech.

        `block` is the full [M, BLOCK] analysis frame; neural mode consumes
        only its newest HOP samples.
        """
        if self.mode == 'off':
            self.last_prob = 1.0
            return True
        if self.mode == 'neural':
            # Only the newest HOP samples -- see the class docstring.
            chunk = block[0, -HOP:]
            self.last_prob = self._vad.process_chunk(chunk)
            return self.last_prob >= self.threshold
        rms = dc.frame_rms(block)
        self._rms_hist.append(rms)
        if len(self._rms_hist) > self._history:
            del self._rms_hist[0]
        self.last_prob = rms
        if len(self._rms_hist) < 20:
            return False
        return rms > float(np.percentile(self._rms_hist, self.percentile))

    def reset(self):
        if self._vad is not None:
            self._vad.reset()
        self._rms_hist.clear()


class BearingEstimator:
    """Frame-by-frame DoA state: VAD gate -> pooled SRP -> bearing.

    Deliberately free of ROS and of any audio device, so the exact code the
    node runs can be driven offline over recorded takes by
    test/replay_session.py and scored against ground truth. The node owns
    threading, parameters and publishing; this owns the estimate.
    """

    def __init__(self, pos, fs=FS, gate=None, avg_tc=0.5, subtract_noise=False,
                 f_hi=None):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.fs = int(fs)
        self.gate = gate if gate is not None else SpeechGate('off')
        # Never let the band exceed what the geometry can support: a wider
        # array aliases at a LOWER frequency, and above that the pooled peak
        # can settle on a grating lobe. doa_core asserts this too; computing it
        # here means a wide array works rather than raising.
        self.f_hi = float(min(dc.F_HI, dc.aliasing_limit_hz(self.pos))
                          if f_hi is None else f_hi)
        self.aperture = max(np.linalg.norm(self.pos[i] - self.pos[j])
                            for i, j in dc.pairs_for(self.pos.shape[0]))
        # decay=avg_tc makes this an exponential-forget pool: it follows a
        # moving talker instead of averaging a whole session. The research
        # tree's live display defaulted to 2.0 s; the accuracy review found
        # 0.5 s saves ~1.1 s of tracking latency at negligible accuracy cost,
        # which is the right trade for a robot that must turn toward a speaker.
        self.acc = dc.SRPAccumulator(
            self.pos, fs=self.fs, f_hi=self.f_hi, decay=avg_tc, hop=HOP,
            subtract_noise=bool(subtract_noise))
        self.n_frames = 0
        self.n_speech = 0
        self.n_consistent = 0
        self.last_active = False
        self.last_est = None

    def push(self, block):
        """Consume one [M, BLOCK] frame. Returns a per-frame diagnostic dict."""
        block = np.asarray(block, dtype=np.float64)
        active = bool(self.gate(block))
        self.n_frames += 1

        # Per-frame least squares is a QUALITY diagnostic, not the answer. Its
        # closure error is geometry-free: for any single source the delays
        # round every triangle must sum to zero, whatever the array looks
        # like, so a large value means the audio carries no single direction
        # -- as opposed to the plane-wave residual, which is equally large for
        # a bad frame and for a wrong geometry and cannot distinguish them.
        est = dc.doa_least_squares(block, pos=self.pos, fs=self.fs,
                                   f_hi=self.f_hi)
        consistent = bool(est['consistent'])

        # Feed BOTH speech and non-speech frames: the accumulator routes them
        # to separate pools, so a stationary interferer can be subtracted.
        # Only active frames move the reported bearing.
        self.acc.update(block, active=active)
        if active:
            self.n_speech += 1
            if consistent:
                self.n_consistent += 1
        self.last_active = active
        self.last_est = est
        return dict(active=active, consistent=consistent,
                    az_frame=est['az_deg'] % 360.0,
                    closure_us=est['closure_us'],
                    band_rms=est['band_rms'],
                    vad_prob=self.gate.last_prob)

    def bearing(self):
        """(azimuth_deg in [0,360), sharpness) from the pooled spectrum.

        sharpness is 1 - mean/peak of the normalized spectrum: ~0 when the
        spectrum is flat (no usable direction), higher when one lobe
        dominates. It is NOT an angular confidence -- with a 64 mm aperture
        the main lobe is ~130 deg wide however good the frame is.
        """
        if self.acc.n_speech == 0:
            return float('nan'), 0.0
        az, sharp = self.acc.bearing()
        return (az % 360.0 if np.isfinite(az) else float('nan')), sharp

    @property
    def consistent_frac(self):
        return (self.n_consistent / self.n_speech) if self.n_speech else 0.0


class TopicSource:
    """Multichannel audio arriving on a ROS topic (voice_mod's /dorai_raw_audio).

    For when voice_mod already owns the device -- two processes cannot open the
    same ALSA capture stream, so this is how doa_mod coexists with it.

    CAVEAT, and it is not a small one: voice_mod's raw tap is the pre-beamformer
    but POST-resample signal. When it is driving separate USB microphones, each
    channel has been through its own asynchronous resampler with a closed-loop
    drift trim -- a per-channel, time-varying delay of exactly the kind TDOA
    reads as a bearing change. That path is fine when voice_mod is fed a single
    multichannel device (one clock, ratio stays 1.0), and unreliable when it is
    fed N independent USB mics. Prefer source='mic' whenever the array can be
    opened directly.
    """

    def __init__(self, node, topic, n_ch_expected=4, fs=FS, capacity_s=8.0):
        self.fs = int(fs)
        self.n_ch = int(n_ch_expected)
        self.ring = RingBuffer(self.n_ch, int(self.fs * capacity_s))
        self.name = topic
        self.n_msgs = 0
        self.n_dropped = 0
        self._warned_shape = False
        self._node = node
        self._sub = node.create_subscription(
            Float32MultiArray, topic, self._on_msg, 10)

    def _on_msg(self, msg):
        data = np.asarray(msg.data, dtype=np.float64)
        if data.size < 4:
            self.n_dropped += 1
            return
        # Repo topic contract: [capture_t, rate, n_channels, seq, interleaved...]
        rate = float(data[1])
        m = int(round(float(data[2])))
        body = data[4:]
        if m <= 0 or body.size % m:
            self.n_dropped += 1
            return
        if m != self.n_ch or int(rate) != self.fs:
            if not self._warned_shape:
                self._node.get_logger().error(
                    f"{self.name}: expected {self.n_ch}ch @ {self.fs} Hz but "
                    f"received {m}ch @ {rate:.0f} Hz. Set -p channels / -p fs "
                    f"to match the publisher, or the geometry will not "
                    f"correspond to the channel order. Dropping these frames.")
                self._warned_shape = True
            self.n_dropped += 1
            return
        self.ring.push(body.reshape(-1, m).T)
        self.n_msgs += 1

    def start(self):
        pass

    def stop(self):
        pass

    def read(self, n):
        return self.ring.latest(n)


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------
class DoaMod(Node):
    def __init__(self):
        super().__init__("doa_mod")

        # Float params use dynamic typing so both `p:=2` and `p:=2.0` are
        # accepted on the command line, matching voice_mod.
        from rcl_interfaces.msg import ParameterDescriptor
        dyn = ParameterDescriptor(dynamic_typing=True)

        # -- source ---------------------------------------------------------
        self.declare_parameter("source", "mic")          # mic | topic
        self.declare_parameter("device", -1)             # -1 = default input
        self.declare_parameter("device_channels", 6)     # ReSpeaker exposes 6
        self.declare_parameter("channels", [1, 2, 3, 4])  # raw capsules
        self.declare_parameter("fs", FS)
        self.declare_parameter("input_topic", "/dorai_raw_audio")

        # -- geometry -------------------------------------------------------
        # radius_mm builds the validated ReSpeaker layout. positions_mm
        # overrides it with an explicit "x1,y1,x2,y2,..." in the SAME order as
        # `channels`. There is no auto-calibration on purpose -- see below.
        self.declare_parameter("radius_mm", 32.0, dyn)
        self.declare_parameter("positions_mm", "")

        # -- VAD ------------------------------------------------------------
        # 'neural' vad.onnx | 'energy' rms percentile | 'off' no gating.
        self.declare_parameter("vad_mode", "neural")
        self.declare_parameter("vad_threshold", 0.45, dyn)
        self.declare_parameter("vad_percentile", 70.0, dyn)
        self.declare_parameter("vad_model_path", "")
        # How long after the last speech frame bearings keep being published.
        self.declare_parameter("vad_hangover_ms", 800.0, dyn)

        # -- estimator ------------------------------------------------------
        self.declare_parameter("avg_tc", 0.5, dyn)
        self.declare_parameter("subtract_noise", False)

        # -- output ---------------------------------------------------------
        self.declare_parameter("publish_period", 0.2, dyn)
        self.declare_parameter("output_topic", "/dorai_doa")
        self.declare_parameter("state_topic", "/dorai_doa/state")
        # A plain-language rendering of the same state, for `ros2 topic echo`.
        # On by default: it is one short string per publish and it is what
        # makes the node debuggable from a terminal. Set publish_human:=false
        # to drop it on a bandwidth-constrained link.
        self.declare_parameter("publish_human", True)
        self.declare_parameter("human_topic", "/dorai_doa/human")
        self.declare_parameter("diag_period", 5.0, dyn)

        gp = self.get_parameter
        self.source_kind = str(gp("source").value).lower().strip()
        device = int(gp("device").value)
        self.device = None if device < 0 else device
        self.device_channels = int(gp("device_channels").value)
        self.channels = [int(c) for c in gp("channels").value]
        self.fs = int(gp("fs").value)
        self.input_topic = str(gp("input_topic").value)
        radius_mm = float(gp("radius_mm").value)
        positions_mm = str(gp("positions_mm").value or "")
        self.vad_mode = str(gp("vad_mode").value).lower().strip()
        self.vad_threshold = float(gp("vad_threshold").value)
        self.vad_percentile = float(gp("vad_percentile").value)
        vad_model_path = str(gp("vad_model_path").value or "") or None
        self.hangover_s = float(gp("vad_hangover_ms").value) / 1000.0
        avg_tc = float(gp("avg_tc").value)
        self.subtract_noise = bool(gp("subtract_noise").value)
        self.publish_period = float(gp("publish_period").value)
        self.diag_period = float(gp("diag_period").value)

        # -- geometry resolution --------------------------------------------
        n_ch = len(self.channels)
        pos = parse_positions_mm(positions_mm, n_ch)
        if pos is None:
            if n_ch != 4:
                raise ValueError(
                    f"the built-in geometry is the 4-mic ReSpeaker layout but "
                    f"{n_ch} channel(s) are selected. Give -p positions_mm "
                    f"with one 'x,y' pair per channel, in millimetres, in "
                    f"channel order.")
            pos = dc.array_positions(radius_mm / 1000.0)
            geom_desc = (f"built-in ReSpeaker 4-mic, radius {radius_mm:.1f} mm "
                         f"(ch1..ch4 at 45/135/225/315 deg CCW)")
        else:
            geom_desc = (f"explicit positions_mm, {pos.shape[0]} mics, "
                         f"aperture "
                         f"{max(np.linalg.norm(pos[i]-pos[j]) for i, j in dc.pairs_for(pos.shape[0]))*1000:.0f} mm")
        self.pos = pos

        if self.subtract_noise and self.vad_mode == "off":
            raise ValueError(
                "subtract_noise=true with vad_mode='off' is meaningless: with "
                "no gate every frame goes to the speech pool, so the noise "
                "pool stays empty and there is nothing to subtract. Use "
                "vad_mode 'neural' or 'energy'.")

        # -- speech gate ------------------------------------------------------
        if vad_model_path is None and self.vad_mode == "neural":
            vad_model_path = self._default_vad_path()
        try:
            self.gate = SpeechGate(
                self.vad_mode, threshold=self.vad_threshold,
                percentile=self.vad_percentile, model_path=vad_model_path)
        except Exception as exc:
            # Explicit failure, never a silent downgrade: a quietly-substituted
            # gate changes the estimator's behaviour with nothing in the log.
            raise RuntimeError(
                f"vad_mode='{self.vad_mode}' could not be initialised: {exc}\n"
                f"Install onnxruntime and ensure vad.onnx is present, or run "
                f"with -p vad_mode:=energy (rms gating, no model) or "
                f"-p vad_mode:=off (no gating at all -- pooled SRP tolerates "
                f"this well; see the SpeechGate docstring).") from exc

        self.est = BearingEstimator(
            self.pos, fs=self.fs, gate=self.gate, avg_tc=avg_tc,
            subtract_noise=self.subtract_noise)

        self.get_logger().info(
            f"doa_mod starting: geometry = {geom_desc}; band "
            f"{dc.F_LO:.0f}-{self.est.f_hi:.0f} Hz (aliasing limit "
            f"{dc.aliasing_limit_hz(self.pos):.0f} Hz); aperture "
            f"{self.est.aperture*1000:.0f} mm; beamwidth ~"
            f"{dc.beamwidth_deg(self.pos, self.est.f_hi):.0f} deg")
        self.get_logger().info(
            f"vad_mode={self.vad_mode} "
            + (f"threshold={self.vad_threshold:.2f} " if self.vad_mode == 'neural'
               else f"percentile={self.vad_percentile:.0f} " if self.vad_mode == 'energy'
               else "(no gating: every frame moves the bearing) ")
            + f"hangover={self.hangover_s*1000:.0f} ms, avg_tc={avg_tc:.2f}s, "
            f"subtract_noise={self.subtract_noise}")
        self.get_logger().warning(
            "GEOMETRY CHECK: a MIRRORED channel map is not a rotation and "
            "cannot be calibrated away. If bearings are right near 90/270 deg "
            "but wrong near 0/180 deg, the channel order is mirrored. Verify "
            "against a known talker position before trusting the output.")

        # -- publishers -------------------------------------------------------
        self.pub = self.create_publisher(Float32, str(gp("output_topic").value), 10)
        self.state_pub = self.create_publisher(
            Float32MultiArray, str(gp("state_topic").value), 10)
        self.human_pub = (
            self.create_publisher(String, str(gp("human_topic").value), 10)
            if bool(gp("publish_human").value) else None)
        self.diag_pub = self.create_publisher(String, "/doa_mod/diagnostics", 10)

        # -- source -----------------------------------------------------------
        if self.source_kind == "topic":
            self.src = TopicSource(self, self.input_topic,
                                   n_ch_expected=n_ch, fs=self.fs)
            self.get_logger().info(
                f"source=topic: subscribing to {self.input_topic} "
                f"({n_ch}ch @ {self.fs} Hz expected)")
        elif self.source_kind == "mic":
            self.src = MicSource(
                device=self.device, channels=tuple(self.channels),
                n_dev_ch=self.device_channels, fs=self.fs,
                logger=self.get_logger())
            self.fs = self.src.fs
            self.get_logger().info(
                f"source=mic: '{self.src.name}', {self.device_channels}ch open, "
                f"using channels {self.channels} @ {self.fs} Hz")
        else:
            raise ValueError(
                f"source must be 'mic' or 'topic', got {self.source_kind!r}")

        self._t0 = time.monotonic()
        self._seq = 0
        self._last_speech_t = None
        self._last_pub = 0.0
        self._last_diag = time.monotonic()
        self._n_pub = 0
        self._busy_s = 0.0
        self._running = True

        self.src.start()
        self._worker = threading.Thread(
            target=self._run, name="doa-worker", daemon=True)
        self._worker.start()
        self.get_logger().info(
            f"Publishing {gp('output_topic').value} (Float32, degrees) every "
            f"{self.publish_period:.2f}s while speech is present")

    # -- helpers -----------------------------------------------------------
    def _default_vad_path(self):
        """vad.onnx from the installed share dir, else next to this file."""
        try:
            from ament_index_python.packages import get_package_share_directory
            cand = os.path.join(
                get_package_share_directory("doa_mod"), "vad.onnx")
            if os.path.isfile(cand):
                return cand
        except Exception:
            pass
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "vad.onnx")

    # -- worker ------------------------------------------------------------
    def _run(self):
        """Pull one hop at a time and keep the estimator fed.

        Paced by sample availability, not by a timer: `read` returns the newest
        BLOCK only when HOP fresh samples have arrived, so this thread
        self-clocks to the audio device.
        """
        period = HOP / float(self.fs)
        next_due = time.monotonic()
        last_written = -1
        while self._running:
            block = self.src.read(BLOCK)
            written = getattr(self.src.ring, 'written', 0)
            if block is None or written - last_written < HOP:
                # Not enough new audio yet; sleep a fraction of a hop.
                time.sleep(period * 0.25)
                if time.monotonic() - next_due > 5.0:
                    next_due = time.monotonic()
                continue
            last_written = written
            t_start = time.monotonic()
            try:
                info = self.est.push(block)
            except Exception as exc:
                self.get_logger().error(f"estimator failed on a frame: {exc}")
                time.sleep(period)
                continue
            self._busy_s += time.monotonic() - t_start

            now = time.monotonic()
            if info['active']:
                self._last_speech_t = now
            self._maybe_publish(now)
            self._maybe_diag(now)

    def _maybe_publish(self, now):
        """Publish at a fixed cadence while speech is recent."""
        if now - self._last_pub < self.publish_period:
            return
        speech_recent = (self._last_speech_t is not None
                         and (now - self._last_speech_t) <= self.hangover_s)
        if not speech_recent:
            # Deliberately silent. Publishing a stale pooled bearing during
            # silence would present the room's noise field as a measurement --
            # exactly what the frame gate exists to prevent. Consumers should
            # treat "no message" as "no talker", with their own timeout.
            return
        az, sharp = self.est.bearing()
        if not np.isfinite(az):
            return
        self._last_pub = now
        self._emit(az, sharp, now)

    def _emit(self, az, sharp, now):
        self.pub.publish(Float32(data=float(az)))
        msg = Float32MultiArray()
        # [elapsed_s, azimuth_deg, sharpness, consistent_frac, speech_flag, seq]
        #
        # elapsed_s counts from node start, matching data[0] on
        # /dorai_clean_audio. Derived from the frame counter rather than the
        # wall clock so it stays correct when frames are driven faster or
        # slower than realtime (offline replay, tests).
        msg.data = [
            float(self.est.n_frames * HOP / float(self.fs)),
            float(az),
            float(sharp),
            float(self.est.consistent_frac),
            1.0 if self.est.last_active else 0.0,
            float(self._seq),
        ]
        msg.layout.dim = [MultiArrayDimension(
            label="doa_state", size=6, stride=6)]
        self.state_pub.publish(msg)

        # A Float32MultiArray echoes as a bare list of numbers, so reading it
        # off a terminal means remembering that index 1 is the bearing. This
        # topic spells the same state out for a human with `ros2 topic echo`.
        # Machine consumers should use /dorai_doa (one float) or the state
        # topic (fixed field order); this one's format is NOT a contract.
        if self.human_pub is not None:
            self.human_pub.publish(String(data=(
                f"bearing {az:5.1f} deg  |  {self._clock_hint(az)}  |  "
                f"confidence {sharp:.2f}  |  frame gate "
                f"{100.0 * self.est.consistent_frac:3.0f}%  |  "
                f"t {msg.data[0]:6.1f}s  seq {self._seq}")))

        self._seq += 1
        self._n_pub += 1

    @staticmethod
    def _clock_hint(az):
        """Coarse o'clock reading of a bearing, for the human topic only.

        Bearings are measured counter-clockwise from the array's 0 deg axis,
        which on the ReSpeaker silkscreen is the mark labelled DOA:0. A clock
        face is the least ambiguous plain-language rendering: it needs no
        assumption about which way the array is bolted to the robot, only
        that you know where the board's own 0 deg mark points. 90 deg is
        therefore 9 o'clock, not 3, because the numbering runs the other way.
        """
        hour = int(round((-az % 360.0) / 30.0)) % 12
        return f"{12 if hour == 0 else hour} o'clock from the array's 0 deg mark"

    def _maybe_diag(self, now):
        if now - self._last_diag < self.diag_period:
            return
        elapsed = now - self._last_diag
        self._last_diag = now
        frames = max(self.est.n_frames, 1)
        duty = 100.0 * self._busy_s / max(elapsed, 1e-9)
        self._busy_s = 0.0
        az, sharp = self.est.bearing()
        xruns = getattr(self.src, 'xruns', 0)
        dropped = getattr(self.src, 'n_dropped', 0)
        line = (
            f"az={az:.1f} deg sharp={sharp:.2f} | frames={self.est.n_frames} "
            f"speech={100.0*self.est.n_speech/frames:.0f}% "
            f"gate_pass={100.0*self.est.consistent_frac:.0f}% | "
            f"published={self._n_pub} | worker_duty={duty:.0f}% "
            f"xruns={xruns} dropped={dropped}")
        self.get_logger().info(line)
        self.diag_pub.publish(String(data=line))
        if duty > 80.0:
            self.get_logger().warning(
                f"worker duty {duty:.0f}% of realtime -- the estimator is "
                f"close to falling behind the audio. Coarsen the SRP grid or "
                f"raise HOP if this persists.")

    def shutdown(self):
        self._running = False
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self.src.stop()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DoaMod()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
