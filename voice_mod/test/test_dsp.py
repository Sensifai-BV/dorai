"""DSP regression tests for voice_mod (no real mics, no ROS required).

These exercise the parts most likely to break silently:
  * the clock-drift loop: channels from independent USB clocks must stay
    aligned to a few samples, and must not be moved against each other from
    block to block (a beamformer needs the delays between channels to stand
    still), with the capture callbacks landing at arbitrary moments;
  * start-up alignment from capture time stamps;
  * the consumer keeping the FIFOs bounded and the frames exact when worker
    ticks stall;
  * the AGC: speech level, no gain pumping in pauses, no clipping;
  * the streaming enhancer driver with the shipped model.

The simulation carries real content through a variable-ratio resampler (each
sample's value is its true capture time), so alignment is measured, not
inferred from buffer lengths.
"""
import os
import random
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

try:
    import voice as V
except ImportError:      # no ROS / PortAudio on this machine: the DSP classes need neither
    for _name in ("sounddevice", "rclpy", "rclpy.node", "rclpy.signals", "std_msgs", "std_msgs.msg", "onnxruntime"):
        try:
            __import__(_name)
        except ImportError:
            sys.modules[_name] = types.ModuleType(_name)
    sys.modules["rclpy.node"].Node = getattr(sys.modules["rclpy.node"], "Node", object)
    sys.modules["rclpy.signals"].SignalHandlerOptions = getattr(
        sys.modules["rclpy.signals"], "SignalHandlerOptions", object)
    for _cls in ("Float32MultiArray", "MultiArrayDimension", "String"):
        if not hasattr(sys.modules["std_msgs.msg"], _cls):
            setattr(sys.modules["std_msgs.msg"], _cls, object)
    sys.modules.pop("voice", None)
    import voice as V

FS = V.OUTPUT_RATE


class _InterpResampler:
    """Variable-ratio linear interpolation that carries the content through."""

    def __init__(self, base):
        self.base, self.variable, self.backend = base, True, "fake-interp"
        self.pos, self.prev = 0.0, None        # position of the next output sample, in input samples

    def process(self, x, ratio_scale=1.0):
        if x.size == 0:
            return np.zeros(0, np.float64)
        xs = np.concatenate([[x[0] if self.prev is None else self.prev], x]).astype(np.float64)
        step = 1.0 / (self.base * ratio_scale)
        n = int(np.floor((len(x) - 1 - self.pos) / step)) + 1 if self.pos <= len(x) - 1 else 0
        q = self.pos + step * np.arange(max(n, 0)) + 1.0
        i = np.floor(q).astype(int)
        y = xs[i] * (1 - (q - i)) + xs[np.minimum(i + 1, len(xs) - 1)] * (q - i)
        self.pos = self.pos + step * max(n, 0) - len(x)
        self.prev = x[-1]
        return y


def _push64(self, samples, capture_mono):
    """MicChannel.push without the float32 cast (the content is a time in seconds)."""
    with self.lock:
        self.in_buf = np.concatenate((self.in_buf, samples)) if self.in_buf.size else np.asarray(samples, np.float64)
        self.last_capture_mono = capture_mono
        self.in_end_mono = capture_mono + len(samples) / float(self.sample_rate)


def _run_pipeline(true_rates, duration_s=150.0, block=4800, stamp_jitter_s=0.0007, tick_jitter=0.2,
                  stall_prob=0.0, drop_at=None, seed=0):
    """Drive MicChannel through the capture / resample / drift / consume loop of
    VoiceMod._tick. Each device has its own true clock and hands its audio over
    in callbacks of `block` samples at its own phase, with a jittered capture
    stamp; worker ticks jitter and can stall. Returns the channels, per popped
    block (time, offsets of each channel against mic0 in samples, ratios, FIFO
    depths), and the sizes of the emitted 10 s frames."""
    rng = random.Random(seed)
    nominal = 48000
    mics = [V.MicChannel(f"mic{i}", i, nominal, hpf_hz=0.0) for i in range(len(true_rates))]
    for ch in mics:
        ch.resampler = _InterpResampler(FS / nominal)
        ch.push = types.MethodType(_push64, ch)
    phase = [rng.uniform(0, block / r) for r in true_rates]
    sent = [0] * len(mics)
    frame_blocks = max(1, round(10.0 * FS / V.L_OUT))
    primed, t, acc, frames, log = False, 0.0, 0, [], []
    while t < duration_s:
        t += V.BLOCK_S * (1 + rng.uniform(-tick_jitter, tick_jitter))
        for i, ch in enumerate(mics):
            while (sent[i] + 1) * block / true_rates[i] + phase[i] <= t:        # callback j has completed
                j = sent[i]
                sent[i] += 1
                start = j * block / true_rates[i] + phase[i]
                if drop_at and drop_at[0] == i and abs(start - drop_at[1]) < 0.05:
                    continue                                                    # one lost callback (xrun)
                ch.push(np.arange(j * block, (j + 1) * block) / true_rates[i] + phase[i],
                        start + rng.gauss(0, stamp_jitter_s))
        if stall_prob and rng.random() < stall_prob:
            continue                                                            # worker busy: input piles up
        for ch in mics:
            ch.resample_in()
        if primed:
            V.lock_channels(mics)
        else:
            depths = [ch.fifo_len() for ch in mics]
            if min(depths) < max(V.TARGET_OUT, V.L_OUT):
                continue
            extra = V.prime_offsets([ch.fifo_end_mono for ch in mics])
            common = min(d - e for d, e in zip(depths, extra))
            if common < V.L_OUT:
                continue
            for ch, e in zip(mics, extra):
                ch.trim_fifo(common + e)
            primed = True
        n_blocks = max(0, (min(ch.fifo_len() for ch in mics) - V.TARGET_OUT) // V.L_OUT)
        for _ in range(int(n_blocks)):
            heads = [float(ch.pop_block()[0]) for ch in mics]                   # true capture time at the block start
            acc += 1
            log.append((t, [(h - heads[0]) * FS for h in heads],
                        [ch.ratio_scale for ch in mics], [ch.fifo_len() for ch in mics]))
        while acc >= frame_blocks:
            frames.append(frame_blocks * V.L_OUT)
            acc -= frame_blocks
    return mics, log, frames


def _offsets(log, t0, t1):
    return np.array([o[1:] for t, o, _, _ in log if t0 <= t < t1])


def test_channels_stay_aligned_under_clock_drift():
    """Three mics on independent clocks (+-100 ppm) stay within a millisecond of
    each other once the loop has settled, and are never moved against each
    other by more than a fraction of a sample from one block to the next."""
    mics, log, frames = _run_pipeline([48000, 48004.8, 47995.2])
    late = _offsets(log, 100, 150)
    assert np.abs(late).max() < 24                         # 1.5 ms while the integral part is still settling
    assert np.abs(np.diff(_offsets(log, 5, 150), axis=0)).max() < 0.5
    fifo = np.array([f for _, _, _, f in log])
    rs = np.array([r for _, _, r, _ in log])
    assert fifo.max() < V.MAX_OUT_BLOCKS * V.L_OUT
    assert sum(m.out_overflows for m in mics) == 0 and sum(m.underruns for m in mics) == 0
    assert np.abs(rs - 1.0).max() < 0.5 * V.RATIO_TRIM_MAX   # far from the clamp
    assert len(frames) >= 10


def test_drift_loop_keeps_one_clock_still():
    """With every device on the same clock there is nothing to correct: the loop
    must not turn the 100 ms granularity of the callbacks into delay changes
    (the length-based loop moved channels by up to 38 samples per block here)."""
    mics, log, _ = _run_pipeline([48000, 48000, 48000])
    off = _offsets(log, 5, 150)
    assert np.abs(np.diff(off, axis=0)).max() < 0.2
    assert np.ptp(_offsets(log, 100, 150), axis=0).max() < 8      # what the 0.7 ms stamp jitter leaves
    assert np.abs(_offsets(log, 100, 150)).max() < 8
    assert sum(m.out_overflows for m in mics) == 0 and sum(m.underruns for m in mics) == 0


def test_alignment_recovers_after_a_lost_block():
    """A dropped capture block puts one channel 100 ms off; the stamps show it
    and the loop pulls the channel back."""
    mics, log, _ = _run_pipeline([48000, 48004.8, 47995.2], duration_s=200.0, drop_at=(1, 60.0))
    assert np.abs(_offsets(log, 60.5, 61.5)[:, 0]).max() > 1000      # the loss is real
    assert np.abs(_offsets(log, 150, 200)).max() < 40


def test_bounded_and_exact_frames_under_tick_jitter():
    """Even when a quarter of ticks stall (input piles up), the FIFO stays
    bounded, nothing is dropped and every frame is exactly 10 s."""
    mics, log, frames = _run_pipeline([48000, 48010, 47990], duration_s=80.0, stall_prob=0.25, seed=1)
    fifo = np.array([f for _, _, _, f in log])
    rs = np.array([r for _, _, r, _ in log])
    assert fifo.max() < V.MAX_OUT_BLOCKS * V.L_OUT
    assert sum(m.out_overflows for m in mics) == 0
    assert sum(m.underruns for m in mics) == 0
    assert frames and set(frames) == {10 * FS}
    assert 0.97 < rs.mean() < 1.03


def test_prime_offsets():
    po = V.prime_offsets
    assert po([10.0, 10.1003, 10.0]) == [0, 1600, 0]         # one device's block landed later: exactly one block
    assert po([5.0, 5.0, 5.0]) == [0, 0, 0]
    assert po([1.0, None]) == [0, 0]                         # no stamps: the old behaviour
    assert po([2.0, 2.2004, 2.037]) == [0, 3200, 592]        # independent device: the measured offset
    assert po([3.0, 3.0995]) == [0, 1600]


def _agc(y, hop=V.L_OUT, **kw):
    ns = types.SimpleNamespace(hop=hop, fs=FS, g=1.0, tgt=0.06, alpha=float(np.exp(-hop / (2.0 * FS))),
                               max_gain=15.0, gate=3e-3, ceil=0.99, **kw)
    return V.ContinuousEnhancer._agc(ns, np.asarray(y, np.float64)), ns


def _speechy(seconds, rms, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    x = np.sin(2 * np.pi * 200 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) + 0.1 * rng.standard_normal(len(t))
    return x * rms / np.sqrt(np.mean(x ** 2))


def test_agc_levels_speech_and_leaves_pauses_alone():
    """Commands over pauses that an enhancer has cleaned: the speech comes out
    near the target, nothing reaches the ceiling, and the gain does not climb
    in the pauses (the first AGC took the residue for signal, drove the gain
    to its maximum and the next word into the limiter)."""
    rng = np.random.default_rng(0)
    parts, marks, pos = [], [], 0
    for k in range(6):
        pause = 0.0008 * rng.standard_normal(int(2.5 * FS))          # residue far below the speech
        word = _speechy(1.2, 0.02, k)
        parts += [pause, word]
        marks.append((pos + len(pause), pos + len(pause) + len(word)))
        pos += len(pause) + len(word)
    x = np.concatenate(parts)
    y, ns = _agc(x)
    assert np.abs(y).max() < 0.99
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    for a, b in marks[2:]:                                           # once the gain has found the level
        assert -6.0 < 20 * np.log10(rms(y[a:b]) / 0.06) < 4.0
    gains = [rms(y[a:b]) / rms(x[a:b]) for a, b in marks[2:]]
    pause_gain = [rms(y[b + FS // 2:b + 2 * FS]) / rms(x[b + FS // 2:b + 2 * FS]) for a, b in marks[2:-1]]
    assert max(pause_gain) < 1.3 * max(gains)                        # no climb between the words
    # the same on 1 s segments (the overlap-add driver's hop): the gain still moves
    y2, _ = _agc(x, hop=FS)
    assert np.abs(y2).max() < 0.99 and rms(y2[marks[-1][0]:marks[-1][1]]) > 2 * rms(x[marks[-1][0]:marks[-1][1]])


def test_agc_limits_a_bang_without_stepping():
    x = np.concatenate([_speechy(4.0, 0.02, 1), 0.8 * np.random.default_rng(2).standard_normal(FS // 10),
                        _speechy(4.0, 0.02, 3)])
    y, _ = _agc(x)
    assert np.abs(y).max() <= 0.99
    steps = np.abs(np.diff(y)) - np.abs(np.diff(x)) * 15.0           # never a jump the input did not have
    assert steps.max() < 0.05


def _model_path():
    return os.path.join(HERE, "..", "dorai_beamformer.ort")


def _have_model():
    if not os.path.exists(_model_path()):
        return False
    try:
        import onnxruntime as ort
        return hasattr(ort, "InferenceSession")
    except ImportError:
        return False


class _Log:
    def error(self, *a, **k): pass
    info = warning = error


@pytest.mark.skipif(not _have_model(), reason="dorai_beamformer.ort or onnxruntime not present")
def test_shipped_model_is_streaming_and_runs_block_by_block():
    """The shipped beamformer is a streaming model: the driver feeds it block by
    block, the output is as long as the input once flushed, and feeding the
    same audio in other frame sizes gives the same samples."""
    assert V.StreamingEnhancer.accepts(_model_path())
    rng = np.random.default_rng(0)
    enh = V.StreamingEnhancer(_model_path(), 1, _Log())
    assert enh.n_mics == 3 and enh.block == 512 and enh.latency == 384
    n = 3 * FS
    x = 0.02 * rng.standard_normal((3, n)).astype(np.float32)
    x[:, FS:2 * FS] += _speechy(1.0, 0.05, 1).astype(np.float32)
    out = np.concatenate([enh.push(x[:, i:i + V.L_OUT]) for i in range(0, n, V.L_OUT)] + [enh.flush()])
    assert out.shape == (n,) and out.dtype == np.float32 and np.all(np.isfinite(out))
    assert np.abs(out).max() < 0.99 and float(np.sqrt(np.mean(out ** 2))) > 1e-4
    enh2 = V.StreamingEnhancer(_model_path(), 1, _Log())
    out2 = np.concatenate([enh2.push(x[:, i:i + 5 * FS]) for i in range(0, n, 5 * FS)] + [enh2.flush()])
    assert out2.shape == (n,)
    assert np.abs(out2 - out).max() < 0.01                           # same model output; the AGC grid differs slightly
    assert np.corrcoef(out, out2)[0, 1] > 0.995
