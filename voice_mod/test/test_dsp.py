"""DSP regression tests for voice_mod (no real mics required).

These exercise the parts most likely to break silently:
  * the consumer keeping the per-mic FIFOs bounded and the channels aligned
    even when worker ticks jitter and independent USB clocks drift;
  * the dorai_beamformer.ort sliding-window beamformer producing aligned mono blocks.

They import only the pure-DSP classes. A faithful fake variable-ratio
resampler stands in for libsamplerate so the drift loop is exercised
deterministically; if onnxruntime or the model file is missing, the
beamformer test self-skips.
"""
import os
import random

import numpy as np
import pytest

import voice as V


class _FakeVarResampler:
    """Variable-ratio resampler: emits round(n*base*scale) samples, carrying
    the fractional remainder so the long-run ratio is exact."""

    def __init__(self, base):
        self.base = base
        self.variable = True
        self.backend = "fake-var"
        self.carry = 0.0

    def process(self, x, ratio_scale=1.0):
        want = x.size * self.base * ratio_scale + self.carry
        n = int(round(want))
        self.carry = want - n
        return np.zeros(max(n, 0), dtype=np.float32)


def _run_pipeline(true_rates, duration_s=60.0, stall_prob=0.0, seed=0):
    """Drive MicChannel through the same capture/resample/drift/consume loop
    as VoiceMod._tick, with the given per-mic true clocks and an optional
    probability that a worker tick is skipped (modelling an inference stall
    that lets input pile up). Returns recorded FIFO depths, ratios and the
    sizes of any emitted N-second frames."""
    nominal = 48000
    mics = [V.MicChannel(f"mic{i}", i, nominal) for i in range(len(true_rates))]
    for ch in mics:
        ch.resampler = _FakeVarResampler(V.OUTPUT_RATE / nominal)

    rng = random.Random(seed)
    frame_blocks = max(1, round(10.0 * V.OUTPUT_RATE / V.L_OUT))
    acc, frames = [], []
    fifo_log, rs_log = [], []
    primed = False
    produced = [0.0] * len(mics)
    t = 0.0

    for _ in range(int(duration_s / V.BLOCK_S)):
        t += V.BLOCK_S
        for i, ch in enumerate(mics):
            target = int(true_rates[i] * t)
            n = target - int(produced[i])
            produced[i] = target
            with ch.lock:
                ch.in_buf = np.concatenate(
                    (ch.in_buf, np.zeros(n, dtype=np.float32)))

        if stall_prob and rng.random() < stall_prob:
            continue  # worker busy: skip processing, input accumulates

        for ch in mics:
            ch.resample_in()
        if len(mics) > 1:
            ref = sum(ch.fifo_len() for ch in mics) / len(mics)
            for ch in mics:
                ch.update_ratio(ref)

        if not primed:
            depths = [ch.fifo_len() for ch in mics]
            if min(depths) >= max(V.TARGET_OUT, V.L_OUT):
                common = min(depths)
                for ch in mics:
                    ch.trim_fifo(common)
                primed = True
            continue

        min_fifo = min(ch.fifo_len() for ch in mics)
        n_blocks = max(0, (min_fifo - V.TARGET_OUT) // V.L_OUT)
        for _ in range(int(n_blocks)):
            for ch in mics:
                acc.append(ch.pop_block())
        # one clean block == one popped block per channel; count whole frames
        while len(acc) >= frame_blocks * len(mics):
            frames.append(frame_blocks * V.L_OUT)
            del acc[:frame_blocks * len(mics)]

        fifo_log.append([ch.fifo_len() for ch in mics])
        rs_log.append([ch.ratio_scale for ch in mics])

    return mics, np.array(fifo_log), np.array(rs_log), frames


def test_channels_stay_aligned_under_clock_drift():
    """Three mics on independent clocks must stay phase-aligned (small FIFO
    spread), bounded, and keep their ratios sane."""
    mics, fifo, rs, frames = _run_pipeline([48000, 48050, 47950])

    spread = (fifo.max(axis=1) - fifo.min(axis=1))[50:]
    assert spread.max() < V.L_OUT            # channels within one block
    assert fifo.max() < V.MAX_OUT_BLOCKS * V.L_OUT
    assert sum(m.out_overflows for m in mics) == 0
    assert sum(m.underruns for m in mics) == 0
    assert 0.97 < rs[50:].mean() < 1.03      # no clamp saturation
    assert len(frames) >= 4


def test_bounded_and_exact_frames_under_tick_jitter():
    """Even when a quarter of ticks stall (input piles up), the FIFO stays
    bounded, no samples are lost, and every published frame is exactly 10 s."""
    mics, fifo, rs, frames = _run_pipeline(
        [48000, 48010, 47990], duration_s=80.0, stall_prob=0.25, seed=1)

    assert fifo.max() < V.MAX_OUT_BLOCKS * V.L_OUT
    assert sum(m.out_overflows for m in mics) == 0
    assert sum(m.underruns for m in mics) == 0
    assert frames and set(frames) == {10 * V.OUTPUT_RATE}   # all exactly 10 s
    assert 0.97 < rs.mean() < 1.03


def _model_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "dorai_beamformer.ort")


@pytest.mark.skipif(
    not os.path.exists(_model_path()), reason="dorai_beamformer.ort not present"
)
def test_beamformer_emits_aligned_mono_blocks():
    pytest.importorskip("onnxruntime")

    class _Log:
        def error(self, *a, **k):
            pass

    beam = V.Beamformer(_model_path(), 1, _Log())
    beam.warmup(3)

    rng = np.random.default_rng(0)
    produced = 0
    for _ in range(20):
        block = (0.05 * rng.standard_normal((3, V.L_OUT))).astype(np.float32)
        for clean in beam.process_block(block):
            assert clean.shape == (V.STEP_SIZE,)
            assert clean.dtype == np.float32
            assert np.all(np.isfinite(clean))
            produced += 1
    assert produced >= 15
