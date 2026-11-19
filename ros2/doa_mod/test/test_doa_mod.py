"""Regression tests for doa_mod — the wiring, not the estimator.

doa_core.py has its own synthetic ground-truth suite in doa/tests/. This file
covers what doa_mod adds on top of it, which is where this package's own bugs
can live:

  * the vendored copy of the validated estimator has not forked;
  * the ring buffer's wraparound is correct;
  * the VAD is called with the convention the model expects;
  * the vad_mode switch actually switches, including 'off';
  * the node publishes when it should and -- more importantly -- STAYS SILENT
    when there is no talker.

Runs without a ROS 2 installation via test/rosmock.py.

    python3 -m pytest doa_mod/test/test_doa_mod.py -v
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PKG)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PKG)

import rosmock                                                     # noqa: E402
rosmock.install()          # must precede `import doa`

import doa as D                                                    # noqa: E402
import doa_core as dc                                              # noqa: E402

FS = 16000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def band_noise(n, seed=0, f_lo=dc.F_LO, f_hi=dc.F_HI, fs=FS):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n * 2)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    X[(f < f_lo) | (f > f_hi)] = 0.0
    return np.fft.irfft(X)[:n]


def _delay(x, d_samples):
    n = len(x)
    nf = 4 * n
    X = np.fft.rfft(x, nf)
    f = np.fft.rfftfreq(nf, 1.0)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * d_samples), nf)[:n]


def plane_wave(az_deg, pos, n=D.BLOCK, fs=FS, seed=0, amp=0.05):
    """[M,n] far-field block. Mic nearest the source arrives FIRST."""
    src = band_noise(n * 2, seed=seed, fs=fs)
    u = np.array([np.cos(np.deg2rad(az_deg)), np.sin(np.deg2rad(az_deg))])
    t = -(np.asarray(pos) @ u) / dc.C_SOUND
    t = t - t.min() + 1e-3
    block = np.stack([_delay(src, ti * fs)[:n] for ti in t])
    return block * (amp / max(np.abs(block).max(), 1e-12))


def wrap180(a):
    return (float(a) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# 1. the vendored estimator must not fork from the validated one
# ---------------------------------------------------------------------------
def test_vendored_core_matches_research_tree():
    """doa_mod's doa_core.py/vad.py must be byte-identical to doa/'s.

    A ROS package cannot reach outside its own directory at build time, so the
    validated estimator has to be duplicated here. Duplication without a guard
    is how two copies silently diverge and the deployed one stops being the
    one that was measured. If this fails: fix the file in doa/, re-run
    doa/score_session.py to confirm the numbers still hold, then re-copy.
    """
    for name in ('doa_core.py', 'vad.py', 'vad.onnx'):
        research = os.path.join(_REPO, 'doa', name)
        vendored = os.path.join(_PKG, name)
        if not os.path.isfile(research):
            pytest.skip(f"research tree not present: {research}")
        h_r = hashlib.sha256(open(research, 'rb').read()).hexdigest()
        h_v = hashlib.sha256(open(vendored, 'rb').read()).hexdigest()
        assert h_r == h_v, (
            f"{name} has FORKED from doa/{name}.\n"
            f"  doa/     {h_r}\n"
            f"  doa_mod/ {h_v}\n"
            f"The deployed estimator is no longer the validated one.")


# ---------------------------------------------------------------------------
# 2. ring buffer
# ---------------------------------------------------------------------------
def test_ring_buffer_matches_naive_reference():
    """Randomised push/read against a grow-forever reference, incl. wraparound."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        cap = int(rng.integers(5, 60))
        n_ch = int(rng.integers(1, 5))
        ring = D.RingBuffer(n_ch, cap)
        ref = np.zeros((n_ch, 0))
        for _ in range(int(rng.integers(1, 12))):
            n = int(rng.integers(1, cap * 2))
            blk = rng.normal(size=(n_ch, n))
            ring.push(blk)
            ref = np.concatenate([ref, blk], axis=1)
            for q in (1, 3, max(cap // 2, 1), cap):
                got = ring.latest(q)
                if ref.shape[1] < q or q > cap:
                    assert got is None
                else:
                    assert np.allclose(got, ref[:, -q:])


def test_ring_buffer_does_not_allocate_per_push():
    """The buffer array must be written IN PLACE, not replaced.

    The research tree's np.roll() version allocated a fresh 1.92 M-sample array
    inside the realtime audio callback. Identity of .buf is the cheap proxy.
    """
    ring = D.RingBuffer(4, 1024)
    before = id(ring.buf)
    for _ in range(10):
        ring.push(np.zeros((4, D.HOP)))
    assert id(ring.buf) == before, "push() replaced the buffer instead of writing in place"


# ---------------------------------------------------------------------------
# 3. the VAD calling convention — audit FINDING 1
# ---------------------------------------------------------------------------
class _SpyVAD:
    """Records every chunk handed to the model."""

    def __init__(self):
        self.calls = []

    def process_chunk(self, audio):
        self.calls.append(np.asarray(audio).shape)
        return 0.9

    def reset(self):
        self.calls.clear()


def test_neural_vad_gets_exactly_one_hop_sized_chunk_per_frame():
    """One call, HOP samples, per analysis frame.

    The model is a STREAMING one: 512 new samples per call with an LSTM state
    carried across calls. The research tree passed the whole 2048-sample block
    every hop, which re-ran the model 4x AND fed each chunk to the LSTM four
    times at four different state positions. Measured cost: 35% of frames
    disagreed with correct streaming, and re-acquiring a moved talker went from
    2.73 s to 5.07 s. This test locks the convention down.
    """
    gate = D.SpeechGate('off')          # construct without loading the model
    gate.mode = 'neural'
    gate._vad = _SpyVAD()
    block = np.random.default_rng(0).normal(0, 0.05, (4, D.BLOCK))
    gate(block)
    assert len(gate._vad.calls) == 1, (
        f"model called {len(gate._vad.calls)}x for one frame -- the whole "
        f"block is being passed instead of just the new hop")
    assert gate._vad.calls[0] == (D.HOP,), (
        f"model received {gate._vad.calls[0]}, expected ({D.HOP},)")


def test_neural_gate_feeds_the_newest_samples():
    """The chunk must be the END of the block, not the start."""
    gate = D.SpeechGate('off')
    gate.mode = 'neural'
    captured = {}

    class _Cap:
        def process_chunk(self, audio):
            captured['x'] = np.asarray(audio).copy()
            return 1.0
    gate._vad = _Cap()
    block = np.arange(4 * D.BLOCK, dtype=float).reshape(4, D.BLOCK)
    gate(block)
    assert np.array_equal(captured['x'], block[0, -D.HOP:]), \
        "the gate is not feeding the newest HOP samples of channel 0"


# ---------------------------------------------------------------------------
# 4. the vad_mode switch
# ---------------------------------------------------------------------------
def test_vad_mode_off_admits_every_frame():
    """'off' must gate nothing at all -- including digital silence."""
    gate = D.SpeechGate('off')
    assert gate(np.zeros((4, D.BLOCK))) is True
    assert gate(np.random.default_rng(0).normal(0, 0.05, (4, D.BLOCK))) is True


def test_vad_mode_off_needs_no_onnxruntime():
    """Constructing an 'off' or 'energy' gate must not import onnxruntime.

    One reason 'off' exists is to drop the dependency on a constrained target.
    """
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name.startswith('onnxruntime'):
            raise ImportError("onnxruntime blocked by test")
        return real_import(name, *a, **k)

    builtins.__import__ = _blocked
    try:
        D.SpeechGate('off')
        D.SpeechGate('energy')
    finally:
        builtins.__import__ = real_import


def test_energy_gate_separates_loud_from_quiet():
    gate = D.SpeechGate('energy', percentile=70.0)
    rng = np.random.default_rng(0)
    for _ in range(30):                      # prime the history with quiet
        gate(rng.normal(0, 1e-4, (4, D.BLOCK)))
    assert gate(rng.normal(0, 0.05, (4, D.BLOCK))) is True, "loud frame rejected"


def test_unknown_vad_mode_is_rejected():
    with pytest.raises(ValueError, match="vad_mode"):
        D.SpeechGate('sometimes')


# ---------------------------------------------------------------------------
# 5. the estimator end-to-end on a synthetic source of known bearing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("truth", [0.0, 45.0, 90.0, 180.0, 270.0, 315.0])
def test_estimator_recovers_synthetic_bearing(truth):
    """A clean plane wave must be recovered within one beamwidth-limited step.

    Tolerance is 6 deg: the SRP grid is 1 deg, but a 64 mm aperture has a very
    broad main lobe, so a couple of degrees of peak wander is physics, not a
    bug.
    """
    pos = dc.array_positions()
    est = D.BearingEstimator(pos, fs=FS, gate=D.SpeechGate('off'), avg_tc=None)
    for k in range(12):
        est.push(plane_wave(truth, pos, seed=k))
    az, sharp = est.bearing()
    assert abs(wrap180(az - truth)) <= 6.0, \
        f"bearing {az:.1f} deg for a source at {truth:.0f} deg"
    assert sharp > 0.2


def test_estimator_reports_nan_when_no_speech_seen():
    """No speech frames => no bearing. Never a confident default."""
    est = D.BearingEstimator(dc.array_positions(), fs=FS,
                             gate=D.SpeechGate('energy'), avg_tc=None)
    for _ in range(10):
        est.push(np.zeros((4, D.BLOCK)))
    az, sharp = est.bearing()
    assert not np.isfinite(az)
    assert sharp == 0.0


def test_degenerate_frames_do_not_pass_the_quality_gate():
    """DC and silence must not count as consistent measurements."""
    est = D.BearingEstimator(dc.array_positions(), fs=FS,
                             gate=D.SpeechGate('off'), avg_tc=None)
    assert est.push(np.ones((4, D.BLOCK)))['consistent'] is False
    assert est.push(np.zeros((4, D.BLOCK)))['consistent'] is False


# ---------------------------------------------------------------------------
# 6. geometry handling
# ---------------------------------------------------------------------------
def test_positions_mm_parses_and_scales():
    pos = D.parse_positions_mm("0,32 -32,0 0,-32 32,0", 4)
    assert pos.shape == (4, 2)
    assert np.isclose(np.linalg.norm(pos[0]), 0.032), "mm were not converted to m"


def test_positions_mm_rejects_collinear_array():
    """A straight line cannot resolve which side of its axis a source is on."""
    with pytest.raises(ValueError, match="(?i)collinear"):
        D.parse_positions_mm("0,0 40,0 80,0", 3)


def test_positions_mm_rejects_channel_count_mismatch():
    with pytest.raises(ValueError, match="channel"):
        D.parse_positions_mm("0,32 -32,0 0,-32", 4)


def test_builtin_geometry_is_not_mirrored():
    """ch1..ch4 must run COUNTER-clockwise from 45 deg.

    The mirrored variant is the fault that shipped undetected for months: it
    reads as a rotation on any source near the mirror axis, so 0/180 deg marks
    cannot reveal it.
    """
    pos = dc.array_positions()
    ang = np.degrees(np.arctan2(pos[:, 1], pos[:, 0])) % 360.0
    assert np.allclose(np.sort(ang), [45.0, 135.0, 225.0, 315.0])
    assert ang[1] > ang[0], "channel order is clockwise -- geometry is mirrored"


# ---------------------------------------------------------------------------
# 7. the node: publish gating
# ---------------------------------------------------------------------------
def _make_node(**overrides):
    params = dict(source='topic', vad_mode='off', publish_period=0.0,
                  vad_hangover_ms=500.0)
    params.update(overrides)
    rosmock.set_overrides(params)
    node = D.DoaMod()
    node._running = False
    if node._worker.is_alive():
        node._worker.join(timeout=2.0)
    return node


def test_node_declares_the_topic_contract():
    node = _make_node()
    try:
        assert '/dorai_doa' in node.publishers_by_topic
        assert '/dorai_doa/state' in node.publishers_by_topic
        assert '/doa_mod/diagnostics' in node.publishers_by_topic
    finally:
        node.shutdown()


def test_node_publishes_bearing_for_a_known_source():
    node = _make_node()
    try:
        pos = dc.array_positions()
        t = 0.0
        for k in range(15):
            node.est.push(plane_wave(120.0, pos, seed=k))
            t += D.HOP / FS
            node._last_speech_t = t
            node._maybe_publish(t)
        sent = node.publishers_by_topic['/dorai_doa'].sent
        assert sent, "nothing was published for a strong synthetic source"
        assert abs(wrap180(sent[-1].data - 120.0)) <= 6.0
        state = node.publishers_by_topic['/dorai_doa/state'].sent[-1]
        assert len(state.data) == 6, "state message must carry 6 fields"
        assert abs(state.data[1] - sent[-1].data) < 1e-6, \
            "state azimuth disagrees with the Float32 topic"
    finally:
        node.shutdown()


def test_node_goes_silent_after_hangover():
    """No talker => no message. A stale bearing must never be re-published.

    Publishing the last pooled bearing during silence would present the room's
    noise field as a measurement. Consumers treat 'no message' as 'no talker'.
    """
    node = _make_node(vad_mode='neural', vad_hangover_ms=300.0,
                      publish_period=0.05)
    try:
        pos = dc.array_positions()
        hop_s = D.HOP / FS
        t = 0.0
        for k in range(20):                      # speech
            info = node.est.push(plane_wave(90.0, pos, seed=k))
            t += hop_s
            if info['active']:
                node._last_speech_t = t
            node._maybe_publish(t)
        n_speech = len(node.publishers_by_topic['/dorai_doa'].sent)
        assert n_speech > 0, "nothing published while a source was present"

        for _ in range(60):                      # then silence
            info = node.est.push(np.zeros((4, D.BLOCK)))
            t += hop_s
            if info['active']:
                node._last_speech_t = t
            node._maybe_publish(t)
        n_end = len(node.publishers_by_topic['/dorai_doa'].sent)

        # A little more may go out during the hangover, then it must stop.
        for _ in range(40):
            node.est.push(np.zeros((4, D.BLOCK)))
            t += hop_s
            node._maybe_publish(t)
        assert len(node.publishers_by_topic['/dorai_doa'].sent) == n_end, \
            "still publishing well past the hangover -- the gate is not closing"
    finally:
        node.shutdown()


def test_human_topic_states_the_bearing_in_words():
    """`ros2 topic echo /dorai_doa/human` must be readable without a decoder ring.

    The state topic is a Float32MultiArray, which echoes as a bare list of
    numbers -- reading a bearing off a terminal means knowing that index 1 is
    the one you want. This topic exists so that a 270 deg bearing prints as
    the words "bearing 270.0 deg".
    """
    node = _make_node()
    try:
        pos = dc.array_positions()
        t = 0.0
        for k in range(15):
            node.est.push(plane_wave(270.0, pos, seed=k))
            t += D.HOP / FS
            node._last_speech_t = t
            node._maybe_publish(t)
        sent = node.publishers_by_topic['/dorai_doa/human'].sent
        assert sent, "nothing published on the human topic"
        text = sent[-1].data
        assert 'bearing' in text and 'deg' in text, f"not human-readable: {text!r}"
        # The number in the text must be the number on the Float32 topic.
        az = node.publishers_by_topic['/dorai_doa'].sent[-1].data
        assert f"{az:5.1f}".strip() in text, \
            f"human text {text!r} disagrees with the published bearing {az}"
    finally:
        node.shutdown()


def test_human_topic_can_be_switched_off():
    node = _make_node(publish_human=False)
    try:
        assert '/dorai_doa/human' not in node.publishers_by_topic
        assert node.human_pub is None
        # and emitting must not blow up with the publisher absent
        node.est.push(plane_wave(90.0, dc.array_positions()))
        node._last_speech_t = 1.0
        node._maybe_publish(1.0)
    finally:
        node.shutdown()


@pytest.mark.parametrize("az,hour", [(0.0, 12), (90.0, 9), (180.0, 6),
                                     (270.0, 3), (359.6, 12)])
def test_clock_hint_runs_counter_clockwise(az, hour):
    """90 deg is 9 o'clock, NOT 3 -- bearings run counter-clockwise.

    Getting this backwards would print a mirrored plain-language reading on
    top of a correct bearing, which is worse than printing nothing: it is the
    same mirror confusion that cost this project months, dressed up as a
    helpful hint.
    """
    assert D.DoaMod._clock_hint(az).startswith(f"{hour} o'clock")


def test_node_rejects_subtract_noise_without_a_gate():
    """subtract_noise with vad_mode='off' is incoherent and must not start.

    With no gate every frame lands in the speech pool, so the noise pool stays
    empty and the subtraction is a silent no-op. Better to refuse than to look
    like it is doing something.
    """
    rosmock.set_overrides(dict(source='topic', vad_mode='off',
                               subtract_noise=True))
    with pytest.raises(ValueError, match="subtract_noise"):
        D.DoaMod()


def test_node_rejects_nondefault_channel_count_without_geometry():
    """3 channels + built-in 4-mic geometry must fail loudly, not guess."""
    rosmock.set_overrides(dict(source='topic', vad_mode='off',
                               channels=[1, 2, 3]))
    with pytest.raises(ValueError, match="positions_mm"):
        D.DoaMod()


def test_topic_source_rejects_mismatched_channel_count():
    """A publisher with the wrong channel count must be dropped, not reshaped.

    Silently accepting it would pair the geometry with the wrong channel
    order, which produces confident wrong bearings.
    """
    node = _make_node()
    try:
        src = node.src
        msg = type('M', (), {})()
        msg.data = [0.0, 16000.0, 2.0, 0.0] + [0.1] * 1024   # 2ch, expected 4
        src._on_msg(msg)
        assert src.n_msgs == 0 and src.n_dropped == 1
        errs = [m for lv, m in node.get_logger().records if lv == 'error']
        assert any('2ch' in m for m in errs), "no diagnostic for the mismatch"
    finally:
        node.shutdown()


# ---------------------------------------------------------------------------
# 8. every frame is analysed, whatever the framing of the input
# ---------------------------------------------------------------------------
def test_ring_read_from_returns_the_samples_at_a_position():
    """Reading by absolute position must give the same samples as slicing the
    whole signal, across the wrap of the ring."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 6000))
    ring = D.RingBuffer(2, 1000)
    for a in range(0, 6000, 137):
        ring.push(x[:, a:a + 137])
        w = ring.written
        for start in (w - 1000, w - 613, w - 200):
            if start < 0:
                continue
            blk, used = ring.read_from(start, 200)
            assert used == start and np.array_equal(blk, x[:, start:start + 200])
    blk, used = ring.read_from(ring.written - 100, 200)        # not all there yet
    assert blk is None and used == ring.written - 100


def test_ring_grow_keeps_its_contents():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 2500))
    ring = D.RingBuffer(3, 1000)
    ring.push(x[:, :700]); ring.push(x[:, 700:1700])
    ring.grow(4000)
    assert ring.capacity == 4000 and ring.written == 1700
    ring.push(x[:, 1700:])
    blk, used = ring.read_from(700, 1800)
    assert used == 700 and np.array_equal(blk, x[:, 700:2500])
    ring.grow(500)                                             # never shrinks
    assert ring.capacity == 4000


def test_frame_reader_skips_ahead_when_too_far_behind_and_counts_it():
    """A reader that cannot keep up must not fall ever further behind: past
    max_lag it jumps to the newest block, in whole hops, and says how much it
    left out."""
    ring = D.RingBuffer(1, 16000)
    x = np.arange(12000, dtype=np.float64)[None, :]
    rd = D.FrameReader(ring, block=2048, hop=512, max_lag=4096)
    ring.push(x[:, :3000])
    assert np.array_equal(rd.next(), x[:, :2048]) and rd.skipped == 0
    ring.push(x[:, 3000:])                                     # 12000 written, reader at 512
    blk = rd.next()
    start = rd.pos - 512
    assert rd.skipped == start - 512 and rd.skipped % 512 == 0
    assert 12000 - (start + 2048) < 512, "did not jump to the newest complete block"
    assert np.array_equal(blk, x[:, start:start + 2048])


def _raw_msg(x, seq=0, fs=FS):
    """voice_mod's raw-tap message for a [M, n] signal."""
    msg = type('M', (), {})()
    msg.data = np.concatenate(([0.0, float(fs), float(x.shape[0]), float(seq)],
                               x.T.reshape(-1)))
    return msg


def _feed(node, x, frame_s):
    n = int(round(frame_s * FS))
    for k, a in enumerate(range(0, x.shape[1], n)):
        node.src._on_msg(_raw_msg(x[:, a:a + n], seq=k))
        while node._step():
            pass


@pytest.mark.parametrize("frame_s", [0.032, 0.1, 1.0, 10.0])
def test_node_analyses_every_frame_whatever_the_message_length(frame_s):
    """voice_mod publishes its raw tap in frames (10 s by default). The worker
    used to analyse the newest block of each message only: 5 blocks a minute
    out of 1875. Every hop of the audio must be analysed, and the bearing must
    be published at the same cadence of AUDIO time, for any frame length."""
    seconds = 20
    x = plane_wave(120.0, dc.array_positions(), n=FS * seconds, seed=5)
    node = _make_node(publish_period=0.2)
    try:
        _feed(node, x, frame_s)
        n_msg = -(-x.shape[1] // int(round(frame_s * FS)))
        assert node.src.n_msgs == n_msg and node.src.n_dropped == 0
        hops = (x.shape[1] - D.BLOCK) // D.HOP + 1
        assert node.est.n_frames == hops, \
            f"{node.est.n_frames} of {hops} blocks analysed with {frame_s} s frames"
        assert node.reader.skipped == 0
        sent = node.publishers_by_topic['/dorai_doa'].sent
        # 0.2 s rounds up to whole 32 ms hops: one bearing every 7 hops.
        cadence = -(-0.2 // (D.HOP / FS)) * D.HOP / FS
        assert abs(len(sent) - seconds / cadence) <= 2, \
            f"{len(sent)} bearings for {seconds} s of audio, one per {cadence:.3f} s expected"
        assert abs(wrap180(sent[-1].data - 120.0)) <= 6.0
        t = [m.data[0] for m in node.publishers_by_topic['/dorai_doa/state'].sent]
        assert np.all(np.diff(t) > 0.15) and np.all(np.diff(t) < 0.3), \
            "bearings are not spread evenly over the audio"
    finally:
        node.shutdown()


def test_topic_source_reports_long_frames_once():
    """Long frames make the bearing late, not wrong; say so, once."""
    node = _make_node()
    try:
        x = np.zeros((4, FS * 2))
        for k in range(3):
            node.src._on_msg(_raw_msg(x, seq=k))
        late = [m for lv, m in node.get_logger().records
                if lv == 'warn' and '2.0 s frames' in m]
        assert len(late) == 1 and 'publish_interval' in late[0]
        assert node.src.ring.capacity >= 2 * x.shape[1], "ring cannot hold two frames"
    finally:
        node.shutdown()


# ---------------------------------------------------------------------------
# 9. channel timing
# ---------------------------------------------------------------------------
def _timing_of(x, pos):
    est = D.BearingEstimator(pos, gate=D.SpeechGate('off'))
    chk = D.TimingCheck(pos)
    for a in range(0, x.shape[1] - D.BLOCK + 1, D.HOP):
        info = est.push(x[:, a:a + D.BLOCK])
        chk.update(est.last_est, info['active'])
    return chk, est


def test_timing_check_flags_a_late_channel():
    """One channel a sample late moves the bearing by ~15 deg and passes the
    closure gate. The plane-wave misfit is what shows it (4+ microphones)."""
    pos = dc.array_positions()
    x = plane_wave(200.0, pos, n=FS * 3, seed=7)
    ok, est_ok = _timing_of(x, pos)
    assert ok.able and not ok.suspect and ok.misfit_us < 5.0
    assert abs(wrap180(est_ok.bearing()[0] - 200.0)) <= 3.0

    late = x.copy()
    late[1] = _delay(x[1], 1.0)
    bad, est_bad = _timing_of(late, pos)
    assert bad.suspect, f"1 sample of lateness not flagged (misfit {bad.misfit_us:.1f} us)"
    assert 20.0 < bad.misfit_us < 32.0
    assert est_bad.consistent_frac > 0.9, "closure was expected to pass: it cannot see this"
    assert abs(wrap180(est_bad.bearing()[0] - 200.0)) > 8.0, \
        "the bearing was expected to be wrong here; the warning has a reason"


def test_timing_check_says_it_cannot_judge_three_microphones():
    """Three pair delays always fit a plane wave: no misfit, so no verdict --
    not a silent 'ok'."""
    tri = np.array([[0.06, 0.0], [-0.03, 0.052], [-0.03, -0.052]])
    x = plane_wave(80.0, tri, n=FS * 2, seed=8)
    x[1] = _delay(x[1], 2.0)
    chk, _ = _timing_of(x, tri)
    assert not chk.able and not chk.suspect and np.isnan(chk.misfit_us)


def _sync_msg(n_ch, n_dev, drift, trim_ppm=12.0, spread_us=300.0):
    msg = type('M', (), {})()
    msg.data = [0.0, float(n_ch), float(n_dev), float(drift), trim_ppm, spread_us]
    return msg


def test_node_warns_when_the_publisher_reports_separate_clocks():
    """voice_mod opens every microphone as its own device. It says so on
    <topic>/sync, and the node must turn that into a warning, a flag in the
    diagnostics line and a flag on the human topic."""
    node = _make_node()
    try:
        assert '/dorai_raw_audio/sync' in node.subscriptions_by_topic
        pos = dc.array_positions()
        for k in range(15):
            node.est.push(plane_wave(120.0, pos, seed=k))
        assert node._timing()[1] is None and node._timing()[0].startswith('ok')

        node.src._on_sync(_sync_msg(4, 1, 1))                  # one device: fine
        assert node._timing()[1] is None
        node.src._on_sync(_sync_msg(4, 4, 0))                  # declared one clock: fine
        assert node._timing()[1] is None

        node.src._on_sync(_sync_msg(4, 4, 1))                  # separate clocks
        label, problem = node._timing()
        assert label.startswith('SEPARATE-CLOCKS') and 'NOT TRUSTWORTHY' in problem
        node._last_diag = -1e9
        node._maybe_diag(1000.0)
        warns = [m for lv, m in node.get_logger().records if lv == 'warn']
        assert any('NOT TRUSTWORTHY' in m for m in warns)
        assert 'timing=SEPARATE-CLOCKS' in node.publishers_by_topic['/doa_mod/diagnostics'].sent[-1].data
        node._last_diag = -1e9
        node._maybe_diag(1010.0)                               # 10 s later: not repeated yet
        assert sum('NOT TRUSTWORTHY' in m for lv, m in node.get_logger().records
                   if lv == 'warn') == 1
        node._last_speech_t = 1.0
        node._maybe_publish(1.0)
        assert 'CHANNEL TIMING SUSPECT' in node.publishers_by_topic['/dorai_doa/human'].sent[-1].data
    finally:
        node.shutdown()


def test_node_flags_a_late_channel_on_the_topic():
    """End to end through the topic path: a late channel in the frames ends
    up as timing=SUSPECT in the diagnostics line."""
    pos = dc.array_positions()
    x = plane_wave(200.0, pos, n=FS * 4, seed=9)
    x[2] = _delay(x[2], 1.0)
    node = _make_node()
    try:
        _feed(node, x, 1.0)
        node._last_diag = -1e9
        node._maybe_diag(1000.0)
        assert 'timing=SUSPECT' in node.publishers_by_topic['/doa_mod/diagnostics'].sent[-1].data
        assert any('channel timing or geometry' in m
                   for lv, m in node.get_logger().records if lv == 'warn')
    finally:
        node.shutdown()
