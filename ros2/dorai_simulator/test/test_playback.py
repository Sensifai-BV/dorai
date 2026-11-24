"""dorai_simulator tests: the capture-device model and message layouts. No ROS needed.

    cd dorai && python -m pytest ros2/dorai_simulator/test -q
"""
import os
import sys

import numpy as np
import pytest

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(_PKG))
sys.path.insert(0, _PKG)

from dorai_simulator import playback_scheduler as devices          # noqa: E402

FS = 16000


def _tone(seconds=4.0, f=440.0):
    t = np.arange(int(seconds * FS)) / FS
    return np.stack([np.sin(2 * np.pi * f * t)] * 3)


def test_a_fast_device_delivers_more_samples_of_the_same_sound():
    """+100 ppm: 100 samples more per million, and the sound is the same sound
    (the tone the device 'hears' is still 440 Hz of real time)."""
    x = _tone(10.0)[0]
    y0 = devices.device_signal(x, FS, 48000, ppm=0.0)
    y1 = devices.device_signal(x, FS, 48000, ppm=100.0)
    assert len(y0) == 480000 and len(y1) == 480048
    # zero crossings per second of real time are unchanged: compare the phase at t = 9 s
    t0 = np.arange(len(y0)) / 48000.0
    t1 = np.arange(len(y1)) / (48000.0 * (1 + 100e-6))
    assert np.abs(y0[1000:-1000] - np.sin(2 * np.pi * 440 * t0[1000:-1000])).max() < 1e-3
    assert np.abs(y1[1000:-1000] - np.sin(2 * np.pi * 440 * t1[1000:-1000])).max() < 1e-3


def test_device_blocks_carry_start_offsets_drift_and_true_stamps():
    blocks = devices.device_blocks(_tone(4.0), FS, rate=48000, block_ms=100.0,
                                   ppm=[0.0, 50.0, -50.0], start_ms=[0.0, 1.0, 2.0])
    assert [b["due_s"] for b in blocks] == sorted(b["due_s"] for b in blocks)
    per = {i: [b for b in blocks if b["device"] == i] for i in range(3)}
    assert all(len(b["samples"]) == 4800 for b in blocks)
    assert [b["seq"] for b in per[1]] == list(range(len(per[1])))
    # the stream of device 1 was opened 1 ms late, and its clock is 50 ppm fast
    assert per[1][0]["stamp_s"] == pytest.approx(0.001, abs=1e-9)
    assert per[1][10]["stamp_s"] - per[1][0]["stamp_s"] == pytest.approx(1.0 / (1 + 50e-6), abs=1e-9)
    assert per[2][10]["stamp_s"] - per[2][0]["stamp_s"] == pytest.approx(1.0 / (1 - 50e-6), abs=1e-9)
    assert per[0][3]["due_s"] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        devices.device_blocks(_tone(1.0), FS, ppm=[0.0])


def test_device_message_is_what_voice_mod_parses():
    """The two ends of the simulated capture path must agree on the layout."""
    sys.path.insert(0, os.path.join(_REPO, "ros2", "voice_mod"))
    try:
        import voice as V
    except Exception as exc:                                        # no ROS / sounddevice here
        pytest.skip(f"voice_mod is not importable: {exc}")
    block = devices.device_blocks(_tone(1.0), FS, start_ms=[0.0, 1.0, 2.0])[4]
    t0 = 123456.789
    seq, rate, stamp, samples = V.parse_device_msg(devices.device_message(block, 48000, t0))
    assert (seq, rate) == (block["seq"], 48000)
    assert abs(stamp - (t0 + block["stamp_s"])) < 1e-4
    assert np.array_equal(samples, block["samples"])
    assert devices.HEADER == V.SIM_HEADER


def test_array_frames_use_the_raw_tap_layout():
    x = np.arange(4 * 3200, dtype=np.float32).reshape(4, 3200)
    frames = devices.array_frames(x, FS, frame_ms=100.0)
    assert len(frames) == 2
    due, data = frames[1]
    assert due == pytest.approx(0.2) and list(data[:4]) == [0.1, 16000.0, 4.0, 1.0]
    assert np.array_equal(data[4:].reshape(-1, 4).T, x[:, 1600:3200])


