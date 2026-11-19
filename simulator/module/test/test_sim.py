"""sim tests. No ROS needed; the render test needs pyroomacoustics and is
skipped without it.

    cd dorai && python -m pytest simulator/module/test -q
"""
import json
import os
import sys

import numpy as np
import pytest

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(_PKG))
sys.path.insert(0, _PKG)

from sim import acoustics, devices                             # noqa: E402

FS = 16000


def test_text_of_a_speech_file(tmp_path):
    assert acoustics.text_of("x/07_Artaban_come_here_raw16k.wav") == "Artaban come here"
    assert acoustics.text_of("x/481-123719-0000.wav") is None       # a corpus id is not a sentence
    wav = tmp_path / "a01.wav"
    (tmp_path / "a01.txt").write_text("Go to  welding station two\n")
    assert acoustics.text_of(str(wav)) == "Go to welding station two"


def test_scenario_defaults_are_filled_in_dict_by_dict():
    sc = acoustics.merge(acoustics.DEFAULTS, dict(room="workshop", talker=dict(commands=2)))
    assert sc["room"] == "workshop" and sc["talker"]["commands"] == 2
    assert sc["talker"]["mouth_height"] == 1.6 and sc["robot"]["height"] == 0.5


def test_every_scenario_file_is_valid():
    yaml = pytest.importorskip("yaml")
    folder = os.path.join(_REPO, "config", "scenarios")
    files = sorted(os.listdir(folder))
    assert len(files) >= 6
    for f in files:
        sc = yaml.safe_load(open(os.path.join(folder, f)))
        full = acoustics.merge(acoustics.DEFAULTS, sc)
        assert acoustics.load_room(full["room"])["size_m"], f
        for a in full["arrays"]:
            assert acoustics.load_array(a).shape[1] == 2, f
        for nz in full["noise"]:
            assert os.path.isfile(os.path.join(_REPO, nz["file"])), f"{f}: {nz['file']} is not in the repo"


# ---------------------------------------------------------------- devices
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


# ----------------------------------------------------------------- render
def test_render_gives_the_delays_and_levels_the_truth_states(tmp_path):
    """One command in a dry room, no noise: the respeaker channels must carry
    the plane-wave delays of the stated bearing, and the noise the stated SNR."""
    pytest.importorskip("pyroomacoustics")
    sf = pytest.importorskip("soundfile")
    rng = np.random.default_rng(0)
    t = np.arange(2 * FS) / FS
    speech = rng.standard_normal(len(t)) * (np.sin(2 * np.pi * 2 * t) > 0)      # bursts of noise
    sf.write(tmp_path / "Artaban_stop.wav", speech * 0.1, FS)
    sf.write(tmp_path / "hum.wav", rng.standard_normal(4 * FS) * 0.1, FS)
    sc = dict(name="t", room=dict(size_m=[8.0, 7.0, 3.5], rt60_s=0.15), robot=dict(place=[4.0, 3.5], heading_deg=30.0, height=1.6),
              talker=dict(commands=1, distance_m=[2.0, 2.0], mouth_height=1.6, lead_in_s=1.0, gap_s=1.0),
              speech=dict(dir=str(tmp_path), pattern="Artaban*.wav"),
              noise=[dict(file=str(tmp_path / "hum.wav"), snr_db=10.0, from_s=0.0)],
              mics=dict(gain_db_sd=0.0, position_error_mm=0.0, self_noise_snr_db=80.0), seed=3)
    signals, truth = acoustics.render(sc)
    assert set(signals) == {"artaban_line3", "respeaker_4mic"} and signals["artaban_line3"].shape[0] == 3
    cmd = truth["commands"][0]
    assert cmd["text"] == "Artaban stop" and cmd["elevation_deg"] == 0.0
    json.dumps(truth)                                                # plain data
    x = signals["respeaker_4mic"].astype(np.float64)
    a, b = int(cmd["t0"] * FS), int(cmd["t1"] * FS)
    quiet = x[0, int(0.1 * FS): int(0.9 * FS)]                        # lead-in: noise only
    snr = 10 * np.log10(np.mean(x[0, a:b] ** 2) / np.mean(quiet ** 2) - 1.0)   # over the command, pauses included
    assert abs(snr - 10.0) < 1.0
    # delays: cross-correlate opposite capsules (0 and 2), 64 mm apart
    pos = np.asarray(truth["arrays"]["respeaker_4mic"])
    u = np.array([np.cos(np.deg2rad(cmd["bearing_deg"])), np.sin(np.deg2rad(cmd["bearing_deg"]))])
    want = (pos[0] - pos[2]) @ u / acoustics.C_SOUND                  # capsule 0 hears it this much earlier
    seg = x[:, a:b]
    X0, X2 = np.fft.rfft(seg[0], 1 << 17), np.fft.rfft(seg[2], 1 << 17)
    R = X2 * np.conj(X0)
    cc = np.fft.irfft(R / np.maximum(np.abs(R), 1e-12), (1 << 17) * 8)
    lags = np.r_[0:200, -200:0]
    got = lags[int(np.argmax(np.r_[cc[:200], cc[-200:]]))] / (FS * 8.0)
    assert abs(got - want) < 15e-6, f"delay {got * 1e6:.0f} us, the bearing in truth.json implies {want * 1e6:.0f} us"


def test_respeaker_config_is_the_geometry_doa_mod_has_built_in():
    """config/arrays/respeaker_4mic.yaml and doa_mod must describe the same
    board: a table that is a mirror of the other gives bearings no offset
    corrects."""
    sys.path.insert(0, os.path.join(_REPO, "ros2", "doa_mod"))
    import doa_core as dc
    assert np.allclose(acoustics.load_array("respeaker_4mic"), dc.array_positions(0.032), atol=1e-5)
