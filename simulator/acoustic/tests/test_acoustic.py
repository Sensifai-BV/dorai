"""Virtual lab tests. No ROS; they need pyroomacoustics.

    cd dorai && python -m pytest simulator/acoustic/tests -q
"""
import json
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)
pytest.importorskip("pyroomacoustics")

from simulator.acoustic import config, hardware_model, microphone_array, robot, room_engine, scene_mixer, source_manager   # noqa: E402
from simulator.acoustic import generators                                                                              # noqa: E402

FS = 16000


def test_text_of_a_speech_file(tmp_path):
    assert source_manager.text_of("x/07_Artaban_come_here_raw16k.wav") == "Artaban come here"
    assert source_manager.text_of("x/481-123719-0000.wav") is None       # a corpus id is not a sentence
    wav = tmp_path / "a01.wav"
    (tmp_path / "a01.txt").write_text("Go to  welding station two\n")
    assert source_manager.text_of(str(wav)) == "Go to welding station two"


def test_scenario_defaults_are_filled_in_dict_by_dict():
    sc = config.merge(scene_mixer.DEFAULTS, dict(room="workshop", talker=dict(commands=2)))
    assert sc["room"] == "workshop" and sc["talker"]["commands"] == 2
    assert sc["talker"]["mouth_height"] == 1.6 and robot.load_robot("artaban")["height_m"] == 0.5


def test_every_scenario_file_is_valid():
    yaml = pytest.importorskip("yaml")
    folder = os.path.join(_REPO, "config", "scenarios")
    files = sorted(os.listdir(folder))
    assert len(files) >= 6
    for f in files:
        sc = yaml.safe_load(open(os.path.join(folder, f)))
        full = config.merge(scene_mixer.DEFAULTS, sc)
        assert room_engine.load_room(full["room"])["size_m"], f
        for a in (full["arrays"] or robot.load_robot()["array_names"]):
            assert microphone_array.load_array(a).shape[1] == 2, f
        for nz in full["noise"]:
            assert nz.get("type") in generators.NOISE or os.path.isfile(os.path.join(_REPO, nz["file"])), f


# ----------------------------------------------------------------- render
def test_render_gives_the_delays_and_levels_the_truth_states(tmp_path):
    """One command in a dry room, no noise: the respeaker channels must carry
    the plane-wave delays of the stated bearing, and the noise the stated SNR."""
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
              hardware=dict(gain_db_sd=0.0, position_error_mm=0.0, self_noise_snr_db=80.0), seed=3)
    signals, truth = scene_mixer.render(sc)
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
    want = (pos[0] - pos[2]) @ u / config.C_SOUND                  # capsule 0 hears it this much earlier
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
    assert np.allclose(microphone_array.load_array("respeaker_4mic"), dc.array_positions(0.032), atol=1e-5)


def test_truth_reports_the_snr_it_reached_and_keeps_clean_parts(tmp_path):
    """A target SNR is a request; the truth file must say what came out, and
    the parts must add up to the mix (they are the reference for SI-SDR)."""
    sf = pytest.importorskip("soundfile")
    rng = np.random.default_rng(1)
    sf.write(tmp_path / "Artaban_stop.wav", rng.standard_normal(FS) * 0.1, FS)
    sc = dict(room=dict(size_m=[6.0, 5.0, 3.0], rt60_s=0.3), talker=dict(commands=1, lead_in_s=0.5, gap_s=0.5),
              speech=dict(dir=str(tmp_path)), noise=[dict(type="ventilation_noise", snr_db=5.0)],
              hardware=dict(self_noise_snr_db=120.0, position_error_mm=0.0), seed=2)
    signals, truth = scene_mixer.render(sc, keep_parts=True)
    assert abs(truth["noise"][0]["snr_measured_db"] - 5.0) < 0.5 and abs(truth["snr_db"] - 5.0) < 0.5
    assert truth["robot"]["model"] == "artaban" and truth["robot"]["capture"]["device_rate"] == 48000
    for name in ("artaban_line3", "respeaker_4mic"):
        parts = signals[name + ".speech"].astype(np.float64) + signals[name + ".noise"]
        assert np.abs(parts - signals[name]).max() < 1e-4


@pytest.mark.parametrize("kind", sorted(generators.NOISE))
def test_noise_generators_are_reproducible_and_bounded(kind):
    a = generators.NOISE[kind](2.0, FS, seed=5)
    b = generators.NOISE[kind](2.0, FS, seed=5)
    c = generators.NOISE[kind](2.0, FS, seed=6)
    assert len(a) == 2 * FS and np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.isfinite(a).all() and 0.5 < np.abs(a).max() <= 1.0


def test_hardware_model_applies_each_imperfection_on_its_own():
    rng = np.random.default_rng(0)
    x = np.stack([np.sin(2 * np.pi * 440 * np.arange(FS) / FS) * 0.3] * 4)
    off = dict(hardware_model.DEFAULTS, gain_db_sd=0.0, self_noise_snr_db=200.0)
    out, rep = hardware_model.apply(x, off, rng, 0.045)
    assert np.allclose(out, x, atol=1e-6) and rep["scale"] == 1.0
    out, rep = hardware_model.apply(x, dict(off, channel_delay_samples=[0, 1.0, 0, 0]), rng, 0.045)
    assert np.allclose(out[1][10:-10], x[1][9:-11], atol=1e-6) and np.allclose(out[0], x[0], atol=1e-6)
    out, _ = hardware_model.apply(x, dict(off, quantization_bits=8), rng, 0.045)
    assert np.abs(out - x).max() <= 1 / 256 + 1e-9 and np.allclose(out * 128, np.round(out * 128))
    out, rep = hardware_model.apply(x * 5, dict(off, clip=True), rng, 0.045)
    assert np.abs(out).max() == 1.0 and rep["clipped_fraction"] > 0.5
    out, rep = hardware_model.apply(x, dict(off, gain_db_sd=1.0), rng, 0.045)
    assert np.allclose(out, x * 10 ** (np.array(rep["gain_db"])[:, None] / 20), atol=1e-6)
