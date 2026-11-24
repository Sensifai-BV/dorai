"""Scenario -> the signal at every microphone of the robot, plus the truth.

A scenario (a dict, normally a YAML file in config/scenarios) names a room,
where the robot stands, who speaks from where, and which noise comes from
where. Every source gets its own room impulse response to every microphone;
the result is, per array on the robot, one multichannel signal on ONE clock at
16 kHz. Separate capture devices with their own clocks are made from that by
ros2/dorai_simulator when the scene is played.

    x_m(t) = sum_k h_mk(t) * s_k(t) + n_m(t)

Kept for evaluation: the command text, time, bearing, distance and elevation;
each noise source's bearing and level; the reverberation time measured from
the impulse responses; and, with keep_parts, each source's own contribution at
the microphones (the clean reference for SNR and SI-SDR).

Not modelled: the robot's body (no shadowing), talker and microphone
directivity, microphone frequency response, sources that move while sounding
(a talker changes place between commands).
"""
from __future__ import annotations

import os

import numpy as np

from . import hardware_model
from .config import FS, merge
from .microphone_array import load_array
from .rir_generator import impulse_responses, propagate
from .robot import load_robot
from .room_engine import load_room, rt60_of
from .source_manager import noise_signal, read_mono, speech_files, text_of, trim

DEFAULTS = dict(
    room="workshop",
    robot=dict(model="artaban", place="open", height=None, heading_deg=None),     # place: open | wall | [x, y]
    arrays=None,                                                                  # default: the robot's arrays
    talker=dict(commands=6, distance_m=[1.0, 4.0], bearing_deg=None, mouth_height=1.6, moves=True, gap_s=2.5, lead_in_s=3.0),
    speech=dict(dir=None, files=None, level_dbfs=-26.0),
    noise=[],
    hardware=None,                                                                # default: the robot's
    seed=0,
)


def render(scenario, root=".", keep_parts=False):
    """Render a scenario. Returns (signals, truth): signals[array] is float32
    [M, n] at FS; truth is a plain dict (JSON-ready). With keep_parts, signals
    also holds "<array>.speech" and "<array>.noise": the two parts of the mix
    as the hardware delivers them (same gains and scale, no self-noise)."""
    sc = merge(DEFAULTS, scenario)
    if "mics" in sc:                                         # the earlier name of the hardware block
        sc["hardware"] = merge(sc.get("hardware") or {}, sc.pop("mics"))
    rng = np.random.default_rng(int(sc["seed"]))
    room_spec = load_room(sc["room"])
    size = np.asarray(room_spec["size_m"], dtype=float)
    rob, tk = sc["robot"], sc["talker"]
    model = load_robot(rob.get("model") or "artaban")
    hw = merge(merge(hardware_model.DEFAULTS, model["hardware"]), sc.get("hardware") or {})
    height = float(model.get("height_m", 0.5) if rob.get("height") is None else rob["height"])

    # ---- where the robot stands and which way it faces
    place = rob["place"]
    if place == "open":
        centre = np.array([rng.uniform(0.35, 0.65) * size[0], rng.uniform(0.35, 0.65) * size[1]])
    elif place == "wall":                                   # parked 0.4 m from the west wall
        centre = np.array([0.4, rng.uniform(0.3, 0.7) * size[1]])
    else:
        centre = np.asarray(place, dtype=float)
    heading = float(rng.uniform(0, 360) if rob.get("heading_deg") is None else rob["heading_deg"])
    cs, sn = np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))
    R = np.array([[cs, -sn], [sn, cs]])

    def bearing_of(p):                                       # in the robot's frame
        return float((np.rad2deg(np.arctan2(p[1] - centre[1], p[0] - centre[0])) - heading) % 360.0)

    names = sc["arrays"] or model["array_names"]
    arrays = {(a if isinstance(a, str) else f"array{i}"): load_array(a) for i, a in enumerate(names)}
    spans, mics = {}, []
    for name, pos in arrays.items():
        world = hardware_model.perturb_positions(pos, hw, rng) @ R.T
        spans[name] = (len(mics), len(mics) + len(pos))
        mics += [[centre[0] + w[0], centre[1] + w[1], height] for w in world]
    M = len(mics)

    # ---- where the talker stands for each command, and where the noise is
    n_cmd = int(tk["commands"])
    spots = []
    while len(spots) < (n_cmd if tk["moves"] else 1):
        fixed = tk.get("bearing_deg") is not None
        az = float(tk["bearing_deg"]) if fixed else rng.uniform(0, 360)
        d = rng.uniform(*tk["distance_m"])
        p = np.array([centre[0] + d * np.cos(np.deg2rad(az + heading)), centre[1] + d * np.sin(np.deg2rad(az + heading)),
                      float(tk["mouth_height"])])
        if np.all(p[:2] > 0.3) and np.all(p[:2] < size[:2] - 0.3):
            spots.append(p)
        elif fixed and tk["distance_m"][0] == tk["distance_m"][1]:
            raise ValueError(f"a talker at {az} deg, {d} m from the robot at {centre.tolist()} is outside the room {size.tolist()}")
    uniq = list(spots)
    spots = spots if tk["moves"] else spots * n_cmd
    noise_specs = list(sc["noise"]) + [dict(s, on_robot=True) for s in model["self_noise"]]
    noise_spots = []
    for nz in noise_specs:
        if nz.get("on_robot"):
            off = np.asarray(nz.get("offset_m", [0.1, 0.0, -0.1]), dtype=float)
            p = np.array([centre[0], centre[1], height]) + np.r_[R @ off[:2], off[2]]
        elif nz.get("position") is not None:
            p = np.asarray(nz["position"], dtype=float)
        elif nz.get("bearing_deg") is not None:
            d = float(nz.get("distance_m", 2.0))
            a = np.deg2rad(float(nz["bearing_deg"]) + heading)
            p = np.array([centre[0] + d * np.cos(a), centre[1] + d * np.sin(a), float(nz.get("height", 1.0))])
        else:
            far = min(2.0, 0.45 * float(np.min(size[:2])))
            while True:
                p = np.array([rng.uniform(0.4, size[0] - 0.4), rng.uniform(0.4, size[1] - 0.4), float(nz.get("height", 1.0))])
                if np.linalg.norm(p[:2] - centre) > far:
                    break
        noise_spots.append(p)

    rirs = impulse_responses(room_spec, mics, uniq + noise_spots)

    # ---- the commands
    files = speech_files(sc["speech"], root)
    files = [files[i] for i in rng.permutation(len(files))[:n_cmd]] if len(files) >= n_cmd \
        else [files[i % len(files)] for i in range(n_cmd)]
    level = 10 ** (float(sc["speech"]["level_dbfs"]) / 20)
    clips = []
    for f in files:
        x = trim(read_mono(f))
        clips.append(x / np.sqrt(np.mean(x ** 2)) * level)
    gap, t = int(tk["gap_s"] * FS), int(tk["lead_in_s"] * FS)
    total = t + sum(len(c) + gap for c in clips) + FS
    speech = np.zeros((M, total))
    commands, active = [], np.zeros(total, bool)
    for k, (c, f) in enumerate(zip(clips, files)):
        dry = np.zeros(len(c) + FS)                           # room for the reverberant tail
        dry[:len(c)] = c
        y = propagate(dry, rirs[k if tk["moves"] else 0])[:, :total - t]
        speech[:, t:t + y.shape[1]] += y
        active[t:t + len(c)] = True
        p = spots[k]
        dist = float(np.linalg.norm(p[:2] - centre))
        commands.append(dict(text=text_of(f), file=os.path.basename(f), t0=round(t / FS, 3), t1=round((t + len(c)) / FS, 3),
                             bearing_deg=round(bearing_of(p), 1), distance_m=round(dist, 2),
                             elevation_deg=round(float(np.rad2deg(np.arctan2(p[2] - height, dist))), 1)))
        t += len(c) + gap
    ref = 0                                                         # levels are set at the first microphone
    speech *= level / np.sqrt(np.mean(speech[ref][active] ** 2))    # the commands arrive at level_dbfs on average
    p_speech = float(np.mean(speech[ref][active] ** 2))

    # ---- the noise
    noise, noises = np.zeros((M, total)), []
    for j, (nz, p) in enumerate(zip(noise_specs, noise_spots)):
        n, label = noise_signal(nz, total, rng, root)
        y = propagate(n, rirs[len(uniq) + j])
        y *= np.sqrt(p_speech / np.mean(y[ref] ** 2)) * 10 ** (-float(nz.get("snr_db", 0.0)) / 20)
        t_on, t_off = float(nz.get("from_s", 0.0)), nz.get("until_s")
        gate = np.zeros(total)
        gate[int(t_on * FS): total if t_off is None else int(float(t_off) * FS)] = 1.0
        noise += y * gate
        on = gate.astype(bool) & active
        noises.append(dict(source=label, on_robot=bool(nz.get("on_robot")), bearing_deg=round(bearing_of(p), 1),
                           distance_m=round(float(np.linalg.norm(p[:2] - centre)), 2), snr_db=float(nz.get("snr_db", 0.0)),
                           from_s=t_on, until_s=t_off,
                           snr_measured_db=round(float(10 * np.log10(np.mean(speech[ref][on] ** 2) / max(np.mean((y * gate)[ref][on] ** 2), 1e-20))), 1) if on.any() else None))

    # ---- the microphones themselves
    hw_rng = np.random.default_rng(int(sc["seed"]) + 7919)
    state = hw_rng.bit_generator.state
    mix, report = hardware_model.apply(speech + noise, hw, hw_rng, p_speech)
    signals = {name: mix[a:b].astype(np.float32) for name, (a, b) in spans.items()}
    if keep_parts:
        g = 10 ** (np.asarray(report["gain_db"])[:, None] / 20) * report["scale"]
        for part, x in (("speech", speech), ("noise", noise)):
            for name, (a, b) in spans.items():
                signals[f"{name}.{part}"] = (x * g)[a:b].astype(np.float32)
    p_noise = float(np.mean(noise[ref][active] ** 2))
    truth = dict(
        name=sc.get("name", "scene"), fs=FS, seconds=round(total / FS, 2), seed=int(sc["seed"]),
        room=dict(name=room_spec["name"], size=size.tolist(), rt60_target_s=room_spec.get("rt60_s"),
                  rt60_s=round(float(np.nanmean([rt60_of(ir) for ir in rirs[0]])), 2)),
        robot=dict(model=model.get("name", "custom"), position=[round(float(v), 2) for v in centre], height=height,
                   heading_deg=round(heading, 1), place=str(place), capture=model["capture"],
                   arrays=[a for a in model.get("arrays", []) if a["name"] in arrays]),
        arrays={name: np.round(pos, 4).tolist() for name, pos in arrays.items()},
        commands=commands, noise=noises,
        snr_db=round(float(10 * np.log10(p_speech / p_noise)), 1) if p_noise > 0 else None,
        snr_convention="power of the commands over power of all noise, after the room, at microphone 1 of the first array, while commands are spoken",
        hardware=dict(settings=hw, applied=report), scale=report["scale"])
    return signals, truth
