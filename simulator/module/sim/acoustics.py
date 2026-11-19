"""Render what the robot's microphones would record in a room.

A scenario (a dict, normally from a YAML file in config/scenarios/) names a room, where
the robot stands, who speaks from where, and which noise comes from where. The
result is one multichannel signal per microphone array on the robot, sampled on
ONE clock at 16 kHz, plus the truth a test needs: which command was spoken
when, from which bearing and distance.

The acoustics are pyroomacoustics': image sources up to order 6 for the direct
sound and the early reflections (exact fractional delays, so the delays between
the capsules are right to a fraction of a sample), ray tracing for the late
reverberation, frequency-dependent wall materials, air absorption. Walls
scatter (SCATTER): a shoebox of perfect mirrors traps sound between its
parallel walls and rings for seconds, which a room with furniture or machines
in it does not.

What is NOT modelled: the robot's body (no shadowing or reflections from it),
the talker's directivity (a mouth radiates forward; here it is a point),
microphone directivity and frequency response, the robot's own motor noise
unless given as a noise source, and moving sources (a talker changes place
between commands, not while speaking).

Clocks are not part of the acoustics. Separate capture devices with their own
clocks are produced from this one-clock render by virtual_mic.py.
"""
from __future__ import annotations

import glob
import os

import numpy as np

FS = 16000
SCATTER = 0.25
C_SOUND = 343.0

# Arrays and rooms are not in this file: they are YAML files in the repo's
# config/ folder (config/arrays/*.yaml, config/rooms/*.yaml), so that a
# geometry or a room is changed without touching code. Positions are in the
# robot's frame, metres: x is the robot's heading (0 deg), y its left (90 deg),
# and bearings are counter-clockwise from x, as doa_mod measures them.


def config_dir():
    """The repo's config/ folder ($DORAI_CONFIG overrides it)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get("DORAI_CONFIG") or os.path.normpath(os.path.join(here, "..", "..", "..", "config"))


def _load_yaml(kind, name):
    import yaml
    path = name if os.path.isfile(str(name)) else os.path.join(config_dir(), kind, f"{name}.yaml")
    if not os.path.isfile(path):
        have = sorted(f[:-5] for f in os.listdir(os.path.join(config_dir(), kind)) if f.endswith(".yaml"))
        raise FileNotFoundError(f"no {kind[:-1]} '{name}' in {os.path.join(config_dir(), kind)} (there: {', '.join(have)})")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_array(name):
    """[M, 2] microphone positions (x, y in metres) in channel order, from
    config/arrays/<name>.yaml. A list of positions is passed through."""
    if not isinstance(name, str):
        return np.asarray(name, dtype=float)[:, :2]
    mics = sorted(_load_yaml("arrays", name)["array"]["microphones"], key=lambda m: m["channel"])
    return np.asarray([m["position_m"][:2] for m in mics], dtype=float)


def load_room(room):
    """A room description: a name in config/rooms, or a dict; a dict with a
    `name` starts from that file and overrides it ({name: rt60, rt60_s: 0.8})."""
    if isinstance(room, str):
        spec = dict(_load_yaml("rooms", room), name=room)
    else:
        spec = dict(room)
        if "name" in spec and "size_m" not in spec and "size" not in spec:
            spec = dict(_load_yaml("rooms", spec["name"]), **spec)
    spec.setdefault("name", "custom")
    spec["size_m"] = spec.pop("size", None) or spec["size_m"]
    if "rt60" in spec:
        spec["rt60_s"] = spec.pop("rt60")
    return spec


DEFAULTS = dict(
    room="workshop",
    robot=dict(place="open", height=0.5, heading_deg=None),     # place: open | wall | [x, y]
    arrays=["artaban_line3", "respeaker_4mic"],
    talker=dict(commands=6, distance_m=[1.0, 4.0], bearing_deg=None, mouth_height=1.6, moves=True, gap_s=2.5, lead_in_s=3.0),
    speech=dict(dir=None, files=None, level_dbfs=-26.0),
    noise=[],
    mics=dict(gain_db_sd=0.5, position_error_mm=1.0, self_noise_snr_db=40.0),
    seed=0,
)


def merge(base, over):
    """`over` on top of `base`, dict by dict."""
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


# ------------------------------------------------------------------ sources
def text_of(path):
    """What is said in a speech file: `<file>.txt` next to it, else the file
    name ('07_Artaban_come_here_raw16k.wav' -> 'Artaban come here'), else None
    for names that are not words (corpus ids such as 481-123719-0000)."""
    side = os.path.splitext(path)[0] + ".txt"
    if os.path.isfile(side):
        return " ".join(open(side, encoding="utf-8").read().split())
    words = [w for w in os.path.splitext(os.path.basename(path))[0].replace("-", "_").split("_")
             if w and not w.isdigit() and w.lower() not in ("raw16k", "raw", "16k", "clean")]
    if not words or not all(w.isalpha() for w in words):
        return None
    return " ".join(words)


def read_mono(path, fs=FS):
    import soundfile as sf
    x, rate = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if rate != fs:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(rate), fs)
        x = resample_poly(x, fs // g, int(rate) // g)
    return x


def speech_files(spec, root="."):
    if spec.get("files"):
        files = [f if os.path.isabs(f) else os.path.join(root, f) for f in spec["files"]]
    else:
        d = spec.get("dir") or os.environ.get("DORAI_SIM_SPEECH") or os.path.join(root, "simulator", "lab", "samples", "speech")
        d = d if os.path.isabs(d) else os.path.join(root, d)
        files = sorted(glob.glob(os.path.join(d, spec.get("pattern", "*.wav"))))
    if not files:
        raise FileNotFoundError(f"no speech files for {spec}")
    return files


def trim(x, pad=0.05, fs=FS):
    """Cut the silence before and after the speech, keeping `pad` seconds.
    Speech is where the level of a 20 ms window is within 30 dB of the loudest
    one (single samples are no guide: a recording's noise floor has peaks)."""
    w = int(0.02 * fs)
    n = len(x) // w
    if n < 3:
        return x
    level = np.sqrt(np.mean(x[:n * w].reshape(n, w) ** 2, axis=1))
    act = np.flatnonzero(level > level.max() * 10 ** (-30 / 20))
    return x[max(0, act[0] * w - int(pad * fs)): (act[-1] + 1) * w + int(pad * fs)]


# -------------------------------------------------------------------- room
def make_room(spec, mics_xyz):
    """The pyroomacoustics room for a description from load_room().

    Three levels, by what the description holds:
      anechoic: true         free field, the direct sound only
      rt60_s: 0.4            image sources only, one absorption for all walls from Sabine's formula
      materials: {...}       image sources to order 6 plus ray tracing, a material per wall, air absorption
    """
    import pyroomacoustics as pra
    size = np.asarray(spec["size_m"], dtype=float)
    scatter = float(spec.get("scatter", SCATTER))
    if spec.get("anechoic"):
        r = pra.ShoeBox(size, fs=FS, materials=pra.Material(1.0), max_order=0)
    elif "materials" in spec:
        mats = {k: pra.Material(v, scatter) for k, v in spec["materials"].items()}
        r = pra.ShoeBox(size, fs=FS, materials=mats, max_order=int(spec.get("max_order", 6)),
                        ray_tracing=True, air_absorption=True)
        r.set_ray_tracing(receiver_radius=0.3, n_rays=int(spec.get("n_rays", 20000)))
    else:
        absorption, order = pra.inverse_sabine(float(spec["rt60_s"]), size)
        trace = bool(spec.get("ray_tracing", False))
        r = pra.ShoeBox(size, fs=FS, materials=pra.Material(absorption, scatter if trace else None),
                        max_order=int(spec.get("max_order", order)), ray_tracing=trace,
                        air_absorption=bool(spec.get("air_absorption", False)))
        if trace:
            r.set_ray_tracing(receiver_radius=0.3, n_rays=int(spec.get("n_rays", 20000)))
    r.add_microphone_array(np.asarray(mics_xyz).T)
    return r, size


def convolve(x, irs, lag):
    from scipy.signal import fftconvolve
    return np.stack([fftconvolve(x, ir)[lag:lag + len(x)] for ir in irs])


def rt60_of(ir, fs=FS):
    """Reverberation time from one impulse response (Schroeder, -5 to -25 dB, times 3)."""
    e = np.cumsum((ir ** 2)[::-1])[::-1]
    db = 10 * np.log10(e / e[0] + 1e-30)
    a, b = np.argmax(db <= -5), np.argmax(db <= -25)
    return float(3.0 * (b - a) / fs) if b > a else float("nan")


def render(scenario, root="."):
    """Render a scenario. Returns (signals, truth): signals[array] is float32
    [M, n] at FS; truth is a plain dict (JSON-ready)."""
    sc = merge(DEFAULTS, scenario)
    rng = np.random.default_rng(int(sc["seed"]))
    room_spec = load_room(sc["room"])
    size = np.asarray(room_spec["size_m"], dtype=float)
    rob, tk = sc["robot"], sc["talker"]

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
    c3 = np.array([centre[0], centre[1], float(rob["height"])])

    def bearing_of(p):                                       # in the robot's frame
        return float((np.rad2deg(np.arctan2(p[1] - centre[1], p[0] - centre[0])) - heading) % 360.0)

    arrays = {(a if isinstance(a, str) else f"array{i}"): load_array(a) for i, a in enumerate(sc["arrays"])}
    spans, mics = {}, []
    for name, pos in arrays.items():
        err = rng.normal(0, sc["mics"]["position_error_mm"] / 1000.0, pos.shape)
        world = (pos + err) @ R.T
        spans[name] = (len(mics), len(mics) + len(pos))
        mics += [[centre[0] + w[0], centre[1] + w[1], c3[2]] for w in world]

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
    spots = spots if tk["moves"] else spots * n_cmd
    noise_spots = []
    for nz in sc["noise"]:
        if nz.get("position") is not None:
            p = np.asarray(nz["position"], dtype=float)
        else:
            far = min(2.0, 0.45 * float(np.min(size[:2])))
            while True:
                p = np.array([rng.uniform(0.4, size[0] - 0.4), rng.uniform(0.4, size[1] - 0.4), float(nz.get("height", 1.0))])
                if np.linalg.norm(p[:2] - centre) > far:
                    break
        noise_spots.append(p)

    room, _ = make_room(room_spec, mics)
    uniq = [spots[0]] if not tk["moves"] else spots
    for p in uniq + noise_spots:
        room.add_source(p)
    room.compute_rir()
    M = len(mics)
    rir = lambda s: [np.asarray(room.rir[m][s]) for m in range(M)]

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
        s = k if tk["moves"] else 0
        irs = rir(s)
        lag = min(int(np.argmax(np.abs(ir))) for ir in irs)        # drop the common travel time, keep the differences
        dry = np.zeros(len(c) + FS)
        dry[:len(c)] = c
        y = convolve(dry, irs, lag)
        y = y[:, :total - t]
        speech[:, t:t + y.shape[1]] += y
        active[t:t + len(c)] = True
        p = spots[k]
        commands.append(dict(text=text_of(f), file=os.path.basename(f), t0=round(t / FS, 3), t1=round((t + len(c)) / FS, 3),
                             bearing_deg=round(bearing_of(p), 1), distance_m=round(float(np.linalg.norm(p[:2] - centre)), 2),
                             elevation_deg=round(float(np.rad2deg(np.arctan2(p[2] - c3[2], np.linalg.norm(p[:2] - centre)))), 1)))
        t += len(c) + gap
    ref = 0                                                         # levels are set at the first microphone
    speech *= level / np.sqrt(np.mean(speech[ref][active] ** 2))    # the commands arrive at level_dbfs on average
    p_speech = float(np.mean(speech[ref][active] ** 2))

    # ---- the noise
    mix, noises = speech.copy(), []
    for j, (nz, p) in enumerate(zip(sc["noise"], noise_spots)):
        src = nz["file"] if os.path.isabs(nz["file"]) else os.path.join(root, nz["file"])
        n = read_mono(src)
        n = np.tile(n, int(np.ceil((total + FS) / len(n))) + 1)
        a = int(rng.integers(0, len(n) - total - 1))
        irs = rir(len(uniq) + j)
        lag = min(int(np.argmax(np.abs(ir))) for ir in irs)
        y = convolve(n[a:a + total], irs, lag)
        y *= np.sqrt(p_speech / np.mean(y[ref] ** 2)) * 10 ** (-float(nz.get("snr_db", 0.0)) / 20)
        t_on, t_off = float(nz.get("from_s", 0.0)), nz.get("until_s")
        gate = np.zeros(total)
        gate[int(t_on * FS): total if t_off is None else int(float(t_off) * FS)] = 1.0
        mix += y * gate
        noises.append(dict(file=os.path.basename(src), bearing_deg=round(bearing_of(p), 1),
                           distance_m=round(float(np.linalg.norm(p[:2] - centre)), 2), snr_db=float(nz.get("snr_db", 0.0)),
                           from_s=t_on, until_s=t_off))

    # ---- the microphones themselves
    mk = sc["mics"]
    mix *= 10 ** (rng.normal(0, mk["gain_db_sd"], (M, 1)) / 20)
    mix += rng.standard_normal(mix.shape) * np.sqrt(p_speech) * 10 ** (-float(mk["self_noise_snr_db"]) / 20)
    peak = float(np.abs(mix).max())
    scale = 0.95 / peak if peak > 0.95 else 1.0
    signals = {name: (mix[a:b] * scale).astype(np.float32) for name, (a, b) in spans.items()}
    truth = dict(
        name=sc.get("name", "scene"), fs=FS, seconds=round(total / FS, 2), seed=int(sc["seed"]),
        room=dict(name=room_spec["name"], size=size.tolist(), rt60_target_s=room_spec.get("rt60_s"),
                  rt60_s=round(float(np.nanmean([rt60_of(ir) for ir in rir(0)])), 2)),
        robot=dict(position=[round(float(v), 2) for v in centre], height=float(rob["height"]), heading_deg=round(heading, 1), place=str(place)),
        arrays={name: np.round(pos, 4).tolist() for name, pos in arrays.items()},
        commands=commands, noise=noises, scale=scale)
    return signals, truth
