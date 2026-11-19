# doa_mod — direction of arrival

Turns raw microphone-array audio into a talker bearing and publishes it.

```
mic array (M mics, ONE clock)
   -> ring buffer -> 2048-sample block every 512-sample hop
   -> VAD gate  [neural | energy | off]
   -> pooled SRP-PHAT over speech frames
   -> bearing in degrees -> /dorai_doa
```

| Topic | Type | Contents |
| :--- | :--- | :--- |
| `/dorai_doa` | `std_msgs/Float32` | azimuth in degrees, 0–360 |
| `/dorai_doa/state` | `std_msgs/Float32MultiArray` | `[elapsed_s, azimuth_deg, sharpness, consistent_frac, speech_flag, seq]` |
| `/dorai_doa/human` | `std_msgs/String` | the same state in words, for `ros2 topic echo` |
| `/doa_mod/diagnostics` | `std_msgs/String` | health line every `diag_period` seconds |

### Reading it from a terminal

```console
$ ros2 topic echo /dorai_doa
data: 268.0

$ ros2 topic echo /dorai_doa/human
data: "bearing 268.0 deg  |  3 o'clock from the array's 0 deg mark  |  confidence 0.62  |  frame gate 100%  |  t    9.4s  seq 16"

$ ros2 topic echo /dorai_doa/state
data: [9.41, 268.0, 0.625, 0.996, 1.0, 16.0]
```

`/dorai_doa` already reads plainly — `data: 268.0` is 268 degrees. The *state*
topic is the one that needs a decoder ring, since a `Float32MultiArray` echoes
as a bare list and you have to know index 1 is the bearing. `/dorai_doa/human`
exists for that: same numbers, spelled out. Turn it off with
`-p publish_human:=false`.

**The human topic's text format is not a contract** — it is for eyes, and it
may change. Machine consumers should subscribe to `/dorai_doa` (one float) or
`/dorai_doa/state` (fixed field order).

The o'clock hint is measured from the board's own `DOA:0` silkscreen mark and
runs **counter-clockwise**, so 90° is 9 o'clock, not 3. A clock face is used
rather than "front/left/right" because the node cannot know which way the
array is bolted to the robot; if you mount it at a fixed yaw, add that offset
in your consumer.

Bearings are published at `publish_period` (default 0.2 s ≈ 5 Hz) **while
speech is present**, and stop `vad_hangover_ms` after the last speech frame.
**Treat "no message" as "no talker"** — a stale bearing republished during
silence would be the room's noise field presented as a measurement.

## Accuracy

Scored on the 2026-08-12 hardware session (ReSpeaker 4-Mic, eight marked
talker positions at 1.5 m), raw reported bearing, **no fitted rotation**:

| VAD mode | mean abs error | max | excl. 315° mark |
| :--- | ---: | ---: | ---: |
| `neural` | **6.1°** | 21° | 4.0° mean, 8° max |
| `energy` | 5.9° | 20° | 3.9° mean, 8° max |
| `off` | 7.1° | 21° | 5.1° mean, 11° max |

Pass bar is ≤15° mean, ≤25° max. The 21° outlier is the 315° mark, which has a
known reflector; it is a room property, not an estimator failure. Reproduce
with:

```bash
python3 test/replay_session.py          # ~4 min, writes doa_mod_scores.csv
python3 test/make_validation_figure.py  # writes doa_mod_validation.png
```

### In simulation and on other real recordings (2026-11-19)

Simulated with `simulator/acoustic` (pyroomacoustics), scored by
`experiments/doa_benchmark.py` against pyroomacoustics' own SRP-PHAT and MUSIC,
which are told when each command is spoken; the node detects it itself.

| Condition | doa_mod | SRP-PHAT | MUSIC |
| :--- | ---: | ---: | ---: |
| Free field, 8 azimuths (0, 45, ... 315°), 2 m | 0.0° | 0.1° | 0.0° |
| Rooms, RT60 0.2 / 0.6 / 1.0 s (target), quiet, MAE | 0.1 / 2.7 / 3.5° | 0.3 / 1.7 / 2.2° | 1.2 / 3.4 / 2.9° |
| Welding from one point, RT60 0.4 s, SNR 20 / 0 / −10 dB, MAE | 23 / 38 / 61° | 28 / 41 / 59° | 1.4 / 4.1 / 4.1° |

* The coordinate convention and channel map agree with the simulator in all
  eight directions.
* **Directional noise is the weak point**, and it is the method's, not the
  node's: PHAT gives every frequency the same vote, so the bins the noise owns
  outvote the speech even at 20 dB. MUSIC over the whole command does not have
  this problem in that idealised test (perfect microphones, image sources
  only). In the more realistic scenes (wall materials, ray-traced tail, 1 mm
  and 0.5 dB microphone errors, a talker who moves between commands, the node's
  own gate) a MUSIC pool inside the node was only slightly better: 78 % of
  last bearings within 15° against 75 %. No estimator change was made on that
  evidence at first; the whole-utterance version that followed is `method:=music`, below.
* **`subtract_noise` should stay off in rooms.** It helped in free field and
  made things worse with reverberation (65 % within 15° against 75 %).
* Two faults found this way are fixed in the node: the neural gate going deaf
  on a long stream (`vad_reset_s`; 54 of 144 simulated commands had no bearing,
  now 1) and the first bearings of a command pointing at the previous talker
  (`utterance_gap_s`; first bearing within 15°: 17 % -> 62 %).

**`method:=music`** (normalised MUSIC on the covariance of the utterance, in the
node, with the node's own gate). Last bearing of each command within 15° of the
talker, 24 simulated room scenes (144 commands, each from another place):

| | `srp_phat` | `music` |
| :--- | ---: | ---: |
| all | 75 % | 83 % |
| quiet lab | 100 % | 100 % |
| welding 10 dB below speech | 89 % | 94 % |
| welding as loud as speech, factory hall | 50 % | 78 % |
| welding + machinery | 72 % | 89 % |
| welding 5 dB above speech | 65 % | 59 % |
| robot at a wall, quiet | 56 % | 67 % |

In the benchmark's noise matrix (welding, RT60 0.4 s) the mean error with
`music` is 6° at 5 dB and 17° at 0 dB (`srp_phat`: 62° and 38°), and no better
than `srp_phat` at −5 and −10 dB: there the gate passes the noise as speech
(about 11 bearings a minute while nobody speaks) and the covariance fills with
it. On the LOCATA recordings, which are quiet and have talkers who walk while
talking, `music` is slightly worse (median 14.9° against 13.7°, 50 % against
54 % within 15°). So the default stays `srp_phat`; use `music` where a noise
source from one direction is the problem, and treat it as unvalidated until it
has been scored on the robot.

Real recordings, LOCATA corpus (`experiments/locata_doa.py`), four equatorial
capsules of the Eigenmike as a planar array: median error 13.7° over 708
bearings (loudspeaker standing still 10.5°, walking talker 19°, walking talker
with moving array 8.7°). pyroomacoustics' SRP-PHAT and MUSIC show the same
+13° to +20° offsets on the static recordings, so this is the capsules sitting
on a rigid sphere (and the rooms), not the node. It is not a statement about
the ReSpeaker.

## Geometry — read this before deploying

The default is the ReSpeaker 4-Mic Array: `ch1..ch4` at 45/135/225/315°,
counter-clockwise, 32 mm radius.

**A mirrored channel map is not a rotation and cannot be calibrated away.** It
reports plausible bearings that are wrong everywhere except on the mirror axis.
This fault shipped undetected for months in the research tree, precisely
because the 0° and 180° test marks lie *on* the mirror axis, so a rotation-only
fit reported a clean offset and hid it.

> **Bring-up check:** put a talker at 90°, then at 0°. If 90° is right and 0°
> is wrong, you have a mirror, not an offset. Fix the channel order or the
> geometry — do not add a correction constant.

For other arrays give the geometry explicitly, in millimetres, in the same
order as `channels`:

```bash
ros2 run doa_mod doa --ros-args \
  -p channels:="[1,2,3,4]" \
  -p positions_mm:="0,32 -32,0 0,-32 32,0"
```

Constraints, all enforced at startup: ≥3 microphones, not collinear, count
matching `channels`, and the analysis band kept below the array's spatial
aliasing limit (a wider array aliases at a *lower* frequency).

**There is no auto-calibration, deliberately.** The research tree's MDS-based
`--calibrate` recovers geometry up to rotation *and reflection*, never resolves
the reflection, and measured 87° mean error on this same session. Measuring
the array once and typing four numbers is both faster and correct.

## VAD modes

`-p vad_mode:=` selects the gate; there is no silent fallback.

* **`neural`** (default) — `vad.onnx`, streaming, one 512-sample chunk per hop.
  Best tracking. Needs `onnxruntime`.
* **`energy`** — adaptive RMS percentile. No model, no `onnxruntime`. Accuracy
  is comparable; it admits far fewer frames, so it settles more slowly.
* **`off`** — no gating; every frame moves the bearing.

Choosing `off` is legitimate, not a debug hack: pooled SRP is markedly
tolerant of bad frames, and it drops the `onnxruntime` dependency entirely.
What you give up is **tracking** and **noise subtraction**:

| mode | accuracy (8 marks) | settling on the walk take |
| :--- | ---: | ---: |
| `neural` | 6.1° | 6.1 s |
| `energy` | 5.9° | 8.9 s |
| `off` | 7.1° | **never converges** |

That last row is the argument for keeping a gate on when the talker moves.
Ungated, every silent and reverberant frame pools in equally, and on a
walk-through take the bearing never settles. `subtract_noise:=true` requires a
gate and is refused with `vad_mode:=off`.

## Parameters

| Parameter | Default | Notes |
| :--- | :--- | :--- |
| `source` | `mic` | `mic` (own the device) or `topic` |
| `device` | `-1` | sounddevice index; -1 = default input |
| `device_channels` | `6` | channels to open (ReSpeaker exposes 6) |
| `channels` | `[1,2,3,4]` | which of them are the raw capsules |
| `fs` | `16000` | 0 = adopt the device's native rate |
| `input_topic` | `/dorai_raw_audio` | with `source:=topic` |
| `radius_mm` | `32.0` | built-in ReSpeaker layout |
| `positions_mm` | `""` | explicit `x,y ...` per mic, overrides `radius_mm` |
| `vad_mode` | `neural` | `neural` \| `energy` \| `off` |
| `vad_threshold` | `0.45` | neural mode |
| `vad_percentile` | `70.0` | energy mode |
| `vad_hangover_ms` | `800` | how long publishing outlives the last speech frame |
| `avg_tc` | `0.5` | pooling time constant, s. Longer = steadier, slower |
| `subtract_noise` | `false` | subtract a stationary interferer; needs a gate; `srp_phat` only, and worse in reverberant rooms |
| `method` | `srp_phat` | `srp_phat` (scored on hardware) or `music` (better against noise from one direction, see Accuracy; not validated on hardware) |
| `music_tc` | `3.0` | `music`: how long the utterance's covariance remembers, s |
| `vad_reset_s` | `2.0` | neural gate: clear the model's state this often while nobody speaks; 0 = never |
| `utterance_gap_s` | `0.8` | a pause this long ends an utterance; the next one starts a new pool; 0 = one pool for ever |
| `max_lag_s` | `1.0` | how far the worker may fall behind before it skips to the newest audio (and reports it) |
| `timing_warn_us` | `0` | plane-wave misfit above which channel timing is reported as suspect; 0 = 12 % of the array's largest delay |
| `publish_period` | `0.2` | seconds between bearings, in audio time |
| `publish_human` | `true` | also publish `/dorai_doa/human` for `ros2 topic echo` |

`avg_tc` is the main accuracy/latency dial: longer pools more frames and is
more accurate on a stationary talker, shorter follows a moving one sooner.

## Running with voice_mod

Both nodes want the microphone array, and two processes cannot open the same
ALSA capture device. Either run `doa` instead of `voice`, or have `voice_mod`
publish its raw tap and point `doa_mod` at the topic:

```bash
ros2 run voice_mod voice --ros-args -p publish_raw:=true
ros2 run doa_mod  doa   --ros-args -p source:=topic
```

Every frame of the topic is analysed, whatever the publisher's frame length,
and bearings are published at the same cadence of audio time. Long frames make
a bearing late, not wrong: with `voice_mod`'s default `publish_interval` of
10 s a bearing is up to 10 s old (the node says so once); use
`-p publish_interval:=0.1` for a live bearing.

**The raw tap of `voice_mod` is not a DoA input on the robot as it is built
now.** `voice_mod` opens every microphone as its own device, each with its own
clock and resampler, aligned by time stamps that are good to about a
millisecond; this array reads tens of microseconds (one sample of lateness on
one channel is 15 degrees). The robot's three microphones are also in a line,
which gives no bearing at all, and `doa_mod` refuses that geometry.
`voice_mod` reports how its channels are clocked on `<topic>/sync`; with
several devices and drift correction on, `doa_mod` warns
(`THE BEARING IS NOT TRUSTWORTHY`), marks `/dorai_doa/human` with
`CHANNEL TIMING SUSPECT` and shows `timing=SEPARATE-CLOCKS` in its diagnostics
line. With four or more microphones it also checks the audio itself
(`timing=ok(misfit 5us)` / `SUSPECT`): a late channel leaves a plane-wave
misfit, 25 us for one sample against 3-14 us when aligned (simulated rooms).
That check cannot see every fault; a pass is not proof.

For a bearing, open the ReSpeaker directly (`source:=mic`), or in simulation
use the one-device topic of the virtual microphones:
`-p source:=topic -p input_topic:=/dorai_sim/array`.

## Why one device, one clock

TDOA measures inter-channel delays of tens of microseconds. N separate USB
microphones are N independent crystals; 1 ppm of relative error is 1 µs of
drift per second, which walks through the array's entire 187 µs delay range in
a couple of minutes. A single multichannel device samples every capsule off one
crystal, so the only inter-channel delay is acoustic. This is why `doa_mod`
does not reuse `voice_mod`'s capture layer, which is built for separate mics
and per-mic resampling — right for beamforming, wrong for TDOA.

## Compute budget

Each hop must finish inside 32 ms (512 samples at 16 kHz). Measured per
`BearingEstimator.push()`, median over 120 hops:

| mode | per hop | share of budget |
| :--- | ---: | ---: |
| `neural` | 12.0 ms | 37% |
| `energy` | 11.8 ms | 37% |
| `off` | 12.0 ms | 37% |

The VAD is not the cost — SRP accumulation is (a 360-point grid × 6 pairs,
recomputed every frame), which is why all three modes land in the same place.
**These are laptop numbers; a Pi 4B is several times slower and this is the
figure to watch during bring-up.** The node logs `worker_duty` every
`diag_period` and warns above 80%. If it gets tight, coarsen the SRP grid to
2° — that halves the cost with negligible accuracy loss, since a 64 mm
aperture has a ~130° main lobe anyway.

## Tests

```bash
python3 -m pytest test/test_doa_mod.py -v     # 57 tests, no ROS needed
```

`doa_core.py`, `vad.py` and `vad.onnx` are **vendored byte-identical** from
`../../simulator/doa` — a ROS package cannot reach outside its own directory at build time.
`test_vendored_core_matches_research_tree` checksums them and fails if the
copies fork. To change the estimator: edit it in `../../simulator/doa`, re-run
`../../simulator/doa` to confirm the numbers still hold, then re-copy.
