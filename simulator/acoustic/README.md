# DorAI virtual acoustic lab (`simulator/acoustic`)

Simulates what the **Artaban robot's microphones** record in a room, so that
voice_mod, stt_mod, intent_mod and doa_mod can be developed and tested without
the robot. The acoustics are [pyroomacoustics](https://github.com/LCAV/pyroomacoustics):
`pra.ShoeBox` rooms with `pra.Material` walls, image sources, ray tracing, and
`room.compute_rir()` for every source-microphone pair. No ROS in this folder;
the ROS 2 side is `ros2/dorai_simulator`.

```
config/robots/artaban.yaml      the robot as it is built now
config/arrays, rooms, scenarios, commands.yaml
        |
generators/tts_generator        commands from text (the system `say` command), text kept with each file
generators/*_noise              welding, machinery, ventilation models (or recorded noise files)
room_engine, rir_generator      pyroomacoustics room -> impulse responses
hardware_model                  gain mismatch, capsule position error, self-noise, delay, quantisation, clipping
scene_mixer                     x_m(t) = sum_k h_mk(t) * s_k(t) + n_m(t), plus the truth
audio_generator                 scenario files -> WAV per array + truth.json
        |
        +-- offline:  evaluation/doa_node.py, experiments/doa_benchmark.py   (the real doa_mod node, no ROS)
        +-- ROS 2:    ros2/dorai_simulator (virtual_mic) -> voice_mod, doa_mod, stt_mod, intent_mod
```

## The simulated robot

`config/robots/artaban.yaml` describes Artaban in its current state, and every
scenario uses it unless it says otherwise:

| | |
|---|---|
| Microphones for `voice_mod` | `artaban_line3`: three USB microphones in a line along the back, 10 cm apart, **each its own capture device with its own clock** (what `dorai_beamformer.ort` is exported for) |
| Microphones for `doa_mod` | `respeaker_4mic`: ReSpeaker 4-Mic Array, 32 mm radius, capsules at 45/135/225/315°, one device, one clock |
| Height | 0.5 m (top of the back) |
| Capture | 48 kHz devices, 100 ms blocks, streams opened 0 / 1 / 2 ms apart, clock errors 0 ppm (not measured on the robot yet), 0.3 ms stamp jitter |
| Microphone imperfections | 0.5 dB gain spread, 1 mm capsule placement error, self-noise 40 dB below speech (assumed, not measured) |
| Own noise (motors, fans) | none yet: nothing has been recorded on the robot |

When something is measured on the real robot (clock errors, gains, the board
revision), put it in that file and every scene follows.

## Quick start (from the repo root)

```bash
pip install -r requirements.txt
python -m simulator.acoustic.generators.tts_generator --out ~/dorai/sim/speech      # commands in config/commands.yaml
python -m simulator.acoustic.audio_generator config/scenarios/*.yaml \
       --out ~/dorai/sim/scenes --speech-dir ~/dorai/sim/speech --seeds 0 1 2
python -m evaluation.doa_node ~/dorai/sim/scenes                                    # doa_mod on every scene
python -m experiments.doa_benchmark --speech ~/dorai/sim/speech                     # the DoA test matrix
```

A scene folder holds `artaban_line3.wav` and `respeaker_4mic.wav` (float32,
16 kHz, one channel per microphone, one clock) and `truth.json`: every command
with its text, start, end, bearing, distance and elevation; every noise source
with its bearing, its requested and its **measured** SNR; the room with the
RT60 **measured** from the impulse responses; the robot; what the hardware
model applied to each channel. `--keep-parts` also writes `<array>.speech.wav`
and `<array>.noise.wav`, the two parts of the mix on their own (the aligned
clean reference for SNR and SI-SDR).

Speech is any folder of 16 kHz WAV files; the text of a file is its `.txt`
neighbour, else its name. Recordings of real people stay out of the repo.

## What a scenario says (`config/scenarios/*.yaml`)

```yaml
room: factory                 # config/rooms: anechoic | rt60 | office | workshop | factory, or {name: rt60, rt60_s: 0.6}
robot: {place: open}          # open | wall (0.4 m from a wall) | [x, y];  model, height, heading_deg
talker: {commands: 6, distance_m: [1.0, 4.0]}     # each command from another place; bearing_deg fixes it
noise:
  - {file: simulator/lab/wavs/welding.wav, snr_db: 0, from_s: 12.0}     # a recording, from a point in the room
  - {type: machinery_noise, snr_db: 10, bearing_deg: 240, distance_m: 3}   # a generated noise, placed by bearing
hardware: {channel_delay_samples: [0, 0.5, 0, 0, 0, 0, 0], quantization_bits: 16}   # on top of the robot's
seed: 0
```

**Room levels.** `anechoic`: free field. `rt60`: image sources only, one
absorption from Sabine's formula. `office`, `workshop`, `factory`: a material
per wall, image sources to order 6, ray tracing for the late part, air
absorption, scattering walls. A target RT60 is a request: targets 0.2 / 0.4 /
0.6 / 0.8 / 1.0 s measured 0.1-0.15 / 0.4 / 0.65 / 0.93 / 1.2 s.

**SNR convention.** Power of the commands over power of the noise, both after
the room, at microphone 1 of the first array, over the time commands are
spoken. Stationary noise lands within 1 dB of the request; noise with pauses
(welding arcs) can differ, and `snr_measured_db` says by how much.

## In ROS 2

```bash
ros2 run dorai_simulator virtual_mic --ros-args -p scene:=$HOME/dorai/sim/scenes/factory_welding_0db_s0
ros2 run voice_mod voice --ros-args -p source:=topic -p publish_interval:=1.0
ros2 run doa_mod doa --ros-args -p source:=topic -p input_topic:=/dorai_sim/array
ros2 launch stt_mod stt.launch.py ; ros2 launch intent_mod intent.launch.py
ros2 topic echo /dorai_sim/truth
```

| Topic | Content |
|---|---|
| `/dorai_sim/mic0..2` | the line microphones as separate capture devices: one message is one 100 ms block of one device at 48 kHz, `[seq, rate, stamp // 1000, stamp % 1000, samples...]`; start offsets, clock errors (`-p ppm:="[0.0, 40.0, -30.0]"`) and stamp jitter per device |
| `/dorai_sim/array` | the ReSpeaker as one device, in the layout of voice_mod's raw tap `[capture_s, rate, channels, seq, interleaved...]` |
| `/dorai_sim/array/sync` | timing status for doa_mod: one device |
| `/dorai_sim/truth` | JSON: the scene, each command when it starts and ends, the end. For evaluation only; no pipeline node reads it |

`voice_mod -p source:=topic` treats each message as a capture callback, so its
resampler, start-up alignment, drift loop and beamformer run as on the robot.

## First live run (2026-11-19, ROS 2, generated speech, Whisper, rule-based intent)

| Scene | Word errors | Commands ending as the right plan or stop | DoA mean / max error |
|---|---:|---:|---:|
| workshop, quiet | 4.3 % | 4 of 6 | 3° / 8° |
| office, robot at a wall | 4.0 % | 5 of 6 | 15° / 39° |
| workshop, welding 10 dB below speech | 32 % | 4 of 6 | 145° / 177° |
| factory, welding as loud as speech | 44 % | 1 of 6 | 73° / 173° |
| factory, welding + machinery | 74 % | 1 of 6 | 50° / 135° |

Where the words are lost (same scenes, decoded offline with the same engine):
speech alone at a microphone is recognised with 4-8 % word errors in every
scene, so reverberation is not the problem. With welding 10 dB below speech one
raw microphone and the beamformer's output both give 16 %; with noise as loud
as speech the raw microphone gives 65 % and 78 %, the beamformed audio 48 % and
57 %. The beamformer helps and is not enough. For the bearing, `doa_mod
-p method:=music` holds up better in these scenes (see ros2/doa_mod/README.md).

One run per scene. No lost blocks or xruns in any run; a transcript arrives
3.3 s after the end of a command (median). With noise from one direction the
bearing points at the noise, and one command was recognised as "stop" by
mistake. With device clocks really drifting (0 / +40 / -30 ppm) the drift loop
found the right corrections (0 / -41 / +29 ppm) but only after about a minute,
and the channels were up to 0.86 ms apart on the way.

Tests: `python -m pytest simulator/acoustic/tests ros2/dorai_simulator/test -q`.
