# sim — simulated microphones for the dorai pipeline

Renders what the robot's microphones would record in a room, and plays it into
the ROS graph as virtual microphones, with the truth to score every module
against. The acoustics are [pyroomacoustics](https://github.com/LCAV/pyroomacoustics).
Nothing here needs the robot, and rendering and scoring need no ROS.

```
config/arrays, rooms, scenarios, commands.yaml
        |
   sim.tts          commands from text (macOS `say`), text kept with each file
   sim.render       scenario -> one WAV per array + truth.json     (pyroomacoustics)
        |
        +-- offline:  sim.score_doa, experiments/doa_benchmark.py   (the real doa_mod node, no ROS)
        +-- ROS 2:    virtual_mic node -> voice_mod (source:=topic), doa_mod (source:=topic)
```

## Quick start (macOS or Linux, no ROS)

```bash
pip install pyroomacoustics soundfile pyyaml scipy numpy onnxruntime
cd simulator/module
python -m sim.tts --out ~/dorai/sim/speech                       # commands from config/commands.yaml
python -m sim.render ../../config/scenarios/*.yaml --out ~/dorai/sim/scenes --speech-dir ~/dorai/sim/speech --seeds 0 1 2
python -m sim.score_doa ~/dorai/sim/scenes                       # doa_mod on every scene
python -m sim.score_doa ~/dorai/sim/scenes subtract_noise=true   # ... with another parameter
```

A scene folder holds `artaban_line3.wav` (voice_mod's three microphones),
`respeaker_4mic.wav` (doa_mod's array), both float32 at 16 kHz on one clock, and
`truth.json`: every command with its text, start, end, bearing, distance and
elevation; every noise source with its bearing and level; the room with the
RT60 **measured** from the rendered impulse responses.

Speech: any folder of 16 kHz WAV files. The text of a file is its `.txt`
neighbour, else its name (`07_Artaban_come_here.wav`). `sim.tts` makes such a
folder from text. Recordings of real people stay out of the repo.

## What a scenario says

```yaml
room: factory                 # a file in config/rooms, or {name: rt60, rt60_s: 0.6}
robot: {place: open}          # open | wall (0.4 m from a wall) | [x, y];  height, heading_deg
arrays: [artaban_line3, respeaker_4mic]      # files in config/arrays, all at the robot's position
talker: {commands: 6, distance_m: [1.0, 4.0]}   # each command from another place; bearing_deg fixes it
noise:
  - {file: simulator/lab/wavs/welding.wav, snr_db: 0, from_s: 12.0}   # a point in the room, with its own reverberation
mics: {gain_db_sd: 0.5, position_error_mm: 1.0, self_noise_snr_db: 40}
seed: 0
```

**SNR convention.** `snr_db` is the power of the commands against the power of
that noise source, both after the room, at the first microphone of the first
array, over the time the commands are spoken (pauses inside a command
included). It is reached to within 1 dB (tested).

**Room levels.** `anechoic` is free field. `rt60` is image sources only with
one absorption from Sabine's formula. `office`, `workshop`, `factory` use a
material per wall, image sources to order 6 and ray tracing for the late part.
The target RT60 is not the RT60 you get: for targets 0.2 / 0.4 / 0.6 / 0.8 /
1.0 s the `rt60` room measured 0.1-0.15 / 0.4 / 0.65 / 0.93 / 1.2 s.
Walls scatter (0.25): a shoebox of mirrors rang for 2 to 10 s.

## In ROS 2

```bash
ros2 run sim virtual_mic --ros-args -p scene:=$HOME/dorai/sim/scenes/factory_welding_0db_s0
ros2 run voice_mod voice --ros-args -p source:=topic -p publish_interval:=1.0
ros2 run doa_mod doa --ros-args -p source:=topic -p input_topic:=/dorai_sim/array
ros2 launch stt_mod stt.launch.py ; ros2 launch intent_mod intent.launch.py
ros2 topic echo /dorai_sim/truth
```

| Topic | Content |
|---|---|
| `/dorai_sim/mic0..2` | the line microphones as **separate capture devices**: one message is one 100 ms block of one device at 48 kHz, `[seq, rate, stamp // 1000, stamp % 1000, samples...]`. Per device: `start_ms` (default 0, 1, 2), `ppm` clock error (default 0), `stamp_jitter_ms`. |
| `/dorai_sim/array` | the ReSpeaker as **one device**, in the layout of voice_mod's raw tap `[capture_s, rate, channels, seq, interleaved...]`, 100 ms frames |
| `/dorai_sim/array/sync` | timing status for doa_mod: one device |
| `/dorai_sim/truth` | JSON: the scene at the start, each command when it starts and ends, the end |

`voice_mod -p source:=topic` treats each message as a capture callback, so its
resampler, start-up alignment, drift loop and beamformer run as with real
devices; with `-p ppm:="[0.0, 40.0, -30.0]"` on the virtual microphones the
drift loop gets clocks that really drift. The truth topic is for evaluation
only; no pipeline node subscribes to it.

## Limits

- No robot body, no talker or microphone directivity, no microphone frequency
  response; talkers change place between commands, not while speaking.
- Generated speech is cleaner and more uniform than people.
- A simulated ReSpeaker is not the board: geometry and channel order in
  `config/arrays/respeaker_4mic.yaml` still have to be confirmed on the robot.
- The virtual microphone node and `voice_mod source:=topic` have unit tests but
  have not been run in a live ROS graph yet.

Tests: `python -m pytest simulator/module/test -q` (the render test needs pyroomacoustics).
