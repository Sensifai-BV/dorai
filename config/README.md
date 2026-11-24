# config

Everything the simulator and the experiments read, so that a geometry, a room
or a test condition is changed without touching code.

| Folder / file | What | Read by |
|---|---|---|
| `robots/artaban.yaml` | the robot as it is built now: its arrays, how they are captured, microphone imperfections, own noise | `simulator.acoustic.robot.load_robot` |
| `arrays/respeaker_4mic.yaml` | doa_mod's array: 4 capsules, 32 mm radius, at 45/135/225/315 deg in channel order | `simulator.acoustic.microphone_array.load_array`; a test checks it equals doa_mod's built-in geometry |
| `arrays/artaban_line3.yaml` | voice_mod's array: 3 microphones in a line, 10 cm apart | the same |
| `rooms/*.yaml` | `anechoic`, `rt60` (target RT60), `office`, `workshop`, `factory` (materials per wall) | `simulator.acoustic.room_engine.load_room` |
| `scenarios/*.yaml` | a room, the robot's place, the talker, the noise | `python -m simulator.acoustic.audio_generator` |
| `commands.yaml` | the commands to synthesise, voices, speaking rates | `python -m simulator.acoustic.generators.tts_generator` |

Positions are metres in the robot's frame: +x is 0 deg (the robot's heading),
+y is 90 deg, bearings counter-clockwise. doa_mod, the simulator and the truth
files all use this one convention.

`arrays/respeaker_4mic.yaml` was first drafted with the capsules on the axes
(90/0/270/180 deg). That is not the geometry doa_mod has built in and was
validated with on hardware, and the two differ by a mirror, which no offset
corrects, so the file now holds the validated positions. The board revision on
the robot still has to be confirmed. The `simulation:`, `ros2:` and `dorai:`
blocks of that file are documentation of intended settings; the code reads
`array.microphones` only.

Node parameters stay with their packages (`ros2/stt_mod/config/stt.yaml`,
`ros2/intent_mod/config/intent.yaml`), where the launch files install them.
Set `DORAI_CONFIG` to use another config folder (scenes are rendered outside the Docker image).
