"""DorAI virtual acoustic lab: what the Artaban robot's microphones record in a room.

Built on pyroomacoustics. No ROS in this package; the ROS 2 side is ros2/dorai_simulator.

  config.py            where the YAML files are, constants
  robot.py             the robot as it is built now (config/robots/artaban.yaml)
  microphone_array.py  array geometry from config/arrays
  room_engine.py       rooms from config/rooms -> a pyroomacoustics room
  rir_generator.py     room impulse responses, convolution
  source_manager.py    speech files and their text, noise sources (recorded or generated)
  generators/          text-to-speech commands, welding / machinery / ventilation noise
  hardware_model.py    microphone imperfections
  scene_mixer.py       scenario -> signals at every microphone + truth
  audio_generator.py   command line: scenarios -> WAV + truth.json
"""
