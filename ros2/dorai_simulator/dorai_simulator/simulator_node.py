#!/usr/bin/env python3
"""virtual_mic: play a rendered scene into the ROS graph as the robot's microphones.

    ros2 run dorai_simulator virtual_mic --ros-args -p scene:=/path/to/scene_folder
    ros2 run voice_mod voice --ros-args -p source:=topic            # hears the three line microphones
    ros2 run doa_mod doa --ros-args -p source:=topic -p input_topic:=/dorai_sim/array

A scene folder comes from `python -m simulator.acoustic.audio_generator` (one WAV per array plus
truth.json). The node publishes, in real time:

  /dorai_sim/mic0..N-1   std_msgs/Float32MultiArray   the `array` microphones as SEPARATE capture devices:
                         one message = one 100 ms block of one device at 48 kHz, each device with its own
                         start time (start_ms), clock error (ppm) and stamp jitter. voice_mod with
                         source:=topic treats each message as a capture callback.
  /dorai_sim/array       std_msgs/Float32MultiArray   the `doa_array` microphones as ONE multichannel device
                         (one clock), in the layout of voice_mod's raw tap, for doa_mod with source:=topic.
  /dorai_sim/array/sync  the matching timing status: one device.
  /dorai_sim/truth       std_msgs/String (JSON)       what is really happening: the scene at the start, then
                         every command when it begins and ends (text, bearing, distance), then the end.

After the scene it publishes `tail_s` seconds of silence so the pipeline can
finish its last utterance, then stops (or starts over with loop:=true).
"""
import json
import os
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from dorai_simulator import jsonsafe, playback_scheduler as devices


def read_scene(folder, array):
    """[M, n] float32 and the sample rate. soundfile if it is installed, else
    scipy (the scene files are float WAV, which the standard `wave` cannot read)."""
    path = os.path.join(folder, array + ".wav")
    try:
        import soundfile as sf
        x, fs = sf.read(path, dtype="float32", always_2d=True)
    except ImportError:
        from scipy.io import wavfile
        fs, x = wavfile.read(path)
        x = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
    return x.T, int(fs)


class VirtualMic(Node):
    def __init__(self):
        super().__init__("virtual_mic")
        from rcl_interfaces.msg import ParameterDescriptor
        dyn = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("scene", "")
        self.declare_parameter("array", "artaban_line3")              # played as separate devices
        self.declare_parameter("doa_array", "respeaker_4mic")     # played as one device; "" = none
        self.declare_parameter("device_rate", 48000)
        self.declare_parameter("block_ms", 100.0, dyn)
        # One value per microphone of `array`. Defaults: one clock, streams
        # opened 0, 1 and 2 ms after each other (what three USB devices
        # opened in a row look like), 0.3 ms of stamp jitter.
        # A scene carries the robot's own values (config/robots/artaban.yaml
        # -> truth.json); they are used while these two are left at their
        # defaults and use_robot_capture is true.
        self.declare_parameter("ppm", [0.0, 0.0, 0.0])
        self.declare_parameter("start_ms", [0.0, 1.0, 2.0])
        self.declare_parameter("use_robot_capture", True)
        self.declare_parameter("stamp_jitter_ms", 0.3, dyn)
        self.declare_parameter("mic_topic_prefix", "/dorai_sim/mic")
        self.declare_parameter("array_topic", "/dorai_sim/array")
        self.declare_parameter("truth_topic", "/dorai_sim/truth")
        self.declare_parameter("start_delay_s", 3.0, dyn)     # time for the other nodes to subscribe
        self.declare_parameter("tail_s", 8.0, dyn)
        self.declare_parameter("loop", False)
        self.declare_parameter("seed", 0)

        gp = self.get_parameter
        self.scene = os.path.expanduser(str(gp("scene").value))
        if not os.path.isfile(os.path.join(self.scene, "truth.json")):
            raise FileNotFoundError(
                f"scene:={self.scene!r} is not a rendered scene (no truth.json). "
                f"Render one with: python -m simulator.acoustic.audio_generator <scenario.yaml> --out <folder>")
        self.truth = json.load(open(os.path.join(self.scene, "truth.json"), encoding="utf-8"))
        self.rate = int(gp("device_rate").value)
        self.tail_s = float(gp("tail_s").value)
        self.loop = bool(gp("loop").value)
        self.start_delay_s = float(gp("start_delay_s").value)

        x, fs = read_scene(self.scene, str(gp("array").value))
        tail = np.zeros((x.shape[0], int(self.tail_s * fs)), np.float32)
        x = np.concatenate((x, tail), axis=1)
        cap = (self.truth.get("robot") or {}).get("capture") or {}
        robot = bool(gp("use_robot_capture").value) and bool(cap)
        ppm_p, start_p = [float(v) for v in gp("ppm").value], [float(v) for v in gp("start_ms").value]
        ppm = (cap["ppm"] if robot and ppm_p == [0.0, 0.0, 0.0] and cap.get("ppm") else ppm_p)
        start = (cap["start_ms"] if robot and start_p == [0.0, 1.0, 2.0] and cap.get("start_ms") else start_p)
        ppm, start = [float(v) for v in ppm][:x.shape[0]], [float(v) for v in start][:x.shape[0]]
        ppm += [0.0] * (x.shape[0] - len(ppm))
        start += [0.0] * (x.shape[0] - len(start))
        self.blocks = devices.device_blocks(
            x, fs, self.rate, float(gp("block_ms").value), ppm, start,
            float(gp("stamp_jitter_ms").value), int(gp("seed").value))
        self.mic_pubs = [self.create_publisher(Float32MultiArray, f"{gp('mic_topic_prefix').value}{i}", 50)
                         for i in range(x.shape[0])]

        self.frames, self.array_pub, self.sync_pub, self.n_array = [], None, None, 0
        doa_array = str(gp("doa_array").value)
        if doa_array and os.path.isfile(os.path.join(self.scene, doa_array + ".wav")):
            a, afs = read_scene(self.scene, doa_array)
            a = np.concatenate((a, np.zeros((a.shape[0], int(self.tail_s * afs)), np.float32)), axis=1)
            self.frames = devices.array_frames(a, afs)
            self.n_array = a.shape[0]
            topic = str(gp("array_topic").value)
            self.array_pub = self.create_publisher(Float32MultiArray, topic, 50)
            self.sync_pub = self.create_publisher(Float32MultiArray, topic + "/sync", 10)
        self.truth_pub = self.create_publisher(String, str(gp("truth_topic").value), 20)

        self.get_logger().info(
            f"scene '{self.truth['name']}': {self.truth['seconds']:.0f} s, room {self.truth['room']['name']} "
            f"(RT60 {self.truth['room']['rt60_s']} s), {len(self.truth['commands'])} commands, "
            f"{len(self.truth['noise'])} noise source(s)")
        self.get_logger().info(
            f"{x.shape[0]} separate devices at {self.rate} Hz on {gp('mic_topic_prefix').value}0.., clock errors {ppm} ppm, "
            f"starts {start} ms" + (f"; one {self.n_array}-channel device on {gp('array_topic').value}" if self.array_pub else ""))
        self._running = True
        self._thread = threading.Thread(target=self._play, name="virtual-mic", daemon=True)
        self._thread.start()

    def _say(self, **event):
        event["stamp"] = time.time()
        self.truth_pub.publish(String(data=jsonsafe.dumps(event)))

    def _events(self):
        """(due_s, kind, payload) of everything to publish, in time order."""
        ev = [(b["due_s"], 0, b) for b in self.blocks] + [(d, 1, p) for d, p in self.frames]
        for i, c in enumerate(self.truth["commands"]):
            ev.append((c["t0"], 2, dict(event="command", phase="start", index=i, **c)))
            ev.append((c["t1"], 2, dict(event="command", phase="end", index=i, **c)))
        ev.sort(key=lambda e: (e[0], e[1]))
        return ev

    def _play(self):
        time.sleep(self.start_delay_s)
        events = self._events()
        while self._running:
            t0 = time.monotonic()
            self._say(event="scene", name=self.truth["name"], seconds=self.truth["seconds"], room=self.truth["room"],
                      commands=self.truth["commands"], noise=self.truth["noise"])
            for due, kind, payload in events:
                wait = t0 + due - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                if not self._running:
                    return
                if kind == 0:
                    msg = Float32MultiArray()
                    msg.data = devices.device_message(payload, self.rate, t0).tolist()
                    self.mic_pubs[payload["device"]].publish(msg)
                elif kind == 1:
                    msg = Float32MultiArray()
                    msg.data = payload.tolist()
                    self.array_pub.publish(msg)
                    sync = Float32MultiArray()
                    sync.data = [float(payload[3]), float(self.n_array), 1.0, 0.0, 0.0, 0.0]
                    self.sync_pub.publish(sync)
                else:
                    self._say(**payload)
            self._say(event="end", name=self.truth["name"])
            self.get_logger().info("scene finished")
            if not self.loop:
                break
        self._running = False

    def finished(self):
        return not self._thread.is_alive()

    def shutdown(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VirtualMic()
        while rclpy.ok() and not node.finished():
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
