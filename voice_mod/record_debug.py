#!/usr/bin/env python3
"""
voice_mod / record_debug.py  —  dorai debug audio recorder.

Records N seconds of audio to WAV for debugging the pipeline. A --source
switch selects what to capture:

  clean   subscribe /dorai_clean_audio  -> <prefix>_clean.wav   (mono, 16 kHz)
  raw     subscribe /dorai_raw_audio    -> <prefix>_raw.wav     (M-ch, 16 kHz)
          (run voice_mod with  -p publish_raw:=true)
  both    clean + raw simultaneously, time-aligned through ROS
  dev     open the USB mics directly with sounddevice and capture the true
          native-rate signal -> <prefix>_dev0.wav, _dev1.wav, ...
          (run with voice_mod STOPPED — ALSA gives one capture client per USB mic)

The 'clean'/'raw'/'both' modes ride the live pipeline so you compare exactly
what the beamformer received vs. produced. 'dev' bypasses ROS entirely to check
raw capture quality / mic wiring when the pipeline is down.

Examples:
  ros2 run voice_mod record --ros-args -p source:=both -p duration:=15.0
  ros2 run voice_mod record --ros-args -p source:=clean -p prefix:=test
  ros2 run voice_mod record --ros-args -p source:=dev -p duration:=10.0 \
      -p device_rate:=48000 -p max_mics:=3
"""

import sys
import wave
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


def _write_wav(filename, pcm16, channels, rate):
    """Write interleaved int16 samples to a WAV file."""
    with wave.open(filename, "wb") as w:
        w.setnchannels(int(channels))
        w.setsampwidth(2)               # 16-bit
        w.setframerate(int(rate))
        w.writeframes(pcm16.tobytes())


def _to_pcm16(x):
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


# ---------------------------------------------------------------------------
# ROS topic recorder (clean / raw / both).
# ---------------------------------------------------------------------------
class _TopicSink:
    """Accumulates audio for one Float32MultiArray topic and writes a WAV."""

    def __init__(self, node, topic, filename, max_samples):
        self.node = node
        self.topic = topic
        self.filename = filename
        self.max_samples = max_samples
        self.frames = []                # list of [N, ch] float arrays
        self.channels = 1
        self.rate = 16000
        self.collected = 0              # per-channel samples
        self.done = False
        self.sub = node.create_subscription(
            Float32MultiArray, topic, self._cb, 10)

    def _cb(self, msg):
        if self.done:
            return
        off = msg.layout.data_offset or 4
        if len(msg.data) <= off:
            return
        self.rate = int(msg.data[1]) or 16000
        self.channels = max(1, int(msg.data[2]))
        flat = np.asarray(msg.data[off:], dtype=np.float32)
        n = (flat.size // self.channels) * self.channels
        if n == 0:
            return
        frame = flat[:n].reshape(-1, self.channels)     # [N, ch], interleaved
        self.frames.append(frame)
        self.collected += frame.shape[0]
        self.node.get_logger().info(
            f"[{self.topic}] {self.collected}/{self.max_samples} samples "
            f"({self.channels}ch @ {self.rate}Hz)",
            throttle_duration_sec=1.0)
        if self.collected >= self.max_samples:
            self.save()

    def save(self):
        if self.done:
            return
        self.done = True
        if not self.frames:
            self.node.get_logger().warning(
                f"[{self.topic}] no audio received; nothing written. "
                f"Is the publisher running?")
            return
        audio = np.concatenate(self.frames, axis=0)[:self.max_samples]
        pcm16 = _to_pcm16(audio).reshape(-1)            # interleave channels
        _write_wav(self.filename, pcm16, self.channels, self.rate)
        self.node.get_logger().info(
            f"[{self.topic}] saved {self.filename} "
            f"({self.channels}ch, {audio.shape[0]} frames @ {self.rate}Hz)")


class TopicRecorder(Node):
    def __init__(self, sources, duration, prefix):
        super().__init__("record_debug")
        max_samples = int(16000 * duration)
        self.sinks = []
        if "clean" in sources:
            self.sinks.append(_TopicSink(
                self, "/dorai_clean_audio", f"{prefix}_clean.wav", max_samples))
        if "raw" in sources:
            self.sinks.append(_TopicSink(
                self, "/dorai_raw_audio", f"{prefix}_raw.wav", max_samples))
        self.get_logger().info(
            f"Recording {sorted(sources)} for {duration:.1f}s "
            f"(prefix='{prefix}')...")
        # Safety timeout: stop ~5 s after the nominal duration even if a topic
        # is silent / not publishing, so we always write what we got.
        self._timeout = self.create_timer(duration + 5.0, self._on_timeout)
        self._tick = self.create_timer(0.2, self._check_done)

    def _check_done(self):
        if all(s.done for s in self.sinks):
            self._finish()

    def _on_timeout(self):
        self.get_logger().warning("Timeout reached; flushing partial captures.")
        self._finish()

    def _finish(self):
        for s in self.sinks:
            s.save()
        raise SystemExit


# ---------------------------------------------------------------------------
# Direct device recorder (dev) — bypasses ROS, captures native-rate raw mics.
# ---------------------------------------------------------------------------
def _usable_input(name):
    low = name.lower()
    exclude = ("blackhole", "loopback", "zoom", "teams", "microsoft", "obs",
               "background music", "monitor", "pulse", "default", "sysdefault",
               "macbook", "built-in", "internal", "hdmi")
    return not any(t in low for t in exclude)


def _looks_usb(name):
    low = name.lower()
    return any(t in low for t in ("usb", "mic", "uac", "card"))


def record_devices(duration, prefix, rate, max_mics):
    import sounddevice as sd

    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices)
              if d.get("max_input_channels", 0) > 0 and _usable_input(d["name"])]
    inputs.sort(key=lambda it: (not _looks_usb(it[1]["name"]), it[0]))
    inputs = inputs[:max_mics]
    if not inputs:
        print("No usable input devices found.", file=sys.stderr)
        return 1

    streams, buffers, rates = [], [], []
    for slot, (idx, dev) in enumerate(inputs):
        # Pick a supported rate: requested first, then the device default.
        use_rate = rate
        try:
            sd.check_input_settings(device=idx, channels=1,
                                    dtype="float32", samplerate=rate)
        except Exception:
            use_rate = int(dev["default_samplerate"])
        buf = []
        buffers.append(buf)
        rates.append(use_rate)

        def _mk(b):
            def _cb(indata, frames, t, status):
                if status:
                    print(f"  (xrun on dev: {status})", file=sys.stderr)
                b.append(indata[:, 0].copy())
            return _cb

        st = sd.InputStream(samplerate=use_rate, device=idx, channels=1,
                            dtype="float32", callback=_mk(buf))
        streams.append(st)
        print(f"dev{slot}: #{idx} '{dev['name']}' @ {use_rate} Hz")

    print(f"Recording {len(streams)} mic(s) for {duration:.1f}s ...")
    for st in streams:
        st.start()
    try:
        sd.sleep(int(duration * 1000))
    finally:
        for st in streams:
            try:
                st.stop()
                st.close()
            except Exception:
                pass

    for slot, buf in enumerate(buffers):
        if not buf:
            print(f"dev{slot}: no audio captured.", file=sys.stderr)
            continue
        audio = np.concatenate(buf)
        fn = f"{prefix}_dev{slot}.wav"
        _write_wav(fn, _to_pcm16(audio), 1, rates[slot])
        print(f"dev{slot}: saved {fn} ({audio.size} samples @ {rates[slot]}Hz)")
    return 0


# ---------------------------------------------------------------------------
def main():
    rclpy.init()
    # A throwaway node just to read parameters consistently with ros2 run.
    # Numeric params use dynamic typing so `duration:=10` (int) and
    # `duration:=10.0` (double) are both accepted from the command line.
    from rcl_interfaces.msg import ParameterDescriptor
    dyn = ParameterDescriptor(dynamic_typing=True)
    cfg = Node("record_debug_cfg")
    cfg.declare_parameter("source", "both")     # clean | raw | both | dev
    cfg.declare_parameter("duration", 15.0, dyn)
    cfg.declare_parameter("prefix", "dorai_debug")
    cfg.declare_parameter("device_rate", 48000, dyn)  # dev mode only
    cfg.declare_parameter("max_mics", 3, dyn)         # dev mode only
    source = cfg.get_parameter("source").value.strip().lower()
    duration = float(cfg.get_parameter("duration").value)
    prefix = cfg.get_parameter("prefix").value
    device_rate = int(cfg.get_parameter("device_rate").value)
    max_mics = int(cfg.get_parameter("max_mics").value)
    cfg.destroy_node()

    if source == "dev":
        rclpy.shutdown()
        sys.exit(record_devices(duration, prefix, device_rate, max_mics))

    if source == "both":
        sources = {"clean", "raw"}
    elif source in ("clean", "raw"):
        sources = {source}
    else:
        print(f"Unknown source '{source}' "
              f"(use clean|raw|both|dev).", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(2)

    node = TopicRecorder(sources, duration, prefix)
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
