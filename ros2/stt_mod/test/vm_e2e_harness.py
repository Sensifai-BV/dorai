#!/usr/bin/env python3
"""Recorder/monitor for the voice_mod -> stt_mod end-to-end test.

Runs next to the real voice_mod and stt_mod nodes. It records every
/dorai_clean_audio frame (exactly what stt_mod receives) to clean.wav, logs
/dorai_transcript finals with wall time, and keeps the latest
/dorai_stt/status. It exits once playback is finished (``--done-file``
appears), voice_mod has been stopped by the shell (no frame for 5 s) and
stt_mod has decoded everything it received.

Outputs in --out: clean.wav (16 kHz mono, the stt input stream),
clean_frames.jsonl (seq, capture_s, n, wall time per frame), finals.jsonl.
"""
import argparse
import json
import os
import threading
import time
import wave

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String


class Harness(Node):
    def __init__(self, a):
        super().__init__("vm_e2e_harness")
        self.a = a
        self.lock = threading.Lock()
        self.chunks, self.frames, self.finals = [], [], []
        self.n = 0
        self.status = {}
        self.create_subscription(Float32MultiArray, "/dorai_clean_audio", self._audio, 200)
        self.create_subscription(String, "/dorai_transcript", self._final, 50)
        self.create_subscription(String, "/dorai_stt/status", self._status, 10)

    def _audio(self, m):
        off = m.layout.data_offset or 4
        x = np.asarray(m.data[off:], dtype=np.float32)
        with self.lock:
            self.frames.append({"seq": int(m.data[3]), "capture_s": float(m.data[0]),
                                "start": self.n, "n": int(x.size), "wall": time.time()})
            self.chunks.append(x)
            self.n += x.size

    def _final(self, m):
        with self.lock:
            self.finals.append({"wall": time.time(), "text": m.data})

    def _status(self, m):
        with self.lock:
            self.status = json.loads(m.data)

    def save(self):
        os.makedirs(self.a.out, exist_ok=True)
        with self.lock:
            x = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)
            frames, finals, st = list(self.frames), list(self.finals), dict(self.status)
        with wave.open(os.path.join(self.a.out, "clean.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
        with open(os.path.join(self.a.out, "clean_frames.jsonl"), "w") as f:
            for r in frames:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(self.a.out, "finals.jsonl"), "w") as f:
            for r in finals:
                f.write(json.dumps(r) + "\n")
        json.dump(st, open(os.path.join(self.a.out, "stt_status_last.json"), "w"))
        seqs = [r["seq"] for r in frames]
        gaps = sum(1 for p, q in zip(seqs, seqs[1:]) if q != p + 1)
        print(f"saved {x.size / 16000:.1f}s clean, {len(frames)} frames (seq gaps {gaps}), "
              f"{len(finals)} finals, stt status {st}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--done-file", required=True, help="touched by the shell when playback ends")
    ap.add_argument("--tail-s", type=float, default=10.0,
                    help="clean audio after playback end that must be decoded before exit")
    ap.add_argument("--timeout-s", type=float, default=7200)
    a = ap.parse_args()
    rclpy.init()
    node = Harness(a)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    t0 = time.time()
    target, t_done = None, None
    while time.time() - t0 < a.timeout_s:
        time.sleep(0.5)
        with node.lock:
            n, st = node.n, dict(node.status)
        if t_done is None and os.path.exists(a.done_file):
            t_done = time.time()
        with node.lock:
            last_frame = node.frames[-1]["wall"] if node.frames else 0.0
        # The shell stops voice_mod --tail-s after playback ends. voice_mod
        # would otherwise keep publishing silence that a slower-than-real-time
        # stt_mod can never catch up with. Done = input has stopped for 5 s and
        # stt_mod has decoded everything it received.
        if t_done is not None and time.time() - t_done >= a.tail_s and time.time() - last_frame > 5.0:
            if target is None:
                target = n / 16000.0
                print(f"input stopped; waiting for stt to decode {target:.1f}s of clean audio", flush=True)
            if st.get("audio_s", 0) >= target - 0.01 and st.get("backlog_s", 1) == 0 \
                    and not st.get("busy", True):
                time.sleep(3.0)          # let the last final arrive
                break
    node.save()
    ex.shutdown(timeout_sec=2.0)     # stop spinning before teardown (else: C++ terminate at exit)
    spin.join(timeout=2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
