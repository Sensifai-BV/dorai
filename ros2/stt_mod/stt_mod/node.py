#!/usr/bin/env python3
"""stt_mod — Stage 2 of the dorai speech pipeline (ROS 2 node ``stt_mod``).

Two engines, both through sherpa-onnx on the CPU (``engine``):

  whisper   (default) Whisper large-v3-turbo with the dorai command fine-tune.
            Not a streaming model: the clean stream is cut into utterances by a
            VAD gate (vad_gated.onnx) and each utterance is decoded once (see whisper.py). Audio
            the gate does not pass is never decoded, so noise produces no text.
  nemotron  NVIDIA nemotron-speech-streaming-en-0.6b (cache-aware FastConformer
            + RNN-T), optionally with a dorai fine-tuned encoder adapter baked
            into the ONNX encoder (see bake_adapter.py). Streaming, with partial
            hypotheses and its own endpointing.

If the Whisper model is not installed the node says so and, unless
``engine_fallback`` is false, runs Nemotron instead.

Topics
  in   /dorai_clean_audio        std_msgs/Float32MultiArray  [capture_s, 16000, 1, seq, samples...]
                                 (layout.data_offset = 4), from voice_mod
  out  /dorai_partial_transcript std_msgs/String  running hypothesis (one update per encoder chunk;
                                                  Nemotron only)
       /dorai_transcript         std_msgs/String  final text per endpoint (punctuated, capitalised)
       /dorai_stt/status         std_msgs/String  JSON {rtf, backlog_s, finals, model, ...}
       /dorai_stt/metrics        std_msgs/String  JSON per final (only with metrics:=true): blind
                                 input SNR estimate, speech/noise dBFS, decode RTF, lag, backlog,
                                 CPU temperature/clock/load (see metrics.py); also appended to
                                 metrics_file if set. Offline WER/CER: `ros2 run stt_mod evaluate`.

Nemotron latency profiles (``latency_ms`` = encoder chunk of the installed export):
   80 / 160 ms  lowest delay, needs a fast CPU (Mac/x86; not a Pi 4)
   560 ms       balanced; the chunk the dorai adapters were trained at
  1120 ms       highest throughput — the only profile near real time on a Pi 4

Decoding runs on a worker thread fed by an unbounded queue, so the ROS callback
never blocks and no audio is dropped when the CPU falls behind; the backlog is
reported on the status topic and warned about above ``backlog_warn_s``.
"""
from __future__ import annotations

import collections
import json
import threading
import time

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node

try:  # raised by spin() when SIGTERM (docker stop, systemd, kill) shuts the context down
    from rclpy.executors import ExternalShutdownException
except ImportError:  # very old rclpy / test mocks
    class ExternalShutdownException(Exception):
        pass
from std_msgs.msg import Float32MultiArray, String

from . import jsonsafe
from .metrics import LevelMeter, SystemProbe
from .models import auto_rule2, default_roots, find_model, find_whisper_model
from .streamer import NemotronStreamer
from .whisper import WhisperStreamer, find_vad_model

FS = 16000

# name -> (default, description). Numbers accept int or double on the CLI.
PARAMS = {
    "input_topic": ("/dorai_clean_audio", "clean audio from voice_mod"),
    "output_topic": ("/dorai_transcript", "final transcripts"),
    "partial_topic": ("/dorai_partial_transcript", "running hypothesis"),
    "status_topic": ("/dorai_stt/status", "JSON status"),
    "metrics_topic": ("/dorai_stt/metrics", "JSON metrics per final (metrics:=true)"),
    # engine
    "engine": ("whisper", "whisper (fine-tuned Whisper, VAD-gated; default) | nemotron (streaming)"),
    "engine_fallback": (True, "whisper model missing -> run nemotron instead of failing"),
    # whisper
    "whisper_model": ("", "Whisper export directory name or its tail; '' = newest dorai fine-tune"),
    "language": ("en", "Whisper language"),
    "vad": ("neural", "utterance gate in front of Whisper: neural | energy | none"),
    "vad_model": ("", "neural VAD file; '' = vad_gated.onnx shipped in this package"),
    "vad_threshold": (0.5, "speech probability threshold of the neural VAD"),
    "vad_min_silence_s": (0.6, "s of silence that ends an utterance"),
    "vad_min_speech_s": (0.25, "shorter utterances are dropped"),
    "vad_max_utt_s": (15.0, "s, an utterance is cut here (Whisper sees at most 30 s)"),
    "vad_pre_roll_s": (0.3, "s of audio kept before the detected start of an utterance"),
    "max_words_per_s": (5.0, "a transcript with more words than this per second of speech is cut"),
    # nemotron
    "latency_ms": (1120, "encoder chunk / latency profile: 80 | 160 | 560 | 1120"),
    "adapter": ("mc-ref", "fine-tuned dorai adapter: mc-ref (default) | mc-das | none"),
    "models_dir": ("", "extra models root searched first"),
    "model_dir": ("", "explicit sherpa-onnx model dir (overrides latency_ms/adapter)"),
    "num_threads": (2, "onnxruntime threads (2 is fastest on a Pi 4)"),
    "block_ms": (200, "feed granularity"),
    # endpointing
    "ep_rule1_sil": (2.0, "s of silence with no text -> reset (floored at rule2+chunk+0.3)"),
    "ep_rule2_sil": (0.0, "s of trailing silence after text -> final; 0 = auto per latency "
                          "profile (80 ms: 0.5, else 0.8; always floored at chunk+0.2)"),
    "ep_max_utt": (20.0, "s, force a final on long utterances"),
    "frame_is_utterance": (False, "true only if the front-end publishes one utterance per frame"),
    "tail_pad_s": (0.8, "silence padded when flushing an utterance"),
    # runtime
    "backlog_warn_s": (5.0, "warn when this much audio waits for decoding"),
    "status_period_s": (5.0, "status topic period"),
    "metrics": (False, "collect per-final metrics: input SNR estimate, RTF, lag, CPU"),
    "metrics_file": ("", "also append metrics as JSON lines to this file"),
    "debug": (False, "verbose logging"),
}


class SttNode(Node):
    def __init__(self):
        super().__init__("stt_mod")
        for k, (v, desc) in PARAMS.items():
            numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
            self.declare_parameter(k, v, ParameterDescriptor(
                description=desc, dynamic_typing=numeric))
        p = {k: self.get_parameter(k).value for k in PARAMS}
        self.p = p
        log = self.get_logger()

        t0 = time.time()
        engine = str(p["engine"]).strip().lower()
        if engine not in ("whisper", "nemotron"):
            raise ValueError(f"engine must be whisper or nemotron, not {p['engine']!r}")
        self.asr = None
        if engine == "whisper":
            try:
                self.asr, self.model_tag, ready = self._make_whisper(p, log)
            except FileNotFoundError as e:
                if not bool(p["engine_fallback"]):
                    raise
                log.error(f"{e} Falling back to engine=nemotron (engine_fallback:=false to fail instead).")
                engine = "nemotron"
                # model_dir, if set, named a Whisper export: let Nemotron search the roots
                self.asr, self.model_tag, ready = self._make_nemotron(dict(p, model_dir=""), log)
        if self.asr is None:
            self.asr, self.model_tag, ready = self._make_nemotron(p, log)
        self.engine = engine

        self.pub = self.create_publisher(String, p["output_topic"], 10)
        self.partial_pub = self.create_publisher(String, p["partial_topic"], 10)
        self.status_pub = self.create_publisher(String, p["status_topic"], 10)
        self.metrics_on = bool(p["metrics"])
        if self.metrics_on:
            self.metrics_pub = self.create_publisher(String, p["metrics_topic"], 50)
            self.meter = LevelMeter()
            self.sysprobe = SystemProbe()
            self._m_last = {"pos_s": 0.0, "compute_s": 0.0, "audio_s": 0.0}
            self._m_file = open(p["metrics_file"], "a") if p["metrics_file"] else None

        self._q = collections.deque()
        self._q_samples = 0
        self._cv = threading.Condition()
        self._stop = False
        self._frame = None            # (t_recv, frame_end_pos_s) of the frame being decoded
        self._busy = False            # worker is inside asr.push()
        self._lags = collections.deque(maxlen=50)
        self._last_seq = None
        self._worker = threading.Thread(target=self._run, name="stt_decode", daemon=True)
        self._worker.start()

        self.sub = self.create_subscription(
            Float32MultiArray, p["input_topic"], self._on_audio, 50)
        self.create_timer(float(p["status_period_s"]), self._status)
        log.info(f"{ready} in {time.time() - t0:.1f}s; {p['input_topic']} -> {p['output_topic']}")

    # -------------------------------------------------------------- engines
    def _make_whisper(self, p, log):
        model_dir, name = find_whisper_model(p["whisper_model"], p["model_dir"], p["models_dir"])
        vad = "none" if bool(p["frame_is_utterance"]) else str(p["vad"]).strip().lower()
        vad = "neural" if vad == "silero" else vad
        vad_model = ""
        if vad == "neural":
            vad_model = find_vad_model(p["vad_model"], model_dir, default_roots(p["models_dir"]))
            if not vad_model:
                log.warning("vad:=neural but vad_gated.onnx was not found (package, model dir, models roots, "
                            "doa_mod); using the energy gate, which also passes loud non-speech.")
                vad = "energy"
        log.info(f"Loading Whisper {name}: {model_dir} (threads={int(p['num_threads'])}, vad={vad}"
                 f"{' ' + vad_model if vad_model else ''})")
        asr = WhisperStreamer(
            model_dir, num_threads=int(p["num_threads"]), vad=vad, vad_model=vad_model,
            vad_threshold=float(p["vad_threshold"]), min_silence_s=float(p["vad_min_silence_s"]),
            min_speech_s=float(p["vad_min_speech_s"]), max_utt_s=float(p["vad_max_utt_s"]),
            pre_roll_s=float(p["vad_pre_roll_s"]),
            language=str(p["language"]), max_words_per_s=float(p["max_words_per_s"]),
            frame_is_utterance=bool(p["frame_is_utterance"]), log=log.info if p["debug"] else None)
        tag = "whisper/" + name[len("sherpa-onnx-whisper-"):] if name.startswith("sherpa-onnx-whisper-") else "whisper/" + name
        return asr, tag, (f"Whisper ready — {tag}, vad={asr.vad_kind}, an utterance ends after "
                          f"{asr.rule2_sil:.2f}s of silence, no partials")

    def _make_nemotron(self, p, log):
        model_dir, lat, adapter = find_model(p["latency_ms"], p["adapter"],
                                             p["model_dir"], p["models_dir"])
        log.info(f"Loading Nemotron {lat} ms, adapter={adapter}: {model_dir} "
                 f"(threads={int(p['num_threads'])})")
        asr = NemotronStreamer(
            model_dir, num_threads=int(p["num_threads"]), block_ms=int(p["block_ms"]),
            rule1_sil=float(p["ep_rule1_sil"]),
            rule2_sil=float(p["ep_rule2_sil"]) or auto_rule2(lat),
            max_utt=float(p["ep_max_utt"]), tail_pad_s=float(p["tail_pad_s"]),
            frame_is_utterance=bool(p["frame_is_utterance"]),
            log=log.info if p["debug"] else None)
        tag = f"nemotron-0.6b/{lat}ms/{adapter}"
        return asr, tag, (f"Nemotron ready — {tag}, rule2={asr.rule2_sil:.2f}s, "
                          f"partials every {int(lat)} ms")

    # ------------------------------------------------------------ ROS input
    def _on_audio(self, msg):
        off = msg.layout.data_offset
        d = msg.data
        if len(d) <= off or off < 4:
            return
        if int(d[1]) != FS or int(d[2]) != 1:
            self.get_logger().warning(
                f"expected 16000 Hz mono, got {int(d[1])} Hz x {int(d[2])}",
                throttle_duration_sec=5.0)
            return
        seq = int(d[3])
        if self._last_seq is not None and seq != self._last_seq + 1:
            self.get_logger().warning(f"clean-audio seq gap {self._last_seq} -> {seq}",
                                      throttle_duration_sec=5.0)
        self._last_seq = seq
        x = np.clip(np.asarray(d[off:], dtype=np.float32), -1.0, 1.0)
        with self._cv:
            self._q.append((time.time(), x))
            self._q_samples += x.size
            self._cv.notify()

    # --------------------------------------------------------------- worker
    def _run(self):
        while True:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait()
                if self._stop and not self._q:
                    return
                t_recv, x = self._q.popleft()
                self._q_samples -= x.size
                self._busy = True
            self._frame = (t_recv, self.asr.pos_s + x.size / FS)
            if self.metrics_on:
                self.meter.push(x)
            try:
                self.asr.push(x, self._on_partial, self._on_final)
            except Exception as e:  # keep the node alive
                self.get_logger().error(f"decode failed: {e}", throttle_duration_sec=2.0)
            finally:
                with self._cv:
                    self._busy = bool(self._q)

    def _on_partial(self, text):
        self.partial_pub.publish(String(data=text))
        if self.p["debug"]:
            self.get_logger().info(f"partial: {text}")

    def _on_final(self, text, pos_s):
        lag = float("nan")
        if self._frame is not None:
            t_recv, end_s = self._frame
            # audio at pos_s arrived with a frame ending at end_s, received at t_recv
            lag = time.time() - t_recv + max(0.0, end_s - pos_s)
            self._lags.append(lag)
        # stream position = seconds of clean audio received before this final;
        # lets an offline harness map each final back to the input timeline.
        self.get_logger().info(f'Transcript: "{text}" (lag {lag:.2f}s, stream {pos_s:.2f}s)')
        self.pub.publish(String(data=text))
        if self.metrics_on:
            self._publish_metrics(text, pos_s, lag)

    def _publish_metrics(self, text, pos_s, lag):
        st, last = self.asr.stats, self._m_last
        d_audio = st["audio_s"] - last["audio_s"]
        d_comp = st["compute_s"] - last["compute_s"]
        m = {"t_wall": round(time.time(), 3), "model": self.model_tag, "text": text,
             "n_words": len(text.split()), "stream_pos_s": round(pos_s, 2),
             "window_s": round(pos_s - last["pos_s"], 2), "lag_s": round(lag, 3),
             # decode speed over the audio decoded since the previous final
             "rtf_window": round(d_comp / d_audio, 3) if d_audio > 0 else None,
             "rtf_total": round(self.asr.rtf(), 3), "backlog_s": round(self.backlog_s(), 2)}
        m.update(self.meter.window(last["pos_s"], pos_s))
        m.update(self.sysprobe.sample())
        self._m_last = {"pos_s": pos_s, "compute_s": st["compute_s"], "audio_s": st["audio_s"]}
        js = jsonsafe.dumps(m)
        self.metrics_pub.publish(String(data=js))
        if self._m_file:
            self._m_file.write(js + "\n")
            self._m_file.flush()
        if self.p["debug"]:
            self.get_logger().info(f"metrics: {js}")

    # --------------------------------------------------------------- status
    def backlog_s(self):
        return self._q_samples / FS

    def _status(self):
        b = self.backlog_s()
        st = {"model": self.model_tag, "rtf": round(self.asr.rtf(), 3),
              "backlog_s": round(b, 2), "busy": self._busy, "audio_s": round(self.asr.stats["audio_s"], 1),
              "finals": self.asr.stats["finals"],
              "lag_median_s": round(float(np.median(self._lags)), 2) if self._lags else None}
        self.status_pub.publish(String(data=jsonsafe.dumps(st)))
        if b > float(self.p["backlog_warn_s"]):
            self.get_logger().warning(
                f"STT is behind real time: {b:.1f}s audio queued (RTF {st['rtf']}). "
"Give it more threads (num_threads), use engine:=nemotron latency_ms:=1120, "
                "or run stt_mod on a faster host.",
                throttle_duration_sec=10.0)
        elif self.p["debug"]:
            self.get_logger().info(jsonsafe.dumps(st))

    def shutdown(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._worker.join(timeout=30.0)
        try:
            text = self.asr.finalize(self._on_partial, None)
            if text:
                self._on_final(text, self.asr.pos_s)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = SttNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass        # SIGINT / SIGTERM: drain and flush below, exit 0
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
