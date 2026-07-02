#!/usr/bin/env python3
"""
stt_mod / stt.py  —  Stage 2 of the dorai speech pipeline.

Consumes /dorai_clean_audio (1 ch, 16 kHz, N-second frames) and publishes
recognized text to /dorai_transcript. Two interchangeable STT engines, chosen
with the `engine` parameter:

  vosk     streaming Kaldi recognizer; emits partials on /dorai_partial_transcript
           and finals on /dorai_transcript.
  whisper  OpenAI Whisper (whisper-tiny) via faster-whisper (CTranslate2, int8).
           Whisper is a whole-utterance model, so each N-second clean frame is
           transcribed in one pass — a natural fit for this batched pipeline.
           No partials (batch model); one final per frame.

Examples:
  ros2 run stt_mod stt --ros-args -p engine:=vosk
  ros2 run stt_mod stt --ros-args -p engine:=whisper -p whisper_model:=tiny.en
  ros2 run stt_mod stt --ros-args -p engine:=whisper -p debug:=true
"""

import os
import sys
import json
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

try:
    from vosk import Model, KaldiRecognizer
except ImportError:
    pass

try:
    from faster_whisper import WhisperModel
except ImportError:
    pass


class SttMod(Node):
    def __init__(self):
        super().__init__("stt_mod")

        # Dynamic typing lets numeric params accept int or double on the CLI.
        from rcl_interfaces.msg import ParameterDescriptor
        dyn = ParameterDescriptor(dynamic_typing=True)

        self.declare_parameter("engine", "vosk")            # vosk | whisper
        self.declare_parameter("input_topic", "/dorai_clean_audio")
        self.declare_parameter("output_topic", "/dorai_transcript")
        self.declare_parameter("partial_topic", "/dorai_partial_transcript")
        self.declare_parameter("debug", False)              # verbose STT logging

        # Vosk
        self.declare_parameter("model_path", "")            # vosk model dir
        self.declare_parameter("model_lang", "en-us")

        # Whisper (faster-whisper)
        self.declare_parameter("whisper_model", "tiny")     # tiny | tiny.en | base...
        self.declare_parameter("whisper_compute_type", "int8")
        self.declare_parameter("language", "en")            # "" = auto-detect
        self.declare_parameter("whisper_beam_size", 1, dyn)
        self.declare_parameter("num_threads", 4, dyn)       # CT2 CPU threads
        # initial_prompt biases decoding toward an expected vocabulary. For a
        # fixed command set this reliably resolves ambiguous words (e.g.
        # "weather" otherwise misheard as "major"/"video"). Keep it a short list
        # of the phrases dorai must recognize, not a paragraph. "" = none.
        self.declare_parameter("initial_prompt", "")

        gp = self.get_parameter
        self.engine = gp("engine").value.strip().lower()
        input_topic = gp("input_topic").value
        output_topic = gp("output_topic").value
        partial_topic = gp("partial_topic").value
        self.debug = bool(gp("debug").value)

        self.get_logger().info(f"STT engine: {self.engine}")
        self.get_logger().info(f"Subscribing to: {input_topic}")
        self.get_logger().info(f"Publishing transcript to: {output_topic}")

        if self.engine == "whisper":
            self._init_whisper()
        elif self.engine == "vosk":
            self.get_logger().info(f"Publishing partials to: {partial_topic}")
            self._init_vosk()
        else:
            raise ValueError(f"Unknown engine '{self.engine}' (use vosk|whisper)")

        self.sub = self.create_subscription(
            Float32MultiArray, input_topic, self.on_clean_audio, 10)
        self.pub = self.create_publisher(String, output_topic, 10)
        self.partial_pub = self.create_publisher(String, partial_topic, 10)

    # ------------------------------------------------------------------ Vosk
    def _init_vosk(self):
        if "Model" not in globals():
            self.get_logger().error(
                "vosk not installed. Run: pip install vosk")
            raise ImportError("Vosk not installed.")
        model_path = self.get_parameter("model_path").value
        model_lang = self.get_parameter("model_lang").value
        try:
            if model_path:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model path not found: {model_path}")
                self.get_logger().info(f"Loading Vosk model: {model_path}")
                self.model = Model(model_path)
            else:
                self.get_logger().info(f"Loading Vosk model lang={model_lang}")
                self.model = Model(lang=model_lang)
            self.rec = KaldiRecognizer(self.model, 16000)
            self.get_logger().info("Vosk recognizer ready.")
        except Exception as e:
            self.get_logger().error(f"Failed to init Vosk: {e}")
            raise

    # --------------------------------------------------------------- Whisper
    def _init_whisper(self):
        if "WhisperModel" not in globals():
            self.get_logger().error(
                "faster-whisper not installed. Run: pip install faster-whisper")
            raise ImportError("faster-whisper not installed.")
        name = self.get_parameter("whisper_model").value
        compute = self.get_parameter("whisper_compute_type").value
        self.language = self.get_parameter("language").value or None
        self.beam_size = int(self.get_parameter("whisper_beam_size").value)
        nthreads = int(self.get_parameter("num_threads").value)
        self.initial_prompt = self.get_parameter("initial_prompt").value or None
        try:
            self.get_logger().info(
                f"Loading Whisper '{name}' (compute={compute}, "
                f"threads={nthreads}); first run downloads the model...")
            self.wmodel = WhisperModel(
                name, device="cpu", compute_type=compute,
                cpu_threads=nthreads if nthreads > 0 else 0)
            self.get_logger().info(
                f"Whisper ready (lang={self.language or 'auto'}, "
                f"beam={self.beam_size}).")
        except Exception as e:
            self.get_logger().error(f"Failed to init Whisper: {e}")
            raise

    # ------------------------------------------------------------- callback
    def on_clean_audio(self, msg):
        data_offset = msg.layout.data_offset
        if len(msg.data) < data_offset:
            return
        sample_rate = int(msg.data[1])
        num_channels = int(msg.data[2])
        if sample_rate != 16000:
            self.get_logger().warning(
                f"Expected 16000 Hz, got {sample_rate} Hz.",
                throttle_duration_sec=2.0)
        if num_channels != 1:
            self.get_logger().warning(
                f"Expected 1 channel, got {num_channels}.",
                throttle_duration_sec=2.0)

        clean_audio = np.array(msg.data[data_offset:], dtype=np.float32)
        if clean_audio.size == 0:
            return

        rms = float(np.sqrt(np.mean(clean_audio ** 2)))
        self.get_logger().info(
            f"Processing clean audio: {clean_audio.size} samples "
            f"(rate={sample_rate}Hz, rms={rms:.5f})",
            throttle_duration_sec=5.0)

        clean_audio = np.clip(clean_audio, -1.0, 1.0)
        if self.engine == "whisper":
            self._run_whisper(clean_audio)
        else:
            self._run_vosk(clean_audio)

    def _run_vosk(self, clean_audio):
        pcm16 = (clean_audio * 32767.0).astype(np.int16).tobytes()
        if self.rec.AcceptWaveform(pcm16):
            res_str = self.rec.Result()
            text = json.loads(res_str).get("text", "").strip()
            if text:
                sys.stdout.write("\r" + " " * 80 + "\r"); sys.stdout.flush()
                self.get_logger().info(f"Transcript: \"{text}\"")
                self.pub.publish(String(data=text))
            else:
                self.get_logger().info(
                    "Vosk: utterance end, no words recognized (empty result).")
            if self.debug:
                self.get_logger().info(f"Vosk raw result: {res_str.strip()}")
        else:
            partial = json.loads(self.rec.PartialResult()).get("partial", "").strip()
            if partial:
                sys.stdout.write(f"\rPartial: {partial}"); sys.stdout.flush()
                self.partial_pub.publish(String(data=partial))
                if self.debug:
                    self.get_logger().info(f"Vosk partial: \"{partial}\"")

    def _run_whisper(self, clean_audio):
        try:
            segments, info = self.wmodel.transcribe(
                clean_audio, language=self.language,
                beam_size=self.beam_size,
                # temperature=0.0 (single value) disables the fallback ladder
                # that retries at higher temperatures and invents confident
                # wrong words ("Good luck", "Absolutely") — it also makes each
                # call fast. condition_on_previous_text=False stops cross-frame
                # repetition loops. Proven correct in lab/whisper_test.py.
                temperature=0.0,
                condition_on_previous_text=False,
                # Hard-stop the in-window repetition loops ("see what's in here,
                # see what's in here, ...") Whisper falls into on hard/ambiguous
                # audio: forbid repeating any 3-gram and penalize repeats. Also
                # bounds the transcribe time (those loops cost 40+ s).
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                # Bias decoding toward the expected command vocabulary.
                initial_prompt=self.initial_prompt)
            text = "".join(seg.text for seg in segments).strip()
        except Exception as e:
            self.get_logger().error(
                f"Whisper transcribe failed: {e}", throttle_duration_sec=2.0)
            return
        if text:
            self.get_logger().info(f"Transcript: \"{text}\"")
            self.pub.publish(String(data=text))
        else:
            self.get_logger().info("Whisper: no speech recognized in frame.")
        if self.debug:
            lang = getattr(info, "language", "?")
            prob = getattr(info, "language_probability", 0.0)
            self.get_logger().info(
                f"Whisper info: lang={lang} p={prob:.2f}")

    def shutdown(self):
        if self.engine != "vosk":
            return
        try:
            text = json.loads(self.rec.FinalResult()).get("text", "").strip()
            if text:
                self.get_logger().info(f"Final transcript on shutdown: \"{text}\"")
                self.pub.publish(String(data=text))
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = SttMod()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
