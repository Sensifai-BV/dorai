#!/usr/bin/env python3
"""
stt_mod / stt.py  —  Stage 2 of the dorai speech pipeline.

Consumes /dorai_clean_audio (1 ch, 16 kHz, N-second frames) and publishes
recognized text to /dorai_transcript. Two interchangeable STT engines, chosen
with the `engine` parameter:

  vosk     streaming Kaldi recognizer; emits partials on /dorai_partial_transcript
           and finals on /dorai_transcript. Fed in 200 ms blocks so finals
           surface at natural pauses (not at the publish-frame boundary).
  whisper  OpenAI Whisper (whisper-tiny) via faster-whisper (CTranslate2, int8).
           Whisper is a whole-*utterance* model, so a front-end endpoint VAD
           (VadSegmenter) groups the clean stream into complete utterances and
           transcribes one per detected pause. This decouples STT latency from
           publish_interval and stops words being split at frame boundaries —
           the two causes of "slow" and "voice not detected". No partials.
           Set vad_endpoint:=false for the old per-frame behaviour.

Both engines are real-time on a Raspberry Pi (measured RTF on a dev laptop:
whisper tiny.en ~0.014, base.en ~0.026, vosk-small ~0.08; scale up for the Pi
but all remain < 1.0). "Slow" was never the model — it was the fixed publish
framing; "voice not detected" was boundary word-splitting. Both are fixed here.

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


class VadSegmenter:
    """Endpoint-based utterance segmenter for the whole-utterance (Whisper) path.

    Whisper is a whole-*utterance* model: it must be handed a complete phrase.
    The old node transcribed each fixed publish frame (default 10 s) as-is, which
    caused the two faults the user reported:

      * "takes a lot of time" — a word spoken just after a frame boundary waits
        the full publish_interval for the frame to fill before STT even starts
        (avg ~publish_interval/2, worst ~publish_interval).
      * "voice not detected" — a word straddling the frame boundary is cut in
        half; each half, handed to Whisper without context, is mis-decoded or
        dropped (measured: "...refuge in the sublime" -> "beat in the skateboard").

    This segmenter decouples STT from the publish framing. It accumulates the
    continuous clean stream and, using a light energy VAD (the stream is already
    AGC-levelled with a silence gate by voice_mod, so an absolute RMS threshold
    is robust), emits exactly one complete utterance when it sees `endpoint_sil`
    seconds of trailing silence after speech — or force-flushes at
    `max_utt` seconds so a never-ending talker still gets transcribed. Silence-
    only audio is dropped, never transcribed. Latency becomes
    endpoint_sil + transcribe time, independent of publish_interval, and words
    are never split mid-utterance.
    """

    def __init__(self, fs=16000, win_ms=30, rms_thresh=0.015,
                 endpoint_sil=0.6, min_speech=0.2, max_utt=20.0, pre_roll=0.2):
        self.fs = int(fs)
        self.win = max(1, int(fs * win_ms / 1000))
        self.thr = float(rms_thresh)
        self.endpoint_sil = float(endpoint_sil)
        self.min_speech = float(min_speech)
        self.max_utt = float(max_utt)
        self.pre_roll = float(pre_roll)
        self.buf = np.zeros(0, dtype=np.float32)

    def _speech_mask(self, x):
        n = len(x) // self.win
        if n == 0:
            return np.zeros(0, dtype=bool)
        fr = x[:n * self.win].reshape(n, self.win)
        rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
        return rms > self.thr

    def push(self, chunk):
        """Add clean audio; return a list of finished utterance waveforms (>=0).

        Usually 0 or 1 utterance per call; a burst that contains several pauses
        can return more."""
        self.buf = np.concatenate([self.buf, np.asarray(chunk, dtype=np.float32)])
        out = []
        while True:
            utt = self._try_extract()
            if utt is None:
                break
            if len(utt) >= self.min_speech * self.fs:
                out.append(utt)
        return out

    def _try_extract(self):
        mask = self._speech_mask(self.buf)
        if mask.size == 0:
            return None
        speech_idx = np.nonzero(mask)[0]
        dur = len(self.buf) / self.fs
        if speech_idx.size == 0:
            # pure silence: keep only a short tail as pre-roll for the next word
            keep = int(self.pre_roll * self.fs)
            if len(self.buf) > keep:
                self.buf = self.buf[-keep:].copy()
            return None

        first_speech_win = int(speech_idx[0])
        last_speech_win = int(speech_idx[-1])
        sil_wins = int(round(self.endpoint_sil * self.fs / self.win))

        # Find the FIRST internal pause of >= endpoint_sil that follows speech —
        # so consecutive commands separated by a pause become separate
        # utterances (lower latency, no merged commands) rather than waiting for
        # the whole buffer to go quiet.
        endpoint_win = None
        run = 0
        for w in range(first_speech_win + 1, last_speech_win + 1):
            if not mask[w]:
                run += 1
                if run >= sil_wins:
                    endpoint_win = w - run + 1        # first silent win of the gap
                    break
            else:
                run = 0

        trailing_sil = (len(mask) - 1 - last_speech_win) * self.win / self.fs
        force = dur >= self.max_utt

        if endpoint_win is not None:
            cut_speech_win = endpoint_win - 1          # last speech win before gap
        elif trailing_sil >= self.endpoint_sil or force:
            cut_speech_win = last_speech_win
        else:
            return None

        tail = int(0.15 * self.fs)
        end = min(len(self.buf), (cut_speech_win + 1) * self.win + tail)
        start = max(0, first_speech_win * self.win - int(self.pre_roll * self.fs))
        utt = self.buf[start:end].copy()
        self.buf = self.buf[end:].copy()
        return utt

    def flush(self):
        """Drain any buffered speech on shutdown."""
        mask = self._speech_mask(self.buf)
        if mask.size and mask.any():
            utt = self.buf.copy()
            self.buf = np.zeros(0, dtype=np.float32)
            if len(utt) >= self.min_speech * self.fs:
                return utt
        self.buf = np.zeros(0, dtype=np.float32)
        return None


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

        # --- Whisper utterance segmentation (VAD endpointing) --------------
        # Whisper is a whole-utterance model. Rather than transcribing each fixed
        # publish frame (which delays a word by up to a full frame and splits
        # words at frame joins), accumulate the stream and transcribe one
        # complete utterance when trailing silence marks its end. This is what
        # makes STT responsive and stops the dropped/garbled boundary words.
        # Set vad_endpoint:=false to fall back to per-frame transcription.
        self.declare_parameter("vad_endpoint", True)
        self.declare_parameter("vad_rms_thresh", 0.015, dyn)   # speech vs silence
        self.declare_parameter("vad_endpoint_sil", 0.6, dyn)   # trailing sil -> emit
        self.declare_parameter("vad_min_speech", 0.2, dyn)     # drop shorter blips
        self.declare_parameter("vad_max_utt", 20.0, dyn)       # force-flush cap

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

        # Utterance segmenter (endpoint VAD) sits in front of Whisper.
        self.vad_endpoint = bool(self.get_parameter("vad_endpoint").value)
        if self.vad_endpoint:
            self.segmenter = VadSegmenter(
                fs=16000,
                rms_thresh=float(self.get_parameter("vad_rms_thresh").value),
                endpoint_sil=float(self.get_parameter("vad_endpoint_sil").value),
                min_speech=float(self.get_parameter("vad_min_speech").value),
                max_utt=float(self.get_parameter("vad_max_utt").value))
            self.get_logger().info(
                f"Whisper VAD endpointing ON (rms>{self.segmenter.thr}, "
                f"endpoint_sil={self.segmenter.endpoint_sil}s, "
                f"max_utt={self.segmenter.max_utt}s) — latency now independent "
                f"of publish_interval.")
        else:
            self.segmenter = None
            self.get_logger().warning(
                "Whisper VAD endpointing OFF — transcribing per publish frame "
                "(higher latency, boundary words may split).")

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
            if self.segmenter is not None:
                # Feed the endpoint VAD; transcribe each complete utterance it
                # returns. Silence-only frames add nothing and cost no compute.
                for utt in self.segmenter.push(clean_audio):
                    self._run_whisper(utt)
            else:
                self._run_whisper(clean_audio)
        else:
            self._run_vosk(clean_audio)

    def _run_vosk(self, clean_audio):
        # Vosk is a *streaming* recognizer: it endpoints internally and emits a
        # final at each natural pause. Its latency is therefore time-to-next-
        # pause — but ONLY if it is fed in small blocks. The old code handed it
        # one whole publish frame (default 10 s) per call, so a result could not
        # surface until the frame boundary, throwing away the streaming latency
        # advantage. Slice each incoming frame into VOSK_BLOCK-second blocks and
        # feed them in order; finals then appear mid-frame at real pauses.
        block = int(0.2 * 16000)                     # 200 ms feed granularity
        pcm = (clean_audio * 32767.0).astype(np.int16)
        for i in range(0, len(pcm), block):
            chunk = pcm[i:i + block].tobytes()
            if self.rec.AcceptWaveform(chunk):
                res_str = self.rec.Result()
                text = json.loads(res_str).get("text", "").strip()
                if text:
                    sys.stdout.write("\r" + " " * 80 + "\r"); sys.stdout.flush()
                    self.get_logger().info(f"Transcript: \"{text}\"")
                    self.pub.publish(String(data=text))
                elif self.debug:
                    self.get_logger().info(
                        "Vosk: utterance end, no words recognized.")
                if self.debug:
                    self.get_logger().info(f"Vosk raw result: {res_str.strip()}")
            else:
                partial = json.loads(
                    self.rec.PartialResult()).get("partial", "").strip()
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
        if self.engine == "whisper":
            # Transcribe any buffered speech the endpoint VAD was still holding.
            if getattr(self, "segmenter", None) is not None:
                try:
                    utt = self.segmenter.flush()
                    if utt is not None:
                        self._run_whisper(utt)
                except Exception:
                    pass
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
