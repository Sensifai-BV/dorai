#!/usr/bin/env python3
"""
stt_mod / stt.py  —  Stage 2 of the dorai speech pipeline.

Consumes /dorai_clean_audio (1 ch, 16 kHz, N-second frames) and publishes
recognized text to /dorai_transcript. Three interchangeable STT engines, chosen
with the `engine` parameter:

  nemotron (default) NVIDIA nemotron-speech-streaming-en-0.6b — a cache-aware
           streaming FastConformer + RNN-T transducer (600 M params, int8 ONNX)
           run through sherpa-onnx on the CPU. Natively streaming: it is fed the
           clean stream in 200 ms blocks, publishes the growing hypothesis on
           /dorai_partial_transcript and a punctuated, capitalised final on
           /dorai_transcript at each endpoint / frame end. Far more accurate
           than tiny Whisper or Vosk-small, no whole-utterance buffering, no
           hallucinated fallbacks. Model files: the k2-fsa sherpa-onnx export
           in ~/dorai/models (see README, section "Nemotron").
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
  ros2 run stt_mod stt                                    # nemotron (default)
  ros2 run stt_mod stt --ros-args -p engine:=nemotron -p num_threads:=4 -p debug:=true
  ros2 run stt_mod stt --ros-args -p engine:=vosk
  ros2 run stt_mod stt --ros-args -p engine:=whisper -p whisper_model:=tiny.en
  ros2 run stt_mod stt --ros-args -p engine:=whisper -p debug:=true
"""

import os
import re
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

try:
    import sherpa_onnx
except ImportError:
    pass


# --------------------------------------------------------------------------
# Nemotron (NVIDIA nemotron-speech-streaming-en-0.6b) via sherpa-onnx
# --------------------------------------------------------------------------
NEMOTRON_DIR_PREFIX = "sherpa-onnx-nemotron-speech-streaming-en-0.6b"
NEMOTRON_DEFAULT_ROOTS = (
    os.environ.get("DORAI_MODELS_DIR", ""),
    os.path.join(os.path.expanduser("~"), "dorai", "models"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
    "/root/.cache/dorai/models",
)


def find_nemotron_model_dir(explicit="", chunk_ms=0):
    """Locate the sherpa-onnx export of nemotron-speech-streaming-en-0.6b.

    Order: the explicit parameter, then $DORAI_MODELS_DIR, ~/dorai/models,
    <repo>/models and /root/.cache/dorai/models. In the first root that has
    any directory starting with NEMOTRON_DIR_PREFIX, prefer one whose name
    carries "<chunk_ms>ms" (the export's encoder chunk; 80/160/560/1120), then
    the newest. chunk_ms=0 = no preference. Returns None when nothing is found.

    Chunk size is a latency/throughput knob: partials arrive every chunk, but
    a larger chunk amortises the cache-aware encoder better — on a Pi 4 the
    1120 ms export decodes ~1.4x faster than the 560 ms one (see CLAUDE.md)."""
    if explicit:
        return os.path.expanduser(explicit) if os.path.isdir(
            os.path.expanduser(explicit)) else None
    for root in NEMOTRON_DEFAULT_ROOTS:
        if not root or not os.path.isdir(root):
            continue
        cands = [os.path.join(root, d) for d in os.listdir(root)
                 if d.startswith(NEMOTRON_DIR_PREFIX)
                 and os.path.isdir(os.path.join(root, d))]
        if not cands:
            continue
        if chunk_ms:
            tagged = [d for d in cands if f"-{int(chunk_ms)}ms-" in os.path.basename(d)]
            if tagged:
                cands = tagged
        return max(cands, key=os.path.getmtime)
    return None


def _pick(model_dir, prefix):
    """Return the single <prefix>*.onnx in model_dir (prefer int8 if several)."""
    names = sorted(n for n in os.listdir(model_dir)
                   if n.startswith(prefix) and n.endswith(".onnx"))
    if not names:
        raise FileNotFoundError(f"no {prefix}*.onnx in {model_dir}")
    int8 = [n for n in names if "int8" in n]
    return os.path.join(model_dir, (int8 or names)[0])


class NemotronStreamer:
    """Native streaming STT front-end around sherpa-onnx's OnlineRecognizer.

    nemotron-speech-streaming-en-0.6b is a cache-aware FastConformer encoder
    with an RNN-T decoder: it is *designed* to be fed audio incrementally and
    emits tokens as they are recognised, with punctuation and capitalisation.
    That removes the two compromises of the other engines — Whisper needs a
    complete utterance (hence the VadSegmenter), Vosk streams but with a small
    Kaldi model whose accuracy is the reason this engine exists.

    Feeding contract
      push(samples, on_partial, on_final)
        `samples` is float32 mono 16 kHz in [-1, 1] — one /dorai_clean_audio
        frame. It is fed in `block_ms` blocks and decoded as the recognizer
        becomes ready; every change of the running hypothesis is reported via
        on_partial(text). Two things produce a *final*:
          * an endpoint inside the frame (trailing silence after speech >=
            rule2, or no speech for rule1, or utterance > rule3) — the
            recognizer is reset and continues on the same stream;
          * frame_is_utterance=True: the end of the frame itself — for a
            front-end that publishes one complete utterance per frame; we pad
            `tail_pad_s` of silence so the encoder's chunk look-ahead flushes
            the last tokens, read the final text and start a fresh stream.
        With frame_is_utterance=False (default; voice_mod publishes fixed
        publish_interval frames, so frame edges fall mid-word) the stream is
        continuous across frames and only endpoints produce finals.

    Model files are the k2-fsa export (encoder/decoder/joiner .onnx + tokens.txt)
    downloaded from the sherpa-onnx `asr-models` release, e.g.
    sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25 (the
    updated March-2026 NVIDIA checkpoint, int8, 560 ms chunk).
    """

    def __init__(self, model_dir, num_threads=4, block_ms=200,
                 rule1_sil=2.0, rule2_sil=0.8, max_utt=20.0,
                 tail_pad_s=0.8, frame_is_utterance=False, fs=16000,
                 debug_log=None):
        if "sherpa_onnx" not in globals():
            raise ImportError(
                "sherpa-onnx not installed. Run: pip install sherpa-onnx")
        if model_dir is None or not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Nemotron model dir not found: {model_dir!r}. Download the "
                f"sherpa-onnx export into ~/dorai/models (see README).")
        self.model_dir = model_dir
        self.fs = int(fs)
        self.block = max(1, int(self.fs * block_ms / 1000))
        self.tail_pad = int(self.fs * float(tail_pad_s))
        self.frame_is_utterance = bool(frame_is_utterance)
        self.log = debug_log or (lambda s: None)

        tokens = os.path.join(model_dir, "tokens.txt")
        if not os.path.exists(tokens):
            raise FileNotFoundError(f"tokens.txt missing in {model_dir}")

        # Encoder chunk (from the export's directory name, e.g. "-1120ms-";
        # 560 ms if untagged). The recognizer only decodes once a whole chunk
        # is available, so (a) sherpa-onnx measures trailing silence in
        # chunk-sized bursts — with rule2 < chunk a pause inside one chunk can
        # fire a false endpoint mid-sentence (seen: "I was a" | "Dreamer, ..."
        # with the 1120 ms export at rule2=0.8) — hence rule2 is floored at
        # chunk + 0.2 s; (b) up to one chunk of fed audio is still undecoded
        # when an endpoint is reported, which sizes the replay window below.
        m = re.search(r"-(\d+)ms-", os.path.basename(model_dir.rstrip("/")))
        self.chunk_s = int(m.group(1)) / 1000.0 if m else 0.56
        rule2_sil = float(rule2_sil)
        if rule2_sil < self.chunk_s + 0.2:
            self.log(f"rule2_sil {rule2_sil}s < chunk {self.chunk_s}s+0.2 -> "
                     f"raising to {self.chunk_s + 0.2:.2f}s")
            rule2_sil = self.chunk_s + 0.2
        self.rule2_sil = rule2_sil
        # rule1 (silence with nothing decoded) must exceed the rollover replay
        # window (rule2 + chunk, see below) or the replayed tail alone re-fires
        # rule1 on the fresh stream, and every block then costs a full replay
        # (measured on silence: RTF 16.8 vs 1.7 on speech on the Pi).
        rule1_sil = float(rule1_sil)
        if rule1_sil < self.rule2_sil + self.chunk_s + 0.3:
            rule1_sil = self.rule2_sil + self.chunk_s + 0.3
        self.rule1_sil = rule1_sil
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=_pick(model_dir, "encoder"),
            decoder=_pick(model_dir, "decoder"),
            joiner=_pick(model_dir, "joiner"),
            num_threads=int(num_threads),
            sample_rate=self.fs,
            feature_dim=80,
            # Only greedy_search is implemented for the Nemotron transducer in
            # sherpa-onnx (modified_beam_search / hotwords are not available).
            decoding_method="greedy_search",
            model_type="nemotron",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=self.rule1_sil,
            rule2_min_trailing_silence=self.rule2_sil,
            rule3_min_utterance_length=float(max_utt),
            provider="cpu",
        )
        self.stream = self.rec.create_stream()
        self._last_partial = ""
        self.stats = {"audio_s": 0.0, "compute_s": 0.0, "finals": 0}
        # Endpoint rollover replay. The encoder decodes in `chunk_s` chunks, so
        # when is_endpoint() turns true we have typically already fed up to a
        # chunk of audio past the decoded position — and if the talker has
        # started the next phrase in that window, recognizer.reset() throws
        # those samples away (measured: "…on end [tucked away in] my corner"
        # lost the bracketed words). Instead of reset() we start a fresh
        # stream and replay exactly the UNDECODED tail into it (fed samples
        # minus decode_stream calls x chunk, tracked per stream) plus a small
        # margin that lies inside the >= rule2 trailing silence. A fixed
        # rule2+chunk window duplicated the last word whenever little was
        # undecoded ("...a lovely child" | "Child whose place..."), while a
        # margin of only 0.35 s lost onsets ("[God] as a direct...") because
        # the transducer emits tokens a few frames late, so the decoded
        # position trails the chunk boundary. A margin equal to rule2 (the
        # measured, hence guaranteed, silence) was loss- and duplicate-free on
        # every test set (margin sweep 0.35/0.5/0.65/0.8).
        self.replay_margin_s = min(0.8, self.rule2_sil)
        self._tail = np.zeros(0, dtype=np.float32)   # last >= chunk+margin s
        self._tail_n = int(self.fs * (self.chunk_s + self.replay_margin_s + 0.2))
        self._fed = 0        # samples fed to the current stream
        self._ndec = 0       # decode_stream calls on the current stream

    # -- helpers ---------------------------------------------------------
    def _drain(self, on_partial):
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
            self._ndec += 1
        text = self.rec.get_result(self.stream).strip()
        if text and text != self._last_partial:
            self._last_partial = text
            if on_partial:
                on_partial(text)
        return text

    def _feed(self, blk):
        self.stream.accept_waveform(self.fs, blk)
        self._fed += len(blk)
        self._tail = np.concatenate([self._tail, blk])[-self._tail_n:]

    def _undecoded_tail(self):
        """Samples fed to the current stream that the encoder has not consumed
        yet (0..chunk), plus replay_margin_s of the preceding silence."""
        und = self._fed / self.fs - self._ndec * self.chunk_s
        und = min(max(und, 0.0), self.chunk_s + 0.3) + self.replay_margin_s
        n = int(und * self.fs)
        return self._tail[-n:] if n > 0 else self._tail[:0]

    def _new_stream(self, replay=None, on_partial=None):
        self.stream = self.rec.create_stream()
        self._fed = 0
        self._ndec = 0
        if replay is not None and replay.size:
            self._feed(replay)
            self._drain(on_partial)

    def _emit_final(self, text, on_final):
        self._last_partial = ""
        if text:
            self.stats["finals"] += 1
            if on_final:
                on_final(text)

    # -- public ----------------------------------------------------------
    def push(self, samples, on_partial=None, on_final=None):
        """Feed one clean frame. See class docstring for the contract."""
        import time
        x = np.asarray(samples, dtype=np.float32)
        if x.size == 0:
            return
        t0 = time.perf_counter()
        for i in range(0, len(x), self.block):
            self._feed(x[i:i + self.block])
            text = self._drain(on_partial)
            if self.rec.is_endpoint(self.stream):
                tail = self._undecoded_tail()
                if text:
                    self.log(f"endpoint inside frame -> final {text!r} "
                             f"(replay {len(tail) / self.fs:.2f}s)")
                    self._emit_final(text, on_final)
                    self._new_stream(tail, on_partial)
                elif tail.size and float(np.abs(tail).max()) > 0.02:
                    # rule1 (long silence, nothing decoded) but the undecoded
                    # tail carries energy: it may be a phrase onset — keep it.
                    self._new_stream(tail, on_partial)
                else:
                    self.rec.reset(self.stream)   # silence: nothing to lose
        if self.frame_is_utterance:
            self.finalize(on_partial, on_final)
        self.stats["audio_s"] += len(x) / self.fs
        self.stats["compute_s"] += time.perf_counter() - t0

    def finalize(self, on_partial=None, on_final=None):
        """Flush the current stream as one final and start a fresh stream."""
        if self.tail_pad:
            self.stream.accept_waveform(
                self.fs, np.zeros(self.tail_pad, dtype=np.float32))
        self.stream.input_finished()
        text = self._drain(on_partial)
        self._new_stream()
        self._tail = np.zeros(0, dtype=np.float32)
        self._emit_final(text, on_final)
        return text

    def rtf(self):
        a = self.stats["audio_s"]
        return self.stats["compute_s"] / a if a > 0 else float("nan")


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

        self.declare_parameter("engine", "nemotron")        # nemotron | whisper | vosk
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

        # Nemotron (nvidia/nemotron-speech-streaming-en-0.6b via sherpa-onnx)
        # "" = auto-locate the sherpa-onnx export under ~/dorai/models (see
        # find_nemotron_model_dir). num_threads above is shared.
        self.declare_parameter("nemotron_model_dir", "")
        # Preferred export chunk size when several exports are installed
        # (80/160/560/1120 ms; 0 = newest). 1120 is the fastest on a Pi 4.
        self.declare_parameter("nemotron_chunk_ms", 1120, dyn)
        self.declare_parameter("nemotron_block_ms", 200, dyn)   # feed granularity
        # Endpointing (sherpa-onnx rules): rule1 = trailing silence with NO text
        # decoded yet; rule2 = trailing silence AFTER text (this is the one
        # that ends a spoken command); rule3 = force-final on long utterances.
        self.declare_parameter("nemotron_ep_rule1_sil", 2.0, dyn)
        self.declare_parameter("nemotron_ep_rule2_sil", 0.8, dyn)
        self.declare_parameter("nemotron_ep_max_utt", 20.0, dyn)
        # False (default): the clean stream is continuous across frames —
        # voice_mod publishes FIXED publish_interval frames, so a frame edge
        # is not an utterance edge and finals come from detected endpoints
        # (rule2 trailing silence). True: each frame is one complete utterance
        # (a VAD-segmenting front-end that publishes one utterance per frame)
        # -> final at frame end, no waiting for trailing silence.
        self.declare_parameter("nemotron_frame_is_utterance", False)
        self.declare_parameter("nemotron_tail_pad_s", 0.8, dyn)

        gp = self.get_parameter
        self.engine = gp("engine").value.strip().lower()
        input_topic = gp("input_topic").value
        output_topic = gp("output_topic").value
        partial_topic = gp("partial_topic").value
        self.debug = bool(gp("debug").value)

        self.get_logger().info(f"STT engine: {self.engine}")
        self.get_logger().info(f"Subscribing to: {input_topic}")
        self.get_logger().info(f"Publishing transcript to: {output_topic}")

        if self.engine == "nemotron":
            self.get_logger().info(f"Publishing partials to: {partial_topic}")
            self._init_nemotron()
        elif self.engine == "whisper":
            self._init_whisper()
        elif self.engine == "vosk":
            self.get_logger().info(f"Publishing partials to: {partial_topic}")
            self._init_vosk()
        else:
            raise ValueError(
                f"Unknown engine '{self.engine}' (use nemotron|whisper|vosk)")

        self.sub = self.create_subscription(
            Float32MultiArray, input_topic, self.on_clean_audio, 10)
        self.pub = self.create_publisher(String, output_topic, 10)
        self.partial_pub = self.create_publisher(String, partial_topic, 10)

    # -------------------------------------------------------------- Nemotron
    def _init_nemotron(self):
        gp = self.get_parameter
        model_dir = find_nemotron_model_dir(
            gp("nemotron_model_dir").value, int(gp("nemotron_chunk_ms").value))
        if model_dir is None:
            self.get_logger().error(
                "Nemotron model not found. Download the sherpa-onnx export "
                "(see README 'Nemotron') into ~/dorai/models, or pass "
                "-p nemotron_model_dir:=/path/to/sherpa-onnx-nemotron-...")
            raise FileNotFoundError("nemotron model dir")
        nthreads = int(gp("num_threads").value)
        try:
            self.get_logger().info(
                f"Loading Nemotron (sherpa-onnx) from {model_dir} "
                f"(threads={nthreads})...")
            self.nemo = NemotronStreamer(
                model_dir, num_threads=nthreads,
                block_ms=int(gp("nemotron_block_ms").value),
                rule1_sil=float(gp("nemotron_ep_rule1_sil").value),
                rule2_sil=float(gp("nemotron_ep_rule2_sil").value),
                max_utt=float(gp("nemotron_ep_max_utt").value),
                tail_pad_s=float(gp("nemotron_tail_pad_s").value),
                frame_is_utterance=bool(gp("nemotron_frame_is_utterance").value),
                debug_log=(self.get_logger().info if self.debug else None))
        except Exception as e:
            self.get_logger().error(f"Failed to init Nemotron: {e}")
            raise
        self.get_logger().info(
            "Nemotron ready — native streaming, partials on, "
            f"final per {'frame' if self.nemo.frame_is_utterance else 'endpoint'} "
            f"(rule2_sil={gp('nemotron_ep_rule2_sil').value}s).")

    def _run_nemotron(self, clean_audio):
        def on_partial(text):
            sys.stdout.write(f"\rPartial: {text[-70:]:<72}"); sys.stdout.flush()
            self.partial_pub.publish(String(data=text))

        def on_final(text):
            sys.stdout.write("\r" + " " * 80 + "\r"); sys.stdout.flush()
            self.get_logger().info(f"Transcript: \"{text}\"")
            self.pub.publish(String(data=text))

        try:
            self.nemo.push(clean_audio, on_partial, on_final)
        except Exception as e:
            self.get_logger().error(
                f"Nemotron decode failed: {e}", throttle_duration_sec=2.0)
            return
        if self.debug:
            st = self.nemo.stats
            self.get_logger().info(
                f"Nemotron RTF so far: {self.nemo.rtf():.3f} "
                f"({st['audio_s']:.1f}s audio, {st['finals']} finals)")

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
        if self.engine == "nemotron":
            self._run_nemotron(clean_audio)
        elif self.engine == "whisper":
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
        if self.engine == "nemotron":
            # Flush whatever the continuous stream still holds (only non-empty
            # when frame_is_utterance is False).
            try:
                text = self.nemo.finalize()
                if text:
                    self.get_logger().info(
                        f"Final transcript on shutdown: \"{text}\"")
                    self.pub.publish(String(data=text))
            except Exception:
                pass
            return
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
