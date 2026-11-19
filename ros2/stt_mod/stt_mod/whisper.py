"""WhisperStreamer: VAD-gated utterance decoding of a continuous 16 kHz stream.

Whisper (large-v3-turbo with the dorai command fine-tune, sherpa-onnx export)
is not a streaming model: it transcribes one utterance at a time. The stream
from voice_mod is therefore cut into utterances by a voice-activity gate, and
each utterance is decoded once, when the talker stops:

    clean audio -> VAD gate -> one utterance -> Whisper -> on_final(text, pos_s)

The gate matters for more than segmentation. A recogniser that is handed
noise writes words: on 20 s of workshop noise with nobody speaking, the
fine-tuned model produced a full command. Audio the gate does not pass is
never decoded, and a transcript that is implausible for the length of its
utterance (a repetition loop) is cut or dropped.

Gates (``vad``):
  neural  neural VAD (``vad_gated.onnx``, shipped in this package; the same
          model doa_mod uses), run by sherpa-onnx. Default. Without the file
          the node falls back to ``energy`` and says so.
  energy  level gate with an adaptive noise floor, hysteresis, pre-roll and
          hangover. No model file. Passes loud non-speech (bangs, arcs).
  none    no gate: every pushed frame is decoded as one utterance (only for a
          front end that already publishes one utterance per frame).

Same interface as NemotronStreamer (push / finalize / rtf / stats / pos_s), so
the node drives either engine. There are no partial hypotheses: ``on_partial``
is never called.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .models import pick_whisper_files

FS = 16000


def tame(text, dur_s, max_words_per_s=5.0):
    """Cut a transcript that cannot belong to ``dur_s`` seconds of speech.

    Greedy Whisper decoding can fall into a loop ("a little bit of a little
    bit of ...") on noisy input. Two checks, both conservative: where a phrase
    of 1 to 4 words comes three times in a row, the text is cut after the last
    complete sentence before the repeats (or, with no sentence end, after the
    first occurrence of the phrase); and a transcript with more words than
    anyone speaks in that time (5 words per second, plus 3) is cut to that
    many words."""
    words = text.split()
    key = [w.lower().strip(".,!?") for w in words]
    cut = None
    for n in (4, 3, 2, 1):
        for i in range(0, len(words) - 3 * n + 1):
            if key[i:i + n] == key[i + n:i + 2 * n] == key[i + 2 * n:i + 3 * n]:
                cut = i + n if cut is None else min(cut, i + n)
                break
    if cut is not None:
        ends = [j + 1 for j in range(cut) if words[j][-1:] in ".!?"]
        words = words[:ends[-1] if ends else cut]
    cap = int(3 + max_words_per_s * max(dur_s, 0.0))
    if len(words) > cap:
        words = words[:cap]
    return " ".join(words).strip()


class EnergyGate:
    """Level gate: adaptive noise floor, hysteresis, pre-roll and hangover.

    Works on 20 ms frames. The floor follows the quietest recent level (it
    falls at once and rises by 1.5 dB per second), so steady noise of any
    level is "no speech". Speech starts when ``start_frames`` consecutive
    frames exceed the floor by ``start_db`` and ends after ``min_silence_s``
    below ``stop_db``. ``accept`` returns finished utterances as
    (start_sample, samples), with ``pre_roll_s`` of audio before the start."""

    def __init__(self, fs=FS, start_db=12.0, stop_db=6.0, min_silence_s=0.6, min_speech_s=0.25,
                 max_utt_s=15.0, pre_roll_s=0.3, abs_gate_dbfs=-60.0):
        self.fs = int(fs)
        self.n = self.fs // 50
        self.start, self.stop = 10 ** (start_db / 20), 10 ** (stop_db / 20)
        self.abs_gate = 10 ** (abs_gate_dbfs / 20)
        self.hang = max(1, int(round(min_silence_s * 50)))
        self.min_frames = max(1, int(round(min_speech_s * 50)))
        self.max_n = int(max_utt_s * self.fs)
        self.pre = int(pre_roll_s * self.fs)
        self.rise = 10 ** (1.5 / 50 / 20)
        self.reset()

    def reset(self):
        self.floor = None
        self.buf = np.zeros(0, np.float32)       # samples not yet framed
        self.hist = np.zeros(0, np.float32)      # pre-roll history
        self.utt = None                          # samples of the open utterance
        self.utt_start = 0
        self.pos = 0                             # samples consumed in frames
        self.above = self.below = self.voiced = 0

    def _close(self, out):
        if self.utt is not None and self.voiced >= self.min_frames:
            out.append((self.utt_start, self.utt))
        self.utt, self.above, self.below, self.voiced = None, 0, 0, 0

    def accept(self, x):
        out = []
        self.buf = np.concatenate([self.buf, np.asarray(x, np.float32)])
        while len(self.buf) >= self.n:
            f, self.buf = self.buf[:self.n], self.buf[self.n:]
            rms = float(np.sqrt(np.mean(f.astype(np.float64) ** 2)))
            self.floor = rms if self.floor is None else min(rms, self.floor * self.rise)
            ref = max(self.floor, self.abs_gate / self.start)
            if self.utt is None:
                self.above = self.above + 1 if rms > ref * self.start else 0
                if self.above >= 2:
                    lead = self.hist[-self.pre:] if self.pre else self.hist[:0]
                    self.utt = np.concatenate([lead, f])
                    self.utt_start = self.pos - len(lead)
                    self.voiced, self.below = self.above, 0
            else:
                self.utt = np.concatenate([self.utt, f])
                if rms > ref * self.stop:
                    self.below = 0
                    self.voiced += 1
                else:
                    self.below += 1
                if self.below >= self.hang or len(self.utt) >= self.max_n:
                    self._close(out)
            self.hist = np.concatenate([self.hist, f])[-max(self.pre, self.n):]
            self.pos += self.n
        return out

    def flush(self):
        out = []
        self._close(out)
        return out


class NeuralGate:
    """Neural VAD (vad_gated.onnx) through sherpa-onnx; same accept / flush / reset as EnergyGate.

    The detector reports an utterance from the frame in which it became sure,
    which is after the first sounds: without help, "Go back to base" reached
    the recogniser as "back to base". The gate therefore keeps the recent
    audio itself and hands out each utterance with ``pre_roll_s`` of audio
    before the detected start."""

    def __init__(self, model, fs=FS, threshold=0.5, min_silence_s=0.6, min_speech_s=0.25, max_utt_s=15.0,
                 num_threads=1, pre_roll_s=0.3):
        import sherpa_onnx
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model
        cfg.silero_vad.threshold = float(threshold)
        cfg.silero_vad.min_silence_duration = float(min_silence_s)
        cfg.silero_vad.min_speech_duration = float(min_speech_s)
        cfg.silero_vad.max_speech_duration = float(max_utt_s)
        cfg.sample_rate = int(fs)
        cfg.num_threads = int(num_threads)
        if not cfg.validate():
            raise ValueError(f"invalid VAD configuration for {model}")
        self.win = int(cfg.silero_vad.window_size)
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=max(30.0, 2 * max_utt_s))
        self.pre = int(float(pre_roll_s) * fs)
        self.keep = int((max_utt_s + min_silence_s + 2.0) * fs) + self.pre     # audio a pending utterance can reach back to
        self.reset_buffers()

    def reset_buffers(self):
        self.buf = np.zeros(0, np.float32)       # samples not yet given to the detector
        self.hist = np.zeros(0, np.float32)      # the audio the detector has seen ...
        self.hist_start = 0                      # ... and the stream position of its first sample
        self.last_end = 0                        # end of the previous utterance (pre-rolls do not overlap it)

    def reset(self):
        self.vad.reset()
        self.reset_buffers()

    def _take(self):
        out = []
        while not self.vad.empty():
            seg = self.vad.front
            start, samples = int(seg.start), np.asarray(seg.samples, np.float32)
            self.vad.pop()
            lead_from = max(start - self.pre, self.last_end, self.hist_start)
            if lead_from < start:
                lead = self.hist[lead_from - self.hist_start:start - self.hist_start]
                samples, start = np.concatenate([lead, samples]), lead_from
            self.last_end = start + len(samples)
            out.append((start, samples))
        return out

    def accept(self, x):
        self.buf = np.concatenate([self.buf, np.asarray(x, np.float32)])
        n = len(self.buf) // self.win * self.win
        for i in range(0, n, self.win):
            self.vad.accept_waveform(self.buf[i:i + self.win])
        self.hist = np.concatenate([self.hist, self.buf[:n]])
        if len(self.hist) > self.keep:
            self.hist_start += len(self.hist) - self.keep
            self.hist = self.hist[-self.keep:]
        self.buf = self.buf[n:]
        return self._take()

    def flush(self):
        self.vad.flush()
        return self._take()


VAD_FILE = "vad_gated.onnx"


def find_vad_model(explicit="", model_dir="", roots=()):
    """Path of the neural VAD file, or "" if none is found: the explicit path,
    then ``vad_gated.onnx`` in this package, the model directory and the models
    roots, then doa_mod's copy of the same model."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = ([explicit] if explicit else []) + [os.path.join(here, VAD_FILE)]
    for d in (model_dir, *roots):
        if d:
            cands += [os.path.join(os.path.expanduser(d), n) for n in (VAD_FILE, "vad.onnx")]
    cands.append(os.path.normpath(os.path.join(here, "..", "..", "doa_mod", "vad.onnx")))   # source tree
    try:
        from ament_index_python.packages import get_package_prefix
        prefix = get_package_prefix("doa_mod")
        for root, _, files in os.walk(prefix):
            if "vad.onnx" in files:
                cands.append(os.path.join(root, "vad.onnx"))
                break
    except Exception:   # not a ROS environment, or doa_mod not installed
        pass
    for c in cands:
        if c and os.path.isfile(os.path.expanduser(c)):
            return os.path.expanduser(c)
    return ""


class WhisperStreamer:
    def __init__(self, model_dir, num_threads=2, vad="neural", vad_model="", vad_threshold=0.5,
                 min_silence_s=0.6, min_speech_s=0.25, max_utt_s=15.0, pre_roll_s=0.3, language="en",
                 max_words_per_s=5.0, frame_is_utterance=False, fs=FS, provider="cpu", log=None,
                 recognizer=None, gate=None):
        self.model_dir = model_dir
        self.fs = int(fs)
        self.log = log or (lambda s: None)
        self.max_words_per_s = float(max_words_per_s)
        self.min_n = int(float(min_speech_s) * self.fs)
        self.rule2_sil = float(min_silence_s)        # what the node reports as the end-of-speech wait
        self.frame_is_utterance = bool(frame_is_utterance)
        if recognizer is None:
            import sherpa_onnx  # imported here so tests can stub it
            enc, dec, tok = pick_whisper_files(model_dir)
            recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=enc, decoder=dec, tokens=tok, language=language, task="transcribe",
                num_threads=int(num_threads), decoding_method="greedy_search", provider=provider)
        self.rec = recognizer

        vad = "none" if self.frame_is_utterance else (vad or "neural").strip().lower()
        vad = "neural" if vad == "silero" else vad            # the model's public name, still accepted
        self.vad_kind = vad
        if gate is not None:
            self.gate = gate
        elif vad == "none":
            self.gate = None
        elif vad == "neural":
            self.gate = NeuralGate(vad_model, self.fs, vad_threshold, min_silence_s, min_speech_s, max_utt_s,
                                   pre_roll_s=pre_roll_s)
        elif vad == "energy":
            self.gate = EnergyGate(self.fs, min_silence_s=min_silence_s, min_speech_s=min_speech_s,
                                   max_utt_s=max_utt_s, pre_roll_s=pre_roll_s)
        else:
            raise ValueError(f"vad must be neural, energy or none, not {vad!r}")
        self.pos = 0
        self.stats = {"audio_s": 0.0, "compute_s": 0.0, "finals": 0, "utterances": 0,
                      "speech_s": 0.0, "dropped": 0}

    @property
    def pos_s(self):
        return self.pos / self.fs

    def _decode(self, samples, end_pos, on_final):
        dur = len(samples) / self.fs
        self.stats["utterances"] += 1
        self.stats["speech_s"] += dur
        if len(samples) < self.min_n:
            self.stats["dropped"] += 1
            return ""
        st = self.rec.create_stream()
        st.accept_waveform(self.fs, np.ascontiguousarray(samples, dtype=np.float32))
        self.rec.decode_stream(st)
        raw = (st.result.text or "").strip()
        text = tame(raw, dur, self.max_words_per_s)
        if text != raw:
            self.log(f"transcript cut ({dur:.1f}s of speech): {raw[:120]!r} -> {text!r}")
        if not text:
            self.stats["dropped"] += 1
            return ""
        self.log(f"utterance {dur:.2f}s -> final {text!r}")
        self.stats["finals"] += 1
        if on_final:
            on_final(text, end_pos / self.fs)
        return text

    def push(self, samples, on_partial=None, on_final=None):
        x = np.asarray(samples, dtype=np.float32)
        if x.size == 0:
            return
        t0 = time.perf_counter()
        base = self.pos
        self.pos += len(x)
        if self.gate is None:
            self._decode(x, self.pos, on_final)
        else:
            for start, seg in self.gate.accept(x):
                self._decode(seg, min(self.pos, start + len(seg)) if start >= 0 else base, on_final)
        self.stats["audio_s"] += len(x) / self.fs
        self.stats["compute_s"] += time.perf_counter() - t0

    def finalize(self, on_partial=None, on_final=None):
        """Decode what the gate still holds (end of stream)."""
        text = ""
        if self.gate is not None:
            for _, seg in self.gate.flush():
                text = self._decode(seg, self.pos, on_final) or text
        return text

    def rtf(self):
        a = self.stats["audio_s"]
        return self.stats["compute_s"] / a if a > 0 else float("nan")
