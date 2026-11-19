"""NemotronStreamer: incremental decoding of a continuous 16 kHz stream.

nvidia/nemotron-speech-streaming-en-0.6b is a cache-aware FastConformer encoder
with an RNN-T decoder, run through sherpa-onnx's OnlineRecognizer. The stream
is fed in ``block_ms`` blocks; every change of the running hypothesis goes to
``on_partial(text)`` and each detected endpoint produces ``on_final(text, pos_s)``
where ``pos_s`` is the stream position (seconds of audio fed so far) at which
the final was emitted — the node turns it into an end-to-end lag.

Endpointing (sherpa-onnx rules): rule1 = trailing silence with nothing decoded,
rule2 = trailing silence after text (ends an utterance), rule3 = force-final
on long utterances. For cache-aware chunked models trailing silence is measured
in chunk-sized decode bursts, so rule2 is floored at chunk + 0.2 s (a smaller
rule2 split "I was a" | "Dreamer, ..." with the 1120 ms export), and rule1 at
rule2 + chunk + 0.3 s (otherwise the replayed tail re-fires rule1 on every
block: RTF 16.8 on silence on the Pi).

Endpoint rollover: when is_endpoint() turns true, up to one chunk of fed audio
is still undecoded; recognizer.reset() would drop it and lose the onset of the
next phrase ("on end [tucked away in] my corner"). Instead a fresh stream is
created and exactly the undecoded tail plus ``replay_margin_s`` (inside the
>= rule2 trailing silence) is replayed into it. A margin equal to rule2 (max
0.8 s) was loss- and duplicate-free in a 0.35/0.5/0.65/0.8 s sweep.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .models import chunk_ms_of, pick_onnx


class NemotronStreamer:
    def __init__(self, model_dir, num_threads=2, block_ms=200, rule1_sil=2.0,
                 rule2_sil=0.8, max_utt=20.0, tail_pad_s=0.8,
                 frame_is_utterance=False, fs=16000, provider="cpu", log=None):
        import sherpa_onnx  # imported here so tests can stub it

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Nemotron model dir not found: {model_dir!r}")
        tokens = os.path.join(model_dir, "tokens.txt")
        if not os.path.exists(tokens):
            raise FileNotFoundError(f"tokens.txt missing in {model_dir}")
        self.model_dir = model_dir
        self.fs = int(fs)
        self.block = max(1, int(self.fs * block_ms / 1000))
        self.tail_pad = int(self.fs * float(tail_pad_s))
        self.frame_is_utterance = bool(frame_is_utterance)
        self.log = log or (lambda s: None)

        self.chunk_s = chunk_ms_of(model_dir) / 1000.0
        self.rule2_sil = max(float(rule2_sil), self.chunk_s + 0.2)
        if self.rule2_sil != float(rule2_sil):
            self.log(f"rule2_sil {rule2_sil}s < chunk {self.chunk_s}s + 0.2 -> "
                     f"{self.rule2_sil:.2f}s")
        self.rule1_sil = max(float(rule1_sil), self.rule2_sil + self.chunk_s + 0.3)
        self.max_utt = float(max_utt)

        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=pick_onnx(model_dir, "encoder"),
            decoder=pick_onnx(model_dir, "decoder"),
            joiner=pick_onnx(model_dir, "joiner"),
            num_threads=int(num_threads),
            sample_rate=self.fs,
            # sherpa-onnx implements only greedy search for the Nemotron
            # transducer (no beam search / hotwords).
            decoding_method="greedy_search",
            model_type="nemotron",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=self.rule1_sil,
            rule2_min_trailing_silence=self.rule2_sil,
            rule3_min_utterance_length=self.max_utt,
            provider=provider,
        )
        self.replay_margin_s = min(0.8, self.rule2_sil)
        self._tail_n = int(self.fs * (self.chunk_s + self.replay_margin_s + 0.5))
        self._tail = np.zeros(0, dtype=np.float32)
        self._last_partial = ""
        self.pos = 0                       # samples fed in total (all streams)
        self.stats = {"audio_s": 0.0, "compute_s": 0.0, "finals": 0}
        self._new_stream()

    # ------------------------------------------------------------ internals
    @property
    def pos_s(self):
        return self.pos / self.fs

    def _new_stream(self, replay=None, on_partial=None):
        self.stream = self.rec.create_stream()
        self._fed = 0
        self._ndec = 0
        if replay is not None and replay.size:
            self.stream.accept_waveform(self.fs, replay)
            self._fed += len(replay)
            self._drain(on_partial)

    def _feed(self, blk):
        self.stream.accept_waveform(self.fs, blk)
        self._fed += len(blk)
        self.pos += len(blk)
        self._tail = np.concatenate([self._tail, blk])[-self._tail_n:]

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

    def _undecoded_tail(self):
        und = self._fed / self.fs - self._ndec * self.chunk_s
        und = min(max(und, 0.0), self.chunk_s + 0.3) + self.replay_margin_s
        n = int(und * self.fs)
        return self._tail[-n:].copy() if n > 0 else self._tail[:0]

    def _emit(self, text, on_final):
        self._last_partial = ""
        if text:
            self.stats["finals"] += 1
            if on_final:
                on_final(text, self.pos_s)

    # --------------------------------------------------------------- public
    def push(self, samples, on_partial=None, on_final=None):
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
                    self.log(f"endpoint -> final {text!r} (replay {len(tail) / self.fs:.2f}s)")
                    self._emit(text, on_final)
                    self._new_stream(tail, on_partial)
                elif tail.size and float(np.abs(tail).max()) > 0.02:
                    self._new_stream(tail, on_partial)   # possible phrase onset
                else:
                    # silence: nothing to lose. Counters keep running because
                    # reset() keeps the encoder's frame position.
                    self.rec.reset(self.stream)
        if self.frame_is_utterance:
            self.finalize(on_partial, on_final)
        self.stats["audio_s"] += len(x) / self.fs
        self.stats["compute_s"] += time.perf_counter() - t0

    def finalize(self, on_partial=None, on_final=None):
        """Flush the current stream as one final and start a fresh stream."""
        if self.tail_pad:
            self.stream.accept_waveform(self.fs, np.zeros(self.tail_pad, np.float32))
        self.stream.input_finished()
        text = self._drain(on_partial)
        self._new_stream()
        self._tail = np.zeros(0, dtype=np.float32)
        self._emit(text, on_final)
        return text

    def rtf(self):
        a = self.stats["audio_s"]
        return self.stats["compute_s"] / a if a > 0 else float("nan")


def transcribe(streamer, x):
    """Offline helper: one utterance -> text (feeds, then finalizes)."""
    finals = []
    streamer.push(x, on_final=lambda t, p: finals.append(t))
    t = streamer.finalize(on_final=lambda t, p: None)
    if t:
        finals.append(t)
    return " ".join(finals).strip()
