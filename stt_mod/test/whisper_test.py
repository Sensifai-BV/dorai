#!/usr/bin/env python3
"""
Offline A/B test: run whisper on one or more WAV files and print the transcript
+ timing, with the SAME decode settings as the live stt_mod node (temperature=0,
no fallback, anti-repetition), so it is representative of the pipeline.

Goal: compare a RAW mic recording vs a dorai_beamformer-CLEAN recording of the SAME speech,
and tiny.en vs base.en, to decide whether the model or dorai_beamformer is the limit.

Usage (on the Pi):
  # record the same phrases two ways first:
  #   ros2 run voice_mod record -p source:=dev   -p duration:=12 -p prefix:=wtw  # raw mics, 48k
  #   ros2 run voice_mod record -p source:=clean -p duration:=12 -p prefix:=wtw  # dorai_beamformer clean (pipeline running)
  python3 lab/whisper_test.py --model tiny.en wtw_dev1.wav wtw_clean.wav
  python3 lab/whisper_test.py --model base.en wtw_dev1.wav wtw_clean.wav

faster-whisper resamples to 16k internally, so 48 kHz raw files are fine.
"""
import time
import argparse
import numpy as np
import wave

from faster_whisper import WhisperModel


def read_wav(path):
    w = wave.open(path, "rb")
    sr = w.getframerate()
    ch = w.getnchannels()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch)[:, 0]
    if sr != 16000:
        idx = (np.arange(int(len(a) * 16000 / sr)) * sr / 16000).astype(np.int64)
        idx = idx[idx < len(a)]
        a = a[idx]
    return a.astype(np.float32), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", default="tiny.en")
    ap.add_argument("--vad", action="store_true", help="enable faster-whisper vad_filter")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--prompt", default=None,
                    help="initial_prompt: bias decoding toward this vocabulary, "
                         "e.g. \"What's the weather? Hello. Turn left. Stop.\"")
    args = ap.parse_args()

    print(f"Loading {args.model} (int8, threads={args.threads})...")
    m = WhisperModel(args.model, device="cpu", compute_type="int8",
                     cpu_threads=args.threads)

    for f in args.files:
        audio, sr = read_wav(f)
        dur = len(audio) / 16000
        rms = float(np.sqrt(np.mean(audio ** 2)))
        t0 = time.monotonic()
        segs, info = m.transcribe(
            audio, language="en", beam_size=args.beam,
            temperature=0.0, condition_on_previous_text=False,
            no_repeat_ngram_size=3, repetition_penalty=1.15,
            initial_prompt=args.prompt,
            vad_filter=args.vad)
        text = "".join(s.text for s in segs).strip()
        dt = time.monotonic() - t0
        print(f"\n=== {f}  ({dur:.1f}s, src_sr={sr}, rms={rms:.4f})")
        print(f"    transcribe={dt:.1f}s  (RTF={dt/max(dur,1e-9):.2f})  "
              f"model={args.model} beam={args.beam} vad={args.vad}")
        print(f"    TEXT: \"{text}\"")


if __name__ == "__main__":
    main()
