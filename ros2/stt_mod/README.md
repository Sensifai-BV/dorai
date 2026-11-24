# stt_mod — speech-to-text (fine-tuned Whisper, or NVIDIA Nemotron) for ROS 2

Stage 2 of the dorai pipeline. Two engines, both int8 ONNX through
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) on the CPU, chosen with `engine`:

| `engine` | Model | How it decodes |
| :--- | :--- | :--- |
| `whisper` (default) | Whisper large-v3-turbo with the dorai command fine-tune (LoRA merged) | The clean stream is cut into utterances by a VAD gate; each utterance is decoded once, when the talker stops. No partial hypotheses. |
| `nemotron` | NVIDIA [`nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) (cache-aware FastConformer + RNN-T, 600 M params) with the dorai workshop adapter **mc-ref** baked into the encoder | Streaming, partial hypotheses, its own endpointing. Ships inside the package (`models`). |

```
/dorai_clean_audio ──► stt_mod ──► /dorai_partial_transcript   running hypothesis (Nemotron only)
                               ├─► /dorai_transcript           final per utterance (punctuated)
                               └─► /dorai_stt/status           JSON: model, rtf, backlog_s, finals, lag_median_s
```

```bash
ros2 launch stt_mod stt.launch.py                       # defaults: Whisper behind the VAD gate, 2 threads
ros2 run stt_mod stt --ros-args -p engine:=nemotron     # the streaming engine (1120 ms, mc-ref)
ros2 run stt_mod stt --ros-args -p engine:=nemotron -p latency_ms:=560 -p num_threads:=4      # Mac / x86
ros2 run stt_mod stt --ros-args -p engine:=nemotron -p adapter:=none        # stock NVIDIA model
```

If the Whisper export is not installed, the node logs an error and runs
Nemotron instead (`engine_fallback:=false` makes it fail instead).

### The VAD gate (Whisper)

Whisper is handed only what the gate passes. That is what keeps noise from
becoming text: given 20 s of workshop noise with nobody speaking, the
fine-tuned model wrote a complete command, while behind the gate nothing is
decoded at all. `vad`:

* `neural` (default): the neural VAD model `vad_gated.onnx`, run by
  sherpa-onnx. The file ships inside the package (`stt_mod`); it is
  the same model as doa_mod's `vad.onnx`. Without it the node warns and uses
  `energy`.
* `energy`: level gate with an adaptive noise floor, hysteresis, 0.3 s
  pre-roll and hangover. Needs no file, but also passes loud non-speech.
* `none`: every incoming frame is one utterance (only for a front end that
  already publishes one utterance per frame).

An utterance ends after `vad_min_silence_s` (0.6 s) of silence, so a transcript
appears that long after the last word plus the decoding time. A transcript
that cannot belong to the length of its utterance (a repetition loop) is cut
back to the last whole sentence (`max_words_per_s`).

## Models

**Download.** All dorai models are in one Hugging Face repo,
[`sensifai/dorai`](https://huggingface.co/sensifai/dorai/tree/main):

| Path in the repo | Size | What |
| :--- | ---: | :--- |
| `stt-model/whisper-finetuned/sherpa-onnx-whisper-large-v3-turbo-dorai-artaban/` | 1.0 GB | Whisper large-v3-turbo with the dorai command fine-tune: `turbo-encoder.int8.onnx`, `turbo-decoder.int8.onnx`, `turbo-tokens.txt` |
| `stt-model/whisper-finetuned/adapter/step-3000/` | 8 MB | the LoRA adapter itself (PEFT), for further training |
| `stt-model/nemo-finetuned/` | 0.67 GB | Nemotron 0.6B, 1120 ms profile, `mc-ref` adapter baked in (NVIDIA Open Model License): `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`, `dorai_adapter.json`, `test_wavs/` |
| `intent-model/intent-qwen35-4b-Q4_0.gguf` | 2.6 GB | intent_mod's model (it downloads this itself on first start) |

```bash
bash ros2/stt_mod/scripts/download_models.sh                         # Whisper + Nemotron -> ~/dorai/models, Whisper checksums verified
MODELS=whisper bash ros2/stt_mod/scripts/download_models.sh          # only the default engine's model
MODELS="whisper nemotron intent" bash ros2/stt_mod/scripts/download_models.sh     # everything, for an offline robot
DEST=/data/models bash ros2/stt_mod/scripts/download_models.sh       # elsewhere; then run the node with -p models_dir:=/data/models
hf download sensifai/dorai --local-dir ~/dorai/models           # the same thing by hand: the whole repo
```

Nothing is bundled in the package: without a download the node has no model
and says where to get one. The files keep the repo's layout under the models
root (`~/dorai/models/stt-model/...`), and the node finds them there: every
root is also searched under `stt-model/` and `stt-model/whisper-finetuned/`,
and `nemo-finetuned/` is recognised by its `dorai_adapter.json`. The neural
VAD of the utterance gate (`vad_gated.onnx`) is not on the Hub; it ships in
the package source.

A model directory copied into `models` is installed with the package
(`setup.py` -> `share/stt_mod/models/`; a symlink with `colcon build
--symlink-install`) and is found too. The encoders are far above GitHub's
100 MB file limit: keep them out of git.

Only the 1120 ms Nemotron profile is published. Other profiles and adapters are looked up, in order, in `models_dir`,
`$DORAI_MODELS_DIR`, `models` (or `share/stt_mod/models`),
`~/dorai/models` and `/root/.cache/dorai/models`. The first root that holds
the requested model wins. To add one, download the stock export and bake the
adapter on a dev machine (needs `pip install onnx`; no torch/NeMo/GPU):

```bash
cd stt_mod/models          # or ~/dorai/models
L=560                      # 80 | 160 | 560 | 1120
N=sherpa-onnx-nemotron-speech-streaming-en-0.6b-${L}ms-int8-2026-04-25
curl -L -o $N.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$N.tar.bz2
tar xjf $N.tar.bz2 && rm $N.tar.bz2                                     # needs bzip2
python3 -m stt_mod.bake_adapter --export $N --adapter <stt repo>/models/mc-ref    # -> $N-dorai-mc-ref/
ros2 run stt_mod transcribe --list                                       # what is installed
```

**How the bake works.** The adapter is NeMo's `LinearAdapter`, which adds
`y + up(silu(down(LayerNorm(y))))` (dim 32, 1.62 M params) after each of the
24 Conformer blocks' `norm_out`. The base model was frozen during training. The
tool therefore splices these 24 fp32 residual MLPs into the stock ONNX encoder
after `/layers.<i>/norm_out/LayerNormalization`. It refuses to bake unless the
export's LayerNorm weights equal the adapter's base checkpoint. For the
2026-04-25 exports they are bit-identical (max |Δ| = 0). A unit test checks the
spliced graph against a numpy reference. On the real-microphone command takes, the int8
sherpa-onnx models reproduce the report's fp32 PyTorch results. At 560 ms,
command-word WER for mc-ref is 36.1 % here and 36.1 % in the report, and the
gain in robot-name recognition over base is the same.

## Defaults and why (measured 2026-10-20)

Data: the real-microphone command test from the stt repo report. One speaker
read 30 short robot commands: 60 quiet takes, 30 takes with welding noise
played from speakers, and the held-out welding/knock/babble noise mixed into
the quiet takes at 0/5/10 dB. That gives 630 decodes per model at 1120 ms.

| 1120 ms, real voice | base | **mc-ref** | mc-das |
| :--- | ---: | ---: | ---: |
| utterance → right command (nearest-of-30), all 630 | 65.1 % | **65.2 %** | 63.5 % |
| Δ vs base, 95 % CI (paired bootstrap over recordings) | — | +0.2 [−4.3, +4.5] | −1.6 [−5.7, +2.8] |
| robot name recognised, quiet | 36.7 % | **61.7 %** | 55.0 % |
| empty transcript, all | 13.0 % | 14.1 % | 16.5 % |
| WER of the words after the name, quiet | **24.3 %** | 35.4 % | 37.5 % |

* **adapter = mc-ref.** On the held-out synthetic workshop test (report) it cuts
  WER by 10.9 % [9.2, 12.6] vs base. On real voice it ties with base on
  getting the command right and recognises the out-of-vocabulary robot name
  about twice as often. mc-das was trained on delay-and-sum input, which
  voice_mod does not produce, and it is never better than mc-ref. The cost is
  the last row: the adapter writes audiobook inflections ("turned right",
  "picked up"). If the downstream consumer needs verbatim words rather than
  command matching, use `adapter:=none`.

### Latency profiles (mc-ref, 60 quiet takes, dev Mac with 12 cores, 4 threads)

| `latency_ms` | rule2 (auto) | end of speech → final, median / p90 | RTF Mac | Pi 4 (2 thr) | right command |
| ---: | ---: | :--- | ---: | :--- | ---: |
| 80 | 0.5 s | 1.32 / 1.65 s | 0.34 | not measured (slower than 560) | 92 % |
| 160 | 0.8 s | 1.71 / 2.08 s | 0.17 | not measured (slower than 560) | 97 % |
| 560 | 0.8 s | 1.89 / 2.28 s | 0.07 | RTF ≈ 3.6–4.3 | 90 % |
| **1120** | 1.32 s (floor) | 2.59 / 3.05 s | 0.04 | **RTF ≈ 1.7** | 95 % |

The lag column is audio time: from the last speech sample to the final,
excluding compute and voice_mod framing. It exceeds rule2 because the model
emits its trailing punctuation late. Shorter rule2 values split utterances at
short pauses: at 80 ms, rule2 = 0.3 s split 12 % of them and dropped
right-command accuracy to 58 %; at 160 ms, 0.4 s gave 82 %. Hence the per-profile auto values (`ep_rule2_sil:=0`). The Pi
numbers come from earlier sessions with the base model; the adapter adds
~0.1 % FLOPs. **No profile is real time on a Pi 4.** 1120 ms is the closest.
For live use, run stt_mod on a faster host (any profile), or on a Pi 5.

The node never drops audio: decoding runs on a worker thread behind an
unbounded queue. When it falls behind, `/dorai_stt/status.backlog_s` grows and
a warning is logged above `backlog_warn_s`.

## Real end-to-end test on the Raspberry Pi 4 (2026-10-20)

Setup: Pi 4 Model B, 8 GB, Ubuntu 24.04.5, ROS 2 Jazzy, provisioned with
`../../tests`. The real `stt_mod` node ran with its defaults (1120 ms,
mc-ref, 2 threads). `test/e2e_realmic.py` played the 90 real-mic takes into
`/dorai_clean_audio` in real time as 1 s voice_mod-format frames, with 3 s of
silence after each take, and scored `/dorai_transcript`. The Mac column is the
same model decoded offline on the Mac.

| condition | takes | WER Pi | WER Mac | WER without the name, Pi / Mac | empty | end of speech → final, median / p90 |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| quiet | 60 | 59.2 % | 60.7 % | 34.7 / 35.4 % | 0 % | 7.59 / 8.26 s |
| welding from speakers | 30 | 108.7 % | 96.1 % | 81.9 / 79.2 % | 16.7 % | 7.68 / 8.81 s |
| all | 90 | 75.7 % | 72.5 % | 50.5 / 50.0 % | 5.6 % | 7.61 / 8.34 s |

* Accuracy on the Pi equals the offline model. The quiet and name-excluded
  WERs agree within 1–3 points, the empty-transcript rate is identical, and no
  take was split into two finals. Plain WER is high mainly because the robot
  name is out of vocabulary and costs 2–3 errors on a 3-word command. On
  quiet takes, WER without the name is 34.7 %.
* Latency is the Pi's limit. The node ran at RTF 1.79 with 2 threads (offline
  `transcribe`: 1.55). The ~2.6 s audio-domain endpoint lag therefore becomes
  ~7.6 s of wall clock. Real-time use needs a faster host than a Pi 4.
* Thermals: the SoC reached 84 °C after ~15 min of continuous decoding, while
  holding 1.75–1.8 GHz under load. That is just below the Pi 4's 85 °C
  throttle point, so fit a heatsink or fan for sustained use.

## Parameters

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `engine` | `whisper` | `whisper` or `nemotron` |
| `engine_fallback` | `true` | Whisper export missing: run Nemotron instead of failing |
| `whisper_model` | `""` | Whisper export directory name or its tail; empty = newest dorai fine-tune found |
| `language` | `en` | Whisper language |
| `vad` | `neural` | utterance gate in front of Whisper: `neural`, `energy`, `none` |
| `vad_model` | `""` | neural VAD file; empty = `vad_gated.onnx` shipped in the package |
| `vad_threshold` | `0.5` | speech probability threshold of the neural VAD |
| `vad_min_silence_s` | `0.6` | silence that ends an utterance |
| `vad_min_speech_s` | `0.25` | shorter utterances are dropped |
| `vad_max_utt_s` | `15.0` | an utterance is cut here |
| `vad_pre_roll_s` | `0.3` | audio kept before the detected start of an utterance (the detector fires after the first sounds) |
| `max_words_per_s` | `5.0` | transcripts longer than speech allows are cut |
| `latency_ms` | `1120` | Nemotron encoder chunk / profile: 80, 160, 560, 1120 |
| `adapter` | `mc-ref` | Nemotron: `mc-ref`, `mc-das`, `none` |
| `models_dir` / `model_dir` | `""` | extra search root / explicit model dir (overrides the model choice of the engine) |
| `num_threads` | `2` | onnxruntime threads (2 is fastest on a Pi 4) |
| `block_ms` | `200` | feed granularity |
| `ep_rule2_sil` | `0` (auto) | trailing silence after text that ends an utterance; floored at chunk + 0.2 s |
| `ep_rule1_sil` | `2.0` | silence with nothing decoded → reset; floored at rule2 + chunk + 0.3 s |
| `ep_max_utt` | `20.0` | force a final on long speech |
| `frame_is_utterance` | `false` | `true` only for a front-end that publishes one utterance per frame |
| `backlog_warn_s`, `status_period_s` | `5.0`, `5.0` | runtime monitoring |
| `metrics`, `metrics_file` | `false`, `""` | per-final metrics on `/dorai_stt/metrics` (+ JSONL file); see Metrics |
| topics | see diagram | `input_topic`, `output_topic`, `partial_topic`, `status_topic` |

`lag_s` in the log is the time from the capture of the endpoint audio to
publication, including queueing and voice_mod framing.

## Metrics: WER, performance, SNR

**Online (on the robot), `metrics:=true`.** With this option, every final
also publishes a JSON record on `/dorai_stt/metrics`. Set
`metrics_file:=/path.jsonl` to also append each record to a file. It is off by
default and costs almost nothing when on.

```bash
ros2 run stt_mod stt --ros-args -p metrics:=true -p metrics_file:=/tmp/stt_metrics.jsonl
ros2 topic echo /dorai_stt/metrics
```

| field | meaning |
| :--- | :--- |
| `text`, `n_words`, `stream_pos_s`, `window_s` | the final; its position in the input stream; audio since the previous final |
| `lag_s` | capture of the endpoint audio → publication (queueing + decoding + framing) |
| `rtf_window`, `rtf_total`, `backlog_s` | decode time / audio time since the previous final and overall; queued audio |
| `snr_est_db`, `speech_dbfs`, `noise_dbfs`, `speech_frac` | **blind** input SNR estimate for this utterance (20 ms frame powers: 10th-percentile noise floor vs frames > floor + 6 dB) |
| `cpu_temp_c`, `cpu_freq_mhz`, `load1`, `proc_cpu_pct` | Pi thermal state, clock, load, and this node's CPU share |

The blind SNR needs no reference. It is reliable at ≥ 5 dB true SNR. With
impulsive noise it reads too high: welding at −10 to −20 dB reads +8 to +12 dB.
Use it to compare conditions (for example, voice_mod input vs output), not as
an absolute value.

**Offline (with reference transcripts), `ros2 run stt_mod evaluate`.** This
runs the same streaming decoder on a manifest. It can sweep latency profiles
and adapters, and mix in noise at given SNRs. It writes `utterances.jsonl`,
`summary.json` and `summary.md`, each giving, per condition × config: WER,
CER, WER without an out-of-vocabulary name (`--name`), empty %,
end-of-speech → final lag, RTF, true and estimated SNR, and max CPU
temperature.

```bash
ros2 run stt_mod evaluate --manifest data/manifest.jsonl --latency-ms 80 160 560 1120 --adapter mc-ref none --out eval
ros2 run stt_mod evaluate --manifest clean.jsonl --noise welding/*.wav --snr 0 5 10 --keep-clean --name artaban
```

Manifest format: one JSON per line with `audio_filepath` (relative to the
manifest) and `text`, plus optional `condition` and `snr_db`.

## Tools and tests

```bash
ros2 run stt_mod transcribe stt_mod/models/*/test_wavs/*.wav         # finals + RTF (no ROS needed)
ros2 run stt_mod transcribe --latency-ms 560 --adapter none --frame-s 5 clean.wav
ros2 run stt_mod record_clean --ros-args -p duration:=20.0             # save /dorai_clean_audio to wav
python3 -m pytest ros2/stt_mod/test -v       # 11 tests; model tests use the bundled default
```

## Known limits / next steps

* The robot name is out of vocabulary. Add speech containing it to adapter
  training (TTS or several real speakers, mixed with the same noises), plus
  10–30 % clean command-style speech, to stop the inflection drift. Re-bake
  afterwards; the node needs no change.
* Babble at 0–5 dB defeats every model.
* sherpa-onnx has only greedy search for this transducer, so it offers no
  hotword boosting.
