# stt_mod — streaming speech-to-text (NVIDIA Nemotron) for ROS 2

Stage 2 of the dorai pipeline. It runs NVIDIA
[`nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
(cache-aware FastConformer + RNN-T, NeMo, 600 M params) as int8 ONNX through
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) on the CPU. The dorai
fine-tuned workshop adapter **mc-ref** is baked into the encoder, and that
default model ships inside the package (`stt_mod/models/`).

```
/dorai_clean_audio ──► stt_mod ──► /dorai_partial_transcript   running hypothesis (every encoder chunk)
                               ├─► /dorai_transcript           final per endpoint (punctuated)
                               └─► /dorai_stt/status           JSON: rtf, backlog_s, finals, lag_median_s
```

```bash
ros2 launch stt_mod stt.launch.py                       # defaults: 1120 ms, mc-ref, 2 threads
ros2 run stt_mod stt --ros-args -p latency_ms:=560 -p num_threads:=4      # Mac / x86
ros2 run stt_mod stt --ros-args -p adapter:=none        # stock NVIDIA model
```

## Models

**Download (Artaban team).** The fine-tuned models are on the Hugging Face Hub at
[`sensifai/dorai-nemotron-speech-0.6b-mc-ref-onnx`](https://huggingface.co/sensifai/dorai-nemotron-speech-0.6b-mc-ref-onnx)
(1120/560/160/80 ms, mc-ref baked in; NVIDIA Open Model License):

```bash
bash stt_mod/scripts/download_models.sh                         # 1120 ms -> ~/dorai/models, checksums verified
LATENCIES="1120 560" DEST=stt_mod/models bash stt_mod/scripts/download_models.sh
```

The encoder file (653 MB) is above GitHub's 100 MB limit, so keep
`stt_mod/models/**/*.onnx` out of the git push. Add it to `.gitignore`, or use
Git LFS.


The default model is bundled in the package, so `ros2 run stt_mod stt` works
with no download:

```
stt_mod/models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-1120ms-int8-2026-04-25-dorai-mc-ref/
    encoder.int8.onnx (653 MB, adapter baked in)  decoder.int8.onnx  joiner.int8.onnx
    tokens.txt  dorai_adapter.json (adapter provenance + encoder sha256)  test_wavs/
```

`setup.py` installs it to `share/stt_mod/models/`; with `colcon build
--symlink-install` that is a symlink, not a copy. The Docker image gets it
through `COPY stt_mod`. The 653 MB encoder exceeds GitHub's 100 MB file
limit, so track `stt_mod/models/**/*.onnx` with Git LFS if you commit it.

Other profiles and adapters are looked up, in order, in `models_dir`,
`$DORAI_MODELS_DIR`, `stt_mod/models` (or `share/stt_mod/models`),
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
`test/pi_provision.sh`. The real `stt_mod` node ran with its defaults (1120 ms,
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
| `latency_ms` | `1120` | encoder chunk / profile: 80, 160, 560, 1120 |
| `adapter` | `mc-ref` | `mc-ref`, `mc-das`, `none` |
| `models_dir` / `model_dir` | `""` | extra search root / explicit model dir (overrides the two above) |
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
python3 -m pytest stt_mod/test -v       # 11 tests; model tests use the bundled default
```

## Known limits / next steps

* The robot name is out of vocabulary. Add speech containing it to adapter
  training (TTS or several real speakers, mixed with the same noises), plus
  10–30 % clean command-style speech, to stop the inflection drift. Re-bake
  afterwards; the node needs no change.
* Babble at 0–5 dB defeats every model.
* sherpa-onnx has only greedy search for this transducer, so it offers no
  hotword boosting.
