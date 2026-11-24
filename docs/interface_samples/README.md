# Interface samples — real recordings of every DORAI topic

Recorded on 2026-11-24 with the full graph running natively on ROS 2 Lyrical
(OrbStack `ros2` VM, Ubuntu 26.04, arm64, CPU only), fed by `dorai_simulator`:

```
virtual_mic (scene) ─► voice_mod (3-mic dorai_beamformer.ort, denoise_level 0.75, 1 s frames)
                    ─► doa_mod (ReSpeaker 4-mic, pooled SRP-PHAT)
voice_mod ─► stt_mod (fine-tuned Whisper large-v3-turbo, int8) ─► intent_mod (Qwen3.5-4B GGUF)
```

Reproduce (in the VM, after `colcon build` of `ros2/*` into `$DORAI_WS`):

```bash
bash tests/e2e/run_scene.sh ~/dorai_sim_scenes/clean_speech /tmp/out
python3 tests/e2e/score.py /tmp/out
```

Files: `<scene>.topics.jsonl` holds every message received, one JSON object per line
(`t` = seconds since the recorder started; audio frames are summarised by their
header, length and level instead of the samples). `<scene>.score.txt` is the scored result.

## Results

| Scene | Room | WER | Plans correct | DoA mean error | Capture → transcript | Intent LLM |
|---|---|---|---|---|---|---|
| clean_speech | workshop, RT60 0.59 s | 0 % (23 words) | 6 / 6 | 3.3° (6/6 within 20°) | 2.5 – 3.3 s | 0.43 – 0.98 s |
| factory_welding_0db | factory, RT60 1.2 s, welding at 0 dB SNR | 52 % | 3 / 6 (2 ignored, 1 clarify) | 73° (3/6 within 20°) | 2.5 – 3.1 s | 0.47 – 1.01 s |

voice_mod per 1 s frame: inference 0.29 – 0.32 s (real-time factor ≈ 0.3), publish < 2 ms, no queueing.

Known issues found by these runs:

* In the factory scene doa_mod locks onto the welding source (~46°) when the talker is
  quieter than the machine; the speech gate does not reject the welder's broadband noise.
* `/dorai_plan.bearing_deg` is the bearing cached when the plan is published, about 3.5 s after
  the utterance, so it can belong to the next talker (clean scene: "go back to base" said at
  191–199°, plan carries 148°). It should be the bearing measured during the utterance.
* "Artaban stop" misheard as "Fargan stop." was ignored: the stop gate only fires when the
  whole utterance is a stop word, so a mangled wake word before "stop" blocks it.
* `/dorai_stt/status` contains `NaN` (`rtf` before the first final), which strict JSON parsers reject.

## One real message per topic (clean_speech)

```jsonc
// /dorai_sim/truth  (ground truth from the simulator)
{"event": "command", "phase": "start", "index": 0, "text": "Artaban go back to base",
 "t0": 3.0, "t1": 4.95, "bearing_deg": 194.9, "distance_m": 1.9, "elevation_deg": 30.1}

// /dorai_clean_audio  (Float32MultiArray; header + 14976 samples)
{"header": {"capture_s": 17.995, "rate": 16000.0, "channels": 1, "seq": 0},
 "samples": 14976, "duration_s": 0.936}

// /dorai_doa  (Float32)
199.0

// /dorai_doa/state  [elapsed_s, azimuth_deg, sharpness, consistent_frac, speech_flag, seq]
[2.976, 199.0, 0.584, 1.0, 1.0, 0.0]

// /dorai_doa/human
"bearing 199.0 deg  |  5 o'clock from the array's 0 deg mark  |  confidence 0.58  |  frame gate 100%  |  t    3.0s  seq 0"

// /dorai_transcript
"Artaban go back to base."

// /dorai_stt/metrics
{"model": "whisper/large-v3-turbo-dorai-artaban", "text": "Artaban go back to base.", "n_words": 5,
 "stream_pos_s": 4.97, "lag_s": 2.847, "rtf_window": 0.012, "snr_est_db": 21.6,
 "speech_dbfs": -28.2, "noise_dbfs": -49.8, "speech_frac": 0.323, "proc_cpu_pct": 15.1}

// /dorai_stt/status
{"model": "whisper/large-v3-turbo-dorai-artaban", "rtf": NaN, "backlog_s": 0.0, "busy": false,
 "audio_s": 0.0, "finals": 0, "lag_median_s": null}

// /dorai_plan
{"text": "Artaban go back to base.", "plan": {"steps": [{"task": "goto", "target": "base"}]},
 "bearing_deg": 148.0, "engine": "llm", "latency_s": 0.9777, "stamp": 1795540920.06,
 "confidence": 1.0, "min_token_prob": 0.999, "wake_edits": 0}

// /dorai_intent/decision
{"text": "Artaban go back to base.", "decision": "plan", "addressed": true, "wake_heard": "artaban",
 "wake_edits": 0, "wake_confidence": 1.0, "command": "go back to base", "engine": "llm",
 "confidence": 1.0, "min_token_prob": 0.999, "latency_s": 0.9777, "tasks": ["goto"]}

// /dorai_intent/status
{"engine": "llm", "model": "intent-qwen35-4b-Q4_0.gguf", "n_plans": 0, "n_clarify": 0, "n_none": 0,
 "n_stop": 0, "n_invalid": 0, "n_ignored": 0, "n_error": 0, "mean_latency_s": null,
 "queue_depth": 0, "require_wake": true, "bearing_deg": null}

// /dorai_say  (factory scene, after an unclear command)
"Sorry, I didn't understand. Could you say that again?"
```
