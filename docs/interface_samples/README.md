# Interface samples — real recordings of every DORAI topic

Recorded on 2026-11-24 with the full graph running natively on ROS 2 Lyrical
(OrbStack `ros2` VM, Ubuntu 26.04, arm64, CPU only), fed by `dorai_simulator`, doa_mod `method:=auto`:

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
| clean_speech | workshop, RT60 0.59 s | 0 % (23 words) | 6 / 6, each with its own talker's bearing | 3.8° (6/6 within 20°) | 2.6 – 3.3 s | 0.78 – 1.30 s |
| factory_welding_0db | factory, RT60 1.2 s, welding at 0 dB SNR | 52 % | 3 / 6 + the stop (1 ignored, 1 clarify) | 21.6° (5/6 within 20°) | 2.9 – 5.7 s | 1.10 – 4.05 s |

voice_mod per 1 s frame: inference 0.31 – 0.41 s (real-time factor ≈ 0.3 – 0.4), publish < 10 ms.
In the factory run the intent LLM took up to 4 s (CPU shared by all five nodes on the VM).

Fixed after the first runs:

* **intent_mod** — `/dorai_plan.bearing_deg` used to be the latest bearing when the plan was
  published, about 3.5 s after the utterance, which could be the next talker's ("go back to base"
  said at 191–199° got 148°). It is now the last bearing of the most recent finished speech burst,
  i.e. its own utterance (194°).
* **intent_mod** — "Artaban stop" misheard as "Fargan stop." was ignored. The name in front of a
  stop is now matched with up to 4 edits (`stop_wake_max_edits`), and `/dorai_stop` fires.
* **doa_mod** — with a welder in the room the bearing locked onto the welder (~46°): SRP-PHAT gives
  every frequency bin one vote and a broadband machine owns most of them. The default is now
  `method:=auto`: SRP-PHAT when the utterance is ≥ 20 dB above the noise between utterances, else
  GEVD-MUSIC (utterance covariance whitened by the noise covariance). Factory scene: 73° → 21.6°
  mean error, 3/6 → 5/6 within 20°; clean scene unchanged. Offline over five scenes, within 15°:
  47 % → 77 % (see `ros2/doa_mod/README.md`).
* **stt_mod, intent_mod, dorai_simulator** — `/dorai_stt/status` sent `NaN` (`rtf` before the
  first final), which strict JSON parsers reject. Every published JSON now goes through a strict
  encoder (non-finite numbers become `null`); a test in each package fails if a bare `json.dumps`
  is used for a published message.

Still open: in the factory scene the quietest command ("go back to base", also lost by STT) gets
the welder's bearing; GEVD-MUSIC has not yet been scored on the robot's own recordings.

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
{"model": "whisper/large-v3-turbo-dorai-artaban", "rtf": null, "backlog_s": 0.0, "busy": false,
 "audio_s": 0.0, "finals": 0, "lag_median_s": null}

// /dorai_plan
{"text": "Artaban go back to base.", "plan": {"steps": [{"task": "goto", "target": "base"}]},
 "bearing_deg": 194.0, "engine": "llm", "latency_s": 1.168, "stamp": 1795542946.94,
 "confidence": 1.0, "min_token_prob": 0.999, "wake_edits": 0}

// /dorai_intent/decision
{"text": "Artaban go back to base.", "decision": "plan", "addressed": true, "wake_heard": "artaban",
 "wake_edits": 0, "wake_confidence": 1.0, "command": "go back to base", "engine": "llm",
 "confidence": 1.0, "min_token_prob": 0.999, "latency_s": 1.168, "tasks": ["goto"]}

// /dorai_stop  (factory scene, "Fargan stop." with the name mangled by welding noise)
"stop"

// /dorai_intent/decision for it
{"text": "Fargan stop.", "decision": "stop", "addressed": false, "wake_heard": "fargan",
 "wake_edits": 4, "wake_confidence": 0.43, "tasks": []}

// /dorai_intent/status
{"engine": "llm", "model": "intent-qwen35-4b-Q4_0.gguf", "n_plans": 0, "n_clarify": 0, "n_none": 0,
 "n_stop": 0, "n_invalid": 0, "n_ignored": 0, "n_error": 0, "mean_latency_s": null,
 "queue_depth": 0, "require_wake": true, "bearing_deg": null}

// /dorai_say  (factory scene, after an unclear command)
"Sorry, I didn't understand. Could you say that again?"
```
