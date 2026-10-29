# intent_mod — Stage 3 of the dorai pipeline

`/dorai_transcript` (stt_mod) → **stop/wake gate** → **engine** (llama.cpp GGUF with a
schema grammar, or rule fallback) → `schema.validate()` → `/dorai_plan` | `/dorai_say`.
Contract and labelling conventions: [intent_design.md](intent_design.md).

## Topics
| Dir | Topic | Type | Content |
|---|---|---|---|
| in | `/dorai_transcript` | String | final text |
| in | `/dorai_doa` | Float32 | talker bearing, degrees (latest cached, attached to the next plan) |
| out | `/dorai_stop` | String | `stop` \| `estop` |
| out | `/dorai_plan` | String | JSON `{"text","plan","bearing_deg","engine","latency_s","stamp"}`; actionable, validated plans only |
| out | `/dorai_say` | String | spoken question/refusal for tts_mod (clarify, invalid or failed engine output) |
| out | `/dorai_intent/status` | String | JSON `{engine, model, n_plans, n_clarify, n_none, n_stop, n_invalid, n_ignored, n_error, mean_latency_s, queue_depth, require_wake, bearing_deg, stamp}` |

The `ExecuteTask` action to task_mod is not implemented (TODO marked in `node.py::_process`);
`/dorai_plan` is the interface until then.

## The stop path
`wake.is_stop()` runs in the subscription callback, **before** the wake gate and the queue.
`/dorai_stop` is therefore published immediately even while the engine is busy, with or
without the wake word, and with no model installed. It fires only when the utterance is
nothing but a stop request (stop / halt / freeze / emergency stop / e-stop / "stop stop",
fuzzy on ASR errors such as `stob`, `stopp`). `stop welding` is a weld task; `don't stop`
is not a stop. Stop utterances are never forwarded to the engine.

## Wake word
`strip_wake()` matches the first 1–3 tokens (after an optional hey/ok) to `wake_word`
within `wake_max_edits` Levenshtein edits: `artiban`, `art a ban`, `arta ban`, `Arthaban`.
With `require_wake: true` (default) unaddressed speech is ignored; with `false` every
transcript is a command. A bare wake word is dropped.

## Parameters (config/intent.yaml)
`engine` (auto|llm|rules), `model` (default `intent-qwen35-4b-Q4_0.gguf`, Qwen3.5-4B + Artaban LoRA merged; a file name,
an absolute path, or a huggingface.co file URL), `hf_repo` sensifai/dorai-intent-model, `auto_download` true, `lora` "", `use_grammar` false, `models_dir`,
`num_threads` 2, `n_ctx` 2048, `locations_file` (default: package `config/locations.yaml`),
`require_wake` true, `wake_word` artaban, `wake_max_edits` 2, `status_period_s` 5.0, topic names, `debug`.
`auto` uses the LLM if the GGUF is found and loads, else logs a warning and uses the rules;
`llm` raises instead of falling back.

## Where the model is found
First hit wins: `models_dir` param → `$DORAI_MODELS_DIR` → package `models/`
(`intent_mod/models` in the source tree, `share/intent_mod/models` installed; `setup.py`
installs `models/*.gguf` and `models/<dir>/*.gguf`) → `~/dorai/models` → `/root/.cache/dorai/models`.
Each root is searched as `<root>/<name>` then `<root>/<subdir>/<name>`.

If no root has it, the node **downloads it at startup** from
[huggingface.co/sensifai/dorai-intent-model](https://huggingface.co/sensifai/dorai-intent-model)
(`hf_repo`) into `models_dir`, or `~/dorai/models` when that is empty, and later starts find it there.
The download resumes from a `.part` file after an interruption. It is checked against the sha256 the Hub reports, and progress is logged every 10 %.
A private repo needs `$HF_TOKEN`. Other ways to pick the model:

    ros2 launch intent_mod intent.launch.py model:=/data/models/intent-qwen35-4b-Q4_0.gguf    # explicit file: never downloaded
    ros2 launch intent_mod intent.launch.py model:=https://huggingface.co/sensifai/dorai-intent-model/blob/main/intent-qwen35-4b-Q4_0.gguf
    ros2 launch intent_mod intent.launch.py auto_download:=false                              # offline: missing model -> rules (engine auto)

The LLM needs `pip install llama-cpp-python>=0.3.36` (pip-only, optional; 0.3.36 is the first with Qwen3.5).

## Run
    colcon build --packages-select intent_mod --symlink-install
    ros2 launch intent_mod intent.launch.py [engine:=rules] [require_wake:=false]
    ros2 topic pub --once /dorai_transcript std_msgs/String "{data: 'Artaban go to welding device x for check meter'}"

## Tests (no ROS, no llama_cpp needed)
    cd dorai && python -m pytest intent_mod/test -q
`test/rosmock.py` fakes rclpy/std_msgs. One optional test runs the real `LlamaIntentEngine`
on 3 utterances if the default GGUF is found in the model roots above (or `$DORAI_TEST_GGUF`)
and `llama_cpp` is importable; otherwise it is skipped.

## The model (train/)
`intent-qwen35-4b-Q4_0.gguf` = Qwen3.5-4B + a LoRA trained on synthetic Artaban commands with
STT noise, merged and quantized (2.6 GB). Rebuild on an Apple-silicon Mac:

    python intent_mod/train/gen_data.py                                   # 1200 train rows + dev/test/paraphrase splits
    python intent_mod/train/lora_throttled.py -- -c intent_mod/train/lora_qwen35_4b.yaml --model <Qwen3.5-4B dir>
    python intent_mod/train/export_gguf.py --base <Qwen3.5-4B dir> --adapter ~/dorai/models/lora/intent-qwen35-4b-mlx \
        --converter <llama.cpp>/convert_hf_to_gguf.py --out ~/dorai/models/qwen35-4b-intent
    python intent_mod/train/eval_intent.py --model ~/dorai/models/qwen35-4b-intent/intent-qwen35-4b-Q4_0.gguf

`lora_throttled.py` caps the run at 70 % GPU duty / half the RAM and trains the Gated DeltaNet
layers through `gdn_chunked.py` (mlx_lm's own training path is one op per token and needs >17 GB).
Exact match, no world seen in training: dev 98.0 %, test 98.3 %, paraphrase 92.6 %; rules 69 / 51 / 46 %.
Published as `intent-qwen35-4b-Q4_0.gguf` in huggingface.co/sensifai/dorai-intent-model (fetched automatically, see above).
