# intent_mod design

```
/dorai_transcript ──► wake.py (deterministic) ──► stop / e-stop phrase? ──yes──► /dorai_stop → twist_mux
                                 │ no
                                 ▼
                     LLM (llama.cpp, grammar = plan_schema(locations.yaml))
                                 ▼
                     schema.validate()  ── problems / clarify ──► tts_mod "did you mean …?"
                                 ▼ ok
                     ExecuteTask(plan JSON) ──► task_mod (behavior tree) ──► Nav2 / vision_mod / tts_mod
```

## Plan format
`{"steps": [{"task": <name>, <args>}, ...]}` with 1–4 steps. Tasks and arguments are in
`intent_mod/schema.py::TASKS`. Location and object arguments are **enums built at runtime
from `config/locations.yaml`**. The grammar therefore makes it impossible to name a place that
doesn't exist, and adding a place needs no retraining.

## Labelling conventions (training data and the runtime contract)
| Utterance | Plan |
|---|---|
| "go to welding device X for check meter" | `[{"task":"inspect_meter","target":"welder_x"}]` (one step; it includes driving there and the spoken report) |
| "go to X and then check the meter on Y" | `[goto X, inspect_meter Y]` |
| charger / charging station / go charge / dock | `[{"task":"dock"}]` (never `goto charging_station`) |
| go home / return to base | `[{"task":"goto","target":"base"}]` |
| place or object not in locations.yaml | single `clarify` step (`unknown_target` / `unknown_object`) |
| unclear fragment ("go to the", "that one") | single `clarify` (`unclear`) |
| chatter not aimed at the robot | single `none` |
| stop, halt, freeze, emergency stop | **never reaches the LLM**: caught by `wake.py` and published straight to twist_mux |

`inspect_meter` on a place with an empty `meters` list is rejected by `validate()` and
answered with a clarify.

## Safety rules
* The LLM output is a *request*. task_mod re-checks preconditions such as battery level, the
  e-stop state and whether the target is reachable.
* `false_action`, meaning the model acted when the gold answer was clarify or none, is the
  key safety metric and is reported separately from accuracy.
