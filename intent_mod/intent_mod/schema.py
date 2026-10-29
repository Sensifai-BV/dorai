"""intent_mod plan schema: task vocabulary, JSON schema, prompt and validation.

Engine-agnostic (no llama.cpp import). The JSON schema built here is turned into a
llama.cpp grammar by the engine, so the model can only emit plans whose task names,
location ids and object ids exist. A plan is

    {"steps": [ {"task": "...", ...}, ... ]}      # 1..MAX_STEPS steps

`clarify` and `none` must be the only step (checked by `validate`, not the grammar).
Stop / emergency stop are NOT tasks: they are caught deterministically before the
LLM (see wake.py) and go straight to twist_mux.
"""
import json

import yaml

MAX_STEPS = 4

# task -> {arg: enum | "LOCATION" | "OBJECT" | "NUMBER"}. Argument order is fixed
# (grammar emits keys in this order), which also keeps exact-match scoring simple.
TASKS = {
    "goto":           {"target": "LOCATION"},
    "inspect_meter":  {"target": "LOCATION"},   # go there + read + speak the gauge(s)
    "inspect_visual": {"target": "LOCATION"},   # go there + look/photo + report
    "read_meter":     {},                       # read the gauge in front of the robot (no place named)
    "inspection":     {"state": ["start", "stop"]},   # the inspection round over all stations
    "look_at":        {"object": "OBJECT"},     # point the camera at an object and describe it
    "dock":           {},                       # go charge
    "come_here":      {},                       # walk to the talker (doa bearing)
    "follow_me":      {},
    "wait":           {},
    "posture":        {"pose": ["sit", "stand", "lie_down"]},
    "go_back":        {},                       # return to previous pose
    "turn":           {"direction": ["left", "right", "around"]},
    "move":           {"direction": ["forward", "backward"], "meters": "NUMBER"},
    "set_speed":      {"speed": ["slow", "normal", "fast"]},
    "weld":           {"state": ["start", "stop"]},
    "gripper":        {"state": ["open", "close"]},
    "arm":            {"state": ["lift", "lower"]},
    "pick":           {"object": "OBJECT"},
    "bring":          {"object": "OBJECT"},     # fetch and bring to the talker
    "hold":           {"object": "OBJECT"},
    "report":         {"what": ["battery", "status"]},
    "lights":         {"state": ["on", "off"]},
    "cancel":         {},
    "confirm":        {"answer": ["yes", "no"]},
    "clarify":        {"reason": ["unknown_target", "unknown_object", "unclear"]},
    "none":           {},                       # not a robot command
}
SOLO_TASKS = {"clarify", "none"}

TASK_HELP = {
    "goto": "go to a place",
    "inspect_meter": "go to a place and read its meter/gauge aloud",
    "inspect_visual": "go to a place and visually inspect it",
    "read_meter": "read the meter/gauge right here (no place named)",
    "inspection": "start/stop the inspection round",
    "look_at": "look at an object",
    "dock": "go to the charger and charge",
    "come_here": "come to the speaker",
    "follow_me": "follow the speaker",
    "wait": "stay / wait here",
    "posture": "sit down / stand up / lie down",
    "go_back": "go back to the previous position",
    "turn": "turn left/right/around",
    "move": "drive forward/backward N meters",
    "set_speed": "slow down / normal / speed up",
    "weld": "start/stop welding",
    "gripper": "open/close gripper",
    "arm": "lift/lower the arm",
    "pick": "pick up an object",
    "bring": "bring an object to the speaker",
    "hold": "hold an object",
    "report": "say battery level or status",
    "lights": "lights on/off",
    "cancel": "cancel the current task",
    "confirm": "answer yes/no to a question",
    "clarify": "the request names a place/object not in the lists, or is unclear",
    "none": "speech not addressed as a robot command",
}


def load_world(path):
    with open(path) as f:
        w = yaml.safe_load(f)
    locs = {l["id"]: l for l in w.get("locations", [])}
    objs = {o["id"]: o for o in w.get("objects", [])}
    return locs, objs


def plan_schema(location_ids, object_ids):
    """JSON schema for a plan over the given world (anyOf one schema per task)."""
    variants = []
    for task, args in TASKS.items():
        props = {"task": {"const": task}}
        for a, kind in args.items():
            if kind == "LOCATION":
                props[a] = {"enum": list(location_ids)}
            elif kind == "OBJECT":
                props[a] = {"enum": list(object_ids)}
            elif kind == "NUMBER":
                props[a] = {"type": "number"}
            else:
                props[a] = {"enum": list(kind)}
        variants.append({"type": "object", "properties": props,
                         "required": list(props), "additionalProperties": False})
    return {"type": "object",
            "properties": {"steps": {"type": "array", "items": {"anyOf": variants},
                                     "minItems": 1, "maxItems": MAX_STEPS}},
            "required": ["steps"], "additionalProperties": False}


def fewshot_examples(locs, objs):
    """~12 worked examples rendered with ids from THIS world (no foreign ids leak in)."""
    meter = [i for i, l in locs.items() if l.get("equipment", {}).get("meters")]
    other = [i for i in locs if i not in meter and i not in ("base", "charging_station")]
    m1 = meter[0] if meter else next(iter(locs))
    m2 = meter[1] if len(meter) > 1 else m1
    o1 = next(iter(objs)) if objs else None
    a = lambda i: locs[i].get("aliases", [i])[0]  # noqa: E731
    ex = [(f"pls read whatever the dial says over at {a(m1)}", [{"task": "inspect_meter", "target": m1}]),
          (f"get yourself next to {a(m2)}", [{"task": "goto", "target": m2}]),
          (f"anything broken around {a(other[0] if other else m2)} have a look", [{"task": "inspect_visual", "target": other[0] if other else m2}]),
          ("you need a recharge go dock", [{"task": "dock"}]),
          ("come to where i am standing", [{"task": "come_here"}]),
          ("swing left then roll ahead one meter", [{"task": "turn", "direction": "left"},
                                                    {"task": "move", "direction": "forward", "meters": 1}]),
          (f"grab a reading at {a(m2)} and afterwards head to base", [{"task": "inspect_meter", "target": m2},
                                                                       {"task": "goto", "target": "base"}]),
          ("go down to the basement", [{"task": "clarify", "reason": "unknown_target"}]),
          ("fetch me the bucket of paint", [{"task": "clarify", "reason": "unknown_object"}]),
          ("uh go over by the", [{"task": "clarify", "reason": "unclear"}]),
          ("the forklift needs new tyres", [{"task": "none"}]),
          ("tell me how full your battery is", [{"task": "report", "what": "battery"}])]
    if o1:
        ex.append((f"carry {objs[o1].get('aliases', [o1])[0]} over to me", [{"task": "bring", "object": o1}]))
    return [(t, {"steps": st}) for t, st in ex]


def system_prompt(locs, objs, with_task_help=True, fewshot=False):
    """Compact, constant system prompt (constant => llama.cpp reuses its KV cache)."""
    L = ["You turn a spoken command for the robot Artaban into a JSON plan.",
         'Answer only with JSON: {"steps":[{"task":...},...]} (1-%d steps).' % MAX_STEPS]
    if with_task_help:
        L.append("Tasks:")
        for t, args in TASKS.items():
            sig = ", ".join(f"{a}" for a in args)
            L.append(f"- {t}({sig}): {TASK_HELP[t]}")
    L.append("Places (id: spoken names):")
    for i, l in locs.items():
        L.append(f"- {i}: {', '.join(l.get('aliases', []))}")
    L.append("Objects (id: spoken names):")
    for i, o in objs.items():
        L.append(f"- {i}: {', '.join(o.get('aliases', []))}")
    L.append("Rules: use only ids listed above. If a place or object is not listed, "
             "answer a single clarify step. 'check the meter/gauge at a place' means inspect_meter; with no place it means read_meter. "
             "Text may contain speech-recognition errors.")
    if fewshot:
        L.append("Use the fewest steps that do what was asked. Examples:")
        for t, plan in fewshot_examples(locs, objs):
            L.append(f"{t} => {json.dumps(plan, separators=(',', ':'))}")
    return "\n".join(L)


def validate(plan, locs, objs):
    """Semantic checks the grammar cannot express. Returns list of problems ([] = ok)."""
    errs = []
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list) or not (1 <= len(steps) <= MAX_STEPS):
        return ["bad steps"]
    for s in steps:
        t = s.get("task")
        if t not in TASKS:
            errs.append(f"unknown task {t}")
            continue
        if set(s) - {"task"} != set(TASKS[t]):
            errs.append(f"{t}: bad args {sorted(s)}")
        for k, spec in TASKS[t].items():            # enums: the grammar is off by default (use_grammar)
            if isinstance(spec, list) and k in s and s[k] not in spec:
                errs.append(f"{t}: bad {k} {s[k]!r}")
        if "target" in s and s["target"] not in locs:
            errs.append(f"{t}: unknown target {s['target']}")
        if "object" in s and s["object"] not in objs:
            errs.append(f"{t}: unknown object {s['object']}")
        if t == "inspect_meter" and s.get("target") in locs and \
                not locs[s["target"]].get("equipment", {}).get("meters"):
            errs.append(f"inspect_meter: {s['target']} has no meters")
        if t == "move" and not (0 < float(s.get("meters", 0)) <= 20):
            errs.append("move: meters out of range")
    if len(steps) > 1 and any(s.get("task") in SOLO_TASKS for s in steps):
        errs.append("clarify/none must be the only step")
    return errs


def canonical(plan):
    """Stable string for exact-match scoring."""
    steps = []
    for s in plan.get("steps", []):
        s = dict(s)
        if "meters" in s:
            s["meters"] = round(float(s["meters"]), 2)
        steps.append(s)
    return json.dumps(steps, sort_keys=True)
