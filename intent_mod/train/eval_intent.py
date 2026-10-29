#!/usr/bin/env python3
"""Score a GGUF (or the rule engine) on the intent splits, through the same path the node uses:

    is_stop -> strip_wake -> engine.parse -> schema.validate -> guard.check

Metrics per split (report section 3 definitions):
  exact          predicted plan == gold plan (schema.canonical)
  unsafe         acted (actionable plan) when the gold answer was clarify/none
  refusal        clarify/none (or invalid) when the gold plan was actionable
  invalid        output failed validate() (node says "please repeat")
  *_guard        the same after the grounding guard

    python intent_mod/train/eval_intent.py --model path/to.gguf [--splits dev test paraphrase] [--limit N]
    python intent_mod/train/eval_intent.py --rules
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from intent_mod import wake  # noqa: E402
from intent_mod.engine import LlamaIntentEngine, RuleIntentEngine  # noqa: E402
from intent_mod.guard import check as guard_check  # noqa: E402
from intent_mod.schema import SOLO_TASKS, canonical, load_world, validate  # noqa: E402

SOLO = SOLO_TASKS


def worlds():
    shop = load_world(os.path.join(os.path.dirname(HERE), "config", "locations.yaml"))
    with open(os.path.join(HERE, "data", "worlds.json")) as f:
        w = json.load(f)
    out = {k: ({l["id"]: l for l in v["locations"]}, {o["id"]: o for o in v["objects"]}) for k, v in w.items()}
    out["sample_shop"] = shop
    return out


def actionable(plan):
    st = plan.get("steps") or []
    return bool(st) and st[0].get("task") not in SOLO


def score(engine, rows, W, verbose=False):
    cur, res = None, []
    for r in rows:
        stop = wake.is_stop(r["text"])
        if r["plan"] == "STOP":
            res.append({"text": r["text"], "gold": "STOP", "pred": stop or "not stopped", "exact": bool(stop),
                        "exact_guard": bool(stop), "lat": 0.0})
            continue
        if stop:                                            # gate swallowed a real command: count as a miss
            res.append({"text": r["text"], "gold": r["plan"], "pred": "STOP", "exact": False, "exact_guard": False,
                        "lat": 0.0, "invalid": False, "unsafe": True, "refusal": False})
            continue
        if r["world"] != cur:
            cur = r["world"]
            engine.reload_world(*W[cur])
        locs, objs = W[cur]
        addressed, rest = wake.strip_wake(r["text"])
        cmd = rest if addressed and rest else wake.normalise(r["text"])
        plan, lat = engine.parse(cmd)
        try:
            bad = bool(validate(plan, locs, objs))
        except Exception:  # noqa: BLE001
            bad = True
        if bad:
            plan = {"steps": [{"task": "clarify", "reason": "unclear"}]}
        gplan, _ = guard_check(cmd, plan, locs, objs) if not bad else (plan, None)
        gplan = {"steps": gplan["steps"]}
        gold_act = actionable(r["plan"])
        e = {"text": r["text"], "gold": r["plan"], "pred": plan, "lat": lat, "invalid": bad,
             "exact": canonical(plan) == canonical(r["plan"]),
             "exact_guard": canonical(gplan) == canonical(r["plan"]),
             "unsafe": (not gold_act) and actionable(plan), "unsafe_guard": (not gold_act) and actionable(gplan),
             "refusal": gold_act and not actionable(plan), "refusal_guard": gold_act and not actionable(gplan)}
        res.append(e)
        if verbose and not e["exact"]:
            print(f"  MISS {r['text']!r}\n       gold {json.dumps(r['plan']['steps'])}\n       pred {json.dumps(plan.get('steps'))}")
    return res


def summary(res):
    n = len(res)
    gold_na = [e for e in res if e["gold"] != "STOP" and not actionable(e["gold"])]
    gold_a = [e for e in res if e["gold"] != "STOP" and actionable(e["gold"])]
    lats = [e["lat"] for e in res if e["lat"]]
    pct = lambda xs, k: 100.0 * sum(bool(x.get(k)) for x in xs) / max(1, len(xs))  # noqa: E731
    return {"n": n, "exact": pct(res, "exact"), "exact_guard": pct(res, "exact_guard"),
            "unsafe": pct(gold_na, "unsafe"), "unsafe_guard": pct(gold_na, "unsafe_guard"),
            "refusal": pct(gold_a, "refusal"), "refusal_guard": pct(gold_a, "refusal_guard"),
            "invalid": pct(res, "invalid"),
            "lat_p50": statistics.median(lats) if lats else 0.0,
            "lat_p95": sorted(lats)[int(0.95 * (len(lats) - 1))] if lats else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["dev", "test", "paraphrase"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--grammar", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    W = worlds()
    shop = W["sample_shop"]
    t0 = time.time()
    eng = RuleIntentEngine(*shop) if a.rules else \
        LlamaIntentEngine(a.model, *shop, n_threads=a.threads, n_ctx=4096, use_grammar=a.grammar)
    print(f"engine {eng.name} {eng.model} loaded in {time.time() - t0:.1f}s")
    out = {"model": eng.model, "grammar": a.grammar, "splits": {}}
    for sp in a.splits:
        with open(os.path.join(HERE, "data", f"{sp}.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        if a.limit:
            rows = rows[:a.limit]
        rows.sort(key=lambda r: r["world"])               # one prefill per world
        print(f"== {sp} ({len(rows)})")
        res = score(eng, rows, W, a.verbose)
        s = summary(res)
        out["splits"][sp] = {"summary": s, "rows": res}
        print("   " + "  ".join(f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
