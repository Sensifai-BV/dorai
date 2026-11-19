"""Deterministic grounding guard: the LLM proposes, this module disposes.

A plan step that names a place (`target`) or an object (`object`) is only accepted if
one of that id's spoken names (aliases in locations.yaml, or the id itself with '_'
read as a space) is actually present in the utterance -- fuzzy, so STT errors such as
'wedding device x' still pass. Otherwise the whole plan is replaced by a confirmation
question ("did you mean <alias>?") routed to tts_mod, and nothing moves.

Rationale: the grammar already guarantees the id EXISTS; it cannot guarantee the
speaker SAID it. Measured on the benchmark (results/guard_eval.csv) this removes
wrong-place motions caused by the model 'snapping' an unknown place (the furnace) to a
known one (the boiler).
"""
import difflib
import re

from .rules import STT_FIX, spell_digits

GENERIC = {"the", "a", "an", "room", "machine", "device", "station", "number", "unit", "cell", "area"}
THRESH = 0.80


def _norm(t):
    t = " " + spell_digits(t.lower()) + " "
    for a, b in STT_FIX:
        t = re.sub(rf"\b{a}\b", b, t)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", t).split())


def _names(i, table):
    names = list(table.get(i, {}).get("aliases", [])) + [i.replace("_", " ")]
    return {_norm(n) for n in names if _norm(n)}


def mentioned(text, ident, table, thresh=THRESH):
    """Best fuzzy score of any spoken name of `ident` against n-grams of `text`."""
    words = _norm(text).split()
    best = 0.0
    for name in _names(ident, table):
        nw = name.split()
        core = [w for w in nw if w not in GENERIC] or nw
        k = len(nw)
        for n in {max(1, k - 1), k, k + 1}:
            for j in range(0, max(1, len(words) - n + 1)):
                g = " ".join(words[j:j + n])
                best = max(best, difflib.SequenceMatcher(None, name, g).ratio())
        # all distinctive words of the name present (any order) also counts
        if all(any(difflib.SequenceMatcher(None, c, w).ratio() >= 0.85 for w in words) for c in core):
            best = max(best, 0.95)
    return best


def check(text, plan, locs, objs, thresh=THRESH):
    """Return (plan_out, reason). reason is None when the plan passes unchanged."""
    for s in plan.get("steps", []):
        if "target" in s and s["target"] not in ("base",):     # 'go home/back to base' has many forms
            if s["target"] == "charging_station":
                continue
            if mentioned(text, s["target"], locs) < thresh:
                alias = (locs.get(s["target"], {}).get("aliases") or [s["target"]])[0]
                return ({"steps": [{"task": "clarify", "reason": "unknown_target"}],
                         "confirm": {"target": s["target"], "say": f"did you mean {alias}?"}}, "ungrounded_target")
        if "object" in s and mentioned(text, s["object"], objs) < thresh:
            alias = (objs.get(s["object"], {}).get("aliases") or [s["object"]])[0]
            return ({"steps": [{"task": "clarify", "reason": "unknown_object"}],
                     "confirm": {"object": s["object"], "say": f"did you mean {alias}?"}}, "ungrounded_object")
    return plan, None
