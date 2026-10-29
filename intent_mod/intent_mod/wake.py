"""Deterministic pre-LLM gate: stop-phrase detection and wake-word stripping.

Both functions are pure, fast (microseconds) and never touch the LLM, so a stop
request cannot be delayed by model latency, a bad generation or a missing model.

* ``is_stop(text)``  -> 'estop' | 'stop' | None
  Fires only when the utterance is *nothing but* a stop request (optionally
  prefixed by the wake word and padded with fillers such as "please" / "now").
  "stop welding" is a weld task, "don't stop" is not a stop.
* ``strip_wake(text)`` -> (addressed, rest)
  Levenshtein match of the first 1-3 tokens against the wake word, tolerant to
  how the ASR spells an out-of-vocabulary name ("artiban", "art a ban", ...).
"""
from __future__ import annotations

import re

GREETINGS = {"hey", "hi", "hello", "ok", "okay", "yo", "hay"}
NEGATIONS = {"dont", "don't", "do", "not", "never", "no", "didnt", "didn't", "cant", "can't",
             "wont", "won't", "without", "nothing"}
# words that may accompany a bare stop request without changing its meaning
STOP_FILLER = {"please", "now", "immediately", "right", "there", "it", "moving", "everything",
               "robot", "here", "at", "once", "all", "just", "you", "the", "and", "quick",
               "quickly", "this", "instant", "moment", "this"}
MAX_STOP_TOKENS = 6


def normalise(text):
    """Lower-case, drop punctuation (keep apostrophes inside words), collapse spaces."""
    t = re.sub(r"[^a-z0-9' ]", " ", str(text or "").lower().replace("\u2019", "'"))
    return " ".join(t.split())


def lev(a, b):
    """Levenshtein edit distance."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] if x == y else 1 + min(prev[j - 1], prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


# ------------------------------------------------------------------ wake word
def strip_wake(text, wake_word="artaban", max_edits=2):
    """(addressed, rest). ``rest`` is the normalised text after the wake word (and an
    optional leading 'hey'/'ok'). If not addressed, ``rest`` is the normalised input."""
    toks = normalise(text).split()
    wake = normalise(wake_word).replace(" ", "")
    start = 1 if toks and toks[0] in GREETINGS else 0
    best = None                               # (edits, n_tokens)
    for n in (1, 2, 3):
        if start + n > len(toks):
            break
        joined = "".join(toks[start:start + n]).replace("'", "")
        if len(joined) < 4:
            continue
        d = lev(joined, wake)
        if d <= max_edits and (best is None or d < best[0]):
            best = (d, n)
    if best is None:
        return False, " ".join(toks)
    return True, " ".join(toks[start + best[1]:])


# ----------------------------------------------------------------- stop words
def _is_stop_word(t):
    return t in ("stop", "stopp", "stops", "stopped") or (     # "stopped": ASR spelling of "stop."
        len(t) >= 4 and t[0] == "s" and lev(t, "stop") <= 1 and t not in ("step", "shop", "stoop", "slop"))


def _is_halt(t):
    return len(t) >= 4 and t[0] == "h" and lev(t, "halt") <= 1


def _is_freeze(t):
    return len(t) >= 5 and t[0] == "f" and lev(t, "freeze") <= 2 and t not in ("freed", "frees")


def _is_emergency(t):
    return len(t) >= 6 and t[0] in "ei" and lev(t, "emergency") <= 3


def is_stop(text, wake_word="artaban", wake_max_edits=2):
    """'estop' for emergency stop, 'stop' for stop/halt/freeze, else None."""
    norm = normalise(text)
    if not norm:
        return None
    _, rest = strip_wake(norm, wake_word, wake_max_edits)
    toks = rest.split()
    if not toks or len(toks) > MAX_STOP_TOKENS or any(t in NEGATIONS for t in toks):
        return None
    # "e stop" / "e-stop" / "estop"
    joined = []
    for i, t in enumerate(toks):
        if t == "e" and i + 1 < len(toks) and _is_stop_word(toks[i + 1]):
            joined.append("emergency")
        elif t in ("estop", "estopp"):
            joined += ["emergency", "stop"]
        else:
            joined.append(t)
    toks = joined
    kinds = []
    for t in toks:
        if _is_emergency(t):
            kinds.append("emergency")
        elif _is_stop_word(t):
            kinds.append("stop")
        elif _is_halt(t) or _is_freeze(t):
            kinds.append("halt")
        elif t in STOP_FILLER or t in GREETINGS:
            continue
        else:
            return None                       # any other content word: not a bare stop ("stop welding")
    if not kinds:
        return None
    if "emergency" in kinds:
        return "estop"
    return "stop"
