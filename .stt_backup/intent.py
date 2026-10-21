"""Map a transcript onto the fixed robot command grammar.

The wake name "Artaban" is not an English word and not in the Nemotron
vocabulary, so the ASR spells it from sub-words ("Arthaban", "R Taban",
"Ortha one"...). Two steps make transcripts usable:

1. wake normalisation: the best 1-3 word span within ``wake_max_edits``
   letter edits of "artaban" is replaced by the name;
2. intent: the nearest grammar command by *character* edit distance on the
   wake-normalised text (inflections such as "turned right" or "stopped
   building" still land on "turn right" / "stop welding").

``score`` = 1 - normalised distance, in [0, 1]. A match is ``accepted`` when
score >= ``min_score``; otherwise the robot should ask the user to repeat
(report 2026-10-20, recommendation 4). Identical scoring to
stt repo scripts/real_mic/artaban_mic_test.py.
"""
from __future__ import annotations

import json
import os
import re

DEFAULT_COMMANDS = [  # (id, text) — same 30 commands as the real-mic test
    ("wake", "hey artaban"), ("wake2", "artaban"),
    ("come", "artaban come here"), ("follow", "artaban follow me"),
    ("stop", "artaban stop"), ("estop", "artaban emergency stop"),
    ("wait", "artaban wait here"), ("back", "artaban go back"),
    ("left", "artaban turn left"), ("right", "artaban turn right"),
    ("fwd2", "artaban move forward two meters"), ("slow", "artaban slow down"),
    ("base", "artaban return to base"), ("charge", "artaban go to the charging station"),
    ("weld_on", "artaban start welding"), ("weld_off", "artaban stop welding"),
    ("grip_open", "artaban open the gripper"), ("grip_close", "artaban close the gripper"),
    ("pick", "artaban pick up the hammer"), ("toolbox", "artaban bring me the toolbox"),
    ("hold", "artaban hold the plate"), ("lift", "artaban lift the arm"),
    ("lower", "artaban lower the arm"), ("battery", "artaban what is your battery level"),
    ("status", "artaban report status"), ("lights_on", "artaban lights on"),
    ("lights_off", "artaban lights off"), ("cancel", "artaban cancel that"),
    ("yes", "artaban yes confirm"), ("no", "artaban no"),
]


def norm_words(s):
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] if x == y else 1 + min(prev[j - 1], prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def load_commands(path=""):
    """Grammar file: JSON or YAML list of {id, text} (or a {id: text} map)."""
    if not path:
        return list(DEFAULT_COMMANDS)
    raw = open(os.path.expanduser(path)).read()
    try:
        data = json.loads(raw)
    except ValueError:
        import yaml
        data = yaml.safe_load(raw)
    if isinstance(data, dict):
        data = data.get("commands", data)
    if isinstance(data, dict):
        bad = [k for k in data if not isinstance(k, str)]
        if bad:  # YAML 1.1 turns unquoted yes/no/on/off keys into booleans
            raise ValueError(f"{path}: quote the command ids {bad} (YAML reads them as booleans)")
        return [(k, str(v)) for k, v in data.items()]
    return [(str(d["id"]), str(d["text"])) for d in data]


class IntentMatcher:
    def __init__(self, commands=None, wake_word="artaban", wake_max_edits=2,
                 min_score=0.6, require_wake=False):
        self.commands = list(commands or DEFAULT_COMMANDS)
        self.wake = wake_word.lower()
        self.wake_max_edits = int(wake_max_edits)
        self.min_score = float(min_score)
        self.require_wake = bool(require_wake)
        self._norm = [(cid, " ".join(norm_words(t)), t) for cid, t in self.commands]

    def wake_normalise(self, words):
        best = None
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                d = lev("".join(words[i:i + n]).replace("'", ""), self.wake)
                if d <= self.wake_max_edits and (best is None or d < best[0]):
                    best = (d, i, n)
        if best is None:
            return words, False
        _, i, n = best
        return words[:i] + [self.wake] + words[i + n:], True

    def match(self, text):
        words = norm_words(text)
        if not words:
            return {"text": text, "normalized": "", "intent": "", "command": "",
                    "score": 0.0, "wake": False, "accepted": False}
        wn, wake = self.wake_normalise(words)
        w = " ".join(wn)
        cid, ctext, orig = min(
            self._norm, key=lambda c: (lev(c[1], w) / max(len(c[1]), 1), -len(c[1])))
        score = max(0.0, 1.0 - lev(ctext, w) / max(len(ctext), len(w), 1))
        ok = score >= self.min_score and (wake or not self.require_wake)
        return {"text": text, "normalized": w, "intent": cid, "command": orig,
                "score": round(score, 3), "wake": wake, "accepted": bool(ok)}
