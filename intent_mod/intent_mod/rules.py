"""Deterministic rule-based intent parser (no ML).

Role: (1) the baseline the LLM must beat, (2) an offline fallback when no LLM model is
installed. Same output contract as the LLM: {"steps": [...]} over schema.TASKS.

Pipeline: STT-confusion fixes -> clause split on connectors -> per clause: keyword
rules pick the task, fuzzy n-gram match against locations.yaml / objects aliases picks
the argument (difflib ratio >= 0.82). Navigation verb with no matching place -> clarify.
"""
import difflib
import re

STT_FIX = [("wedding", "welding"), ("welling", "welding"), ("metre", "meter"), ("meeter", "meter"),
           ("gage", "gauge"), ("gripe er", "gripper"), ("charting", "charging"), ("chuck", "check"),
           ("tool books", "toolbox"), ("lice", "lights"), ("ready", "reading")]
CONNECT = r"\b(?:and then|and after that|after that|then|and)\b"
FILLER = {"please", "okay", "hey", "uh", "um", "now", "thanks", "right", "can", "you", "the", "a", "to", "me",
          "of", "on", "at", "for", "go", "over", "i", "need", "want", "and"}
DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
NUMW = {"half a": 0.5, "one and a half": 1.5, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10}
DIST = r"(half a|one and a half|one|two|three|four|five|six|seven|eight|nine|ten|\d+(?:\.\d+)?) meters?"


def spell_digits(t):
    """'welding station 2' -> 'welding station two' (aliases are spelled); '1.5 meters' is left alone."""
    return re.sub(r"(?<![\d.])(\d)(?!\.?\d)", lambda m: DIGITS[int(m.group(1))], t)


def _norm(t):
    t = " " + spell_digits(t.lower()) + " "
    for a, b in STT_FIX:
        t = re.sub(rf"\b{a}\b", b, t)
    return " ".join(re.sub(r"(?!(?<=\d)\.(?=\d))[^a-z0-9' ]", " ", t).split())   # keep '1.5'


def _best(text, table, thr=0.82):
    """Fuzzy-match any alias in `table` {id: [aliases]} against n-grams of text."""
    words = text.split()
    best = (0.0, None)
    for i, aliases in table.items():
        for a in aliases:
            a = _norm(a).replace("the ", "")
            n = len(a.split())
            for k in range(max(1, n - 1), n + 2):
                for j in range(0, max(1, len(words) - k + 1)):
                    g = " ".join(words[j:j + k])
                    r = difflib.SequenceMatcher(None, a, g).ratio()
                    if r > best[0] or (r == best[0] and best[1] and len(a) > len(best[1][1])):
                        best = (r, (i, a))
    return best[1][0] if best[0] >= thr else None


class RuleParser:
    def __init__(self, locs, objs):
        self.locs = {i: l.get("aliases", []) for i, l in locs.items()}
        self.loc_meters = {i: bool(l.get("equipment", {}).get("meters")) for i, l in locs.items()}
        self.objs = {i: o.get("aliases", []) for i, o in objs.items()}

    def parse(self, text):
        t = _norm(text)
        clauses = [c.strip() for c in re.split(CONNECT, t) if c.strip()]
        steps = []
        for c in clauses:
            s = self.clause(c)
            if s is None:
                continue
            steps.append(s)
        acts = [s for s in steps if s["task"] not in ("clarify", "none")]
        if not steps:
            return {"steps": [{"task": "none"}]}
        if any(s["task"] == "clarify" for s in steps):
            return {"steps": [next(s for s in steps if s["task"] == "clarify")]}
        return {"steps": acts[:4] if acts else [{"task": "none"}]}

    def clause(self, c):
        has = lambda *ws: any(re.search(rf"\b{w}\b", c) for w in ws)  # noqa: E731
        # --- navigation-like tasks with a place
        dist = re.search(DIST, c)                       # "two meters forward" is a distance, not a gauge
        meter = has("meter", "meters", "gauge", "gauges", "reading") and not dist
        visual = has("inspect", "look", "visual", "visually", "damage", "see if") and not meter
        nav = has("go", "goes", "head", "walk", "move to", "navigate", "get over", "take yourself", "at")
        if has("charge", "charger", "charging", "recharge", "dock yourself", "plug yourself"):
            return {"task": "dock"}
        if has("back to base", "back to the base", "back home", "return to base", "return to the base",
               "return home", "go home", "head home"):
            return {"task": "goto", "target": "base"}
        # --- posture, inspection round, look at an object
        if has("sit", "sit down", "take a seat"):
            return {"task": "posture", "pose": "sit"}
        if has("stand up", "get up", "stand"):
            return {"task": "posture", "pose": "stand"}
        if has("lie down", "lay down"):
            return {"task": "posture", "pose": "lie_down"}
        if has("patrol", "rounds", "inspection round") or \
                (has("inspection") and has("start", "begin", "stop", "end", "finish", "resume")
                 and not _best(re.sub(r"\binspection\b", "", c), self.locs)):
            return {"task": "inspection", "state": "stop" if has("stop", "end", "finish") else "start"}
        if has("look at", "look over at", "show me", "point the camera at"):
            o = _best(c, self.objs)
            if o and not _best(c, self.locs):
                return {"task": "look_at", "object": o}
        if meter or visual or (nav and not has("forward", "back", "backward", "ahead", "reverse", "meters", "meter")):
            tgt = _best(c, self.locs)
            if tgt == "charging_station":
                return {"task": "dock"}
            if tgt:
                if meter and self.loc_meters.get(tgt):
                    return {"task": "inspect_meter", "target": tgt}
                if visual or has("check on", "check") and not meter:
                    return {"task": "inspect_visual", "target": tgt}
                return {"task": "goto", "target": tgt}
            if has("home", "base"):
                return {"task": "goto", "target": "base"}
            if meter or visual or has("go to", "head to", "move to", "navigate to", "walk over to", "get over to"):
                rest = [w for w in c.split() if w not in FILLER and w not in
                        ("check", "meter", "gauge", "inspect", "read", "go", "head", "walk", "navigate")]
                if meter and not [w for w in rest if w not in ("meters", "gauges", "reading", "this", "that", "what", "says", "is", "it")]:
                    return {"task": "read_meter"}
                return {"task": "clarify", "reason": "unknown_target" if rest else "unclear"}
        # --- motion
        if has("come here", "come over", "come to me", "over here", "get over here", "come closer", "walk to me"):
            return {"task": "come_here"}
        if has("follow", "come with me", "behind me", "walk with me"):
            return {"task": "follow_me"}
        if has("turn", "rotate", "face", "spin"):
            d = "left" if has("left") else "right" if has("right") else "around" if has("around") else None
            return {"task": "turn", "direction": d} if d else {"task": "clarify", "reason": "unclear"}
        m = dist
        if m and has("forward", "ahead", "back", "backward", "reverse", "move", "drive", "step", "go"):
            v = NUMW.get(m.group(1)) or float(m.group(1))
            back = has("back up", "move back", "reverse", "backward")
            return {"task": "move", "direction": "backward" if back else "forward", "meters": v}
        if has("go back", "return to where", "previous spot", "last place"):
            return {"task": "go_back"}
        if has("wait", "stay", "hold your position", "stay put", "don't move", "park"):
            return {"task": "wait"}
        if has("slow", "slower", "easy"):
            return {"task": "set_speed", "speed": "slow"}
        if has("faster", "speed up", "quicker"):
            return {"task": "set_speed", "speed": "fast"}
        if has("normal speed"):
            return {"task": "set_speed", "speed": "normal"}
        # --- tools
        if has("weld", "welding", "arc"):
            return {"task": "weld", "state": "stop" if has("stop", "end", "finish", "kill") else "start"}
        if has("gripper", "hand", "grip", "let go"):
            return {"task": "gripper", "state": "open" if has("open", "release", "let go") else "close"}
        if has("arm"):
            return {"task": "arm", "state": "lower" if has("lower", "down", "drop") else "lift"}
        if has("lights", "light", "headlights"):
            return {"task": "lights", "state": "off" if has("off", "kill") else "on"}
        for verb, task in (("bring", "bring"), ("fetch", "bring"), ("hand me", "bring"), ("get me", "bring"),
                           ("pick", "pick"), ("grab", "pick"), ("take", "pick"), ("get", "pick"), ("lift", "pick"),
                           ("hold", "hold")):
            if has(verb):
                o = _best(c, self.objs)
                if o:
                    return {"task": task, "object": o}
                rest = [w for w in c.split() if w not in FILLER and w not in verb.split()]
                return {"task": "clarify", "reason": "unknown_object" if rest else "unclear"}
        # --- status / dialog
        if has("battery", "charge left"):
            return {"task": "report", "what": "battery"}
        if has("status", "how are you", "are you ok"):
            return {"task": "report", "what": "status"}
        if has("cancel", "never mind", "forget it", "abort", "scratch that", "don't do that"):
            return {"task": "cancel"}
        if has("no", "negative", "don't"):
            return {"task": "confirm", "answer": "no"}
        if has("yes", "correct", "that's right", "confirm"):
            return {"task": "confirm", "answer": "yes"}
        if has("do the thing", "you know what", "that one", "do it again") or c in ("go", "check"):
            return {"task": "clarify", "reason": "unclear"}
        return None
