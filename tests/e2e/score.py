#!/usr/bin/env python3
"""Score one recorded run against the scene's truth and print a summary + sample messages."""
import json, re, sys, glob, os, csv

out = sys.argv[1]
L = [json.loads(l) for l in open(os.path.join(out, "topics.jsonl"))]
by = {}
for r in L:
    by.setdefault(r["topic"], []).append(r)

truth = [r["data"] for r in by.get("/dorai_sim/truth", [])]
cmds = []
for t in truth:
    if isinstance(t, dict) and t.get("event") in ("command_start", "start", "command") or (isinstance(t, dict) and "text" in t and "bearing_deg" in t):
        cmds.append(t)
scene = next((t for t in truth if isinstance(t, dict) and "commands" in t), None)
if scene:
    cmds = scene["commands"]


def norm(s):
    return re.sub(r"[^a-z ]", "", s.lower()).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = [[i + j if i * j == 0 else 0 for j in range(len(h) + 1)] for i in range(len(r) + 1)]
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1][-1], len(r)


trans = [r["data"] for r in by.get("/dorai_transcript", [])]
errs = words = 0
print(f"== {os.path.basename(out)}: scene '{scene['name'] if scene else '?'}' rt60 {scene['room']['rt60_s'] if scene else '?'} s")
print(f"{'reference':40s} | {'transcript':40s}")
for i, c in enumerate(cmds):
    hyp = trans[i] if i < len(trans) else ""
    e, n = wer(c["text"], hyp)
    errs += e; words += n
    print(f"{c['text'][:40]:40s} | {hyp[:40]:40s}  ({e}/{n})")
print(f"WER {100 * errs / max(words, 1):.1f}% over {words} words; {len(trans)} transcripts for {len(cmds)} commands")

dec = [r["data"] for r in by.get("/dorai_intent/decision", [])]
for d in dec:
    print(f"  decision {d.get('decision'):8s} {d.get('text')!r:45s} tasks={d.get('tasks')} conf={d.get('confidence')} {d.get('latency_s')}s")

# DoA: bearing messages during each command window (truth times are scene-relative;
# align with the first truth message carrying a command start)
doa = by.get("/dorai_doa", [])
starts = [r for r in by.get("/dorai_sim/truth", []) if isinstance(r["data"], dict) and r["data"].get("event") == "command" and r["data"].get("phase") == "start"]
errs_deg = []
for r in starts:
    c = r["data"]
    t0 = r["t"]
    dur = c.get("t1", 0) - c.get("t0", 0) if "t1" in c else 2.0
    bs = [x["data"] for x in doa if t0 <= x["t"] <= t0 + dur + 0.6]
    if bs and "bearing_deg" in c:
        b = bs[-1]
        e = abs((b - c["bearing_deg"] + 180) % 360 - 180)
        errs_deg.append(e)
        print(f"  doa  truth {c['bearing_deg']:6.1f}  est {b:6.1f}  err {e:5.1f}  ({len(bs)} msgs)  {c.get('text','')}")
if errs_deg:
    print(f"DoA mean abs error {sum(errs_deg)/len(errs_deg):.1f} deg, within 20 deg: {sum(e <= 20 for e in errs_deg)}/{len(errs_deg)}")

dl = os.path.join(out, "delay.csv")
if os.path.exists(dl):
    rows = list(csv.DictReader(open(dl)))
    if rows:
        k = [c for c in rows[0] if c.endswith("_ms")]
        print("voice_mod delay (mean ms): " + ", ".join(f"{c}={sum(float(r[c]) for r in rows)/len(rows):.0f}" for c in k))
m = [r["data"] for r in by.get("/dorai_stt/metrics", [])]
if m:
    print("stt lag_s: " + ", ".join(str(round(x.get("lag_s", 0), 2)) for x in m))
