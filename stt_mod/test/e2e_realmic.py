#!/usr/bin/env python3
"""End-to-end ROS 2 test of a RUNNING stt_mod node on real-microphone takes.

Plays each take of a NeMo-style manifest (one JSON per line: audio_filepath,
text, optional condition) into /dorai_clean_audio in real time, framed
exactly as voice_mod frames it ([capture_s, 16000, 1, seq] + samples,
data_offset 4). It collects /dorai_transcript and scores WER per condition.
After each take it waits until the node has drained its queue
(/dorai_stt/status backlog_s == 0) and has stayed quiet, so takes never
overlap even when the CPU is slower than real time.

  terminal 1:  ros2 run stt_mod stt --ros-args -p status_period_s:=0.5
  terminal 2:  python3 stt_mod/test/e2e_realmic.py --manifest data/manifest.jsonl --out e2e_out

Reported per condition: WER (all words), name-excluded WER (the leading
"hey"/robot-name span removed by alignment, because that name is out of the
model's vocabulary), empty-transcript rate, end-of-speech -> final lag
(wall clock, includes decoding), and the node's RTF.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import wave

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

FS = 16000


def read_wav_16k(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dt = {2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dt).astype(np.float32) / float(np.iinfo(dt).max + 1)
    x = x.reshape(-1, ch).mean(axis=1)
    if sr != FS:
        n = int(round(len(x) * FS / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def speech_end_s(x, win=480):
    n = len(x) // win
    r = np.sqrt((x[:n * win].reshape(n, win) ** 2).mean(1))
    thr = max(0.1 * r.max(), 3 * np.median(r))
    idx = np.nonzero(r > thr)[0]
    return (idx[-1] + 1) * win / FS if idx.size else len(x) / FS


def words(s):
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] if x == y else 1 + min(prev[j - 1], prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def strip_name(r, h, name):
    """Drop 'hey'/<name> from the reference and the best leading 0-3 hyp words."""
    rc = [w for w in r if w not in ("hey", name)]
    if len(rc) == len(r):
        return r, h
    lead = 1 + (r[:1] == ["hey"])
    k = min(range(0, lead + 3), key=lambda k: (lev(rc, h[k:]), k))
    return rc, h[k:]


class Player(Node):
    def __init__(self, a):
        super().__init__("stt_e2e_realmic")
        self.pub = self.create_publisher(Float32MultiArray, a.input_topic, 50)
        self.create_subscription(String, a.output_topic, self._final, 50)
        self.create_subscription(String, a.status_topic, self._status, 10)
        self.lock = threading.Lock()
        self.finals = []          # (wall_time, text)
        self.status = {}
        self.status_t = 0.0
        self.seq = 0
        self.t_start = time.time()

    def _final(self, msg):
        with self.lock:
            self.finals.append((time.time(), msg.data))

    def _status(self, msg):
        with self.lock:
            self.status = json.loads(msg.data)
            self.status_t = time.time()

    def publish_frame(self, x, t_capture):
        m = Float32MultiArray()
        m.layout.dim = [MultiArrayDimension(label="samples", size=len(x), stride=len(x))]
        m.layout.data_offset = 4
        m.data = [float(t_capture - self.t_start), float(FS), 1.0, float(self.seq)] + x.tolist()
        self.seq += 1
        self.pub.publish(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="e2e_out")
    ap.add_argument("--frame-s", type=float, default=1.0,
                    help="frame length (voice_mod publish_interval); 1.0 isolates STT lag")
    ap.add_argument("--gap-s", type=float, default=3.0, help="silence after each take")
    ap.add_argument("--quiet-s", type=float, default=2.5,
                    help="no new final for this long + empty backlog = take done")
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--name", default="artaban", help="out-of-vocabulary name stripped for name-excluded WER")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--input-topic", default="/dorai_clean_audio")
    ap.add_argument("--output-topic", default="/dorai_transcript")
    ap.add_argument("--status-topic", default="/dorai_stt/status")
    a = ap.parse_args()

    base = os.path.dirname(os.path.abspath(a.manifest))
    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    os.makedirs(a.out, exist_ok=True)

    rclpy.init()
    node = Player(a)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    print(f"waiting for stt_mod on {a.input_topic} ...", flush=True)
    t0 = time.time()
    while node.pub.get_subscription_count() == 0:
        time.sleep(0.2)
        if time.time() - t0 > 120:
            raise SystemExit("no subscriber on the input topic: is `ros2 run stt_mod stt` running?")
    time.sleep(1.0)

    fr = int(a.frame_s * FS)
    hyp_f = open(os.path.join(a.out, "hyps.jsonl"), "w")
    results = []
    for i, r in enumerate(rows):
        p = r["audio_filepath"]
        p = p if os.path.isabs(p) else os.path.join(base, p)
        x = read_wav_16k(p)
        sp_end = speech_end_s(x)
        y = np.concatenate([x, np.zeros(int(a.gap_s * FS), np.float32)])
        with node.lock:
            n0 = len(node.finals)
        t_play = time.time()
        for j in range(0, len(y), fr):
            blk = y[j:j + fr]
            t_cap = t_play + (j + len(blk)) / FS       # frame leaves the "mic" when complete
            time.sleep(max(0.0, t_cap - time.time()))
            node.publish_frame(blk, t_cap)
        t_speech_end = t_play + sp_end
        t_wait = time.time()
        while True:
            time.sleep(0.2)
            with node.lock:
                last = node.finals[-1][0] if len(node.finals) > n0 else t_wait
                fresh = (node.status_t > t_wait and node.status.get("backlog_s", 1) == 0
                         and not node.status.get("busy", False))
            if fresh and time.time() - max(last, t_wait) > a.quiet_s:
                break
            if time.time() - t_wait > a.timeout_s:
                print(f"  timeout on {p}", flush=True)
                break
        with node.lock:
            got = node.finals[n0:]
            rtf = node.status.get("rtf")
        hyp = " ".join(t for _, t in got).strip()
        lag = (got[0][0] - t_speech_end) if got else float("nan")
        rw, hw = words(r["text"]), words(hyp)
        rc, hc = strip_name(rw, hw, a.name)
        res = {"audio_filepath": r["audio_filepath"], "condition": r.get("condition", "all"),
               "ref": r["text"], "hyp": hyp, "n_finals": len(got),
               "n_ref": len(rw), "err": lev(rw, hw), "n_ref_cmd": len(rc), "err_cmd": lev(rc, hc),
               "empty": int(not hw), "lag_s": round(lag, 3), "node_rtf": rtf}
        results.append(res)
        hyp_f.write(json.dumps(res) + "\n"); hyp_f.flush()
        print(f"[{i + 1}/{len(rows)}] {res['condition']:16s} lag {lag:5.2f}s  "
              f"{r['text']!r} -> {hyp!r}", flush=True)

    by = {}
    for res in results:
        by.setdefault(res["condition"], []).append(res)
    by["ALL"] = results
    lines = ["| condition | takes | WER % | name-excluded WER % | empty % | lag median s | lag p90 s |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    summary = {}
    for c, rs in by.items():
        lags = np.array([q["lag_s"] for q in rs if q["lag_s"] == q["lag_s"]])
        s = {"takes": len(rs),
             "wer": 100 * sum(q["err"] for q in rs) / max(1, sum(q["n_ref"] for q in rs)),
             "wer_name_excluded": 100 * sum(q["err_cmd"] for q in rs) / max(1, sum(q["n_ref_cmd"] for q in rs)),
             "empty": 100 * np.mean([q["empty"] for q in rs]),
             "lag_median_s": float(np.median(lags)) if lags.size else None,
             "lag_p90_s": float(np.percentile(lags, 90)) if lags.size else None}
        summary[c] = s
        lines.append(f"| {c} | {s['takes']} | {s['wer']:.1f} | {s['wer_name_excluded']:.1f} | "
                     f"{s['empty']:.1f} | {s['lag_median_s'] or float('nan'):.2f} | "
                     f"{s['lag_p90_s'] or float('nan'):.2f} |")
    with node.lock:
        summary["node_status"] = node.status
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("node status:", json.dumps(summary["node_status"]))
    ex.shutdown(timeout_sec=2.0)
    spin.join(timeout=2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
