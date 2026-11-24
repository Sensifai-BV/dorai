#!/usr/bin/env python3
"""intent_mod — Stage 3 of the dorai speech pipeline (ROS 2 node ``intent_mod``).

Transcript text -> robot plan. A deterministic gate (wake.py) runs first; only
addressed, non-stop utterances reach the engine (llama.cpp GGUF with a schema
grammar, or the rule fallback).

Topics
  in   /dorai_transcript   std_msgs/String   final text from stt_mod
       /dorai_doa          std_msgs/Float32  bearing of the talker, degrees (bursts kept; a plan gets its utterance's)
  out  /dorai_stop         std_msgs/String   'stop' | 'estop'. Published from the subscription
                                             callback itself, never queued behind the engine.
       /dorai_plan         std_msgs/String   JSON {"text","plan","bearing_deg","engine",
                                             "latency_s","stamp","confidence","min_token_prob",
                                             "wake_edits"}; valid actionable plans only
       /dorai_say          std_msgs/String   spoken question / refusal for tts_mod (clarify,
                                             invalid engine output, engine failure)
       /dorai_intent/status std_msgs/String  JSON {engine, model, n_plans, n_clarify, n_stop,
                                             mean_latency_s, queue_depth, ...} every status_period_s
       /dorai_intent/decision std_msgs/String JSON, one per transcript: what was heard, the
                                             decision (stop | estop | ignored | wake_only | plan |
                                             clarify | none | invalid | guarded | error) and its
                                             evidence (see "Decisions" below). With debug:=true the
                                             same is logged, one line per transcript.

Decisions. Every transcript ends in exactly one decision, published on
/dorai_intent/decision and, with debug:=true, logged:
  wake        what stood in the place of the wake word ("arthaban"), its edit distance to the
              wake word and wake_confidence = 1 - edits / len(wake word)
  command     the text handed to the engine (after the wake word)
  engine      llm | rules, and its latency
  confidence  LLM only: geometric mean of the probabilities the model gave the tokens of the
              plan it wrote (1.0 = no alternative at any token); min_token_prob is its least
              certain token. None for the rule engine (a rule matches or not) and for stop,
              which is a word match. This is the engine's certainty about its own output for
              the text it was given; it says nothing about whether the transcript was right.
  tasks       the plan's step names

Flow: stop check -> wake gate (require_wake) -> queue -> worker thread (engine) ->
schema.validate -> plan | say | drop. The engine runs on one worker thread so the ROS
callbacks never block.
"""
from __future__ import annotations

import collections
import json
import os
import threading
import time

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node

try:  # raised by spin() when SIGTERM shuts the context down
    from rclpy.executors import ExternalShutdownException
except ImportError:  # very old rclpy / test mocks
    class ExternalShutdownException(Exception):
        pass
from std_msgs.msg import Float32, String

from . import jsonsafe, wake
from .bearing import BearingHistory
from .engine import make_engine
from .models import DEFAULT_LORA, DEFAULT_MODEL, HF_REPO, HF_SUBDIR
from .guard import check as guard_check
from .schema import load_world, validate

# name -> (default, description). Numbers accept int or double on the CLI.
PARAMS = {
    "input_topic": ("/dorai_transcript", "final transcripts from stt_mod"),
    "doa_topic": ("/dorai_doa", "talker bearing, degrees (Float32)"),
    "stop_topic": ("/dorai_stop", "'stop' | 'estop' (to twist_mux)"),
    "plan_topic": ("/dorai_plan", "JSON plans"),
    "say_topic": ("/dorai_say", "text to speak (tts_mod)"),
    "status_topic": ("/dorai_intent/status", "JSON status"),
    "decision_topic": ("/dorai_intent/decision", "JSON, one per transcript: decision and its evidence"),
    "locations_file": ("", "world file; '' = package share config/locations.yaml"),
    "engine": ("auto", "auto | llm | rules"),
    "model": (DEFAULT_MODEL, "GGUF file name, absolute path, or huggingface.co file URL"),
    "models_dir": ("", "extra models root searched first; downloads go here ('' = ~/dorai/models)"),
    "hf_repo": (HF_REPO, "Hugging Face repo to download `model` from when no root has it"),
    "hf_subdir": (HF_SUBDIR, "folder of that repo the model file is in ('' = the repo's top level)"),
    "auto_download": (True, "download a missing model from Hugging Face at startup ($HF_TOKEN for a private repo)"),
    "num_threads": (2, "llama.cpp threads"),
    "n_ctx": (2048, "llama.cpp context size"),
    "require_wake": (True, "only act on utterances addressed to the wake word (stop always works)"),
    "wake_word": ("artaban", "robot name"),
    "wake_max_edits": (2, "max Levenshtein edits when matching the wake word"),
    "stop_wake_max_edits": (4, "looser wake-word match in front of a stop (a missed stop is the worse error)"),
    "doa_burst_gap_s": (0.6, "silence between /dorai_doa messages that ends one talker's burst"),
    "doa_max_age_s": (20.0, "a finished burst older than this is not attached to a plan"),
    "lora": (DEFAULT_LORA, "optional GGUF LoRA adapter file name (normally empty: the default model has it merged)"),
    "use_grammar": (False, "JSON-schema grammar during decoding (off: faster on ARM; plans are validated anyway)"),
    "grounding_guard": (True, "ask 'did you mean X?' if the plan names a place/object the speaker did not say"),
    "status_period_s": (5.0, "status topic period"),
    "debug": (False, "verbose logging: one line per transcript with the decision, wake-word match, "
                     "engine, confidence, latency and tasks"),
}

QUESTIONS = {
    "unknown_target": "I don't know that place. Where do you want me to go?",
    "unknown_object": "I don't know that object. Which one do you mean?",
    "unclear": "Sorry, I didn't understand. Could you say that again?",
}
INVALID_QUESTION = "Sorry, I couldn't make a safe plan from that. Could you say it again?"


def default_locations_file():
    """Source-tree config first (dev / symlink-install), then the installed share dir."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.normpath(os.path.join(here, "..", "config", "locations.yaml"))]
    try:
        from ament_index_python.packages import get_package_share_directory
        cands.append(os.path.join(get_package_share_directory("intent_mod"), "config", "locations.yaml"))
    except Exception:  # noqa: BLE001
        pass
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError("locations.yaml not found; set the locations_file parameter")


class IntentNode(Node):
    def __init__(self):
        super().__init__("intent_mod")
        for k, (v, desc) in PARAMS.items():
            numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
            self.declare_parameter(k, v, ParameterDescriptor(description=desc, dynamic_typing=numeric))
        p = {k: self.get_parameter(k).value for k in PARAMS}
        self.p = p
        log = self.get_logger()

        self.locations_file = p["locations_file"] or default_locations_file()
        self.locs, self.objs = load_world(self.locations_file)
        t0 = time.time()
        self.engine = make_engine({
            "locs": self.locs, "objs": self.objs, "engine": p["engine"], "model": p["model"],
            "models_dir": p["models_dir"], "hf_repo": p["hf_repo"], "hf_subdir": p["hf_subdir"], "auto_download": bool(p["auto_download"]), "num_threads": int(p["num_threads"]), "lora": p.get("lora", ""), "use_grammar": p.get("use_grammar", False),
            "n_ctx": int(p["n_ctx"])}, log=log)

        self.pub_stop = self.create_publisher(String, p["stop_topic"], 10)
        self.pub_plan = self.create_publisher(String, p["plan_topic"], 10)
        self.pub_say = self.create_publisher(String, p["say_topic"], 10)
        self.pub_status = self.create_publisher(String, p["status_topic"], 10)
        self.pub_decision = self.create_publisher(String, p["decision_topic"], 20)

        self.bearings = BearingHistory(float(p["doa_burst_gap_s"]), float(p["doa_max_age_s"]))
        self.stats = collections.Counter()
        self._lat = collections.deque(maxlen=50)
        self._q = collections.deque()
        self._cv = threading.Condition()
        self._shutdown = False
        self._busy = False
        self._worker = threading.Thread(target=self._run, name="intent_worker", daemon=True)
        self._worker.start()

        self.create_subscription(Float32, p["doa_topic"], self._on_doa, 10)
        self.create_subscription(String, p["input_topic"], self._on_transcript, 20)
        self.create_timer(float(p["status_period_s"]), self._status)
        log.info(f"intent_mod ready in {time.time() - t0:.1f}s - engine={self.engine.name} "
                 f"({self.engine.model}), world={self.locations_file} "
                 f"[{len(self.locs)} places, {len(self.objs)} objects], "
                 f"require_wake={bool(p['require_wake'])}; {p['input_topic']} -> {p['plan_topic']}")

    # ------------------------------------------------------------ ROS input
    def _on_doa(self, msg):
        self.bearings.add(time.monotonic(), msg.data)

    def _on_transcript(self, msg):
        """Runs on the ROS callback thread: stop path first, everything else is queued."""
        text = (msg.data or "").strip()
        if not text:
            return
        p = self.p
        stop = wake.is_stop(text, p["wake_word"], int(p["wake_max_edits"]), int(p["stop_wake_max_edits"]))
        if stop:
            self.pub_stop.publish(String(data=stop))          # immediate, never behind the engine
            self.stats["n_stop"] += 1
            self.get_logger().warning(f"STOP ({stop}) from '{text}'")
            self._decide(stop, text, wake.match_wake(text, p["wake_word"], int(p["wake_max_edits"])))
            return
        m = wake.match_wake(text, p["wake_word"], int(p["wake_max_edits"]))
        if m["addressed"]:
            if not m["rest"]:                                 # wake word alone
                self.stats["n_wake_only"] += 1
                self._decide("wake_only", text, m)
                return
            cmd = m["rest"]
        elif p["require_wake"]:
            self.stats["n_ignored"] += 1
            self._decide("ignored", text, m)
            return
        else:
            cmd = wake.normalise(text)
        with self._cv:
            # the bearing of THIS utterance: the last finished speech burst, not the
            # latest message, which may already be the next talker (see bearing.py)
            bearing = self.bearings.for_utterance(time.monotonic())
            self._q.append((text, cmd, bearing, time.time(), m))
            self._cv.notify()

    def _decide(self, decision, text, m, **more):
        """Publish (and with debug log) what was done with one transcript and why."""
        d = {"text": text, "decision": decision, "addressed": bool(m["addressed"]),
             "wake_heard": m["heard"], "wake_edits": m["edits"], "wake_confidence": m["confidence"],
             "command": more.pop("command", m["rest"] if m["addressed"] else None),
             "engine": None, "confidence": None, "min_token_prob": None, "latency_s": None,
             "tasks": [], "stamp": time.time()}
        d.update(more)
        self.pub_decision.publish(String(data=jsonsafe.dumps(d)))
        if self.p["debug"]:
            wk = (f"wake '{m['heard']}' {m['edits']} edit(s), conf {m['confidence']:.2f}" if m["edits"] is not None
                  else "no wake-word candidate")
            eng = ""
            if d["engine"]:
                conf = "n/a" if d["confidence"] is None else f"{d['confidence']:.2f} (min token {d['min_token_prob']:.2f})"
                took = "" if d["latency_s"] is None else f" {d['latency_s']:.2f}s"
                eng = (f" | engine {d['engine']}{took}, confidence {conf}"
                       f" | tasks {d['tasks']}" + (f" | {d['why']}" if d.get("why") else ""))
            self.get_logger().info(f"decision {decision.upper()}: '{text}' | {wk}"
                                   + (f" | command '{d['command']}'" if d["command"] else "") + eng)

    # ----------------------------------------------------------- worker
    def _run(self):
        while True:
            with self._cv:
                while not self._q and not self._shutdown:
                    self._cv.wait()
                if self._shutdown and not self._q:
                    return
                item = self._q.popleft()
                self._busy = True
            try:
                self._process(*item)
            except Exception as e:  # noqa: BLE001  never kill the worker
                self.stats["n_error"] += 1
                self.get_logger().error(f"intent worker error on '{item[0]}': {e!r}")
                self._say(INVALID_QUESTION)
                try:
                    self._decide("error", item[0], item[4], command=item[1], engine=self.engine.name,
                                 why=repr(e)[:200])
                except Exception:  # noqa: BLE001  reporting must not kill the worker either
                    pass
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def _process(self, text, cmd, bearing, t_recv, m=None):
        m = m or wake.match_wake(text, self.p["wake_word"], int(self.p["wake_max_edits"]))
        plan, lat = self.engine.parse(cmd)
        self._lat.append(lat)
        last = getattr(self.engine, "last", None) or {}
        ev = {"command": cmd, "engine": self.engine.name, "latency_s": round(lat, 4),
              "confidence": last.get("confidence"), "min_token_prob": last.get("min_token_prob")}
        tasks = lambda pl: [str(s.get("task")) for s in pl.get("steps", []) if isinstance(s, dict)]  # noqa: E731
        try:
            problems = validate(plan, self.locs, self.objs)
        except Exception as e:  # noqa: BLE001  malformed engine output
            problems = [f"malformed plan: {e!r}"]
        if problems:
            self.stats["n_invalid"] += 1
            self.get_logger().warning(f"invalid plan for '{text}': {problems}")
            self._say(INVALID_QUESTION)
            self._decide("invalid", text, m, why="; ".join(map(str, problems))[:200], **ev)
            return
        if self.p.get("grounding_guard", True) and self.engine.name != "rules":
            plan, why = guard_check(cmd, plan, self.locs, self.objs)
            if why:
                self.stats["n_guarded"] = self.stats.get("n_guarded", 0) + 1
                self.get_logger().info(f"grounding guard ({why}) for '{text}'")
                self._say(plan["confirm"]["say"])
                self._decide("guarded", text, m, why=str(why), tasks=tasks(plan), **ev)
                return
        first = plan["steps"][0]
        if first["task"] == "clarify":
            self.stats["n_clarify"] += 1
            self._say(QUESTIONS.get(first.get("reason"), QUESTIONS["unclear"]))
            self._decide("clarify", text, m, why=str(first.get("reason")), tasks=tasks(plan), **ev)
            return
        if first["task"] == "none":
            self.stats["n_none"] += 1
            self._decide("none", text, m, tasks=tasks(plan), **ev)
            return
        self.stats["n_plans"] += 1
        # TODO(task_mod): replace this JSON topic by an ExecuteTask action goal once the
        # task_mod action interface exists; keep /dorai_plan as the debug/log interface.
        self.pub_plan.publish(String(data=jsonsafe.dumps({
            "text": text, "plan": plan, "bearing_deg": bearing, "engine": self.engine.name,
            "latency_s": round(lat, 4), "stamp": t_recv, "confidence": ev["confidence"],
            "min_token_prob": ev["min_token_prob"], "wake_edits": m["edits"] if m["addressed"] else None})))
        self._decide("plan", text, m, tasks=tasks(plan), **ev)

    def _say(self, text):
        self.pub_say.publish(String(data=text))

    # ----------------------------------------------------------- utilities
    def wait_idle(self, timeout=5.0):
        """Block until the queue is empty and the worker idle (tests / shutdown)."""
        end = time.time() + timeout
        with self._cv:
            while self._q or self._busy:
                left = end - time.time()
                if left <= 0:
                    return False
                self._cv.wait(left)
        return True

    def _status(self):
        lat = list(self._lat)
        s = self.stats
        self.pub_status.publish(String(data=jsonsafe.dumps({
            "engine": self.engine.name, "model": self.engine.model,
            "n_plans": s["n_plans"], "n_clarify": s["n_clarify"], "n_none": s["n_none"],
            "n_stop": s["n_stop"], "n_invalid": s["n_invalid"], "n_ignored": s["n_ignored"],
            "n_error": s["n_error"],
            "mean_latency_s": round(sum(lat) / len(lat), 4) if lat else None,
            "queue_depth": len(self._q) + (1 if self._busy else 0),
            "require_wake": bool(self.p["require_wake"]), "bearing_deg": self.bearings.latest,
            "stamp": time.time()})))

    def shutdown(self):
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()
        self._worker.join(timeout=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = IntentNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
