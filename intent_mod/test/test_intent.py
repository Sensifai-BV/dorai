"""intent_mod tests. They need NO ROS and NO llama_cpp (rosmock fakes rclpy/std_msgs).

  cd dorai && python -m pytest intent_mod/test -q

One optional test runs the real LlamaIntentEngine; it is skipped unless the default GGUF is
installed in a models root (or $DORAI_TEST_GGUF). Hugging Face downloads are disabled here
(autouse fixture); the download tests fake the HTTP layer.
"""
import json
import os
import sys
import threading

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import rosmock  # noqa: E402

rosmock.install()

from intent_mod import engine as eng  # noqa: E402
from intent_mod import models, node as node_mod, wake  # noqa: E402
from intent_mod.schema import load_world, validate  # noqa: E402

REAL_HF_DOWNLOAD = models.hf_download
WORLD = os.path.join(HERE, "..", "config", "locations.yaml")
LOCS, OBJS = load_world(WORLD)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """A missing model must never start a real 2.6 GB download from a unit test."""
    def refuse(*a, **k):
        raise IOError("network disabled in tests")
    monkeypatch.setattr(models, "hf_download", refuse)


# ------------------------------------------------------------------ helpers
class FakeEngine:
    name, model = "fake", "fake.gguf"

    def __init__(self, plan=None, gate=None):
        self.plan, self.gate, self.calls = plan, gate, []

    def reload_world(self, locs, objs): pass

    def parse(self, text):
        self.calls.append(text)
        if self.gate is not None:
            self.gate.wait(5)
        return self.plan, 0.01


@pytest.fixture
def make_node():
    nodes = []

    def _make(**over):
        rosmock.Node.overrides = {"engine": "rules", "locations_file": WORLD, **over}
        n = node_mod.IntentNode()
        nodes.append(n)
        return n
    yield _make
    for n in nodes:
        n.shutdown()
    rosmock.Node.overrides = {}


def say(n, text):
    n.subs[n.p["input_topic"]](rosmock.String(text))


def msgs(n, key):
    return n.pubs[n.p[key]].msgs


# ------------------------------------------------------------- wake / stop
@pytest.mark.parametrize("text,exp", [
    ("stop", "stop"), ("Stop!", "stop"), ("halt", "stop"), ("freeze", "stop"),
    ("stop stop", "stop"), ("stop stop stop", "stop"), ("emergency stop", "estop"),
    ("Emergency stop!", "estop"), ("e stop", "estop"), ("artaban stop", "stop"),
    ("hey artaban emergency stop", "estop"), ("stop please", "stop"), ("stop moving", "stop"),
    ("stob", "stop"), ("stopp", "stop"), ("halt now", "stop"),
    ("Arduban stopped", "stop"), ("Artaban stopped.", "stop"),            # ASR spellings of "Artaban stop."
])
def test_is_stop_positive(text, exp):
    assert wake.is_stop(text) == exp


@pytest.mark.parametrize("text", [
    "stop welding", "artaban stop welding", "start welding", "don't stop", "do not stop",
    "never stop", "go to welder a", "step forward", "the weather is nice", "", "   ",
    "stop the gripper", "I stopped by the shop",
])
def test_is_stop_negative(text):
    assert wake.is_stop(text) is None


@pytest.mark.parametrize("text,rest", [
    ("artaban come here", "come here"), ("hey artaban come here", "come here"),
    ("artiban come here", "come here"), ("art a ban come here", "come here"),
    ("arta ban come here", "come here"), ("Arthaban, follow me", "follow me"),
    ("OK Artaban turn left.", "turn left"), ("artaban", ""),
])
def test_strip_wake_addressed(text, rest):
    assert wake.strip_wake(text) == (True, rest)


@pytest.mark.parametrize("text", ["come here", "go to welder a", "hey there", "urban renewal today",
                                  "the tartan is here", ""])
def test_strip_wake_not_addressed(text):
    assert wake.strip_wake(text)[0] is False


def test_strip_wake_respects_max_edits():
    assert wake.strip_wake("artiban come", max_edits=0)[0] is False
    assert wake.strip_wake("artiban come", max_edits=1)[0] is True


def test_levenshtein():
    assert wake.lev("kitten", "sitting") == 3 and wake.lev("", "abc") == 3


# ---------------------------------------------------------------- engines
def test_rule_engine_inspect_meter():
    e = eng.RuleIntentEngine(LOCS, OBJS)
    plan, lat = e.parse("go to welding device x for check meter")
    assert plan == {"steps": [{"task": "inspect_meter", "target": "welder_x"}]}
    assert lat >= 0 and validate(plan, LOCS, OBJS) == []


def test_rule_engine_unknown_place_is_clarify():
    plan, _ = eng.RuleIntentEngine(LOCS, OBJS).parse("go to the paint shop")
    assert plan["steps"][0]["task"] == "clarify" and plan["steps"][0]["reason"] == "unknown_target"


# the 10 basic Artaban commands of the partner test sheet, plus the STT spellings seen for them
@pytest.mark.parametrize("text,step", [
    ("come here", {"task": "come_here"}),
    ("go to welding station two", {"task": "goto", "target": "welder_2"}),
    ("go to welding station 2", {"task": "goto", "target": "welder_2"}),       # digit, was welder_a
    ("go to wedding station two", {"task": "goto", "target": "welder_2"}),
    ("turn left", {"task": "turn", "direction": "left"}),
    ("go back to base", {"task": "goto", "target": "base"}),
    ("go back to the base", {"task": "goto", "target": "base"}),               # was go_back
    ("check the meter", {"task": "read_meter"}),
    ("follow me", {"task": "follow_me"}),
    ("sit down", {"task": "posture", "pose": "sit"}),
    ("look at the welding torch", {"task": "look_at", "object": "welding_torch"}),
    ("start inspection", {"task": "inspection", "state": "start"}),
    ("move 1.5 meters forward", {"task": "move", "direction": "forward", "meters": 1.5}),
])
def test_rule_engine_basic_commands(text, step):
    plan, _ = eng.RuleIntentEngine(LOCS, OBJS).parse(text)
    assert plan == {"steps": [step]} and validate(plan, LOCS, OBJS) == []


def test_validate_rejects_out_of_enum_values():
    # Seen live (E6, welding 0 dB): "Both welding state reduced" -> weld state "reduce" was published.
    assert validate({"steps": [{"task": "weld", "state": "reduce"}]}, LOCS, OBJS)
    assert validate({"steps": [{"task": "turn", "direction": "up"}]}, LOCS, OBJS)
    assert validate({"steps": [{"task": "weld", "state": "stop"}]}, LOCS, OBJS) == []


def test_make_engine_falls_back_to_rules_when_gguf_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DORAI_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(models, "default_roots", lambda extra="": [str(tmp_path)])
    log = rosmock._Logger()
    e = eng.make_engine({"locs": LOCS, "objs": OBJS, "engine": "auto", "models_dir": str(tmp_path)}, log=log)
    assert isinstance(e, eng.RuleIntentEngine)
    assert any(l == "warning" and "not found" in m for l, m in log.lines)
    with pytest.raises(FileNotFoundError):
        eng.make_engine({"locs": LOCS, "objs": OBJS, "engine": "llm", "models_dir": str(tmp_path)}, log=log)
    assert isinstance(eng.make_engine({"locs": LOCS, "objs": OBJS, "engine": "rules"}), eng.RuleIntentEngine)


def test_make_engine_falls_back_when_llama_fails(tmp_path, monkeypatch):
    g = tmp_path / models.DEFAULT_MODEL
    g.write_bytes(b"not a gguf")
    monkeypatch.setattr(models, "default_roots", lambda extra="": [str(tmp_path)])
    e = eng.make_engine({"locs": LOCS, "objs": OBJS, "engine": "auto"}, log=rosmock._Logger())
    assert isinstance(e, eng.RuleIntentEngine)


# ------------------------------------------------------------------ models
def test_find_model_root_order(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    for r in (a, b):
        r.mkdir()
        (r / "m.gguf").write_text("x")
    os.utime(a / "m.gguf", (0, 0))                       # older, but first root still wins
    assert models.find_model("m.gguf", roots=[str(a), str(b)]) == str(a / "m.gguf")
    assert models.find_model("m.gguf", roots=[str(b), str(a)]) == str(b / "m.gguf")
    assert models.find_model("none.gguf", roots=[str(a), str(b)]) is None


def test_find_model_subdir_and_absolute(tmp_path):
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "m.gguf"
    f.write_text("x")
    assert models.find_model("m.gguf", roots=[str(tmp_path)]) == str(f)
    assert models.find_model(str(f)) == str(f)
    assert models.find_model(str(tmp_path / "nope.gguf")) is None


def test_default_roots_order(monkeypatch, tmp_path):
    monkeypatch.setenv("DORAI_MODELS_DIR", "/env/models")
    r = models.default_roots("/param/models")
    assert r[0] == "/param/models" and r[1] == "/env/models"
    assert os.path.normpath(r[2]).endswith(os.path.join("intent_mod", "models"))
    assert r[-2].endswith(os.path.join("dorai", "models")) and r[-1] == "/root/.cache/dorai/models"


# -------------------------------------------------------------------- node
def test_node_plan_inspect_meter(make_node):
    n = make_node()
    n.subs["/dorai_doa"](rosmock.Float32(42.5))
    say(n, "Artaban, go to welding device X for check meter.")
    assert n.wait_idle()
    out = json.loads(msgs(n, "plan_topic")[0])
    assert out["plan"] == {"steps": [{"task": "inspect_meter", "target": "welder_x"}]}
    assert out["bearing_deg"] == 42.5 and out["engine"] == "rules"
    assert {"text", "plan", "bearing_deg", "engine", "latency_s", "stamp"} <= set(out)
    assert msgs(n, "stop_topic") == [] and msgs(n, "say_topic") == []


def test_node_unknown_place_goes_to_say_not_plan(make_node):
    n = make_node()
    say(n, "artaban go to the paint shop")
    assert n.wait_idle()
    assert msgs(n, "plan_topic") == []
    assert len(msgs(n, "say_topic")) == 1 and "place" in msgs(n, "say_topic")[0]


def test_node_chatter_publishes_nothing(make_node):
    n = make_node(require_wake=False)
    say(n, "the weather is really nice today")
    assert n.wait_idle()
    assert msgs(n, "plan_topic") == [] and msgs(n, "say_topic") == [] and msgs(n, "stop_topic") == []


def test_node_ignores_unaddressed_when_wake_required(make_node):
    n = make_node()
    say(n, "go to welder a")
    assert n.wait_idle()
    assert msgs(n, "plan_topic") == [] and n.stats["n_ignored"] == 1


def test_node_require_wake_false_acts_without_wake(make_node):
    n = make_node(require_wake=False)
    say(n, "go to welder a")
    assert n.wait_idle()
    assert json.loads(msgs(n, "plan_topic")[0])["plan"] == {"steps": [{"task": "goto", "target": "welder_a"}]}


def test_node_stop_works_without_wake_and_not_queued(make_node):
    n = make_node()
    say(n, "stop")
    say(n, "artaban emergency stop")
    assert msgs(n, "stop_topic") == ["stop", "estop"]      # published synchronously, no wait_idle
    assert n.engine.name == "rules" and len(n._q) == 0


def test_node_stop_published_while_worker_blocked(make_node):
    n = make_node()
    gate = threading.Event()
    n.engine = FakeEngine({"steps": [{"task": "goto", "target": "welder_a"}]}, gate=gate)
    say(n, "artaban go to welder a")
    for _ in range(200):                                    # wait until the worker is inside the engine
        if n.engine.calls:
            break
        threading.Event().wait(0.01)
    assert n.engine.calls == ["go to welder a"]
    say(n, "stop")                                          # worker is stuck: stop must still go out
    assert msgs(n, "stop_topic") == ["stop"] and msgs(n, "plan_topic") == []
    say(n, "artaban stop welding")                          # a weld task is queued, not a stop
    assert msgs(n, "stop_topic") == ["stop"] and len(n._q) == 1
    gate.set()
    assert n.wait_idle()
    assert len(msgs(n, "plan_topic")) == 2
    assert n.stats["n_stop"] == 1


def test_node_invalid_llm_output_becomes_clarify(make_node):
    n = make_node()
    n.engine = FakeEngine({"steps": [{"task": "goto", "target": "narnia"}]})
    say(n, "artaban go to narnia")
    assert n.wait_idle()
    assert msgs(n, "plan_topic") == [] and len(msgs(n, "say_topic")) == 1
    assert n.stats["n_invalid"] == 1


def test_node_engine_exception_does_not_kill_worker(make_node):
    n = make_node()

    class Boom(FakeEngine):
        def parse(self, text):
            raise RuntimeError("boom")
    n.engine = Boom()
    say(n, "artaban go to welder a")
    assert n.wait_idle() and len(msgs(n, "say_topic")) == 1 and n.stats["n_error"] == 1
    n.engine = FakeEngine({"steps": [{"task": "wait"}]})
    say(n, "artaban wait")
    assert n.wait_idle() and len(msgs(n, "plan_topic")) == 1


def test_node_llm_clarify_and_none(make_node):
    n = make_node()
    n.engine = FakeEngine({"steps": [{"task": "clarify", "reason": "unclear"}]})
    say(n, "artaban do that thing")
    n.wait_idle()
    n.engine = FakeEngine({"steps": [{"task": "none"}]})
    say(n, "artaban nice weather")
    n.wait_idle()
    assert msgs(n, "plan_topic") == [] and len(msgs(n, "say_topic")) == 1 and n.stats["n_none"] == 1


def test_node_status_json_fields(make_node):
    n = make_node()
    say(n, "artaban go to welder a")
    say(n, "artaban go to the paint shop")
    say(n, "stop")
    n.wait_idle()
    n._status()
    s = json.loads(msgs(n, "status_topic")[-1])
    for k in ("engine", "model", "n_plans", "n_clarify", "n_stop", "mean_latency_s", "queue_depth"):
        assert k in s
    assert (s["engine"], s["n_plans"], s["n_clarify"], s["n_stop"], s["queue_depth"]) == ("rules", 1, 1, 1, 0)
    assert s["mean_latency_s"] is not None and s["mean_latency_s"] >= 0


def test_node_falls_back_to_rules_without_model(make_node, tmp_path, monkeypatch):
    monkeypatch.setattr(models, "default_roots", lambda extra="": [str(tmp_path)])
    n = make_node(engine="auto")
    assert n.engine.name == "rules"
    assert any(l == "warning" for l, _ in n.get_logger().lines)


# --------------------------------------------------------- optional real LLM
def _gguf():
    env = os.environ.get("DORAI_TEST_GGUF")
    if env and os.path.isfile(env):
        return env
    return models.find_model(models.DEFAULT_MODEL)                    # deployed: merged LoRA, Q4_0


@pytest.mark.skipif(_gguf() is None, reason="intent-qwen35-4b-Q4_0.gguf not installed (see models.default_roots)")
@pytest.mark.parametrize("use_grammar", [False, True])
def test_real_llama_engine_three_utterances(use_grammar):
    pytest.importorskip("llama_cpp")
    e = eng.LlamaIntentEngine(_gguf(), LOCS, OBJS, n_threads=4, n_ctx=2048, use_grammar=use_grammar)
    for text in ("go to welding device x for check meter", "come here", "bring me the hammer"):
        plan, lat = e.parse(text)
        assert validate(plan, LOCS, OBJS) == [] or plan["steps"][0]["task"] in ("clarify", "none"), plan
        assert lat > 0
        print(f"{text!r} -> {json.dumps(plan)} ({lat:.2f}s)")


# ------------------------------------------------------------------ grounding guard
def _world():
    from intent_mod.schema import load_world
    return load_world(os.path.join(os.path.dirname(__file__), "..", "config", "locations.yaml"))


def test_guard_passes_spoken_target():
    from intent_mod.guard import check
    locs, objs = _world()
    plan = {"steps": [{"task": "inspect_meter", "target": "welder_x"}]}
    out, why = check("go to welding device x for check meter", plan, locs, objs)
    assert why is None and out == plan


def test_guard_passes_stt_error():
    from intent_mod.guard import check
    locs, objs = _world()
    plan = {"steps": [{"task": "inspect_meter", "target": "welder_x"}]}
    assert check("go to wedding device x for check meeter", plan, locs, objs)[1] is None


def test_guard_blocks_unsaid_target():
    from intent_mod.guard import check
    locs, objs = _world()
    out, why = check("go read the meter on the furnace", {"steps": [{"task": "inspect_meter", "target": "compressor"}]},
                     locs, objs)
    assert why == "ungrounded_target" and out["steps"][0]["task"] == "clarify"
    assert "did you mean" in out["confirm"]["say"]


def test_guard_blocks_unsaid_object():
    from intent_mod.guard import check
    locs, objs = _world()
    out, why = check("bring me the ladder", {"steps": [{"task": "bring", "object": "hammer"}]}, locs, objs)
    assert why == "ungrounded_object"


def test_guard_allows_dock_and_home():
    from intent_mod.guard import check
    locs, objs = _world()
    assert check("go home", {"steps": [{"task": "goto", "target": "base"}]}, locs, objs)[1] is None
    assert check("go charge", {"steps": [{"task": "dock"}]}, locs, objs)[1] is None


# ------------------------------------------------------------ HF download
class _Resp:
    """Minimal urllib response: honours a Range request header like the HF CDN."""
    def __init__(self, data, req):
        rng = req.headers.get("Range")
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        self.status, self._b = (206 if rng else 200), data[start:]
        self.headers = {"Content-Length": str(len(self._b))}

    def read(self, n):
        out, self._b = self._b[:n], self._b[n:]
        return out

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_hub(monkeypatch, data, sha=None):
    import hashlib
    monkeypatch.setattr(models, "_hf_sha256", lambda url, hdr: sha or hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda req, timeout=0: _Resp(data, req))


def test_hf_download_resumes_and_verifies(tmp_path, monkeypatch):
    data = os.urandom(3 * 1024 * 1024 + 17)
    _fake_hub(monkeypatch, data)
    (tmp_path / "m.gguf.part").write_bytes(data[:1_000_000])          # interrupted earlier
    log = rosmock._Logger()
    p = REAL_HF_DOWNLOAD("org/repo", "m.gguf", str(tmp_path), log=log, chunk=1 << 20)
    assert open(p, "rb").read() == data and not (tmp_path / "m.gguf.part").exists()
    assert any("resuming" in m for _, m in log.lines)


def test_hf_download_rejects_bad_checksum(tmp_path, monkeypatch):
    _fake_hub(monkeypatch, b"corrupted", sha="0" * 64)
    with pytest.raises(IOError, match="sha256 mismatch"):
        REAL_HF_DOWNLOAD("org/repo", "m.gguf", str(tmp_path), log=rosmock._Logger())
    assert list(tmp_path.iterdir()) == []                              # no half file left behind


def test_resolve_model_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "default_roots", lambda extra="": [str(tmp_path)])
    calls = []

    def fake(repo, fname, dest, rev="main", log=None):
        calls.append((repo, fname, dest, rev))
        return os.path.join(dest, os.path.basename(fname))
    monkeypatch.setattr(models, "hf_download", fake)
    d = str(tmp_path / "dl")
    # bare name, not installed -> hf_repo/name into models_dir
    assert models.resolve_model("x.gguf", d, hf_repo="sensifai/dorai-intent-model") == os.path.join(d, "x.gguf")
    # huggingface.co URL (blob or resolve) -> that repo, revision and file
    url = "https://huggingface.co/sensifai/dorai-intent-model/blob/main/intent-qwen35-4b-Q4_0.gguf"
    models.resolve_model(url, d)
    assert calls == [("sensifai/dorai-intent-model", "x.gguf", d, "main"),
                     ("sensifai/dorai-intent-model", "intent-qwen35-4b-Q4_0.gguf", d, "main")]
    # an explicit path that is missing is an error, never a download; download=False stays offline
    assert models.resolve_model(str(tmp_path / "nope.gguf"), d) is None
    assert models.resolve_model("y.gguf", d, download=False) is None
    assert len(calls) == 2
    # already installed -> no download
    (tmp_path / "z.gguf").write_bytes(b"x")
    assert models.resolve_model("z.gguf", d) == str(tmp_path / "z.gguf") and len(calls) == 2
