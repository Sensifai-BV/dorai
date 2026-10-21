"""Intent engines: text -> plan. Both share one interface:

    engine.parse(text) -> (plan_dict, latency_s)
    engine.reload_world(locs, objs)
    engine.name / engine.model        (strings, for the status topic)

``LlamaIntentEngine`` runs a GGUF through llama.cpp with a JSON-schema grammar built
from locations.yaml (so only existing places/objects can be named); ``RuleIntentEngine``
is the deterministic fallback. llama_cpp is imported lazily, so this module imports
(and the rules engine works) without it.
"""
from __future__ import annotations

import json
import logging
import threading
import time

from .models import DEFAULT_MODEL, find_model
from .rules import RuleParser
from .schema import plan_schema, system_prompt

BAD_PLAN = {"steps": []}        # fails schema.validate() -> node asks the user to repeat


class RuleIntentEngine:
    name = "rules"
    model = "rules"

    def __init__(self, locs, objs):
        self.reload_world(locs, objs)

    def reload_world(self, locs, objs):
        self.locs, self.objs = locs, objs
        self.parser = RuleParser(locs, objs)

    def parse(self, text):
        t0 = time.perf_counter()
        plan = self.parser.parse(text)
        return plan, time.perf_counter() - t0


_SENT = "\u2063UTTERANCE\u2063"


def split_template(llm, system):
    """Render the GGUF's own chat template around a sentinel -> (prefix, suffix) text.
    Works for LFM2/LFM2.5, Gemma 4 and Qwen3.5 templates; reasoning templates get an
    empty think block (no-think mode)."""
    import re

    import jinja2
    tpl = llm.metadata.get("tokenizer.chat_template") or ""
    tpl = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", tpl)   # HF training-only tags
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    env.globals["strftime_now"] = lambda f: time.strftime(f)
    out = env.from_string(tpl).render(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": _SENT}],
        add_generation_prompt=True, bos_token="", eos_token="", enable_thinking=False)
    pre, post = out.split(_SENT)
    if post.rstrip().endswith("<think>"):
        post = post.rstrip() + "</think>\n"
    return pre, post


class LlamaIntentEngine:
    """Grammar-constrained llama.cpp engine.

    The system prompt (task list + places + objects) is prefilled ONCE per world and its
    model state snapshotted; each utterance restores the snapshot and decodes only the
    user turn. This matters on hybrid/recurrent models (LFM2) where llama.cpp cannot
    rewind to a shared prefix: measured TTFT 1.3 s -> 0.08 s on a 4-thread CPU.
    """
    name = "llm"

    def __init__(self, model_path, locs, objs, n_threads=2, n_ctx=2048, lora_path=None, max_tokens=200):
        from llama_cpp import Llama                       # lazy: optional dependency
        self.model_path = model_path
        self.model = model_path.rsplit("/", 1)[-1]
        self.max_tokens = int(max_tokens)
        self._lock = threading.Lock()                     # llama.cpp contexts are not thread-safe
        kw = {"lora_path": lora_path} if lora_path else {}
        self.llm = Llama(model_path=model_path, n_ctx=int(n_ctx), n_threads=int(n_threads),
                         n_threads_batch=int(n_threads), n_gpu_layers=0, verbose=False, **kw)
        self.reload_world(locs, objs)

    def reload_world(self, locs, objs):
        from llama_cpp import LlamaGrammar
        with self._lock:
            self.locs, self.objs = locs, objs
            self.grammar = LlamaGrammar.from_json_schema(json.dumps(plan_schema(list(locs), list(objs))),
                                                         verbose=False)
            self.system = system_prompt(locs, objs)
            self.pre_txt, self.post_txt = split_template(self.llm, self.system)
            self.pre = self.llm.tokenize(self.pre_txt.encode(), add_bos=True, special=True)
            t0 = time.perf_counter()
            self.llm.reset()
            self.llm.eval(self.pre)
            self.prefill_s = time.perf_counter() - t0
            self.state = self.llm.save_state()

    def parse(self, text):
        t0 = time.perf_counter()
        with self._lock:
            self.llm.load_state(self.state)
            full = self.llm.tokenize((self.pre_txt + text + self.post_txt).encode(), add_bos=True, special=True)
            raw, n = "", 0
            for tok in self.llm.generate(full[len(self.pre):], temp=0.0, grammar=self.grammar, reset=False):
                if tok == self.llm.token_eos() or n >= self.max_tokens:
                    break
                raw += self.llm.detokenize([tok]).decode("utf-8", "ignore")
                n += 1
                if raw.rstrip().endswith("}") and raw.count("{") == raw.count("}"):
                    break
        try:
            plan = json.loads(raw)
        except (ValueError, TypeError):
            plan = dict(BAD_PLAN)
        return plan, time.perf_counter() - t0


def make_engine(cfg, log=None):
    """cfg keys: locs, objs, engine ('auto'|'llm'|'rules'), model, models_dir, num_threads, n_ctx.

    'auto' -> LLM if the GGUF is found and llama_cpp loads it, else rules (warning logged).
    'llm'  -> LLM or raise. 'rules' -> rules.
    """
    log = log or logging.getLogger("intent_mod")
    locs, objs = cfg["locs"], cfg["objs"]
    mode = cfg.get("engine", "auto")
    if mode not in ("auto", "llm", "rules"):
        raise ValueError(f"engine must be auto|llm|rules, got {mode!r}")
    if mode == "rules":
        return RuleIntentEngine(locs, objs)
    name = cfg.get("model") or DEFAULT_MODEL
    path = find_model(name, cfg.get("models_dir", ""))
    if path is None:
        msg = f"intent model '{name}' not found in any models root"
        if mode == "llm":
            raise FileNotFoundError(msg)
        log.warning(msg + " - falling back to the rule engine")
        return RuleIntentEngine(locs, objs)
    try:
        lora = cfg.get("lora") or ""
        lora_path = find_model(lora, cfg.get("models_dir", "")) if lora else None
        return LlamaIntentEngine(path, locs, objs, n_threads=cfg.get("num_threads", 2),
                                 n_ctx=cfg.get("n_ctx", 2048), lora_path=lora_path)
    except Exception as e:  # noqa: BLE001  (ImportError, bad GGUF, OOM ...)
        if mode == "llm":
            raise
        log.warning(f"could not load LLM {path}: {e!r} - falling back to the rule engine")
        return RuleIntentEngine(locs, objs)
