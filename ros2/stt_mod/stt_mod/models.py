"""Locate the sherpa-onnx model directories: Whisper (default engine) and Nemotron.

Whisper (``engine: whisper``): one directory per export, named

    sherpa-onnx-whisper-<size>[-dorai-<tag>]

holding ``*encoder*.onnx``, ``*decoder*.onnx`` and ``*tokens.txt`` (int8 files are
preferred when both are present). ``-dorai-`` marks a dorai fine-tune; with no
name given the newest dorai fine-tune is used, else the newest Whisper export.

Nemotron (``engine: nemotron``): the directory for a latency/adapter choice.

Model directory naming (all under one models root):

    sherpa-onnx-nemotron-speech-streaming-en-0.6b-<L>ms-int8-2026-04-25               stock NVIDIA model
    sherpa-onnx-nemotron-speech-streaming-en-0.6b-<L>ms-int8-2026-04-25-dorai-<A>     + dorai adapter <A>

<L> is the encoder chunk = the latency profile (80 | 160 | 560 | 1120 ms), and
<A> is a fine-tuned dorai adapter baked in by ``stt_mod.bake_adapter``
(``mc-ref``: single-mic workshop adapter, ``mc-das``: beamformed-input adapter).

Search roots, in order (the first root holding the requested model wins):
the ``models_dir`` parameter, $DORAI_MODELS_DIR, the package's own ``models/``
dir (stt_mod/models in the source tree, share/stt_mod/models when installed),
~/dorai/models, /root/.cache/dorai/models.

The models are published in one Hugging Face repo, ``sensifai/dorai``:

    stt-model/whisper-finetuned/sherpa-onnx-whisper-large-v3-turbo-dorai-artaban/   Whisper (default engine)
    stt-model/nemo-finetuned/                                                       Nemotron 1120 ms + mc-ref
    intent-model/intent-qwen35-4b-Q4_0.gguf                                         intent_mod's model

A root may hold that layout as downloaded (``hf download sensifai/dorai
--local-dir ~/dorai/models``, or scripts/download_models.sh): every root is
also searched under ``stt-model/`` and ``stt-model/whisper-finetuned/``, and a
directory without the sherpa-onnx name (``nemo-finetuned``) is identified by
its ``dorai_adapter.json``.
"""
from __future__ import annotations

import json
import os
import re

HF_REPO = "sensifai/dorai"                 # one Hub repo for every dorai model
# where a download of that repo puts the speech-to-text models, relative to a models root
HF_STT_DIRS = ("stt-model", os.path.join("stt-model", "whisper-finetuned"), "whisper-finetuned")

PREFIX = "sherpa-onnx-nemotron-speech-streaming-en-0.6b"
LATENCIES_MS = (80, 160, 560, 1120)
ADAPTERS = ("none", "mc-ref", "mc-das")


# Endpoint trailing silence per latency profile, measured on 60 short real
# utterances (mc-ref): a shorter rule2 splits utterances at short pauses
# (80 ms @ 0.3 s: 12 % split; 160 ms @ 0.4 s: 5 %), while 80 ms @ 0.5 s split
# almost none (2 %) and cut the end-of-speech lag 1.67 -> 1.32 s.
AUTO_RULE2_S = {80: 0.5, 160: 0.8, 560: 0.8, 1120: 0.8}


def auto_rule2(latency_ms):
    return AUTO_RULE2_S.get(int(latency_ms), 0.8)


def _package_models_dirs():
    here = os.path.dirname(os.path.abspath(__file__))
    out = [os.path.normpath(os.path.join(here, "..", "models"))]   # source tree / symlink-install
    try:
        from ament_index_python.packages import get_package_share_directory
        out.append(os.path.join(get_package_share_directory("stt_mod"), "models"))
    except Exception:  # not a ROS environment, or package not installed
        pass
    return out


def default_roots(extra=""):
    roots = [extra, os.environ.get("DORAI_MODELS_DIR", ""), *_package_models_dirs(),
             os.path.join(os.path.expanduser("~"), "dorai", "models"),
             "/root/.cache/dorai/models"]
    seen, out = set(), []
    for r in roots:
        if r and os.path.realpath(os.path.expanduser(r)) not in seen:
            seen.add(os.path.realpath(os.path.expanduser(r)))
            out.append(r)
    return out


def expand_roots(roots):
    """Each root, followed by the places a download of the Hub repo puts the STT models in it."""
    out = []
    for r in roots:
        r = os.path.expanduser(r)
        out += [r] + [os.path.join(r, s) for s in HF_STT_DIRS]
    return out


def parse_dir(name):
    """'...-560ms-int8-2026-04-25-dorai-mc-ref' -> (560, 'mc-ref'); stock -> (560, 'none').

    A directory with another name (the Hub's ``nemo-finetuned``) is read from its
    ``dorai_adapter.json``: the adapter name and the latency of the export it was baked
    into. Anything else -> None."""
    base = os.path.basename(name.rstrip("/"))
    if not base.startswith(PREFIX):
        info = os.path.join(name, "dorai_adapter.json")
        if not os.path.isfile(info):
            return None
        try:
            with open(info) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return None
        m = re.search(r"-(\d+)ms-", str(d.get("base_export", "")))
        if not (m and d.get("adapter")):
            return None
        return int(m.group(1)), str(d["adapter"])
    m = re.search(r"-(\d+)ms-", base)
    lat = int(m.group(1)) if m else 560
    a = re.search(r"-dorai-(.+)$", base)
    return lat, (a.group(1) if a else "none")


def chunk_ms_of(model_dir):
    p = parse_dir(model_dir)
    return p[0] if p else 560


def list_models(roots):
    """All (latency_ms, adapter, path) found, root order preserved."""
    out = []
    for root in expand_roots(roots):
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            if not (os.path.isdir(p) and os.path.exists(os.path.join(p, "tokens.txt"))):
                continue
            parsed = parse_dir(p)
            if parsed:
                out.append((parsed[0], parsed[1], p))
    return out


def find_model(latency_ms, adapter="none", model_dir="", models_dir=""):
    """Return (path, latency_ms, adapter). Raises FileNotFoundError with the
    list of installed models if the exact combination is missing."""
    if model_dir:
        p = os.path.expanduser(model_dir)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"model_dir not found: {p}")
        lat, ad = parse_dir(p) or (int(latency_ms), adapter)
        return p, lat, ad
    adapter = (adapter or "none").strip().lower()
    roots = default_roots(models_dir)
    found = list_models(roots)
    for root in roots:   # root order decides; newest wins only within one root
        hits = [p for lat, ad, p in list_models([root])
                if lat == int(latency_ms) and ad == adapter]
        if hits:
            return max(hits, key=os.path.getmtime), int(latency_ms), adapter
    have = ", ".join(f"{lat}ms/{ad}" for lat, ad, _ in found) or "none"
    raise FileNotFoundError(
        f"no Nemotron model for latency_ms={latency_ms} adapter={adapter} in "
        f"{roots}. Installed: {have}. Get it with stt_mod/scripts/download_models.sh "
        f"(from huggingface.co/{HF_REPO}; see stt_mod/README.md).")


def pick_onnx(model_dir, prefix):
    names = sorted(n for n in os.listdir(model_dir)
                   if n.startswith(prefix) and n.endswith(".onnx"))
    if not names:
        raise FileNotFoundError(f"no {prefix}*.onnx in {model_dir}")
    int8 = [n for n in names if "int8" in n]
    return os.path.join(model_dir, (int8 or names)[0])


# ------------------------------------------------------------------ Whisper
WHISPER_PREFIX = "sherpa-onnx-whisper"


def pick_whisper_files(model_dir):
    """(encoder, decoder, tokens) of a sherpa-onnx Whisper export; int8 preferred."""
    names = sorted(os.listdir(model_dir))

    def one(key, ext):
        hits = [n for n in names if key in n and n.endswith(ext)]
        if not hits:
            raise FileNotFoundError(f"no *{key}*{ext} in {model_dir}")
        int8 = [n for n in hits if "int8" in n]
        return os.path.join(model_dir, (int8 or hits)[0])

    return one("encoder", ".onnx"), one("decoder", ".onnx"), one("tokens", ".txt")


def list_whisper_models(roots):
    """All (name, path) of Whisper exports found, root order preserved."""
    out = []
    for root in expand_roots(roots):
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            if not (d.startswith(WHISPER_PREFIX) and os.path.isdir(p)):
                continue
            try:
                pick_whisper_files(p)
            except FileNotFoundError:
                continue
            out.append((d, p))
    return out


def find_whisper_model(name="", model_dir="", models_dir=""):
    """Return (path, name). ``name`` is a directory name or its tail (e.g.
    ``dorai-artaban``); empty = newest dorai fine-tune, else newest export, of
    the first root that has one. Raises FileNotFoundError listing what is installed."""
    if model_dir:
        p = os.path.expanduser(model_dir)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"model_dir not found: {p}")
        pick_whisper_files(p)
        return p, os.path.basename(p.rstrip("/"))
    name = (name or "").strip()
    roots = default_roots(models_dir)
    for root in roots:
        found = list_whisper_models([root])
        if name:
            hits = [(d, p) for d, p in found if d == name or d.endswith("-" + name)]
        else:
            hits = [(d, p) for d, p in found if "-dorai-" in d] or found
        if hits:
            d, p = max(hits, key=lambda it: os.path.getmtime(it[1]))
            return p, d
    have = ", ".join(d for d, _ in list_whisper_models(roots)) or "none"
    raise FileNotFoundError(
        f"no Whisper model{' ' + repr(name) if name else ''} in {roots}. Installed: {have}. "
        f"Get it with stt_mod/scripts/download_models.sh (from huggingface.co/{HF_REPO}; "
        "see stt_mod/README.md).")
