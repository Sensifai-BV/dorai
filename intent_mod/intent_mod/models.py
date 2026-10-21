"""Locate the intent GGUF model.

Search roots, in order (the first root holding the file wins):
the ``models_dir`` parameter, $DORAI_MODELS_DIR, the package's own ``models/`` dir
(intent_mod/models in the source tree, share/intent_mod/models when installed),
~/dorai/models, /root/.cache/dorai/models. Inside a root the file is looked up as
``<root>/<name>`` and then ``<root>/<subdir>/<name>`` (one level, sorted).
``name`` may also be an absolute path.
"""
from __future__ import annotations

import glob
import os

DEFAULT_LORA = "intent-lfm25-1.2b-lora-f16.gguf"
DEFAULT_MODEL = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"   # base; + lora intent-lfm25-1.2b-lora-f16.gguf


def _package_models_dirs():
    here = os.path.dirname(os.path.abspath(__file__))
    out = [os.path.normpath(os.path.join(here, "..", "models"))]   # source tree / symlink-install
    try:
        from ament_index_python.packages import get_package_share_directory
        out.append(os.path.join(get_package_share_directory("intent_mod"), "models"))
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


def find_model(name=DEFAULT_MODEL, models_dir="", roots=None):
    """Path of the GGUF, or None when it is not installed in any root."""
    name = name or DEFAULT_MODEL
    if os.path.isabs(name):
        return name if os.path.isfile(name) else None
    for root in (roots if roots is not None else default_roots(models_dir)):
        root = os.path.expanduser(root)
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return p
        hits = sorted(glob.glob(os.path.join(glob.escape(root), "*", name)))
        if hits:
            return hits[0]
    return None
