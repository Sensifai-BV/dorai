"""Constants and the config folder."""
from __future__ import annotations

import os

FS = 16000
C_SOUND = 343.0


def config_dir():
    """The repo's config/ folder ($DORAI_CONFIG overrides it)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get("DORAI_CONFIG") or os.path.normpath(os.path.join(here, "..", "..", "config"))


def load_yaml(kind, name):
    import yaml
    path = name if os.path.isfile(str(name)) else os.path.join(config_dir(), kind, f"{name}.yaml")
    if not os.path.isfile(path):
        have = sorted(f[:-5] for f in os.listdir(os.path.join(config_dir(), kind)) if f.endswith(".yaml"))
        raise FileNotFoundError(f"no {kind[:-1]} '{name}' in {os.path.join(config_dir(), kind)} (there: {', '.join(have)})")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge(base, over):
    """`over` on top of `base`, dict by dict."""
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out
