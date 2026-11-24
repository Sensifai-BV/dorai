"""The robot the lab simulates: config/robots/<name>.yaml (default: artaban, as
it is built now -- three separate USB microphones in a line for voice_mod, a
ReSpeaker 4-Mic Array for doa_mod, 0.5 m above the floor)."""
from __future__ import annotations

from .config import load_yaml


def load_robot(name="artaban"):
    r = load_yaml("robots", name) if isinstance(name, str) else dict(name)
    r.setdefault("capture", {})
    r.setdefault("hardware", {})
    r.setdefault("self_noise", [])
    r["array_names"] = [a["name"] for a in r.get("arrays", [])]
    return r
