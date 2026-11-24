"""Strict JSON for published messages.

Python's json writes float('nan') / inf as the bare tokens NaN / Infinity, which
are not JSON: strict parsers (JavaScript JSON.parse, Go, jq, rosbridge clients)
reject the whole message. Every JSON a node publishes goes through dumps(): a
non-finite number becomes null, and allow_nan=False guarantees nothing slips by.
"""
import json
import math


def clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):     # numpy scalar
        try:
            return clean(o.item())
        except (TypeError, ValueError):
            return o
    return o


def dumps(o, **kw):
    return json.dumps(clean(o), allow_nan=False, **kw)
