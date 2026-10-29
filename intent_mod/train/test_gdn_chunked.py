"""gdn_chunked must equal mlx_lm's per-token reference (values, final state and gradients).

    python intent_mod/train/test_gdn_chunked.py      # needs mlx + mlx_lm (training venv)
"""
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import gated_delta_ops

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_chunked import gated_delta_chunked  # noqa: E402


def inputs(B=2, T=77, Hk=2, Hv=4, Dk=32, Dv=16, seed=0):
    mx.random.seed(seed)
    r = mx.random.normal
    q, k = r((B, T, Hk, Dk)), r((B, T, Hk, Dk))
    q = q / mx.linalg.norm(q, axis=-1, keepdims=True) * Dk ** -0.5
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    v, a, b = r((B, T, Hv, Dv)), r((B, T, Hv)), r((B, T, Hv))
    A_log, dt_bias = r((Hv,)) * 0.5, r((Hv,)) * 0.5
    state = r((B, Hv, Dv, Dk)) * 0.1
    mask = mx.arange(T)[None, :] < mx.array([[T], [T - 9]])
    return q, k, v, a, b, A_log, dt_bias, state, mask


def ref(q, k, v, a, b, A_log, dt_bias, state, mask):
    g = mx.exp(-mx.exp(A_log) * nn.softplus(a + dt_bias))
    return gated_delta_ops(q, k, v, g, mx.sigmoid(b), state, mask)


def new(q, k, v, a, b, A_log, dt_bias, state, mask, chunk):
    lg = -mx.exp(A_log) * nn.softplus(a + dt_bias)
    return gated_delta_chunked(q, k, v, lg, mx.sigmoid(b), state, mask, chunk)


def main():
    x = inputs()
    valid = x[8][..., None, None]          # outputs at masked (padding) steps are unspecified
    y0, s0 = ref(*x)
    worst = 0.0
    for chunk in (8, 16, 64):
        y1, s1 = new(*x, chunk)
        e = max(mx.abs(mx.where(valid, y0 - y1, 0)).max().item(), mx.abs(s0 - s1).max().item())
        print(f"chunk {chunk:3d}  forward max|err| {e:.2e}")
        worst = max(worst, e)

        def loss(fn, *p):
            y, s = fn(*p)
            return (mx.where(valid, y, 0) * mx.cos(mx.arange(y.size).reshape(y.shape))).sum() + (s ** 2).sum()

        argn = [0, 1, 2, 3, 4, 5, 6, 7]
        g0 = mx.grad(lambda *p: loss(ref, *p, x[8]), argn)(*x[:8])
        g1 = mx.grad(lambda *p: loss(lambda *pp: new(*pp, chunk), *p, x[8]), argn)(*x[:8])
        ge = max((mx.abs(u - w).max() / (mx.abs(u).max() + 1e-6)).item() for u, w in zip(g0, g1))
        print(f"            grad rel max|err| {ge:.2e}")
        worst = max(worst, ge)
    assert worst < 1e-3, worst
    print("OK")


if __name__ == "__main__":
    main()
