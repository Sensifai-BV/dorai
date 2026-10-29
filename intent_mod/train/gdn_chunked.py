"""Chunked, differentiable Gated DeltaNet recurrence for training Qwen3.5 under mlx_lm.

mlx_lm only has a forward Metal kernel for the gated delta rule; in training mode
(`use_kernel=not self.training`) it falls back to `gated_delta_ops`, a Python loop of one
step per token. For a ~700-token prompt that is 700 x 24 layers of tiny ops per step: the
compiled graph takes many minutes to build, and backward keeps a [Hv, Dv, Dk] state per
token (2 MB/token/layer), so a 4B LoRA run needed more than 17 GB and barely progressed.

This is the chunkwise-parallel (WY) form of the same recurrence, as in flash-linear-attention's
chunk_gated_delta_rule: within a chunk of C tokens everything is matmuls, and state is only
carried across T/C chunk boundaries. The intra-chunk unit-lower-triangular inverse is done by
recursive 2x2 blocking (numerically the same as forward substitution), so it needs no
triangular solve and no vjp beyond matmul. (The shorter Neumann product
(I + A)(I + A^2)...(I + A^32) is exact in theory but A^32 reaches ~1e18 on real activations
and NaN'd training.) Matches `gated_delta_ops` to fp32 rounding (see test_gdn_chunked.py).

    import gdn_chunked; gdn_chunked.patch()     # before mlx_lm builds the model
"""
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def _unit_lower_inv(M):
    """Inverse of a unit lower-triangular [..., n, n] (n a power of two) by 2x2 block recursion."""
    n = M.shape[-1]
    if n == 1:
        return mx.ones_like(M)
    h = n // 2
    X11 = _unit_lower_inv(M[..., :h, :h])
    X22 = _unit_lower_inv(M[..., h:, h:])
    X21 = -(X22 @ M[..., h:, :h]) @ X11
    top = mx.concatenate([X11, mx.zeros_like(X11)], axis=-1)
    return mx.concatenate([top, mx.concatenate([X21, X22], axis=-1)], axis=-2)


def gated_delta_chunked(q, k, v, log_g, beta, state=None, mask=None, chunk=64):
    """q, k [B,T,Hk,Dk]; v [B,T,Hv,Dv]; log_g, beta [B,T,Hv]; state [B,Hv,Dv,Dk]; mask [B,T].

    Returns (y [B,T,Hv,Dv] in q.dtype, state [B,Hv,Dv,Dk] fp32). Same contract as gated_delta_ops,
    except the decay is passed as its log (exp(-exp(A_log) * softplus(.)) underflows to 0)."""
    assert chunk & (chunk - 1) == 0, "chunk must be a power of two"
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    out_dtype = q.dtype
    if Hv // Hk > 1:
        q = mx.repeat(q, Hv // Hk, -2)
        k = mx.repeat(k, Hv // Hk, -2)
    f32 = mx.float32
    q, k, v, log_g, beta = (x.astype(f32) for x in (q, k, v, log_g, beta))
    if mask is not None:                                   # masked step = no decay, no write
        log_g = mx.where(mask[..., None], log_g, 0.0)
        beta = mx.where(mask[..., None], beta, 0.0)
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=f32)

    C = chunk
    N = -(-T // C)
    pad = N * C - T
    if pad:                                                # padded tail: no decay, no write, zero q
        pw = [(0, 0), (0, pad), (0, 0), (0, 0)]
        q, k, v = mx.pad(q, pw), mx.pad(k, pw), mx.pad(v, pw)
        log_g, beta = mx.pad(log_g, pw[:3]), mx.pad(beta, pw[:3])

    # [B, H, N, C, D] / [B, H, N, C]
    split = lambda x: x.reshape(B, N, C, Hv, -1).transpose(0, 3, 1, 2, 4)  # noqa: E731
    q, k, v = split(q), split(k), split(v)
    log_g = log_g.reshape(B, N, C, Hv).transpose(0, 3, 1, 2)
    beta = beta.reshape(B, N, C, Hv).transpose(0, 3, 1, 2)

    gc = mx.cumsum(log_g, axis=-1)                         # log decay from chunk start, inclusive
    idx = mx.arange(C)
    lower = idx[:, None] >= idx[None, :]
    strict = idx[:, None] > idx[None, :]
    diff = gc[..., :, None] - gc[..., None, :]
    decay = mx.exp(mx.where(lower, diff, -mx.inf))         # Gamma_ij = prod g_(j+1..i), i >= j

    kb = k * beta[..., None]
    A = mx.where(strict, -(kb @ k.swapaxes(-1, -2)) * decay, 0.0)
    Tinv = _unit_lower_inv(mx.eye(C, dtype=f32) - A)      # (I - A)^-1
    w = Tinv @ (kb * mx.exp(gc)[..., None])                # [B,H,N,C,Dk]
    u = Tinv @ (v * beta[..., None])                       # [B,H,N,C,Dv]
    attn = mx.where(lower, q @ k.swapaxes(-1, -2) * decay, 0.0)
    q_dec = q * mx.exp(gc)[..., None]
    k_dec = k * mx.exp(gc[..., -1:] - gc)[..., None]
    g_end = mx.exp(gc[..., -1])                            # [B,H,N]

    S = state.swapaxes(-1, -2)                             # [B,H,Dk,Dv]
    ys = []
    for n in range(N):
        v_new = u[:, :, n] - w[:, :, n] @ S
        ys.append(q_dec[:, :, n] @ S + attn[:, :, n] @ v_new)
        S = S * g_end[:, :, n, None, None] + k_dec[:, :, n].swapaxes(-1, -2) @ v_new
    y = mx.stack(ys, axis=2).reshape(B, Hv, N * C, Dv)[:, :, :T].transpose(0, 2, 1, 3)
    return y.astype(out_dtype), S.swapaxes(-1, -2)


def patch(chunk=64):
    """Route mlx_lm's training-mode (use_kernel=False) gated delta calls through the chunked form."""
    from mlx_lm.models import gated_delta, qwen3_5
    orig = gated_delta.gated_delta_update

    def update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *, use_kernel=True,
               lower_bound=None, allow_neg_eigval=False):
        if use_kernel or lower_bound is not None:
            return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel,
                        lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
        beta = mx.sigmoid(b)
        if allow_neg_eigval:
            beta = beta * 2.0
        log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
        return gated_delta_chunked(q, k, v, log_g, beta, state, mask, chunk)

    qwen3_5.gated_delta_update = update
    return orig
