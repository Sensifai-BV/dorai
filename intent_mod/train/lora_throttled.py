#!/usr/bin/env python3
"""Run `mlx_lm.lora` without taking over the Mac: same CLI/config, capped GPU duty and memory.

Stock mlx_lm wires the device's whole recommended working set (~75% of RAM on a 24 GB M4 Pro)
and keeps the GPU at 100% back to back, so the desktop stutters for the whole run. Here:

  * GPU duty cycle: after every evaluated step the trainer sleeps so the GPU is busy at most
    `--duty` of wall time (0.7 -> ~30% idle gaps the window server can use).
  * Memory: wired limit off, MLX allocation limit = `--mem-frac` of RAM, small buffer cache.
  * CPU: the process lowers its own priority (nice 10).
  * Qwen3.5 DeltaNet layers train through gdn_chunked (chunkwise matmuls) instead of mlx_lm's
    one-op-per-token fallback, which needed >17 GB and stalled for minutes per step.

    python intent_mod/train/lora_throttled.py --duty 0.7 --mem-frac 0.5 -- \
        -c intent_mod/train/lora_qwen35_4b.yaml --model /path/to/Qwen3.5-4B
"""
import argparse
import os
import sys
import time

import mlx.core as mx


class _ThrottledMX:
    """Stands in for `mlx.core` inside the trainer: identical except eval() keeps a duty cycle."""

    def __init__(self, duty):
        self._rest = (1.0 - duty) / duty

    def __getattr__(self, name):
        return getattr(mx, name)

    def eval(self, *args):
        t0 = time.perf_counter()
        mx.eval(*args)
        time.sleep(self._rest * (time.perf_counter() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duty", type=float, default=0.7, help="max GPU busy fraction (0-1]")
    ap.add_argument("--mem-frac", type=float, default=0.5, help="MLX memory limit as a fraction of RAM")
    ap.add_argument("--nice", type=int, default=10)
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    os.nice(a.nice)
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    mx.set_memory_limit(int(a.mem_frac * ram))
    mx.set_cache_limit(1 << 30)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gdn_chunked
    gdn_chunked.patch()
    from mlx_lm import lora
    from mlx_lm.tuner import trainer
    trainer.maybe_set_recommended_wired_limit = lambda: None
    trainer.mx = _ThrottledMX(a.duty)
    print(f"throttled: duty {a.duty:.0%}, mem limit {a.mem_frac * ram / 1e9:.1f} GB, nice {a.nice}, no wired memory, chunked GDN",
          flush=True)

    sys.argv = ["mlx_lm.lora"] + rest
    lora.main()


if __name__ == "__main__":
    main()
