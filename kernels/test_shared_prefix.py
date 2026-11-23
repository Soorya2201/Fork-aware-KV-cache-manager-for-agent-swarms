"""
Correctness tests for the shared-prefix decomposition.

Mac (CPU): checks the pure-torch decomposition against naive attention.
CARC (CUDA + Triton): also checks the Triton kernel.

  uv run python kernels/test_shared_prefix.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_prefix_attn import (HAS_TRITON, attention_decomposed_torch,  # noqa: E402
                                attention_reference, shared_prefix_attention)


def make(G, F, Hq, Hk, D, P, Smax, device, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).to(device=device, dtype=dtype)  # noqa: E731
    suf_len = torch.randint(0, Smax + 1, (G, F), generator=g, dtype=torch.int32).to(device)
    return (r(G, F, Hq, D), r(G, P, Hk, D), r(G, P, Hk, D),
            r(G, F, Smax, Hk, D), r(G, F, Smax, Hk, D), suf_len)


def main():
    cases = [(1, 4, 4, 2, 32, 100, 17), (2, 8, 28, 4, 64, 300, 40), (3, 3, 8, 8, 16, 1, 0)]
    for c in cases:
        args = make(*c, device="cpu", dtype=torch.float32)
        ref = attention_reference(*args)
        dec = attention_decomposed_torch(*args)
        err = (ref - dec).abs().max().item()
        assert err < 1e-4, (c, err)
        print(f"decomposed torch == reference  {c}  max err {err:.2e}")
    if HAS_TRITON and torch.cuda.is_available():
        for c in cases + [(4, 16, 28, 4, 128, 6000, 300)]:
            args = make(*c, device="cuda", dtype=torch.float16)
            ref = attention_reference(*args)
            out = shared_prefix_attention(*args, chunk=512)
            err = (ref - out).abs().max().item()
            assert err < 2e-2, (c, err)
            print(f"triton == reference  {c}  max err {err:.2e}")
    else:
        print("Triton/CUDA not available: kernel test skipped (run on CARC)")


if __name__ == "__main__":
    main()
