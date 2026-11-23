"""
kernels/bench_cascade.py — shared-prefix decode attention: naive vs ours vs FlashInfer cascade.

Runs on CARC (CUDA). Reports median latency (CUDA events) and effective HBM traffic.
Report honestly even if ours loses: FlashInfer's cascade kernels are heavily tuned.

  python kernels/bench_cascade.py --out results/kernel_a100.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_prefix_attn import attention_reference, shared_prefix_attention  # noqa: E402


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def naive_sdpa(q, k_pre, v_pre, k_suf, v_suf, suf_len):
    """Prefix duplicated per fork, one padded SDPA call (what a fork-unaware engine does)."""
    G, F, Hq, D = q.shape
    P, Hk = k_pre.shape[1], k_pre.shape[2]
    R = Hq // Hk
    Smax = k_suf.shape[2]
    K = torch.cat([k_pre[:, None].expand(G, F, P, Hk, D), k_suf], 2).reshape(G * F, P + Smax, Hk, D)
    V = torch.cat([v_pre[:, None].expand(G, F, P, Hk, D), v_suf], 2).reshape(G * F, P + Smax, Hk, D)
    K = K.repeat_interleave(R, 2).transpose(1, 2)
    V = V.repeat_interleave(R, 2).transpose(1, 2)
    mask = torch.arange(P + Smax, device=q.device)[None] < (P + suf_len.reshape(-1))[:, None]
    return torch.nn.functional.scaled_dot_product_attention(
        q.reshape(G * F, Hq, 1, D), K, V, attn_mask=mask[:, None, None, :])


def flashinfer_cascade(q, k_pre, v_pre, k_suf, v_suf, suf_len):
    """FlashInfer: shared-prefix single-request attention + per-fork batch decode, LSE merge."""
    import flashinfer
    G, F, Hq, D = q.shape
    outs = []
    for g in range(G):
        o1, s1 = flashinfer.single_prefill_with_kv_cache(
            q[g], k_pre[g], v_pre[g], causal=False, return_lse=True)
        o2s, s2s = [], []
        for f in range(F):
            n = int(suf_len[g, f])
            if n == 0:
                o2s.append(torch.zeros_like(o1[f:f + 1])); s2s.append(torch.full_like(s1[f:f + 1], -1e30))
                continue
            o2, s2 = flashinfer.single_prefill_with_kv_cache(
                q[g, f:f + 1], k_suf[g, f, :n], v_suf[g, f, :n], causal=False, return_lse=True)
            o2s.append(o2); s2s.append(s2)
        o, _ = flashinfer.merge_state(o1, s1, torch.cat(o2s), torch.cat(s2s))
        outs.append(o)
    return torch.stack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev, dt = "cuda", torch.float16
    Hq, Hk, D = 28, 4, 128          # Qwen2.5-7B attention shape
    rows = []
    for G, F, P, S in [(1, 4, 4096, 256), (1, 13, 6000, 300), (4, 13, 6000, 300),
                       (1, 32, 16384, 512), (8, 8, 2048, 128)]:
        torch.manual_seed(0)
        q = torch.randn(G, F, Hq, D, device=dev, dtype=dt)
        kp, vp = (torch.randn(G, P, Hk, D, device=dev, dtype=dt) for _ in range(2))
        ks, vs = (torch.randn(G, F, S, Hk, D, device=dev, dtype=dt) for _ in range(2))
        sl = torch.full((G, F), S, device=dev, dtype=torch.int32)
        a = (q, kp, vp, ks, vs, sl)
        ref = attention_reference(*a)
        row = {"G": G, "F": F, "P": P, "S": S}
        for name, fn in (("naive_sdpa", naive_sdpa), ("ours_triton", shared_prefix_attention),
                         ("flashinfer_cascade", flashinfer_cascade)):
            try:
                out = fn(*a).float().reshape(ref.shape)
                row[name + "_max_err"] = (out - ref).abs().max().item()
                row[name + "_ms"] = timeit(lambda: fn(*a))
            except Exception as e:  # e.g. FlashInfer missing or API changed
                row[name + "_error"] = f"{type(e).__name__}: {e}"[:200]
        kv_bytes_shared = 2 * G * P * Hk * D * 2
        row["prefix_bytes_read_once_MB"] = kv_bytes_shared / 2**20
        row["prefix_bytes_naive_MB"] = kv_bytes_shared * F / 2**20
        rows.append(row)
        print(json.dumps(row))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "rows": rows}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
