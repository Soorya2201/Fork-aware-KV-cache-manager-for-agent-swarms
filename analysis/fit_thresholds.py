"""
analysis/fit_thresholds.py — export distribution-aware keep thresholds for SGLang.

The server-side `distaware` strategy (sglang_ext/) cannot see agent ids, so it uses
one threshold per context-length bucket. For each bucket this fits
theta* = argmin r E[min(T, theta)] + B P(T > theta) over the bucket's idle gaps
(paper/model.md, section 4), with r = L / pool_tokens and B = bL + cL^2.

  python analysis/fit_thresholds.py --trace trace/zage_trace.jsonl \
      --curve results/prefill/a100_qwen7b_curve.json --pool-tokens 300000 \
      --out results/thresholds.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import orjson


def best_threshold(gaps, rent, buy):
    g = np.sort(np.asarray(gaps, float))
    n = len(g)
    best_t, best_c = 0.0, buy * np.mean(g > 0)
    pre = np.concatenate([[0.0], np.cumsum(g)])
    for k in range(n):
        m = np.searchsorted(g, g[k], side="right")
        c = rent * (pre[m] + g[k] * (n - m)) / n + buy * (n - m) / n
        if c < best_c:
            best_t, best_c = float(g[k]), float(c)
    return best_t, best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--curve", required=True)
    ap.add_argument("--pool-tokens", type=int, required=True)
    ap.add_argument("--buckets", type=int, default=6)
    ap.add_argument("--time-scale", type=float, default=1.0, help="divide trace gaps by this")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cur = json.load(open(args.curve))
    recs = [orjson.loads(l) for l in open(args.trace, "rb") if l.strip()]
    rows = [(r["prompt_tokens"] + r["output_tokens"], r["next_call_same_agent_gap_s"] / args.time_scale)
            for r in recs if r["next_call_same_agent_gap_s"] is not None]
    L = np.array([x[0] for x in rows]); T = np.array([x[1] for x in rows])
    edges = np.unique(np.quantile(L, np.linspace(0, 1, args.buckets + 1)))
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (L >= lo) & (L <= hi)
        Lm = float(np.median(L[sel]))
        rent, buy = Lm / args.pool_tokens, cur["b"] * Lm + cur["c"] * Lm * Lm
        theta, _ = best_threshold(T[sel], rent, buy)
        table.append([float(hi), theta])
        print(f"L in [{lo:6.0f}, {hi:6.0f}]  n={sel.sum():4d}  tau={buy / rent:7.2f}s  "
              f"gap P50={np.median(T[sel]):6.2f}s  theta*={theta:6.2f}s")
    table[-1][0] = 1e12
    json.dump(table, open(args.out, "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
