"""
analysis/compare_sim_real.py — does the simulator predict real SGLang? (Phase 3, result #3)

Compares per-call cached prompt tokens reported by SGLang (bench/replay.py with
--enable-cache-report) against forkcache-sim's prediction for the same schedule
(--per-call-out, LRU policy in budget mode at the server's KV pool size).

  python analysis/compare_sim_real.py --real results/radix_<job>.json \
      --sim results/sim/per_call_lru.jsonl --policy lru --pool-gb 18.7

SGLang caches at token granularity (page size 1) while the simulator uses 16-token
blocks, so the simulator can under-predict by up to 15 tokens per call; the
report shows the error with and without that rounding.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--policy", default="lru")
    ap.add_argument("--pool-gb", type=float, required=True)
    ap.add_argument("--block", type=int, default=16)
    args = ap.parse_args()

    real = {c["call_id"]: c for c in json.load(open(args.real))["calls"]}
    sim = {}
    for l in open(args.sim):
        r = json.loads(l)
        if r["policy"] == args.policy and r["mode"] == "budget" and abs(r["pool_gb"] - args.pool_gb) < 1e-3:
            sim[r["call_id"]] = r["hit_tokens"]
    ids = sorted(set(real) & set(sim))
    if not ids:
        raise SystemExit("no overlapping calls: check --policy / --pool-gb")
    rc = np.array([real[i]["cached_tokens"] or 0 for i in ids], float)
    rc_blk = np.floor(rc / args.block) * args.block
    sc = np.array([sim[i] for i in ids], float)
    P = np.array([real[i]["prompt_tokens"] for i in ids], float)
    tot_real, tot_blk, tot_sim = rc.sum(), rc_blk.sum(), sc.sum()
    print(f"calls compared: {len(ids)}")
    print(f"cached fraction  real {tot_real / P.sum():.3f}   real (block-rounded) {tot_blk / P.sum():.3f}"
          f"   simulated {tot_sim / P.sum():.3f}")
    print(f"aggregate error vs block-rounded real: {100 * (tot_sim - tot_blk) / max(tot_blk, 1):+.1f}%")
    print(f"per-call MAE: {np.abs(sc - rc_blk).mean():.1f} tokens; calls exactly right: "
          f"{100 * np.mean(sc == rc_blk):.1f}%")


if __name__ == "__main__":
    main()
