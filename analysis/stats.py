"""
analysis/stats.py — Phase 1 statistics over a workload trace (result #1).

  uv run python analysis/stats.py --trace trace/zage_trace.jsonl

Prints prefix-reuse, idle-gap, fork-tree and KV-memory statistics, writes a JSON
summary and an idle-gap histogram to results/.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import orjson

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# KV bytes per token = 2 (K and V) x layers x KV heads x head dim x bytes per element (fp16)
KV_BYTES = {
    "qwen2.5-7b":   2 * 28 * 4 * 128 * 2,   # 57,344
    "qwen2.5-0.5b": 2 * 24 * 2 * 64 * 2,    # 12,288
}


def load(path):
    with open(path, "rb") as f:
        return [orjson.loads(l) for l in f if l.strip()]


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def block_reuse(recs):
    """Prefill tokens servable from an unbounded block cache (16-token blocks)."""
    seen, hit = set(), 0
    for r in recs:
        bs = r["block_size"]
        n_full = r["prompt_tokens"] // bs          # only full prompt blocks are cacheable
        for i, h in enumerate(r["block_hashes"][:n_full]):
            if h in seen:
                hit += bs
            else:
                break                             # prefix property: first miss ends reuse
        seen.update(r["block_hashes"])
    return hit


def fork_tree(recs):
    """Tree edge = fork_of if set, else parent_call. Returns per-session depth/branching."""
    tparent = {r["call_id"]: (r["fork_of"] if r["fork_of"] is not None else r["parent_call"])
               for r in recs}
    depth = {}
    for r in recs:                                # call_ids are increasing: parents first
        p = tparent[r["call_id"]]
        depth[r["call_id"]] = 0 if p is None else depth[p] + 1
    children = defaultdict(list)
    for c, p in tparent.items():
        if p is not None:
            children[p].append(c)
    out = {}
    for s in sorted({r["session"] for r in recs}):
        ids = [r["call_id"] for r in recs if r["session"] == s]
        internal = [i for i in ids if children.get(i)]
        out[s] = {
            "calls": len(ids),
            "max_depth": max(depth[i] for i in ids),
            "mean_branching": float(np.mean([len(children[i]) for i in internal])) if internal else 0.0,
            "max_branching": max((len(children[i]) for i in internal), default=0),
            "fork_edges_in": sum(1 for r in recs if r["session"] == s and r["fork_of"] is not None),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=os.path.join(ROOT, "trace", "zage_trace.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "stats"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = load(args.trace)
    recs.sort(key=lambda r: r["call_id"])

    P = np.array([r["prompt_tokens"] for r in recs])
    O = np.array([r["output_tokens"] for r in recs])
    S = np.array([r["shared_prefix_tokens"] for r in recs])
    SP = np.array([r["shared_with_parent_tokens"] for r in recs])
    lat = np.array([r["t_end"] - r["t_start"] for r in recs])
    total = int(P.sum())
    rep = int(S.sum())
    fork_mask = np.array([r["fork_of"] is not None for r in recs])
    blk = block_reuse(recs)

    gaps = np.array([r["next_call_same_agent_gap_s"] for r in recs
                     if r["next_call_same_agent_gap_s"] is not None])
    idle_L = np.array([r["prompt_tokens"] + r["output_tokens"] for r in recs
                       if r["next_call_same_agent_gap_s"] is not None])

    # Concurrent idle KV (tokens) within the trace's own timeline: an agent's KV is
    # idle from t_end of a call to t_start of its next call.
    events = []
    for r in recs:
        g = r["next_call_same_agent_gap_s"]
        if g is not None:
            L = r["prompt_tokens"] + r["output_tokens"]
            events += [(r["t_end"], L), (r["t_end"] + g, -L)]
    events.sort()
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)

    summary = {
        "calls": len(recs),
        "sessions": len({r["session"] for r in recs}),
        "agents": len({r["agent"] for r in recs}),
        "rounds": max(r["round"] for r in recs),
        "prefill_tokens_total": total,
        "prefill_tokens_repeating_earlier_prefix": rep,
        "repeat_pct": 100.0 * rep / total,
        "repeat_pct_from_own_previous_call": 100.0 * int(SP.sum()) / total,
        "repeat_pct_block_granular_16": 100.0 * blk / total,
        "fork_calls": int(fork_mask.sum()),
        "repeat_pct_in_fork_calls": 100.0 * int(S[fork_mask].sum()) / total,
        "prompt_tokens": {"mean": float(P.mean()), "p50": pct(P, 50), "p90": pct(P, 90), "max": int(P.max())},
        "output_tokens": {"mean": float(O.mean()), "p50": pct(O, 50), "max": int(O.max())},
        "call_latency_s": {"p50": pct(lat, 50), "p90": pct(lat, 90)},
        "idle_gap_s": {"n": int(len(gaps)), "mean": float(gaps.mean()), "p50": pct(gaps, 50),
                       "p90": pct(gaps, 90), "p99": pct(gaps, 99), "max": float(gaps.max())},
        "idle_context_tokens": {"p50": pct(idle_L, 50), "p90": pct(idle_L, 90)},
        "kv_bytes_per_token": KV_BYTES,
        "kv_per_call_MB": {m: {"p50": pct((P + O) * b, 50) / 2**20, "max": float(((P + O) * b).max()) / 2**20}
                           for m, b in KV_BYTES.items()},
        "peak_concurrent_idle_tokens_sequential_replay": int(peak),
        "peak_concurrent_idle_GB": {m: peak * b / 2**30 for m, b in KV_BYTES.items()},
        "idle_KV_GB_seconds": {m: float((idle_L * gaps).sum() * b / 2**30) for m, b in KV_BYTES.items()},
        "fork_tree_per_session": fork_tree(recs),
    }

    by_round = defaultdict(lambda: [0, 0])
    for r in recs:
        by_round[r["round"]][0] += r["prompt_tokens"]
        by_round[r["round"]][1] += r["shared_prefix_tokens"]
    summary["repeat_pct_by_round"] = {k: 100.0 * v[1] / v[0] for k, v in sorted(by_round.items())}

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.save(os.path.join(args.out, "idle_gaps.npy"), gaps)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(gaps, bins=40, color="#3a6ea5")
    for q, ls in ((50, "-"), (90, "--"), (99, ":")):
        ax.axvline(pct(gaps, q), color="#222", ls=ls, lw=1, label=f"P{q} = {pct(gaps, q):.1f} s")
    ax.set_xlabel("idle gap before the agent's next call (s)")
    ax.set_ylabel("idle periods")
    ax.set_title("KV idle time per agent turn")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "idle_gap_hist.png"), dpi=150)

    s = summary
    print(f"Trace: {s['calls']} calls, {s['sessions']} sessions x {s['rounds']} rounds x {s['agents']} agents")
    print(f"Prefill tokens: {total:,}; repeating an earlier prefix: {rep:,} ({s['repeat_pct']:.1f}%)")
    print(f"  from the agent's own previous call: {s['repeat_pct_from_own_previous_call']:.1f}%   "
          f"block-granular (16): {s['repeat_pct_block_granular_16']:.1f}%   "
          f"in fork calls ({s['fork_calls']}): {s['repeat_pct_in_fork_calls']:.1f}%")
    print("  by round: " + ", ".join(f"R{k} {v:.0f}%" for k, v in s["repeat_pct_by_round"].items()))
    g = s["idle_gap_s"]
    print(f"Idle gaps (n={g['n']}): P50 {g['p50']:.1f}s  P90 {g['p90']:.1f}s  P99 {g['p99']:.1f}s  max {g['max']:.1f}s")
    print(f"Idle context: P50 {s['idle_context_tokens']['p50']:.0f} tokens; KV per call (7B fp16) "
          f"P50 {s['kv_per_call_MB']['qwen2.5-7b']['p50']:.0f} MB, max {s['kv_per_call_MB']['qwen2.5-7b']['max']:.0f} MB")
    print(f"Peak concurrent idle KV (as recorded, one session at a time): {peak:,} tokens = "
          f"{s['peak_concurrent_idle_GB']['qwen2.5-7b']:.2f} GB at 7B")
    ft = list(s["fork_tree_per_session"].values())
    print(f"Fork tree: max depth {max(t['max_depth'] for t in ft)}, mean branching "
          f"{np.mean([t['mean_branching'] for t in ft]):.2f}, max branching {max(t['max_branching'] for t in ft)}")
    print(f"Wrote {args.out}/summary.json and idle_gap_hist.png")


if __name__ == "__main__":
    main()
