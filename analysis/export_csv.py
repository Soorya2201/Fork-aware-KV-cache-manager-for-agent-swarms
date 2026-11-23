"""
analysis/export_csv.py — export a trace and its simulator results as plain CSV files.

  python analysis/export_csv.py --trace trace/zage_trace.jsonl --sim results/sim/mac.jsonl --out data --prefix mac_
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import orjson
import pandas as pd
from scipy import stats


def ci95(x):
    x = np.asarray(x, float)
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--sim", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--kv-bytes-per-token", type=int, default=57344)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    p = lambda name: os.path.join(a.out, f"{a.prefix}{name}.csv")

    recs = [orjson.loads(l) for l in open(a.trace, "rb") if l.strip()]
    calls = pd.DataFrame([{k: v for k, v in r.items() if k != "block_hashes"} | {"n_blocks": len(r["block_hashes"])}
                          for r in recs])
    calls["context_tokens"] = calls.prompt_tokens + calls.output_tokens
    calls["latency_s"] = (calls.t_end - calls.t_start).round(4)
    calls["kv_mb_7b"] = (calls.context_tokens * a.kv_bytes_per_token / 2**20).round(1)
    calls.to_csv(p("calls"), index=False)

    ctx = calls.set_index("call_id").context_tokens
    ch = calls[calls.parent_call.notna()].copy()
    ch["reused_fraction"] = (ch.shared_with_parent_tokens / ch.parent_call.astype(int).map(ctx)).round(4)
    idle = calls[calls.next_call_same_agent_gap_s.notna()][
        ["call_id", "session", "agent", "round", "context_tokens", "next_call_same_agent_gap_s"]]
    idle = idle.merge(ch[["parent_call", "reused_fraction"]].rename(columns={"parent_call": "call_id"}),
                      on="call_id", how="left").rename(columns={"next_call_same_agent_gap_s": "idle_gap_s"})
    idle.to_csv(p("idle_periods"), index=False)

    rs = calls.groupby("round").agg(calls=("call_id", "count"), mean_prompt_tokens=("prompt_tokens", "mean"),
                                    mean_output_tokens=("output_tokens", "mean"), mean_latency_s=("latency_s", "mean"),
                                    repeat_tokens=("shared_prefix_tokens", "sum"),
                                    prompt_tokens=("prompt_tokens", "sum")).reset_index()
    rs["repeat_pct"] = (100 * rs.repeat_tokens / rs.prompt_tokens).round(1)
    rs = rs.merge(ch.groupby("round").reused_fraction.mean().rename("mean_reused_fraction_of_prev_kv").reset_index(),
                  on="round", how="left")
    rs.round(3).to_csv(p("per_round_summary"), index=False)
    out = [p("calls"), p("idle_periods"), p("per_round_summary")]

    if a.sim:
        sim = pd.DataFrame([orjson.loads(l) for l in open(a.sim, "rb") if l.strip()])
        sim["pool_gb"] = sim.pool_gb.round(2)
        rows = []
        for (m, g, pol), d in sim.groupby(["mode", "pool_gb", "policy"]):
            rows.append(dict(mode=m, pool_gb=g, policy=pol, seeds=len(d),
                             total_cost_gpu_s=d.total_cost_gpu_s.mean(), total_cost_ci95=ci95(d.total_cost_gpu_s),
                             recompute_tokens=d.recompute_tokens.mean(), recompute_ci95=ci95(d.recompute_tokens),
                             hit_rate=d.hit_rate.mean(), hit_rate_ci95=ci95(d.hit_rate), peak_kv_gb=d.peak_kv_gb.mean()))
        pd.DataFrame(rows).round(4).to_csv(p("policy_results"), index=False)
        out.append(p("policy_results"))
    for f in out:
        print(f"{f}: {sum(1 for _ in open(f)) - 1} rows")


if __name__ == "__main__":
    main()
