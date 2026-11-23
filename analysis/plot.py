"""
analysis/plot.py — Phase 2/6 evaluation of forkcache-sim results.

  uv run python analysis/plot.py --results results/sim/sim_results.jsonl --out results/sim

* Budget mode: recompute tokens vs KV pool size, per policy, mean and 95% CI over seeds.
* Cost mode: total cost (memory + recompute GPU-s) and competitive ratio vs the oracle.
* Paired comparisons: each seed replays the same schedule under every policy, so
  differences are paired per seed (t-interval on the per-seed difference).
* "Where we lose": every (mode, pool) cell where the proposed policy is not better
  than the best baseline, written out explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

PROPOSED = "dist_aware_agent"
BASELINES = ["lru", "ttl_10s", "ttl_30s", "ttl_60s", "continuum_style", "ski_rental", "ski_rental_rand"]
COLORS = {"lru": "#7f7f7f", "ttl_10s": "#c7c7c7", "ttl_30s": "#bcbd22", "ttl_60s": "#dbdb8d",
          "continuum_style": "#9467bd", "ski_rental": "#1f77b4", "ski_rental_rand": "#aec7e8",
          "dist_aware": "#ff9896", "dist_aware_agent": "#d62728",
          "dist_aware_agent_robust3": "#ff7f0e", "oracle": "#2ca02c",
          "fa_ski_rental": "#17becf", "fa_ski_rental_rand": "#9edae5", "fa_dist_aware_agent": "#8c1c13",
          "fa_dist_aware_agent_robust3": "#e377c2", "fa_oracle": "#98df8a"}


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--proposed", default=None,
                    help="policy under test (default: fa_dist_aware_agent if present, else dist_aware_agent)")
    args = ap.parse_args()
    global PROPOSED
    os.makedirs(args.out, exist_ok=True)
    rows = [json.loads(l) for l in open(args.results) if l.strip()]

    # data[mode][policy][pool] = {seed: row}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        data[r["mode"]][r["policy"]][round(r["pool_gb"], 3)][r["seed"]] = r
    names = {r["policy"] for r in rows}
    PROPOSED = args.proposed or ("fa_dist_aware_agent" if "fa_dist_aware_agent" in names else "dist_aware_agent")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = {"budget": {}, "cost": {}, "losses": []}
    for mode, metric, ylabel in (("budget", "recompute_tokens", "recomputed prompt tokens"),
                                 ("cost", "total_cost_gpu_s", "memory + recompute cost (GPU-s)")):
        if mode not in data:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.2))
        pols = sorted(data[mode], key=lambda p: list(COLORS).index(p) if p in COLORS else 99)
        for p in pols:
            pools = sorted(data[mode][p])
            m = [ci95([r[metric] for r in data[mode][p][g].values()]) for g in pools]
            ax.errorbar(pools, [x[0] for x in m], yerr=[x[1] for x in m], marker="o", ms=3.5,
                        capsize=2, lw=2.2 if p in (PROPOSED, "oracle", "lru") else 1.1,
                        color=COLORS.get(p), label=p)
            report[mode][p] = {str(g): {"mean": mm, "ci95": cc} for g, (mm, cc) in zip(pools, m)}
        ax.set_xscale("log", base=2)
        ax.set_xlabel("KV pool (GB)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{args.title} {mode} mode".strip())
        ax.legend(fontsize=7, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"{mode}_{metric}.png"), dpi=150)
        plt.close(fig)

        # Paired differences vs LRU and vs the best baseline, per pool.
        if PROPOSED not in data[mode]:
            continue
        for g in sorted(data[mode][PROPOSED]):
            ours = data[mode][PROPOSED][g]
            cands = {b: data[mode][b][g] for b in BASELINES if b in data[mode] and g in data[mode][b]}
            if not cands:
                continue
            best = min(cands, key=lambda b: np.mean([r[metric] for r in cands[b].values()]))
            for name, base in (("lru", cands.get("lru")), ("best_baseline:" + best, cands[best])):
                if base is None:
                    continue
                seeds = sorted(set(ours) & set(base))
                d = [ours[s][metric] - base[s][metric] for s in seeds]
                bm = np.mean([base[s][metric] for s in seeds])
                mean, half = ci95(d)
                rel = 100 * mean / bm if bm else float("nan")
                entry = {"mode": mode, "pool_gb": g, "vs": name, "metric": metric,
                         "diff_mean": mean, "diff_ci95": half, "rel_pct": rel,
                         "significant": abs(mean) > half}
                report.setdefault("paired", []).append(entry)
                if name.startswith("best") and mean > -half:   # not clearly better than the best baseline
                    report["losses"].append(entry)

    # Competitive ratios (cost mode).
    if "cost" in data:
        report["competitive_ratio"] = {
            p: {str(g): ci95([r["competitive_ratio"] for r in data["cost"][p][g].values()])[0]
                for g in sorted(data["cost"][p])} for p in data["cost"]}

    with open(os.path.join(args.out, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nPaired comparison of {PROPOSED} (negative = better), mean ± 95% CI over seeds:")
    for e in report.get("paired", []):
        print(f"  {e['mode']:<6} pool {e['pool_gb']:>6} GB vs {e['vs']:<32} {e['metric']:<17} "
              f"{e['diff_mean']:>+12.2f} ± {e['diff_ci95']:<10.2f} ({e['rel_pct']:+.1f}%)"
              f"{'' if e['significant'] else '  [not significant]'}")
    print(f"\nCells where {PROPOSED} is NOT clearly better than the best baseline: {len(report['losses'])}")
    for e in report["losses"]:
        print(f"  {e['mode']} pool {e['pool_gb']} GB vs {e['vs']}: {e['rel_pct']:+.1f}%")
    print(f"wrote {args.out}/eval_report.json and plots")


if __name__ == "__main__":
    main()
