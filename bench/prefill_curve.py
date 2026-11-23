"""
bench/prefill_curve.py — measure the prefill cost curve R(L) (Phase 3.2).

Sends fresh random prompts of each length with max_tokens=1 (so no prefix can be
reused) and records time-to-first-token. Fits R(L) = a + b L + c L^2 with
non-negative coefficients and writes it in the format forkcache-sim reads.

CARC (SGLang, launched with --disable-radix-cache):
  python bench/prefill_curve.py --backend sglang --url http://127.0.0.1:30000 \
      --name a100_qwen7b --reps 10
Mac (Ollama):
  uv run python bench/prefill_curve.py --backend ollama --url http://localhost:11434 \
      --model qwen2.5-0.5b-32k --name mac_qwen05b --reps 5
"""

from __future__ import annotations

import argparse
import json
import os
import time

import httpx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))


def random_ids(rng, n):
    # Ordinary vocabulary range of Qwen2.5 (special tokens start at 151643).
    return rng.integers(1000, 150000, size=n).tolist()


def measure_sglang(client, url, ids):
    t0 = time.perf_counter()
    r = client.post(f"{url}/generate", json={
        "input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
    r.raise_for_status()
    dt = time.perf_counter() - t0
    meta = r.json().get("meta_info", {})
    return dt, meta.get("prompt_tokens", len(ids)), meta.get("cached_tokens")


def measure_openai(client, url, model, ids):
    t0 = time.perf_counter()
    r = client.post(f"{url}/v1/completions", json={
        "model": model, "prompt": ids, "max_tokens": 1, "temperature": 0})
    r.raise_for_status()
    dt = time.perf_counter() - t0
    u = r.json().get("usage", {})
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return dt, u.get("prompt_tokens", len(ids)), cached


def measure_ollama(client, url, model, text):
    r = client.post(f"{url}/api/generate", json={
        "model": model, "prompt": text, "raw": True, "stream": False,
        "options": {"num_predict": 1, "temperature": 0}, "keep_alive": "10m"})
    r.raise_for_status()
    j = r.json()
    # prompt_eval_duration is the server's own prefill time; more precise than wall clock.
    return j["prompt_eval_duration"] / 1e9, j["prompt_eval_count"], None


def fit_curve(L, R):
    from scipy.optimize import nnls
    L = np.asarray(L, float)
    A = np.stack([np.ones_like(L), L, L * L], axis=1)
    # Relative-error weighting so short prompts are not swamped by long ones.
    w = 1.0 / np.maximum(np.asarray(R, float), 1e-6)
    coef, _ = nnls(A * w[:, None], np.asarray(R, float) * w)
    return {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sglang", "openai", "ollama"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192,16384,32768")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "prefill"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tok = None
    if args.backend == "ollama":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    rng = np.random.default_rng(0)
    rows = []
    with httpx.Client(timeout=600) as client:
        def one(n):
            ids = random_ids(rng, n)
            if args.backend == "sglang":
                return measure_sglang(client, args.url, ids)
            if args.backend == "openai":
                return measure_openai(client, args.url, args.model, ids)
            return measure_ollama(client, args.url, args.model, tok.decode(ids))

        for _ in range(args.warmup):
            one(512)
        for n in [int(x) for x in args.lengths.split(",")]:
            for rep in range(args.reps):
                dt, ptoks, cached = one(n)
                rows.append({"target_len": n, "rep": rep, "prompt_tokens": ptoks,
                             "ttft_s": dt, "cached_tokens": cached})
                print(f"L={n:>6} rep={rep} tokens={ptoks:>6} ttft={dt*1000:8.1f} ms"
                      + (f" cached={cached}" if cached else ""), flush=True)

    by = {}
    for r in rows:
        by.setdefault(r["target_len"], []).append(r)
    table = []
    for n, rs in sorted(by.items()):
        t = np.array([r["ttft_s"] for r in rs])
        table.append({"target_len": n,
                      "prompt_tokens_median": float(np.median([r["prompt_tokens"] for r in rs])),
                      "ttft_median_s": float(np.median(t)), "ttft_p95_s": float(np.percentile(t, 95))})
    curve = fit_curve([r["prompt_tokens"] for r in rows], [r["ttft_s"] for r in rows])
    curve["source"] = {"backend": args.backend, "model": args.model, "name": args.name,
                       "reps": args.reps, "measured": time.strftime("%Y-%m-%d")}
    with open(os.path.join(args.out_dir, f"{args.name}_raw.json"), "w") as f:
        json.dump({"rows": rows, "table": table}, f, indent=2)
    with open(os.path.join(args.out_dir, f"{args.name}_curve.json"), "w") as f:
        json.dump(curve, f, indent=2)
    print("\n  L (target)  tokens  TTFT median   P95")
    for r in table:
        print(f"  {r['target_len']:>9} {r['prompt_tokens_median']:>7.0f} {r['ttft_median_s']*1000:>9.1f} ms"
              f" {r['ttft_p95_s']*1000:>8.1f} ms")
    print(f"fit: R(L) = {curve['a']:.4g} + {curve['b']:.4g} L + {curve['c']:.4g} L^2  (seconds)")


if __name__ == "__main__":
    main()
