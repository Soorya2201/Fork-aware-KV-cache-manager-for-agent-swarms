"""
bench/replay.py — replay a workload trace against a live server (Phase 3.3).

Prompts are rebuilt from the trace's block hashes: every unique block hash maps
to a fixed pseudo-random run of token ids, so calls that shared a prefix in the
original workload share exactly the same token prefix here. Output length is
reproduced with max_new_tokens = output_tokens (ignore_eos where supported).

Timing: sessions start at the offsets in --schedule (written by forkcache-sim
--schedule-out, so the simulator and the replay overlap sessions identically).
Within a session calls run in order; each call waits for its recorded start
time and for the previous call of its session to finish.

CARC, SGLang (run once with --disable-radix-cache and once with the radix cache):
  python bench/replay.py --backend sglang --url http://127.0.0.1:30000 \
      --trace trace/zage_trace.jsonl --schedule results/schedule.json --out results/radix.json
Mac smoke test (Ollama; reports TTFT and prompt_eval_count, no cache report):
  uv run python bench/replay.py --backend ollama --url http://localhost:11434 \
      --model qwen2.5-0.5b-32k --trace trace/zage_trace.jsonl --schedule results/schedule.json \
      --limit-sessions 2 --out results/replay_mac_smoke.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time

import httpx
import numpy as np
import orjson

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
VOCAB_LO, VOCAB_HI = 1000, 150000


def block_tokens(h: str, n: int) -> list[int]:
    seed = int.from_bytes(hashlib.sha256(h.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).integers(VOCAB_LO, VOCAB_HI, size=n).tolist()


def prompt_ids(rec) -> list[int]:
    bs, P, O = rec["block_size"], rec["prompt_tokens"], rec["output_tokens"]
    seq_len, hs = P + O, rec["block_hashes"]
    ids = []
    for i, h in enumerate(hs):
        n = bs if i + 1 < len(hs) else seq_len - bs * (len(hs) - 1)
        ids += block_tokens(h, n)
        if len(ids) >= P:
            break
    return ids[:P]


async def call_sglang(client, url, model, ids, out_toks):
    body = {"input_ids": ids, "stream": True,
            "sampling_params": {"max_new_tokens": max(out_toks, 1), "temperature": 0,
                                "ignore_eos": True}}
    t0 = time.perf_counter()
    ttft, meta = None, {}
    async with client.stream("POST", f"{url}/generate", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            meta = json.loads(line[5:]).get("meta_info", meta)
    return {"ttft_s": ttft, "e2e_s": time.perf_counter() - t0,
            "server_prompt_tokens": meta.get("prompt_tokens"),
            "cached_tokens": meta.get("cached_tokens"),
            "server_output_tokens": meta.get("completion_tokens")}


async def call_openai(client, url, model, ids, out_toks):
    body = {"model": model, "prompt": ids, "max_tokens": max(out_toks, 1), "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True}
    t0 = time.perf_counter()
    ttft, usage = None, {}
    async with client.stream("POST", f"{url}/v1/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            j = json.loads(line[5:])
            if ttft is None and j.get("choices"):
                ttft = time.perf_counter() - t0
            usage = j.get("usage") or usage
    return {"ttft_s": ttft, "e2e_s": time.perf_counter() - t0,
            "server_prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "server_output_tokens": usage.get("completion_tokens")}


async def call_ollama(client, url, model, text, out_toks):
    body = {"model": model, "prompt": text, "raw": True, "stream": True, "keep_alive": "10m",
            "options": {"num_predict": max(out_toks, 1), "temperature": 0}}
    t0 = time.perf_counter()
    ttft, last = None, {}
    async with client.stream("POST", f"{url}/api/generate", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.strip():
                continue
            j = json.loads(line)
            if ttft is None:
                ttft = time.perf_counter() - t0
            last = j
    return {"ttft_s": ttft, "e2e_s": time.perf_counter() - t0,
            # Ollama counts only the prompt tokens it had to evaluate (not cached ones).
            "server_prompt_tokens": last.get("prompt_eval_count"), "cached_tokens": None,
            "server_output_tokens": last.get("eval_count"),
            "prefill_s": (last.get("prompt_eval_duration") or 0) / 1e9}


async def main_async(args):
    with open(args.trace, "rb") as f:
        recs = sorted((orjson.loads(l) for l in f if l.strip()), key=lambda r: r["call_id"])
    sched = json.load(open(args.schedule))
    offsets = {int(k): v for k, v in sched["offsets"].items()}
    sessions = sorted({r["session"] for r in recs}, key=lambda s: offsets[s])
    if args.limit_sessions:
        sessions = sessions[:args.limit_sessions]
    keep = set(sessions)
    recs = [r for r in recs if r["session"] in keep]
    t_base = min(offsets[s] + min(r["t_start"] for r in recs if r["session"] == s) for s in sessions)

    tok = None
    if args.backend == "ollama":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    fn = {"sglang": call_sglang, "openai": call_openai, "ollama": call_ollama}[args.backend]

    results = []
    sem = asyncio.Semaphore(args.max_inflight)
    t0 = time.perf_counter()

    async def run_session(client, s):
        for r in (x for x in recs if x["session"] == s):
            due = (offsets[s] + r["t_start"] - t_base) / args.time_scale
            await asyncio.sleep(max(0.0, due - (time.perf_counter() - t0)))
            ids = prompt_ids(r)
            payload = tok.decode(ids) if tok else ids
            async with sem:
                sent = time.perf_counter() - t0
                res = await fn(client, args.url, args.model, payload, r["output_tokens"])
            res.update({"call_id": r["call_id"], "session": s, "agent": r["agent"],
                        "prompt_tokens": r["prompt_tokens"], "output_tokens": r["output_tokens"],
                        "scheduled_s": due, "sent_s": sent})
            results.append(res)
            if len(results) % 25 == 0:
                print(f"[replay] {len(results)}/{len(recs)} calls", flush=True)

    async with httpx.AsyncClient(timeout=None) as client:
        await asyncio.gather(*(run_session(client, s) for s in sessions))

    results.sort(key=lambda r: r["call_id"])
    ttft = np.array([r["ttft_s"] for r in results if r["ttft_s"] is not None])
    cached = [r["cached_tokens"] for r in results if r["cached_tokens"] is not None]
    wall = time.perf_counter() - t0
    summary = {
        "backend": args.backend, "label": args.label, "calls": len(results), "wall_s": wall,
        "ttft_p50_s": float(np.median(ttft)), "ttft_p90_s": float(np.percentile(ttft, 90)),
        "ttft_p99_s": float(np.percentile(ttft, 99)),
        "throughput_tok_s": sum(r["prompt_tokens"] + r["output_tokens"] for r in results) / wall,
        "cached_token_frac": (sum(cached) / sum(r["prompt_tokens"] for r in results)) if cached else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "calls": results}, f, indent=1)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sglang", "openai", "ollama"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--trace", default=os.path.join(ROOT, "trace", "zage_trace.jsonl"))
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--limit-sessions", type=int, default=0)
    ap.add_argument("--max-inflight", type=int, default=64)
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help=">1 compresses the schedule (e.g. a Mac-speed trace on a faster GPU)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
