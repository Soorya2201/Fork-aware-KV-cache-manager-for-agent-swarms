"""
compression/kv_int8_check.py — does compressing idle KV change what the model says? (Phase 5.2)

Agent scenario: an agent's history is prefilled, then it goes idle. While idle its
KV is quantized (INT8 per-token-per-head symmetric, or FP8 e4m3) and later
dequantized when the agent returns with a new message. We compare greedy
continuations from the original and the compressed history KV:

  * token agreement: fraction of positions identical, and first-divergence index
  * task accuracy: each prompt has a checkable short answer
  * loops: repeated 4-gram fraction (the failure mode Robin Jia's VaSE paper documents
    for compressed reasoning models)

Runs on a Mac CPU with Qwen2.5-0.5B-Instruct (CUDA on CARC with --device cuda and a
bigger --model).

  uv run python compression/kv_int8_check.py --max-new 160
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import torch  # noqa: E402

# (history shared by the agent, new message on return, regex the answer must match)
TASKS = [
    ("You are the water negotiator for State A. State A holds 4.4 units, State B 2.8 units, State C 0.3 units.",
     "What is the total of the three allocations? Think step by step, then give the number.", r"7\.5"),
    ("A reservoir holds 12 units. Each month 1.5 units flow out and 0.5 units flow in.",
     "After 4 months, how many units are left? Show the arithmetic, then the final number.", r"\b8(\.0)?\b"),
    ("Three agents speak in order Alpha, Beta, Gamma, and the order repeats every round.",
     "Who speaks seventh? Explain briefly, then give the name.", r"Alpha"),
    ("The treaty floor for Mexico is 1.5 units. In a shortage year Mexico's delivery is cut by 5%.",
     "What is the delivery in the shortage year? Compute it, then state the number.", r"1\.425|1\.43"),
    ("An agent's context is 6,000 tokens. Each token of KV cache needs 57,344 bytes.",
     "How many bytes does the context need? Multiply, then give the number.", r"344,?064,?000|344\.064"),
    ("Upper Basin states share 7.5 units: Colorado 51.75%, Utah 23%, Wyoming 14%, New Mexico 11.25%.",
     "How many units does Utah get? Calculate, then give the number.", r"1\.725|1\.73"),
    ("A negotiation has 6 rounds and 13 agents, one call per agent per round.",
     "How many calls are made in total? Then explain in one sentence.", r"\b78\b"),
    ("Arizona proposed 2.8 units, then conceded 10% in round 2 and another 0.1 units in round 3.",
     "What is Arizona's round-3 proposal? Work it out, then give the number.", r"2\.42"),
    ("The idle times observed were 20, 22, 25, 30 and 23 seconds.",
     "What is the median idle time? Sort them first, then answer.", r"\b23\b"),
    ("Recomputing 2,000 tokens costs 0.2 seconds. Keeping them costs 0.01 seconds per second of idling.",
     "After how many seconds of idling is keeping as costly as recomputing? Divide, then answer.", r"\b20\b"),
    ("A tribal nation holds rights to 0.6 units but currently receives 0.25 units.",
     "How large is the shortfall as a percentage of its rights? Compute, then answer.", r"58\.3|58%"),
    ("Five agents each hold 1,200 tokens of KV while idle; the pool fits 4,000 tokens.",
     "How many agents can stay cached at once? Explain, then give the number.", r"\b3\b"),
]


def cache_tensors(cache):
    """Yield (layer, 'k'|'v', tensor) for DynamicCache across transformers versions."""
    if hasattr(cache, "layers"):
        for i, layer in enumerate(cache.layers):
            yield i, "k", layer.keys
            yield i, "v", layer.values
    else:
        for i in range(len(cache.key_cache)):
            yield i, "k", cache.key_cache[i]
            yield i, "v", cache.value_cache[i]


def set_tensor(cache, i, kind, t):
    if hasattr(cache, "layers"):
        if kind == "k":
            cache.layers[i].keys = t
        else:
            cache.layers[i].values = t
    elif kind == "k":
        cache.key_cache[i] = t
    else:
        cache.value_cache[i] = t


def quant_dequant(x: torch.Tensor, scheme: str) -> torch.Tensor:
    """x: [batch, heads, seq, head_dim]. Returns the dequantized tensor (same dtype)."""
    if scheme == "none":
        return x
    xf = x.float()
    if scheme == "int8":
        scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0  # per token, per head
        q = torch.clamp(torch.round(xf / scale), -127, 127).to(torch.int8)
        return (q.float() * scale).to(x.dtype)
    if scheme == "fp8":
        scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0   # e4m3 max
        q = (xf / scale).to(torch.float8_e4m3fn)
        return (q.float() * scale).to(x.dtype)
    raise ValueError(scheme)


def loop_score(ids: list[int], n: int = 4) -> float:
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


@torch.no_grad()
def run_one(model, tok, history, message, scheme, max_new, device):
    import copy
    hist_ids = tok.apply_chat_template([{"role": "system", "content": history}], tokenize=False)
    full = tok.apply_chat_template([{"role": "system", "content": history},
                                    {"role": "user", "content": message}],
                                   tokenize=False, add_generation_prompt=True)
    assert full.startswith(hist_ids)
    h = tok(hist_ids, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    rest = tok(full[len(hist_ids):], return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    out = model(input_ids=h, use_cache=True)          # agent's history: prefilled, then idle
    cache = out.past_key_values
    cache = copy.deepcopy(cache)
    err = 0.0
    for i, kind, t in list(cache_tensors(cache)):       # compress while idle
        qt = quant_dequant(t, scheme)
        err = max(err, float((qt.float() - t.float()).abs().max()))
        set_tensor(cache, i, kind, qt)

    out = model(input_ids=rest, past_key_values=cache, use_cache=True)   # agent returns
    cache, logits = out.past_key_values, out.logits[:, -1]
    gen = []
    for _ in range(max_new):
        nxt = int(logits.argmax(-1))
        gen.append(nxt)
        if nxt == tok.eos_token_id or nxt == tok.convert_tokens_to_ids("<|im_end|>"):
            break
        out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache, use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1]
    return gen, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--schemes", default="int8,fp8")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "compression", "kv_check.json"))
    args = ap.parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 4))   # leave the rest of the Mac free

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()

    rows = []
    t0 = time.time()
    for k, (hist, msg, ans) in enumerate(TASKS):
        base, _ = run_one(model, tok, hist, msg, "none", args.max_new, args.device)
        base_text = tok.decode(base, skip_special_tokens=True)
        row = {"task": k, "baseline_correct": bool(re.search(ans, base_text)),
               "baseline_loop": loop_score(base), "baseline_len": len(base)}
        for s in args.schemes.split(","):
            g, err = run_one(model, tok, hist, msg, s, args.max_new, args.device)
            n = min(len(g), len(base))
            first_div = next((i for i in range(n) if g[i] != base[i]), n)
            text = tok.decode(g, skip_special_tokens=True)
            row[s] = {"identical": g == base, "first_divergence": first_div,
                      "agreement": sum(a == b for a, b in zip(g, base)) / max(len(base), 1),
                      "correct": bool(re.search(ans, text)), "loop": loop_score(g),
                      "len": len(g), "max_abs_kv_err": err}
        rows.append(row)
        print(f"task {k:>2}: base {'ok ' if row['baseline_correct'] else 'bad'} | " + " | ".join(
            f"{s}: {'same' if row[s]['identical'] else 'diverge@' + str(row[s]['first_divergence'])}"
            f" {'ok' if row[s]['correct'] else 'bad'}" for s in args.schemes.split(",")), flush=True)

    summary = {"model": args.model, "dtype": args.dtype, "tasks": len(rows), "max_new": args.max_new,
               "baseline_accuracy": sum(r["baseline_correct"] for r in rows) / len(rows),
               "seconds": time.time() - t0}
    for s in args.schemes.split(","):
        summary[s] = {
            "identical_outputs": sum(r[s]["identical"] for r in rows) / len(rows),
            "mean_token_agreement": sum(r[s]["agreement"] for r in rows) / len(rows),
            "accuracy": sum(r[s]["correct"] for r in rows) / len(rows),
            "answer_flips": sum(r[s]["correct"] != r["baseline_correct"] for r in rows),
            "mean_loop_score": sum(r[s]["loop"] for r in rows) / len(rows),
            "baseline_mean_loop_score": sum(r["baseline_loop"] for r in rows) / len(rows),
        }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
