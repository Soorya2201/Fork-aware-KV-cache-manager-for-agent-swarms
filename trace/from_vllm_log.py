"""
trace/from_vllm_log.py — rebuild a per-call trace from a vLLM server log.

vLLM (with request logging on) writes every request's fully templated prompt and
its received/finished times. This turns such a log into the same trace format as
trace/logger.py (see trace/SCHEMA.md), so the simulator and stats run on it
unchanged. Prompt text is tokenized in memory only; nothing textual is written.

What the log does not contain, and how it is filled:
- Output token counts. Estimated per call as decode_rate * (duration - prompt/prefill_rate),
  clipped to [1, max_tokens]. decode_rate is the median generation throughput vLLM
  reports while one request runs; prefill_rate is solved so the estimated outputs sum
  to the total generated tokens implied by vLLM's own throughput counters. Field
  `output_tokens_estimated: true` marks this.
- Output token ids. Output blocks get unique placeholder ids: they occupy KV but never
  match a later prompt (this app re-summarizes past replies instead of replaying them).
- Sub-second timing. vLLM log timestamps have 1 s resolution.
Agents are identified by their system prompt (index = order of first appearance),
rounds by "Round N" in the last user turn, and a new session starts when the round
number goes back down.

  python trace/from_vllm_log.py --log data/carc/vllm_server.log --out trace/carc_v100_trace.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import statistics
import sys
from datetime import datetime

import numpy as np
import orjson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import BLOCK_SIZE, TOKENIZER_ID, chained_block_hashes, finalize, lcp_len  # noqa: E402

TS = r"(?P<ts>\d\d-\d\d \d\d:\d\d:\d\d)"
RX_RECV = re.compile(TS + r" logger\.py:\d+\] Received request (?P<id>\S+): prompt: (?P<p>.*?), params: SamplingParams\((?P<sp>.*?)\), prompt_token_ids")
RX_FIN = re.compile(TS + r" async_llm_engine\.py:\d+\] Finished request (?P<id>\S+)\.")
RX_MET = re.compile(TS + r" metrics\.py:\d+\] Avg prompt throughput: (?P<pt>[\d.]+) tokens/s, Avg generation throughput: (?P<gt>[\d.]+) tokens/s, Running: (?P<run>\d+) reqs")
RX_ROUND = re.compile(r"Round (\d+)")
RX_MAXTOK = re.compile(r"max_tokens=(\d+)")


def parse_ts(s: str, year: int) -> float:
    return datetime.strptime(f"{year}-{s}", "%Y-%m-%d %H:%M:%S").timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--year", type=int, default=datetime.now().year)
    args = ap.parse_args()

    recv, fin, met = {}, {}, []
    order = []
    with open(args.log, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Received request" in line:
                m = RX_RECV.search(line)
                if m:
                    recv[m["id"]] = (parse_ts(m["ts"], args.year), ast.literal_eval(m["p"]),
                                     int(RX_MAXTOK.search(m["sp"])[1]))
                    order.append(m["id"])
            elif "Finished request" in line:
                m = RX_FIN.search(line)
                if m:
                    fin[m["id"]] = parse_ts(m["ts"], args.year)
            elif "metrics.py" in line:
                m = RX_MET.search(line)
                if m:
                    met.append((parse_ts(m["ts"], args.year), float(m["pt"]), float(m["gt"]), int(m["run"])))
    ids = [i for i in order if i in fin]
    print(f"requests: {len(order)} received, {len(ids)} finished")

    # vLLM's counters: throughput is averaged over the interval since the previous log line.
    tot_prompt = sum(pt * (t - met[k - 1][0]) for k, (t, pt, gt, r) in enumerate(met) if k)
    tot_gen = sum(gt * (t - met[k - 1][0]) for k, (t, pt, gt, r) in enumerate(met) if k)
    decode_rate = statistics.median(gt for (t, pt, gt, r) in met if r == 1 and pt == 0 and gt > 0)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    rows = []
    for rid in ids:
        t0, prompt, maxtok = recv[rid]
        p_ids = tok.encode(prompt, add_special_tokens=False)
        system = prompt.split("<|im_end|>", 1)[0]
        last_user = prompt.rsplit("<|im_start|>user", 1)[-1]
        rm = RX_ROUND.search(last_user)
        rows.append(dict(t0=t0, t1=fin[rid], p_ids=p_ids, system=system,
                         round=int(rm[1]) if rm else None, maxtok=maxtok))
    del recv
    P = np.array([len(r["p_ids"]) for r in rows], float)
    D = np.array([max(r["t1"] - r["t0"], 0.0) for r in rows])
    print(f"prompt tokens: tokenizer {P.sum():,.0f} vs vLLM counters {tot_prompt:,.0f} "
          f"({100 * (P.sum() / tot_prompt - 1):+.1f}%)")

    def est(prefill_rate):
        o = decode_rate * (D - P / prefill_rate)
        return np.clip(np.round(o), 1, [r["maxtok"] for r in rows])
    lo, hi = 100.0, 1e6          # bisection on prefill rate: more prefill speed -> more output
    for _ in range(80):
        mid = (lo * hi) ** 0.5
        if est(mid).sum() < tot_gen:
            lo = mid
        else:
            hi = mid
    prefill_rate = (lo * hi) ** 0.5
    O = est(prefill_rate).astype(int)
    print(f"decode rate {decode_rate:.1f} tok/s (median, 1 running); prefill rate solved {prefill_rate:,.0f} tok/s; "
          f"output tokens est {O.sum():,} vs vLLM counters {tot_gen:,.0f}")

    # agents, sessions, rounds
    agent_of, session, prev_round = {}, 1, None
    t_base = rows[0]["t0"]
    seqs, last_call, out = [], {}, []
    next_placeholder = -1
    for k, r in enumerate(rows):
        a = agent_of.setdefault(r["system"], len(agent_of))
        rnd = r["round"] if r["round"] is not None else (prev_round or 1)
        if prev_round is not None and rnd < prev_round:
            session += 1
        prev_round = rnd
        prompt = np.asarray(r["p_ids"], np.int64)
        outp = np.arange(next_placeholder, next_placeholder - O[k], -1, dtype=np.int64)
        next_placeholder -= int(O[k])
        seq = np.concatenate([prompt, outp])
        parent = last_call.get((session, a))
        best_len, best_id, parent_len = 0, None, 0
        for cid, ids_ in seqs:
            l = lcp_len(prompt, ids_)
            if cid == parent:
                parent_len = l
            if l > best_len:
                best_len, best_id = l, cid
        significant = best_len >= 2 * BLOCK_SIZE and best_len >= 0.25 * len(prompt)
        fork_of = best_id if (best_id is not None and best_id != parent and significant
                              and best_len > parent_len) else None
        out.append({
            "call_id": k, "session": session, "agent": a, "round": rnd,
            "parent_call": parent, "fork_of": fork_of,
            "t_start": round(r["t0"] - t_base, 4), "t_end": round(r["t1"] - t_base, 4),
            "prompt_tokens": len(prompt), "output_tokens": int(O[k]),
            "output_tokens_estimated": True,
            "shared_prefix_tokens": best_len, "shared_with_parent_tokens": parent_len,
            "server_prompt_tokens": None, "server_output_tokens": None,
            "next_call_same_agent_gap_s": None, "waiting_on": None,
            "block_size": BLOCK_SIZE, "block_hashes": chained_block_hashes(seq.tolist()),
        })
        seqs.append((k, seq))
        last_call[(session, a)] = k
    raw = args.out + ".raw"
    with open(raw, "wb") as f:
        for rec in out:
            f.write(orjson.dumps(rec) + b"\n")
    n = finalize(raw, args.out)
    os.remove(raw)
    meta = {"source": os.path.basename(args.log), "calls": n, "sessions": session, "agents": len(agent_of),
            "decode_rate_tok_s": decode_rate, "prefill_rate_tok_s_solved": prefill_rate,
            "prompt_tokens_vs_vllm_counter_pct": 100 * (P.sum() / tot_prompt - 1),
            "output_tokens": "estimated from duration (1 s timestamps)"}
    json.dump(meta, open(args.out.replace(".jsonl", "_meta.json"), "w"), indent=2)
    print(f"wrote {n} calls, {session} sessions, {len(agent_of)} agents -> {args.out}")


if __name__ == "__main__":
    main()
