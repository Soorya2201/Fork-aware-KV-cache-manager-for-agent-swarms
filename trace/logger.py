"""
trace/logger.py — anonymized per-call trace recorder for multi-agent LLM workloads.

For every LLM call it records timing, token counts, the fork/parent structure
and prefix-sharing statistics. Prompt and output text never touch disk: token
IDs are kept in memory only long enough to compute exact shared-prefix lengths
and chained SHA-256 block hashes (the same keying vLLM/SGLang use for prefix
caching). The hashes let a simulator or replay benchmark reproduce exactly
which KV blocks are shared, without the text.

Output: one JSON object per line. See trace/SCHEMA.md for every field.
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass, field

import numpy as np
import orjson

BLOCK_SIZE = 16           # tokens per KV block (SGLang page / vLLM block granularity)
HASH_HEX_CHARS = 16       # 64-bit truncated SHA-256 per block: collision-safe at this scale
TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"   # same vocab/template across all Qwen2.5 sizes


def chained_block_hashes(token_ids: list[int], block_size: int = BLOCK_SIZE) -> list[str]:
    """Hash each block together with the hash of everything before it.

    Block i's hash depends on blocks 0..i, so two sequences share block i's hash
    iff they share the entire token prefix up to the end of block i. The final
    block may be partial; its hash includes its length so a partial block never
    collides with a full one.
    """
    out, prev = [], b""
    for start in range(0, len(token_ids), block_size):
        blk = token_ids[start:start + block_size]
        h = hashlib.sha256()
        h.update(prev)
        h.update(len(blk).to_bytes(2, "little"))
        h.update(np.asarray(blk, dtype=np.int32).tobytes())
        prev = h.digest()
        out.append(prev.hex()[:HASH_HEX_CHARS])
    return out


def lcp_len(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the longest common prefix of two int arrays."""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    neq = np.nonzero(a[:n] != b[:n])[0]
    return int(neq[0]) if len(neq) else n


@dataclass
class _Seq:
    call_id: int
    session: int
    agent: int
    ids: np.ndarray            # prompt + output token ids (in memory only)


@dataclass
class TraceRecorder:
    path: str
    tokenizer_id: str = TOKENIZER_ID
    _seqs: list[_Seq] = field(default_factory=list)
    _last_call: dict = field(default_factory=dict)   # (session, agent) -> call_id
    _next_id: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        from transformers import AutoTokenizer   # tokenizer only; no torch needed
        self.tok = AutoTokenizer.from_pretrained(self.tokenizer_id)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._fh = open(self.path, "ab")

    def record(self, *, session: int, agent: int, round_num: int, messages: list[dict],
               output_text: str, t_start: float, t_end: float,
               server_prompt_tokens: int | None = None,
               server_output_tokens: int | None = None) -> dict:
        prompt_ids = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                                  tokenize=True)
        if hasattr(prompt_ids, "input_ids"):       # newer transformers return BatchEncoding
            prompt_ids = prompt_ids["input_ids"]
        prompt_ids = list(prompt_ids)
        out_ids = self.tok.encode(output_text or "", add_special_tokens=False)
        seq = np.asarray(prompt_ids + out_ids, dtype=np.int32)
        prompt = seq[:len(prompt_ids)]

        with self._lock:
            call_id = self._next_id
            self._next_id += 1
            parent = self._last_call.get((session, agent))

            # Longest prefix this prompt shares with anything already in KV:
            # every earlier call's prompt+output sequence.
            best_len, best_id, parent_len = 0, None, 0
            for s in self._seqs:
                l = lcp_len(prompt, s.ids)
                if s.call_id == parent:
                    parent_len = l
                if l > best_len:
                    best_len, best_id = l, s.call_id
            # A fork: the longest shared prefix comes from a different lineage
            # than this agent's own previous call (e.g. another session that
            # started from the same context). A few shared chat-template tokens
            # are not a fork: require >= 2 blocks and >= 25% of the prompt.
            significant = best_len >= 2 * BLOCK_SIZE and best_len >= 0.25 * len(prompt_ids)
            fork_of = best_id if (best_id is not None and best_id != parent and significant
                                  and best_len > parent_len) else None

            rec = {
                "call_id": call_id,
                "session": session,
                "agent": agent,
                "round": round_num,
                "parent_call": parent,
                "fork_of": fork_of,
                "t_start": round(t_start, 4),
                "t_end": round(t_end, 4),
                "prompt_tokens": len(prompt_ids),
                "output_tokens": len(out_ids),
                "shared_prefix_tokens": best_len,
                "shared_with_parent_tokens": parent_len,
                "server_prompt_tokens": server_prompt_tokens,
                "server_output_tokens": server_output_tokens,
                "next_call_same_agent_gap_s": None,     # filled by finalize()
                "waiting_on": None,                      # filled by finalize()
                "block_size": BLOCK_SIZE,
                "block_hashes": chained_block_hashes(seq.tolist()),
            }
            self._seqs.append(_Seq(call_id, session, agent, seq))
            self._last_call[(session, agent)] = call_id
            self._fh.write(orjson.dumps(rec) + b"\n")
            self._fh.flush()
        return rec

    def close(self):
        self._fh.close()


def finalize(raw_path: str, out_path: str, wait_kind: str = "agent") -> int:
    """Fill next_call_same_agent_gap_s and waiting_on from the raw call log.

    gap = (start of this agent's next call in the same session) - (end of this call):
    how long this agent's KV sits idle before it is needed again. The last call of
    each agent in a session has gap None and waiting_on "none".
    wait_kind labels what the idle agent waits on ("agent" for turn-taking systems
    like Mulholland, where an agent waits for the other agents' turns; "tool" when
    a tool call is in flight).
    """
    with open(raw_path, "rb") as f:
        recs = [orjson.loads(l) for l in f if l.strip()]
    recs.sort(key=lambda r: r["call_id"])
    by_id = {r["call_id"]: r for r in recs}
    for r in recs:
        r["next_call_same_agent_gap_s"] = None
        r["waiting_on"] = "none"
    for r in recs:
        p = r["parent_call"]
        if p is not None and p in by_id:
            gap = r["t_start"] - by_id[p]["t_end"]
            by_id[p]["next_call_same_agent_gap_s"] = round(max(gap, 0.0), 4)
            by_id[p]["waiting_on"] = wait_kind
    with open(out_path, "wb") as f:
        for r in recs:
            f.write(orjson.dumps(r) + b"\n")
    return len(recs)
