"""
trace/run_zage_traced.py — run the 13-agent negotiation system with trace logging.

The agent code in zage/ is used unmodified: this script wraps it from outside.
  * A TracingClient wraps any OpenAI-compatible client (Ollama on a Mac, vLLM or
    SGLang on CARC) and times every chat.completions.create call.
  * NegotiationAgent.generate_proposal and run_simulation are wrapped to tell the
    tracer which session / agent / round each call belongs to.

Mac (Ollama, smallest model):
  uv run python trace/run_zage_traced.py --sessions 10 --rounds 5 \
      --base-url http://localhost:11434/v1 --model qwen2.5-0.5b-32k
CARC (SGLang or vLLM serving Qwen2.5-7B on port 30000):
  python trace/run_zage_traced.py --sessions 10 --rounds 5 \
      --base-url http://127.0.0.1:30000/v1 --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import contextvars
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "zage"))
sys.path.insert(0, os.path.join(ROOT, "trace"))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

from logger import TraceRecorder, finalize   # noqa: E402

_ctx = contextvars.ContextVar("trace_ctx", default=None)


def _num(v, default: float) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v)) if v is not None else None
    return float(m.group()) if m else float(default)


def _sanitize(resp) -> None:
    """Make small-model JSON safe for the app's float() calls (e.g. "87%" -> 87).

    Runs after the call was recorded, so the trace keeps the model's raw output.
    """
    msg = resp.choices[0].message
    try:
        d = json.loads(msg.content or "")
    except (json.JSONDecodeError, TypeError):
        return              # the app has its own fallback for unparseable output
    if not isinstance(d, dict):
        msg.content = "not json"
        return
    for k, default in (("proposed_allocation_maf", 0.0), ("conservation_commitment_pct", 0.0)):
        if k in d:
            d[k] = _num(d[k], default)
    msg.content = json.dumps(d)


class _Completions:
    def __init__(self, inner, recorder, t0):
        self._inner, self._rec, self._t0 = inner, recorder, t0

    def create(self, **kw):
        ctx = _ctx.get()
        t_start = time.perf_counter() - self._t0
        resp = self._inner.create(**kw)
        t_end = time.perf_counter() - self._t0
        if ctx is not None:
            usage = getattr(resp, "usage", None)
            self._rec.record(
                session=ctx["session"], agent=ctx["agent"], round_num=ctx["round"],
                messages=kw["messages"],
                output_text=resp.choices[0].message.content or "",
                t_start=t_start, t_end=t_end,
                server_prompt_tokens=getattr(usage, "prompt_tokens", None),
                server_output_tokens=getattr(usage, "completion_tokens", None),
            )
        _sanitize(resp)
        return resp


class TracingClient:
    def __init__(self, inner, recorder, t0):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(inner.chat.completions, recorder, t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen2.5-0.5b-32k")
    ap.add_argument("--raw", default=os.path.join(ROOT, "trace", "raw_calls.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "trace", "zage_trace.jsonl"))
    ap.add_argument("--first-session", type=int, default=1,
                    help="session id to start at (to extend an existing raw log)")
    args = ap.parse_args()

    from openai import OpenAI
    import agents
    import negotiation

    # Rounds are read from module globals at call time.
    negotiation.NUM_ROUNDS_PER_SIM = args.rounds

    agent_index = {name: i for i, name in enumerate(negotiation.AGENT_ORDER)}
    session_var = {"id": None}

    orig_gen = agents.NegotiationAgent.generate_proposal

    def traced_gen(self, round_number, other_proposals, basin_context=None):
        tok = _ctx.set({"session": session_var["id"], "agent": agent_index[self.name],
                        "round": round_number})
        try:
            proposal = orig_gen(self, round_number, other_proposals, basin_context)
        finally:
            _ctx.reset(tok)
        # Small models (0.5B) sometimes emit null / "4.4 MAF" / missing numeric
        # fields, which crash the downstream rule engine. Coerce them like a
        # schema validator would. Prompts and call timing are unaffected.
        proposal["proposed_allocation_maf"] = _num(
            proposal.get("proposed_allocation_maf"), self.ceiling)
        proposal["conservation_commitment_pct"] = _num(
            proposal.get("conservation_commitment_pct"), 0.0)
        for k in ("shortage_sharing_position", "key_demand", "concession_offered"):
            if not isinstance(proposal.get(k), str):
                proposal[k] = "" if proposal.get(k) is None else json.dumps(proposal.get(k))
        return proposal

    agents.NegotiationAgent.generate_proposal = traced_gen

    rec = TraceRecorder(args.raw)
    t0 = time.perf_counter()
    inner = OpenAI(base_url=args.base_url, api_key="not-needed")
    agents.set_active_client(TracingClient(inner, rec, t0), model_name=args.model)

    meta = {"model": args.model, "base_url": args.base_url, "sessions": args.sessions,
            "rounds": args.rounds, "agents": len(agent_index),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    results = []
    try:
        for s in range(args.first_session, args.first_session + args.sessions):
            session_var["id"] = s
            ts = time.perf_counter()
            results.append(negotiation.run_simulation(s))
            print(f"[trace] session {s} done in {time.perf_counter() - ts:.1f}s", flush=True)
    finally:
        rec.close()
        n = finalize(args.raw, args.out)
        meta["calls"] = n
        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        # Negotiation outcomes contain generated text: keep them local (gitignored).
        os.makedirs(os.path.join(ROOT, "results", "private"), exist_ok=True)
        with open(os.path.join(ROOT, "results", "private", "zage_outcomes.json"), "w") as f:
            json.dump(results, f, default=str)
        print(f"[trace] wrote {n} calls to {args.out}")


if __name__ == "__main__":
    main()
