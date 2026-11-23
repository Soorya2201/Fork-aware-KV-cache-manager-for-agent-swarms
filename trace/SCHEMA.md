# Trace schema (`trace/zage_trace.jsonl`)

One JSON object per LLM call, ordered by `call_id` (= issue order). No text is stored.

| Field | Meaning |
|---|---|
| `call_id` | Global index of the call. |
| `session` | One full negotiation (a Monte Carlo run of all agents over all rounds). |
| `agent` | Agent index 0–12 (order: 7 basin states, federal, 2 tribal, Mexico, IID, environmental). |
| `round` | Negotiation round, 1-based. |
| `parent_call` | This agent's previous call in the same session (its own continuation), or null. |
| `fork_of` | The earlier call from a *different* lineage that this prompt shares the longest prefix with, when that share is ≥ 32 tokens, ≥ 25% of the prompt, and longer than the share with `parent_call`. Typical case: round-1 prompts of session *k* are identical to session 1's. |
| `t_start`, `t_end` | Seconds since the trace started, measured around the HTTP call. |
| `prompt_tokens`, `output_tokens` | Qwen2.5 tokenizer counts (chat template applied). `server_*` are the server's own counts, kept as a cross-check. |
| `shared_prefix_tokens` | Longest token prefix this prompt shares with any earlier call's prompt + output (everything that could be in a KV cache). |
| `shared_with_parent_tokens` | The same, but only against `parent_call`. |
| `next_call_same_agent_gap_s` | Idle time of this call's KV: start of the agent's next call in the session minus this call's end. Null for the last call. |
| `waiting_on` | What the idle agent waits for: `agent` (other agents' turns), `tool`, or `none` (no next call). |
| `block_size` | Tokens per hashed block (16). |
| `block_hashes` | Chained SHA-256 (64-bit truncated) of each 16-token block of prompt + output. Block *i* matches across calls iff the full prefix through block *i* matches. The last block may be partial. |

## Workload notes

The Mulholland system is turn-based: in each round all 13 agents speak in a fixed order, so an
agent's KV is idle while the other 12 speak. It makes no external tool calls (its rule engine runs
synchronously in microseconds), so every idle period is `waiting_on: agent`. Tool waits would need
a tool-using workload; the schema supports them.

Timing depends on the serving hardware. A Mac trace (Ollama, Qwen2.5-0.5B) and a CARC trace
(SGLang, Qwen2.5-7B) share structure and token counts but not gap lengths.

## Traces rebuilt from server logs

`trace/from_vllm_log.py` builds the same format from a vLLM request log. Those traces add
`output_tokens_estimated: true`: output counts come from duration × decode rate, calibrated to
vLLM's counters, and output blocks use placeholder token ids. Timestamps have 1 s resolution.
