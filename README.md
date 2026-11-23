# Fork-aware KV-cache manager for agent swarms

When an LLM agent pauses (for a tool call, or while other agents take their turn), should its
KV cache stay in GPU memory, be compressed, or be evicted, and can that choice come with a
guarantee instead of a guess?

This repository builds the pieces needed to answer that on a real 13-agent workload:

| Part | What it is | Where | Status |
|---|---|---|---|
| 1. Workload trace | Per-call logger for a 13-agent negotiation system: token counts, fork tree, idle gaps, chained block hashes (no text) | `trace/` | Runs on a Mac (Ollama) and on GPU servers |
| 2. Keep-or-evict math | Ski-rental model: 2-competitive threshold, e/(e−1) randomized, distribution-aware threshold, memory pressure, and a fork-aware rule that keeps only the prefix the agent's next call will reuse | `paper/model.md` | Written; bounds checked numerically |
| 3. Rust simulator | Radix-tree prefix cache plus 14 policies (LRU, TTL, ski rental, randomized, Continuum-style, distribution-aware, their fork-aware versions, and two oracles), event-driven replay | `sim/forkcache-sim/` | Built, 7 known-answer tests pass |
| 4. SGLang integration | `skirental` and `distaware` radix eviction strategies, plugged in without forking SGLang | `sglang_ext/` | Logic tested; GPU run pending |
| 5. Kernel | Triton decode attention that reads a shared prefix once per fork group, LSE-merged with per-fork suffixes; benchmark vs FlashInfer cascade | `kernels/` | Math verified on CPU; kernel run pending (needs CUDA) |
| 5. Compression | INT8 / FP8 quantization of idle KV, checked for token agreement, accuracy and loops | `compression/` | Runs on a Mac (Qwen2.5-0.5B) |
| 6. Evaluation | 95% CIs over replay seeds, paired differences vs LRU and the best baseline, explicit list of where the proposed policy loses | `analysis/plot.py` | Runs |

## How the parts fit

```
13-agent app ──► trace/run_zage_traced.py ──► trace.jsonl (hashes only)
                                                 │
            bench/prefill_curve.py ──► R(L) ─────┤
                                                 ▼
                          sim/forkcache-sim (Rust) ── policies ──► analysis/plot.py
                                                 │
            bench/replay.py on SGLang ◄── schedule ┘ ──► analysis/compare_sim_real.py
            sglang_ext/ (same policies inside SGLang)
```

## Run on a Mac (no GPU)

Requirements: [uv](https://docs.astral.sh/uv/), Rust (`rustup`), [Ollama](https://ollama.com).

```bash
uv sync
ollama pull qwen2.5:0.5b
printf 'FROM qwen2.5:0.5b\nPARAMETER num_ctx 32768\n' > /tmp/Modelfile && ollama create qwen2.5-0.5b-32k -f /tmp/Modelfile

# 1. Trace (needs the private agent app in zage/; about 1 h for 20 sessions x 6 rounds)
uv run python trace/run_zage_traced.py --sessions 20 --rounds 6
uv run python analysis/stats.py

# 2. Prefill curve on this machine
uv run python bench/prefill_curve.py --backend ollama --url http://localhost:11434 \
    --model qwen2.5-0.5b-32k --name mac_qwen05b --reps 5

# 3. Simulator: tests, then the policy sweep
cd sim/forkcache-sim && cargo test --release && cd ../..
sim/forkcache-sim/target/release/forkcache-sim --trace trace/zage_trace.jsonl \
    --prefill-curve results/prefill/mac_qwen05b_curve.json --kv-bytes-per-token 12288 \
    --kv-budget-gb 0.25,0.5,1,2,4 --concurrency 10 --seeds 20 --out results/sim/mac.jsonl
uv run python analysis/plot.py --results results/sim/mac.jsonl --out results/sim

# 5. Kernel math and KV compression
uv run python kernels/test_shared_prefix.py
uv run python compression/kv_int8_check.py
```

## Run on GPUs (USC CARC)

See `docs/CARC_RUNBOOK.md`: environment setup, prefill curve, re-recording the trace with
Qwen2.5-7B, replay on stock SGLang, simulator-vs-real check, the SGLang policies, the kernel
benchmark and the 7B compression check, each as one SLURM job.

## Data policy

The trace comes from a system that Marshall stakeholders use. The agent app (`zage/`), the
traces and every derived result are excluded from this repository. They will be released,
anonymized, only after the system's PI approves. The logger never writes prompt or output
text: only token counts, timing, and truncated SHA-256 hashes of token blocks.

## Related work

- Continuum (Li et al., 2025): KV time-to-live for multi-turn agents; `continuum_style` here is
  a reimplementation of the idea, not the authors' code.
- Karlin, Manasse, McGeoch, Owicki (1994): randomized ski rental, e/(e−1).
- Purohit, Svitkina, Kumar (NeurIPS 2018): learning-augmented ski rental (robustness clamp).
- SGLang RadixAttention; FlashInfer cascade attention.
