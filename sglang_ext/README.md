# sglang_ext: idle-agent eviction strategies for SGLang (Phase 4.1)

Registers two radix eviction policies without forking SGLang:

| policy | behaviour |
|---|---|
| `skirental` | Pins a node for τ(L) = R(L)/(c_mem·L) after its last access, then plain LRU. Pinned nodes are evicted only when nothing unpinned is left, earliest deadline first. |
| `distaware` | Pins for a threshold θ(L) from a table fitted offline on the trace (θ = 0: never pin). |

```bash
PYTHONPATH=$PROJ/sglang_ext python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct \
  --port 30000 --enable-cache-report --enable-metrics \
  --radix-eviction-policy skirental \
  --radix-eviction-policy-config '{"b": 5.1e-5, "c": 1.2e-10, "pool_tokens": 300000}'
```

Status: written against SGLang's `mem_cache/evict_policy.py` / `mem_cache/utils.py`
interface (Oct 2026 main). **Not yet run on a GPU.** `slurm/replay_policies.sbatch`
runs it; check the server log for `[forkcache]` errors first. If the installed SGLang
predates the strategy registry, the same `get_priority` logic goes into a subclass of
`RadixCache` instead.

Limitation: the server cannot tell which agent a node belongs to, so `distaware` uses a
threshold per context-length bucket, not the per-agent fit. Per-agent thresholds need a
request-level hint (for example `extra_body` passed through to the scheduler); that is the
next step.
