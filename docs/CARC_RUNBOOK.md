# CARC runbook: Phases 3–5 on real GPUs

Everything below runs from your Mac terminal or a CARC login shell. Nothing here
publishes data. The trace and the negotiation app are copied privately with `rsync`
and never go to GitHub.

## 0. Before you start

- Get the Zage PI's yes for using the logs (draft: `docs/permission_email.md`), and confirm
  which CARC allocation you may use. Use the WiDES allocation only if that PI agrees.
- On the Mac, the Mac-side results should already exist (see `README.md`, "Run on a Mac").

## 1. Log in and get the code onto CARC

```bash
ssh <netid>@discovery.usc.edu                 # Duo 2FA
myaccount                                     # note your account, e.g. <pi>_123
mkdir -p /scratch1/$USER && cd /scratch1/$USER
git clone https://github.com/Soorya2201/Fork-aware-KV-cache-manager-for-agent-swarms.git forkcache
cd forkcache && mkdir -p logs results
cp slurm/carc.env.example slurm/carc.env && nano slurm/carc.env   # set ACCOUNT, GPU
sinfo -p gpu -o "%G" | sort -u                # which GPU types exist right now
```

From the **Mac**, copy the private pieces (trace, schedule, and the app for re-tracing):

```bash
cd /Volumes/Soorya_1_TB/Projects/Multi_Agent_LLM_Serving
rsync -av trace/zage_trace.jsonl results/schedule.json <netid>@discovery.usc.edu:/scratch1/<netid>/forkcache/trace_mac/
rsync -av zage/ <netid>@discovery.usc.edu:/scratch1/<netid>/forkcache/zage/
```

## 2. One-time environment (inside a GPU allocation, about 20–30 min)

```bash
salloc --partition=gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=1:00:00 --account=<pi>_123
cd /scratch1/$USER/forkcache
bash slurm/setup_env.sh            # uv venv, sglang[all], deps, downloads Qwen2.5-7B to $HF_HOME
python sglang_ext/test_strategy.py # should say "base class: real SGLang"
exit
```

## 3. Jobs, in order

All are submitted from `/scratch1/$USER/forkcache`; `squeue -u $USER` shows them,
and `tail -f logs/<name>_<jobid>.out` follows one.

| # | Command | Gives you | Time |
|---|---|---|---|
| 3.1 | `bash slurm/submit.sh slurm/prefill_curve.sbatch` | R(L) at 512–32K tokens: `results/prefill/<gpu>_qwen7b_curve.json` | ~20 min |
| 3.2 | `bash slurm/submit.sh slurm/trace_7b.sbatch` | The 13-agent trace re-recorded on the GPU with Qwen2.5-7B: `trace/zage_trace_7b.jsonl` (real GPU idle gaps) | ~1–2 h |
| 3.3 | schedule + `replay.sbatch` (below) | Stock SGLang with the radix cache off and on: TTFT, cached tokens, throughput, KV pool size | ~1 h |
| 3.4 | `compare_sim_real.py` (below) | Result #3: simulator vs real SGLang cached tokens | 1 min |
| 4.1 | `replay_policies.sbatch` (below) | LRU vs ski-rental vs distribution-aware *inside SGLang* | ~2 h |
| 5.1 | `bash slurm/submit.sh slurm/kernel.sbatch` | Triton kernel correctness + benchmark vs FlashInfer cascade | ~15 min |
| 5.2 | `bash slurm/submit.sh slurm/compression.sbatch` | INT8/FP8 idle-KV agreement at 7B | ~30 min |

### 3.3 Replay on stock SGLang

The simulator writes the schedule so the replay overlaps sessions exactly as simulated.
Build the simulator on CARC once (`curl https://sh.rustup.rs -sSf | sh -s -- -y`), or run this
step on the Mac and `rsync` the file over:

```bash
cd sim/forkcache-sim && cargo build --release && cd ../..
CURVE=results/prefill/a100_qwen7b_curve.json
sim/forkcache-sim/target/release/forkcache-sim --trace trace/zage_trace_7b.jsonl \
  --prefill-curve $CURVE --policy lru --mode budget --kv-budget-gb 20 --seeds 1 \
  --concurrency 10 --schedule-out results/schedule.json --out /dev/null
TRACE=trace/zage_trace_7b.jsonl bash slurm/submit.sh slurm/replay.sbatch --export=ALL
cat results/kv_pool_<jobid>.txt        # max_total_num_tokens = the server's KV pool
```

### 3.4 Simulator vs real (result #3)

```bash
POOL=<max_total_num_tokens>; GB=$(python -c "print($POOL*57344/2**30)")
sim/forkcache-sim/target/release/forkcache-sim --trace trace/zage_trace_7b.jsonl \
  --prefill-curve $CURVE --policy lru --mode budget --kv-budget-gb $GB --seeds 1 \
  --concurrency 10 --per-call-out results/sim_per_call.jsonl --out results/sim_check.jsonl
python analysis/compare_sim_real.py --real results/radix_<jobid>.json \
  --sim results/sim_per_call.jsonl --policy lru --pool-gb $GB
```

The target is aggregate error within about 10–15%. If the real cache hit rate is far higher than
the simulator's, the pool is too big for any pressure: lower `MEM_FRACTION` in 4.1 or raise
`--concurrency`.

### 4.1 Policies inside SGLang

```bash
python analysis/fit_thresholds.py --trace trace/zage_trace_7b.jsonl --curve $CURVE \
  --pool-tokens $POOL --out results/thresholds.json
CURVE=$CURVE POOL_TOKENS=$POOL THRESHOLDS=results/thresholds.json TRACE=trace/zage_trace_7b.jsonl \
  MEM_FRACTION=0.6 bash slurm/submit.sh slurm/replay_policies.sbatch --export=ALL
```

### Full simulator sweep with GPU numbers (CPU only, seconds)

```bash
sim/forkcache-sim/target/release/forkcache-sim --trace trace/zage_trace_7b.jsonl \
  --prefill-curve $CURVE --kv-budget-gb 5,10,20,40 --kv-bytes-per-token 57344 \
  --concurrency 10 --seeds 20 --out results/sim_7b.jsonl
python analysis/plot.py --results results/sim_7b.jsonl --out results/sim_7b --title "A100 / Qwen2.5-7B"
```

## 4. Bring results back to the Mac

```bash
rsync -av <netid>@discovery.usc.edu:/scratch1/<netid>/forkcache/results/ \
  /Volumes/Soorya_1_TB/Projects/Multi_Agent_LLM_Serving/results/carc/
```

## If something fails

- `sglang.launch_server` dies: read `logs/sglang_*_<jobid>.log`. OOM means a smaller
  `--context-length` or a lower `MEM_FRACTION`. "no kernel image" means a GPU too old for the
  installed wheel (V100): use a100/a40/l40s.
- `[forkcache] eviction strategies not registered`: the installed SGLang has no strategy
  registry; see `sglang_ext/README.md`.
- The job pends for a long time: `squeue -u $USER --start`; try another GPU type in `carc.env`.
