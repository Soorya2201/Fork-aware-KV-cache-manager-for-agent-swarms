# CARC runbook

One script submits every GPU job in order. The last job pushes the results to the private
data repo (`Soorya2201/forkcache-data`). Nothing workload-derived goes to this public repo.

```
fc_setup ──┬── fc_prefill ──┐
           ├── fc_replay ───┼── fc_policies ──┐
           ├── fc_kernel    │                 │
           ├── fc_compress  │                 ├── fc_publish  -> private data repo
           └── fc_trace7b ──┴─────────────────┘
```

| Job | What it produces | Time |
|---|---|---|
| fc_setup | uv venv, SGLang, Qwen2.5-7B weights (skipped once done) | ~30 min, once |
| fc_prefill | prefill curve R(L), 512–32K tokens: `results/prefill/<gpu>_qwen7b_curve.json` | ~20 min |
| fc_replay | V100 trace replayed on SGLang: radix off, radix on, radix on with a capped pool | ~3 h |
| fc_policies | LRU vs skirental vs distaware inside SGLang, capped pool | ~3 h |
| fc_kernel | Triton shared-prefix kernel vs FlashInfer cascade | ~15 min |
| fc_compress | INT8/FP8 idle-KV check on 7B | ~30 min |
| fc_trace7b | 20 more 7B sessions of the agent app (for tighter CIs) | ~2 h |
| fc_publish | copies `results/` and job logs to `carc_runs/<time>/` in the data repo, pushes | 1 min |

GPU: SGLang needs an A100, A40, L40S or H100. Current SGLang/torch builds have no V100 kernels.

## One-time setup (login node)

1. Key that can write to the data repo only:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/forkcache_data -N "" -C carc-forkcache
   cat ~/.ssh/forkcache_data.pub      # add it as a deploy key with write access
   cat >> ~/.ssh/config <<'CFG'
   Host github-forkcache
     HostName github.com
     User git
     IdentityFile ~/.ssh/forkcache_data
     IdentitiesOnly yes
   CFG
   chmod 600 ~/.ssh/config
   ```
2. Code and data:
   ```bash
   mkdir -p /scratch1/$USER && cd /scratch1/$USER
   git clone https://github.com/Soorya2201/Fork-aware-KV-cache-manager-for-agent-swarms.git forkcache
   cd forkcache
   git clone git@github-forkcache:Soorya2201/forkcache-data.git private
   bash slurm/run_all.sh            # first run only writes slurm/carc.env; check it
   ```

## Every run

```bash
cd /scratch1/$USER/forkcache && git pull
bash slurm/run_all.sh                     # or SKIP="trace7b kernel" bash slurm/run_all.sh
squeue -u $USER                           # watch; logs in logs/
```

When `fc_publish` finishes, the results are in the data repo under `carc_runs/<time>/`.
If it failed (for example no internet on the CPU node), run `bash slurm/publish.sh` on the login node.

## If something fails

- `logs/fc_<job>_<id>.out` has the error; `logs/sglang_*` has the server log.
- "compute capability 7.0": you got a V100. Set `GPU=a100` (or a40 / l40s) in `slurm/carc.env`.
- Server OOM: lower `--context-length` in `prefill_curve.sbatch`, or use a larger GPU.
- fc_policies "SKIPPED skirental/distaware": the installed SGLang has no pluggable radix eviction;
  see `sglang_ext/README.md`. The LRU run still gives the baseline.
- A job pends for long: `squeue -u $USER --start`; try another GPU type in `slurm/carc.env`.
