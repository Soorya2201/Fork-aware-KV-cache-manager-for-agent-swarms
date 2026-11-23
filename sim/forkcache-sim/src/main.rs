//! forkcache-sim CLI.
//!
//!   cargo run --release -- --trace ../../trace/zage_trace.jsonl --policy all \
//!       --kv-budget-gb 5,10,20,40 --prefill-curve ../../results/prefill/mac_curve.json
//!
//! Writes one JSON line per (mode, policy, budget, seed) to --out and prints a
//! summary table (mean over seeds).

use clap::Parser;
use forkcache_sim::policy::PrefillCurve;
use forkcache_sim::sim::{run, Metrics, Mode, SimConfig};
use forkcache_sim::trace::{load, Workload};
use forkcache_sim::{make_policies, ALL_POLICIES};
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::Mutex;

#[derive(Parser, Debug)]
#[command(about = "Replay a multi-agent LLM trace and compare KV keep-or-evict policies")]
struct Args {
    #[arg(long)]
    trace: String,
    /// Comma-separated policy names, or "all".
    #[arg(long, default_value = "all")]
    policy: String,
    /// budget, cost, or both.
    #[arg(long, default_value = "both")]
    mode: String,
    /// KV pool sizes in GB (capacity in budget mode; sets c_mem in cost mode).
    #[arg(long, default_value = "5,10,20,40", value_delimiter = ',')]
    kv_budget_gb: Vec<f64>,
    /// 57344 = Qwen2.5-7B fp16; 12288 = Qwen2.5-0.5B fp16.
    #[arg(long, default_value_t = 57344)]
    kv_bytes_per_token: u64,
    /// JSON file {"a":..,"b":..,"c":..}: prefill GPU-seconds R(L) = a + bL + cL^2.
    #[arg(long)]
    prefill_curve: String,
    /// Sessions overlapping in time during replay.
    #[arg(long, default_value_t = 10)]
    concurrency: usize,
    /// Repeated replays (session order and start jitter) for confidence intervals.
    #[arg(long, default_value_t = 10)]
    seeds: u64,
    #[arg(long, default_value = "10,30,60", value_delimiter = ',')]
    ttl: Vec<f64>,
    #[arg(long, default_value = "sim_results.jsonl")]
    out: String,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Write the replay schedule (session start offsets) for seed 0 here, for bench/replay.py.
    #[arg(long)]
    schedule_out: Option<String>,
    /// Write per-call cache hits (seed 0 runs) here, for analysis/compare_sim_real.py.
    #[arg(long)]
    per_call_out: Option<String>,
}

fn main() {
    let args = Args::parse();
    let calls = load(&args.trace).unwrap_or_else(|e| panic!("{e}"));
    let w = Workload::new(calls);
    let curve: PrefillCurve = serde_json::from_str(
        &std::fs::read_to_string(&args.prefill_curve).expect("read prefill curve"),
    ).expect("parse prefill curve");
    eprintln!("trace: {} calls, {} sessions, {} unique blocks", w.calls.len(), w.sessions().len(), w.blocks.len());

    let modes: Vec<Mode> = match args.mode.as_str() {
        "budget" => vec![Mode::Budget],
        "cost" => vec![Mode::Cost],
        _ => vec![Mode::Budget, Mode::Cost],
    };
    let names: Vec<&str> = if args.policy == "all" { ALL_POLICIES.to_vec() } else { args.policy.split(',').collect() };

    let mut jobs = Vec::new();
    for &mode in &modes {
        for &gb in &args.kv_budget_gb {
            for seed in 0..args.seeds {
                for &n in &names {
                    jobs.push((mode, gb, seed, n));
                }
            }
        }
    }
    let results: Mutex<Vec<Metrics>> = Mutex::new(Vec::new());
    let next = Mutex::new(0usize);
    std::thread::scope(|s| {
        for _ in 0..args.threads.max(1) {
            s.spawn(|| loop {
                let j = {
                    let mut g = next.lock().unwrap();
                    if *g >= jobs.len() { break; }
                    *g += 1;
                    *g - 1
                };
                let (mode, gb, seed, name) = jobs[j];
                let cfg = SimConfig {
                    mode,
                    pool_tokens: ((gb * (1u64 << 30) as f64) / args.kv_bytes_per_token as f64) as u64,
                    kv_bytes_per_token: args.kv_bytes_per_token,
                    curve,
                    concurrency: args.concurrency,
                    seed,
                };
                for mut p in make_policies(name, &w, mode == Mode::Cost, &args.ttl, seed) {
                    let m = run(&w, &cfg, p.as_mut());
                    results.lock().unwrap().push(m);
                }
            });
        }
    });
    let results = results.into_inner().unwrap();

    if let Some(path) = &args.schedule_out {
        let offs = forkcache_sim::sim::session_offsets(&w, args.concurrency, 0);
        let offs: BTreeMap<u32, f64> = offs.into_iter().collect();
        let j = serde_json::json!({"concurrency": args.concurrency, "seed": 0, "offsets": offs});
        std::fs::write(path, serde_json::to_string_pretty(&j).unwrap()).expect("write schedule");
    }
    if let Some(path) = &args.per_call_out {
        let mut f = std::fs::File::create(path).expect("create per-call out");
        for m in results.iter().filter(|m| m.seed == 0) {
            for &(cid, hit) in &m.per_call_hits {
                writeln!(f, "{}", serde_json::json!({"policy": m.policy, "mode": m.mode,
                    "pool_gb": m.pool_gb, "call_id": cid, "hit_tokens": hit})).unwrap();
            }
        }
    }

    let mut f = std::fs::File::create(&args.out).expect("create out");
    for m in &results {
        writeln!(f, "{}", serde_json::to_string(m).unwrap()).unwrap();
    }

    // Summary: mean over seeds.
    let mut agg: BTreeMap<(String, String, u64), Vec<&Metrics>> = BTreeMap::new();
    for m in &results {
        agg.entry((m.mode.clone(), m.policy.clone(), (m.pool_gb * 1000.0) as u64)).or_default().push(m);
    }
    let mean = |v: &Vec<&Metrics>, f: fn(&Metrics) -> f64| v.iter().map(|m| f(m)).sum::<f64>() / v.len() as f64;
    for &mode in &modes {
        let mname = format!("{:?}", mode).to_lowercase();
        println!("\n== mode: {mname} (mean over {} seeds, concurrency {}) ==", args.seeds, args.concurrency);
        println!("{:<28} {:>7} {:>13} {:>8} {:>9} {:>9} {:>11} {:>9} {:>7}",
                 "policy", "pool_GB", "recompute_tok", "hit%", "peakGB", "meanGB", "total_GPUs", "ratio", "unpins");
        for ((md, pol, gbk), v) in &agg {
            if *md != mname { continue; }
            println!("{:<28} {:>7.1} {:>13.0} {:>8.2} {:>9.2} {:>9.2} {:>11.2} {:>9.3} {:>7.0}",
                     pol, *gbk as f64 / 1000.0,
                     mean(v, |m| m.recompute_tokens as f64),
                     100.0 * mean(v, |m| m.hit_rate),
                     mean(v, |m| m.peak_kv_gb),
                     mean(v, |m| m.mean_kv_gb),
                     mean(v, |m| m.total_cost_gpu_s),
                     mean(v, |m| m.competitive_ratio),
                     mean(v, |m| m.pressure_unpins as f64));
        }
    }
    eprintln!("wrote {} runs to {}", results.len(), args.out);
}
