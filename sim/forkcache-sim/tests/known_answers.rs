//! Hand-made traces with known answers, plus checks of the ski-rental math.

use forkcache_sim::policies::dist_aware::best_threshold;
use forkcache_sim::policies::lru::Lru;
use forkcache_sim::policies::oracle::Oracle;
use forkcache_sim::policies::ski_rental::{sample_threshold_fraction, SkiRental};
use forkcache_sim::policy::PrefillCurve;
use forkcache_sim::sim::{run, Mode, SimConfig};
use forkcache_sim::trace::{Call, Workload};

fn hashes(agent: u32, n: usize) -> Vec<String> {
    // Chained-hash stand-ins: block i of agent a; unique per (agent, i), shared across that agent's calls.
    (0..n).map(|i| format!("a{agent}b{i}")).collect()
}

/// Three agents take turns A,B,C,A,B,C,A,B,C. Each prompt is the agent's own
/// 4-block (64-token) context. The pool holds two agents' contexts.
fn round_robin_3() -> Workload {
    let mut calls = Vec::new();
    let mut last: [Option<u64>; 3] = [None; 3];
    for k in 0..9u64 {
        let agent = (k % 3) as u32;
        let t = k as f64 * 10.0;
        let gap = if k + 3 < 9 { Some(30.0 - 1.0) } else { None };
        calls.push(Call {
            call_id: k, session: 0, agent, round: (k / 3) as u32 + 1,
            parent_call: last[agent as usize], fork_of: None,
            t_start: t, t_end: t + 1.0, prompt_tokens: 64, output_tokens: 0,
            next_call_same_agent_gap_s: gap, block_size: 16, block_hashes: hashes(agent, 4),
        });
        last[agent as usize] = Some(k);
    }
    Workload::new(calls)
}

fn cfg(mode: Mode, pool_tokens: u64) -> SimConfig {
    SimConfig {
        mode, pool_tokens, kv_bytes_per_token: 1,
        curve: PrefillCurve { a: 0.0, b: 1e-3, c: 0.0 },
        concurrency: 1, seed: 0,
    }
}

#[test]
fn lru_thrashes_on_round_robin_and_belady_does_not() {
    let w = round_robin_3();
    let lru = run(&w, &cfg(Mode::Budget, 128), &mut Lru { cost_mode: false });
    let opt = run(&w, &cfg(Mode::Budget, 128), &mut Oracle { cost_mode: false });
    // LRU always evicts exactly the agent that speaks next: zero hits.
    assert_eq!(lru.hit_tokens, 0);
    assert_eq!(lru.recompute_tokens, 6 * 64);
    // Belady keeps the next speaker: returns at calls 3, 5, 7 hit (worked out by hand).
    assert_eq!(opt.hit_tokens, 3 * 64);
    assert_eq!(opt.compulsory_tokens, 3 * 64);
    assert_eq!(lru.overflow_events, 0);
}

#[test]
fn everything_hits_when_pool_is_large() {
    let w = round_robin_3();
    let lru = run(&w, &cfg(Mode::Budget, 10_000), &mut Lru { cost_mode: false });
    assert_eq!(lru.hit_tokens, 6 * 64);
    assert_eq!(lru.recompute_tokens, 0);
}

#[test]
fn cost_mode_ski_rental_matches_hand_computation() {
    // One agent, idle 29 s. rent = c_mem * 64 = 64/pool; buy = 1e-3 * 64.
    // pool = 64_000 tokens -> rent = 1e-3, buy = 0.064 -> tau = 64 s > 29 s: keep; cost = rent*29 = OPT.
    let w = round_robin_3();
    let m = run(&w, &cfg(Mode::Cost, 64_000), &mut SkiRental);
    assert!((m.competitive_ratio - 1.0).abs() < 1e-9, "ratio {}", m.competitive_ratio);
    assert_eq!(m.recompute_tokens, 0);
    // pool = 640 tokens -> rent = 0.1/s, tau = 0.64 s < 29 s: rent tau then buy = 2 * buy, OPT = buy.
    let m = run(&w, &cfg(Mode::Cost, 640), &mut SkiRental);
    assert!((m.competitive_ratio - 2.0).abs() < 1e-6, "ratio {}", m.competitive_ratio);
    assert_eq!(m.recompute_tokens, 6 * 64);
}

#[test]
fn deterministic_threshold_is_2_competitive_on_every_gap() {
    let (rent, buy) = (0.37, 5.0);
    let tau = buy / rent;
    for k in 0..2000 {
        let t = k as f64 * 0.05;
        let cost = rent * t.min(tau) + if t > tau { buy } else { 0.0 };
        let opt = (rent * t).min(buy);
        assert!(cost <= 2.0 * opt + 1e-9, "t={t}: {cost} > 2*{opt}");
    }
}

#[test]
fn randomized_threshold_expected_ratio_is_e_over_e_minus_1() {
    let (rent, buy) = (1.0, 1.0);
    let target = std::f64::consts::E / (std::f64::consts::E - 1.0);
    let n = 200_000;
    for &t in &[0.3f64, 0.7, 1.0, 3.0] {
        let mut sum = 0.0;
        for i in 0..n {
            let x = sample_threshold_fraction((i as f64 + 0.5) / n as f64); // stratified uniform
            let theta = x * buy / rent;
            sum += rent * t.min(theta) + if t > theta { buy } else { 0.0 };
        }
        let ratio = (sum / n as f64) / (rent * t).min(buy);
        assert!((ratio - target).abs() < 1e-3, "t={t}: ratio {ratio} vs {target}");
    }
}

#[test]
fn best_threshold_matches_brute_force() {
    let mut gaps = vec![3.0, 3.5, 4.0, 4.2, 25.0, 30.0, 31.0, 0.5, 4.1, 28.0];
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut pre = vec![0.0];
    for &g in &gaps { pre.push(pre.last().unwrap() + g); }
    for &(rent, buy) in &[(1.0, 2.0), (1.0, 10.0), (0.1, 10.0), (1.0, 100.0)] {
        let (theta, cost) = best_threshold(&gaps, &pre, rent, buy);
        let ec = |th: f64| gaps.iter().map(|&t| rent * t.min(th) + if t > th { buy } else { 0.0 }).sum::<f64>() / gaps.len() as f64;
        assert!((ec(theta) - cost).abs() < 1e-9);
        for k in 0..4000 {
            let th = k as f64 * 0.01;
            assert!(cost <= ec(th) + 1e-9, "rent {rent} buy {buy}: theta {theta} cost {cost} > ec({th}) = {}", ec(th));
        }
    }
}

/// Two agents share a 4-block prefix; each adds 2 private blocks. While B is
/// still running, idle A exclusively holds only its 2 private blocks (32 tokens).
/// Once B is idle and pinned, A's exclusive part is still those 2 blocks.
#[test]
fn exclusive_suffix_counts_only_private_blocks() {
    use forkcache_sim::radix::{Cache, Order};
    use forkcache_sim::trace::BlockMeta;
    // blocks 0-3 shared chain, 4-5 A's suffix (parent 3), 6-7 B's suffix (parent 3)
    let parent = [None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(3), Some(6)];
    let depth = [0, 1, 2, 3, 4, 5, 4, 5];
    let meta: Vec<BlockMeta> = (0..8).map(|i| BlockMeta { parent: parent[i], tokens: 16, depth: depth[i] }).collect();
    let mut cache = Cache::new(meta, Order::Lru, Vec::new());
    let a = [0u32, 1, 2, 3, 4, 5];
    let b = [0u32, 1, 2, 3, 6, 7];
    for &x in a.iter().chain(&b[4..]) {
        cache.insert(x);
    }
    for &x in &b {
        cache.active[x as usize] += 1;
    }
    assert_eq!(cache.exclusive_suffix_tokens(&a), 32);
    // B finishes and is pinned: shared prefix is still held by B's pins.
    for &x in &b {
        cache.active[x as usize] -= 1;
        cache.pins[x as usize] += 1;
    }
    assert_eq!(cache.exclusive_suffix_tokens(&a), 32);
    // B's KV is gone entirely: A now holds the whole chain alone.
    for &x in &b {
        cache.pins[x as usize] -= 1;
    }
    cache.evict(7);
    cache.evict(6);
    assert_eq!(cache.exclusive_suffix_tokens(&a), 96);
    // Marginal recompute of the suffix on top of a cached prefix.
    let curve = PrefillCurve { a: 0.0, b: 1e-3, c: 1e-7 };
    let fa = curve.recompute_suffix(64, 32);
    assert!((fa - (32e-3 + 1e-7 * (96.0f64 * 96.0 - 64.0 * 64.0))).abs() < 1e-12);
    assert!(fa < curve.recompute(96));
}
