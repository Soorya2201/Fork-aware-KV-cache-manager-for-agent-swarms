//! Event-driven replay of a trace through the prefix cache under one policy.
//!
//! Two modes:
//!  * Budget: a fixed KV pool. When a policy releases an idle agent its blocks
//!    stay cached but unprotected (LRU-evictable); pins only change who is evicted
//!    under pressure. Headline metric: recompute tokens at a given pool size.
//!  * Cost: no capacity limit. Released KV is freed at once and memory is rented
//!    at c_mem per token-second. Metric: memory GPU-s + recompute GPU-s, and the
//!    per-idle-period competitive ratio against the offline optimum.
//!
//! Sessions are re-timed so that `concurrency` of them overlap; each seed draws
//! a different session order and start jitter. Within a session the recorded
//! timing is kept as is (a cache simulator, not a queueing simulator).

use crate::policy::{EvictionPolicy, Hold, IdleCtx, PrefillCurve, Victim};
use crate::radix::{Cache, Order};
use crate::trace::Workload;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Budget,
    Cost,
}

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub mode: Mode,
    /// KV pool size in tokens (capacity in budget mode; sets c_mem in both modes).
    pub pool_tokens: u64,
    pub kv_bytes_per_token: u64,
    pub curve: PrefillCurve,
    pub concurrency: usize,
    pub seed: u64,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct Metrics {
    pub policy: String,
    pub mode: String,
    pub pool_gb: f64,
    pub seed: u64,
    pub concurrency: usize,
    pub calls: u64,
    pub prompt_tokens: u64,
    pub hit_tokens: u64,
    pub computed_tokens: u64,
    /// Prompt tokens recomputed although their KV had been computed before.
    pub recompute_tokens: u64,
    /// Prompt tokens never seen before (no policy can avoid these).
    pub compulsory_tokens: u64,
    pub hit_rate: f64,
    pub peak_kv_gb: f64,
    pub mean_kv_gb: f64,
    pub prefill_gpu_s: f64,
    pub mem_cost_gpu_s: f64,
    pub total_cost_gpu_s: f64,
    pub idle_episodes: u64,
    pub episode_cost: f64,
    pub episode_opt: f64,
    pub competitive_ratio: f64,
    pub worst_episode_ratio: f64,
    pub pressure_unpins: u64,
    pub overflow_events: u64,
    pub duration_s: f64,
    /// (call_id, prompt tokens served from cache) for every call.
    #[serde(skip)]
    pub per_call_hits: Vec<(u64, u64)>,
}

struct IdleAgent {
    blocks: Vec<u32>,
    pinned: bool,
    deadline: f64,
    generation: u64,
    ctx: IdleCtx,
}

const END: u8 = 0;
const EXPIRE: u8 = 1;
const START: u8 = 2;

fn us(t: f64) -> i64 {
    (t * 1e6).round() as i64
}

/// Absolute start offset of every session for this seed.
pub fn session_offsets(w: &Workload, concurrency: usize, seed: u64) -> HashMap<u32, f64> {
    let mut first: HashMap<u32, f64> = HashMap::new();
    let mut last: HashMap<u32, f64> = HashMap::new();
    for c in &w.calls {
        let f = first.entry(c.session).or_insert(c.t_start);
        *f = f.min(c.t_start);
        let l = last.entry(c.session).or_insert(c.t_end);
        *l = l.max(c.t_end);
    }
    let mut sessions = w.sessions();
    let dur: f64 = sessions.iter().map(|s| last[s] - first[s]).sum::<f64>() / sessions.len() as f64;
    let slot = dur / concurrency.max(1) as f64;
    let mut rng = StdRng::seed_from_u64(seed);
    sessions.shuffle(&mut rng);
    sessions
        .iter()
        .enumerate()
        .map(|(k, &s)| {
            let jitter = rng.r#gen::<f64>() * 0.5 * slot;
            (s, k as f64 * slot + jitter - first[&s])
        })
        .collect()
}

pub fn run(w: &Workload, cfg: &SimConfig, policy: &mut dyn EvictionPolicy) -> Metrics {
    let offsets = session_offsets(w, cfg.concurrency, cfg.seed);
    let t_of = |i: usize, t: f64| t + offsets[&w.calls[i].session];

    // Belady needs each block's future use times (as a cacheable prompt block).
    let belady = policy.belady();
    let mut uses: Vec<Vec<i64>> = if belady { vec![Vec::new(); w.blocks.len()] } else { Vec::new() };
    if belady {
        for (i, c) in w.calls.iter().enumerate() {
            let t = us(t_of(i, c.t_start));
            for &b in &w.seq_blocks[i][..w.n_prompt_blocks[i]] {
                uses[b as usize].push(t);
            }
        }
        for u in uses.iter_mut() {
            u.sort();
        }
    }
    // Blocks of each call's context that the same agent's next call reuses (prefix match).
    let true_reuse = reuse_blocks(w);
    let mut cache = Cache::new(w.blocks.clone(), if belady { Order::Belady } else { Order::Lru }, uses);
    let c_mem = 1.0 / cfg.pool_tokens as f64; // GPU-s per token-second: the whole pool for 1 s = 1 GPU-s
    let cap = if cfg.mode == Mode::Budget { Some(cfg.pool_tokens) } else { None };

    let mut heap: BinaryHeap<Reverse<(i64, u8, u64, u64)>> = BinaryHeap::new();
    for (i, c) in w.calls.iter().enumerate() {
        heap.push(Reverse((us(t_of(i, c.t_start)), START, i as u64, 0)));
        heap.push(Reverse((us(t_of(i, c.t_end)), END, i as u64, 0)));
    }

    let mut idle: HashMap<(u32, u32), IdleAgent> = HashMap::new();
    let mut gen_counter: u64 = 0;
    let mut seen = vec![false; w.blocks.len()];
    let mut m = Metrics {
        policy: policy.name(),
        mode: format!("{:?}", cfg.mode).to_lowercase(),
        pool_gb: (cfg.pool_tokens * cfg.kv_bytes_per_token) as f64 / (1u64 << 30) as f64,
        seed: cfg.seed,
        concurrency: cfg.concurrency,
        ..Default::default()
    };
    let (mut last_t, mut first_t) = (f64::NAN, f64::NAN);
    let mut token_seconds = 0.0;
    let mut peak_tokens: u64 = 0;
    let gb = |tok: f64| tok * cfg.kv_bytes_per_token as f64 / (1u64 << 30) as f64;

    while let Some(Reverse((t_us, kind, a, b))) = heap.pop() {
        let t = t_us as f64 / 1e6;
        if first_t.is_nan() {
            first_t = t;
        }
        if !last_t.is_nan() {
            token_seconds += cache.cached_tokens as f64 * (t - last_t);
        }
        last_t = t;
        cache.now = t_us;

        match kind {
            START => {
                let i = a as usize;
                let c = &w.calls[i];
                let seq = &w.seq_blocks[i];
                let key = (c.session, c.agent);
                let returning = idle.remove(&key);
                if let Some(ia) = &returning {
                    policy.on_return(&ia.ctx, t - ia.ctx.now);
                }

                // Prefix hits: leading cacheable prompt blocks already cached.
                let np = w.n_prompt_blocks[i];
                let mut hits = 0;
                while hits < np && cache.is_cached(seq[hits]) {
                    hits += 1;
                }
                let hit_tokens: u64 = seq[..hits].iter().map(|&x| w.blocks[x as usize].tokens as u64).sum();
                for &x in &seq[hits..np] {
                    let tk = w.blocks[x as usize].tokens as u64;
                    if seen[x as usize] { m.recompute_tokens += tk } else { m.compulsory_tokens += tk }
                }
                // Prompt tail shorter than a block is always computed; count it as compulsory.
                let tail = c.prompt_tokens as u64
                    - seq[..np].iter().map(|&x| w.blocks[x as usize].tokens as u64).sum::<u64>();
                m.compulsory_tokens += tail;

                // Protect what we hit, make room for the rest, then insert it.
                let cached_len = seq.iter().take_while(|&&x| cache.is_cached(x)).count();
                for &x in &seq[..cached_len] {
                    cache.active[x as usize] += 1;
                    cache.touch(x);
                }
                // The returning agent's own pins are no longer needed: what it reuses is
                // active now; the rest of its old context (e.g. output not replayed) is not.
                if let Some(ia) = returning {
                    if ia.pinned {
                        for &x in &ia.blocks {
                            cache.pins[x as usize] -= 1;
                            cache.refresh(x);
                        }
                    }
                    if cfg.mode == Mode::Cost {
                        cache.free_unreferenced(&ia.blocks);
                    }
                }
                let need: u64 = seq[cached_len..].iter().map(|&x| w.blocks[x as usize].tokens as u64).sum();
                if let Some(cap) = cap {
                    while cache.cached_tokens + need > cap {
                        if cache.evict_one() {
                            continue;
                        }
                        let keys: Vec<(u32, u32)> = idle.iter().filter(|(_, v)| v.pinned).map(|(k, _)| *k).collect();
                        if keys.is_empty() {
                            m.overflow_events += 1;
                            break;
                        }
                        let victims: Vec<Victim> = keys
                            .iter()
                            .map(|k| {
                                let v = &idle[k];
                                Victim { session: k.0, agent: k.1, tokens: v.ctx.tokens,
                                         idle_since: v.ctx.now, deadline: v.deadline, buy: v.ctx.buy }
                            })
                            .collect();
                        let pick = policy.on_pressure(&victims, t).unwrap_or(0);
                        let v = idle.get_mut(&keys[pick]).unwrap();
                        v.pinned = false;
                        for &x in &v.blocks {
                            cache.pins[x as usize] -= 1;
                            cache.refresh(x);
                        }
                        m.pressure_unpins += 1;
                    }
                }
                for &x in &seq[cached_len..] {
                    cache.insert(x);
                    cache.active[x as usize] += 1;
                    cache.refresh(x);
                }
                for &x in seq {
                    seen[x as usize] = true;
                }
                peak_tokens = peak_tokens.max(cache.cached_tokens);
                m.calls += 1;
                m.prompt_tokens += c.prompt_tokens as u64;
                m.hit_tokens += hit_tokens;
                m.per_call_hits.push((c.call_id, hit_tokens));
                m.computed_tokens += c.prompt_tokens as u64 - hit_tokens;
                m.prefill_gpu_s += cfg.curve.prefill(c.prompt_tokens as u64, hit_tokens);
            }
            END => {
                let i = a as usize;
                let c = &w.calls[i];
                let seq = w.seq_blocks[i].clone();
                for &x in &seq {
                    cache.active[x as usize] -= 1;
                    cache.touch(x);
                }
                match c.next_call_same_agent_gap_s {
                    None => {
                        if cfg.mode == Mode::Cost {
                            cache.free_unreferenced(&seq);
                        }
                    }
                    Some(gap) => {
                        let tokens = (c.prompt_tokens + c.output_tokens) as u64;
                        let excl = cache.exclusive_suffix_tokens(&seq).min(tokens);
                        let ctx = IdleCtx {
                            now: t, session: c.session, agent: c.agent, round: c.round, tokens,
                            rent: c_mem * tokens as f64,
                            buy: cfg.curve.recompute(tokens),
                            excl_tokens: excl,
                            fa_rent: c_mem * excl as f64,
                            fa_buy: cfg.curve.recompute_suffix(tokens - excl, excl),
                            n_blocks: seq.len(),
                            block_tokens: w.block_size as f64,
                            c_mem,
                            curve: cfg.curve,
                            true_reuse_blocks: true_reuse[i],
                        };
                        let hold = if policy.wants_true_gap() {
                            policy.on_idle_with_gap(&ctx, gap)
                        } else {
                            policy.on_idle(&ctx)
                        };
                        let theta = match hold {
                            Hold::Release => 0.0,
                            Hold::PinUntil(d) => (d - t).max(0.0),
                            Hold::PinForever => f64::INFINITY,
                        };
                        let cost = ctx.rent * gap.min(theta) + if gap > theta { ctx.buy } else { 0.0 };
                        let opt = (ctx.rent * gap).min(ctx.buy);
                        m.idle_episodes += 1;
                        m.episode_cost += cost;
                        m.episode_opt += opt;
                        if opt > 0.0 {
                            m.worst_episode_ratio = m.worst_episode_ratio.max(cost / opt);
                        }
                        gen_counter += 1;
                        let (pinned, deadline) = match hold {
                            Hold::Release => (false, t),
                            Hold::PinUntil(d) => (true, d),
                            Hold::PinForever => (true, f64::INFINITY),
                        };
                        let keep = if pinned { policy.keep_prefix_blocks(&ctx).min(seq.len()) } else { seq.len() };
                        let seq: Vec<u32> = if pinned && keep < seq.len() {
                            if cfg.mode == Mode::Cost {
                                // drop the dead tail now (pins on the prefix keep it alive)
                                for &x in &seq[..keep] {
                                    cache.pins[x as usize] += 1;
                                }
                                cache.free_unreferenced(&seq);
                                for &x in &seq[..keep] {
                                    cache.pins[x as usize] -= 1;
                                }
                            }
                            seq[..keep].to_vec()
                        } else {
                            seq
                        };
                        if pinned {
                            for &x in &seq {
                                cache.pins[x as usize] += 1;
                                cache.refresh(x);
                            }
                            if deadline.is_finite() {
                                heap.push(Reverse((us(deadline), EXPIRE, i as u64, gen_counter)));
                            }
                        } else if cfg.mode == Mode::Cost {
                            cache.free_unreferenced(&seq);
                        }
                        idle.insert((c.session, c.agent),
                                    IdleAgent { blocks: seq, pinned, deadline, generation: gen_counter, ctx });
                    }
                }
            }
            EXPIRE => {
                let c = &w.calls[a as usize];
                if let Some(v) = idle.get_mut(&(c.session, c.agent)) {
                    if v.generation == b && v.pinned {
                        v.pinned = false;
                        for &x in &v.blocks {
                            cache.pins[x as usize] -= 1;
                            cache.refresh(x);
                        }
                        if cfg.mode == Mode::Cost {
                            let blocks = v.blocks.clone();
                            cache.free_unreferenced(&blocks);
                        }
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    m.duration_s = last_t - first_t;
    m.hit_rate = m.hit_tokens as f64 / m.prompt_tokens.max(1) as f64;
    m.peak_kv_gb = gb(peak_tokens as f64);
    m.mean_kv_gb = gb(token_seconds / m.duration_s.max(1e-9));
    m.mem_cost_gpu_s = c_mem * token_seconds;
    m.total_cost_gpu_s = m.mem_cost_gpu_s + m.prefill_gpu_s;
    m.competitive_ratio = if m.episode_opt > 0.0 { m.episode_cost / m.episode_opt } else { f64::NAN };
    m
}

/// For each call: leading blocks of its sequence that the same agent's next call
/// has as its prompt prefix (0 for an agent's last call).
pub fn reuse_blocks(w: &Workload) -> Vec<usize> {
    let idx: HashMap<u64, usize> = w.calls.iter().enumerate().map(|(i, c)| (c.call_id, i)).collect();
    let mut out = vec![0usize; w.calls.len()];
    for (j, c) in w.calls.iter().enumerate() {
        if let Some(p) = c.parent_call.and_then(|p| idx.get(&p).copied()) {
            let (a, b) = (&w.seq_blocks[p], &w.seq_blocks[j][..w.n_prompt_blocks[j]]);
            out[p] = a.iter().zip(b).take_while(|(x, y)| x == y).count();
        }
    }
    out
}
