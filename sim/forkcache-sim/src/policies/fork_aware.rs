//! Fork-aware wrapper (paper/model.md, section 5.3).
//!
//! An idle agent's KV is not one item to keep or evict. Its next prompt forks off
//! the old context at some point K (in this workload the app rebuilds a windowed
//! history every turn, so on median only about half the old context is reused).
//! Everything past K is dead: keeping it pays rent and buys nothing. And of the
//! kept prefix, the part siblings also hold stays cached anyway.
//!
//! The wrapper therefore
//!   1. predicts K from the other sessions (same agent, same round), choosing it by
//!      a newsvendor rule: keep block j iff the expected rent on it is below the
//!      expected recompute it saves,
//!   2. releases the tail past K immediately,
//!   3. hands the inner threshold rule the marginal costs of the kept prefix:
//!      shared part P' (held by siblings), own part K - P',
//!      rent = c_mem (K - P'),  buy = b (K - P') + c (K^2 - P'^2).
//! The inner rule's competitive guarantee then holds for the kept prefix.

use crate::policy::{EvictionPolicy, Hold, IdleCtx, Victim};
use std::collections::HashMap;

/// One observed idle period: (session, agent, round, reused fraction of the context, gap s).
pub type ReuseSample = (u32, u32, u32, f64, f64);

pub enum ForkPoint {
    /// Predict K from other sessions (leave-one-session-out).
    Predict(Vec<ReuseSample>),
    /// Offline: the true K (for the fork-aware oracle).
    Oracle,
}

pub struct ForkAware<P: EvictionPolicy> {
    pub inner: P,
    pub fork: ForkPoint,
    last_keep: usize,
    /// Budget mode: memory has no price, so keep every block the next call might
    /// reuse (the largest fork point seen) and drop only the tail that is dead.
    pub budget_mode: bool,
    fit: HashMap<(u32, u32, u32), (Vec<f64>, f64)>,
}

impl<P: EvictionPolicy> ForkAware<P> {
    pub fn new(inner: P, fork: ForkPoint) -> Self {
        ForkAware { inner, fork, last_keep: usize::MAX, budget_mode: false, fit: HashMap::new() }
    }

    pub fn budget(mut self, on: bool) -> Self {
        self.budget_mode = on;
        self
    }

    /// Kept prefix length in blocks.
    fn fork_blocks(&mut self, ctx: &IdleCtx) -> usize {
        let samples = match &self.fork {
            ForkPoint::Oracle => return ctx.true_reuse_blocks,
            ForkPoint::Predict(s) => s,
        };
        let key = (ctx.session, ctx.agent, ctx.round);
        let (fracs, mean_gap) = self.fit.entry(key).or_insert_with(|| {
            let mine: Vec<&ReuseSample> = samples
                .iter()
                .filter(|x| x.0 != ctx.session && x.1 == ctx.agent && x.2 == ctx.round)
                .collect();
            let f: Vec<f64> = mine.iter().map(|x| x.3).collect();
            let g = if mine.is_empty() { 0.0 } else { mine.iter().map(|x| x.4).sum::<f64>() / mine.len() as f64 };
            (f, g)
        });
        if fracs.is_empty() {
            return ctx.n_blocks; // no history: keep everything
        }
        let n = ctx.n_blocks;
        let bt = ctx.block_tokens;
        let reuse: Vec<f64> = fracs.iter().map(|f| f * n as f64).collect();
        if self.budget_mode {
            let hi = reuse.iter().cloned().fold(0.0, f64::max);
            return ((hi.ceil() as usize) + 1).min(n);
        }
        // Expected cost of keeping k blocks: rent on k blocks for the expected gap,
        // plus recompute of whatever the next call reuses beyond k.
        let cost = |k: usize| -> f64 {
            let kt = k as f64 * bt;
            let rent = ctx.c_mem * kt * *mean_gap;
            let miss: f64 = reuse
                .iter()
                .map(|&r| {
                    let rt = r * bt;
                    if rt > kt { ctx.curve.recompute_suffix(kt as u64, (rt - kt) as u64) } else { 0.0 }
                })
                .sum::<f64>()
                / reuse.len() as f64;
            rent + miss
        };
        let mut cands: Vec<usize> = reuse.iter().map(|r| (r.floor() as usize).min(n)).collect();
        cands.push(0);
        cands.push(n);
        cands.sort_unstable();
        cands.dedup();
        *cands
            .iter()
            .min_by(|&&a, &&b| cost(a).partial_cmp(&cost(b)).unwrap())
            .unwrap()
    }

    fn marginal(&self, ctx: &IdleCtx, keep: usize) -> IdleCtx {
        let mut m = ctx.clone();
        let l = ctx.tokens as f64;
        let k = (keep as f64 * ctx.block_tokens).min(l);
        // exclusive part is a suffix of the whole context; shared part of the kept prefix:
        let shared = k.min(l - ctx.excl_tokens as f64).max(0.0);
        let own = k - shared;
        m.rent = ctx.c_mem * own;
        m.buy = ctx.curve.recompute_suffix(shared as u64, own as u64);
        m
    }
}

impl<P: EvictionPolicy> EvictionPolicy for ForkAware<P> {
    fn name(&self) -> String {
        match self.fork {
            ForkPoint::Predict(_) => format!("fa_{}", self.inner.name()),
            ForkPoint::Oracle => format!("fa_{}_true_fork", self.inner.name()),
        }
    }
    fn on_idle(&mut self, ctx: &IdleCtx) -> Hold {
        let keep = self.fork_blocks(ctx);
        self.last_keep = keep;
        if keep == 0 {
            return Hold::Release;
        }
        let m = self.marginal(ctx, keep);
        self.inner.on_idle(&m)
    }
    fn keep_prefix_blocks(&mut self, _ctx: &IdleCtx) -> usize {
        self.last_keep
    }
    fn on_return(&mut self, ctx: &IdleCtx, gap: f64) {
        self.inner.on_return(ctx, gap)
    }
    fn on_pressure(&mut self, victims: &[Victim], now: f64) -> Option<usize> {
        self.inner.on_pressure(victims, now)
    }
    fn belady(&self) -> bool {
        self.inner.belady()
    }
    fn wants_true_gap(&self) -> bool {
        self.inner.wants_true_gap()
    }
    fn on_idle_with_gap(&mut self, ctx: &IdleCtx, gap: f64) -> Hold {
        let keep = self.fork_blocks(ctx);
        self.last_keep = keep;
        if keep == 0 {
            return Hold::Release;
        }
        let m = self.marginal(ctx, keep);
        self.inner.on_idle_with_gap(&m, gap)
    }
}
