//! Keep-or-evict policy interface and the cost model shared by all policies.

/// Prefill cost model R(L) = a + b L + c L^2 GPU-seconds (fitted from TTFT measurements).
#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize)]
pub struct PrefillCurve {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl PrefillCurve {
    /// GPU-seconds to prefill tokens [cached, total) of a prompt.
    pub fn prefill(&self, total: u64, cached: u64) -> f64 {
        let (t, p) = (total as f64, cached.min(total) as f64);
        self.a + self.b * (t - p) + self.c * (t * t - p * p)
    }
    /// Cost to recompute `extra` tokens on top of `cached` tokens already in cache.
    pub fn recompute_suffix(&self, cached: u64, extra: u64) -> f64 {
        let (p, e) = (cached as f64, extra as f64);
        self.b * e + self.c * ((p + e) * (p + e) - p * p)
    }
    /// Recompute cost of an evicted context of L tokens (the per-call overhead `a`
    /// is paid on return either way, so it is not part of the eviction penalty).
    pub fn recompute(&self, l: u64) -> f64 {
        let l = l as f64;
        self.b * l + self.c * l * l
    }
}

/// Everything a policy may know when an agent goes idle.
#[derive(Clone, Debug)]
pub struct IdleCtx {
    pub now: f64,
    pub session: u32,
    pub agent: u32,
    pub round: u32,
    /// Tokens of KV the agent holds (prompt + output).
    pub tokens: u64,
    /// Memory rent while kept: c_mem * L, GPU-seconds per second.
    pub rent: f64,
    /// Recompute cost if evicted: R(L), GPU-seconds.
    pub buy: f64,
    /// Fork-aware marginal costs. Only the exclusive suffix (E tokens) is freed by
    /// evicting this agent; the shared prefix P stays cached while siblings hold it.
    /// fa_rent = c_mem E, fa_buy = b E + c ((P+E)^2 - P^2).
    pub excl_tokens: u64,
    pub fa_rent: f64,
    pub fa_buy: f64,
    /// Blocks in the agent's sequence, tokens per block, memory price, cost curve:
    /// what a policy needs to price keeping only a prefix of the context.
    pub n_blocks: usize,
    pub block_tokens: f64,
    pub c_mem: f64,
    pub curve: PrefillCurve,
    /// Offline only (oracles): blocks the agent's next call actually reuses.
    pub true_reuse_blocks: usize,
}

impl IdleCtx {
    /// Ski-rental break-even time tau = R(L) / (c_mem L).
    pub fn tau(&self) -> f64 {
        if self.rent > 0.0 { self.buy / self.rent } else { f64::INFINITY }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Hold {
    /// Do not protect: unpinned (budget mode) or freed now (cost mode).
    Release,
    /// Protect until this absolute time, then release.
    PinUntil(f64),
    /// Protect until the agent returns.
    PinForever,
}

/// A pinned idle agent that could be unpinned under memory pressure.
#[derive(Clone, Debug)]
pub struct Victim {
    pub session: u32,
    pub agent: u32,
    pub tokens: u64,
    pub idle_since: f64,
    pub deadline: f64,
    pub buy: f64,
}

pub trait EvictionPolicy {
    fn name(&self) -> String;
    /// Called when a call ends and its agent will be needed again.
    fn on_idle(&mut self, ctx: &IdleCtx) -> Hold;
    /// Called when the agent returns after `gap` seconds idle (online learning hook).
    fn on_return(&mut self, _ctx: &IdleCtx, _gap: f64) {}
    /// Memory is full and nothing unpinned is left: pick a pinned agent to unpin.
    /// Default: earliest deadline first (it was about to be released anyway).
    fn on_pressure(&mut self, victims: &[Victim], _now: f64) -> Option<usize> {
        victims
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.deadline.partial_cmp(&b.1.deadline).unwrap())
            .map(|(i, _)| i)
    }
    /// How many leading blocks of the agent's context to keep (called right after
    /// on_idle). The tail past this point is released immediately. Default: all.
    fn keep_prefix_blocks(&mut self, _ctx: &IdleCtx) -> usize {
        usize::MAX
    }
    /// Whether the cache should evict in Belady order (oracle only).
    fn belady(&self) -> bool {
        false
    }
    /// Offline knowledge of the true gap (oracle only).
    fn wants_true_gap(&self) -> bool {
        false
    }
    fn on_idle_with_gap(&mut self, ctx: &IdleCtx, _gap: f64) -> Hold {
        self.on_idle(ctx)
    }
}

/// Furthest predicted return first: what a predictor of return times should evict.
pub fn furthest_deadline(victims: &[Victim]) -> Option<usize> {
    victims
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.deadline.partial_cmp(&b.1.deadline).unwrap())
        .map(|(i, _)| i)
}
