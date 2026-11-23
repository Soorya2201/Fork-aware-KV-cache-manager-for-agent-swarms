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
