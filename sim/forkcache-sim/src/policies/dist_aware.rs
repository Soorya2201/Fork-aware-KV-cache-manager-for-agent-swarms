//! Distribution-aware threshold (paper/model.md, section 4).
//!
//! Picks the threshold theta that minimizes expected cost under the empirical
//! idle-gap distribution:  E[C(theta)] = rent * E[min(T, theta)] + buy * P(T > theta).
//! Between sample points E[C] rises linearly in theta and drops at each sample,
//! so the minimum is at theta = 0 or at one of the observed gaps: exact in O(n log n).
//!
//! To keep the evaluation honest the distribution for a call in session s is
//! fitted only on the other sessions (leave-one-session-out cross-fitting).
//! `per_agent` conditions on the agent index (an agent's turn position fixes
//! most of its idle time in a turn-based system). `alpha` optionally clamps theta
//! to [tau/(alpha-1), (alpha-1) tau], which bounds the worst-case competitive
//! ratio by alpha even when the fitted distribution is wrong.

use crate::policy::{furthest_deadline, EvictionPolicy, Hold, IdleCtx, Victim};
use std::collections::HashMap;

pub struct DistAware {
    /// (session, agent, gap) for every observed idle period in the trace.
    samples: Vec<(u32, u32, f64)>,
    per_agent: bool,
    alpha: Option<f64>,
    cache: HashMap<(u32, u32), (Vec<f64>, Vec<f64>)>,
    /// Budget mode (fixed pool, memory has no price): pin every idle agent until a
    /// safe upper bound on its predicted return and, under pressure, unpin the one
    /// predicted to return last (per-agent Belady, paper/model.md section 5).
    pub budget_mode: bool,
}

impl DistAware {
    pub fn new(samples: Vec<(u32, u32, f64)>, per_agent: bool, alpha: Option<f64>) -> Self {
        DistAware { samples, per_agent, alpha, cache: HashMap::new(), budget_mode: false }
    }

    pub fn budget(mut self, on: bool) -> Self {
        self.budget_mode = on;
        self
    }

    fn dist(&mut self, session: u32, agent: u32) -> &(Vec<f64>, Vec<f64>) {
        let key = (session, if self.per_agent { agent } else { u32::MAX });
        let (samples, per_agent) = (&self.samples, self.per_agent);
        self.cache.entry(key).or_insert_with(|| {
            let mut g: Vec<f64> = samples
                .iter()
                .filter(|(s, a, _)| *s != session && (!per_agent || *a == agent))
                .map(|x| x.2)
                .collect();
            g.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut pre = Vec::with_capacity(g.len() + 1);
            pre.push(0.0);
            for &x in &g {
                pre.push(pre.last().unwrap() + x);
            }
            (g, pre)
        })
    }
}

/// argmin over theta in {0} U samples of rent*E[min(T,theta)] + buy*P(T>theta).
/// Returns (theta, expected cost). theta = 0 means evict immediately.
pub fn best_threshold(sorted: &[f64], prefix: &[f64], rent: f64, buy: f64) -> (f64, f64) {
    let n = sorted.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let nf = n as f64;
    let mut best = (0.0, buy * sorted.iter().filter(|&&t| t > 0.0).count() as f64 / nf);
    let mut k = 0;
    while k < n {
        let theta = sorted[k];
        let mut m = k;
        while m < n && sorted[m] <= theta {
            m += 1;
        }
        let e_min = (prefix[m] + theta * (n - m) as f64) / nf;
        let cost = rent * e_min + buy * (n - m) as f64 / nf;
        if cost < best.1 {
            best = (theta, cost);
        }
        k = m;
    }
    best
}

impl EvictionPolicy for DistAware {
    fn name(&self) -> String {
        let base = if self.per_agent { "dist_aware_agent" } else { "dist_aware" };
        match self.alpha {
            Some(a) => format!("{base}_robust{a}"),
            None => base.into(),
        }
    }

    fn on_idle(&mut self, ctx: &IdleCtx) -> Hold {
        if self.budget_mode {
            let (g, _) = self.dist(ctx.session, ctx.agent);
            return match g.last() {
                // predicted return within the largest gap seen for this agent (+20%)
                Some(&max_gap) => Hold::PinUntil(ctx.now + 1.2 * max_gap + 1e-6),
                None => Hold::Release,
            };
        }
        let tau = ctx.tau();
        let alpha = self.alpha;
        let (g, pre) = self.dist(ctx.session, ctx.agent);
        let (mut theta, _) = best_threshold(g, pre, ctx.rent, ctx.buy);
        if theta.is_nan() {
            theta = tau; // no data: fall back to deterministic ski rental
        }
        if let Some(a) = alpha {
            theta = theta.clamp(tau / (a - 1.0), (a - 1.0) * tau);
        }
        if theta <= 0.0 {
            Hold::Release
        } else {
            Hold::PinUntil(ctx.now + theta + 1e-6)
        }
    }

    fn on_pressure(&mut self, victims: &[Victim], _now: f64) -> Option<usize> {
        furthest_deadline(victims)
    }
}
