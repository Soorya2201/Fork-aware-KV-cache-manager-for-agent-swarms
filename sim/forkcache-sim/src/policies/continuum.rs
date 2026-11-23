//! Continuum-style TTL (our reimplementation of the idea, not the authors' code).
//!
//! Continuum (Li et al., 2025) pins a multi-turn agent's KV with a time-to-live
//! derived from the predicted duration of the pause, and only when pinning is
//! cheaper than recomputing. Here the prediction is the running mean of the last
//! `window` observed gaps of the same agent, learned online as the replay runs.

use crate::policy::{furthest_deadline, EvictionPolicy, Hold, IdleCtx, Victim};
use std::collections::{HashMap, VecDeque};

pub struct ContinuumStyle {
    pub window: usize,
    pub margin: f64,
    history: HashMap<u32, VecDeque<f64>>,
}

impl ContinuumStyle {
    pub fn new(window: usize, margin: f64) -> Self {
        ContinuumStyle { window, margin, history: HashMap::new() }
    }
}

impl EvictionPolicy for ContinuumStyle {
    fn name(&self) -> String {
        "continuum_style".into()
    }

    fn on_idle(&mut self, ctx: &IdleCtx) -> Hold {
        match self.history.get(&ctx.agent).filter(|h| !h.is_empty()) {
            None => Hold::PinUntil(ctx.now + ctx.tau()), // cold start: ski rental
            Some(h) => {
                let pred = h.iter().sum::<f64>() / h.len() as f64;
                if ctx.rent * pred < ctx.buy {
                    Hold::PinUntil(ctx.now + self.margin * pred)
                } else {
                    Hold::Release
                }
            }
        }
    }

    fn on_return(&mut self, ctx: &IdleCtx, gap: f64) {
        let h = self.history.entry(ctx.agent).or_default();
        h.push_back(gap);
        while h.len() > self.window {
            h.pop_front();
        }
    }

    fn on_pressure(&mut self, victims: &[Victim], _now: f64) -> Option<usize> {
        furthest_deadline(victims)
    }
}
