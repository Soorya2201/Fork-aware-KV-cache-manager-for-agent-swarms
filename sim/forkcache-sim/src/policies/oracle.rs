//! Offline oracle: knows every idle gap in advance.
//!
//! Cost mode: pays min(rent * t, buy) per idle period, the per-episode optimum
//! that competitive ratios are measured against.
//! Budget mode: Belady/MIN eviction (evict the block needed furthest in the
//! future), optimal for hit count with uniform-size blocks.

use crate::policy::{EvictionPolicy, Hold, IdleCtx, Victim};

pub struct Oracle {
    pub cost_mode: bool,
}

impl EvictionPolicy for Oracle {
    fn name(&self) -> String {
        "oracle".into()
    }
    fn on_idle(&mut self, _ctx: &IdleCtx) -> Hold {
        Hold::Release
    }
    fn wants_true_gap(&self) -> bool {
        true
    }
    fn on_idle_with_gap(&mut self, ctx: &IdleCtx, gap: f64) -> Hold {
        if !self.cost_mode {
            return Hold::Release; // Belady order does all the work
        }
        if ctx.rent * gap < ctx.buy {
            Hold::PinUntil(ctx.now + gap + 1e-6)
        } else {
            Hold::Release
        }
    }
    fn on_pressure(&mut self, victims: &[Victim], _now: f64) -> Option<usize> {
        crate::policy::furthest_deadline(victims)
    }
    fn belady(&self) -> bool {
        !self.cost_mode
    }
}
