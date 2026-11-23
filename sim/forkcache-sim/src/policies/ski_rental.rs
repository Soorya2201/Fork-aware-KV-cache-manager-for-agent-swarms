//! Ski-rental thresholds (paper/model.md, sections 2-3).
//!
//! Deterministic: keep until tau(L) = R(L) / (c_mem L), then evict. 2-competitive.
//! Randomized: keep until tau * X with X ~ e^x / (e - 1) on [0, 1]. e/(e-1)-competitive.

use crate::policy::{EvictionPolicy, Hold, IdleCtx};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct SkiRental;

impl EvictionPolicy for SkiRental {
    fn name(&self) -> String {
        "ski_rental".into()
    }
    fn on_idle(&mut self, ctx: &IdleCtx) -> Hold {
        Hold::PinUntil(ctx.now + ctx.tau())
    }
}

pub struct SkiRentalRandom {
    rng: StdRng,
}

impl SkiRentalRandom {
    pub fn new(seed: u64) -> Self {
        SkiRentalRandom { rng: StdRng::seed_from_u64(seed ^ 0x5eed_5a1) }
    }
}

/// Inverse CDF of the density e^x / (e - 1) on [0, 1].
pub fn sample_threshold_fraction(u: f64) -> f64 {
    (1.0 + u * (std::f64::consts::E - 1.0)).ln()
}

impl EvictionPolicy for SkiRentalRandom {
    fn name(&self) -> String {
        "ski_rental_rand".into()
    }
    fn on_idle(&mut self, ctx: &IdleCtx) -> Hold {
        let x = sample_threshold_fraction(self.rng.r#gen::<f64>());
        Hold::PinUntil(ctx.now + ctx.tau() * x)
    }
}
