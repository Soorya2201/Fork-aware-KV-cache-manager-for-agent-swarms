//! Fixed TTL: keep every idle agent's KV for the same T seconds.

use crate::policy::{EvictionPolicy, Hold, IdleCtx};

pub struct FixedTtl {
    pub ttl: f64,
}

impl EvictionPolicy for FixedTtl {
    fn name(&self) -> String {
        format!("ttl_{}s", self.ttl)
    }
    fn on_idle(&mut self, ctx: &IdleCtx) -> Hold {
        Hold::PinUntil(ctx.now + self.ttl)
    }
}
