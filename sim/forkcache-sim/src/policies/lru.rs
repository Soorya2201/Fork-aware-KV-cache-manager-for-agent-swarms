//! LRU: the default in vLLM/SGLang. Idle KV is never protected; under pressure the
//! least recently used blocks go first. In cost mode it keeps everything until
//! the agent returns (nothing is ever proactively freed).

use crate::policy::{EvictionPolicy, Hold, IdleCtx};

pub struct Lru {
    pub cost_mode: bool,
}

impl EvictionPolicy for Lru {
    fn name(&self) -> String {
        "lru".into()
    }
    fn on_idle(&mut self, _ctx: &IdleCtx) -> Hold {
        if self.cost_mode { Hold::PinForever } else { Hold::Release }
    }
}
