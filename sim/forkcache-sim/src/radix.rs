//! Radix-tree prefix cache over chained block hashes.
//!
//! Because block hashes are chained, the blocks form a tree: a block's parent is
//! the block before it in any sequence that contains it. The cache keeps the
//! radix invariant (a cached block's parent is cached) by evicting leaves only.
//! Each block carries an `active` refcount (running requests) and a `pins`
//! refcount (idle agents whose policy chose to keep their KV). Only blocks with
//! both counts at zero and no cached children are evictable.

use crate::trace::BlockMeta;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Order {
    /// Least recently used first; deeper blocks first among ties.
    Lru,
    /// Belady/MIN: the block whose next use is furthest in the future first.
    Belady,
}

pub struct Cache {
    pub meta: Vec<BlockMeta>,
    cached: Vec<bool>,
    pub active: Vec<u32>,
    pub pins: Vec<u32>,
    cached_children: Vec<u32>,
    last_use: Vec<i64>,
    key: Vec<Option<(i64, i64)>>,
    evictable: BTreeSet<((i64, i64), u32)>,
    pub cached_tokens: u64,
    order: Order,
    /// For Belady: sorted use times (us) of each block as a cacheable prompt block.
    uses: Vec<Vec<i64>>,
    pub now: i64,
}

impl Cache {
    pub fn new(meta: Vec<BlockMeta>, order: Order, uses: Vec<Vec<i64>>) -> Self {
        let n = meta.len();
        Cache {
            meta,
            cached: vec![false; n],
            active: vec![0; n],
            pins: vec![0; n],
            cached_children: vec![0; n],
            last_use: vec![0; n],
            key: vec![None; n],
            evictable: BTreeSet::new(),
            cached_tokens: 0,
            order,
            uses,
            now: 0,
        }
    }

    /// Tokens at the end of `seq` that only this sequence holds: cached, not
    /// active, not pinned, and with no cached child outside `seq`. Because eviction
    /// is leaf-only, these blocks form a suffix; everything before it stays cached
    /// for as long as some other agent or request holds it.
    pub fn exclusive_suffix_tokens(&self, seq: &[u32]) -> u64 {
        let mut tokens = 0u64;
        let mut own_child = 0u32;
        for &b in seq.iter().rev() {
            let i = b as usize;
            if !self.cached[i] {
                own_child = 0;
                continue;
            }
            if self.active[i] != 0 || self.pins[i] != 0 || self.cached_children[i] > own_child {
                break;
            }
            tokens += self.meta[i].tokens as u64;
            own_child = 1;
        }
        tokens
    }

    #[inline]
    pub fn is_cached(&self, b: u32) -> bool {
        self.cached[b as usize]
    }

    fn evictable_now(&self, b: u32) -> bool {
        let i = b as usize;
        self.cached[i] && self.active[i] == 0 && self.pins[i] == 0 && self.cached_children[i] == 0
    }

    fn priority(&self, b: u32) -> (i64, i64) {
        let i = b as usize;
        let depth = -(self.meta[i].depth as i64);
        match self.order {
            Order::Lru => (self.last_use[i], depth),
            Order::Belady => {
                let u = &self.uses[i];
                let p = u.partition_point(|&t| t <= self.now);
                let next = if p < u.len() { u[p] } else { i64::MAX };
                (-next, depth) // smallest key evicted first => furthest next use
            }
        }
    }

    /// Recompute whether `b` belongs in the evictable set.
    pub fn refresh(&mut self, b: u32) {
        let i = b as usize;
        if let Some(k) = self.key[i].take() {
            self.evictable.remove(&(k, b));
        }
        if self.evictable_now(b) {
            let k = self.priority(b);
            self.key[i] = Some(k);
            self.evictable.insert((k, b));
        }
    }

    pub fn insert(&mut self, b: u32) {
        let i = b as usize;
        if self.cached[i] {
            return;
        }
        debug_assert!(self.meta[i].parent.map_or(true, |p| self.cached[p as usize]),
            "radix invariant: parent must be cached before child");
        self.cached[i] = true;
        self.cached_tokens += self.meta[i].tokens as u64;
        self.last_use[i] = self.now;
        if let Some(p) = self.meta[i].parent {
            self.cached_children[p as usize] += 1;
            self.refresh(p);
        }
        self.refresh(b);
    }

    pub fn evict(&mut self, b: u32) {
        let i = b as usize;
        debug_assert!(self.evictable_now(b));
        if let Some(k) = self.key[i].take() {
            self.evictable.remove(&(k, b));
        }
        self.cached[i] = false;
        self.cached_tokens -= self.meta[i].tokens as u64;
        if let Some(p) = self.meta[i].parent {
            self.cached_children[p as usize] -= 1;
            self.refresh(p);
        }
    }

    /// Evict one block chosen by the cache order. Returns false if nothing is evictable.
    pub fn evict_one(&mut self) -> bool {
        match self.evictable.iter().next().copied() {
            Some((_, b)) => {
                self.evict(b);
                true
            }
            None => false,
        }
    }

    pub fn touch(&mut self, b: u32) {
        self.last_use[b as usize] = self.now;
        self.refresh(b);
    }

    /// Free every block of `blocks` (deepest first) that nobody references.
    pub fn free_unreferenced(&mut self, blocks: &[u32]) {
        for &b in blocks.iter().rev() {
            if self.evictable_now(b) {
                self.evict(b);
            }
        }
    }
}
