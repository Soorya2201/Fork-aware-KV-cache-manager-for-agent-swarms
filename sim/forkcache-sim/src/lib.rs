//! forkcache-sim: replay multi-agent LLM traces through a prefix KV cache and
//! compare keep-or-evict policies for idle agents (see paper/model.md).

pub mod policies;
pub mod policy;
pub mod radix;
pub mod sim;
pub mod trace;

use policies::{continuum::ContinuumStyle, dist_aware::DistAware, fixed_ttl::FixedTtl, fork_aware::{ForkAware, ForkPoint, ReuseSample}, lru::Lru,
               oracle::Oracle, ski_rental::{SkiRental, SkiRentalRandom}};
use policy::EvictionPolicy;
use trace::Workload;

pub const ALL_POLICIES: &[&str] = &[
    "lru", "ttl", "ski_rental", "ski_rental_rand", "continuum_style",
    "dist_aware", "dist_aware_agent", "dist_aware_agent_robust", "oracle",
    "fa_ski_rental", "fa_ski_rental_rand", "fa_dist_aware_agent", "fa_dist_aware_agent_robust", "fa_oracle",
];

/// Idle-gap samples (session, agent, gap) from the trace, for distribution-aware fitting.
pub fn gap_samples(w: &Workload) -> Vec<(u32, u32, f64)> {
    w.calls.iter()
        .filter_map(|c| c.next_call_same_agent_gap_s.map(|g| (c.session, c.agent, g)))
        .collect()
}

/// Fork-point samples (session, agent, round, reused fraction, gap) for fork-aware policies.
pub fn reuse_samples(w: &Workload) -> Vec<ReuseSample> {
    let reuse = sim::reuse_blocks(w);
    w.calls.iter().enumerate()
        .filter_map(|(i, c)| c.next_call_same_agent_gap_s.map(|g| {
            let n = w.seq_blocks[i].len().max(1) as f64;
            (c.session, c.agent, c.round, reuse[i] as f64 / n, g)
        }))
        .collect()
}

/// Build fresh policy instances by name ("ttl" expands to one per TTL value).
pub fn make_policies(name: &str, w: &Workload, cost_mode: bool, ttls: &[f64], seed: u64)
    -> Vec<Box<dyn EvictionPolicy + Send>> {
    match name {
        "lru" => vec![Box::new(Lru { cost_mode })],
        "ttl" => ttls.iter().map(|&t| Box::new(FixedTtl { ttl: t }) as Box<dyn EvictionPolicy + Send>).collect(),
        "ski_rental" => vec![Box::new(SkiRental)],
        "ski_rental_rand" => vec![Box::new(SkiRentalRandom::new(seed))],
        "continuum_style" => vec![Box::new(ContinuumStyle::new(8, 1.25))],
        "dist_aware" => vec![Box::new(DistAware::new(gap_samples(w), false, None))],
        "dist_aware_agent" => vec![Box::new(DistAware::new(gap_samples(w), true, None))],
        "dist_aware_agent_robust" => vec![Box::new(DistAware::new(gap_samples(w), true, Some(3.0)))],
        "oracle" => vec![Box::new(Oracle { cost_mode })],
        "fa_ski_rental" => vec![Box::new(ForkAware::new(SkiRental, ForkPoint::Predict(reuse_samples(w))))],
        "fa_ski_rental_rand" =>
            vec![Box::new(ForkAware::new(SkiRentalRandom::new(seed), ForkPoint::Predict(reuse_samples(w))))],
        "fa_dist_aware_agent" => vec![Box::new(ForkAware::new(
            DistAware::new(gap_samples(w), true, None), ForkPoint::Predict(reuse_samples(w))))],
        "fa_dist_aware_agent_robust" => vec![Box::new(ForkAware::new(
            DistAware::new(gap_samples(w), true, Some(3.0)), ForkPoint::Predict(reuse_samples(w))))],
        "fa_oracle" => vec![Box::new(ForkAware::new(Oracle { cost_mode }, ForkPoint::Oracle))],
        other => panic!("unknown policy {other}; known: {ALL_POLICIES:?}"),
    }
}
