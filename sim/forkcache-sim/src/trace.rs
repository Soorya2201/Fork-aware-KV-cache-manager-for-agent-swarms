//! Trace loading: one JSON object per LLM call (see trace/SCHEMA.md).

use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Clone, Deserialize)]
pub struct Call {
    pub call_id: u64,
    pub session: u32,
    pub agent: u32,
    #[serde(default)]
    pub round: u32,
    pub parent_call: Option<u64>,
    #[serde(default)]
    pub fork_of: Option<u64>,
    pub t_start: f64,
    pub t_end: f64,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub next_call_same_agent_gap_s: Option<f64>,
    pub block_size: u32,
    pub block_hashes: Vec<String>,
}

/// Static description of one KV block (fixed by the trace, independent of policy).
#[derive(Debug, Clone)]
pub struct BlockMeta {
    pub parent: Option<u32>,
    pub tokens: u32,
    pub depth: u32,
}

/// A trace with block hashes interned to dense ids.
pub struct Workload {
    pub calls: Vec<Call>,
    /// Interned block ids of each call's full sequence (prompt + output).
    pub seq_blocks: Vec<Vec<u32>>,
    /// Number of leading blocks that lie entirely inside the prompt (cacheable on reuse).
    pub n_prompt_blocks: Vec<usize>,
    pub blocks: Vec<BlockMeta>,
    pub block_size: u32,
}

pub fn load(path: &str) -> Result<Vec<Call>, String> {
    let f = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut calls = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let c: Call = serde_json::from_str(&line).map_err(|e| format!("line {}: {e}", i + 1))?;
        calls.push(c);
    }
    calls.sort_by_key(|c| c.call_id);
    Ok(calls)
}

impl Workload {
    pub fn new(calls: Vec<Call>) -> Self {
        let block_size = calls.first().map(|c| c.block_size).unwrap_or(16);
        let mut ids: HashMap<String, u32> = HashMap::new();
        let mut blocks: Vec<BlockMeta> = Vec::new();
        let mut seq_blocks = Vec::with_capacity(calls.len());
        let mut n_prompt_blocks = Vec::with_capacity(calls.len());
        for c in &calls {
            let seq_len = c.prompt_tokens + c.output_tokens;
            let n = c.block_hashes.len();
            let mut v = Vec::with_capacity(n);
            let mut parent: Option<u32> = None;
            for (i, h) in c.block_hashes.iter().enumerate() {
                let tokens = if i + 1 < n { block_size } else { seq_len - block_size * (n as u32 - 1) };
                let id = *ids.entry(h.clone()).or_insert_with(|| {
                    blocks.push(BlockMeta { parent, tokens, depth: i as u32 });
                    (blocks.len() - 1) as u32
                });
                v.push(id);
                parent = Some(id);
            }
            seq_blocks.push(v);
            n_prompt_blocks.push(((c.prompt_tokens / block_size) as usize).min(n));
        }
        Workload { calls, seq_blocks, n_prompt_blocks, blocks, block_size }
    }

    pub fn sessions(&self) -> Vec<u32> {
        let mut s: Vec<u32> = self.calls.iter().map(|c| c.session).collect();
        s.sort();
        s.dedup();
        s
    }
}
