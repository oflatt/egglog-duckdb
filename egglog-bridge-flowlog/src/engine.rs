//! Thin wrapper around the build-time-generated flowlog-rs
//! `DatalogIncrementalEngine`.
//!
//! The generated module (from `build.rs` compiling `transitive_step.dl` in
//! `DatalogInc` mode) is `include!`d here so the generated symbols stay
//! confined to this file. Per SPIKE_RESULTS.md (confirmed empirically) the
//! generated API is:
//!
//! - `DatalogIncrementalEngine::new(workers) -> Self` (spawns timely workers),
//! - `engine.begin()` (auto-called by the first `insert_*`/`remove_*`),
//! - `engine.insert_<rel>(Vec<rel::T>)` / `remove_<rel>(Vec<rel::T>)`
//!   (relation names lowercased: `insert_edge`, `insert_path`),
//! - `engine.commit() -> IncrementalResults` (steps ONE epoch to fixpoint,
//!   blocks, returns this epoch's per-output deltas),
//! - `rel::Edge` / `rel::Path` / `rel::Hop` are **tuple aliases** `(i32, i32)`,
//! - `IncrementalResults.hop: Vec<(rel::Hop, i32)>` (field = lowercase rel name;
//!   the `i32` is the differential-dataflow multiplicity diff).
//!
//! Because the `.dl` join `hop(x,z) :- path(x,y), edge(y,z)` is **non-recursive**,
//! one `commit()` is exactly one round of the join over the staged delta — which
//! is what makes one egglog `run_rules` call = one bounded hop.

#[allow(clippy::all)]
#[allow(dead_code)]
#[allow(unused)]
mod gen {
    // The generated file is fully self-contained (it pulls the
    // flowlog-runtime re-exports of timely / differential-dataflow itself).
    include!(concat!(env!("OUT_DIR"), "/transitive_step.rs"));
}

use gen::DatalogIncrementalEngine;

/// Owns the live flowlog incremental engine plus the host-side per-round
/// feedback buffer (`pending_path`): the `path` rows derived in the last
/// committed hop, to be staged as the next round's `insert_path` delta.
pub struct FlowEngine {
    engine: DatalogIncrementalEngine,
    /// `path` rows derived last round, awaiting feed-back next round.
    pending_path: Vec<(i32, i32)>,
}

impl FlowEngine {
    /// Spawn a fresh single-worker incremental engine.
    pub fn new() -> Self {
        FlowEngine {
            engine: DatalogIncrementalEngine::new(1),
            pending_path: Vec::new(),
        }
    }

    /// Stage `edge(src, dst)` insert deltas (auto-begins the txn).
    pub fn insert_edge(&mut self, rows: &[(i32, i32)]) {
        if rows.is_empty() {
            return;
        }
        self.engine.insert_edge(rows.to_vec());
    }

    /// Stage `path(src, dst)` insert deltas (auto-begins the txn).
    pub fn insert_path(&mut self, rows: &[(i32, i32)]) {
        if rows.is_empty() {
            return;
        }
        self.engine.insert_path(rows.to_vec());
    }

    /// Step exactly one epoch to fixpoint and return this epoch's `hop` deltas
    /// as `(x, z, diff)`. The non-recursive join means these are exactly the
    /// new one-hop extensions caused by the rows staged since the last commit.
    pub fn commit_hop(&mut self) -> Vec<(i32, i32, i32)> {
        let results = self.engine.commit();
        results
            .hop
            .into_iter()
            .map(|(t, d): ((i32, i32), i32)| (t.0, t.1, d))
            .collect()
    }

    /// Record the `path` rows to feed back next round.
    pub fn set_pending_path(&mut self, rows: Vec<(i32, i32)>) {
        self.pending_path = rows;
    }

    /// Take (and clear) the pending feed-back `path` rows.
    pub fn take_pending_path(&mut self) -> Vec<(i32, i32)> {
        std::mem::take(&mut self.pending_path)
    }
}
