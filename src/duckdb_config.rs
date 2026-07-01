//! Configuration for the DuckDB backend.
//!
//! The flags here govern how
//! [`EGraph::with_duckdb_backend`](crate::EGraph::with_duckdb_backend)
//! sets up its underlying [`egglog_bridge_duckdb::EGraph`] and how the
//! egglog frontend instruments the program before handing it off:
//!
//! - [`fast_rebuild`](DuckBackendConfig::fast_rebuild) engages the relational
//!   δuf fast-rebuild.
//! - [`proofs`](DuckBackendConfig::proofs) runs term encoding in
//!   proof-tracking mode.
//!
//! NOTE: DuckDB has been migrated OFF native-UF onto the fast RELATIONAL
//! term-encoding. There is no longer a `native_uf` knob here — congruence is
//! rule-encoded (`@congruence_rule*`) and the relational δuf fast-rebuild
//! canonicalizes it.

/// Knobs for [`EGraph::with_duckdb_backend`](crate::EGraph::with_duckdb_backend).
#[derive(Clone, Default, Debug)]
pub struct DuckBackendConfig {
    /// `--fast-rebuild`: engage the DuckDB backend's RELATIONAL δuf fast-rebuild
    /// (`enable_fast_rebuild`), which drops the always-empty `δview ⋈ uf_old`
    /// rebuild term (sound under canonicalize-at-creation). Bit-exact with the
    /// full rebuild; default ON for the migrated plain-`--duckdb` path.
    pub fast_rebuild: bool,
    /// Run the encoder in proof-tracking mode. The term-encoded
    /// program then threads explicit proofs through every union and
    /// rewrite, exercising a different control-flow path through the
    /// encoder. Set by tests to confirm the duckdb-side optimizations
    /// (inline-congruence, hash-cons) don't regress proof mode.
    pub proofs: bool,
}
