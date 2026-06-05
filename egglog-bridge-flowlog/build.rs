//! Compile the build-time-fixed `transitive_step.dl` into a Rust module, in
//! INCREMENTAL mode so the generated engine is a long-lived
//! `DatalogIncrementalEngine` (insert_/remove_ staging + `commit()`), not a
//! one-shot batch `run()`. `lib.rs` include!s the generated module.
//!
//! API per `crates/flowlog-build/src/lib.rs` (confirmed by the spike):
//!   Builder::default().mode(ExecutionMode::DatalogInc).compile(&["..dl"], &[])
//!
//! `compile()` writes `$OUT_DIR/<stem>.rs` (stem = `.dl` file stem, here
//! `transitive_step.rs`). The generated file is self-contained: it
//! `pub use`s an inner module that itself pulls in the flowlog-runtime
//! re-exports of timely / differential-dataflow, so the include site needs no
//! extra imports.
//!
//! ## Milestone-1 scope note (the FlowLog crux)
//!
//! egglog defines rules at RUNTIME, but flowlog compiles `.dl` -> Rust at BUILD
//! time. For the M1 PROOF a fixed `.dl` is acceptable (per the brief). Runtime
//! rule installation (regenerate + rustc + dlopen, or another mechanism) is the
//! FlowLog analog of Feldera's static-circuit-rebuild risk and is investigated
//! in ../MILESTONE1.md; it is deferred to M2.
use flowlog_build::{Builder, ExecutionMode};

#[allow(clippy::disallowed_macros)] // for println! (cargo: directives)
fn main() {
    println!("cargo:rerun-if-changed=transitive_step.dl");
    Builder::default()
        // M1's transitive-closure proof is integer-only; no string interning.
        .string_intern(false)
        .mode(ExecutionMode::DatalogInc)
        .compile(&["transitive_step.dl"], &[] as &[&str])
        .expect("flowlog-build failed to compile transitive_step.dl");
}
