// Compile path.dl into a Rust module at build time, in INCREMENTAL mode so the
// generated engine is a long-lived `DatalogIncrementalEngine` (insert/remove +
// commit), not a one-shot batch `run()`.
//
// flowlog-build writes the generated code into OUT_DIR; main.rs include!s it.
//
// API per crates/flowlog-build/src/lib.rs:
//   Builder::default().mode(ExecutionMode::DatalogInc).compile(&["path.dl"], &[])
//
// NOTE: the exact constant name (`ExecutionMode::DatalogInc`) and the exact
// "where is the generated file" contract are the two things to confirm against
// the real crate the first time this compiles; both are read straight from
// flowlog-build's source. If `compile()` writes to OUT_DIR with a fixed name,
// adjust the include! path in main.rs accordingly.
use flowlog_build::{Builder, ExecutionMode};

fn main() {
    println!("cargo:rerun-if-changed=path.dl");
    Builder::default()
        .string_intern(false) // numbers only in this spike; no string interning needed
        .mode(ExecutionMode::DatalogInc)
        .compile(&["path.dl"], &[] as &[&str])
        .expect("flowlog-build failed to compile path.dl");
}
