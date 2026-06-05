//! Runtime translation of an egglog rule (the trait-level `RuleBuilderOps` IR)
//! into a FlowLog `.dl` program plus a thin **driver** crate, for the M2
//! shell-out architecture.
//!
//! ## What is generated, at runtime
//!
//! M1 compiled a *build-time-fixed* `transitive_step.dl` and recognized the
//! rule shape structurally. M2 instead takes the rule the frontend builds **at
//! runtime** (`StepShape`, recognized from the live `RuleIr`) and emits:
//!
//! 1. a `.dl` for it (a non-recursive single-join step — one `commit()` = one
//!    bounded hop, exactly the M1 per-iteration model, but the relation names,
//!    join columns, and head come from the runtime rule, not a fixed file);
//! 2. a driver `main.rs` that `include!`s the flowlog-build-generated engine and
//!    speaks a line-based **stdin/stdout command protocol** (see `protocol`);
//! 3. a `Cargo.toml` + `build.rs` so the temp crate compiles standalone.
//!
//! The caller (`subprocess.rs`) hashes the `.dl`, builds the crate **once** per
//! rule-set (`cargo build`, ~tens of seconds cold), caches the binary by hash,
//! then spawns it and drives it over the pipe for the whole program — so the
//! flowlog incremental engine stays warm and incrementality is preserved.
//!
//! ## The non-recursive single-join shape
//!
//! The recognized step is `head(x, z) :- path(x, y), edge(y, z)`. We emit two
//! command-staged input relations (`path`, `edge`) and one output `hop`:
//!
//! ```text
//! .decl edge(src: int32, dst: int32)
//! .input edge(IO="command", delimiter=",")
//! .decl path(src: int32, dst: int32)
//! .input path(IO="command", delimiter=",")
//! .decl hop(src: int32, dst: int32)
//! hop(x, z) :- path(x, y), edge(y, z).
//! .output hop
//! ```
//!
//! The host (`lib.rs::run_one_hop_shellout`) folds each commit's `hop` deltas
//! into the Rust-side mirror and re-stages the new `path` rows next round, the
//! same bounded host-feedback loop M1 used — only now driving a *subprocess*
//! over a pipe instead of an in-process engine.

/// The fixed relation names the generated `.dl` and driver use. The runtime
/// rule's actual `FunctionId`s are mapped onto these three roles by the host
/// (`StepShape`); the `.dl` itself only needs stable lowercase idents so the
/// generated engine's API (`insert_path` / `insert_edge` / `IncrementalResults.
/// hop`) and the driver's protocol dispatch are predictable.
pub const REL_EDGE: &str = "edge";
pub const REL_PATH: &str = "path";
pub const REL_HOP: &str = "hop";

/// Emit the runtime `.dl` for a recognized transitive-closure step.
///
/// The shape is fixed (non-recursive single join), but this is produced *at
/// rule-install time from the runtime rule IR* — it is not a checked-in file.
/// Future shapes (different arities / multiple joins) extend here.
pub fn emit_dl() -> String {
    let mut s = String::new();
    s.push_str("// AUTO-GENERATED at runtime by egglog-bridge-flowlog (M2 shell-out).\n");
    s.push_str("// One non-recursive join = one bounded egglog hop per commit().\n\n");
    s.push_str(&format!(".decl {REL_EDGE}(src: int32, dst: int32)\n"));
    s.push_str(&format!(
        ".input {REL_EDGE}(IO=\"command\", delimiter=\",\")\n\n"
    ));
    s.push_str(&format!(".decl {REL_PATH}(src: int32, dst: int32)\n"));
    s.push_str(&format!(
        ".input {REL_PATH}(IO=\"command\", delimiter=\",\")\n\n"
    ));
    s.push_str(&format!(".decl {REL_HOP}(src: int32, dst: int32)\n"));
    s.push_str(&format!(
        "{REL_HOP}(x, z) :- {REL_PATH}(x, y), {REL_EDGE}(y, z).\n"
    ));
    s.push_str(&format!(".output {REL_HOP}\n"));
    s
}

/// The driver crate's `Cargo.toml`, parameterized by the absolute path to the
/// local flowlog-rs clone (so the temp crate finds `flowlog-runtime` /
/// `flowlog-build`). The crate is intentionally NOT a workspace member
/// (`[workspace]` empty table) so it builds standalone in its temp dir.
pub fn emit_cargo_toml(crate_name: &str, flowlog_root: &str) -> String {
    format!(
        r#"[package]
name = "{crate_name}"
version = "0.0.0"
edition = "2021"
publish = false

# Standalone: do NOT inherit the egglog workspace.
[workspace]

[[bin]]
name = "{crate_name}"
path = "src/main.rs"

[dependencies]
flowlog-runtime = {{ path = "{flowlog_root}/crates/flowlog-runtime" }}

[build-dependencies]
flowlog-build = {{ path = "{flowlog_root}/crates/flowlog-build" }}
"#
    )
}

/// The driver crate's `build.rs`: compile the runtime-emitted `program.dl` into
/// `$OUT_DIR/program.rs` in incremental mode. Identical in spirit to the M1
/// build.rs, but the `.dl` it compiles was written at runtime.
pub fn emit_build_rs() -> String {
    r#"//! AUTO-GENERATED driver build.rs (M2 shell-out).
use flowlog_build::{Builder, ExecutionMode};

fn main() {
    println!("cargo:rerun-if-changed=program.dl");
    Builder::default()
        .string_intern(false)
        .mode(ExecutionMode::DatalogInc)
        .compile(&["program.dl"], &[] as &[&str])
        .expect("flowlog-build failed to compile program.dl");
}
"#
    .to_string()
}

/// The driver `main.rs`: include the generated engine, then run the line-based
/// stdin/stdout command protocol. The host side that speaks this wire format is
/// [`crate::subprocess::DriverHandle`].
///
/// Protocol (one command per line on stdin; responses on stdout):
///   - `insert <rel> <c0> <c1>`  -> stage an insert delta (no response)
///   - `remove <rel> <c0> <c1>`  -> stage a remove delta (no response)
///   - `commit`  -> step ONE epoch; emit `delta hop <x> <z> <diff>` for every
///     output delta, then a final `ok` line
///   - `read <rel>`              -> (reserved; emits `ok`)
///   - `quit`                    -> exit 0
///
/// The driver flushes stdout after every command so the host never deadlocks.
pub fn emit_main_rs() -> String {
    r#"//! AUTO-GENERATED driver (M2 shell-out): embeds a flowlog-rs
//! `DatalogIncrementalEngine` for the runtime-emitted `program.dl` and drives
//! it over a line-based stdin/stdout command protocol. One `commit` command =
//! one engine `commit()` = one bounded egglog hop.

#[allow(clippy::all)]
#[allow(dead_code)]
#[allow(unused)]
mod gen {
    include!(concat!(env!("OUT_DIR"), "/program.rs"));
}

use gen::DatalogIncrementalEngine;
use std::io::{BufRead, Write};

fn main() {
    let mut engine = DatalogIncrementalEngine::new(1);

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let cmd = it.next().unwrap_or("");
        match cmd {
            "insert" | "remove" => {
                let rel = it.next().unwrap_or("");
                let cols: Vec<i32> = it.filter_map(|t| t.parse::<i32>().ok()).collect();
                if cols.len() != 2 {
                    let _ = writeln!(out, "err expected 2 columns, got {}", cols.len());
                    let _ = out.flush();
                    continue;
                }
                let tup = (cols[0], cols[1]);
                let items = vec![tup];
                match (cmd, rel) {
                    ("insert", "edge") => engine.insert_edge(items),
                    ("insert", "path") => engine.insert_path(items),
                    ("remove", "edge") => engine.remove_edge(items),
                    ("remove", "path") => engine.remove_path(items),
                    _ => {
                        let _ = writeln!(out, "err unknown rel {rel}");
                        let _ = out.flush();
                    }
                }
            }
            "commit" => {
                let results = engine.commit();
                for (t, d) in results.hop.into_iter() {
                    let t: (i32, i32) = t;
                    let _ = writeln!(out, "delta hop {} {} {}", t.0, t.1, d);
                }
                let _ = writeln!(out, "ok");
                let _ = out.flush();
            }
            "read" => {
                // Reserved: the host keeps its own mirror; nothing to stream.
                let _ = writeln!(out, "ok");
                let _ = out.flush();
            }
            "quit" => {
                break;
            }
            other => {
                let _ = writeln!(out, "err unknown command {other}");
                let _ = out.flush();
            }
        }
    }
}
"#
    .to_string()
}
