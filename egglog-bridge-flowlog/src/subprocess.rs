//! The M2 shell-out mechanism: compile a runtime-emitted driver crate, cache
//! the built binary by rule-set hash, spawn it once, and drive its flowlog
//! incremental engine over a line-based stdin/stdout pipe.
//!
//! ## Lifecycle
//!
//! 1. `DriverHandle::build_or_cached(dl)` — given the runtime `.dl` text: hash
//!    it (FNV-1a) to a stable cache key; if `$cache/<hash>/driver` already
//!    exists, reuse it (instant); else materialize a temp crate (`Cargo.toml` +
//!    `build.rs` + `src/main.rs` + `program.dl`) under `$cache/<hash>/` and
//!    shell out to `cargo build --release` (cold builds compile timely +
//!    differential-dataflow + the generated engine, ~tens of seconds; the
//!    binary is then reused). Stale cache entries are pruned so runtime-compile
//!    artifacts don't accumulate unbounded (DISK constraint).
//! 2. `DriverHandle::spawn()` — launch the built binary ONCE as a child with
//!    piped stdin/stdout. The flowlog engine inside stays warm across every
//!    `commit`, preserving incrementality.
//! 3. `insert` / `remove` / `commit` — speak the protocol over the pipe; one
//!    `commit` returns this epoch's output deltas.
//!
//! ## Cache location
//!
//! `$EGGLOG_FLOWLOG_CACHE` if set, else `<system tmp>/egglog-flowlog-cache`.
//! All runtime-compile target dirs live UNDER this single root.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{anyhow, Context, Result};

use crate::codegen;

/// Absolute path to the local flowlog-rs clone. The runtime driver crate
/// depends on `flowlog-runtime` / `flowlog-build` by path from here.
const FLOWLOG_ROOT: &str = "/tmp/flowlog-main";

/// Keep at most this many cached driver builds; older ones are pruned to bound
/// disk use from runtime compilation (DISK constraint in the brief).
const MAX_CACHE_ENTRIES: usize = 8;

/// FNV-1a 64-bit hash of the `.dl` text — the rule-set cache key. A given
/// rule-set hashes identically every run, so it compiles once and is reused.
fn hash_dl(dl: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in dl.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// The single cache root for all runtime-compile artifacts.
fn cache_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("EGGLOG_FLOWLOG_CACHE") {
        PathBuf::from(dir)
    } else {
        std::env::temp_dir().join("egglog-flowlog-cache")
    }
}

/// Prune all but the most-recently-modified `MAX_CACHE_ENTRIES` cache dirs,
/// never removing `keep`. Best-effort; failures are ignored.
fn prune_cache(root: &Path, keep: &Path) {
    let Ok(rd) = std::fs::read_dir(root) else {
        return;
    };
    let mut entries: Vec<(std::time::SystemTime, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir() && e.path() != keep)
        .filter_map(|e| {
            let mtime = e.metadata().and_then(|m| m.modified()).ok()?;
            Some((mtime, e.path()))
        })
        .collect();
    if entries.len() < MAX_CACHE_ENTRIES {
        return;
    }
    entries.sort_by_key(|(t, _)| *t);
    let excess = entries.len() + 1 - MAX_CACHE_ENTRIES;
    for (_, path) in entries.into_iter().take(excess) {
        let _ = std::fs::remove_dir_all(&path);
    }
}

/// A built, possibly-running driver subprocess.
pub struct DriverHandle {
    /// Path to the compiled driver binary (cached by rule-set hash).
    binary: PathBuf,
    /// The spawned child + its piped stdio, once `spawn()` is called.
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout: Option<BufReader<ChildStdout>>,
}

impl DriverHandle {
    /// Build the driver for `dl` (or reuse a cached binary keyed by its hash),
    /// returning a not-yet-spawned handle. This is where the (cold) `cargo
    /// build` happens.
    pub fn build_or_cached(dl: &str) -> Result<Self> {
        let root = cache_root();
        std::fs::create_dir_all(&root)
            .with_context(|| format!("creating cache root {}", root.display()))?;

        let hash = hash_dl(dl);
        let crate_dir = root.join(&hash);
        // The release binary name matches the crate name (`driver`).
        let binary = crate_dir.join("target").join("release").join("driver");

        if binary.exists() {
            // Cache hit: touch the dir so LRU pruning keeps it warm.
            let _ = std::fs::write(crate_dir.join(".touch"), hash.as_bytes());
            prune_cache(&root, &crate_dir);
            return Ok(DriverHandle {
                binary,
                child: None,
                stdin: None,
                stdout: None,
            });
        }

        // Cache miss: materialize the temp crate and shell out to cargo build.
        Self::materialize_crate(&crate_dir, dl)?;
        Self::cargo_build(&crate_dir)?;

        if !binary.exists() {
            return Err(anyhow!(
                "driver build reported success but binary {} is missing",
                binary.display()
            ));
        }
        prune_cache(&root, &crate_dir);
        Ok(DriverHandle {
            binary,
            child: None,
            stdin: None,
            stdout: None,
        })
    }

    /// Build the driver for a generalized join: a `.dl` and a matching driver
    /// `main.rs` are supplied directly (the dd-join path's flexible-arity,
    /// multi-relation protocol), keyed by the hash of `dl ++ main_rs` so each
    /// distinct join shape compiles once and is reused.
    pub fn build_or_cached_with(dl: &str, main_rs: &str) -> Result<Self> {
        let root = cache_root();
        std::fs::create_dir_all(&root)
            .with_context(|| format!("creating cache root {}", root.display()))?;

        let combined = format!("{dl}\n//---main---\n{main_rs}");
        let hash = hash_dl(&combined);
        let crate_dir = root.join(&hash);
        let binary = crate_dir.join("target").join("release").join("driver");

        if binary.exists() {
            let _ = std::fs::write(crate_dir.join(".touch"), hash.as_bytes());
            prune_cache(&root, &crate_dir);
            return Ok(DriverHandle {
                binary,
                child: None,
                stdin: None,
                stdout: None,
            });
        }

        Self::materialize_crate_with(&crate_dir, dl, main_rs)?;
        Self::cargo_build(&crate_dir)?;
        if !binary.exists() {
            return Err(anyhow!(
                "driver build reported success but binary {} is missing",
                binary.display()
            ));
        }
        prune_cache(&root, &crate_dir);
        Ok(DriverHandle {
            binary,
            child: None,
            stdin: None,
            stdout: None,
        })
    }

    /// Write the driver crate's files to `crate_dir`.
    fn materialize_crate(crate_dir: &Path, dl: &str) -> Result<()> {
        Self::materialize_crate_with(crate_dir, dl, &codegen::emit_main_rs())
    }

    /// Write the driver crate's files with an explicit `main.rs`.
    fn materialize_crate_with(crate_dir: &Path, dl: &str, main_rs: &str) -> Result<()> {
        let src = crate_dir.join("src");
        std::fs::create_dir_all(&src).with_context(|| format!("creating {}", src.display()))?;
        std::fs::write(
            crate_dir.join("Cargo.toml"),
            codegen::emit_cargo_toml("driver", FLOWLOG_ROOT),
        )?;
        std::fs::write(crate_dir.join("build.rs"), codegen::emit_build_rs())?;
        std::fs::write(crate_dir.join("program.dl"), dl)?;
        std::fs::write(src.join("main.rs"), main_rs)?;
        Ok(())
    }

    /// Shell out to `cargo build --release` in the temp crate dir.
    fn cargo_build(crate_dir: &Path) -> Result<()> {
        let output = Command::new("cargo")
            .arg("build")
            .arg("--release")
            .current_dir(crate_dir)
            // Confine all build artifacts to this crate's own target dir
            // (under the single cache root).
            .env("CARGO_TARGET_DIR", crate_dir.join("target"))
            .output()
            .with_context(|| format!("spawning cargo build in {}", crate_dir.display()))?;
        if !output.status.success() {
            return Err(anyhow!(
                "cargo build failed for driver in {}:\n--- stdout ---\n{}\n--- stderr ---\n{}",
                crate_dir.display(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            ));
        }
        Ok(())
    }

    /// Spawn the driver binary once, wiring up the stdin/stdout pipe. The
    /// flowlog engine inside stays warm for the whole program.
    pub fn spawn(&mut self) -> Result<()> {
        if self.child.is_some() {
            return Ok(());
        }
        let mut child = Command::new(&self.binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawning driver {}", self.binary.display()))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
        self.stdin = Some(stdin);
        self.stdout = Some(BufReader::new(stdout));
        self.child = Some(child);
        Ok(())
    }

    fn send(&mut self, line: &str) -> Result<()> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("driver not spawned"))?;
        stdin.write_all(line.as_bytes())?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        Ok(())
    }

    /// Stage an `insert <rel> <a> <b>` delta.
    pub fn insert(&mut self, rel: &str, a: i32, b: i32) -> Result<()> {
        self.send(&format!("insert {rel} {a} {b}"))
    }

    /// Stage a `remove <rel> <a> <b>` delta (M2 protocol; retraction wiring is
    /// M3, but the command is supported end-to-end).
    #[allow(dead_code)]
    pub fn remove(&mut self, rel: &str, a: i32, b: i32) -> Result<()> {
        self.send(&format!("remove {rel} {a} {b}"))
    }

    /// Send `clear` and wait for the `ok` reply: reset the dd-join engine's
    /// staged relations to empty before re-staging the next iteration's read
    /// view (the non-recursive join is recomputed from scratch each call).
    pub fn send_clear(&mut self) -> Result<()> {
        self.send("clear")?;
        let stdout = self
            .stdout
            .as_mut()
            .ok_or_else(|| anyhow!("driver not spawned"))?;
        let mut buf = String::new();
        loop {
            buf.clear();
            let n = stdout.read_line(&mut buf)?;
            if n == 0 {
                return Err(anyhow!("driver closed pipe before `ok` (clear)"));
            }
            let line = buf.trim();
            if line == "ok" {
                break;
            }
            if let Some(err) = line.strip_prefix("err ") {
                return Err(anyhow!("driver error: {err}"));
            }
        }
        Ok(())
    }

    /// Stage an `ins <rel_idx> <c0> <c1> ...` delta for the generalized
    /// dd-join protocol (arbitrary-arity row into relation `rel_idx`).
    pub fn insert_row(&mut self, rel_idx: usize, cols: &[i32]) -> Result<()> {
        let mut line = format!("ins {rel_idx}");
        for c in cols {
            line.push(' ');
            line.push_str(&c.to_string());
        }
        self.send(&line)
    }

    /// `commit` one epoch for the generalized dd-join protocol; read back
    /// `row <c0> <c1> ...` lines (one per join binding) until the terminating
    /// `ok`. Returns the binding rows as `Vec<Vec<i32>>`.
    pub fn commit_rows(&mut self) -> Result<Vec<Vec<i32>>> {
        self.send("commit")?;
        let stdout = self
            .stdout
            .as_mut()
            .ok_or_else(|| anyhow!("driver not spawned"))?;
        let mut rows = Vec::new();
        let mut buf = String::new();
        loop {
            buf.clear();
            let n = stdout.read_line(&mut buf)?;
            if n == 0 {
                return Err(anyhow!("driver closed pipe before `ok`"));
            }
            let line = buf.trim();
            if line == "ok" {
                break;
            }
            if let Some(rest) = line.strip_prefix("row ") {
                let cols: Vec<i32> = rest
                    .split_whitespace()
                    .filter_map(|t| t.parse::<i32>().ok())
                    .collect();
                rows.push(cols);
            } else if let Some(err) = line.strip_prefix("err ") {
                return Err(anyhow!("driver error: {err}"));
            }
        }
        Ok(rows)
    }

    /// `commit` one epoch; read back `delta <rel> <x> <z> <diff>` lines until
    /// the terminating `ok`. Returns the `hop` deltas as `(x, z, diff)`.
    pub fn commit(&mut self) -> Result<Vec<(i32, i32, i32)>> {
        self.send("commit")?;
        let stdout = self
            .stdout
            .as_mut()
            .ok_or_else(|| anyhow!("driver not spawned"))?;
        let mut deltas = Vec::new();
        let mut buf = String::new();
        loop {
            buf.clear();
            let n = stdout.read_line(&mut buf)?;
            if n == 0 {
                return Err(anyhow!("driver closed pipe before `ok`"));
            }
            let line = buf.trim();
            if line == "ok" {
                break;
            }
            if let Some(rest) = line.strip_prefix("delta ") {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                // `hop <x> <z> <diff>`
                if toks.len() == 4 {
                    let x: i32 = toks[1].parse().context("delta x")?;
                    let z: i32 = toks[2].parse().context("delta z")?;
                    let d: i32 = toks[3].parse().context("delta diff")?;
                    deltas.push((x, z, d));
                }
            } else if let Some(err) = line.strip_prefix("err ") {
                return Err(anyhow!("driver error: {err}"));
            }
        }
        Ok(deltas)
    }
}

impl Drop for DriverHandle {
    fn drop(&mut self) {
        // Ask the driver to quit, then reap it so we don't leak timely worker
        // threads / zombie processes across egraphs.
        let _ = self.send("quit");
        if let Some(mut child) = self.child.take() {
            let _ = child.wait();
        }
    }
}
