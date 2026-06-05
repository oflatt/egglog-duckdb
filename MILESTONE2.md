# Milestone 2 — FlowLog backend: runtime rule installation via shell-out

**Verdict: ACHIEVED.** A rule **defined at runtime** through the `Backend`
trait (`RuleBuilderOps`, exactly as the egglog frontend builds rules) is
installed by **translating it to FlowLog `.dl` on the fly, compiling a thin
driver crate via `cargo build` (shell-out), and driving the resulting
subprocess over a stdin/stdout pipe** — one `commit` per `run_rules` call = one
bounded egglog hop. Bounded `(run N)` matches the reference backend
(`egglog_bridge::EGraph`) round-for-round. M1's in-process proof stays green.

This supersedes MILESTONE1.md's dlopen/cdylib/FFI plan: the manager chose
**shell-out** (compile a standalone driver binary, drive it over a pipe) over
`dlopen`-ing a cdylib through a C ABI. Same unblocking insight (flowlog codegen
is cheap and dir-agnostic; the `rustc`/DD compile dominates and amortizes if you
compile a rule-set **once** and cache the binary), but no `unsafe` FFI surface
and no C-ABI shim around the non-`#[repr(C)]` generated types — the engine is
reached through a normal Rust `main()`.

## The shell-out architecture as built

```
 RuleBuilderOps (runtime rule)        egglog-bridge-flowlog (this crate)
        │                                       │
        │ recognize_step  ──► StepShape         │  src/lib.rs
        ▼                                       │
   codegen::emit_dl()  ──► program.dl (runtime, not a checked-in file)
   codegen::emit_main_rs/build_rs/cargo_toml ──► driver crate files
        │                                       │  src/codegen.rs
        ▼
   subprocess::DriverHandle::build_or_cached(dl)
        │   hash(.dl) ─► cache key
        │   cache hit?  ─► reuse $cache/<hash>/target/release/driver  (instant)
        │   cache miss? ─► materialize temp crate + `cargo build --release`
        ▼                                       │  src/subprocess.rs
   DriverHandle::spawn()  ──► child process, piped stdin/stdout (ONCE)
        │
        │  insert/remove/commit over the pipe   ◄──► driver: flowlog
        │  one commit = one epoch = one hop           DatalogIncrementalEngine
        ▼                                             (stays warm across commits)
   fold_hops_shellout ──► Rust-side mirror; re-stage new `path` rows next round
```

Files:
- `src/codegen.rs` — runtime translation of the rule IR to (a) a `.dl` program,
  (b) a driver `main.rs` speaking the wire protocol, (c) a standalone
  `Cargo.toml` + `build.rs`.
- `src/subprocess.rs` — `DriverHandle`: rule-set-hash compile cache, `cargo
  build` shell-out, subprocess spawn/lifecycle, and the pipe driver
  (`insert` / `remove` / `commit`).
- `src/lib.rs` — `ExecMode::{InProcess, ShellOut}`; `EGraph::new_shellout()`;
  `run_one_hop_shellout` (the M2 per-iteration loop driving the subprocess).
- `tests/run_n_shellout_proof.rs` — the M2 proof (`#[ignore]` by default; it
  shells out to a cold `cargo build`).

## The wire protocol

Line-based text over the child's stdin/stdout, one command per line; the driver
**flushes stdout after every command** so the host never deadlocks:

| command                  | direction | meaning |
|--------------------------|-----------|---------|
| `insert <rel> <c0> <c1>` | host→drv  | stage an insert delta for `<rel>` (no reply) |
| `remove <rel> <c0> <c1>` | host→drv  | stage a remove delta (no reply) |
| `commit`                 | host→drv  | step ONE epoch; reply with zero or more `delta hop <x> <z> <diff>` lines, terminated by a single `ok` |
| `read <rel>`             | host→drv  | reserved (host keeps its own mirror); replies `ok` |
| `quit`                   | host→drv  | exit 0 |
| `err <msg>`              | drv→host  | error (surfaced as an `anyhow` error host-side) |

Relations are 2-arity `(i32, i32)` tuples (the projection columns of the
recognized transitive-closure step); the format is whitespace-delimited and
robust to large integer tuples. `commit`'s `delta` lines carry the
differential-dataflow multiplicity `<diff>` so retraction (`diff < 0`) is
representable on the wire (M3 will consume it; M2 is monotone and folds only
`diff > 0`, matching M1).

## The runtime-rule-install proof vs reference (actual output)

`tests/run_n_shellout_proof.rs` builds the rule
`path(x, z) :- path(x, y), edge(y, z)` **at runtime through `RuleBuilderOps`**
(no build-time `.dl`), seeds the chain `1->2->3->4`, and drives bounded `(run N)`
through the shell-out path. Verbatim:

```
run(1) reference = {(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
run(1) flowlog(shellout) = {(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
run(3) reference = {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
run(3) flowlog(shellout) = {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
test run1_vs_run3_shellout_runtime_install_matches_reference ... ok
```

- `(run 1)` adds one hop (`(1,3)`, `(2,4)`) but does **NOT** contain the 3-hop
  pair `(1,4)` — **bounded**, not saturated.
- `(run 3)` reaches the full closure including `(1,4)`.
- The shell-out FlowLog backend **equals the reference backend** at both N — the
  faithfulness proof — and `run(1) != run(3)` is the bounded-iteration proof.
- The rule is installed **at runtime** via codegen → shell-out compile →
  subprocess (the M2 bar), not a build-time-fixed `.dl` (M1).

Reproduce:
```
EGGLOG_FLOWLOG_CACHE=/tmp/egglog-flowlog-m2cache \
cargo test --release --manifest-path egglog-bridge-flowlog/Cargo.toml \
  --test run_n_shellout_proof -- --ignored --nocapture
```

M1's in-process proof (`--test run_n_proof`) still passes unchanged.

## Compile-caching & costs observed

- **Cache key:** FNV-1a hash of the runtime `.dl` text. Same rule-set → same
  hash → compiles **once**, reused thereafter. (The two fresh egraphs in the
  proof share one cached driver: the first builds it, the second reuses it.)
- **Cold compile:** ~18.7s wall for the test (single rule-set compiled once;
  this includes the second egraph reusing the cache). The cold `cargo build`
  itself dominates — it compiles timely + differential-dataflow + the generated
  engine. (MILESTONE1.md's ~45s figure was a fully-cold DD/timely build; here
  the host machine's cargo registry/deps were already warm.)
- **Warm (cache hit):** ~0.01–0.02s for the whole test — the binary is reused
  and only spawned + driven.
- **IPC cost:** negligible. 8 `commit`s across two egraphs complete in ~0.02s
  warm; the line protocol + flush-per-command is not a bottleneck at this scale.
- **Disk:** one cache entry is ~265M (its own `target/` with the DD/timely build
  graph). Pruning keeps at most `MAX_CACHE_ENTRIES = 8` entries (LRU by mtime),
  so runtime-compile artifacts do **not** accumulate unbounded. All artifacts
  live under a single root: `$EGGLOG_FLOWLOG_CACHE` or
  `<system tmp>/egglog-flowlog-cache`. Each driver crate sets
  `CARGO_TARGET_DIR` to its own `target/` under that root.

## Subprocess lifecycle

- The driver is **spawned once per rule-set** (`DriverHandle::spawn`), with
  piped stdin/stdout and inherited stderr. The flowlog engine inside stays
  **warm for the whole program**, so incrementality is preserved across every
  `commit` (each commit applies only the freshly-staged delta).
- One `run_rules` = one `commit` over the pipe = one egglog iteration (the M1
  per-iteration model, now across a process boundary).
- The host keeps the same bounded **host-feedback loop** as M1: first round
  stages all `edge` + `path` rows; each later round re-stages only the previous
  round's NEW `path` rows (`EGraph::pending_path`). N rounds = N bounded hops.
- `Drop for DriverHandle` sends `quit` and reaps the child, so timely worker
  threads / zombie processes don't leak across egraphs.

## What is stubbed / deferred (correctly out of scope for M2)

- **Rule shape:** the runtime translation currently emits the **non-recursive
  single-join step** (`head(x,z) :- path(x,y), edge(y,z)`), recognized from the
  IR by `recognize_step` (reused from M1). Arbitrary multi-atom bodies,
  primitives in the body, multiple rules per `run_rules`, and non-2-arity
  projections are not yet translated (they error clearly at `run_rules`). The
  IR (`compile.rs`) already records the richer ops; extending `emit_dl` +
  driver dispatch is the natural next step.
- **Retraction / union-find rebuild (M3):** the protocol carries `remove` and
  signed `diff`s end-to-end, and the generated engine has `remove_<rel>`, but
  the host folds only `diff > 0` (monotone, as M1). Rebuild reuses Feldera M2's
  retraction pattern, per the brief — deferred to M3.
- **Full frontend integration (M3):** the proof drives the trait directly; it is
  not yet wired into the egglog CLI/schedule.
- Containers, complex merges, proofs, push/pop (`clone_boxed`), primitives:
  unchanged from M1 (stubbed/error).

## New trait-friction observed

- **No "schema frozen" / "rules installed" signal.** The trait installs rules
  one at a time (`new_rule`/`build`) and the first `run_rules` is where we learn
  the rule-set; there is no hook saying "all rules are in, freeze and compile."
  M2 compiles lazily on the first `run_rules` (which works because the frontend
  installs rules up front), but a `freeze_rules()` / `seal()` hook would make
  the compile point explicit and let multi-rule rule-sets compile into one `.dl`
  deterministically. This is the same friction flagged in M1, made sharper by
  the `rustc` boundary.
- **`run_rules` can't surface "I had to recompile" latency.** A cold rule-set
  triggers a multi-second `cargo build` inside `run_rules`; the trait has no way
  to report "this call paid a compile cost," so a caller can't distinguish a
  slow first iteration from a hang. A progress/affordance channel would help.
- **`FunctionId` ↔ relation-name mapping is host-private.** The trait gives
  `FunctionId`s; the `.dl`/driver need stable relation names. M2 maps the three
  recognized roles to fixed names (`edge`/`path`/`hop`) in `StepShape`; a
  general translation will need the table names the frontend used (available via
  `add_table`'s `FunctionConfig.name`, which the backend already stores).
- Everything else matches the M1 surface; no `egglog-backend-trait` changes were
  required, and duckdb + bridge still build.
