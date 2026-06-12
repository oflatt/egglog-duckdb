# SPIKE: in-process raw differential-dataflow for the FlowLog backend

Branch: `spike-flowlog-dd` (worktree `.perf-wt/flowlog-dd`). No git ops performed.

## Verdict: **YES**

In-process raw `differential-dataflow` 0.24.0 + `timely` 0.30.0 CAN evaluate
egglog's term-encoded rule IR **incrementally across epochs, including
rebuild/retraction, bit-exact** with the reference backend — bypassing
flowlog-rs's build-time `.dl` codegen entirely. This is the FlowLog analog of
what the feldera backend does with DBSP's runtime `RootCircuit::build`, and the
two are structurally near-identical (both signed-weight incremental dataflows).

The smallest end-to-end demonstration: the **full `tests/math-microbenchmark.egg`
at `(run 2)` and `(run 3)`** — a real eqsat workload with congruence closure,
unions, and the complete `@uf` rebuild fixpoint — produces **bit-exact
per-function tuple counts** vs the reference backend, with every body join
running on the in-process DD dataflow.

## What was built

- `egglog-bridge-flowlog/src/dd_native.rs` (NEW, ~470 lines): a
  `PersistentDdJoin` that owns a single-threaded `timely::worker::Worker`,
  builds ONE non-recursive DD dataflow per rule (`worker.dataflow(...)`), one
  `InputSession` per body atom occurrence, a left-deep chain of DD `.join`s with
  per-row constant/repeated-var/shared-var constraints in `.flat_map`,
  `.distinct()` for set semantics, and output captured via raw timely
  `.inspect_batch` into a shared `Rc<RefCell<Vec<(Row, isize)>>>`, probed for
  fixpoint detection. Row model: fixed `[u32; 16]` (DD `Data` needs
  `Sized+Ord+Hash`; an array gives that without dbsp's `declare_tuples!`).
- `interpret.rs`: a new `run_iteration_dd_native` + `dd_native_bindings` path,
  gated behind `eg.dd_native_enabled` (`EGGLOG_FLOWLOG_DD_NATIVE=1`), mirroring
  feldera's `run_iteration`/`persistent_bindings`. Computes the per-rule
  per-relation signed delta vs last-fed rows, steps the persistent join,
  re-runs body prims host-side over the bindings, applies heads + FD-merge with
  the exact same write logic as the default host path (so results are bit-exact).
- `lib.rs`: `dd_native_enabled` / `dd_native` (per-rule joins) / `dd_native_fed`
  (last-fed snapshots) fields on `EGraph`; module registration.
- `Cargo.toml`: `differential-dataflow = "=0.24.0"`, `timely = "=0.30.0"` as
  DIRECT deps at the EXACT versions already in `Cargo.lock` (pulled by
  flowlog-runtime) — confirmed no duplicate copies compiled.

The DEFAULT host interpreter and the existing shell-out `dd_join.rs` path are
**untouched** (flag-gated new path only). Confirmed: `--flowlog` without the flag
is still bit-exact on math N=3.

## Evidence for the 5 crux questions

**1. Build-once + drive-epochs feeding only deltas works in-process.** Yes.
Unit test `dd_native::tests::dd_native_join_is_incremental` builds once and steps
3 epochs feeding only deltas. End-to-end: bounded transitive closure
`path(x,z) :- path(x,y), edge(y,z)` gives `(run 2)` → 7 paths, `(run 4)` → 10
paths (full closure), BOTH bit-exact with reference — two different `(run N)`
producing correctly different bounded results.

**2. The DD join maintains its arrangement across epochs (genuinely
incremental).** Yes. The `InputSession`s are NEVER cleared — only the per-epoch
delta is `update()`d, so the DD arrangements persist. Evidence from
`FLOWLOG_DD_NATIVE_TRACE=1`: on bounded TC, every epoch's `input_delta_rows` is
1–8 and the `output_binding_delta` tracks the INPUT delta (`pos=1`, not the
growing integral). Epoch K does only delta·integral work, never a full
recompute — this is the anti-pattern the shell-out path (clear+restage = O(state)
every call) commits and this spike avoids. Reasoning from DD semantics confirms
it: `.join` on persistent arrangements emits `δL⋈R + L⋈δR + δL⋈δR`.

**3. Retraction is bit-exact.** Yes — THE KEY RISK, cleared. The tiny union
program (`(rewrite (Add a b) (Add b a))` + assoc) lowers `union` to `@uf`
relational writes + rebuild rules that DELETE non-canonical rows. Under the DD
path these become negative-weight input deltas. Trace shows **22 epochs with
negative-weight outputs** (e.g. `pos=1 neg=4`) — retractions flowing through DD's
signed weights — and the final `(print-size)` is `Add=12, Lit=3`, bit-exact with
reference. The full math-microbenchmark (heavy rebuild) is bit-exact at N=2/3
across all 13 functions. DD's signed weights handle the retract+rewrite fixpoint
natively; this was the original "engine does the join is hard" semantic worry and
it is a NON-issue with raw DD.

**4. Primitives evaluable inline.** Yes (demonstrated, host-side tail). A rule
`summ(a, a+b) :- seed(a, b)` with the `+` arithmetic prim runs on the DD path:
`summ` size 3 and all `(check ...)` pass, bit-exact. In the spike, the relational
join runs on DD and the prim tail is re-run host-side over the bindings (exactly
the existing `dd_join` split, and feldera M1–M3's split). Putting prims
ON-circuit (feldera M4/Stage-C `apply_steps` via a shared `PrimEngine` closure in
`.flat_map`) is a direct port and is the recommended Stage 4 below.

**5. Fixpoint structure.** EXTERNAL epoch-drive (host loop advances epochs;
non-recursive dataflow). Chosen because it matches egglog's bounded `(run N)`
fire→rebuild→repeat and sidesteps DD `iterate()`'s monotonicity constraints under
retraction (a rebuild RETRACTS rows, which `iterate()` cannot express cleanly).
One epoch = one bounded hop = one `run_rules` call — identical to feldera.

## Hard parts hit & how solved

- **Owning a steppable timely worker.** `timely::execute*` spawns threads and
  blocks; I instead construct `Worker::new(WorkerConfig::default(),
  Allocator::Thread(Thread::default()), Some(Instant::now()))` directly, build the
  dataflow with `worker.dataflow(...)`, and `worker.step_while(|| probe.less_than(&epoch))`
  between host calls. The worker is held in the `EGraph` (single-threaded; the
  egraph already asserts `unsafe impl Send+Sync`).
- **Reading per-epoch output deltas.** DD `Collection` has no
  non-integrated-output handle like DBSP's `OutputHandle`. Solution: drop to the
  raw timely stream (`out.inner`) and `inspect_batch` into a shared
  `Rc<RefCell<Vec>>`, cleared each epoch, then `probe_with` a `ProbeHandle` for
  fixpoint. `consolidate` + host-side accumulation collapses multiplicities.
- **API surface (DD 0.24/timely 0.30).** `join`/`distinct`/`consolidate` are
  inherent on `Collection` (no trait import); `inspect_batch`/`probe_with` need
  the timely `Inspect`/`Probe` traits; `Worker` is not generic over `Allocator`;
  `Collection` ops consume `self` (clone the input collections).
- **Retraction correctness.** Solved for free by DD signed weights + feeding the
  exact `+1/-1` set-diff vs last-fed rows (the feldera `fed` pattern). Negative
  binding deltas are integral bookkeeping (a body row retracted); egglog heads are
  monotone-fire so we do NOT re-fire on disappearance — the mirror row stays until
  an explicit `@uf` `remove` head retracts it, which it does, bit-exactly.

## Staged plan for full coverage

| Stage | Scope | Effort | Risk |
|---|---|---|---|
| **1. Widen row + var cap** | Replace `[u32;16]` with a generated wide row (or `Vec`-keyed via a newtype impl'ing DD `Data`) so rules with >16 vars don't panic. Survey peak is ~23 vars. | S | Low — feldera proved `declare_tuples!`-style 32-wide works; DD `Data` is `Ord+Hash+Clone`, an array or small newtype suffices. |
| **2. On-circuit `!=`/guard inlining** | Lower recognized rep-comparison prims to DD `.filter` inside the join (feldera `Cond`/`apply_guards`). Removes the host prim re-run for guards. | S | Low — direct port; pure rep arithmetic. |
| **3. Multi-occurrence self-join fan-out + join-order** | The `occ_of_func` fan-out exists; verify N-way self-joins and pick a non-naive left-deep order for wide bodies. | M | Med — correctness is fine (proven on the 2-atom self-join); perf of bad orders. |
| **4. On-circuit value prims (`+`, `int-div`, `string-concat`, …)** | Port feldera Stage-C `apply_steps`: a shared `PrimEngine` (`Arc<Mutex<Database>>`) captured into `.flat_map` closures, re-evaluating the REAL prim on-circuit. Removes the host tail entirely. | M | Med — needs the prim engine threaded into the DD closures; feldera has the exact blueprint. |
| **5. Full rebuild / retraction at scale** | Already correct on math-microbenchmark; validate the whole `@uf`/`@uff` rebuild ruleset family on the suite (luminal, eggcc, cykjson). | M | Med — semantics proven; watch row-width cap and self-join blowup. |
| **6. Extraction + read-back** | Unchanged — extraction reads the Rust mirror, which the DD path keeps bit-exact. Wire `print-size`/`extract` parity into `tests/files.rs` under a new treatment gate. | S | Low. |
| **7. Replace the host interpreter join entirely** | Make the DD path the ONLY join path (retire the nested-loop + `dd_join` shell-out), as feldera did. Requires Stages 1–4 complete (no graceful fallback). | L | Med-High — every rule shape must be DD-expressible or explicitly atom-less-fired; the panic-on-unsupported becomes a hard correctness gate. |

**Rough total: M–L.** Stages 1–2 + 6 are quick wins; Stage 4 (on-circuit prims)
and Stage 7 (sole join path) are the real work. The feldera `dbsp_join.rs` is a
line-for-line template for Stages 1, 2, 4, and 7.

## Walls? None fundamental.

No wall makes full coverage impractical. The two things to budget for: (a) a wide
fixed-arity DD `Data` row (feldera solved this with `declare_tuples!`; DD needs
only `Ord+Hash+Clone`, easier than DBSP's rkyv stack), and (b) the per-rule timely
`Worker` cost — each rule owns a worker + dataflow. For programs with hundreds of
rebuild rules this is many workers; a single shared worker hosting all rule
dataflows (one `worker.dataflow` per rule on the same worker) is the obvious
consolidation and is a Stage-3 refinement, not a blocker. Memory/threading posture
matches the sibling backends (single-threaded, `unsafe impl Send+Sync`).

## How to reproduce

```
cd .perf-wt/flowlog-dd
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 cargo build --release --bin egglog
cargo test -p egglog-bridge-flowlog --release dd_native   # unit: build-once/incremental/retraction

# bit-exact end-to-end (math-microbenchmark trimmed to (run 2)/(run 3) lives in .tmp_math_n2.egg):
diff <(./target/release/egglog .tmp_math_n2.egg) \
     <(EGGLOG_FLOWLOG_DD_NATIVE=1 ./target/release/egglog --flowlog .tmp_math_n2.egg)   # identical

# incrementality + retraction trace:
FLOWLOG_DD_NATIVE_TRACE=1 EGGLOG_FLOWLOG_DD_NATIVE=1 ./target/release/egglog --flowlog .tmp_math_n2.egg 2>&1 | grep dd_native
```
