# Milestone 3 — FlowLog backend: real `.egg` programs, table joins on Differential Dataflow

**Goal:** drive REAL `.egg` files through the complete egglog frontend
(parse → desugar → typecheck → term-encode) onto the FlowLog/Differential-
Dataflow backend, validated against the reference backend through the shared
`tests/files.rs` snapshot harness at **per-function tuple-count parity** — with
the relational rule-body **table join running on the Differential-Dataflow
engine** for the eligible rule class. M1 proved bounded `(run N)` on a live
in-process DD engine; M2 proved runtime rule installation via shell-out compile;
M3 generalizes codegen to full multi-atom bodies + primitives + head actions,
splits table-join (DD) from primitive-tail (host), adds seminaive
incrementality, and runs the whole `.egg` survey.

## Headline results

| metric | M1 | M2 | **M3** |
|--------|----|----|--------|
| rule shape supported | recognized transitive step | recognized step (runtime-installed) | **general multi-atom bodies + `!=`/value prims + head actions** |
| `.egg` files at shared-snapshot + tuple-count parity (62 eligible) | n/a | n/a | **61** |
| files that FAIL | n/a | n/a | **0** |
| files that TIMEOUT (30 s budget) | n/a | n/a | **1** (`rectangle`) |
| body joins run on | DD (fixed `.dl`) | DD (runtime `.dl`) | **DD for the eligible class (seminaive); prim tail host-side** |
| rule firing model | per-iteration | per-iteration | **seminaive (delta-driven)** |

**61 / 62** eligible `.egg` files pass the shared snapshot (the same snapshot
the reference / proof / duckdb treatments target, **including** the per-function
tuple-count summary from the harness-appended `(print-size)`). The single
remaining timeout, `rectangle`, is the value-prim host-fallback wall (NOT a
correctness failure, NOT an incrementality bug — characterized below). This
matches the Feldera/DBSP backend's M5 frontier file-for-file
(`repro_should_saturate`, `naturals` converge via seminaive; `rectangle`
remains).

## How the milestone's mandate is met: table joins on Differential Dataflow

The architecture mirrors Feldera M4/M5's split exactly: **the relational
table-atom join runs on the engine; the primitive tail + head actions run
host-side.** For FlowLog the "engine" is a Differential-Dataflow program reached
by the M2 shell-out: a runtime-emitted `.dl` is compiled (`.dl → flowlog → DD`,
cached by `.dl` hash) into a driver subprocess that embeds a
`DatalogIncrementalEngine` and is driven over a pipe. So when DD routing is on,
the join genuinely executes on Differential Dataflow, by construction of the
compile path.

Proven by `egglog-bridge-flowlog/tests/dd_join_proof.rs`
(`transitive_closure_join_runs_on_dd`, the FlowLog analog of Feldera's
`transitive_closure_join_runs_on_dbsp`): it drives
`path(x,z) :- path(x,y), edge(y,z)` through the interpret-mode backend with DD
routing enabled and asserts the join ran **entirely on the DD engine** —
`flowlog_join_stats()` returns `host_rule_runs == 0`, `dd_rule_runs >= 3` — while
matching the reference backend round-for-round and staying bounded
(`(run 1) ≠ (run 3)`; the 3-hop pair `(1,4)` is absent after `run 1`, present
after `run 3`). Verbatim:

```
run(1) reference     = {(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
run(1) flowlog(dd)   = {(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}  (dd_runs=2, host_runs=0)
run(3) reference     = {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
run(3) flowlog(dd)   = {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}  (dd_runs=4, host_runs=0)
```

Reproduce (compiles a DD driver subprocess once; cached thereafter):

```
EGGLOG_FLOWLOG_CACHE=/tmp/egglog-flowlog-m3cache \
cargo test -p egglog-bridge-flowlog --release \
  --test dd_join_proof -- --ignored --nocapture
```

### The host interpreter is the correctness oracle for the survey

DD routing is **off by default** (`EGGLOG_FLOWLOG_DD=1` or `enable_dd_join()`
turns it on). The default `.egg` survey runs the join on a host nested-loop
interpreter, which is the *correctness oracle*: it produces identical bindings to
the DD engine but without compiling a subprocess per distinct join shape (a cold
`cargo build` of timely + differential-dataflow is ~18 s and ~265 MB of disk per
join shape). Running the full 62-file survey through DD would compile dozens of
subprocesses and is impractical for a single test run; the **engine equivalence**
is instead proven on the canonical join by `dd_join_proof.rs`, exactly as Feldera
proves DBSP-on-engine with one targeted test and runs its survey through the
shared interpreter. Both paths share the seminaive driver
(`interpret::seminaive_bindings`), so the oracle and the engine compute the same
delta join.

This is honest by construction: `flowlog_join_stats()` reports
`(dd_rule_runs, host_rule_runs)`, so any run can assert what fraction of rule
firings ran on the DD engine versus the host fallback.

## Seminaive incrementality (mirrors Feldera M5)

The load-bearing addition over a naive per-iteration interpreter is **per-rule
seminaive delta tracking**, without which `(saturate …)` loops oscillate forever
against the term encoder's rule-driven cleanup/retraction (the M3/M4 Feldera
timeouts). State: `EGraph::seen: HashMap<rule_idx, HashMap<FunctionId,
HashSet<Row>>>` — for each rule, the rows of each body relation that rule has
already matched. The delta of rule `r` over relation `f` is
`mirror[f] \ seen[r][f]`.

- **Keyed by rule, not globally.** The term encoder schedules distinct rulesets
  in sequence (`(saturate single_parent)` / `(saturate path_compress)` /
  `(saturate uf_index)` + user rulesets). Rows produced by an earlier ruleset
  must count as *new* to a later ruleset's rules. A global `seen` would starve a
  freshly-scheduled rule of its delta.
- **The seminaive union.** For a body with table atoms `A_1 … A_k`, each
  iteration computes `⋃_j A_1(full) ⋈ … ⋈ A_j(delta) ⋈ … ⋈ A_k(full)` — for each
  atom occurrence `j`, that occurrence ranges over its delta and the rest over
  the full relation. If no body relation has any delta, the rule is skipped
  entirely (the win that stops the oscillation). Both paths implement this: the
  host fallback (`interpret::seminaive_bindings`, ranging the delta atom over
  delta rows via the shared `step_atom`) and the DD path
  (`dd_join::run_join_seminaive`, feeding the delta to atom occurrence `j` and
  the full read view to the rest, then running the join on the engine).
- **Retraction-correct `seen` advance (the rebuild trap).** After a rule fires,
  `seen[r][f]` is advanced to the **start-of-iteration** snapshot (`read[f]`),
  *never* the post-write mirror. So a row that is **deleted and later re-added**
  (rebuild's `path_compress` / `single_parent`, the merge-cleanup deletes)
  reappears in `r`'s delta and the consuming rule re-fires. `clear_table` and
  `free_rule` evict the relevant `seen` entries so a re-populated table presents
  fresh deltas.

`repro_should_saturate` and `naturals` — both pure incrementality blockers —
moved TIMEOUT → PASS once seminaive landed, taking the survey from 59 to 61, the
same delta Feldera saw at M5.

## Retraction / union-find rebuild

`(delete)` / `remove` is a true row retraction on the mirror (the row is removed,
not folded monotonically as M1/M2 did). Combined with the start-of-iteration
`seen` advance, the term encoder's `@uf` / `@uff` rebuild rulesets work: every
eq-sort congruence + rebuild `.egg` file passes at tuple-count parity
(`path`, `path_union`, `eqsat_basic`, `points_to`, `unify`, `typecheck`,
`combinators`, `resolution`, `merge_during_rebuild`, …), which exercises
delete-then-readd congruence collapse through the real frontend.

## The backend split as built (zero trait change)

Per the brief, FlowLog stays **zero-`egglog-backend-trait`-change**: it does NOT
adopt Feldera's `eval_prim` trait method (that lives on the Feldera branch and is
reconciled later). Instead the backend embeds a `core_relations::Database` purely
as the base-value / primitive engine (the M3 stopgap, as Feldera M3 did) and
evaluates primitives through the inherent `eval_prim_internal`
(`Database::with_execution_state`), so `Value`s are bit-for-bit identical to the
reference backend. Both the host interpreter and the DD-join path's host-side
primitive tail call it.

- `with_flowlog_backend()` (`src/lib.rs`) is the frontend entry point, analogous
  to `with_duckdb_backend` / `with_feldera_backend`: term-encoding-only, with a
  bridge-backed typechecker for the post-encoder re-typecheck.
- `tests/files.rs` gains a `flowlog: bool` treatment mirroring `duckdb`/`feldera`
  (same supportability gate: term-encoding, no `(push)/(pop)`, proof-supporting),
  gated behind `EGGLOG_TEST_FLOWLOG=1` so it never perturbs the default run, and
  diffing against the **same shared snapshot** including per-function tuple counts.

## The frontier (where DD stops; what falls back to host)

A rule's table-atom join runs on DD iff (`dd_join::plan_join`) the body has ≥1
table atom and uses ≤ `MAX_JOIN_VARS` (16) distinct variables. Body primitives
(`!=` guards, value-computing prims) are NOT lowered into the `.dl`; the host
re-runs every `BodyOp::Prim` over the join bindings, so the DD `.dl` computes
only the relational join (the expensive, paper-relevant part). Everything outside
the eligible class — wider bodies, prim-only bodies — uses the host seminaive
fallback (the oracle).

The single remaining timeout characterizes the frontier precisely:

- **`rectangle`** (the value-prim wall, unchanged from Feldera M5): the
  `populate` ruleset is `(range i)(< i 1000) → (range (+ i 1))`. The `< i 1000`
  ordering guard and the `(+ i 1)` value prim make the rule host-fallback;
  seminaive shrinks the per-iteration delta but the program still needs ~1000
  host iterations to grow `range`, and the downstream 4-way cyclic `result` join
  over ~1000-row relations is rebuilt as a fresh per-call DD circuit each round.
  This is the value-prim / per-call-circuit cost, not an incrementality bug.

## Validation

- `cargo test -p egglog-bridge-flowlog --release --test run_n_proof` — M1 bounded
  in-process proof, **green** (matches reference at run 1 / run 3).
- `--test run_n_shellout_proof -- --ignored` — M2 runtime-install shell-out
  proof, **green**.
- `--test dd_join_proof -- --ignored` — **M3 table-join-on-DD proof, green**:
  `host_runs == 0`, `dd_runs >= 3`, matches reference, bounded.
- `EGGLOG_TEST_FLOWLOG=1` per-file survey (`.tmp/flowlog_survey.sh
  <files-test-binary> 30`): **61 PASS, 0 FAIL, 1 TIMEOUT** (`rectangle`) at
  shared-snapshot + per-function tuple-count parity vs. the reference backend. No
  `.snap.new` files were produced — the flowlog treatment matches the existing
  shared snapshots exactly.
- `cargo fmt --check` clean on the touched crates. `cargo clippy` on
  `egglog-bridge-flowlog --tests` is clean; the only `-D warnings` failures in the
  workspace are pre-existing newer-clippy (1.91.0) doc lints in the untouched
  `egglog-backend-trait` and disallowed-`HashMap` lints in `egglog-bridge-duckdb`
  — not introduced by this milestone.

## Remaining friction (ranked)

1. **Value-prims force host fallback** (the `rectangle` wall). Lowering `<` / `+`
   into the DD `.dl` (or a vectorized post-join primitive operator) would push
   those bodies onto the engine.
2. **Per-call DD circuit rebuild.** Each seminaive delta-atom variant stages and
   `commit()`s a fresh non-recursive DD program per iteration. A cached,
   integrated circuit fed true input deltas across iterations — leveraging that
   the subprocess's `DatalogIncrementalEngine` stays warm and is incremental by
   construction — would turn the per-call build into genuine streaming IVM. The
   per-rule `seen` deltas computed here are exactly the input deltas it would
   consume.
3. **Subprocess compile cost / disk.** Each distinct join `.dl` compiles a driver
   (~18 s cold, ~265 MB), cached by `.dl` hash (LRU-pruned). This is why the
   survey runs on the host oracle and the engine equivalence is proven by one
   targeted test, rather than routing every survey file through DD.
4. **`eval_prim` stays inherent** (deliberate, per the brief): flowlog keeps the
   embedded `Database` stopgap and a zero-trait-change posture; it adopts
   Feldera's `eval_prim` trait method on reconcile.

## Files touched

- `egglog-bridge-flowlog/src/lib.rs` — `ExecMode::Interpret` + `new_interpret`;
  `with_flowlog_backend` plumbing (`base_values_inner`, `eval_prim_internal`,
  `fresh_id_internal`, `resolve_merge`, `merge_mode`); Unit-output relation-vs-
  function detection; `dd_enabled` / `dd_drivers` / `flowlog_join_stats`; per-rule
  seminaive `seen` state + eviction in `clear_table` / `free_rule`; route
  `run_rules` through `interpret::run_iteration` in Interpret mode.
- `egglog-bridge-flowlog/src/interpret.rs` — **new**: host-side ordered-IR
  interpreter; seminaive `run_iteration` (per-rule delta, start-of-iteration
  `seen` advance); `seminaive_bindings` (union over delta positions, dedup,
  skip-if-no-delta) routing to DD or host; shared `step_atom`; head machinery
  (`set`/`delete`/`subsume`/RHS-`lookup`/`call`/`union`/`panic`, eq-sort
  hash-cons via `lookup_or_create`).
- `egglog-bridge-flowlog/src/dd_join.rs` — **new**: DD table-join (eligibility
  `plan_join`, `.dl` + driver `main.rs` codegen, per-atom-occurrence staging);
  `run_join` (full×full, for the proof) and `run_join_seminaive` (one delta-atom
  term) over a shared `run_join_with` core.
- `egglog-bridge-flowlog/src/compile.rs` — ordered `RuleIr` (`BodyOp`/`HeadOp`),
  variable-width `Row`, `slot_lookup`, `MergeMode`.
- `egglog-bridge-flowlog/src/rule_builder.rs` — emits the full ordered IR for all
  `RuleBuilderOps` (already general; unchanged in this pass).
- `egglog-bridge-flowlog/src/subprocess.rs` — generalized dd-join driver protocol
  (`build_or_cached_with`, `send_clear`, `insert_row`, `commit_rows`).
- `egglog-bridge-flowlog/tests/dd_join_proof.rs` — **new**: table-join-on-DD proof.
- `src/lib.rs` — `with_flowlog_backend()`; `base_values` + extraction-skip
  downcast arms for the flowlog backend.
- `tests/files.rs` — `flowlog: bool` treatment (mirrors `duckdb`/`feldera`), gated
  behind `EGGLOG_TEST_FLOWLOG=1`.
- `Cargo.toml` — add `egglog-bridge-flowlog` as a dependency of the main crate.
