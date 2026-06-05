# Milestone 3 — Feldera/DBSP backend: real `.egg` programs through the full egglog frontend

**Goal:** drive REAL `.egg` files through the complete egglog frontend
(parse → desugar → typecheck → term-encode) onto the Feldera backend, and
validate against the reference backend through the existing `tests/files.rs`
shared-snapshot harness, targeting **per-function tuple-count parity**.

## What runs end-to-end

`EGraph::with_feldera_backend()` (in `src/lib.rs`) is the analog of
`with_duckdb_backend`: it builds an `egglog_bridge_feldera::EGraph`, wraps it in
the `Backend` trait, and (like DuckDB) runs term-encoding-only with a
bridge-backed typechecker for the post-encoder re-typecheck. `tests/files.rs`
gains a `feldera: bool` treatment that mirrors the `duckdb` one: it shares the
**same** snapshot every other treatment targets, including the per-function
tuple-count summary.

### Goal 2 — per-function tuple-count parity (already in the shared snapshot)

The harness appends `(print-size)` to every test file. `(print-size)` emits a
`PrintAllFunctionsSize(Vec<(name, size)>)` **sorted by function name**, and that
output flows through `outputs_to_snapshot_preserved_across_treatments`
unfiltered (only `OverallStatistics` timing and extraction outputs are dropped).
So the shared snapshot **already encodes a deterministic per-function
tuple-count summary** that every treatment — normal / proof / duckdb / feldera —
must agree on. Example (`tests/snapshots/files__shared_snapshot_path.snap`):

```
((edge 3)
 (path 6))
```

A feldera run that produces a different size for `edge` or `path` fails the
shared snapshot. **No new mechanism was needed** for Goal 2; the milestone
framing predated this existing behavior. The contribution here is making the
feldera treatment *subject* to that same parity check.

## The backend: host-side rule interpreter over the embedded value engine

Milestones 1–2 ran rules as a non-recursive DBSP circuit with host-side
feedback (per-iteration stepping + retraction-rebuild). Milestone 3 needs
**primitives to actually evaluate** — the egglog frontend's term encoding and
the default scheduler both emit `call_external_func` / `query_prim` /
RHS-`lookup` ops, and a DBSP `map`/`filter` closure (`Send + 'static`) cannot
call back into a primitive's `invoke(&mut ExecutionState, …)`. So M3:

1. **Embeds a `core_relations::Database`** in the Feldera `EGraph` purely as the
   **base-value + primitive engine.** It owns the `BaseValues` registry (so
   `Value`s are bit-for-bit identical to the reference backend) and the
   registered external functions. `register_external_func` registers into the
   Database (ids stay aligned with the local name/panic side-table), and
   primitives are invoked host-side via `Database::with_execution_state`.
2. **Replaces the per-subset DBSP circuit with a host-side ordered-IR
   interpreter** (`egglog-bridge-feldera/src/interpret.rs`). A rule is an ordered
   list of body ops (table-atom matches + primitive evals) and head ops
   (`set`/`delete`/`subsume`/RHS-`lookup`/RHS-`call`/`union`/`panic`). One
   `run_rules` call = one bounded iteration: snapshot the mirror, run a
   nested-loop join over the body, evaluate primitives against the Database,
   apply head actions, then resolve functional-dependency conflicts per each
   function's merge mode (the M2 `Old`/`New`/`Min` recognition is retained).

The M1/M2 narrative (bounded per-iteration stepping; retraction-rebuild;
`@uf`/`@uff` min-merge congruence collapse) is preserved at the **semantics**
level: the host interpreter reproduces the DBSP results exactly, and both proof
tests (`run_n_proof`, `rebuild_proof`) stay green — including the load-bearing
diamond / star+chain congruence-collapse scenarios. The row representation moved
from a fixed `Tup8` (DBSP zset element) to a variable-width `Box<[u32]>`, lifting
the arity-8 cap. Integrating the DBSP incremental join *under* this interpreter
(so the relational join runs in DBSP and only the primitive tail is host-side)
is deferred to M4 — it is a performance refactor, not a correctness one, and the
DoD signal is tuple-count parity.

## Results: 59/62 files passing with tuple-count parity

Of the **62** `.egg` files eligible for the `feldera` treatment (the same gate
as the plain-`duckdb` treatment: term-encoding-only, proof-supporting, no
`(push)/(pop)`), driven through the full frontend onto the Feldera backend and
diffed against the **shared snapshot** (including the per-function tuple-count
summary from the appended `(print-size)`):

| outcome | count |
|---------|-------|
| **PASS** (shared snapshot + tuple-count parity) | **59** |
| FAIL    | **0** |
| TIMEOUT (25s budget) | **3** |

Reproduce (per-file, timeout-guarded):
`EGGLOG_TEST_FELDERA=1 cargo test -p egglog --release --test files -- feldera`
(or `.tmp/feldera_survey.sh <files-test-binary> 25`).

Example passing programs, spanning the tractable class and well beyond it:

- **i64 / f64 / string base values + primitives:** `i64`, `f64`, `string`,
  `bitwise`, `integer_math`, `primitives`, `repro_new_backend_prims`.
- **eq-sort datatypes + rewrites + congruence + run/saturate:** `path`,
  `path_union`, `eqsat_basic`, `eqsolve`, `antiunify`, `birewrite`,
  `combinators`, `matrix`, `resolution`, `rw_analysis`,
  `points_to`, `stratified`, `fibonacci_demand`, `intersection`.
- **merges (incl. complex/lattice):** `complex_merge_func`, `merge_saturates`,
  `merge_during_rebuild`.
- **schedules:** `until`, `test_combined`, `test_combined_steps`,
  `schedule_demo`, `repro_typechecking_schedule`.
- **`(check …)` (side-channel external-func path), `(extract …)` (skipped like
  duckdb), `(print-size)`, relations, globals, `repro_*` regression files.**

Every PASS matched the reference backend's **per-function tuple counts exactly**
(that summary is part of the shared snapshot), which is the milestone's
definition-of-done signal.

## Failure frontier (ranked)

Only **3** files are unresolved (`rectangle`, `naturals`, `repro_should_saturate`),
none a correctness FAIL. All three still time out at a **90-second** budget — and
since `repro_should_saturate` is a 7-line program and `rectangle` 58 lines, this
points at **non-convergence** (`(saturate …)` never reaching a fixpoint), not raw
slowness:

1. **Merge-driven non-convergence** (`repro_should_saturate`:
   `(function MyMap () i64 :merge (min old new))` with a rule that `set`s both 1
   and 2 every iteration; `rectangle`, `naturals` appear to share the shape).
   These still time out at a **60-second** budget, so they are not converging.
   The interpreter applies functional-dependency merge resolution *per
   iteration* against a read-snapshot of the mirror, rather than the reference's
   *atomic flush-time* merge; under `(saturate …)` the set/merge interaction (the
   term encoder routes a custom `:merge` through a generated merge ruleset)
   keeps reporting progress. A determinism fix for the conflict tiebreak (fold
   rows in sorted order, see `resolve_merge`) was added to remove a latent
   nondeterminism in snapshot resolution, but it does NOT by itself make these
   converge — matching the reference's flush-time merge semantics is the M4 item.

2. **Host-interpreter performance on heavy saturation** (secondary). The
   interpreter clones the full relation mirror as the per-iteration read view and
   runs a nested-loop (hash-indexed) join with **no semi-naive incrementality** —
   it re-derives all matches every iteration. This is fine for the 59 passing
   programs but is the obvious scaling limit, and is exactly what M4's
   DBSP-incremental-join integration is meant to remove (run the relational join
   in DBSP under the host primitive tail).

What is **NOT** on the frontier (these now work): i64/f64/string base values and
their primitives, eq-sort congruence + rebuild via the term encoder's `@uf` /
`@uff` rulesets, `(check)`, complex/lattice merges, arity > 8 (the row is now
variable-width), and run/saturate/combined schedules.

Not yet attempted (gated out of the treatment, same as duckdb): proof mode,
`(push)/(pop)`, container sorts, subsumption-dependent checks.

## New trait-friction (for the pre-FlowLog refactor)

1. **Primitive invocation has no backend-agnostic entry point.** To evaluate a
   primitive a backend must own a `core_relations::Database` (or reimplement
   `ExecutionState`), because `ExternalFunction::invoke` takes
   `&mut core_relations::ExecutionState`. DuckDB sidesteps this by translating
   prims to SQL by name; Feldera had to embed a whole Database just to call
   `invoke`. A trait method like `eval_prim(id, &[Value]) -> Option<Value>` (or a
   pure-closure primitive form) would remove the per-backend engine.

2. **External-func id alignment is implicit.** `register_external_func` must
   return an id the *frontend* then uses in `call_external_func`. We keep the
   Database's `DenseIdMapWithReuse` and a local side-table in lockstep by hand.
   A backend that stores primitives elsewhere has to replicate this aliasing.

3. **`query_prim`'s bind-vs-guard is positional.** The last entry is the return
   slot; whether it *binds* or *asserts equality* depends on whether the var is
   already grounded — a fact the trait does not surface, so the backend
   re-derives it at interpret time. A first-class "filter atom" vs "compute atom"
   distinction would be cleaner (refines M2 friction #4).

4. **`run_rules(&[RuleId])` ruleset identity is still rediscovered each call**
   (carried from M1/M2). With the interpreter the cost is just collecting the
   live rule subset, but a stable ruleset handle would still help.

5. **`get_canon_repr` / `union` are inert under term encoding.** Both the DuckDB
   and Feldera term-encoding paths never consult trait `union` /
   `get_canon_repr` (the encoder lowers unions to `@uf` `set`s and canonicalizes
   via view tables). The trait surface for native union-find is unused dead
   weight on these backends; a "term-encoding backend" capability flag could let
   the frontend skip wiring it.

## Files touched

- `src/lib.rs` — `EGraph::with_feldera_backend()`.
- `Cargo.toml` — add `egglog-bridge-feldera` as a dependency of the main crate.
- `tests/files.rs` — `feldera: bool` treatment (mirrors `duckdb`), gated behind
  `EGGLOG_TEST_FELDERA=1`.
- `egglog-bridge-feldera/src/lib.rs` — embed `Database`; back base-values /
  external-funcs with it; route `run_rules` through the interpreter;
  `resolve_merge` FD resolution; variable-width rows.
- `egglog-bridge-feldera/src/interpret.rs` — the host-side rule interpreter (new).
- `egglog-bridge-feldera/src/compile.rs` — ordered `RuleIr` (`BodyOp` / `HeadOp`);
  removed the DBSP circuit assembly (execution is host-side in M3).
- `egglog-bridge-feldera/src/rule_builder.rs` — emit the ordered IR for all
  `RuleBuilderOps` (table atoms, prims, lookups, calls, set/delete/subsume/union/panic).
- `egglog-bridge-feldera/src/base_values.rs` — `BaseValues`→`BaseValuePool` cast
  over the Database's registry.
- `egglog-bridge-feldera/src/external_func.rs` — id-aligned name/panic side-table
  + `PanicFunc`.
