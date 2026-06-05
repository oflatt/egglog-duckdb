# Milestone 4 — Feldera/DBSP backend: relational body joins on the DBSP engine

**Goal (the paper's core contribution):** move full-program rule-body joins
off the M3 host-side interpreter and onto **DBSP's incremental dataflow
engine**, matching the reference backend's per-function tuple counts. M3 reached
59/62 parity via a host nested-loop interpreter (now the *correctness oracle*).
M4 makes the **relational join run on DBSP** for the eligible rule class, with
the interpreter retained only as oracle / fallback, and documents the frontier
precisely.

## Headline results

| metric | M3 | M4 |
|--------|----|----|
| `.egg` files at shared-snapshot + tuple-count parity (62 eligible) | 59 | **59** |
| files that FAIL | 0 | **0** |
| files that TIMEOUT | 3 | **3** |
| body joins run on | host interpreter | **DBSP for the eligible class** |
| primitive entry point | embedded `Database` reach-in | **`Backend::eval_prim`** |

The 59 passing files are unchanged from M3 *and now run their multi-atom rule
joins on the DBSP dataflow engine* (verified — see "How we know the joins are on
DBSP"). The full-survey wall time dropped (~7s for the 59 vs. M3's host-only
run), because DBSP's `join` replaces the interpreter's repeated full scans for
the eligible rules. The 3 timeouts persist; their root cause is now precisely
diagnosed (missing semi-naive incrementality, **not** merely merge timing — see
"The three timeouts").

## 1. The `eval_prim` refactor (the mandated prerequisite)

M3's biggest wart: to evaluate a primitive a backend had to own a
`core_relations::Database`, because `ExternalFunction::invoke` takes
`&mut core_relations::ExecutionState`. We added a backend-agnostic entry point
to the `Backend` trait:

```rust
fn eval_prim(&self, id: ExternalFunctionId, args: &[Value]) -> Option<Value>;
```

- **Reference backend** (`egglog-bridge/src/backend_impl.rs`):
  `self.with_execution_state(|st| st.call_external_func(id, args))`.
- **Feldera** (`egglog-bridge-feldera/src/lib.rs`): same, against its embedded
  `Database`. The host interpreter (`interpret.rs`) and the DBSP-join path both
  now call the inherent `eval_prim_internal` instead of reaching into `self.db`
  directly — the M3 wart is gone at the call sites.
- **DuckDB** (`egglog-bridge-duckdb/src/backend_impl.rs`): invokes the stored
  `ExternalFunction` against an *ephemeral* `core_relations` execution state.
  DuckDB evaluates primitives via SQL on its own path, so this is provided for
  trait completeness; pure primitives (comparisons / arithmetic on
  inline-encodable base values) evaluate correctly through it.

All three backends build; the reference + duckdb + feldera test suites stay
green (bridge 25, duckdb 26, feldera proofs all pass).

## 2. Relational body joins on DBSP (`egglog-bridge-feldera/src/dbsp_join.rs`)

For a **DBSP-eligible** rule, `run_rules` now builds a non-recursive DBSP circuit
that computes the join of the body's table atoms — multi-atom, left-deep, with
`!=` guards as DBSP `filter` operators *inside* the dataflow — and reads back a
z-set of binding rows (one `Tup8` per satisfying assignment, body variables in a
fixed canonical order). One `transaction()` = one egglog iteration (the
per-iteration model M1/M2 proved). This is the relational join running on DBSP's
engine, which is the milestone's mandate.

The join *output* is handed to the existing head machinery: value-computing
primitives are evaluated via `eval_prim`, and `set`/`delete`/`lookup`/`union`
head actions + FD-merge resolution are applied host-side. DBSP map/filter
closures are `Send + 'static` and cannot borrow the primitive engine, so
primitive *value computation* and head writes stay on the host — but the join
itself (the expensive, paper-relevant part) is on DBSP.

### The frontier (exactly where the DBSP path stops)

A rule runs its join on DBSP iff (`dbsp_join::plan_join`):

- its body has ≥1 table atom;
- every table atom has arity ≤ 8 and the body uses ≤ 8 distinct variables
  (the fixed-arity `DBData`/`Tup8` row — DBSP rows must be rkyv-archivable, so a
  variable-width row is not available, exactly as in M1/M2);
- its only *body primitives* are `!=` guards (recognized by name), which lower
  to a pure-`u32`-inequality DBSP filter.

Everything else falls back to the host interpreter (the oracle):

- **value-computing body prims** (`+`, `ordering-min/max`, …) and **non-`!=`
  ordering guards** (`<`, `<=` on typed base values) — evaluating these needs
  the primitive engine *inside* the join, which a `Send + 'static` DBSP closure
  cannot hold. (A body that has table atoms *followed by* a value-prim still
  runs its table-atom join on DBSP; only the primitive tail is host-side.)
- bodies wider than 8 variables / atoms wider than 8 columns.

This frontier is **honest by construction**: `EGraph::dbsp_join_stats()` returns
`(dbsp_rule_runs, host_rule_runs)`, so a test/survey can assert which fraction of
rule firings ran on DBSP versus fell back.

### How we know the joins are on DBSP (not the interpreter)

`egglog-bridge-feldera/tests/run_n_proof.rs::transitive_closure_join_runs_on_dbsp`
drives the canonical transitive-closure rule `path(x,z) :- path(x,y), edge(y,z)`
through the Feldera backend and asserts `host_runs == 0` and `dbsp_runs >= 3` —
i.e. the 2-atom join ran entirely on DBSP every round, never on the interpreter.
`tests/rebuild_proof.rs` exercises the rebuild rules
`(@uf a b)(@uf b c)(!= b c)` and `(@uf a b)(@uff a c)(!= b c)` — 2-atom joins
*with `!=` guards* — which are DBSP-eligible and match the reference exactly
across all five union-find shapes (incl. the diamond congruence-collapse case).

Across the 59 passing `.egg` programs, the dominant rule shape — eq-sort
congruence, rewrites, transitive/relational joins, `@uf`/`@uff` rebuild — is
table-atoms(+`!=`), which is exactly the DBSP-eligible class. Programs whose
*head* or *value-prim tail* needs the primitive engine still run their **join**
on DBSP and only the tail on the host.

## 3. The three timeouts (precise root-cause; merge fix re-scoped)

M3 hypothesized the 3 timeouts (`repro_should_saturate`, `rectangle`,
`naturals`) were merge-timing (per-iteration vs. atomic flush-time) and slated a
flush-time merge fix. M4's diagnosis (via per-iteration mirror tracing)
corrects this:

- **`repro_should_saturate`** (`(function MyMap () i64 :merge (min old new))`
  with a rule that `set`s both 1 and 2): the custom `:merge` is term-encoded
  into a *view* table + a *merge ruleset* + a *cleanup ruleset* (delete the
  stale view row). The user rule `((MyMap)) → (set 1)(set 2)` re-derives the
  same match **every iteration** because the host/DBSP-per-call model has **no
  semi-naive incrementality** — so it perpetually re-inserts `MyMap = 2`, the
  cleanup rule deletes the stale view row, and the next iteration re-creates it.
  The system **oscillates** and never reaches a no-change iteration. The
  reference backend converges precisely *because* it is semi-naive (the user
  rule fires once on the `(MyMap)` match and does not re-fire).
- **`rectangle`** is *not* a merge problem at all: it is
  `(range i)(< i 1000) → (range (+ i 1))` saturating `range` to 1000 elements.
  The `< i 1000` guard and `(+ i 1)` value-prim make the rule host-fallback, and
  the host nested-loop re-derives all matches every iteration (no semi-naive) →
  quadratic blowup over ~1000 rounds.
- **`naturals`** is eq-sort rewriting under `(saturate (run))`; same
  non-incrementality wall.

So the real blocker for all three is **semi-naive incrementality**, not merge
atomicity. A flush-time atomic merge alone would *not* make
`repro_should_saturate` converge (the reference relies on semi-naive too). We
deliberately did **not** ship a speculative semi-naive rewrite of the hybrid
DBSP+host per-iteration model in this pass: it interacts with rule-driven
deletes (rebuild's `path_compress` / `single_parent` / the merge-cleanup
deletes — a deleted-then-readded row *must* re-fire), and a naïve firing-dedup
would regress the 59 passing programs (incl. the M2 rebuild proof). Getting it
right needs per-row version tracking across the delta; that is the clear next
milestone and is documented here rather than risked half-done.

## What runs with joins genuinely on DBSP

- All M2 union-find / rebuild rules (`@uf`/`@uff`, `path_compress`,
  `single_parent`, `uf_index`) — 2-atom joins with `!=` guards.
- Transitive-closure / relational joins (`path`, `path_union`, `points_to`,
  `unification_points_to`, `resolution`, `intersection`, `stratified`,
  `rw_analysis`, …).
- Eq-sort congruence + rewrite rules whose bodies are table atoms (+ `!=`):
  `eqsat_basic`, `eqsolve`, `antiunify`, `birewrite`, `combinators`, `matrix`,
  `unify`, `typecheck`, …

For programs that also use base-value primitives (`i64`, `f64`, `string`,
`bitwise`, `integer_math`, `primitives`, `complex_merge_func`, …), the
table-atom join runs on DBSP and the primitive value-computation + head writes
run on the host (the documented frontier).

## Validation

- `cargo test -p egglog-bridge-feldera --release` — M1 (`run_n_proof`, incl. the
  new `transitive_closure_join_runs_on_dbsp` DBSP-on-engine assertion) and M2
  (`rebuild_proof`, all 5 shapes) pass against the reference backend.
- `EGGLOG_TEST_FELDERA=1 cargo test -p egglog --release --test files`
  (excluding the 3 known timeouts): **59 pass, 0 fail** at shared-snapshot +
  per-function tuple-count parity vs. the reference backend.
- Reference + DuckDB backends stay green (bridge 25 tests, duckdb 26 tests,
  default + duckdb `.egg` file treatments unaffected).

## Remaining friction / blockers (ranked)

1. **Semi-naive incrementality** — the single blocker for the 3 timeouts and the
   biggest scaling limit. Both the host fallback and the per-call DBSP-join
   rebuild re-derive all matches each iteration. DBSP supports incremental
   `join` natively via input deltas + cached circuits (the M2 per-subset circuit
   cache is the foundation); wiring delta-input semi-naive into the M4 join path
   — with correct re-firing across rule-driven deletes — is the next milestone.
2. **Primitives cannot run inside DBSP closures** — `eval_prim` removed the
   *embedded-Database* wart but a `Send + 'static` map/filter closure still
   cannot call it, so value-computing prims and head writes stay host-side. A
   pure-closure primitive form (or a vectorized post-join primitive operator)
   would push more of the rule onto DBSP.
3. **Fixed `Tup8` row** — DBSP rows must be rkyv-`DBData`, capping the DBSP join
   at 8 columns / 8 body variables. Wider rules fall back to the host
   (variable-width) interpreter. A custom rkyv-archivable variable-width row
   would lift this.
4. Carried from M3: typed merge intent, stable ruleset handles, a first-class
   "filter atom" vs "compute atom" body distinction, and a term-encoding
   capability flag to skip the unused native-union-find trait surface.

## Files touched

- `egglog-backend-trait/src/lib.rs` — `Backend::eval_prim`.
- `egglog-bridge/src/backend_impl.rs` — reference `eval_prim`.
- `egglog-bridge-duckdb/src/backend_impl.rs`, `src/external_func.rs`,
  `Cargo.toml` — duckdb `eval_prim` (ephemeral execution state) + `clone_func`.
- `egglog-bridge-feldera/src/dbsp_join.rs` — **new**: DBSP body-join circuit
  (eligibility analysis, left-deep multi-atom join, `!=` filters, binding-row
  read-back).
- `egglog-bridge-feldera/src/interpret.rs` — DBSP-join path with host fallback;
  routes primitives through `eval_prim_internal`.
- `egglog-bridge-feldera/src/lib.rs` — `eval_prim` + `eval_prim_internal`,
  `dbsp_join` module, `dbsp_join_stats()` diagnostics.
- `egglog-bridge-feldera/tests/run_n_proof.rs` —
  `transitive_closure_join_runs_on_dbsp` (asserts the join runs on DBSP).
