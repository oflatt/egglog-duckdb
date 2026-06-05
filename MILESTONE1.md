# Milestone 1 — Feldera/DBSP backend: per-iteration stepping behind the trait

**Verdict: ACHIEVED.** Bounded `(run N)` works end-to-end behind the
`egglog_backend_trait::Backend` interface on real `dbsp` (0.150.0, in-process),
and matches the reference backend (`egglog_bridge::EGraph`) round-for-round on
the same program. The load-bearing correction in the brief — that the backend
must do **one egglog iteration per `run_rules` call**, NOT saturate in one
transaction — is realized with a **non-recursive** DBSP circuit driven N times.

Crate: `egglog-bridge-feldera/` (added to the workspace `members`). File layout
mirrors `egglog-bridge-duckdb/`: `lib.rs`, `compile.rs`, `rule_builder.rs`,
`base_values.rs`, `external_func.rs`.

## What works

1. **Relations + a single-join rule.** `add_table` registers a relation; the
   rule builder accumulates a `RuleIr` (1–2 table body atoms + `set`/`insert`
   heads); `build()` registers it and invalidates the cached circuit.
2. **Bounded `(run N)`.** `run_rules` does exactly one round of rule application
   (one `transaction()` over a non-recursive circuit), then folds the
   integrated output back into a Rust-side mirror and feeds the new rows back as
   the next round's input delta. N calls = N hops, not closure.
3. **Read-back through the trait.** `for_each` / `for_each_while` / `lookup_id`
   / `table_size` read the materialized mirror (refreshed from the circuit's
   `integrate().output()` after each transaction — the spike's mirror pattern).

## The `(run 1)` vs `(run 3)` proof (actual test output)

Program (single-join derivation over the chain `1->2->3->4`):

```
edge(x,y), path(x,y)   seeded with {(1,2),(2,3),(3,4)}
path(x,z) :- path(x,y), edge(y,z).
```

Test `run1_vs_run3_bounded_and_matches_reference` (in
`egglog-bridge-feldera/tests/run_n_proof.rs`) drives BOTH backends through the
`Backend` trait and prints:

```
run(1) reference = {(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
run(1) feldera   = {(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
run(3) reference = {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
run(3) feldera   = {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
```

- `(run 1)` adds **one hop** (`(1,3)`, `(2,4)`) but does **NOT** contain the
  3-hop pair `(1,4)` — bounded, not saturated.
- `(run 3)` reaches the full closure including `(1,4)`.
- Feldera **equals the reference backend** at both N — this is the faithfulness
  proof. `run(1) != run(3)` is the bounded-iteration proof.

Reproduce:
```
cargo test -p egglog-bridge-feldera --test run_n_proof -- --nocapture
```

## How the per-iteration model maps onto DBSP

- The circuit is **non-recursive** (NO `recursive` scope). For each relation
  `r`: `r_out = r_in ∪ ⋃_{rules→r} (body-join → head)`, then
  `r_out.integrate().output()`. One `transaction()` = one hop because the body
  is not a fixpoint.
- The host (`run_rules`) realizes egglog's bounded loop by **feeding the
  previous round's derived rows back** as input deltas (a Rust-side feedback
  loop), instead of DBSP's internal recursive feedback. This is the deliberate
  divergence from the spike, which (wrongly, for egglog) used the recursive
  scope.
- `run_rules` reports `changed` honestly (mirror grew?), so the frontend's outer
  loop terminates.
- Rows use a single uniform type `Row = Tup8<u32,…>` (egglog `Value`s are
  `u32`, 0-padded past a relation's arity), so a *dynamically assembled* circuit
  can wire `OrdZSet<Row>` streams without per-relation generics. Arity is capped
  at 8 for milestone 1.
- Circuits are static once built, so the circuit is rebuilt lazily whenever a
  relation or rule is added/removed (PLAN §4.2 #2); relation contents replay
  from the mirror as input deltas.

## Trait methods: implemented vs stubbed

Implemented (real bodies): `add_table`, `table_size`, `approx_table_size`,
`for_each`, `for_each_while`, `lookup_id`, `add_values`, `add_term`,
`insert_rows`, `lookup_constructor_rows`, `get_canon_repr` (identity — no UF
yet), `fresh_id`, `clear_table`, `base_values`, `base_value_pool(_mut)`,
`base_value_constant_dyn`, `new_rule`, `free_rule`, `run_rules`,
`flush_updates`, `register_external_func`/`free_external_func`/`new_panic`
(storage only), capability flags, `set_report_level`, `dump_debug_info`,
`as_any`/`as_any_mut`.

`RuleBuilderOps` implemented: `new_var`, `new_var_named`, `query_table`
(1–2 atoms), `set`/`insert`, `build`.

Stubbed / errors (mirroring DuckDB's gating; deferred to later PLAN phases):
`with_execution_state_dyn`, `action_registry_any`, `clone_boxed` (push/pop —
PLAN Phase 5 snapshot-and-replay), container pool (all empty/error),
`supports_inline_table_lookups`/`supports_subsumption`/`supports_complex_merge`/
`supports_containers` = `false`. `RuleBuilderOps::{query_prim,
call_external_func, lookup, subsume, remove, union, panic}` error at `build()`.

## Trait-friction notes (feed the minimal interface refactor)

1. **`run_rules(&[RuleId])` runs a *subset* of rules, but a DBSP circuit is a
   monolithic graph of *all* rules.** Honoring an arbitrary subset per call
   would require per-subset circuits (or gating operators by a runtime mask).
   Milestone 1 runs the whole built circuit; this is exact for single-ruleset
   programs but not for `(run R1)` then `(run R2)` over disjoint rulesets. A
   trait hint like "rulesets are stable / declared up front" — or grouping rules
   into named circuits — would remove the impedance.

2. **`run_rules` = one pass vs DBSP's natural "saturate in one step".** Resolved
   here by *not* using the recursive scope (one transaction = one hop). The
   optional `runs_ruleset_to_fixpoint() -> bool` hint from PLAN §4.3 is **not**
   wanted by this backend (we deliberately do one hop); but a backend that *did*
   saturate would want it. The trait should let a backend *declare* which model
   it implements so the frontend's loop can adapt.

3. **No retraction/`remove` surface needed yet, but the model is monotone-only.**
   `run_rules` currently pushes only positive deltas (no `-1` weights). When
   rebuild/union lands (PLAN Phase 3), the mirror→circuit diff must emit
   retractions; the trait already supports this shape via `remove`, but there is
   no batched "apply these retractions then step" entry point distinct from
   rule-driven removal.

4. **Static-circuit rebuild is invisible to the trait.** Adding a rule/relation
   silently invalidates and lazily rebuilds the circuit (replaying mirror state
   as deltas). This works but is O(state) per rebuild; the proof encoder emits
   many rules, so a "freeze schema / no more rules" signal (or batched rule
   registration) would let the backend build the circuit once.

5. **`FunctionConfig` carries no explicit "is this a relation vs a function
   with an output column" bit.** We infer `has_output` from `DefaultVal`
   (`FreshId`/`Const` ⇒ output; `Fail`+`AssertEq` ⇒ relation). This is a
   heuristic shared with DuckDB's `add_table`; an explicit flag (or a separate
   `add_relation`) would be cleaner and avoids each backend re-deriving it.

6. **`run_rules` returns an `IterationReport` whose only field this backend
   sets is `changed`.** The richer per-rule timing shape doesn't fit the
   one-transaction model; fine for now, but the report is mostly empty.

## Not attempted (correctly out of scope for milestone 1)

Union-find / rebuild as non-monotone recursive views (the real research crux,
PLAN Phase 3), primitives (PLAN Phase 2), merges beyond plain insert, proofs,
containers, push/pop. None are contradicted by this milestone; they are the
next gates.
