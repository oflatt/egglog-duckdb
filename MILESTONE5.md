# Milestone 5 — Seminaive incrementality on DBSP: egglog seminaive = DBSP IVM

**Goal (the paper's headline equivalence):** realize egglog's seminaive
evaluation as DBSP incremental view maintenance. Each `run_rules` iteration must
fire rules only against the **delta** (newly-derived/changed facts) joined with
the full relations — so a rule that already fired on the existing facts does NOT
re-fire. This is exactly the incremental dataflow DBSP exists for, and it is what
makes bounded `(saturate …)` loops *converge* instead of oscillating forever
against rule-driven cleanup/retraction.

## Headline results

| metric | M3 | M4 | **M5** |
|--------|----|----|--------|
| `.egg` files at shared-snapshot + tuple-count parity (62 eligible) | 59 | 59 | **61** |
| files that FAIL | 0 | 0 | **0** |
| files that TIMEOUT | 3 | 3 | **1** |
| body joins run on | host interp | DBSP (eligible) | **DBSP (eligible), seminaive** |
| rule firing model | full×full re-derive | full×full re-derive | **delta-driven (seminaive)** |

**The two oscillation timeouts converged.** `repro_should_saturate` and
`naturals` — both blocked in M3/M4 purely by *missing seminaive incrementality*
— now reach a fixpoint and **pass at shared-snapshot + per-function tuple-count
parity** vs. the reference backend. The remaining timeout, `rectangle`, is the
value-prim host-fallback frontier M4 flagged (NOT incrementality); it is
characterized below.

The 59 previously-passing files stayed green (now 61 total). Validated against
BOTH the reference backend (the shared snapshot every treatment shares) and the
M3 host-interpreter oracle (the seminaive driver reuses the same head machinery /
merge resolution), plus the M1 `run_n_proof` and M2 `rebuild_proof` (incl. the
retraction-heavy diamond / star+chain congruence-collapse shapes).

## The seminaive approach as built

### 1. Per-rule delta tracking (`EGraph::seen`)

The load-bearing state is a **per-rule** map
`seen: HashMap<rule_idx, HashMap<FunctionId, HashSet<Row>>>` — for each rule, the
contents of each body relation that rule has **already matched against**. The
seminaive delta of rule `r` over relation `f` is:

```text
delta[r][f] = mirror[f] \ seen[r][f]
```

Keying by **rule** (not globally) is essential and was the first design trap:
the frontend schedules distinct rulesets in sequence (the term encoder's
`(saturate single_parent)` / `(saturate path_compress)` / `(saturate uf_index)`,
and user rulesets). Rows produced by an earlier ruleset must count as *new* to a
later ruleset's rules, which have never matched them. A single global "seen"
snapshot would starve a freshly-scheduled rule of its entire delta and silently
drop derivations.

### 2. The seminaive join (the union over delta positions)

For a body with table atoms `A_1 … A_k` (plus prim guards), each iteration
computes the standard seminaive union:

```text
Δbindings(r) = ⋃_j  A_1(full) ⋈ … ⋈ A_j(delta) ⋈ … ⋈ A_k(full)
```

i.e. for each table-atom position `j`, atom `j` ranges over only its delta rows
and the others over the full relation; the union over `j` is exactly the set of
bindings touching ≥1 newly-derived fact. Variants are deduplicated (a binding
with new facts in two positions appears in two terms). If **no** body relation
has any delta, the rule is skipped entirely — the seminaive win that stops the
oscillation. Implemented in `interpret.rs::seminaive_bindings`.

### 3. Both engines are seminaive (DBSP **and** the host fallback)

The seminaive partition happens at join input, so it had to reach **both** rule
paths — `repro_should_saturate`'s user rule `((MyMap)) → (set 1)(set 2)` is
itself DBSP-eligible (one table atom, no value prims), so a host-only fix would
not have converged it:

- **DBSP-eligible rules** (`dbsp_join.rs`): the join circuit was refactored to
  use **one input stream per atom *occurrence*** (not per relation), so the same
  relation appearing in several atoms can be fed different row sets. `run_join`
  (full×…×full, retained for the M4 proof) and the new `run_join_seminaive`
  (occurrence `j` fed its delta, the rest full) both route through one
  `run_join_with` core. The relational join — the paper-relevant part — stays on
  DBSP's dataflow engine, now incrementally.
- **Host fallback** (the oracle, for value-prim / wide bodies): the nested-loop
  body scan ranges the delta atom over delta rows and the others over the full
  mirror, via the shared `step_atom` helper.

### 4. Retraction-correct `seen` advance (the rebuild trap)

After a rule generates its bindings, `seen[r][f]` is advanced to the
**start-of-iteration** snapshot (`read[f]`), *never* the post-write mirror. This
is what the milestone flagged as delicate, and it is exactly what makes
rule-driven deletes (rebuild's `path_compress` / `single_parent`, the
merge-cleanup deletes) correct: a row that is **deleted and later re-added**
reappears in `r`'s delta (it is in `mirror` but the start-of-iteration snapshot
that re-added it did not contain it), so the consuming rule re-fires. The M2
`rebuild_proof` (all five union-find shapes, incl. the diamond congruence
collapse that hinges on delete-then-readd) passes unchanged. `clear_table` and
`free_rule` evict the relevant `seen` entries so a re-populated table presents
fresh deltas.

## Why the oscillation cases now converge

`repro_should_saturate`:
`(function MyMap () i64 :merge (min old new))`, seeded `(set (MyMap) 1)`, rule
`((MyMap)) → (set (MyMap) 1)(set (MyMap) 2)` under `(saturate (run))`.

- **Before (M4):** every iteration re-matched the persistent `(MyMap)` row,
  re-`set` 1 and 2, the term-encoder cleanup deleted the stale view row, and the
  next iteration re-created it — `changed` stayed true forever (timeout).
- **After (M5):** `(MyMap)` is new only in the first iteration. The rule fires
  once (min-merges to 1), advances its `seen`, and thereafter sees an empty
  delta → does not re-fire. The cleanup has nothing to fight; the mirror reaches
  a no-change iteration and `(saturate)` terminates — matching the reference,
  which converges precisely *because* it is seminaive.

`naturals` (eq-sort rewriting under `(saturate (run))`): same wall removed — the
rewrite rules fire only on newly-derived terms instead of re-deriving the full
set every round.

## The remaining timeout: `rectangle` (expected, not incrementality)

`rectangle` is the M4-predicted value-prim frontier, confirmed unchanged by M5:
the `populate` ruleset is `(range i)(< i 1000) → (range (+ i 1))`. The `< i 1000`
ordering guard and the `(+ i 1)` value-computing primitive make the rule
**host-fallback** (neither is a DBSP `!=`-filter; evaluating them needs the
primitive engine, which a `Send + 'static` DBSP closure cannot hold — the M4
frontier). Seminaive makes the *delta* small (one new `range` row/iteration), but
the program still requires ~1000 host iterations to grow `range` to 1000, and the
downstream `result` join `R(x,y) S(y,z) T(z,w) U(w,x)` is a 4-way cyclic join
over ~1000-row relations rebuilt as a fresh per-call DBSP circuit each iteration.
This is the value-prim / per-call-circuit cost the milestone explicitly scoped
*out* of the seminaive fix, not an incrementality bug. Removing it needs the DBSP
frontier extended to value-prims (a vectorized post-join primitive operator or a
pure-closure primitive form) and/or a cached incremental circuit — the M4
remaining-friction items, not this milestone's mandate.

## Validation (vs. reference AND oracle)

- `cargo test -p egglog-bridge-feldera --release` — **green**: M1
  `run_n_proof` (incl. `transitive_closure_join_runs_on_dbsp`, which still
  asserts `host_runs == 0` / `dbsp_runs ≥ 3`: the seminaive join still runs
  entirely on DBSP) and M2 `rebuild_proof` (all five union-find shapes) match the
  reference backend exactly.
- `EGGLOG_TEST_FELDERA=1` per-file survey (`.tmp/feldera_survey.sh`, 30 s
  budget): **61 PASS, 0 FAIL, 1 TIMEOUT** (`rectangle`) at shared-snapshot +
  per-function tuple-count parity vs. the reference. `repro_should_saturate` and
  `naturals` moved TIMEOUT → PASS; every other file unchanged.
- Reference + DuckDB backends untouched (changes are confined to the
  `egglog-bridge-feldera` crate) and stay green; the shared-snapshot files
  treatment for the default/proof/duckdb paths is unaffected.

## Remaining friction (ranked)

1. **Value-prims still force host fallback** (the `rectangle` wall). A DBSP
   pure-closure / vectorized post-join primitive operator would push `<` / `+`
   bodies onto the engine and let the per-call circuit be cached incrementally.
2. **Per-call circuit rebuild.** Each delta-atom variant builds a fresh
   non-recursive DBSP circuit per iteration (`k` builds/rule/round). The natural
   next step is a cached, integrated DBSP circuit fed true input deltas across
   iterations — turning the per-call build into genuine streaming IVM. The
   per-rule `seen` deltas computed here are exactly the input deltas such a
   circuit would consume.
3. **Fixed `Tup8` row** (carried from M4): DBSP rows must be rkyv-`DBData`,
   capping the DBSP join at 8 columns / 8 body variables; wider rules use the
   host (variable-width) seminaive path.
4. Carried: typed merge intent, stable ruleset handles, filter-vs-compute atom
   distinction, term-encoding capability flag.

## Files touched

- `egglog-bridge-feldera/src/lib.rs` — per-rule `seen` seminaive state;
  `free_rule` / `clear_table` evict it.
- `egglog-bridge-feldera/src/interpret.rs` — `seminaive_bindings` (the union over
  delta positions, dedup, skip-if-no-delta), `step_atom` helper, per-rule delta
  computation + start-of-iteration `seen` advance in `run_iteration`.
- `egglog-bridge-feldera/src/dbsp_join.rs` — per-atom-occurrence input streams;
  `run_join_with` core; `run_join_seminaive` (one delta-atom term).
