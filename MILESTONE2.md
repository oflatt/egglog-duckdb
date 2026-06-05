# Milestone 2 — Feldera/DBSP backend: union-find + rebuild (the research crux)

**Verdict: ACHIEVED.** A real, term-encoded **union-find + rebuild** program —
the `@uf` parent table plus the term encoder's path-compress / single-parent /
uf-index rulesets, with `(delete …)` / `(set …)` actions run in egglog's
scheduled order — runs end-to-end on the Feldera/DBSP backend behind the
`egglog_backend_trait::Backend` interface, and **matches the reference backend
(`egglog_bridge::EGraph`) on per-function tuple counts AND exact contents**
across five union-find shapes (including multi-parent congruence collapse).

This is the milestone's success bar: a monotone-incremental engine (DBSP) hosts
egglog's delete-and-rewrite rebuild, faithfully, by RUNNING the rules the term
encoder already emits — the backend does not invent rebuild.

## What runs end-to-end

`tests/rebuild_proof.rs` hand-encodes exactly the term-mode rulesets from
`src/proofs/proof_encoding.rs::declare_sort` (the encoder's `:merge old` /
`:merge (ordering-min old new)` recognition included) and drives them through
the `Backend` trait on **both** backends:

```
@uf  : function (child parent) -> Unit   :merge old              -- parent edges
@uff : function (child) -> leader        :merge (ordering-min …) -- the UF index

uf_index     : (@uf a b)                   -> (set (@uff a) b)
path_compress: (@uf a b)(@uf b c)(!= b c)  -> (delete (@uf a b))(set (@uf a c))
single_parent: (@uf a b)(@uff a c)(!= b c) -> (delete (@uf a b))
                                              (set (@uf b c))(set (@uff b) c)
```

Schedule, from `proof_encoding.rs::rebuild`:
`loop { saturate single_parent; saturate path_compress; saturate uf_index }`
until a whole pass makes no change.

## The tuple-count comparison (actual numbers)

Each scenario seeds union edges, runs the full rebuild schedule on both
backends, and reads `@uf` / `@uff` back via `for_each`. Reproduce:

```
cargo test -p egglog-bridge-feldera --test rebuild_proof -- --nocapture
```

| scenario      | seed edges (child,parent)      | `@uf` ref/fel | `@uff` ref/fel | final contents (both) |
|---------------|--------------------------------|---------------|----------------|-----------------------|
| single-union  | (2,1)                          | 1 / 1         | 1 / 1          | {(2,1)}               |
| 2-chain       | (2,1)(3,2)                     | 2 / 2         | 2 / 2          | {(2,1),(3,1)}         |
| 3-chain       | (2,1)(3,2)(4,3)                | 3 / 3         | 3 / 3          | {(2,1),(3,1),(4,1)}   |
| diamond       | (3,1)(3,2)(2,1)  *(3 has 2 parents)* | 2 / 2    | 2 / 2          | {(2,1),(3,1)}         |
| star+chain    | (5,1)(4,1)(3,2)(2,1)           | 4 / 4         | 4 / 4          | {(2,1),(3,1),(4,1),(5,1)} |

Feldera equals the reference **exactly** (counts and contents) in every case,
and the semantic invariant holds: after rebuild every non-leader child resolves
to leader `1`. The **diamond** scenario is the load-bearing one — node 3 starts
with two parents {1,2}, so `single_parent` must fire (the
`(@uf a b)(@uff a c)(!= b c)` redirect), exercising the congruence-collapse half
of rebuild, not merely path compression.

Milestone 1's bounded-iteration proof (`tests/run_n_proof.rs`) stays green.

## How the new machinery was implemented

### Ruleset-scoped execution (friction #1, resolved)

`run_rules(&[RuleId])` runs an arbitrary **subset** of rules; a DBSP circuit is a
static monolithic graph. We build and **cache one circuit per distinct sorted
rule subset** (`EGraph::circuits: HashMap<Vec<u32>, CircuitState>`). The Rust
mirror is the single source of truth shared across all circuits. Each call:

1. looks up / builds the circuit for that exact subset,
2. syncs that circuit's input handles to the current mirror by pushing the
   **delta** (additions `+1`, removals `−1`) vs. what was last pushed into *this*
   circuit,
3. runs one `transaction()` (one hop — the circuit is non-recursive),
4. folds the per-relation diff back into the mirror.

This lets the frontend schedule `(saturate single_parent)` then
`(saturate path_compress)` then `(saturate uf_index)` as distinct
`run_rules` calls hitting distinct cached circuits, exactly the term encoder's
rebuild schedule. (M1 ran one monolithic circuit; this is the main new
machinery.)

### Retraction = rebuild's delete half (spike-proven negative weights)

`(delete …)` / the trait `remove` becomes a `delete` head action that feeds a
per-relation **delete diff stream** (separate from the **insert diff stream**).
Both streams are `integrate().output()`'d. At fold time the host computes
`new = (old ∪ inserts) \ deletes`, where a delete addresses a row by its **key**
(input columns for a function, whole row for a relation) — matching egglog's
key-addressed `(delete (@uf a b))`. The mirror→circuit sync also pushes `−1`
weights when the mirror shrinks, so the circuit's integrated input always equals
the current mirror (the spike's confirmed retraction path).

### `@uf` / `@uff` merge recognition (DuckDB-style stopgap)

`add_table` recognizes the merge mode into a `MergeMode` enum (`compile.rs`),
mirroring the DuckDB backend's stopgap (`backend_impl.rs` ~577):

- `AssertEq` / `Old` ⇒ keep old; `New` ⇒ keep new;
- `UnionId` and `Primitive(…)` ⇒ **lattice-min** — this is how the term encoder's
  `@uff` `:merge (ordering-min old new)` arrives;
- a function with no output column ⇒ `Relation` (whole row is the key).

FD-conflict resolution runs **host-side at fold time**: for each key, pick a
single surviving output column per the mode (`min` for the UF index). Keeping
merge resolution in the host (rather than wiring a DBSP keyed aggregate) mirrors
egglog's own separation of rule firing from flush-time merge, and keeps the
circuit a pure relational diff engine.

### Body filters and lookups

`query_prim` recognizes the inequality guard `!=` (by name, recorded via
`rename_prim`) and lowers it to a circuit `Filter::Ne` evaluated inside the join.
The `(= c (@uff a))` lookup in `single_parent` is just a function-table body
atom `query_table(@uff, [a, c])` (output `c` bound by the join). 2-atom joins
with cross-atom filters are supported (the filter is evaluated on the joined
`(ra, rb)` pair, carried through the join as a `Tup2<u8, Row>` marker).

## What is stubbed / deferred

- **Full term encoder integration via the frontend.** This milestone drives the
  encoded rulesets directly through the `Backend` trait (M1's proven posture),
  not by routing a `.egg` source file through `EGraph::with_backend`. Running
  arbitrary `.egg` programs through the frontend on Feldera needs the rest of
  the encoder surface (view-table congruence rules, base values, arity > 8,
  arbitrary primitives, schedules) and is the next integration step. The rebuild
  *rules themselves* — the research crux — are exercised faithfully here.
- **Merge typing.** Min-merge is recognized via the stopgap (any `Primitive`
  ⇒ min); a typed-merge cleanup is a later refactor (same as DuckDB).
- **Containers, subsumption, proofs, push/pop (`clone_boxed`), n-ary joins
  (>2 atoms), arity > 8.** Unchanged from M1 — gated/stubbed.
- **DBSP-native rebuild (PLAN path A).** Rebuild here is run as the encoder's
  rules over a non-recursive per-hop circuit with host-side feedback (the M1
  model extended with retraction), NOT as a single non-monotone *recursive*
  DBSP view. The recursive-view encoding (`find`-via-aggregate inside one
  recursive scope) remains the open research question; this milestone proves the
  rules converge correctly under DBSP retraction, which is the prerequisite.

## NEW trait-friction (for the minimal-interface refactor)

1. **No name on `MergeFn::Primitive`.** Recognizing `:merge (ordering-min …)`
   requires matching the primitive by name, but the trait carries only an
   `ExternalFunctionId`. DuckDB reaches into its own name registry; Feldera now
   tracks names via `rename_prim` too. A typed merge enum (or a name on the
   `MergeFn::Primitive` variant) would remove this per-backend re-recognition.

2. **`remove` addresses by key, but relations vs functions differ.** The trait's
   `remove(func, entries)` passes the key columns. Whether that key is "all
   columns" (relation) or "inputs only" (function) is inferred from
   `has_output`, the same `DefaultVal`-based heuristic M1 flagged (friction #5).
   An explicit relation-vs-function bit would make delete semantics unambiguous.

3. **Ruleset identity is rediscovered every call.** `run_rules(&[RuleId])` hands
   a raw id list; the backend has to sort/dedup it into a cache key each time.
   A stable "ruleset handle" (or a "rulesets are declared up front" hint) would
   let the backend build each ruleset's circuit once at registration instead of
   lazily per distinct subset. (Refines M1 friction #1 — now concretely needed
   because the rebuild schedule cycles through three fixed rulesets.)

4. **`!=` (and other guards) arrive as `query_prim` with a threaded return
   var.** The bridge identifies primitives by id and the predicate's "return
   value" is a side-channel; a backend that wants to lower `!=` to a filter must
   recognize the primitive and ignore the synthetic return entry. A first-class
   "filter atom" in the rule IR would be cleaner than the prim-with-unit-return
   idiom.

5. **No batched "apply retractions then step" entry point.** Rebuild's deletes
   are rule-driven here (fine), but a backend that wanted to push external
   retractions between steps still has no trait surface distinct from
   rule-driven removal (carried over from M1 friction #3, now exercised).

## Files

- `egglog-bridge-feldera/src/lib.rs` — ruleset-scoped `run_rules`, per-subset
  circuit cache, merge recognition in `add_table`, `fold_diffs_into_mirror`
  (delete-by-key + FD merge resolution).
- `egglog-bridge-feldera/src/compile.rs` — `MergeMode`, `Filter`, delete head
  actions, insert/delete diff streams, 2-atom join with cross-atom filters.
- `egglog-bridge-feldera/src/rule_builder.rs` — `remove` → delete action,
  `query_prim` (`!=`) → filter, `rename_prim` → name registry.
- `egglog-bridge-feldera/src/external_func.rs` — primitive name tracking.
- `egglog-bridge-feldera/tests/rebuild_proof.rs` — the M2 proof.
