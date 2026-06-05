# Milestone 6 — Killing the super-linear per-iteration blowup (Feldera backend)

## Symptom

On `tests/math-microbenchmark.egg` (symbolic-math eqsat, `(run N)`), the
Feldera backend's runtime exploded super-linearly with `N`, while the reference
bridge stayed flat. Reproduce:

```sh
sed 's/(run 11)/(run N)/' tests/math-microbenchmark.egg > /tmp/mm.egg
./target/release/egglog --feldera /tmp/mm.egg
```

## The bottleneck (measured)

The Feldera `run_rules` path runs a **host-side interpreter**
(`egglog-bridge-feldera/src/interpret.rs::run_iteration`); the DBSP join itself
turned out to be cheap. Per-iteration profiling (env flag, since removed)
showed the cost growing as **O(state²)**, dominated by three operations whose
cost was proportional to the *total accumulated e-graph state* (which grows
~exponentially in eqsat), not to the per-iteration delta:

| component (N=9, total over all iterations) | before | scaling N=7→N=9 |
|--|--|--|
| **`writes` (Remove retraction loop)** | **2675 ms** | **41 ms → 2675 ms ≈ 65×** (super-linear) |
| `apply_head` (`lookup_or_create` mirror rescans) | 733 ms | ~27× (super-linear) |
| `seen` advance (full snapshot clone per rule) | 479 ms | ~8.6× |
| `join` (DBSP body join + env/dedup) | 1486 ms | ~8.2× (≈ linear in state) |

The single worst-offender iteration retracted **15 049 rows** against a state of
**130 k rows**, spending **1291 ms in that one iteration's writes phase alone**.

### Root mechanism

1. **Retraction is O(removes × state).** `Write::Remove(f, key)` was applied one
   at a time, each doing a full `HashSet::retain` scan of the function's mirror.
   During rebuild the `@uf` rewrites retract many rows per iteration, so with
   `R` removes on a table of size `S` the cost is `O(R·S)` — quadratic, since
   both `R` and `S` grow with the e-graph. This was the dominant blowup.
2. **`lookup_or_create` is O(bindings × state).** Each RHS eq-sort constructor
   lookup linearly scanned the (growing) mirror to hash-cons. With many new
   terms created per round, this is `O(bindings · S)` per iteration.
3. **`seen` advance is O(rules × state).** Every rule cloned the full
   start-of-iteration snapshot of each body relation into its own `seen` set,
   re-cloning the same growing relations once per rule per call.

## The fix

All three were made cost-proportional to the work actually done, not to total
state. Changes are confined to `egglog-bridge-feldera/`:

1. **Batch retractions (`interpret.rs`).** Collect all `Remove` keys per function
   into a hash set, then do a **single** `retain` pass per touched function:
   `O(state)` total regardless of how many rows are retracted, instead of
   `O(removes × state)`. Removes are applied before sets, preserving the term
   encoder's `(@uf)` "delete old leader, set new leader" delete-then-set
   ordering.
2. **Iteration-scoped lookup index (`interpret.rs`).** `lookup_or_create` now
   uses a lazily-built `key → output` map per function: built once per iteration
   (`O(state)` the first time a function is touched) and kept updated as new rows
   are hash-consed, so every subsequent lookup is `O(1)`.
3. **Shared `seen` snapshots (`interpret.rs` + `lib.rs`).** `EGraph::seen` now
   stores `Rc<HashSet<Row>>`. The start-of-iteration snapshot of each relation is
   built once per `run_rules` call and shared by refcount across every rule's
   `seen` advance, replacing `O(rules × state)` cloning with `O(state)`.

These keep the load-bearing per-iteration model intact (one engine round per
egglog iteration; `(run N)` stays bounded — no switch to saturate-to-fixpoint)
and preserve exact seminaive semantics.

## Before / after (math-microbenchmark `(run N)`)

| run N | bridge | feldera before | feldera after |
|--|--|--|--|
| 3 | 0.01 s | 0.04 s | 0.04 s |
| 5 | 0.01 s | 0.10 s | 0.11 s |
| 7 | 0.02 s | 0.39 s | **0.31 s** |
| 9 | 0.02 s | 6.20 s | **2.28 s** |
| 11 | 0.02 s | >90 s (timeout) | **~43 s** |

The dominant `writes` component dropped from **2675 ms → 41 ms** at N=9 (65× →
linear); N=9 overall dropped **6.2 s → 2.3 s** and N=11 went from a timeout to
completing. Output matches the reference bridge **exactly at N=7 and N=9**.

## What remains

After the fix the remaining per-iteration cost (`join`, `read` snapshot, `delta`
computation, `seen`) is each `O(state)` per iteration — linear, not quadratic.
Because eqsat state itself grows ~exponentially in `N`, total time still grows
with `N`, but the **per-iteration super-linear (quadratic-in-state) term is
gone**. Fully matching the bridge's flatness would require a persistent DBSP
circuit fed only deltas across iterations (the host interpreter currently
re-reads the full mirror each `run_rules`); that is a larger architectural
change and is left as future work.

### Note: a pre-existing correctness divergence at N ≥ 10

At `N ≥ 10` the Feldera interpreter and the reference bridge produce different
table sizes (feldera produces *fewer* rows). This was confirmed **pre-existing
and independent of this fix** (it reproduces with the original per-write
ordering via a temporary probe, and was previously unobservable only because
N≥10 timed out before completing). It is a separate interpreter-semantics issue,
out of scope for this performance milestone, and is noted here for follow-up.

## Validation

- math-microbenchmark `(run 7/9/11)`: large drops (see table), output matches
  the bridge at N=7/9.
- `EGGLOG_TEST_FELDERA=1 cargo test --release --test files -- --skip rectangle`:
  **61/62 feldera cases pass**, zero feldera failures.
- Feldera proofs green: `rebuild_proof` (rebuild_matches_reference_tuple_counts),
  `run_n_proof` (run1_vs_run3_bounded_and_matches_reference,
  transitive_closure_join_runs_on_dbsp).
- duckdb + bridge still build; `cargo clippy -p egglog-bridge-feldera` clean.

## Flowlog

Flowlog is a distinct backend and was not exercised by this benchmark; the fix
lives entirely in `egglog-bridge-feldera` and does not touch shared code, so it
does not transfer automatically. If flowlog's host path shares the same
per-remove `retain` / linear hash-cons pattern, the same three fixes apply.
