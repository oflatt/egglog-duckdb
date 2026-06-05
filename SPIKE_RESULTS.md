# SPIKE RESULTS — egglog-on-DBSP (Phase 0, confirmatory)

**Verdict: GO (with caveats).** The `dbsp` Rust crate can be driven
in-process, synchronously, exactly the way an egglog backend needs. All four
load-bearing assumptions in `PLAN.md` were CONFIRMED by a real, compiled,
executed spike — not by reading docs alone.

- Project: `/tmp/egglog-feldera/spike-dbsp/` (standalone Cargo project, NOT a
  member of the egglog workspace — its `Cargo.toml` has an empty `[workspace]`
  table to detach it; egglog itself was never built).
- Engine: `dbsp` crate, pinned `=0.150.0` (see "Version note" below).
- Source of truth for the API: the actual extracted crate source at
  `/tmp/dbsp-src/dbsp-0.305.0/` and `/tmp/dbsp-0.150.0.crate`, plus the run.

---

## Version note (a surprise worth recording)

The latest published `dbsp` is **0.305.0** (verified via the crates.io API on
2026-06-04: `max_stable_version == newest_version == 0.305.0`). However
`dbsp >= 0.270.0` declares `rust-version = "1.91.1"`, and the newest toolchain
available in this environment is 1.91.0 (1.91.1 could not be installed —
`rustup`/toolchain-install was blocked). `dbsp 0.150.0` declares
`rust-version = "1.87.0"` (installed) and exposes the **identical public API**
used by an egglog backend:

| API used here | present in 0.150.0 | present in 0.305.0 |
|---|---|---|
| `RootCircuit::build -> (CircuitHandle, T)` | yes | yes |
| `RootCircuit::add_input_zset() -> (Stream, ZSetHandle)` | yes | yes |
| `ZSetHandle::push(k, w)` / `append` | yes | yes |
| `Circuit::recursive(closure)` | yes | yes |
| `Stream::delta0(child)` | yes | yes |
| `Stream::{map_index, join, plus, distinct, filter, map}` | yes | yes |
| `Stream::integrate()` | yes | yes |
| `CircuitHandle::transaction()` / `step()` | yes | yes |
| `OutputHandle::consolidate()` | yes | yes |

So the spike was built and run on 0.150.0; the code is API-compatible with
0.305.0. **For the real backend, budget a toolchain bump to >= 1.91.1 (ideally
current stable) so it can pin a recent `dbsp`.** The `dbsp` dep tree is heavy
(actix, tokio, tarpc, feldera-storage, etc.) but compiled fine (~2 min cold).

---

## What was tested (the program)

egglog graph reachability, modelled on a **plain 2-column relation** (no
weight/hop columns — this is the egglog shape, and it is what makes cyclic
graphs terminate):

```
path(x,y) :- edge(x,y).
path(x,z) :- path(x,y), edge(y,z).
```

`edge` is an **input relation** fed by the host through a push handle; `path`
is a **recursive view**. Four transactions exercise insert, retraction, cycle
termination, and a no-op.

---

## The concrete `dbsp` API (as used)

### Build the circuit once, up front (static dataflow graph)

```rust
let (circuit, (edge_in, path_out)) = RootCircuit::build(|root| {
    // input relation: returns (stream, push-handle)
    let (edges, edge_in): (Stream<RootCircuit, OrdZSet<Edge>>, ZSetHandle<Edge>) =
        root.add_input_zset::<Edge>();

    // the recursive Datalog ruleset, in ONE recursive scope
    let path: Stream<RootCircuit, OrdZSet<Edge>> = root.recursive(
        |child, path: Stream<_, OrdZSet<Edge>>| {
            let edges = edges.delta0(child);          // import parent stream
            let base  = edges.clone();                // path(x,y) :- edge(x,y)
            let step  = path.map_index(|Tup2(x,y)| (*y, *x))   // key on join col
                .join(&edges.map_index(|Tup2(y,z)| (*y, *z)),
                      |_y, x, z| Tup2(*x, *z));        // path(x,z) :- path(x,y),edge(y,z)
            Ok(base.plus(&step))                       // UNION; `recursive` adds distinct
        },
    )?;

    // integrate() => the output handle holds the ACCUMULATED relation
    Ok((edge_in, path.integrate().output()))
})?;
```

### Feed deltas, run to fixpoint, read back (the per-step loop)

```rust
edge_in.push(Tup2(0u64, 1u64), 1);   // +1 weight = insert
edge_in.push(Tup2(1u64, 2u64), -1);  // -1 weight = retract
circuit.transaction()?;              // one logical clock tick (see below)
for (Tup2(x, z), (), w) in path_out.consolidate().iter() { /* read */ }
```

### `transaction()` vs `step()` — the one runtime gotcha

`CircuitHandle::step()` **panics/errors if called outside a transaction**
(`"step called outside of a transaction"` — observed on the first run).
`CircuitHandle::transaction()` is the right synchronous call: its doc says
*"Start and instantly commit a transaction, waiting for the commit to
complete."* That is exactly "absorb this batch of input deltas, run all
recursive scopes to their internal fixpoints, emit/commit output." **The
`run_rules -> step` adapter in PLAN.md should be `run_rules -> transaction()`,
not the raw `step()`.** (Raw `step()` exists for manual transaction control:
`start_commit_transaction()` + loop on `step()` until `is_commit_complete()`.)

---

## CONFIRMED claims (actual run output)

```
STEP 1 (insert chain 0->1->2->3->4)
  path = { 0->1, 0->2, 0->3, 0->4, 1->2, 1->3, 1->4, 2->3, 2->4, 3->4 }  (n=10)
  OK: full closure (10 pairs) computed to fixpoint in ONE step

STEP 2 (retract edge 1->2)
  path = { 0->1, 2->3, 2->4, 3->4 }  (n=4)
  OK: closure SHRANK correctly after retraction (10 -> 4 pairs)

STEP 3 (restore 1->2, add 4->2 forming cycle 2->3->4->2)
  path = { 0->1, 0->2, 0->3, 0->4, 1->2, 1->3, 1->4, 2->2, 2->3, 2->4,
           3->2, 3->3, 3->4, 4->2, 4->3, 4->4 }  (n=16)
  OK: cyclic graph reached FIXPOINT (distinct => termination), self-loops present

STEP 4 (no input delta): path unchanged (n=16) => adapter reports changed=false

ALL ASSERTIONS PASSED
```

1. **Recursion runs to fixpoint inside one transaction — CONFIRMED.** STEP 1
   yields the *complete* transitive closure (10 pairs of a 5-node chain) after
   a single `transaction()`. The recursive scope iterates `y = distinct(f(i,y))`
   internally to convergence; the host calls the engine exactly once per
   logical tick. This is the core "one `run_rules` ruleset call = one
   saturating step" assumption — validated.

2. **Cyclic Datalog terminates — CONFIRMED (and stronger than the tutorial).**
   The official dbsp tutorial computes a *weighted* closure that **diverges on
   cycles** (weights grow without bound). egglog's `path(x,y)` has no such
   columns, and STEP 3 introduces a real cycle `2->3->4->2`: it terminated and
   produced the correct closure *including self-loops* `2->2, 3->3, 4->4`. The
   built-in `distinct` on the recursive body (confirmed in the operator's own
   doc diagram: `y = distinct(f(i+Δi, y))`) is the termination guarantee. This
   is the egglog-relevant case and it works.

3. **Read the relation back between steps — CONFIRMED.** Reading happens
   between transactions via `OutputHandle::consolidate()`. **Caveat that
   changes the mental model:** the *recursive view's stream is a DELTA stream*
   (DBSP recursion emits `Δx = new - old`; a retraction appears as
   negative-weight rows). To get the *accumulated* relation egglog needs, you
   must `integrate()` the stream before `.output()`. With `integrate()`,
   `consolidate()` returns the full current relation — done here and verified
   across all four steps.

4. **Retraction shrinks results — CONFIRMED.** STEP 2 pushes `-1` for edge
   `1->2`; the closure correctly shrinks 10 -> 4 pairs (the `0->{2,3,4}` and
   `1->{2,3,4}` paths that depended on `1->2` vanish). This is precisely the
   delete-and-rewrite primitive egglog's rebuild relies on, and DBSP's
   negative-weight algebra does it natively, incrementally.

---

## The `run_rules -> transaction` adapter (concrete shape observed)

```
run_rules(ruleset):
  1. drain staged inserts/retractions into the relation input handles:
        edge_in.push(row, +1)   // insert_rows
        edge_in.push(row, -1)   // retraction (rebuild delete)
  2. circuit.transaction()?     // ONE logical tick; recursive scope saturates
  3. refresh the Rust-side materialized mirror from the integrated output
        handles (consolidate()), so lookup_id / for_each / point lookups read
        committed state
  4. report changed = (mirror changed this tick)   // STEP 4 shows a no-op
                                                    // tick leaves it unchanged
```

- **The Rust-side materialized mirror is just the integrated output handle.**
  `path.integrate().output()` *is* the mirror: `consolidate()` yields the full
  relation as an `OrdZSet`, iterable as `(key, (), weight)`. For point lookups
  the backend keeps a `HashMap`/`BTreeMap` refreshed from `consolidate()` after
  each transaction (rebuilt once per tick, not per query). Reads *during* a
  rule body still cannot reenter the engine, so keep
  `supports_inline_table_lookups = false` (same as DuckDB).
- **Base values / rows are arbitrary Rust types.** Rows here are
  `Tup2<u64,u64>`; any `Clone + Hash + Ord + ...` (the `DBData` bound) works,
  confirming PLAN §3.4 — a `Value` enum can live directly in a row, no SQL
  type-squeezing or string interning needed.
- **No hand-written seminaive variants.** The recursive join is incrementalized
  by the engine; we wrote one `join`, not N timestamp-focused variants
  (confirms PLAN §3.2).

---

## Surprises that feed the Phase-B trait refactor

1. **`run_rules` should map to `transaction()`, and the trait wants a
   `runs_ruleset_to_fixpoint()` hint.** One `transaction()` already saturates
   the ruleset. STEP 4 proves a redundant tick is a safe no-op (relation
   unchanged), so the frontend's outer loop terminates either way — but the
   hint lets it skip the wasted tick. Low-risk, worth adding (PLAN §4.3 #1).

2. **Output is a DELTA stream; egglog wants accumulated state — interpose
   `integrate()`.** This is a real interface requirement, not a footnote: every
   relation the backend exposes for reading must be `integrate()`d before
   `.output()`. The integrated handle doubles as the point-lookup mirror.
   (Refines PLAN §1's "consolidate yields the accumulated Z-set" — it only does
   so *after* integrate; the raw recursive stream is incremental.)

3. **Static circuit vs. incremental rule addition is the next real plumbing
   risk (not validated here).** The circuit graph is fixed at `build` time
   (`RootCircuit::build(|root| ...)`), so adding/removing rules after the fact
   (the proof encoder emits many; `free_rule`) means **rebuild the circuit and
   replay current relation contents as initial deltas**. The push-handle +
   integrate machinery makes replay mechanical (snapshot via `consolidate()`,
   re-`push` into the fresh circuit), which also gives `clone_boxed` / push-pop
   — but this was *assumed*, not exercised, by this spike. It is the first
   thing Phase 1 must prove.

4. **Toolchain/version coupling.** Pin a recent `dbsp` (0.3xx) and bump the
   project MSRV to >= 1.91.1 before building the real backend; the heavy
   transitive dep tree (actix/tokio/feldera-storage) is the build-cost reality
   the PLAN's risk #6 flagged.

---

## GO / NO-GO

**GO — with caveats.** The make-or-break plumbing of Phases 0-1 is real:
in-process synchronous driving, recursion-to-fixpoint in one tick (including
cyclic termination via `distinct`), reading accumulated state between ticks,
and native incremental retraction all work on the actual crate. The
`run_rules -> transaction()` adapter and the integrate-as-mirror pattern are
concrete and small.

Caveats / not yet de-risked (deliberately out of Phase-0 scope):
- **circuit rebuild + state replay** for incremental rule add/remove and
  push/pop (Phase 1 must prove this cheap enough);
- **rebuild as non-monotone recursive views** — the union-find / congruence
  `find`-via-aggregate + key-rewrite story (PLAN §2.3 path A) remains the
  genuine research crux and is the Phase-3 go/no-go gate;
- **toolchain bump** to pin a current `dbsp`.

None of these is contradicted by the spike; they are simply the next gates.
The engine itself behaves exactly as the PLAN bet on.

---

## How to reproduce

```
cargo run --release --manifest-path /tmp/egglog-feldera/spike-dbsp/Cargo.toml
```

(Uses `rust-toolchain.toml` pinning 1.87.0, which `dbsp 0.150.0` supports.)
Files: `/tmp/egglog-feldera/spike-dbsp/{Cargo.toml,rust-toolchain.toml,src/main.rs}`.
