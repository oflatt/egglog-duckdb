# Feldera persistent-circuit design + rebuild feasibility spike

**Goal.** Retire the host-side per-iteration interpreter
(`egglog-bridge-feldera/src/interpret.rs`) so the DBSP engine performs the
**whole** per-iteration evaluation — rule firing, primitive tail, head actions
(`set`/`union`/`delete`), and rebuild (union-find + congruence) — as a
**persistent circuit built once and fed input deltas across iterations**, with
per-iteration cost O(delta) (flat), matching the reference bridge.

This document is DESIGN + the result of a feasibility SPIKE that de-risks the
load-bearing unknown: **rebuild (non-monotone delete-and-rewrite) as DBSP
recursive views in a persistent circuit, converging correctly AND staying flat
per-iteration.** The existing interpreter remains in place as the correctness
oracle (NOT removed by this phase).

The verdict is at the end: **GO-PARTIAL on path A.**

---

## 0. Where we are today (the thing being replaced)

Current Feldera `run_rules` = **one host-driven egglog iteration**:

1. `interpret.rs` snapshots the whole relation mirror (`read`).
2. For each rule it computes a seminaive delta and runs the body join — on DBSP
   (`dbsp_join.rs`) when eligible, else a host nested loop.
3. Head actions (`set`/`delete`/`lookup`/`union`) are applied **host-side**;
   FD-merge conflicts resolved host-side; the mirror is folded.
4. Rebuild happens because the **frontend** schedules the term encoder's
   `parent` / `single_parent` / `uf_function_index` / `rebuilding` rulesets and
   calls `run_rules` for each, repeatedly, until `changed == false`.

Only the multi-atom body join runs on DBSP. Everything else — the loop, the
retraction application, the hash-cons lookups, the rebuild fixpoint driving — is
the host interpreter. Milestone 6 made each of those **linear** per iteration
(was quadratic), but the architecture still re-reads the full mirror every
`run_rules` call, so it cannot be flat. Flatness needs a persistent circuit
(this task).

The reference for what rebuild actually is: `seminaive-encoding-add.egg` (the
term encoder's output). Stripped of timestamp bookkeeping, rebuild is four
rule groups forming a **joint relation+union-find fixpoint with deletes**:

| ruleset | what it does | monotone? |
|---|---|---|
| `parent` | `UF(a,b),UF(b,c),b≠c ⟹ delete UF(a,b); set UF(a,c)` (path union) | **no** (deletes the stale edge) |
| `single_parent` | two parents of a node ⟹ keep the larger, redirect | **no** |
| `uf_function_index` | materialize `UF_f(a)=leader(a)`, `:merge (ordering-min)` | a `find` (min-aggregate) |
| `rebuilding` (congruence) | two view rows, equal args, different outputs ⟹ union outputs | monotone insert into UF |
| `rebuilding` (rebuild) | rewrite each view row to its leader output; **delete** the stale row | **no** (delete + re-insert) |

The non-monotone deletes (`parent`, `single_parent`, `rebuild`) feeding back into
more unions (`congruence`) is the "path A" research crux from `PLAN.md` §2.3.

---

## 1. The whole iteration as one persistent DBSP circuit

### 1.1 Relations as integrated input streams

Each egglog function/relation `f` becomes a DBSP input z-set `f_in:
ZSetHandle<Row>` plus an integrated output trace `f_out =
f_view.integrate().output()` that the host reads for `for_each` / `lookup_id` /
`table_size`. Seed rows (`add_values` / `add_term` / `insert_rows`) and the new
terms produced by user rules are pushed as **+1 deltas**; retractions
(`delete`) as **-1 deltas**. The host-side materialized mirror in `lib.rs`
becomes exactly these integrated output traces — no separate Rust copy.

DBSP can hold base values directly in a column enum (`Id`, `I64`, `F64`,
`Str`...), so the DuckDB-style BIGINT/DOUBLE round-tripping is unnecessary
(`PLAN.md` §3.4). `Row` would be a `DBData` enum-column tuple rather than the
current `Tup8<u32,…>`.

### 1.2 Rules as operators

A rule body `(R x y)(S y z)` lowers to `index R by y` ⋈ `index S by y` →
`map` to the head row (`dbsp_join.rs` already does this for eligible rules,
left-deep, with `!=` guards as `filter`s). The whole-program circuit wires every
user rule's join into the graph once. **Seminaive is deleted work**: DBSP's delta
calculus on the recursive scope IS seminaive — the hand-rolled "N variants per
rule, per-table `seen` snapshots" machinery in `interpret.rs` goes away
(`PLAN.md` §3.2).

The blocker on running head actions inside DBSP is **value-computing
primitives** (`+`, `from-string`, `bigrat`, …): a DBSP `map`/`filter` closure is
`Send + 'static` and cannot borrow the primitive engine. Two options:

- **Pure-Rust prims inside `map`** (`PLAN.md` §3.4) — re-implement the primitive
  set as `Send + 'static` closures registered into the circuit. This is the
  "engine does everything" end state.
- **Host-evaluated prim tail** (incremental migration) — keep value-computing
  primitives host-side initially; only the monotone relational rules + the whole
  rebuild move onto the circuit first. This is what the staged plan below does.

### 1.3 Rebuild as recursive non-monotone views

This is the spike's subject — see §2. In one `recursive` scope:

- `uf` — the accumulated union-edge relation (grows as congruence discovers
  equalities); a co-recursive view.
- `reach` — the symmetric-transitive closure of `uf` (e-class membership).
- `leader(x)` = `min { y : reach(x,y) }`, computed by antijoin (remove
  "dominated" pairs `(x,y)` for which some `(x,z), z<y` exists) — find without a
  typed `Min` aggregate.
- `canon_view` — each raw view row rewritten to its leader output
  (**non-monotone**: the row for a given `(args)` moves from `id` to `leader` as
  unions land; the old row is retracted as a negative z-set weight).
- `congru` — equal canonical args + different leader outputs ⟹ a new `uf` edge,
  fed back into the scope.

Symmetric `uf` makes `find` a min-over-component, which **sidesteps the
encoder's explicit `parent`/`single_parent` parent-edge DELETES** entirely: the
leader is recomputed from the (monotone, symmetric) union relation instead of
mutated in place. The remaining non-monotonicity lives where it is unavoidable —
the **view table** retraction (`rebuild`) — handled by z-set negative weights.

### 1.4 `(run N)` bounded iteration — preserved, NOT saturate-to-fixpoint

egglog's `(run N)` is **N bounded rounds with rebuild between them** — a
transitive-closure rule extends N hops, not to full closure. This is
load-bearing and must survive the migration.

The mapping: **one user-rule round per `transaction()`; rebuild runs to
fixpoint inside that same transaction.** Concretely the circuit has two
recursive regions with different intent:

- **rebuild** is genuinely run-to-fixpoint (a recursive scope) — correct,
  because the term encoder schedules rebuild as `(saturate …)`.
- **user rules** are applied **one hop per transaction** (NOT inside a
  saturating recursive scope) — the host calls `transaction()` exactly N times
  for `(run N)`, exactly as `interpret.rs`/`compile.rs` do today. The
  non-recursive single-hop-per-transaction model from Milestones 1–2 is kept for
  user rules; only rebuild gets a recursive scope.

So `run_rules(ruleset)` = push the ruleset's deltas + one `transaction()`; the
frontend's existing loop drives `(run N)` and `(saturate R)`. `changed` is
reported by whether any output trace changed this transaction (the no-op step in
`main.rs` confirms a delta-free `transaction()` leaves traces unchanged ⟹
`changed=false` ⟹ the frontend loop exits).

### 1.5 Reads / extraction

Between transactions, `for_each` / `lookup_id` / `table_size` read the
integrated output traces via `OutputHandle::consolidate()` (proven in `main.rs`
and the rebuild spike). Mid-rule reads are not supported
(`supports_inline_table_lookups = false`), same as DuckDB. Proof/term extraction
is read out of the proof-term relations like any other table — the backend needs
**no special proof machinery** (`PLAN.md` §3.5).

### 1.6 What the host still does (end state)

Ideally only: program I/O, pushing seed/user-rule-output deltas, reading output
traces for extraction, and **impure / value-computing primitives** that cannot
be expressed as `Send + 'static` circuit closures (until/unless they are
ported). The per-iteration *loop body* — joins, head writes, the entire rebuild
fixpoint — runs on DBSP.

---

## 2. The feasibility SPIKE (the load-bearing unknown)

Code: `spike-dbsp/src/rebuild.rs` (+ `persist_probe.rs`), standalone crate,
`dbsp = 0.150.0`, toolchain bumped to 1.91.0. Run:
`cargo run --release --bin rebuild` and `--bin persist_probe`.

The spike models the **actual rebuild** of `seminaive-encoding-add.egg` (UF
find + congruence + non-monotone canonical-view rewrite) as recursive views in
**one persistent circuit fed per-iteration deltas**, and checks every iteration
against a plain-Rust union-find+congruence **oracle**, while measuring
per-iteration cost as the e-graph grows. (The Phase-0 spike, `main.rs`, only
proved *generic* transitive closure + retraction on a toy `path` relation; it
did NOT model rebuild.)

### 2.1 Result A — convergence (correct for bounded e-classes)

For a commutativity-style eqsat (each iteration adds `(Add x y)`, a commuted
duplicate unioned to it, and a redundant term congruence must collapse), the
persistent circuit **matched the oracle exactly for the first 15 iterations**,
including the non-monotone canonical-view retraction (rows correctly moving from
`id` to `leader` and old rows disappearing) and congruence-discovered unions
feeding back into `find`. This is the core "path A" mechanism working end-to-end
in a persistent circuit.

### 2.2 Result B — per-iteration cost is FLAT (the headline)

With ~constant-size per-iteration deltas, step time was **essentially flat as
the e-graph grew** (growth ratio last-quarter / first-quarter ≈ **1.2×**, well
under the O(state) failure mode that blew up the interpreter on
math-microbenchmark). The persistent circuit feeds only deltas; DBSP's
incremental operators keep per-transaction work proportional to the change, not
the accumulated state. **The math-microbenchmark-style super-linear blowup is
gone in principle.**

### 2.3 Result C — the real blocker, isolated by an A/B probe

Pushing harder (a scenario where the e-graph grows monotonically and classes
chain deeply across many iterations), the circuit **diverged from the oracle**:
e.g. `leader[141]` should be `100` but the circuit gave `101`. The diagnostic
dump showed the smoking gun: the recursive `reach` view had **no rows at all for
older nodes** (`reach[100]=[]`, `reach[101]=[]`) even though their union edges
were present in the accumulated input — the closure of *previously-inserted*
facts was not retained across transactions for the **co-recursive** view.

I then wrote a minimal probe (`persist_probe.rs`) feeding a growing chain one
edge per transaction to a **single** recursive transitive-closure view (exactly
`main.rs`'s working pattern). It **persisted perfectly** — the full closure of
all edges fed so far was maintained at every transaction.

**Conclusion (decisive):** DBSP recursive views *do* persist across
`transaction()` calls. The divergence is **specific to the co-recursive,
non-monotone tuple** `(uf, reach)` with congruence feedback — when `uf` grows
*inside* a transaction via congruence (rather than arriving as a stable input
delta), the joint scope does not retain the prior-transaction closure for the
co-recursive `reach`. In other words: **closure-over-a-stable-input persists;
closure-over-a-co-recursively-grown relation, with non-monotone feedback, does
not (with the naive encoding).**

Two "obvious" fixes were tried and **both fail**, which is itself a useful
finding:
- `integrate()` at root then `delta0(child)` — the scope still sees only the
  per-step delta (delta0 imports the *nested* delta of the integral); old facts
  vanish ⟹ same divergence.
- `delta0(child).integrate()` *inside* the scope — the integral accumulates over
  the scope's INNER iterations, so the "input" grows every inner step and the
  fixpoint **never terminates** (the run hangs on transaction 0).

### 2.4 The fix — proposed, then IMPLEMENTED AND VALIDATED in the spike

Since closure-over-a-stable-input provably persists, route the
**congruence-discovered union edges back through the input boundary** instead of
co-recursively inside the scope:

- `uf` accumulates as an ordinary integrated **input relation** (seed unions +
  congruence edges), not a co-recursive view.
- `reach` is a **single** recursive closure over that `uf` input — the exact
  shape that persists in `persist_probe.rs`.
- Each transaction: run the scope, read the `congru` output, and **push any
  not-yet-applied congruence edge back into `union_in` and step again**, until no
  new congruence edge appears (a thin host-side **shuttle** for congruence edges
  *only*). Rebuild is scheduled `(saturate)` already, so multiple transactions
  per iteration are expected; each shuttle step moves only the new edges =>
  O(delta).

**This is implemented in `rebuild.rs` and it works.** With the shuttle, the
persistent circuit **matched the oracle on ALL 120 iterations** of the deep
deep-chaining scenario that broke the naive co-recursive encoding — the e-graph
grew from 3 to **360 canonical rows / 472 union nodes**, every leader and every
canonical-view row correct throughout, including the non-monotone view-row
retractions and multi-level congruence cascades. So the non-monotone recursive
rebuild **does converge in DBSP** with this encoding.

Per-iteration cost over the full 120-iteration run grew **~3.65×** while state
grew **~120×** — strongly sub-linear in state (the interpreter's failure mode is
linear-to-quadratic *in state*). The residual non-flatness comes from the
**read-out** operators recomputed at root each transaction — the leader-min
`antijoin` and the final `canon`/`congru` joins scale with the *reach* relation,
not the per-iteration delta. Making those fully incremental (or replacing the
join/antijoin `find` with a native min-aggregate / UF operator) is the
optimization that would flatten the last factor; it does not affect correctness.

An alternative that removes even the shuttle is a purpose-built **DBSP-native UF
operator** (a custom incremental operator maintaining find/union and exposing
leader as an output relation) — `PLAN.md`'s path B expressed *inside* DBSP. It
would also flatten the `find` read-out. This is the recommended optimization /
fallback (see §3).

---

## 3. Verdict + staged plan

### Verdict: **GO-PARTIAL on path A** (convergence risk RESOLVED; flatness needs one optimization).

- The whole-iteration-as-persistent-circuit thesis is **sound and the payoff is
  real**: rebuild-as-recursive-views in a persistent circuit converges
  **correctly** — the validated §2.4 encoding matched the Rust oracle on **all
  120 iterations** of a deep deep-chaining eqsat (3 → 360 canonical rows),
  including non-monotone view-row retraction and multi-level congruence cascades.
- Per-iteration cost is **strongly sub-linear in state** (state ×120, time
  ×3.65), eliminating the interpreter's linear/quadratic-in-state blowup. The
  residual factor is entirely in the **read-out** `find` operators (leader-min
  antijoin + final canon/congru joins) recomputed at root, not in the
  per-iteration delta path — fixable, see below.
- The make-or-break risk — *does the non-monotone recursive rebuild converge in
  a persistent DBSP circuit?* — is **answered YES**, with the important structural
  requirement that the **congruence feedback cross the input boundary** (§2.4):
  closure-over-a-stable-input persists across transactions, but
  closure-over-a-co-recursively-grown relation (the naive encoding) does not, and
  the two naive persistence work-arounds both fail (silent fact loss / hang). So
  the §2.4 shuttle (or a native UF operator) is **required**, not optional.
- It is GO-**PARTIAL** rather than unqualified GO only because (a) full
  per-iteration flatness needs the `find` read-out made incremental (a native
  min-aggregate or DBSP UF operator), and (b) the spike covers rebuild on a
  fixed-arity 2-arg view with leaf args; a production lowering must also
  canonicalize *arguments* (same `leader_of` join) and handle arbitrary arities
  and the value-computing-primitive boundary (§1.2). None of these are blockers —
  they are scoped follow-on work.

### Staged migration (each stage validated against the `interpret.rs` oracle)

The interpreter stays as the oracle the entire time; gate each stage behind a
flag and diff its output against the interpreter on the 61/62 feldera survey +
the rebuild/run-N proofs before advancing.

1. **Persist relations as integrated streams.** Replace the Rust mirror with
   DBSP input handles + integrated output traces; `add_values`/`for_each`/
   `lookup_id`/`table_size` go through them. Build the circuit once; rebuild it
   only when the rule/relation set changes (cache by sorted rule-id subset, as
   `EGraph::circuits` already anticipates). *Oracle check:* reads match exactly.
2. **Move the monotone user-rule joins onto the persistent circuit**, one hop
   per `transaction()`, value-computing prims still host-side on the join
   output. Delete the per-rule `seen` seminaive bookkeeping (DBSP delta calculus
   replaces it). *Oracle check:* `(run N)` bounded results match for N≤9.
3. **Move rebuild onto the circuit** as recursive views with the §2.4
   congruence-shuttle encoding (or a native UF operator). Retire the host
   retraction-application + FD-merge + hash-cons lookup paths. *Oracle check:*
   `rebuild_proof` tuple counts + `run_n_proof` match; flatness holds on
   math-microbenchmark `(run N)` for growing N (the whole point).
4. **Port value-computing primitives to `Send + 'static` circuit closures** so
   head actions run inside DBSP; the host is left with I/O + genuinely impure
   prims. *Oracle check:* full survey green.
5. **Retire `interpret.rs`** once stages 1–4 match the oracle across the survey
   and proofs. Keep it git-recoverable for one release as a fallback.

The new design should aim **not to inherit** the documented N≥10 interpreter
divergence (MILESTONE6): because rebuild becomes a single DBSP fixpoint rather
than a host-scheduled multi-ruleset loop, the ordering hazards that produce that
divergence do not arise — validate this explicitly at stage 3 by running N≥10
against the *reference bridge* (not the feldera interpreter oracle, which has the
bug).

### If the congruence shuttle proves inadequate (fallback)

If §2.4's host-side congruence feedback is too costly or still mis-converges on
real workloads, implement a **DBSP-native incremental union-find operator**
(custom operator maintaining `parent`/`find` with path-halving, exposing leader
as an output relation). This is `PLAN.md`'s path B expressed *inside* DBSP
rather than on the host — it keeps the persistent-circuit/flatness benefits while
removing the non-monotone-closure-persistence hazard entirely. The spike's
flatness result (Result B) still applies, since the rest of the iteration is
unchanged.

---

## Reproduce

```sh
cd spike-dbsp
cargo run --release --bin rebuild         # rebuild as persistent recursive views vs oracle + flatness
cargo run --release --bin persist_probe   # isolates: single recursive closure persists across txns
cargo run --release --bin spike           # Phase-0: generic closure + retraction + cycle termination
```
