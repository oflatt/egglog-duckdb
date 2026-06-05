# Plan: A Feldera/DBSP Backend for egglog

A research-phase analysis (EXPLORE + PLAN only — no implementation yet) of
whether **Feldera** / the **DBSP** incremental-computation engine can serve
as an execution backend for egglog, behind the existing
`egglog-backend-trait` `Backend` interface, with term/proof encoding as a
goal.

This document is the Feldera counterpart to
[`duckdb-backend-plan.md`](duckdb-backend-plan.md) and assumes the same
**term-encoded IR** as the cut point (see
[`src/proofs/proof_encoding.md`](src/proofs/proof_encoding.md)): after term
encoding, `union`, congruence, rebuild, deletion, and custom merges are all
compiled into *ordinary egglog rules* over ordinary relations. The backend
sees plain Datalog + two upsert modes (`:merge old`, `:merge new`) +
primitives + a schedule.

This is a planning document, not a commitment.

---

## 1. Feldera / DBSP model summary

**DBSP** is the formal model and Rust library behind Feldera. Its world is
**streams of changes** rather than tables:

- **Z-sets.** The universal collection type. A Z-set is a map from rows to
  *integer weights*. Weight `+1` = "this row is present"; `+n` = "present
  with multiplicity n"; **negative weights = retraction/deletion**. Z-sets
  form a commutative group (you can add and negate them), which is what
  makes incremental math work: a delta is just another Z-set, and applying
  it is group addition.
- **IndexedZSet.** A Z-set keyed by a column for fast joins/aggregates.
- **Circuits.** You build a static dataflow graph once (`RootCircuit::build`
  or `build_with`), wiring operators: `map`, `filter`, `flat_map`, `join`,
  `antijoin`, `distinct`, `aggregate`, `plus`/`minus`, etc. Operators
  consume and produce *streams of Z-sets*.
- **Incrementality is the default.** DBSP's whole point: feed a *delta* to
  inputs, get the *delta* to every downstream view, in time proportional to
  the change — not the database size. There is a formal `integrate` /
  `differentiate` pair and a delay operator `z⁻¹`; the incremental version
  of any operator is mechanically derived.
- **Recursion / fixpoint.** `ChildCircuit::recursive` (a.k.a. the recursive
  scope) builds a nested circuit with a feedback edge. `delta0` imports a
  parent stream into the child scope. Inside, you express "new = base ∪
  step(old)" and the scope **iterates to a fixed point within one outer
  step**, automatically halting when the stream stops changing. `distinct`
  inside the loop is what guarantees termination for monotone-ish queries.
- **Stepping.** You push input deltas through input handles
  (`ZSetHandle::push` / `append`), call `circuit.step()` (or
  `transaction()`), then read output handles (`consolidate()` yields the
  accumulated Z-set or the delta). One `step()` = "absorb this batch of
  input changes, run all recursive scopes to their internal fixpoints,
  emit output deltas."

**Two ways to use Feldera:**

1. **SQL pipeline interface.** Write SQL (Feldera dialect), Feldera compiles
   it to a DBSP circuit and runs it as a managed pipeline (REST/HTTP, Kafka,
   etc.). Feldera SQL notably supports **mutually-recursive, non-monotone
   recursive views** (no `WITH RECURSIVE` CTE restriction; recursive views
   are auto-`DISTINCT`; everything except window functions is allowed in a
   recursive definition).
2. **The `dbsp` Rust crate directly.** Build the circuit programmatically in
   the host process, push Z-set deltas through input handles, `step()`, read
   output handles. In-process, no server, full control over the circuit
   graph and over what you put in a row (arbitrary `Clone + Hash + Ord` Rust
   types).

---

## 2. The single most important question: does DBSP's incremental model
## *help* or *fight* egglog's fixpoint-then-read model?

This is where the analysis must be honest, because it is the crux.

### 2.1 What egglog actually does

egglog's outer loop (in the **frontend**, not the backend — see §4) is:

```
loop:
  for each rule in ruleset: run_rules([rule])   # one seminaive pass
  flush_updates()                               # apply staged inserts/unions
  rebuild()                                      # canonicalize until UF stable
  if nothing changed: break                     # saturation
read database
```

Two kinds of "change-to-fixpoint" are happening:

- **Datalog seminaive saturation** (monotone growth of relations).
- **Rebuild** — congruence closure over the union-find, which **deletes and
  rewrites rows** (a row `f(a)=x` whose `a` got unioned to `b` is *replaced*
  by `f(b)=x`; the old row is removed). This is **non-monotone, in-place
  mutation**.

### 2.2 The good news: DBSP is *built* for the monotone part

Datalog-to-fixpoint is exactly DBSP's recursive-scope use case (the
transitive-closure tutorial is literally a Datalog recursion). The
term-encoded egglog program is "plain Datalog + upserts," and:

- Each egglog relation → a DBSP stream/IndexedZSet.
- Each rule → `join`(body atoms) → `map`(head) → fold into the head
  relation's stream, all inside a recursive scope.
- Seminaive is **subsumed for free**: DBSP's incrementalization derives the
  delta form of each join automatically. egglog hand-rolls "N variants per
  rule, one focused on each atom's `ts >= last_run`" (see DuckDB
  `compile.rs`); in DBSP you do **not** write those variants — the engine's
  `z⁻¹`/integrate machinery produces them. This is a genuine *elegance win*:
  DBSP's incremental join *is* seminaive.

So for the pure-Datalog skeleton, DBSP is arguably a *better* fit than
DuckDB: the engine does seminaive for you and runs the recursion to fixpoint
inside one `step()`.

### 2.3 The bad news (the central risk): rebuild is deletion+mutation in a
### monotone-incremental engine

DBSP *can* delete — negative weights retract rows. But egglog's rebuild is
not "the input changed, propagate the delta." It is "an *internal*
equality fact (a union) forces us to **rewrite the keys of existing derived
rows** and then re-run congruence, which can force more unions, … to
fixpoint." Concretely:

- A union `a ~ b` means every row mentioning `a` must be canonicalized to
  mention `find(a)`. In Z-set terms that is: emit `-1·f(a,…)` and
  `+1·f(find(a),…)` for *every* affected row.
- After canonicalization, two rows may collide on the key (congruence),
  which **creates a new union**, which feeds back into the union-find, which
  triggers more canonicalization. This is a fixpoint *over the union-find
  and the relations jointly* — exactly the cyclic, non-monotone recursion
  Feldera's recursive-views feature is advertised to support, but it is the
  hardest case.

There are two viable encodings, both with real cost:

- **(A) Encode union-find as relations and rebuild as recursive rules**
  (the term-encoding approach). `UF_<Sort>(child, parent)` plus the
  canonicalization/congruence rules become DBSP recursive views. The
  `find`-min and congruence-collapse are non-monotone (they retract
  stale parent edges / stale rows), but Feldera's recursive views *do*
  allow non-monotone bodies and auto-`DISTINCT`. **This is the
  DBSP-idiomatic path**: rebuild becomes data, and DBSP's own fixpoint
  runs it. The risk is whether the `find`-as-aggregate (`MIN` over a
  reachability closure) and the "rewrite every row's key" join converge
  efficiently and *correctly* under DBSP's distinct/retraction semantics.
- **(B) Keep a native union-find in host Rust** (like DuckDB's `uf.rs`) and
  drive canonicalization by pushing retraction+insertion deltas into the
  circuit between steps. This breaks DBSP's "internal fixpoint" — you'd
  step, read displaced ids, compute the rewrite delta in Rust, push it,
  step again, loop until stable. That **re-introduces egglog's outer loop**
  and throws away much of DBSP's incremental advantage for the rebuild part
  (though seminaive for the Datalog part still benefits).

**Honest assessment:** the monotone Datalog core is an *excellent* fit; the
union-find/rebuild core is the make-or-break risk, and path (A) — express
rebuild as non-monotone recursive views and let DBSP own the fixpoint — is
the bet that justifies choosing Feldera at all. If (A) doesn't converge
correctly/efficiently, the backend degrades to (B), which is strictly worse
than the DuckDB backend (more moving parts, same outer loop).

---

## 3. How egglog maps onto Feldera/DBSP

### 3.1 Rule compilation

| egglog construct | DBSP construct |
|---|---|
| Relation / function table | input `ZSetHandle` + an `IndexedZSet` stream |
| Rule body (join of atoms) | `join` / `antijoin` chain over indexed streams |
| Filter / primitive guard | `filter` / `map` closure (pure Rust) |
| Rule head `set`/`insert` | `map` producing rows, summed into the relation's stream |
| `:merge old` / `:merge new` | `distinct` (old) or a keyed `aggregate` picking new |
| Whole ruleset, run to fixpoint | one `recursive` scope containing all the rules |
| Schedule (`run`, `saturate`, `seq`) | sequencing of `step()`s / nested scopes (see §4) |

A rule LHS like `(R x y) (S y z)` compiles to: index `R` by `y`, index `S`
by `y`, `join` → bind `x,y,z` → `map` to the head row → add to head stream.
**No seminaive variants are written by hand** — DBSP incrementalizes the
join.

### 3.2 Seminaive — subsumed, not reimplemented

The DuckDB backend reimplements seminaive (per-table `ts` column,
`last_run_at`, N focused variants). On DBSP this is **deleted work**: the
recursive scope + DBSP's delta calculus *is* seminaive. We should *not* add
a timestamp column. This is the clearest place Feldera improves on the
DuckDB design.

### 3.3 Union-find + rebuild + congruence — the hard part (see §2.3)

Recommended primary plan: **path (A)** — lower exactly the term-encoded
program. Term encoding already emits `UF_<Sort>`, `UF_<Sort>f`, view tables,
deferred-delete/subsume helpers, and congruence/rebuild *as egglog rules*
(`proof_encoding.rs`). So:

- `UF_<Sort>(child → parent)` is a relation; `find` is "follow to the
  representative," expressible as a recursive view with a `MIN`/`argmin`
  aggregate over the closure (Feldera supports non-monotone recursive
  aggregates).
- Canonicalization rules rewrite view-table keys; collisions trigger new
  `UF` edges; congruence rules fire — all inside the same recursive scope,
  so DBSP runs the joint relation+UF fixpoint itself.
- Deletion/subsumption (deferred-delete rulesets) become **negative-weight
  retractions** — natural in DBSP.

Fallback: **path (B)** native UF in Rust if (A) misbehaves; this is a
milestone-gating experiment, not a default.

### 3.4 Primitives and base values

- **Base values** (`i64`, `f64`, `bool`, `String`, `BigInt`, …): rows in a
  DBSP Z-set can hold arbitrary `Clone+Hash+Ord` Rust values, so unlike
  DuckDB (which must squeeze everything into BIGINT/DOUBLE columns and
  intern strings — see `base_values.rs`), **DBSP can store base values
  directly** in a column enum (`Value` ≈ `enum { Id(u64), I64(i64),
  F64(OrderedF64), Str(InternId), … }`). This is *simpler* than DuckDB.
- **Primitives** (`+`, `<`, `from-string`, `bigrat`, …): pure Rust closures
  inside `map`/`filter` — no UDF/ABI dance, no "can't reenter the database"
  restriction that forced DuckDB to set `supports_inline_table_lookups =
  false`. Primitives that *do* need to read a table can be encoded as joins,
  or (worst case) need the same gating DuckDB uses.

### 3.5 Term / proof encoding — how proofs work (a paper goal)

**Key finding: the backend needs essentially zero special proof
machinery.** Proof/term encoding (`src/proofs/proof_encoding.rs`) is a
**frontend source-to-source transformation**: it rewrites the user program
into one with explicit term constructors, proof-term constructors, UF
tables, and congruence/rebuild rules — all expressed in *the same egglog
primitives the backend already implements* (tables, rules, `union` →
UF-edge inserts). So once the Feldera backend can faithfully run plain
rules + relations + retraction, it runs the proof-encoded program for free,
and proofs are read out of the proof-term relations like any other table
(`for_each`).

This is the same posture the DuckDB backend takes (it targets *term-encoded
mode only*), and it is the reason this whole effort is plausible: **proofs
are a property of the encoded IR, not of the backend.** The Feldera
backend's job for proofs is purely: (1) preserve row identity / term
hash-consing so proof terms point at the right things, and (2) iterate the
proof relations out at the end.

---

## 4. Does the `Backend` / `RuleBuilderOps` interface fit? (most likely
## break point)

Short answer: **the data-shaped methods fit fine; the control-flow methods
are a semantic mismatch that needs an adapter, not a rewrite.**

### 4.1 What fits cleanly

`add_table`, `add_term`, `add_values`, `insert_rows`,
`lookup_constructor_rows`, `lookup_id`, `for_each` / `for_each_while`,
`clear_table`, `base_values`, `base_value_pool`, `fresh_id`,
`get_canon_repr`, capability flags, `clone_boxed`. These are all
"describe/poke/read the relational state" and map onto building input deltas
and reading output handles.

`RuleBuilderOps` also fits structurally: it is an accumulator (the DuckDB
impl already "accumulates calls into an internal IR and submits on
`build()`"). The Feldera impl does the same — accumulate body atoms/actions,
and on `build()` **wire that rule's operators into the circuit graph**
rather than compile SQL.

### 4.2 Where it breaks: `run_rules` / `flush_updates` and the circuit
### lifecycle

The trait (and the frontend that drives it) assume:

> *build rules incrementally, then call `run_rules([ids])` to run a
> ruleset's rules **once**, call `flush_updates()`, and repeat in an outer
> loop until saturation; the database is mutable and queryable at every step.*

DBSP wants the opposite:

> *define the entire circuit graph **once, up front**, then `step()` it over
> input deltas; the recursive scope runs the ruleset **to fixpoint** in a
> single step.*

Concrete frictions:

1. **"Run once" vs "run to fixpoint."** The frontend calls `run_rules`
   repeatedly expecting one pass each. A DBSP recursive scope does the whole
   fixpoint in one `step()`. The adapter must either (a) treat one
   `run_rules` ruleset call as "one `step()` that internally saturates that
   ruleset" and report `changed=false` once stable so the frontend's outer
   loop exits after one or two iterations, or (b) build *non-recursive*
   per-pass circuits and lose DBSP's main advantage. **(a) is the right
   choice** and is mostly compatible with how `run_rules` returns an
   `IterationReport{changed}`.

2. **Rules added incrementally after the circuit exists.** egglog can add
   rules between runs (and the proof encoder generates many rules). DBSP
   circuits are *static* once built. Options: **rebuild the circuit lazily**
   whenever the rule set changes (acceptable — rule-set changes are far
   rarer than `step`s, and a built circuit can be re-derived from the
   accumulated IR + current relation contents pushed back in as initial
   deltas), or pre-declare a "universe" circuit. The lazy-rebuild approach
   means the backend must be able to **snapshot relation contents and replay
   them into a fresh circuit** — which also gives us `clone_boxed`
   (push/pop) for free.

3. **`free_rule`.** Removing a rule = rebuild the circuit without it. Same
   mechanism as #2.

4. **Mutable mid-run reads (`lookup_id`, `for_each` between iterations).**
   DBSP output handles expose the consolidated relation after a `step()`, so
   reads between `run_rules` calls are fine — but reads *during* a rule's
   apply body (the `supports_inline_table_lookups` path) are not, exactly as
   on DuckDB. Set `supports_inline_table_lookups = false`.

5. **`with_execution_state` / `action_registry` / `TableAction` escape
   hatch.** These are bridge-specific (the trait already documents them as
   reference-backend-only, gated by capability flags). Feldera stubs/errors
   them like DuckDB.

### 4.3 Verdict on the interface

The trait's *shape* (accumulate-a-rule, run-a-ruleset, read-the-db) is
serviceable behind an adapter. **The one real conceptual tension is
"`run_rules` = one pass" vs "`step` = to fixpoint."** It is resolvable by
making a `run_rules(ruleset)` call equal one transactional `step()` that
saturates that ruleset's recursive scope, and reporting `changed` honestly.
No trait signature change is strictly *required* for a first cut — but two
optional additions would make the fit cleaner and are worth proposing:

- A capability/hint method like `runs_ruleset_to_fixpoint() -> bool` so the
  frontend can skip its own redundant outer loop when the backend already
  saturates (avoids re-stepping a converged circuit). Without it, the
  frontend's loop still terminates (the backend reports `changed=false`),
  just with one wasted no-op step.
- Confirmation that `clear_table` + bulk re-`insert_rows` is an acceptable
  way to implement circuit rebuild / push-pop, or a dedicated
  snapshot/restore pair. The DuckDB plan flags push/pop as a known gap;
  Feldera's "replay into a fresh circuit" model can support it.

---

## 5. Proposed crate structure: `egglog-bridge-feldera`

Mirror `egglog-bridge-duckdb/` (it already implements the same trait):

```
egglog-bridge-feldera/
  Cargo.toml            # depends on egglog-backend-trait, egglog-core-relations, dbsp
  src/
    lib.rs              # EGraph struct: circuit handle(s), rule IR store, relation
                        #   handles, base-value pool, fresh-id counter, schedule state
    backend_impl.rs     # impl Backend for EGraph (run_rules→step, for_each→output
                        #   handle consolidate, add_table→register relation, etc.)
    rule_builder.rs     # impl RuleBuilderOps: accumulate body/actions into a
                        #   `feldera::Rule` IR; build()=wire operators into circuit
    compile.rs          # Rule IR → DBSP operator graph (join/map/filter/distinct,
                        #   recursive scope assembly, merge-mode lowering)
    circuit.rs          # circuit lifecycle: (re)build from current IR + relation
                        #   snapshots, step()/transaction(), output consolidation
    uf.rs               # ONLY if path (B) fallback is needed (native UF in Rust)
    base_values.rs      # Value enum / column representation; intern table for
                        #   exotic base & container values
    external_func.rs    # primitive registry → Rust closures used in map/filter
  examples/             # mirror duckdb examples for parity testing
```

Decision to make early: target the **`dbsp` Rust crate**, not the SQL
pipeline (argued in §7).

---

## 6. Phased implementation plan

**Phase 0 — Spike (throwaway).** In a scratch binary, build a DBSP circuit
by hand for one hard-coded egglog program (e.g. transitive closure / a
path-rewrite) using the `dbsp` crate. Goal: learn `RootCircuit::build`,
`recursive`, `delta0`, input/output handles, `distinct`, `join`, and
how `step()` + recursive scope actually saturates. **Deliverable: convince
ourselves the recursive scope runs a Datalog ruleset to fixpoint in one
step and we can read the result.** No trait involved.

**Phase 1 — Minimal first milestone (see §8).** Plain relations + rules, no
UF. Implement enough of `Backend`/`RuleBuilderOps` to run a Datalog-only
egglog program (no `union`, no merges beyond `:merge new`) and read it back.
Prove the `run_rules → step` adapter and circuit-rebuild-on-rule-add work.

**Phase 2 — Merges + base values + primitives.** `:merge old`/`:merge new`
via `distinct`/keyed aggregate; the `Value` column enum; primitives as Rust
closures. Run term-encoded programs that don't exercise heavy rebuild.

**Phase 3 — Union-find + rebuild via path (A).** Lower the term-encoded
`UF_*` tables and congruence/rebuild rules as DBSP recursive views with
non-monotone aggregates + retraction. **This is the research crux**; gate
with a decision point: if (A) is correct+fast, proceed; else fall back to
path (B) (native `uf.rs`, retraction deltas pushed between steps).

**Phase 4 — Proof mode + parity.** Because proofs are a frontend encoding
(§3.5), this should largely "just work" once Phases 1–3 run encoded
programs faithfully. Run the proof test suite; iterate the proof relations
out via `for_each`.

**Phase 5 — Performance + push/pop.** Tune indexing, batch sizes; implement
`clone_boxed` / push-pop via snapshot-and-replay.

---

## 7. Recommendation: `dbsp` crate, not the SQL pipeline

Target the **`dbsp` Rust crate directly**, embedded in-process.

- **For:** in-process (no server/Kafka/REST hop), full control of the
  operator graph (egglog's rebuild needs custom recursion shapes the SQL
  compiler may not expose), arbitrary Rust types in rows (base values,
  interned terms — no SQL type-squeezing like DuckDB needs), Rust closures
  as primitives (no UDF ABI), and it matches the existing
  `egglog-bridge-duckdb` in-process posture and the `Backend` trait's
  synchronous `&mut self` methods.
- **Against:** lower-level API, less documentation than the SQL surface, and
  we hand-build seminaive/recursion wiring (but that wiring *is* the
  research contribution).
- **Why not the SQL pipeline:** it is built for managed streaming
  deployments (HTTP/Kafka I/O, a running pipeline service). Bending egglog's
  synchronous, embedded, push/pop, fresh-id-allocating loop onto a streaming
  SQL *service* is a worse impedance match than building circuits directly —
  even though Feldera SQL's *non-monotone recursive views* are exactly the
  feature that makes rebuild expressible. We borrow that **idea** (rebuild
  as recursive views) but implement it on the `dbsp` crate.

---

## 8. Recommended first milestone (concrete)

**"Datalog to fixpoint, behind the trait."**

A `egglog-bridge-feldera` crate that implements just enough of `Backend` +
`RuleBuilderOps` to run a **union-free, primitive-light, single-ruleset
egglog program** (e.g. graph reachability / a non-equational rewrite set):

1. `add_table` registers a relation (input `ZSetHandle` + indexed stream).
2. `RuleBuilderOps` accumulates body atoms + a `set`/`insert` head; `build()`
   wires `join`→`map`→fold into one shared **recursive scope**.
3. `run_rules([ids])` = build (or reuse) the circuit, push seeded rows as a
   delta, `step()` once (scope saturates internally), report
   `changed=false` once stable.
4. `for_each` reads the consolidated output handle.
5. Verify the result equals the reference backend on the same program.

Hitting this proves the three things that make-or-break the whole approach:
the **`run_rules`→`step` adapter**, **DBSP recursion = Datalog seminaive to
fixpoint**, and **circuit (re)build from the accumulated rule IR**. It
deliberately defers the genuine research risk — union-find/rebuild as
non-monotone recursive views (Phase 3) — until the plumbing is proven.

---

## 9. Risks and unknowns

1. **(Biggest) Rebuild as non-monotone recursive views (path A).** Whether
   `find`-via-aggregate + key-rewriting + congruence-collapse converges
   *correctly and efficiently* under DBSP's `distinct`/retraction semantics
   inside one recursive scope. If not, fall back to native-UF path (B),
   which loses much of the DBSP advantage.
2. **Static circuit vs incremental rule addition.** Rebuilding the circuit
   when rules/relations change (and replaying state) must be cheap enough;
   the proof encoder generates many rules.
3. **`run_rules`-once vs scope-to-fixpoint semantics.** The adapter must
   report `changed` so the frontend's outer loop terminates without infinite
   re-stepping; a `runs_ruleset_to_fixpoint()` trait hint would make this
   crisp.
4. **Schedules.** egglog's `run N`, `saturate`, `seq`, `repeat` must map
   onto `step()` sequencing / nested scopes; bounded `run N` (not to
   fixpoint) needs a non-recursive or counted scope.
5. **Push/pop & `clone_boxed`.** Needs snapshot-and-replay; the DuckDB plan
   lists this as an open gap, so no reference impl to copy.
6. **DBSP API maturity / version churn** and build-time/disk cost of pulling
   in `dbsp` (note: this phase avoided heavy builds; a real spike must
   budget for the dependency).
7. **Determinism / row identity for proofs.** `add_term` hash-consing must
   yield stable ids so proof terms reference the right rows across circuit
   rebuilds.

---

## 10. Honest recommendation

Feldera/DBSP is a **genuinely interesting and partly better-fitting** target
than DuckDB *for the monotone Datalog core*: DBSP's incremental recursion
*is* seminaive, so a large chunk of the DuckDB backend's hand-rolled
machinery (timestamp columns, focused seminaive variants, string interning,
UDF ABI) simply disappears. Base values and primitives are *simpler* on
DBSP. Proofs ride for free on the frontend term encoding, exactly as on
DuckDB.

The **entire bet** rides on the union-find/rebuild story (§2.3, §9.1):
expressing egglog's delete-and-rewrite rebuild as DBSP non-monotone
recursive views. That is simultaneously the **biggest risk** and the
**most interesting research result** — it is precisely the question "can a
monotone-incremental engine host an equality-saturation rebuild?" If yes,
this is a strong paper; if no, the backend degrades to a native-UF design
that is strictly more complex than DuckDB's for no win.

Recommendation: **pursue it, via the `dbsp` crate**, front-load the Phase 0
spike and the Phase 3 rebuild experiment as the go/no-go gate, and keep the
trait unchanged for the first milestone (adapter `run_rules`→`step`), adding
only the optional `runs_ruleset_to_fixpoint()` hint if the redundant
outer-loop step proves wasteful.
