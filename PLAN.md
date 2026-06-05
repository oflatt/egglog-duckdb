# Plan: A Flowlog-based egglog backend (`egglog-bridge-flowlog`)

Status: **EXPLORE + PLAN only.** No backend code is written yet, and no heavy
builds were run. This document mirrors the structure and scope discipline of
`duckdb-backend-plan.md`, but targets **FlowLog** instead of DuckDB.

---

## 0. What is FlowLog? (Step 1 — target identification)

**Confidence: High.** "Flowlog" here is **FlowLog**, the Datalog engine
"Efficient and Extensible Datalog via Incrementality" (Hangdong Zhao et al.,
VLDB 2026, arXiv `2511.00865`).

- Repo / org: <https://github.com/flowlog-rs/flowlog> (org
  <https://github.com/flowlog-rs>), site <https://www.flowlog-rs.com/>,
  paper <https://arxiv.org/abs/2511.00865> /
  <https://www.vldb.org/pvldb/vol19/p361-zhao.pdf>, VLDB-26 artifact
  <https://github.com/flowlog-rs/FlowLog-VLDB>.

**Ambiguity note (flagged for the manager):** there is an *older, unrelated*
"Flowlog" — a 2014-era SDN / network-controller Datalog-ish language (Nelson et
al., Brown). That one is a network-programming DSL with no Rust embedding, no
recursion-aware Datalog optimizer, and no relevance to an egraph backend. Given
the project context (Rust, relational/incremental backends, comparison against
the DuckDB backend), the **flowlog-rs / VLDB-26 system is the clearly intended
target.** The rest of this plan assumes it; the SDN Flowlog is dismissed.

### FlowLog's model, in the terms this project cares about

- **Programming/data model:** classic Datalog (EDB input relations → IDB derived
  relations), written in **Soufflé-compatible `.dl` syntax**. `.decl` declares
  typed relations (`int32`, `string`/`symbol`, `f64`); `.input`/`.output`
  bind relations to CSV files; rules are `Head(x) :- Body1(...), Body2(...).`.
- **Features:** stratified negation (negated atom must be EDB or a lower
  stratum), arithmetic/comparison constraints, (possibly recursive)
  aggregations (`min`, `max`, `sum`, `count`, `average`), and **user-defined
  operators/functors** (the "extensible" in the title — custom relational
  operators implemented in Rust). No e-graphs, no union-find, no built-in
  equality/congruence (we supply those via the term encoding, exactly as DuckDB
  does).
- **How a host embeds it:** FlowLog is a **Datalog→Timely/Differential-Dataflow
  compiler + runtime**, *not* an embeddable mutable database handle like
  `duckdb-rs`. Three workspace crates:
  - `flowlog-compiler` — CLI (`flowlog-compiler PROG.dl -o exe ...`) that emits
    a **standalone Rust executable**.
  - `flowlog-build` — library form callable from a `build.rs` to compile a
    `.dl` program into Rust at build time.
  - `flowlog-runtime` — linked into the generated binary (string interning, CSV
    IO, incremental-txn state, worker management).
  - Pipeline: parse → typecheck → stratify → planner (per-rule relational IR) →
    codegen → DD operators executed **stratum by stratum to fixpoint**.
- **Evaluation strategy:** **semi-naïve** at the logical level, realized on
  **Differential Dataflow**. DD tuples are `(data, time, diff)`:
  - *batch mode* (`datalog-batch`) uses a presence-only diff (`Present`,
    zero-bit) — static Datalog, fastest;
  - *incremental mode* (`datalog-inc` / `extend-inc`) uses `isize` diffs, so
    inserts (+1) **and retractions (−1)** are first-class and the fixpoint is
    maintained incrementally as inputs change.
- **Stated limits:** ~2–3× Soufflé's memory; weaker on batch workloads with few
  expensive iterations; structural (not cost-based) join planner; distributed
  execution is future work.

### The one fact that shapes the whole design

FlowLog's natural unit of work is *"compile a fixed program, then stream facts
through it and read the fixpoint"* — the **program is fixed at build time**,
data flows at runtime. egglog instead wants *"hold mutable egraph state, run
ruleset A to saturation, inspect (check/extract), then run ruleset B, then add
more facts"*. Reconciling FlowLog's fixed-program/streaming-data model with
egglog's interactive, schedule-driven, inspect-between-steps model is the
central engineering problem (see §5, §6).

---

## 1. Why FlowLog is a plausible backend (and the caveats)

The same linchpin that makes DuckDB plausible makes FlowLog plausible — the
**term encoding** (`src/proofs/proof_encoding.{rs,md}`). After term encoding:

- every `union` is gone; equality lives in explicit `UF_<Sort>` /
  `UF_<Sort>f` tables maintained by ordinary rules;
- every constructor is an ordinary relation (term table + view table +
  deferred delete/subsume helpers);
- rebuild = congruence + view canonicalization is expressed *as egglog rules*,
  not backend magic;
- the only merge modes that survive are `:merge old` (≈ "keep first / dedup")
  and `:merge new` (≈ "overwrite", only on `UF_<Sort>f`).

What's left is **plain stratified Datalog with two trivial upsert modes,
primitives, and a schedule** — which is *exactly FlowLog's wheelhouse*, and a
much closer fit than DuckDB on two axes:

1. **Seminaive is native.** DuckDB "has nothing seminaive-shaped"
   (`duckdb-backend-plan.md` §2.6) and the DuckDB backend re-implements the
   N-variant delta expansion in SQL by hand. FlowLog does seminaive *for us*
   inside DD — we hand it rules, it maintains the fixpoint incrementally. The
   single most fiddly part of the DuckDB backend essentially disappears.
2. **Recursion/fixpoint is native.** DuckDB can't even express mutual recursion
   across tables in one statement; the DuckDB backend drives iteration from
   Rust. FlowLog's executor runs strata to fixpoint as its core loop.

**Caveats (why it isn't free):**

- **Program is fixed at compile time.** FlowLog wants the whole `.dl` up front
  and then streams facts. egglog adds rules and functions interactively and
  inspects state between schedule steps. We must either (a) regenerate/recompile
  the dataflow per "epoch", or (b) target the long-lived **incremental
  (`*-inc`) runtime** and feed facts as DD input batches, reading outputs
  between `worker.step()`s. (b) is the real target; (a) is the fallback.
- **No mutable random-access table API.** Unlike DuckDB's `SELECT … WHERE key=?`
  / `INSERT … RETURNING`, FlowLog exposes relations as DD collections, not
  point-queryable tables. `lookup_id`, `add_term` (lookup-or-allocate fresh id),
  `for_each`, and `table_size` need a side structure we maintain in Rust
  alongside the dataflow (see §3, §4).
- **`add_term`'s fresh-id allocation is imperative.** egglog constructors
  allocate a fresh eq-sort id on first insert (DuckDB uses a sequence +
  `INSERT … RETURNING`, the materialized path in `compile.rs`). Pure Datalog has
  no "allocate a fresh skolem id" primitive; we provide it as a FlowLog functor
  or as a Rust-side hash-cons keyed by `(func, args)` (see §3.4).
- **Output type story.** FlowLog types are `int32`/`string`/`f64`. egglog packs
  *everything* into a 32-bit `Value`. We follow the DuckDB precedent: store
  every `Value` as an integer column (FlowLog `int32` / interned), and keep the
  typed meaning in the Rust-side `BaseValuePool`. (Caution: egglog `Value` is
  `u32`; FlowLog's documented integer type is `int32` — verify FlowLog has an
  unsigned/64-bit integer type or that the top-bit-set intern fallback values
  fit; this is a concrete unknown, §6.)

---

## 2. How egglog maps onto FlowLog

### 2.1 Where we cut into the egglog pipeline

Identical to DuckDB (`duckdb-backend-plan.md` §2.5): cut at the
**`Backend` / `RuleBuilderOps` trait** (`egglog-backend-trait/src/lib.rs`).
Term encoding has already run, so the backend sees UF tables, view tables, and
rebuild rules as ordinary rules. We build a new crate
`egglog-bridge-flowlog` that implements `Backend`, mirroring
`egglog-bridge-duckdb`. The frontend `EGraph` holds a `Box<dyn Backend>` and is
unchanged.

### 2.2 Schema / type mapping

| egglog (post-term-encoding) | FlowLog |
| --- | --- |
| eq-sort `S` | integer id column (`int32`/interned handle); rows in a per-function relation |
| base value (i64/f64/String/bool/Unit) | integer column holding the `Value` handle; typed meaning kept in Rust `BaseValuePool` |
| `Proof` sort | integer id handle into proof relations (same as any eq-sort) |
| container sorts (Vec/Set/Map) | **out of scope v1** (gated out, as on DuckDB) |
| function `f(a,b)->c` | FlowLog relation `f(a,b,c)` (+ the seminaive/`ts` machinery handled by DD, *not* a manual column) |
| `:merge old` | dedup / "first wins" — a distinct relation; on conflict keep existing |
| `:merge new` (only `UF_<Sort>f`) | "last wins" overwrite — modeled via retraction (`-1` old, `+1` new) in incremental mode |
| rule body (conjunctive query) | a FlowLog rule body (join + constraints) |
| rule action insert/set | derive into the head relation |
| `delete` / `subsume` | retraction in incremental mode (`diff = -1`); subsume = a flag relation |
| schedule (run/saturate/seq) | one stratum-fixpoint per ruleset, driven from Rust between `worker.step()`s |
| primitives (`+`, `<`, …) | FlowLog built-in constraints/functors where they line up; else Rust functors |
| seminaive timestamp filter | **native to DD — we do nothing** (the big win over DuckDB) |

### 2.3 Rule compilation

`RuleBuilderOps` accumulates body atoms and actions (exactly as the DuckDB
`DuckRuleBuilderOps` accumulates a `crate::Rule` IR). On `build()` we translate
the accumulated rule into **FlowLog `.dl` rule text** (or, better, directly into
FlowLog's per-rule relational IR if `flowlog-build` exposes it as a library
API — to be confirmed, §6). Each egglog body atom →

- table atom → FlowLog body literal over that relation;
- primitive atom → FlowLog constraint or functor call;
- RHS `set`/insert → FlowLog rule head;
- RHS `union` → after term encoding this is already a `set` into `UF_<Sort>`,
  so no special handling;
- RHS `delete`/`subsume` → retraction / flag relation (incremental mode).

Because FlowLog does seminaive itself, we **do not** emit N delta variants the
way the DuckDB backend does — we emit one rule and let DD's semi-naïve
operators handle deltas. This deletes the most error-prone slice of the DuckDB
compiler.

### 2.4 Fresh-id allocation / hash-consing (`add_term`)

The hard primitive. egglog constructors do "lookup-or-allocate-fresh-id keyed by
`(func, args)`". Options:

1. **Rust-side hash-cons** (recommended v1): the backend keeps a
   `HashMap<(FunctionId, Vec<Value>), Value>` and an id counter, exactly like a
   union-find seq. `add_term` / `lookup_constructor_rows` consult it, allocate on
   miss, and feed the resulting full row into FlowLog as an input fact. This
   keeps id allocation deterministic and outside the dataflow — matching how
   DuckDB uses a sequence + materialized path, and how the native-UF path keeps
   the UF in Rust (`uf.rs`).
2. **FlowLog functor** that allocates ids inside a rule. Cleaner in principle but
   needs a stateful functor with deterministic, replay-stable ids; risky under
   DD's out-of-order/incremental execution. Defer.

### 2.5 Union-find / rebuild / congruence

**All in egglog rules already** (term encoding). The backend does **not**
implement congruence — it just runs the `parent`, `single_parent`,
`uf_function_index`, `rebuilding`, `rebuilding_cleanup`,
`delete_subsume_ruleset` rulesets that term encoding emits, in the scheduled
order. This is the whole point of cutting below term encoding: the backend is a
plain Datalog evaluator.

Two backend-visible consequences:

- These rules use `delete` + `set` (e.g. `uf_update` deletes `(UF a b)` and sets
  `(UF a c)`). On FlowLog this is a **retraction + assertion**, i.e. requires
  **incremental mode** (`isize` diffs). Batch/`Present` mode cannot retract, so
  it cannot run the term encoding's rebuild rules. **→ The FlowLog backend must
  target incremental mode.** (This is a hard requirement, not a preference.)
- `:merge old` vs `:merge new` map to "first wins" vs "last wins"; "last wins"
  (`UF_<Sort>f`) is implemented by retracting the old output tuple and asserting
  the new one when a key already has a different value.

### 2.6 Primitives / base values

Same split as DuckDB:

- Primitives that map to FlowLog built-ins (arithmetic, comparison, string ops)
  → emit the built-in constraint/functor.
- Everything else (egglog's `ExternalFunction`s: `from-string`, `bigrat`,
  user UDFs) → **FlowLog Rust functors** backed by a shared `BaseValuePool`,
  exactly mirroring the DuckDB `VScalar` UDFs in `compile.rs`
  (`FromStringScalar`, `BigratScalar`, …). The pool is `BaseValues`-backed and
  `Clone`-shares its intern tables so a value interned inside a functor is
  visible to the egraph (`base_values.rs` precedent).
- `BaseValuePool` impl: copy `DuckdbBaseValuePool` almost verbatim — it is a thin
  wrapper over `egglog_core_relations::BaseValues` and is backend-agnostic.

### 2.7 Term / proof encoding (the paper goal)

**Proofs need *no new backend mechanism* beyond what running ordinary rules
already provides** — this is the elegant part, and it is the same on FlowLog as
on DuckDB:

- Proof tracking is turned on *in the frontend* via term encoding
  (`EGraph::new_with_term_encoding`). With proofs enabled, the encoding changes
  the *output column type* of `UF_<Sort>`, the view tables, and per-sort
  `…Proof` tables from `Unit` to `Proof`, and the emitted rules build proof
  terms (`Rule`, `Trans`, `Sym`, `Fiat`, `PCons`/`PNil`) as ordinary
  constructor applications stored in those columns
  (`proof_encoding.md`, `proof_encoding_helpers.rs`).
- To the backend a `Proof` is **just another eq-sort id**: a proof term is built
  by the same `add_term` hash-cons path as any other constructor, and stored in
  an integer column. The backend needs **zero proof-specific code**; it only
  needs to faithfully run the (proof-instrumented) rules and `set`s the encoding
  emits.
- Therefore the FlowLog backend gets proofs "for free" the moment it can:
  (a) allocate fresh ids for constructor/proof terms (`add_term`, §2.4),
  (b) run rules with `set`/`delete`/`merge old`/`merge new` correctly
  (incremental mode, §2.5), and
  (c) read back a column value (`lookup_id` / `for_each`) so the frontend's
  proof extraction (`proof_extraction.rs`, `proof_checker.rs`) can walk the
  proof DAG.
- **Net:** proof support reduces to "correctly evaluate the encoded program and
  expose `lookup`/`for_each`". No congruence, no proof combinators in the
  backend. FlowLog's incremental retraction model is actually a *better* fit
  than DuckDB here because the rebuild rules (which delete+set) are exactly the
  retraction pattern DD is built for.

**One FlowLog-specific risk for proofs:** the rebuild rules rely on
`ordering-min`/`ordering-max` (an insertion-order total order on terms) to pick
deterministic parents. That ordering is currently realized via egglog id
ordering. We must ensure FlowLog preserves a stable, deterministic id order
under incremental/parallel DD execution, or proofs (and even confluence) could
vary run-to-run. Keeping id allocation Rust-side (§2.4 option 1) is what makes
this deterministic.

---

## 3. Proposed crate structure (`egglog-bridge-flowlog`)

Mirror `egglog-bridge-duckdb/src/` one-for-one:

```
egglog-bridge-flowlog/
  Cargo.toml                # depends on flowlog-build/-runtime (or vendored), egglog-backend-trait,
                            # egglog-core-relations, egglog-numeric-id
  src/
    lib.rs                  # the `EGraph` struct: holds the FlowLog dataflow handle / input sessions,
                            #   the Rust-side hash-cons + id counter, function/relation registry,
                            #   the BaseValuePool, rule registry. Mirrors duckdb lib.rs.
    backend_impl.rs         # `impl Backend for EGraph` (mirrors duckdb backend_impl.rs)
    compile.rs              # egglog rule IR -> FlowLog .dl text or FlowLog relational IR
    rule_builder.rs         # `impl RuleBuilderOps` accumulator (mirrors DuckRuleBuilderOps)
    uf.rs                   # OPTIONAL: only if we keep a Rust-side UF for a native-UF fast path;
                            #   in the pure-encoding path the UF lives in FlowLog rules and this is unused
    base_values.rs          # `FlowlogBaseValuePool` — near-verbatim copy of DuckdbBaseValuePool
    external_func.rs        # FlowLog Rust functors backed by the shared pool (mirrors duckdb UDFs)
  examples/
```

### 3.1 `Backend` methods: needed vs stubbed (v1)

Following the DuckDB precedent (`backend_impl.rs` header lists exactly what is
real vs `unimplemented!`):

**Must implement for a trivial program to run:**
- `add_table` (register relation + its merge mode)
- `new_rule` / `RuleBuilderOps::*` / `build` / `run_rules` / `free_rule`
- `flush_updates` (push staged input facts into DD, step the worker)
- `insert_rows` / `lookup_constructor_rows` / `add_term` / `add_values`
  (feed EDB facts; `add_term` does the hash-cons)
- `lookup_id` / `for_each` / `for_each_while` / `table_size`
  (read fixpoint state back out — needs the Rust-side materialized mirror, §4)
- `get_canon_repr` (UF canonicalization — read from the maintained `UF_<Sort>f`
  relation mirror)
- `fresh_id` (id counter)
- `base_values` / `base_value_pool` / `base_value_pool_mut` /
  `base_value_constant_dyn`
- `register_external_func` / `free_external_func` / `new_panic` (functors)
- `clear_table`
- capability flags, `set_report_level`, `dump_debug_info`, `as_any[_mut]`

**Stub / error in v1 (same gates as DuckDB):**
- `supports_inline_table_lookups` → `false` (functors can't reenter the egraph
  during a DD operator)
- `supports_subsumption` → start `false`; can be `true` once subsume-flag
  relations are wired (term encoding needs subsume, so this may need to be true
  earlier than on DuckDB — see §6)
- `supports_complex_merge` → `false` (term encoding compiles complex merges away)
- `supports_containers` → `false`
- `container_pool[_mut]` → zero-sized stub (copy `DuckdbContainerPool`)
- `with_execution_state_dyn` / `action_registry_any` → bridge-only;
  `unimplemented!`/panic as DuckDB does
- `clone_boxed` → hard (a running DD dataflow isn't trivially cloneable). v1:
  rebuild from a replay log of input facts + registered rules (the design note
  in the trait already anticipates a "replay buffer" for DuckDB). Needed for
  push/pop; can be deferred behind a "push/pop unsupported" error initially.

### 3.2 `RuleBuilderOps` methods

Implement `new_var[_named]`, `query_table`, `query_prim`,
`call_external_func`, `lookup`, `set`, `remove`, `union`, `panic`, `build`,
`build_check`. `subsume` and `is_subsumed` per §6. `rename_prim` for overloaded
primitives (`^`, `+`) as on DuckDB.

---

## 4. The state-mirroring problem (FlowLog-specific design)

FlowLog relations are DD collections, not point-queryable tables. But `Backend`
demands `lookup_id`, `for_each`, `table_size`, `get_canon_repr`. Two designs:

- **(A) Maintain a Rust-side materialized mirror.** Mark every relation
  `.output` (or attach an inspection probe / `Trace`), and after each
  `worker.step()` to fixpoint, drain the relation's current contents into a
  Rust `HashMap`/index keyed for lookup. `lookup_id` / `for_each` /
  `get_canon_repr` read the mirror. Cost: memory (a second copy) + drain time
  per epoch. This is the most robust v1 approach and is analogous to how the
  DuckDB backend just queries its tables.
- **(B) Query DD `Trace` handles directly.** DD/differential keeps arrangements
  (indexed traces) we could probe with `cursor`s for point lookups. Lower memory,
  but couples us tightly to DD internals and FlowLog's arrangement choices.
  Defer to a perf pass.

Recommend **(A)** for the first milestone.

A related decision: **epoch model.** Map one egglog `run_rules`/schedule step to
one DD logical time. Stage all `insert_rows`/`add_term` facts for the step,
advance the input session to the next time, `worker.step()` until the frontier
passes, then refresh the mirror. This gives egglog its "run to local fixpoint,
then inspect" semantics on top of DD's incremental engine.

---

## 5. Phased implementation plan

**Milestone 0 — spike (no `Backend` impl).** Confirm the *embedding* story is
viable at all: drive `flowlog-build`/`flowlog-runtime` from Rust to (a) build a
hand-written `.dl` (e.g. transitive closure), (b) feed input facts at runtime in
incremental mode, (c) step the worker, (d) read the output relation back into
Rust between steps. This validates the §4 mirror loop and the §0 "fixed program
vs interactive" tension *before* committing to the trait. **Gate the whole plan
on this.**

**Milestone 1 — trivial program end-to-end.** Implement `add_table`,
`RuleBuilderOps`→`.dl`, `run_rules`, `flush_updates`, `insert_rows`,
`for_each`/`table_size`, `lookup_id`, base-value pool, the container/exec stubs,
and capability flags. Target: a relations-only egglog program with `:merge old`
and a non-recursive rule runs and `check`s pass. (Mirrors DuckDB Phase 1.1.)

**Milestone 2 — eq-sorts + hash-cons + UF.** Add `add_term`/`fresh_id`/
`lookup_constructor_rows`, `get_canon_repr`, `:merge new`, `delete`/retraction.
Run a term-encoded program with `(union …)` and rebuilding to fixpoint; verify
congruence via `check (= …)`. This is the real "egraph on FlowLog" milestone.

**Milestone 3 — primitives.** Wire FlowLog built-ins + Rust functors and the
shared `BaseValuePool`; get arithmetic/string programs (and `from-string` /
`bigrat`) working.

**Milestone 4 — proofs.** Turn on term encoding with proofs; ensure proof
columns round-trip and `proof_extraction.rs` / `proof_checker.rs` validate
proofs produced by the FlowLog backend. Most of this should "just work" if
M2/M3 are correct (§2.7); budget for ordering-determinism debugging.

**Milestone 5 — robustness/perf.** push/pop via replay-log `clone_boxed`,
subsumption, the `Trace`-cursor lookup path (§4 B), benchmarking vs
bridge/DuckDB via `script/bench.py`.

---

## 6. Risks / unknowns / where the models fight

1. **Embedding API maturity (highest risk).** It is unconfirmed whether
   `flowlog-build`/`flowlog-runtime` expose a *stable, programmatic* way to:
   register a dynamically-built program, feed facts at runtime, step, and read
   relations back — versus only the "compile a `.dl` to a standalone exe" path.
   If only the latter exists, the backend must **regenerate + recompile** the DD
   program whenever rules/functions change (slow; an out-of-process exe per
   epoch). Milestone 0 exists to settle this.
2. **Interactive schedule vs fixed dataflow.** egglog interleaves rule running
   with `check`/`extract`/new facts. DD wants a long-lived dataflow with
   streamed input. The epoch model (§4) bridges this *if* incremental input
   sessions are exposed; otherwise we recompile (risk 1).
3. **Retraction is mandatory.** The term encoding's rebuild rules `delete`+`set`,
   so we **must** use incremental (`isize`) mode, not batch `Present` mode. This
   forecloses FlowLog's fastest mode — a perf caveat to measure.
4. **Custom/stateful functors under DD.** Functors run inside DD operators and
   may be invoked out of order / multiple times / on retraction. egglog
   `ExternalFunction`s must be **pure and deterministic** (they are, mostly).
   `supports_inline_table_lookups=false` (no reentry), same as DuckDB.
5. **Fresh-id determinism.** Id allocation must be stable for confluent results
   and valid proofs. Keep it Rust-side (§2.4 option 1); never let DD's
   parallel/incremental ordering pick ids.
6. **Integer width.** egglog `Value` is `u32`; FlowLog's documented int type is
   `int32`. Need an unsigned/64-bit integer column or a guarantee that handles
   fit. Concrete thing to verify against FlowLog's type system in Milestone 0.
7. **Subsumption.** Term encoding *uses* `subsume` on view tables
   (`delete_subsume_ruleset`). Unlike DuckDB (which gated subsume off), a fully
   term-encoded FlowLog backend likely needs subsume working in M2 — model it as
   a boolean flag relation joined into queries (`is_subsumed`), or as retraction
   of the subsumed view row.
8. **State mirror cost.** Design (A) double-stores every relation in Rust.
   Acceptable for correctness-first milestones; revisit with (B) for perf.
9. **`clone_boxed` / push-pop.** No cheap dataflow snapshot. Replay-log rebuild
   is the plan; expensive. May ship "push/pop unsupported" first.

---

## 7. Interface gaps in `egglog-backend-trait` (feeds the interface assessment)

The trait was shaped around two backends (in-memory bridge, DuckDB). FlowLog
exposes a few rough edges:

- **Batch-of-facts + explicit epoch boundary.** `flush_updates() -> bool` is the
  only "advance the world" hook and it conflates "merge staged inserts" with
  "run rebuild". For a streaming DD backend it would be cleaner to have an
  explicit `begin_epoch` / `end_epoch` (or `advance_time`) so the backend knows
  exactly when to seal an input batch and step the worker. Today FlowLog must
  infer epoch boundaries from the `flush_updates`/`run_rules` call pattern.
- **`run_rules(&[RuleId])` assumes the backend re-runs named rules each call.**
  FlowLog would rather install all rules once into a persistent dataflow and just
  push data. The trait's per-call rule list fits a re-evaluating engine, not a
  standing dataflow. A capability flag like `prefers_standing_dataflow()` plus an
  `install_rules`/`step` split would let FlowLog avoid re-submitting rule plans.
- **Retraction isn't first-class.** `remove` exists on `RuleBuilderOps`, but
  there's no top-level "retract these input facts" on `Backend` analogous to
  `insert_rows`. A `retract_rows` would map directly to DD `-1` diffs and make
  the incremental story explicit instead of emulated via rule `delete`s.
- **Point-lookup vs scan.** `lookup_id` + `for_each` assume cheap random access.
  A scan/streaming-friendly backend benefits from a `subscribe`/probe hook
  (read changes since last epoch) rather than full `for_each` every inspection.
  Optional; design (A) makes the current API workable.
- **`clone_boxed` assumes snapshotability.** For dataflow backends a
  `try_clone() -> Option<...>` or an explicit "checkpoint token" would be more
  honest than a mandatory deep clone the backend can't cheaply provide.

None of these block a first implementation (DuckDB lived within the same trait),
but they're where FlowLog's grain runs against the current surface.

---

## Executive summary

- **Target:** **FlowLog** (`flowlog-rs/flowlog`, VLDB 2026, arXiv 2511.00865) —
  a Soufflé-syntax Datalog engine that compiles to **Differential Dataflow**,
  with native semi-naïve + incremental (insert/retract) evaluation. *High
  confidence*; the only other "Flowlog" (a 2014 SDN DSL) is unrelated and
  dismissed.
- **Top opportunity:** FlowLog gives us **seminaive and fixpoint for free** —
  the two hardest, hand-rolled parts of the DuckDB backend — because the
  term-encoded egglog program is *already* plain stratified Datalog with two
  upsert modes. Incremental retraction (`isize` diffs) is a natural fit for the
  rebuild rules' `delete`+`set`. **Proofs need no backend-specific code**: a
  `Proof` is just another eq-sort id, so proofs fall out of correctly running
  the (proof-instrumented) encoded rules plus `add_term` + `lookup`.
- **Top risks:** (1) FlowLog's embedding model is *fixed-program / streamed-data*
  and possibly only "compile a `.dl` to an exe" — reconciling that with egglog's
  interactive, inspect-between-steps schedule is the crux and is unproven; (2)
  no point-queryable mutable table API, so we must maintain a Rust-side
  materialized mirror and a Rust-side hash-cons/id counter for `add_term` /
  `lookup_id` / `get_canon_repr`; (3) retraction is mandatory (rebuild rules
  delete), forcing incremental mode and forgoing FlowLog's fastest batch mode.
- **Recommended first milestone:** **Milestone 0 — an embedding spike** that
  drives `flowlog-build`/`flowlog-runtime` from Rust to feed facts at runtime
  into a hand-written incremental `.dl`, step the worker, and read the relation
  back between steps. Everything else is gated on this proving that FlowLog can
  be driven as a *live, incrementally-fed* engine rather than a one-shot
  compiler. If it can, the trait-level backend (Milestones 1–4) closely follows
  the DuckDB template, minus the seminaive machinery.
