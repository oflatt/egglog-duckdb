# Backend Interface Assessment

**Purpose**: gate the decision to build two more backends (Flowlog: relational/Datalog;
Feldera: incremental DBSP) on the existing `Backend` trait.
**Branch**: `egglog-encoding-main`. **Mode**: read-only review, no code changes.
**Inputs**: `egglog-backend-trait/src/lib.rs` (the trait), `egglog-bridge/src/backend_impl.rs`
(reference passthrough), `egglog-bridge-duckdb/src/*` (only non-reference impl),
`src/lib.rs` + `src/proofs/*` (frontend driver), and the prior
`docs/backend_trait_inventory.md`.

---

## TL;DR recommendation

**The interface is good enough to *prototype* a Flowlog backend on as-is, but it is NOT
yet a neutral interface, and I would not commit to two backends without first making 2–3
targeted changes.** The trait is honest about being a one-for-one mirror of
`egglog_bridge::RuleBuilder`, and that shape transfers cleanly to another iterate-to-fixpoint
relational engine (Flowlog). It does *not* transfer cleanly to Feldera, whose evaluation
model is natively incremental — the `run_rules(&[RuleId]) -> IterationReport` /
`flush_updates() -> bool` "drive one iteration, ask did-anything-change" contract is the
reference backend's fixpoint loop leaking through the seam.

The single most important finding is about **proofs**: proof/term encoding emits a *textual
egglog program* that flows through the normal `parse → typecheck → add_table / new_rule`
path. A faithful `Backend` impl (like the bridge) runs it with zero proof-specific hooks —
**that is the good news, and it is genuinely encouraging for the paper.** But the duckdb
backend does *not* run it faithfully: it re-recognizes proof-encoding constructs
(`ordering-min`/`ordering-max` merges, `pname` UF tables, Unit-output relations,
native-UF) by **string-matching primitive names inside `add_table`**. That special-casing
is invisible to the trait and will have to be re-implemented from scratch in every new
backend. The trait does not currently *carry* the semantic information (this column is a
union-find leader; this merge is a min-join; this relation is a proof table) that a backend
needs — it forces each backend to reverse-engineer it from names.

### Top issues (ranked)

1. **Proof/term-encoding semantics travel as string-matched names, not typed trait data.**
   `egglog-bridge-duckdb/src/backend_impl.rs:577-580` keys merge behavior off the literal
   string `"ordering-min"`; `:684-601` detects Unit-output proof relations by comparing
   `BaseValueId`s. A new backend must duplicate all of this. *This is the #1 risk to the
   paper's "encode proofs across backends" goal.*

2. **`run_rules` / `flush_updates` / `IterationReport` bake in the bridge's
   iterate-to-fixpoint model.** Fine for Flowlog, actively wrong for Feldera (incremental).
   `egglog-backend-trait/src/lib.rs:419-425`.

3. **Capability flags are a leak, not an abstraction.** Four `supports_*` booleans
   (`lib.rs:505-527`) plus an `as_any()` downcast escape hatch mean the frontend *already*
   branches on backend identity in ~60 sites (`grep` count across `src/`). The trait is
   "neutral" only because the frontend compensates with `downcast_ref::<bridge::EGraph>()`
   and `is::<duckdb::EGraph>()`.

4. **The `deferred_err` pattern** (duckdb `rule_builder.rs:159` + 7 sites) is forced by
   infallible trait methods (`new_var`, `set`, `union`, …). Errors get smuggled to
   `build()`. This is a direct symptom of mirroring an *infallible* in-memory builder onto
   backends that can fail mid-build.

5. **The trait split forces real unsafe in both the frontend and the reference backend.**
   Frontend: 6 `unsafe { &*backend_ptr }` raw-pointer reborrows (`src/lib.rs:1228, 1273,
   1428, 1507, 1541, 2058`) to work around `new_rule(&mut self)` + pool-read aliasing.
   Bridge: `*const BaseValues as *const BaseValuesAsPool` repr-transparent transmute
   (`backend_impl.rs:413-436`) to satisfy the `base_values()` (concrete) vs
   `base_value_pool()` (dyn) split.

Concrete effort to de-risk before building both backends: ~1 week (see §6).

---

## (a) Trait surface map

`Backend` (object-safe, `Send + Sync`, `clone_boxed`). Companion dyn traits:
`RuleBuilderOps`, `BaseValuePool`, `ContainerPool`. Reference impl
(`egglog-bridge/src/backend_impl.rs`) is a near-perfect one-liner-per-method passthrough —
strong evidence the surface matches *the bridge* exactly.

### `Backend` methods

| Method | lib.rs | Maps to bridge | Clean? | Notes for new backends |
|---|---|---|---|---|
| `add_table(FunctionConfig)->FunctionId` | 262 | `add_table` | **Leaky** | Carries `merge: MergeFn`, `default: DefaultVal`. DuckDB reverse-engineers proof semantics here. |
| `table_size` / `approx_table_size` | 267/272 | yes | clean | |
| `for_each` / `for_each_while` | 284/302 | yes | clean | HRTB `for<'r> FnMut(FunctionRow)`; default `for_each` is `unimplemented!()` (a wart, see §b). |
| `lookup_id` | 309 | yes | clean | |
| `add_values(Box<dyn Iterator>)` | 315 | yes | ok | boxed iterator only to stay dyn-safe. |
| `add_term` | 322 | yes | clean | "stage inputs, return fresh canonical id". |
| `insert_rows` / `lookup_constructor_rows` | 338/350 | `with_execution_state`+TableAction loop | ok | These *replaced* the old `with_execution_state` callers — a genuine improvement. |
| `get_canon_repr(val, ty)` | 358 | yes | clean | the right UF abstraction; `ty` disambiguates. |
| `fresh_id` | 363 | yes | clean | |
| `clear_table` | 368 | yes | clean | |
| `base_values()->&BaseValues` | 376 | yes | **Leaky** | returns the *concrete* core-relations type; coexists with dyn `base_value_pool()`. Forces a transmute in the bridge. |
| `with_execution_state_dyn` | 384 | yes | **Leaky** | duckdb `unimplemented!()` (`backend_impl.rs:933`). Bridge-only. |
| `action_registry_any()->&dyn Any` | 398 | yes | **Hack** | type-erased to dodge a dep cycle; duckdb `unimplemented!()` (`:940`). |
| `new_rule(&mut self,..)->Box<dyn RuleBuilderOps>` | 407 | yes | **Awkward** | `&mut self` here is what forces the frontend's 6 unsafe reborrows. |
| `free_rule` / `run_rules` / `flush_updates` | 412/419/425 | yes | **Model-leak** | fixpoint contract; see §d Feldera. |
| `register_external_func` / `free_external_func` / `new_panic` | 436/444/451 | yes | partial | duckdb stores but does not wire to SQL (`backend_impl.rs:1034`); gated by `supports_inline_table_lookups`. |
| `base_value_pool[_mut]` | 460/466 | yes | clean (dyn) | |
| `container_pool[_mut]` | 476/481 | yes | stub on duckdb | empty stub; never reached because container programs are proof-gated out. |
| `base_value_constant_dyn` | 494 | yes | clean | |
| `supports_inline_table_lookups / _subsumption / _complex_merge / _containers` | 505-527 | n/a | **Leak** | capability flags (see §b). |
| `set_report_level` / `dump_debug_info` | 534/539 | yes | clean | |
| `clone_boxed` | 550 | yes | partial | duckdb `unimplemented!()` (`:1172`) — no push/pop on duckdb. |
| `as_any` / `as_any_mut` | 569/573 | self | **Escape hatch** | the de-facto neutrality breaker. |

### `RuleBuilderOps` (the "mirror")
`new_var`, `new_var_named`, `query_table`, `query_prim`, `call_external_func`, `lookup`,
`subsume`, `set`, `remove`, `union`, `panic`, `new_panic`, `set_no_decomp`, `build`,
`build_check`, `rename_prim` (`lib.rs:654-788`). Most are **infallible** (`-> ()` or
`-> QueryEntry`); only `query_table`/`query_prim`/`subsume`/`build`/`build_check` return
`Result`. This infallibility is exactly what forces duckdb's `deferred_err`.

### `BaseValuePool` / `ContainerPool`
Clean Any-based dyn dispatch with free-function generic sugar (`pool_get<T>`,
`pool_unwrap<T>`, `container_register_val<C>`). This is the *best-designed* part of the
interface and should be the template for the rest.

---

## (b) Concrete smells / leaks (file:line evidence)

### Capability flags — leakage by enumeration
`egglog-backend-trait/src/lib.rs:505-527` defines `supports_inline_table_lookups`,
`supports_subsumption`, `supports_complex_merge`, `supports_containers`. Each is a place
where the frontend must know what a backend can't do. They imply the trait is a *union* of
all backends' features, with runtime opt-outs, rather than a neutral contract. Adding
Feldera/Flowlog will add more flags (e.g. `supports_fixpoint`, `supports_seminaive_delta`).
Flags multiply combinatorially with backends — a classic interface smell.

### `as_any` downcast escape hatch — the real neutrality breaker
The trait documents `as_any` as the supported path to reach bridge-only state
(`lib.rs:221-255, 552-573`). In practice the frontend leans on it hard:
- `EGraph::bridge()` / `bridge_mut()` — `src/lib.rs:539-552` — `expect("this code path is
  bridge-only")`.
- ~60 downcast/`as_any`/`is::<duckdb>` sites across `src/` (by file:
  `lib.rs` 33, `prelude.rs` 9, `extract.rs`/`typechecking.rs` 4 each, `scheduler.rs`/
  `serialize.rs` 3 each, `sort/fn.rs` 1).
- `has_duckdb_backend()` branches on `self.backend.as_any().is::<duckdb::EGraph>()`
  (`src/lib.rs:1961`); several call sites then do duckdb-only work
  (`src/lib.rs:1395, 1495-1501, 1658`).

A "neutral" trait whose consumers downcast to concrete backends in 60 places is neutral
only on paper. Every such site is a place a third/fourth backend must be taught about.

### `deferred_err` — infallible-builder mismatch
`egglog-bridge-duckdb/src/rule_builder.rs:159` plus deferral at `:554, :572, :624, :664,
:683, :709`. Because `RuleBuilderOps::union/set/remove/...` are `-> ()` (mirroring the
bridge's infallible builder), duckdb cannot report "I can't do this" at the call; it stashes
the first error and surfaces it at `build()`. `union` is the worst case
(`rule_builder.rs:697-716`): duckdb literally cannot implement a generic `union(l,r)`
because it needs the per-sort `pname`, which the `FunctionId`/`Value` surface doesn't carry
— so it always defers an error and relies on the frontend never calling it (term encoding
emits `(set (pname ...) ())` instead). That's a semantic gap papered over by "the frontend
happens not to hit it."

### Eager/lossy choices (the `PanicMsg` symptom generalizes)
The recently-fixed `panic_msg: String` → `PanicMsg = Box<dyn FnOnce()->String>`
(`lib.rs:632, 693-711`) was one lossy choice. duckdb drops the closure unevaluated
(`rule_builder.rs:549` `_panic_msg`) — fine, but it shows the trait is sized to the bridge's
needs and other backends just ignore fields. Similar "accept for parity, ignore" cases:
`new_rule`'s `seminaive` flag is ignored by duckdb (`backend_impl.rs:948-949`);
`set_no_decomp` is a no-op default (`lib.rs:755`).

### `unimplemented!()` / stub / no-op methods on duckdb
- `with_execution_state_dyn` — `unimplemented!()` (`backend_impl.rs:933`).
- `action_registry_any` — `unimplemented!()` (`:940`).
- `clone_boxed` — `unimplemented!()` (`:1172`); push/pop unsupported.
- `flush_updates` — hardcoded `false` no-op (`:1018`).
- `subsume` — silent `Ok(())` no-op (`rule_builder.rs:646-657`).
- `for_each` trait default is `unimplemented!()` (`lib.rs:291`) — every backend MUST
  override; the "default" exists only to placate dyn-compat lints. A no-default required
  method would be more honest.

### Type-erasure hacks
- Bridge `*const BaseValues as *const BaseValuesAsPool` / `*mut … as *mut …` repr-transparent
  transmutes (`egglog-bridge/src/backend_impl.rs:413, 420, 429, 436`) — needed only because
  the trait exposes both a concrete `base_values()` and a dyn `base_value_pool()`.
- `action_registry_any() -> &dyn Any` + generic `action_registry<R>()` downcast
  (`lib.rs:398, 612-616`) — erases `Arc<RwLock<ActionRegistry>>` to dodge a dependency cycle.
- Frontend `unsafe { &*backend_ptr }` raw reborrows (`src/lib.rs:1228, 1273, 1428, 1507,
  1541, 2058`) — to read the pool while `new_rule` holds `&mut self`.

### `unsafe impl Send/Sync for duckdb::EGraph`
`egglog-bridge-duckdb/src/backend_impl.rs:1201-1202`. The trait requires `Send + Sync`
(matching the parallel bridge), but duckdb's `Connection` isn't naturally `Sync`, so it is
asserted unsafely. Feldera (DBSP runtime/threads) may have the same friction.

---

## (c) Proof-encoding readiness

**How it works** (good): proofs/term encoding is a *source-to-source* pass
(`src/proofs/proof_encoding.rs`) that emits ordinary egglog text — per-sort union-find
tables `(function {pname} (S S) ... :merge old :internal-hidden)`, transitivity/congruence
rules using `(ordering-min a b)`/`(ordering-max a b)`, `delete`/`set`, proof tables. This
generated program is then parsed, typechecked, and lowered through the **same**
`add_table` / `new_rule` path as user code (`src/lib.rs:2203-2223`). There is **no proof
hook on the `Backend` trait**: `proofs_enabled` lives entirely in the frontend
(`proof_encoding_helpers.rs:245`, `src/lib.rs:1949`), and `command_supports_proof_encoding`
(`proof_encoding_helpers.rs:521`) gates *which programs* get encoded, before the backend is
involved.

**Consequence for new backends (the encouraging part):** a backend that faithfully
implements `add_table` (honoring `MergeFn`), `new_rule`/`RuleBuilderOps`, `union`, `lookup`,
`set`, `remove`, and the primitives `ordering-min`/`ordering-max` will run proof-encoded
programs *with no proof-specific code*. The reference bridge does exactly this.

**The catch (the #1 paper risk):** the duckdb backend does **not** run the generated
program faithfully — it re-derives the encoding's intent from names/shapes:
- `add_table` keys union-find merge on the string `"ordering-min"`
  (`backend_impl.rs:566-581`): `Some("ordering-min") => MergeMode::Min`, else `Old`. If a
  backend doesn't recognize this it "silently drops UFf upserts" (their words) and proofs
  diverge.
- Unit-output proof relations (`@UF_...` tables) are detected by comparing the trailing
  column's `BaseValueId` to the `()` type id and stripped to a relation
  (`backend_impl.rs:590-616`).
- `union` is *not* implemented; the encoder is required to emit `(set (pname ...) ())`
  instead (`rule_builder.rs:697-716`).
- Native-UF (`--duck-native-uf`) reimplements congruence/canonicalization in-process
  (`backend_impl.rs:870-885`, `compile.rs:72-121, 310-378, 772-787`), keyed off pname tables.

So the trait exposes everything proofs need *only if the backend chooses to interpret the
generated egglog literally*. It does **not** carry the typed signal "this is a UF leader
column / this merge is a lattice-min / this is a proof table." Each backend that wants to
optimize (as duckdb did, and as Feldera will be forced to for a min-merge) must string-match
to recover that intent. **For the paper this is the thing to fix:** either commit to "every
backend runs the generated egglog literally and we measure that" (clean, but duckdb already
deviated for performance), or **promote the encoding's intent into typed `FunctionConfig`
fields** (e.g. a `FunctionRole::UnionFind { sort }` / `MergeFn::LatticeMin` variant) so
backends don't reverse-engineer strings.

---

## (d) Per-backend fit prediction

### Flowlog (relational / Datalog, SQL-compilation-ish — like duckdb)

| Trait area | Fit | Notes |
|---|---|---|
| `add_table` / schema / `FunctionConfig` | **Good** | relational tables map directly. |
| `RuleBuilderOps` query side | **Good** | `query_table`/`query_prim`/`Filter`-style fits Datalog bodies. duckdb's data-IR translation (`duck::Rule`) is a working template. |
| `run_rules` / `flush_updates` fixpoint | **Good** | Flowlog is iterate-to-fixpoint; the contract matches. |
| `union` / congruence / proofs | **Awkward (same as duckdb)** | will need the same `pname`/`ordering-min` recognition or a native UF. Directly inherits issue #1. |
| `subsume`, complex merge | **Gate off initially** | reuse the `supports_*` flags. |
| primitives w/ inline table lookups | **Likely false** | same as duckdb (`rust_rule`/`query` unsupported). |

**Prediction**: Flowlog is a ~duckdb-shaped effort. The trait fits; you'll re-walk the same
proof-encoding string-matching path. Most of the duckdb `compile.rs`/`rule_builder.rs`
structure transfers. **Low risk, but you will copy the proof-recognition smell verbatim
unless issue #1 is fixed first.**

### Feldera (incremental DBSP / Z-sets / native seminaive)

| Trait area | Fit | Notes |
|---|---|---|
| `run_rules(&[RuleId]) -> IterationReport` + `flush_updates() -> bool` | **Fights the model** | DBSP is a standing dataflow that consumes input deltas and emits output deltas continuously. "Run these rules once, tell me if anything changed" is the bridge's outer fixpoint loop. Feldera *is* the fixpoint; being driven one iteration at a time inverts control. You'd either (a) build the whole rule set into one DBSP circuit and emulate `run_rules` by stepping it, or (b) ignore the per-call rule subsetting. Either way `IterationReport.changed` becomes a fixed-point-detection shim. |
| seminaive `new_rule(.., seminaive)` flag | **Redundant/awkward** | DBSP is *natively* incremental; the bridge's seminaive flag is meaningless (duckdb already ignores it — `backend_impl.rs:948`). Feldera would too, but for a deeper reason: incrementality isn't a per-rule toggle there. |
| `RuleBuilderOps` callback shape | **Workable** | you can accumulate atoms into a DBSP circuit description at `build()`, like duckdb accumulates `duck::Rule`. The mirror shape isn't fatal here. |
| union-find / congruence / `union` | **Hard** | egglog's UF is mutate-in-place + rebuild; DBSP wants monotone-ish dataflow. Encoding UF as `pname` tables with `ordering-min` merge is *more* DBSP-friendly than mutable UF (it's a lattice min over Z-sets) — but only if the trait carries "this merge is a lattice min" (issue #1) instead of a string. |
| `add_term` / fresh-id allocation | **Awkward** | allocating fresh ids inside an incremental circuit (per new term) is stateful; DBSP prefers deterministic/content-addressed ids. May need content-hash ids. |
| push/pop (`clone_boxed`) | **Hard** | snapshotting a running circuit. duckdb already punted (`unimplemented!`). |

**Prediction**: Feldera is **not a clean fit for the current trait.** The query/action
builder shape survives, but the *driver* (`run_rules`/`flush_updates`/`IterationReport`) and
the *seminaive flag* encode the reference engine's iterate-to-fixpoint loop. Feldera's value
is precisely that it does seminaive natively; the trait makes you pretend to be a
batch-iterating engine. This is the strongest argument that "mirror `RuleBuilder`" leaked
the reference model. **Medium-high risk.** Recommend a thin driver abstraction (issue #2)
before starting Feldera.

---

## (e) RECOMMENDATION (ranked, with effort)

**Verdict: do NOT start both backends on the trait as-is. Make changes 1–2 first (they
gate the paper's proof story and the Feldera fit), then build Flowlog as the
validation case, then revisit before Feldera.**

1. **Promote proof/term-encoding intent into typed trait data — do this FIRST.**
   *Why*: it's the #1 paper risk and the only change that prevents every new backend from
   re-implementing duckdb's `"ordering-min"` string-matching. *What*: add explicit
   `FunctionConfig` signal for union-find/proof roles and a first-class `MergeFn::LatticeMin`
   (or a `MergeFn::Builtin(name)` the frontend already knows), so `add_table` carries
   "this column is a UF leader / this merge is min" instead of a primitive name.
   *Effort*: medium, ~3 days. Touches `proof_encoding.rs` emission, `FunctionConfig`, the
   two backend `add_table`s. Highest leverage.

2. **Abstract the driver to decouple from iterate-to-fixpoint — do this before Feldera.**
   *Why*: `run_rules`/`flush_updates`/`IterationReport` is the reference engine's loop; it's
   the part that fights Feldera (§d). *What*: split into a "submit work" + "advance until
   fixpoint or N steps" contract, where an incremental backend can implement
   "advance" as "step the circuit" and a batch backend keeps today's behavior. Let the
   *backend* own the fixpoint loop, not the frontend. *Effort*: medium, ~2-3 days; mostly
   `src/lib.rs`/`src/scheduler.rs` run loop + trait method reshape.

3. **Make `RuleBuilderOps` fallible where backends can fail; delete `deferred_err`.**
   *Why*: removes a whole class of "smuggle the error to build()" bugs and makes
   unsupported ops explicit at the call. *What*: change `set`/`remove`/`union`/
   `call_external_func`/`new_var` to `-> Result<…>`. *Effort*: small-medium, ~1-2 days,
   mechanical but wide.

4. **Shrink the `as_any` surface / kill the unsafe reborrows.** *Why*: 60 downcast sites and
   6 `unsafe` reborrows are the practical neutrality breakers. *What*: give `new_rule` a
   `&self`-compatible shape (builder borrows pool immutably, registers on `build(&mut self)`)
   to remove the reborrows; lift the handful of genuinely bridge-only ops (`unstable-fn`,
   action registry) behind narrow trait methods instead of downcasts. *Effort*: medium,
   ~2-3 days. Can be deferred until after Flowlog proves the shape.

5. **Collapse capability flags into one negotiated descriptor** (e.g.
   `fn capabilities(&self) -> Capabilities`). *Why*: stops the boolean explosion as backends
   grow. *Effort*: small, ~half a day. Cosmetic; do opportunistically.

**Build order**: (1) → (2) → build **Flowlog** (it's duckdb-shaped and will validate that
proofs run faithfully once intent is typed) → reassess (3)/(4) with two real consumers →
then **Feldera**. Do not build Feldera until (2) lands; the driver mismatch is structural,
not cosmetic.
