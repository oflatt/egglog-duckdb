# Backend Trait Refactor — Implementation Plan

**Audience**: An engineer plus agent team familiar with egglog3 but not part of the planning session.
**Goal**: Unify `src/lib.rs::EGraph` (uses `egglog_bridge::EGraph`) and `src/backend_duckdb.rs::DuckdbBackend` (uses `egglog_bridge_duckdb::EGraph`) behind a single `Backend` trait so that frontend features — `(extract …)`, `(prove-exists …)`, serialization — work uniformly across backends. Delete the parallel pipeline in `backend_duckdb.rs`.

## Design principle (added)

**Change the existing backend and frontend IRs as little as possible.** Rather than introducing a new neutral `RuleIr` that both backends translate from, the trait surface mirrors the existing bridge `RuleBuilder` API. DuckDB adapts to it; the bridge is a thin passthrough; the frontend's rule-building code is unchanged.

Implications:

- **Bridge IR (`RuleBuilder`, `QueryEntry`, `Query`, callback chain in `egglog-bridge/src/rule.rs`)**: unchanged.
- **DuckDB IR (`Rule`, `Atom`, `Action`, `Term`, `compile.rs` pipeline in `egglog-bridge-duckdb/src/`)**: unchanged.
- **`BackendRule` in `src/lib.rs:1985-2007`**: unchanged — it still calls `.query_table(...)`, `.lookup(...)`, `.set(...)` etc. through the trait. Surface looks identical.
- The trait's rule-building method is `Backend::new_rule(name, seminaive) -> Box<dyn RuleBuilderOps + '_>`. `RuleBuilderOps` mirrors the existing `egglog_bridge::RuleBuilder` method signatures one-for-one.
- Bridge `RuleBuilderOps` impl: trivial newtype wrapping `egglog_bridge::RuleBuilder`. Each method delegates.
- DuckDB `RuleBuilderOps` impl: accumulates calls into a `duck::Rule` struct (the existing data IR), submits to `compile_rule` on `build()`. The translation layer lives entirely inside this impl.
- `QueryEntry`, `MergeFn`, `DefaultVal`, `FunctionConfig`, `FunctionRow`, `ColumnTy` continue to live in `egglog-bridge`. The new trait crate re-exports them. DuckDB depends on egglog-bridge to use these types (or to translate to its own internal forms). **No types move crates.**

What we DON'T do:

- ~~Define a neutral `RuleIr`~~ — drop. The trait surface IS the IR.
- ~~Rewrite `BackendRule` to emit `RuleIr`~~ — drop. `BackendRule` stays.
- ~~Move `FunctionId`/`RuleId`/etc. into a new crate~~ — they stay in egglog-bridge, the trait re-exports.
- ~~Refactor `egglog-bridge`'s internals~~ — only add a thin `impl Backend` block.

This roughly halves the refactor's blast radius. New cost: ~12–15 engineer-days (was 16–24).

## Scope constraints (updated)

- **DuckDB backend is term-encoding only**. DuckDB always runs programs through term encoding; programs that don't lower (containers, custom presorts, primitives without validators, missing merge functions, etc.) are *already* gated out by `program_supports_proofs` in `src/proofs/proof_encoding_helpers.rs:462`, which the test harness consults via `file_supports_proofs` for both `duckdb` and `duckdb_proofs` combos. So the 28 container-sort test files never reach DuckDB — no new skip mechanism is needed.
- **Impact on the plan**:
  - Phase 2 Commit 13 (DuckDB `ContainerPool`) is **removed entirely**. DuckDB's `ContainerPool` impl is an empty stub; container-touching API surface is unreachable from any DuckDB run, so its defensive error paths never fire in practice.
  - Phase 0.3 / Phase 3.2 container-rebuild interaction sections become NOT-APPLICABLE — DuckDB never holds containers, never rebuilds them.
  - The container-rebuild ↔ native-UF handshake (originally the most subtle integration) **disappears** as a blocker.
  - DuckDB-side `BaseValuePool` still needed (i64/f64/String/bool/Unit and user-defined `BaseValue` impls), but the more complex `ContainerValues` mirror does not.
  - **No new test-skip work**: the existing `supports_proofs` gate already excludes every container-using file from DuckDB's two combos.
- The trait's `container_pool()` / `register_container` methods remain in the surface (the bridge needs them), but the DuckDB impl is a stub — callers that reach it from DuckDB indicate a bug (an upstream gate failed).

**Estimated total effort**: 12–15 engineer-days (down from 18–28 with containers cut AND minimal-IR-change posture). Parallelizable to ~6–8 wall-clock days with 2 agents after Phase 0.

**Key file paths used throughout**:
- `/Users/oflatt/egglog3/src/lib.rs` — frontend `EGraph` (2446 lines), holds `backend: egglog_bridge::EGraph`
- `/Users/oflatt/egglog3/src/backend_duckdb.rs` — parallel pipeline to be deleted (1551 lines)
- `/Users/oflatt/egglog3/src/scheduler.rs` — heavy `backend.X` consumer (heavy ExecutionState use)
- `/Users/oflatt/egglog3/src/extract.rs` — `for_each_while`, container/base values
- `/Users/oflatt/egglog3/src/serialize.rs` — `for_each_while`, canonicalization
- `/Users/oflatt/egglog3/src/prelude.rs` — `rust_rule`, `query`, `TableAction`/`UnionAction` users
- `/Users/oflatt/egglog3/src/sort/mod.rs:55` — `Sort::column_ty(&self, &egglog_bridge::EGraph)` (a leak point)
- `/Users/oflatt/egglog3/src/proofs/proof_extraction.rs:55` — `for_each_while`
- `/Users/oflatt/egglog3/egglog-bridge/src/lib.rs` (1491 lines) and `rule.rs` (869 lines) — the reference backend's public API
- `/Users/oflatt/egglog3/egglog-bridge-duckdb/src/lib.rs` (1676 lines), `compile.rs` (1354 lines) — DuckDB backend
- `/Users/oflatt/egglog3/core-relations/src/common.rs:103` — `define_id!(pub Value, u32, …)`
- `/Users/oflatt/egglog3/core-relations/src/base_values/mod.rs:66` — `pub struct BaseValues`
- `/Users/oflatt/egglog3/core-relations/src/containers/mod.rs:70` — `pub struct ContainerValues`

---

## Phase 0 — Research and Inventory (1–2 days, single agent, blocking)

Phase 0 produces three artifacts. **No code changes**. Output: a single markdown doc committed to the repo (`docs/backend_trait_inventory.md`).

### 0.1 Catalog of `backend.X(...)` call sites

There are **69 call sites** across `src/` (counted: `grep -rn "self\.backend\.\|egraph\.backend\.\|\.backend\.\b" /Users/oflatt/egglog3/src/ | wc -l`). The **distinct method surface is 19 methods**:

| Method | Callers | Purpose | DuckDB difficulty |
|---|---|---|---|
| `add_table(FunctionConfig) -> FunctionId` | `lib.rs:742`, `scheduler.rs:338` | Register a function/relation | Medium — maps to `add_function`/`add_relation`/`add_eq_sort_constructor`; DuckDB needs to interpret `DefaultVal`/`MergeFn` |
| `add_values(iter<(FunctionId, Vec<Value>)>)` | `lib.rs:1017,1040` | Bulk insert | Easy — maps to `insert` with literal args |
| `base_value_constant<T>(x) -> QueryEntry` | `scheduler.rs:327`, used in primitives | Build a typed const for rule IR | Tangled with QueryEntry; see 0.2 |
| `base_values()` (15+ uses) | `lib.rs`, `scheduler.rs`, `prelude.rs`, etc. | Returns `&BaseValues` for `get<T>` / `unwrap<T>` | **Hard** — DuckDB has no typed pool; primitives are inline i64/VARCHAR/DOUBLE in tables |
| `base_values_mut()` | (only internal) | Mutate base value registry | Same as above |
| `container_values()` (4+ uses) | `lib.rs:1924`, `serialize.rs:347,365`, `extract.rs:324,364,527,535` | Returns `&ContainerValues` for `get_val<T>`/`register_val<T>` | **Hardest** — DuckDB has no container model at all |
| `container_values_mut()` | (internal in bridge) | — | Same |
| `dump_debug_info()` | `lib.rs:1970` | Diagnostic log dump | Trivial — call `db.conn_for_dump()` and SELECT |
| `flush_updates() -> bool` | `lib.rs:1626`, `scheduler.rs:240` | Drain staged inserts, returns "changed" | Medium — DuckDB has no staged-update concept; either no-op or maps to running pending sync |
| `for_each(FunctionId, FnMut(FunctionRow))` | `lib.rs:1128,2344`, `extract.rs:457,697` | Iterate rows | Easy — `SELECT * FROM <table>` |
| `for_each_while(FunctionId, FnMut(FunctionRow) -> bool)` | `serialize.rs:146`, `extract.rs:912`, `proofs/proof_extraction.rs:55` | Iterate with early stop | Easy with row-by-row cursor; mind lifetime — see 0.3 |
| `free_external_func(ExternalFunctionId)` | `lib.rs:1205,1269`, `prelude.rs:671` | Drop user-registered primitive | DuckDB equivalent: drop registered VScalar |
| `free_rule(RuleId)` | `lib.rs:1102,1267`, `prelude.rs:671` | Drop rule | DuckDB: remove from `self.rules` |
| `lookup_id(FunctionId, &[Value]) -> Option<Value>` | `lib.rs:1013,1952` | Key→output lookup | Easy — `SELECT cN FROM t WHERE c0=… LIMIT 1` |
| `new_panic(String) -> ExternalFunctionId` | `prelude.rs:540`, internal | Register a deferred-panic primitive | Already mirrored — `Action::Panic` |
| `new_rule(name, seminaive) -> RuleBuilder<'_>` | `lib.rs:1056,1095,1169,1255`, `scheduler.rs:348,367` | **Build rule via mutable builder** | **Hardest** — DuckDB builds rules from the duck::Rule IR; the bridge's RuleBuilder is callback-based with QueryEntry. See Phase 3.4 |
| `register_external_func(Box<dyn ExternalFunction>) -> ExternalFunctionId` | `typechecking.rs:187`, scheduler, `lib.rs:1162,1249` | Register a user primitive | Medium — DuckDB has VScalar UDFs (see `UfFindScalar`); needs general ExecutionState bridge |
| `run_rules(&[RuleId]) -> Result<IterationReport>` | `lib.rs:1101,1203,1266`, `scheduler.rs:221,252` | Run one iteration of given rule set | Medium — maps to DuckDB's `run_iteration_in_set` with set of rule names |
| `set_report_level(ReportLevel)` | `lib.rs:1963` | Adjust report verbosity | Trivial — store on struct |
| `table_size(FunctionId) -> usize` | `lib.rs:862,878,1939,1860` | Row count | Easy — `SELECT COUNT(*) FROM t` (already in `db.count`) |
| `with_execution_state<R>(FnOnce(&mut ExecutionState) -> R) -> R` | `lib.rs:1611,1618,1929`, `scheduler.rs:225` | Borrow execution state for staging mid-callback (e.g. `TableAction::insert`, `register_val`) | **Hardest** — see Phase 3.3 |

In addition, `egglog_bridge::EGraph` itself is leaked through public API:
- `Sort::column_ty(&self, backend: &egglog_bridge::EGraph) -> ColumnTy` — declared at `src/sort/mod.rs:55`, implemented for each sort
- `RuleBuilder<'_>` (returned by `new_rule`) — used by `BackendRule` in `lib.rs:2007`
- `TableAction::new(&egraph.backend, FunctionId)` and `UnionAction::new(&egraph.backend)` — used in `lib.rs:1608,1981`, `scheduler.rs:236`, `prelude.rs:551,544`
- Several types re-exported from `egglog_bridge::{FunctionRow, ColumnTy, QueryEntry, MergeFn, DefaultVal, FunctionConfig, FunctionId, RuleId, ExecutionState}` are passed around as if they were stable IR.

### 0.2 `Value` type semantics

`Value` is a transparent `u32` newtype defined at `core-relations/src/common.rs:103` (`define_id!(pub Value, u32, …)`). Three coexisting meanings inside the bridge:

1. **EqSort id**: bare interned u32 owned by the bridge's id counter; resolves via `get_canon_in_uf` → UF table
2. **Base value (primitive)**: interned in `BaseValues` (`core-relations/src/base_values/mod.rs:66`) — a registry of typed `InternTable<P, Value>` per `BaseValue` trait impl. For "may-unbox" types (e.g. `i64`), the high bit signals whether the bytes are stored inline (`VAL_OFFSET = 1<<31`) or are an index into the intern table.
3. **Container id**: produced by `ContainerValues::register_val::<C>` (`core-relations/src/containers/mod.rs`) — same u32 namespace.

The disambiguation lives in `ColumnTy`: `Id` (eq-sort or container) vs `Base(BaseValueId)` (`egglog-bridge/src/lib.rs:45`).

**DuckDB equivalent today**: it sidesteps `Value` entirely. All columns are concrete SQL types (`BIGINT`, `VARCHAR`, `DOUBLE`, `BOOLEAN`, `STRUCT`). EqSort IDs are `BIGINT` allocated from a DuckDB sequence. Primitives are inlined. The `i64::Literal` and `String::Literal` are passed through as `ToSql`.

**Implication for the trait**: We cannot make `Value` a backend-agnostic newtype without forcing both impls to materialize their state into the same shape. Two options:

- **(A) Keep `Value = u32`** as today (it's already in `egglog_core_relations`, neutral crate). DuckDB needs an internal lossless mapping `Value <-> (DuckDB SQL value)`. For i64s this is the identity (`Value::new(x as u32)` only works for values fitting in u32; we'd need to widen). For Strings/F64 we need a tiny intern table on the DuckDB side.
- **(B) Make `Value` a backend associated type**. This pushes a generic parameter through every API that returns/accepts a Value — including `Sort`, `TermDag` reconstruction, `extract.rs`'s cost model, `prelude.rs` user-facing helpers, `add_primitive_with_validator`. The blast radius is enormous.

**Recommendation: Option A.** Pin `Value = egglog_core_relations::Value` as the trait's value type. Widen to `u64` if needed (Phase 1.1 spike to confirm `u32` suffices; DuckDB EqSort sequence already starts at 1 and won't overflow u32 in practice — but eg. integer literals like `1_000_000_000_000` do not fit). The simpler concrete decision: **widen `Value` from u32 to u64** as a pre-refactor housekeeping commit (Phase 0 ships it). This is a controlled change inside `core-relations`.

### 0.3 `BaseValues` / `ContainerValues` model

- `BaseValues` (`core-relations/src/base_values/mod.rs:66`): a `HashMap<TypeId, BaseValueId> + DenseIdMap<BaseValueId, Box<dyn DynamicInternTable>>`. Each entry is a `BaseInternTable<P: BaseValue>` (`Hash + Eq + Clone + Any + Send + Sync`). `unwrap::<P>(Value)` and `get::<P>(P)` are the public surface.
- `ContainerValues` (`core-relations/src/containers/mod.rs:70`): similar shape but with rebuilding semantics (the rebuild loop in `egglog-bridge/src/lib.rs:500` interleaves table rebuild with `ContainerValues::rebuild_containers`).

**How DuckDB would model these**:

- **For `BaseValues`**: a backend-local `BaseValuePool` that stores in-memory intern tables for non-trivially-encodable types. The trait gives DuckDB control of the pool. For i64/bool/f64/String/Unit, `Value` is the SQL value (sign-extended). For exotic `BaseValue` impls (BigInt, BigRat, Rational64, user-defined), use an in-memory intern table just like the bridge — `Value` is the index. The handful of call sites (`base_values().get::<i64>(7)`, `unwrap::<i64>(v)`) become trait methods `pool.get<T>(&self, T)` and `pool.unwrap<T>(&self, Value)`, where the impl picks inline vs interned per registered `BaseValue::MAY_UNBOX`.
- **For `ContainerValues`**: this is the harder one. DuckDB has no rebuild model for in-memory containers. Two paths:
  - **(a) In-process container pool on DuckDB side too.** Keep an exact mirror of the bridge's `ContainerValues` in the DuckDB backend, with the rebuild loop firing whenever DuckDB completes an iteration that may have changed an EqSort UF. Concretely: after each `run_iteration_in`, scan native UFs for new unions, then call `container_values.rebuild_containers(uf_table_view)`. The bridge already exposes this logic — extract it into `egglog-core-relations` so both backends reuse it. **This is what the plan recommends.**
  - **(b) Encode containers as DuckDB STRUCT/LIST.** Far more work; DuckDB has LIST and STRUCT types but no native UF rebuild; requires per-iteration UPDATE statements scanning every container-bearing table.

**Recommendation: (a)**. The `ContainerValues` pool stays in-process, owned by the backend impl. Containers are NOT stored in DuckDB tables — only their `Value` id ever appears in a SQL column. The pool ↔ DuckDB hash-cons / rebuild handshake mirrors what the in-memory bridge already does.

### 0.4 `with_execution_state` semantics

`ExecutionState<'_>` (`core-relations::ExecutionState`) is the bridge's per-call mutable cursor over the database: it carries access to `BaseValues`/`ContainerValues`, the counter for fresh ids, staging buffers for in-rule inserts, and the side-channel for panic propagation. It's borrowed by:

- Rule actions during `run_rules` (the inner-loop callbacks built by `RuleBuilder`)
- Top-level `with_execution_state` calls (4 sites in `src/`) for: container registration (`container_to_value` in `lib.rs:1929`), file-input bulk insert (`lib.rs:1611,1618`), scheduler match canonicalization (`scheduler.rs:225`)

The reason it takes a `FnOnce(&mut ExecutionState) -> R` rather than returning `&mut ExecutionState`: ExecutionState borrows from `Database` for the duration of the callback; you can stage many updates then drop. The borrow checker forbids a long-lived `&mut`.

**DuckDB equivalent**: DuckDB rule actions are pure SQL — they don't have an analog of "stage these inserts into a buffer then merge". The 4 top-level `with_execution_state` callers each do one of:
1. Register a container value (`container_to_value`) — DuckDB: call into the in-process container pool directly (no DB transaction needed)
2. Input-file bulk insert (`input_file`) — DuckDB: build a single multi-row `INSERT … VALUES` SQL
3. Scheduler match canonicalization (`scheduler.rs:225`) — this calls `TableAction::insert` per match; on DuckDB, equivalent is a per-match SQL insert

The cleanest abstraction: **don't include `with_execution_state` in the trait**. Instead, refactor each caller to use higher-level trait methods:
- `Backend::register_container<C>(C) -> Value`
- `Backend::insert_rows(FunctionId, &[Vec<Value>])`
- Push the scheduler-specific use into a `Backend::add_match_table` helper

This re-frames 4 cases of "give me state to stage updates" into 3 use-case-specific methods, none of which need a callback.

### 0.5 Rule compilation IR

Two distinct IRs today:

- **`egglog_bridge` IR**: callback-based builder. `RuleBuilder` (rule.rs:122) accumulates `BuildRuleCallback`s. The `Query` struct (rule.rs:101) holds atoms as `(TableId, Vec<QueryEntry>, SchemaMath)`. Actions are folded into the callback list. The "AST" only exists as the closure chain; once built, it's compiled to a `core_relations::CachedPlan`.
- **`egglog_bridge_duckdb` IR**: data IR. `Rule { name, ruleset, body: Vec<Atom>, actions: Vec<Action> }` (lib.rs:454). `Atom = Func { name, args } | Filter(Term) | Bind { var, expr }` (lib.rs:403). `Action = Insert | Delete | LetCtor | LetExpr | Panic` (lib.rs:421). `Term = Var | Lit | Prim | FuncCall`.

The duckdb IR is what `BackendRule` in `src/lib.rs:1985` would emit if we asked it to. It's strictly *less expressive* than the bridge IR (no subsume support — though that's tracked in TODOs; relies on names not FunctionIds; no MergeFn::Function tied to a TableAction; no `query_table` with `is_subsumed`).

**Two strategies for the trait**:

1. **Lower-bound IR**: Define a new neutral `RuleIr` enum in `egglog-bridge-trait` (a new crate, or a module in `egglog-bridge`) that both backends consume. `Backend::new_rule(RuleIr) -> RuleId`. Each backend translates. `BackendRule` in `src/lib.rs` emits `RuleIr` directly.
2. **Keep callback API**: `Backend` exposes `new_rule(...) -> Box<dyn RuleBuilderTrait + '_>` and both bridges implement the builder. DuckDB's impl materializes calls into a `duck::Rule` then submits at `build()`.

**Recommendation: (1) for the body, (2) for the builder seam during migration.** Step 1: introduce a neutral `RuleIr` and have the bridge's `RuleBuilder` get a `.build_from_ir(RuleIr)` shortcut. Have `BackendRule` produce `RuleIr` instead of calling `rb.query_table` etc. directly. Then `Backend::new_rule(RuleIr) -> RuleId` is the only seam. (This also has the side benefit of giving us a stable inspectable rule format for debugging and proof checking.)

The IR needs:
- `RuleIr { name, ruleset, seminaive, body: Vec<Atom>, actions: Vec<Action> }`
- `Atom = TableAtom { func: FunctionId, args, is_subsumed } | PrimAtom { prim: ExternalFunctionId, args, ret_ty } | … (rebuild-specific atoms)`
- `Action = TableInsert | TableLookup | TableSubsume | Union | Panic | LetPrim | Set | Remove | Change | … `
- Both `FunctionId` and `ExternalFunctionId` are stable handles owned by the backend (see Phase 1.3).

### 0.6 Deliverable

Commit `docs/backend_trait_inventory.md` with the four tables above plus a sketch of `RuleIr`. No code changes.

---

## Phase 1 — Design the `Backend` trait (1–2 days)

Output: a design doc (`docs/backend_trait_design.md`) AND a stub commit introducing the trait file (`src/backend/mod.rs` or new `egglog-backend-trait/` crate) with the trait definition but no impl. **No callers updated**.

### 1.1 Crate / module placement

- New crate: `egglog-backend-trait` at top-level, alongside `egglog-bridge` and `egglog-bridge-duckdb`. Avoids a circular dep (the trait can't live in `egglog-bridge` because `egglog-bridge-duckdb` would have to depend on it; but `egglog-bridge-duckdb` already exists as a peer).
- Depends only on `egglog-core-relations` (for `Value`), `egglog-reports`, `egglog-numeric-id`.
- Both `egglog-bridge` and `egglog-bridge-duckdb` add a dev-only/optional dep on it during the migration; in the final state both are required.

### 1.2 Core trait shape (sketch — final names TBD)

```text
pub trait Backend: Send + Sync {
    // --- table lifecycle ---
    fn add_table(&mut self, config: FunctionConfig) -> FunctionId;
    fn table_size(&self, table: FunctionId) -> usize;

    // --- iteration ---
    fn for_each_while(&self, table: FunctionId, f: &mut dyn FnMut(BackendRow<'_>) -> bool);
    fn for_each(&self, table: FunctionId, f: &mut dyn FnMut(BackendRow<'_>)) { … default impl }

    // --- direct access ---
    fn lookup_id(&self, func: FunctionId, key: &[Value]) -> Option<Value>;
    fn insert_rows(&mut self, table: FunctionId, rows: &[Vec<Value>]);

    // --- rule management ---
    fn new_rule_id(&mut self, name: &str, seminaive: bool) -> RuleId;
    fn finish_rule(&mut self, id: RuleId, ir: RuleIr);
    fn free_rule(&mut self, id: RuleId);
    fn run_rules(&mut self, rules: &[RuleId]) -> Result<IterationReport>;
    fn flush_updates(&mut self) -> bool;

    // --- primitives ---
    fn register_external_func(&mut self, f: Box<dyn ExternalFunction>) -> ExternalFunctionId;
    fn free_external_func(&mut self, id: ExternalFunctionId);
    fn new_panic(&mut self, msg: String) -> ExternalFunctionId;

    // --- typed value handles ---
    fn base_value_pool(&self) -> &dyn BaseValuePool;
    fn base_value_pool_mut(&mut self) -> &mut dyn BaseValuePool;
    fn container_pool(&self) -> &dyn ContainerPool;
    fn container_pool_mut(&mut self) -> &mut dyn ContainerPool;
    fn register_container<C: ContainerValue>(&mut self, c: C) -> Value;

    // --- canonicalization ---
    fn get_canon_repr(&self, val: Value, ty: ColumnTy) -> Value;

    // --- diagnostics ---
    fn set_report_level(&mut self, level: ReportLevel);
    fn dump_debug_info(&self);
}
```

The handles `FunctionId`, `RuleId`, `ExternalFunctionId`, `ColumnTy` move into `egglog-backend-trait`. `BaseValueId`, `ContainerValueId` go there too. `BackendRow<'_>` is the renamed `FunctionRow<'_>`, identical shape.

**Why `&mut dyn FnMut` and not `impl FnMut`**: trait must be `dyn`-compatible because the frontend will hold `Box<dyn Backend>` or similar (Phase 1.5).

**Why a separate `BaseValuePool` / `ContainerPool` trait**: the methods are generic over `T: BaseValue` / `C: ContainerValue`, which doesn't fit in a `dyn Backend` directly. Splitting them out lets `dyn Backend` work; the pool sub-traits use `&dyn Any`-style downcast methods.

### 1.3 Handles: backend-owned, opaque to caller

- `FunctionId`, `RuleId`, `ExternalFunctionId` are `u32` newtypes (same as today). Each backend interprets them in its own internal map. The trait promises only "we return one to you, you give it back".
- Bridge interprets them as `core_relations::TableId`, `egglog_bridge::RuleId`, `core_relations::ExternalFunctionId` indices.
- DuckDB interprets them as indices into `Vec<FunctionInfo>`, `Vec<CompiledRule>`, `Vec<DuckExternalFn>`.
- Existing `egglog_bridge::FunctionId` becomes a re-export of `egglog_backend_trait::FunctionId`.

### 1.4 Rule building surface (was: `RuleIr` — now: `RuleBuilderOps` trait)

Under the minimal-change posture (see "Design principle" above), there's no neutral `RuleIr`. The trait's rule-building surface is a `RuleBuilderOps` trait object that mirrors the existing bridge `RuleBuilder` API:

```text
pub trait RuleBuilderOps {
    fn query_table(&mut self, table: FunctionId, args: &[QueryEntry], is_subsumed: Option<bool>);
    fn query_prim(&mut self, func: ExternalFunctionId, args: &[QueryEntry], ret: QueryEntry);
    fn lookup(&mut self, table: FunctionId, args: &[QueryEntry], panic_msg: String) -> QueryEntry;
    fn set(&mut self, table: FunctionId, args: &[QueryEntry]);
    fn insert(&mut self, table: FunctionId, args: &[QueryEntry]);
    fn subsume(&mut self, table: FunctionId, key: &[QueryEntry]);
    fn remove(&mut self, table: FunctionId, key: &[QueryEntry]);
    fn union(&mut self, l: QueryEntry, r: QueryEntry);
    fn panic(&mut self, msg: String);
    fn build(self: Box<Self>) -> Result<RuleId, BuildError>;
    // (final method list mirrors whatever the existing RuleBuilder exposes)
}
```

`Backend::new_rule(name, seminaive) -> Box<dyn RuleBuilderOps + '_>`. The lifetime ties to `&mut Backend`.

- **Bridge impl** (`egglog-bridge::EGraph::BridgeRuleBuilderOps`): a thin newtype around `RuleBuilder`. Every method delegates. Zero behavioral change.
- **DuckDB impl** (`egglog_bridge_duckdb::DuckRuleBuilderOps`): accumulates each call into an in-progress `duck::Rule { body, actions, … }`. On `build()`, hands off to the existing `compile_rule(&duck::Rule, &functions)` pipeline. Subsume / complex MergeFn calls return errors when DuckDB doesn't support them.

`QueryEntry`, `FunctionId`, `ExternalFunctionId`, etc. stay in `egglog-bridge` (or `core-relations`). The new trait crate re-exports them.

```text
pub struct RuleIr {
    pub name: String,
    pub seminaive: bool,
    pub vars: DenseIdMap<VarId, ColumnTy>,
    pub atoms: Vec<RuleAtom>,
    pub actions: Vec<RuleAction>,
}

pub enum RuleAtom {
    Table { func: FunctionId, args: Vec<RuleEntry>, is_subsumed: Option<bool> },
    Prim  { func: ExternalFunctionId, args: Vec<RuleEntry>, ret_ty: ColumnTy },
}

pub enum RuleEntry { Var(VarId), Const { val: Value, ty: ColumnTy } }

pub enum RuleAction {
    Let       { var: VarId, source: LetSource },
    Set       { func: FunctionId, args: Vec<RuleEntry> },
    Insert    { func: FunctionId, args: Vec<RuleEntry> },
    Subsume   { func: FunctionId, key: Vec<RuleEntry> },
    Remove    { func: FunctionId, key: Vec<RuleEntry> },
    Union     { l: RuleEntry, r: RuleEntry },
    Panic     { msg: String },
}
pub enum LetSource {
    Lookup    { func: FunctionId, args: Vec<RuleEntry>, panic_msg: String },
    CallPrim  { func: ExternalFunctionId, args: Vec<RuleEntry>, ret_ty: ColumnTy },
    Entry     (RuleEntry),
}
```

### 1.5 Frontend `EGraph::backend` becomes a trait object

```text
pub struct EGraph {
    backend: Box<dyn Backend>,
    …
}
```

`EGraph::default()` constructs a `Box::new(egglog_bridge::EGraph::default())`. A new `EGraph::with_duckdb()` constructs `Box::new(egglog_bridge_duckdb::EGraph::new()?)`.

The `Sort::column_ty(&self, backend: &egglog_bridge::EGraph)` signature changes to `column_ty(&self, backend: &dyn Backend)` — call sites in `src/sort/*.rs` update.

`TableAction`/`UnionAction` are problematic: they currently hold `egglog_bridge`-specific fields. Plan: move them behind the trait too via a `TableActionOps` trait.

**The deepest blocker**: `TableAction` is used in `prelude.rs::RustRuleRhs` *inside* a primitive's `apply()`. For DuckDB, primitives run inside DuckDB's SQL via a VScalar UDF, and they cannot synchronously call back into DuckDB to do a `TableAction::lookup`. Two options:

- **(α)** Disable user primitives that touch tables on DuckDB. Document. `rust_rule` and `query` are runtime errors when backend is DuckDB.
- **(β)** Refactor `RustRuleRhs` to be two-phase. Much more work; defer.

**Recommendation: (α) for the refactor.** Add a `Backend::supports_inline_table_lookups() -> bool` flag.

### 1.6 Migration-friendly seam

During migration commits, `Box<dyn Backend>` is the long-term shape. First commit can leave `backend: egglog_bridge::EGraph` as-is and add a trait that's implemented for it.

### 1.7 Per-method DuckDB impl sketches

(See Phase 0.1 table; one paragraph per method in the full plan above.)

---

## Phase 2 — Migration plan (commit by commit, each ≤ 500 lines diff, tests green)

### Commit 1 — Pre-refactor: confirm `Value` u32 sufficient or widen to u64

### Commit 2 — Introduce `egglog-backend-trait` crate with no impls

New crate at top-level. Re-exports `Value` from `egglog-core-relations`. Defines all the types. No impl yet.

### Commit 3 — Move id types from `egglog-bridge` to `egglog-backend-trait`

`egglog-bridge` depends on the new crate. Existing definitions become `pub use`.

### Commit 4 — Implement `Backend` trait for `egglog_bridge::EGraph`

Trait impl block delegates to existing inherent methods. Frontend not yet using the trait. Trait impl is dead code for now.

### Commit 5 — (CUT under minimal-change posture)

Originally: refactor `BackendRule` to emit a neutral `RuleIr`. **Cut.** Under the new posture the trait's rule-building method exposes a `RuleBuilderOps` trait object that mirrors the existing `RuleBuilder` API one-for-one. `BackendRule` continues to call `.query_table(…)`, `.lookup(…)`, `.set(…)` etc. through the trait — no rewrite. The bridge passes through. DuckDB adapts.

What this commit does instead: introduce the `RuleBuilderOps` trait and the bridge's trivial passthrough impl. Small commit (~150 lines).

### Commit 6 — Restructure `with_execution_state` callers

Add new trait methods. Migrate the 4 call sites. Bridge impl internally calls `with_execution_state`.

### Commit 7 — Replace `Sort::column_ty(&egglog_bridge::EGraph)` with `Sort::column_ty(&dyn Backend)`

Touches all sort impls and call sites.

### Commit 8 — Frontend EGraph field becomes `Box<dyn Backend>`

`EGraph::default()` boxes a fresh bridge. All `self.backend.X` go through trait dispatch (benchmark for regression). `TableAction`/`UnionAction` move behind trait method. Handle `Clone` via `Backend::clone_boxed`.

**At this point, lib.rs's EGraph runs on any `dyn Backend`, but only bridge exists. All tests pass.**

### Commit 9 — Stub DuckDB `Backend` impl (compiles, doesn't run)

### Commit 10 — Wire DuckDB `RuleBuilderOps` impl

New file `egglog-bridge-duckdb/src/rule_builder.rs`: a struct that implements `RuleBuilderOps`, accumulating each method call into a `duck::Rule` data IR. On `build()`, submits to the existing `compile_rule` pipeline. Unsupported operations (subsume on duckdb, `MergeFn::Function`, etc.) error at the corresponding `RuleBuilderOps` method call with a clear "unsupported on duckdb" message. **No changes to `duck::Rule`, `compile.rs`, or the existing DuckDB rule pipeline.**

### Commit 11 — Wire DuckDB `BaseValuePool`

In-process pool, same shape as core-relations's.

### Commit 12 — Wire DuckDB external functions (limited)

VScalar wrapper. Primitives needing TableAction error at registration.

### Commit 13 — DuckDB `ContainerPool` stub (NOT supporting containers)

DuckDB backend explicitly does not implement containers. `container_pool()` returns a fixed empty pool. `register_container` errors. Sort registration errors with a clear "container sorts not supported on duckdb backend" message if it sees `presort_and_args` for `Vec`/`Set`/`Map`/etc. (Pair is kept as a special case for proof mode — already implemented, stays on the DuckDB side.)

Original full ContainerPool implementation (~2 days) is **cut** from this plan. Reinstate only if/when container support on DuckDB becomes a goal.

### Commit 14 — Flip the test harness to drive DuckDB through the unified pipeline

`tests/files.rs` uses `egraph.parse_and_run_program` for both paths. Catalog new failures.

### Commit 15+ — Iterate on extract/prove-exists/serialize correctness on DuckDB

Sub-commits:
- 15a — `for_each_while` correctness
- 15b — `get_canon_repr` for `ColumnTy::Id`
- 15c — `prove_exists` works on DuckDB
- 15d — `serialize.rs` works on DuckDB

---

## Phase 3 — DuckDB-specific implementation details (parallel with Phase 2 from Commit 9)

### 3.1 `for_each_while` lifetime

DuckDB: `Connection::prepare(...)` → `Statement` → `Rows` iterator. Pack each row into a `BackendRow { vals: &[Value] }` via a per-row buffer. Subsumption: today no DuckDB column models it. Plan: always `subsumed: false`, document degradation.

### 3.2 `BaseValues` strategy (containers not applicable)

- **Inline-friendly types** (i64, bool, f64, Unit): encode directly into a 64-bit `Value`.
- **String / BigInt / BigRat / user-defined `BaseValue` impls**: in-memory intern table on the DuckDB-side `BaseValuePool`.
- **ContainerValues**: NOT IMPLEMENTED on DuckDB. The trait method returns an empty pool; sort registration errors on any `(sort X (Vec/Set/Map/etc A))`. Pair container is a special case — already handled by `ColumnTy::PairI64` and `pair_sorts` in the current DuckDB backend; that path stays.

### 3.3 `with_execution_state` elimination

Trait does not expose `ExecutionState` to frontend. 4 call sites become dedicated methods. User-defined primitives in `apply()` get a stripped-down ExecutionState; staged inserts forbidden on DuckDB.

### 3.4 Rule compiler unification

No new IR. The trait's `RuleBuilderOps` mirrors the existing bridge `RuleBuilder` one-for-one (Phase 1.4). Bridge passes through. DuckDB's impl accumulates calls into the existing `duck::Rule` data IR, then runs the existing `compile_rule` pipeline. Both existing IRs are untouched.

---

## Phase 4 — Cleanup (2 days, after Phase 3 green)

### What deletes

- `src/backend_duckdb.rs` (entire 1551-line file)
- `tests/files.rs` lines 159–197 (duckdb-specific branch)
- CLI `src/cli.rs:145-147`
- The `DuckdbBackend` typechecker field

### What survives

- Everything in `egglog-bridge-duckdb/src/` — storage/execution-side optimizations
- CLI `--duckdb` flag (now constructs `EGraph::with_duckdb_backend`)
- The `_duckdb` snapshot suffix

### Naming cleanup

- `egglog_bridge::EGraph` → `egglog_bridge::ReferenceBackend`
- `egglog_bridge_duckdb::EGraph` → `egglog_bridge_duckdb::DuckdbBackend`

---

## Phase 5 — Effort estimates and parallelization

| Phase | Days |
|---|---|
| 0 — Research + inventory | 1–2 |
| 1 — Trait design | 1–2 |
| 2.C1–C2 — Pre-refactor + new crate | 1 |
| 2.C3 — Move id types | 0.5 |
| 2.C4 — Bridge impls trait | 1 |
| 2.C5 — RuleBuilderOps trait + bridge passthrough | 0.5 |
| 2.C6 — ExecutionState callers | 1 |
| 2.C7 — Sort::column_ty | 0.5 |
| 2.C8 — Box<dyn Backend> | 1.5 |
| 2.C9 — Stub DuckDB impl | 1 |
| 2.C10 — DuckDB rule IR | 2 |
| 2.C11 — DuckDB BaseValuePool | 1 |
| 2.C12 — DuckDB primitives | 1.5 |
| 2.C13 — DuckDB ContainerPool stub | 0.5 |
| 2.C14 — Flip test harness | 0.5 |
| 2.C15 — Iterate extract/prove/serialize | 3–5 |
| 4 — Cleanup | 1 |
| **Total** | **12–15** |

### Parallelization (2 agents)

**Strict sequential blockers**: Phase 0 → Phase 1 → Phase 2 Commits 1–8. ~7 days, single agent.

**After Commit 8**, two agents parallel:
- **Agent A**: 9 → 10 → 13 → 15c/d (rule-side / functional)
- **Agent B**: 11 → 12 → 14 → 15a/b (data/value side)

**Wall clock with 2 agents**: ~9 days (7 sequential + 2 parallel after containers cut). The cut Commit 13 (ContainerPool full impl) was originally the longest parallel-track item.

### Risks

- **Performance regression** from `Box<dyn Backend>` indirect dispatch. Mitigate: benchmark; add monomorphized fast path if >5%.
- **Clone semantics for `dyn Backend`** (push/pop snapshots). DuckDB connection clone is non-trivial; may need checkpoint/rollback model.
- **Subsumption on DuckDB**: not modeled today. Either add column or degrade.
- **User primitives needing table access**: `rust_rule`/`query` DuckDB-incompatible v1.
- **Container rebuild ↔ DuckDB native UF** interaction: most subtle part of Commit 13.

### Recommended agent assignments

- **Phases 0–1, Commits 1–8**: Single agent (one coherent author)
- **Commits 9–14**: Two agents in parallel
- **Commits 15+**: Either agent picks failing test files
- **Cleanup**: Same agent who did Commit 8
