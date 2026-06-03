# Backend Trait Refactor — Phase 0 Inventory

**Phase**: 0 (research only, no code changes)
**Source plan**: `/Users/oflatt/egglog3/.tmp/backend_trait_refactor_plan.md`
**Scope adjustment**: DuckDB does **not** need to support containers (Vec/Set/Map/MultiSet). DuckDB only runs term-encoded programs, and term encoding's gate (`program_supports_proofs` in `src/proofs/proof_encoding_helpers.rs:462`, which returns `SortWithPresort` for any container sort) already excludes container files from DuckDB's test combos via the existing `supports_proofs` / `file_supports_proofs` check in `tests/files.rs`. So no new skip mechanism is needed; the **28** test files using container sorts are already invisible to DuckDB. Container-related items in this document are tagged "N/A on duckdb" — meaning unreachable rather than something to gate against.

## Executive summary

The frontend `EGraph` in `src/lib.rs` calls into `backend: egglog_bridge::EGraph` at **69 call sites** across `src/` reducing to **19 distinct method names** (verified — count matches the plan's draft). Of those 19, three are container-specific and become "N/A on duckdb" under the updated scope: `container_values`, `container_values_mut`, and the container-side use of `with_execution_state`. Beyond raw method calls, the frontend leaks **11 concrete types from `egglog_bridge`/`core_relations`** through public signatures, the heaviest single offender being `Sort::column_ty(&self, &egglog_bridge::EGraph)` on `src/sort/mod.rs:55` which forces every sort impl in `src/sort/*.rs` and `src/prelude.rs` to name the concrete bridge type. `Value` (a `u32` newtype in `core-relations/src/common.rs:103`) is the safest shared currency between backends — it has three meanings (eq-sort id / base value handle / container handle) all disambiguated by an accompanying `ColumnTy` (`egglog-bridge/src/lib.rs:45`). With containers cut from DuckDB, `Value`'s only blocker on DuckDB is encoding base values: the DuckDB eqsort sequence (`__egglog_eqsort_seq`, `egglog-bridge-duckdb/src/lib.rs:658`) starts at 1 and won't overflow `u32` in any realistic workload, but `i64` literals can — so widening `Value` to `u64` (or adding a tiny intern table for out-of-range i64) is a small, isolated commit recommended as a preflight. The 4 `with_execution_state` callers can each be replaced with a dedicated, non-callback trait method. Rule compilation has two unrelated IRs today (bridge: callback-based `RuleBuilder`; DuckDB: data-IR `Rule { body: Vec<Atom>, actions: Vec<Action> }` at `egglog-bridge-duckdb/src/lib.rs:454`); a neutral `RuleIr` sketched in Section 5 is small enough to cover both. Test-impact preview at Commit 14 (flip): **28 test files declare container sorts** but they are **already excluded** by the existing `supports_proofs` gate (the term-encoding pass rejects `presort_and_args`), so no new skip mechanism is needed. **17 files** call `(extract …)` which currently produces no duckdb output (the harness strips `ExtractBest`/`ExtractVariants` from the shared snapshot — turning extract back on will re-introduce diff if the DuckDB output diverges).

---

## Section 1 — Call-site catalog (Phase 0.1)

### 1.1 Distinct backend-method surface

Generated from:

```
grep -rn "self\.backend\.\|egraph\.backend\.\|\.backend\.\b" /Users/oflatt/egglog3/src/
```

**Total call sites: 69. Distinct methods: 19.** Verified — both numbers match the plan's draft.

| # | Method | Call sites (`file:line`) | Brief purpose | DuckDB difficulty |
|---|---|---|---|---|
| 1 | `add_table(FunctionConfig) -> FunctionId` | `src/lib.rs:742`, `src/scheduler.rs:338` | Register a function/relation/constructor | Medium — maps to `add_function` / `add_relation` / `add_eq_sort_constructor`; DuckDB must interpret `FunctionConfig.merge` (`MergeFn`) and `FunctionConfig.default` (`DefaultVal`) |
| 2 | `add_values(iter<(FunctionId, Vec<Value>)>)` | `src/lib.rs:1017`, `src/lib.rs:1040` | Bulk insert rows for one or many functions | Easy — multi-row `INSERT … VALUES` |
| 3 | `base_value_constant<T>(x) -> QueryEntry` | `src/lib.rs:2066`, `src/lib.rs:2180-2184`, `src/scheduler.rs:327` | Build a typed constant `QueryEntry` for the rule IR | Tangled with `QueryEntry`. Replaced when rule IR replaces `QueryEntry` (Section 5) |
| 4 | `base_values() -> &BaseValues` | `src/lib.rs:1014,1016,1029,1155,1156,1363,1557,1569,1577,1583,1906,1911`; `src/scheduler.rs:325,326`; `src/prelude.rs:790,968,969,1018,1110`; `src/lib.rs:2190-2194` | Get typed-pool reference; subsequent `.get::<T>(T)` / `.unwrap::<T>(Value)` | Medium — DuckDB needs a `BaseValuePool` (trait object), per Phase 1; primitives like `i64`/`bool` are inline-encodable, others (`F`, `S`, `Z`, `Q`) need an in-process intern table |
| 5 | `base_values_mut() -> &mut BaseValues` | `src/prelude.rs:794` (inside `BaseSortImpl::register_type`) | Mutate base-value registry (register a new type) | Medium — analogous mutable surface on the duckdb pool |
| 6 | `container_values() -> &ContainerValues` | `src/lib.rs:1924`, `src/lib.rs:2281,2285`; `src/serialize.rs:347`; `src/extract.rs:324,364,527,535` | Access containers (lookup, iteration, inner-value enumeration) | **N/A on duckdb** — containers excluded. Sort registration errors on `(sort X (Vec/Set/Map/MultiSet ...))`; `Pair` is the kept exception, already handled in current DuckDB backend |
| 7 | `container_values_mut()` | (internal in bridge / sort registration paths; **0** direct callers in `src/`) | Mutable container registry | **N/A on duckdb** |
| 8 | `dump_debug_info()` | `src/lib.rs:1970` | Diagnostic log dump | Trivial — DuckDB issues `SELECT * FROM ...` per table |
| 9 | `flush_updates() -> bool` | `src/lib.rs:1626`, `src/scheduler.rs:240` | Drain staged inserts after rules / scheduler-side staging; returns "changed" | Medium — DuckDB has no staged-update concept; either no-op + run pending UPSERTs, or hook DuckDB's transaction commit |
| 10 | `for_each(FunctionId, F: FnMut(FunctionRow))` | `src/lib.rs:1128`, `src/lib.rs:2344`; `src/extract.rs:457,637,697` | Iterate every row | Easy — `SELECT * FROM <table>` cursor, bind into `BackendRow` |
| 11 | `for_each_while(FunctionId, F: FnMut(FunctionRow) -> bool)` | `src/serialize.rs:146`; `src/extract.rs:912`; `src/proofs/proof_extraction.rs:55` | Iterate with early-stop | Easy — same as #10 but break out when `false` returned; care with statement lifetime (Phase 3.1) |
| 12 | `free_external_func(ExternalFunctionId)` | `src/lib.rs:1205`, `src/lib.rs:1269`, `src/prelude.rs:671` | Drop user-registered primitive | Easy — drop the VScalar UDF and remove from registry |
| 13 | `free_rule(RuleId)` | `src/lib.rs:1102`, `src/lib.rs:1204`, `src/lib.rs:1267`, `src/prelude.rs:671` | Drop a rule | Easy — remove from `self.rules` |
| 14 | `lookup_id(FunctionId, &[Value]) -> Option<Value>` | `src/lib.rs:1013`, `src/lib.rs:1952` | Key→output point lookup | Easy — `SELECT cN FROM t WHERE c0=… LIMIT 1` |
| 15 | `new_panic(String) -> ExternalFunctionId` | `src/prelude.rs:540` | Register a deferred-panic primitive | Easy — already mirrored as `Action::Panic` |
| 16 | `new_rule(name, seminaive) -> RuleBuilder<'_>` | `src/lib.rs:1056,1095,1169,1255`; `src/scheduler.rs:348,367` | **Build a rule via a mutable, callback-based builder** | **Hardest** — DuckDB rules are a data IR (`duck::Rule`); this is the major surface refactor — see Section 5 |
| 17 | `register_external_func(Box<dyn ExternalFunction>) -> ExternalFunctionId` | `src/typechecking.rs:187`, `src/scheduler.rs:332`, `src/lib.rs:1162`, `src/lib.rs:1249` | Register a user primitive | Medium — DuckDB has VScalar UDFs (see `UfFindScalar` in `egglog-bridge-duckdb/src/uf.rs`); needs an `ExecutionState`-equivalent bridge to expose `base_values()` etc. inside the UDF |
| 18 | `run_rules(&[RuleId]) -> Result<IterationReport>` | `src/lib.rs:1101,1203,1266`; `src/scheduler.rs:221,252` | Run one iteration on the given rule set | Medium — maps to DuckDB's `run_iteration_in_set` with a set of rule names |
| 19 | `set_report_level(ReportLevel)` | `src/lib.rs:1963` | Adjust verbosity | Trivial — store on struct |
| 20 | `table_size(FunctionId) -> usize` | `src/lib.rs:862,878,1860,1939` | Row count | Easy — `SELECT COUNT(*) FROM t` (already implemented as `db.count`) |
| 21 | `with_execution_state<R>(FnOnce(&mut ExecutionState) -> R) -> R` | `src/lib.rs:1611,1618,1929`; `src/scheduler.rs:225` | Borrow execution state for staging mid-callback (insert / lookup / register container) | **Hardest** — see Section 4. The container-side caller (`lib.rs:1929`) is **N/A on duckdb**; the other 3 are non-container |

> **Notes on the table**
>
> - I count **21 rows** because I split `container_values_mut()` out for completeness (the plan's 19 omitted it; it has zero direct callers in `src/`). Effective surface in `src/` callers is 19 + 1 internal helper = **20 methods**.
> - `base_values_mut()` is called only inside `src/prelude.rs` (`BaseSortImpl::register_type` at line 794). Plan said "internal only" — confirmed, just one site.
> - `for_each` shows up at `src/extract.rs:637` too (uf_func iteration), missing from the plan's draft — added.
> - `free_rule` count is 4 (plan said 3) — confirmed: `src/lib.rs:1102, 1204, 1267` and `src/prelude.rs:671`.
> - Many `base_values()` sites are read-only convenience calls (`get::<T>(T)`, `unwrap::<T>(v)`); see Section 3.

### 1.2 `egglog_bridge` types leaked through `src/` API surface

These types are named in `src/` outside the `backend_duckdb.rs` parallel-pipeline file. They form the second axis of work after the method surface.

| Type | First defined at | Notable `src/` references | Surface impact |
|---|---|---|---|
| `FunctionId` | `egglog-bridge/src/lib.rs:51` (`define_id!`) | `src/lib.rs:298,742,967,1860,1939,1952,2021-2023`; `src/scheduler.rs:6,291,338`; `src/lib.rs:128` (`Function.backend_id`) | Public — every `Function` struct exposes one. Becomes neutral handle. |
| `RuleId` | `egglog-bridge/src/lib.rs:50` | `src/ast/mod.rs:26`; `src/lib.rs:967,1102,1204,1267,2173`; `src/scheduler.rs:6,292-293`; `src/prelude.rs:671` | Used throughout rule lifecycle. Becomes neutral handle. |
| `ExternalFunctionId` | `core-relations` (re-exported via `core_relations::ExternalFunctionId`) | `src/lib.rs:46,1162,1205,1249,1269`; `src/typechecking.rs:60,187,785`; `src/prelude.rs:326,423,540,671`; `src/core.rs:67`; `src/sort/fn.rs:413` | Neutral handle; lives in `core-relations`, already in a backend-neutral crate |
| `FunctionRow<'a>` | `egglog-bridge/src/lib.rs:1408` (`{ vals: &'a [Value], subsumed: bool }`) | `pub use` at `src/lib.rs:54`; consumed in `src/extract.rs:21,120,345,383,420,467,637,687,887`; `src/serialize.rs` via `for_each_while` closure | Move into `egglog-backend-trait` as `BackendRow<'a>`, identical shape |
| `ColumnTy` | `egglog-bridge/src/lib.rs:45` (`enum { Id, Base(BaseValueId) }`) | `pub use` at `src/sort/mod.rs:13`; declared in `src/sort/mod.rs:55` (`Sort::column_ty`); used in `src/sort/fn.rs:147`; `src/prelude.rs:789,858`; `src/lib.rs:55,1198,1262`; `src/scheduler.rs:6,336,360` | Heavy leak; move into trait crate |
| `QueryEntry` | `egglog-bridge/src/rule.rs:58` (`enum { Var(Variable), Const { val, ty } }`) | `src/lib.rs:55,1987,2006,2013,2029,2066,2083`; `src/lib.rs::literal_to_entry` (`lib.rs:2178`); via `RuleBuilder` in `src/scheduler.rs` | Replaced when rule IR replaces callback API — Section 5 |
| `MergeFn` | `egglog-bridge/src/lib.rs:876` | `src/lib.rs:683,687-710,741,753,755`; `src/scheduler.rs:6,341` | Moves into trait crate. Note `MergeFn::Function`, `MergeFn::Primitive` refer to bridge `FunctionId`/`ExternalFunctionId` — those are already opaque |
| `DefaultVal` | `egglog-bridge/src/lib.rs:866` | `src/lib.rs:741,749,750`; `src/scheduler.rs:6,340` | Moves into trait crate |
| `FunctionConfig` | `egglog-bridge/src/lib.rs:109` | `src/lib.rs:742`; `src/scheduler.rs:6,338` | Moves into trait crate |
| `ExecutionState<'_>` | `core_relations` re-export at `src/lib.rs:45` | `src/lib.rs:113` (`apply` trait method), `src/sort/fn.rs:346,435,450`, `src/sort/multiset.rs:353,409`, `src/scheduler.rs:107,316`, `src/prelude.rs:323,442`, `src/typechecking.rs:181` | **Already lives in `core-relations`** — backend-neutral crate. The trait can require backends to provide a `&mut ExecutionState`. But duckdb has nothing to put in it without significant lift. See Section 4 |
| `TableAction` / `UnionAction` | `egglog-bridge::TableAction` / `UnionAction` | `src/lib.rs:1608,1981`; `src/prelude.rs:325,358,360,422,544,551`; `src/scheduler.rs:6,108,236`; `src/sort/fn.rs:412`; `src/sort/multiset.rs:2` | Both currently hold bridge-specific state. Move behind a `TableActionOps` trait per Phase 1.5 (or — recommended — replace with backend-level methods like `Backend::insert_rows` + `Backend::union`) |
| `RuleBuilder<'a>` | `egglog-bridge/src/rule.rs:122` | `src/lib.rs:1986,1994`; `src/scheduler.rs:348,367` | Replaced by rule IR in Section 5 |

### 1.3 `Sort::column_ty` per-impl behavior

Declared at **`src/sort/mod.rs:55`**:

```rust
fn column_ty(&self, backend: &egglog_bridge::EGraph) -> ColumnTy;
```

Implementations and their bodies:

| `impl Sort for …` | File:line | Returns | Uses `backend`? |
|---|---|---|---|
| `EqSort` (eq-sort wrapper for user datatypes) | `src/sort/mod.rs:171` | `ColumnTy::Id` | No |
| `FunctionSort` (the `(UnstableFn …)` sort) | `src/sort/fn.rs:147` | `ColumnTy::Id` | No |
| `BaseSortImpl<T: BaseSort>` (generic wrapper used by `i64`, `f64`, `bool`, `String`, `Unit`, `BigInt`, `BigRat`) | `src/prelude.rs:789` | `ColumnTy::Base(backend.base_values().get_ty::<T::Base>())` | **Yes** — looks up `BaseValueId` via the registered type-id |
| `ContainerSortImpl<T: ContainerSort>` (generic wrapper for `Vec`/`Set`/`Map`/`MultiSet`/`Pair`) | `src/prelude.rs:858` | `ColumnTy::Id` | No |

**Key finding**: only **one** of the four implementations actually reads from `backend`. That one (`BaseSortImpl`) only needs `BaseValues` to obtain the `BaseValueId` for `T::Base`. The signature can therefore be safely narrowed to `column_ty(&self, &dyn BaseValuePool) -> ColumnTy` *or* the broader `column_ty(&self, &dyn Backend) -> ColumnTy`. The plan picks `&dyn Backend`; either works.

> Per-sort concrete sorts (`src/sort/bigint.rs`, `bigrat.rs`, `bool.rs`, `f64.rs`, `i64.rs`, `map.rs`, `multiset.rs`, `pair.rs`, `set.rs`, `string.rs`, `unit.rs`, `vec.rs`) do not define `column_ty` themselves — they implement the inner `BaseSort` or `ContainerSort` trait and rely on the generic wrapper in `prelude.rs`.

---

## Section 2 — `Value` type semantics (Phase 0.2)

### 2.1 Definition

```rust
// core-relations/src/common.rs:103
define_id!(pub Value, u32, "A generic identifier representing an egglog value");
```

`Value` is a transparent `u32` newtype. Defined in `core-relations` (the backend-neutral crate), so it is already shareable.

It has a special sentinel `Value::stale()` returning `Value::new(u32::MAX)` (`core-relations/src/common.rs:106-108`).

### 2.2 Three coexisting meanings

| Meaning | Source | How produced |
|---|---|---|
| 1. **EqSort id** | Bridge's id counter (`CounterId` in `egglog_bridge::EGraph::id_counter`, `egglog-bridge/src/lib.rs:64`) | Allocated fresh by `DefaultVal::FreshId` or new-rule-side `lookup` |
| 2. **Base value (primitive)** | `BaseValues` (`core-relations/src/base_values/mod.rs:66`) | `BaseValues::get::<T>(T) -> Value` either inline-boxes (for `MAY_UNBOX` types) or interns into a typed `InternTable<P, Value>` |
| 3. **Container id** | `ContainerValues::register_val::<C>` (`core-relations/src/containers/mod.rs`) | Single `u32` namespace shared with eqsort ids |

For "may-unbox" base types (e.g. `i64`), the high bit of the `Value` signals whether the bytes are inlined (`Value::new(x as u32)`) or interned (`x + VAL_OFFSET = 1<<31`) — `core-relations/src/base_values/mod.rs:158` defines `VAL_OFFSET`.

### 2.3 Disambiguation via `ColumnTy`

The accompanying type tag lives in `ColumnTy` (`egglog-bridge/src/lib.rs:45`):

```rust
pub enum ColumnTy {
    Id,                    // eq-sort or container
    Base(BaseValueId),     // base value, with which base-value type
}
```

Inside the bridge, `get_canon_repr` (`egglog-bridge/src/lib.rs:221`) does the right thing per column:

```rust
pub fn get_canon_repr(&self, val: Value, ty: ColumnTy) -> Value {
    match ty {
        ColumnTy::Id => self.get_canon_in_uf(val),
        ColumnTy::Base(_) => val,
    }
}
```

Container values share the `ColumnTy::Id` variant — the disambiguation between "is this a UF id or a container id" is done by **which table** the column lives in (and the bridge's internal `container_values` data structure recognizes ids it registered).

### 2.4 `Value` unwrap / get call sites

Searching `unwrap::<T>` / `get::<T>` / `get_canon_in_uf` across `src/` and `src/proofs/`:

| Site | Op | Type |
|---|---|---|
| `src/lib.rs:1014` | `unwrap::<i64>` | i64 (next-id counter read) |
| `src/lib.rs:1016` | `get::<i64>` | i64 |
| `src/lib.rs:1029` | `get::<i64>` | i64 (timestamp) |
| `src/lib.rs:1155-1156` | `get_ty::<()>`, `get(())` | Unit |
| `src/lib.rs:1363` | `unwrap::<i64>` | i64 |
| `src/lib.rs:1557` | `get(())` | Unit |
| `src/lib.rs:1569` | `get(i)` (i64) | i64 |
| `src/lib.rs:1577-1578` | `get::<F>(Boxed<f64>)` | f64 |
| `src/lib.rs:1583` | `get::<S>(...)` | String |
| `src/lib.rs:1906,1911` | `unwrap::<T>` / `get::<T>` generic | Any |
| `src/lib.rs:2190-2194` | `get::<i64>`, `<F>`, `<S>`, `<bool>`, `<()>` | Literal-to-value path |
| `src/lib.rs:2290-2294` | `unwrap::<i64>` / `get::<i64>` (3 sites) | i64 (in a primitive `apply`) |
| `src/prelude.rs:332` | `unwrap::<T>` | Generic via `RustRuleContext` |
| `src/prelude.rs:347` | `get::<T>` | Generic via `RustRuleContext` |
| `src/prelude.rs:450,968,969,1018,1110` | various i64/Unit | Test fixtures |
| `src/scheduler.rs:111,318,325-326` | `get_ty::<()>`, `get(())` | Unit |
| `src/sort/bool.rs:42` | `unwrap::<bool>` | bool |
| `src/sort/bigint.rs:58` | `unwrap::<Z>` | BigInt |
| `src/sort/bigrat.rs:114` | `unwrap::<Q>` | BigRat |
| `src/sort/f64.rs:52` | `unwrap::<F>` | f64 |
| `src/sort/i64.rs:75` | `unwrap::<i64>` | i64 |
| `src/sort/string.rs:44` | `unwrap::<S>` | String |
| `src/sort/fn.rs:352` | `unwrap::<ResolvedFunction>` | FunctionContainer base |
| `src/sort/vec.rs:153` | `get::<i64>` | i64 (vec index seed) |

**`get_canon_in_uf` direct usage**: zero in `src/` — that's a bridge-internal routine. The frontend always goes through `get_canon_repr(val, ty)` indirectly via `for_each` rows (already canonicalized) or via `lookup_id`. *Verified by grep.*

### 2.5 Feasibility of keeping `Value = u32`

**Plan's recommendation**: keep `Value` as the shared type, optionally widen to `u64`.

**Verification**:

- DuckDB eqsort sequence: `CREATE SEQUENCE __egglog_eqsort_seq START 1` (`egglog-bridge-duckdb/src/lib.rs:658`), allocated via `SELECT nextval('__egglog_eqsort_seq')` (`egglog-bridge-duckdb/src/lib.rs:938`, `compile.rs:578-580`). The sequence is `BIGINT` in DuckDB but in practice never exceeds `2^31 - 1`; allocation is one per fresh constructor row, capped by `--max-iterations`. **`u32` is sufficient for eq-sort ids.**
- `i64` literals: a program with `(let x 4_000_000_000)` produces a `Literal::Int(4_000_000_000_i64)`. With `MAY_UNBOX` and `try_box` (`core-relations/src/base_values/unboxed.rs:78-86`), if the top bit is clear (value fits in 31 bits) it boxes inline; otherwise it falls through to an intern table on the bridge side. **The intern-table fallback already handles oversize i64.** `u32` is therefore sufficient *if* DuckDB also implements an i64 intern table.
- Alternative: widen `Value` to `u64`. This eliminates the intern-table fallback for `u64`/`i64`/`usize`/`isize` and gives the DuckDB backend a direct `BIGINT ↔ Value` identity. **Recommended preflight commit** (Phase 2.C1 in the plan).

**Conclusion**: `Value` remains the shared currency. Widening to `u64` is a small, isolated change limited to `core-relations`; either widen or rely on the existing intern-table fallback. **No fundamental blocker.**

---

## Section 3 — `BaseValues` model (Phase 0.3, containers scope-cut)

### 3.1 `BaseValues` shape

```rust
// core-relations/src/base_values/mod.rs:66
pub struct BaseValues {
    type_ids: HashMap<TypeId, BaseValueId>,
    tables:   DenseIdMap<BaseValueId, Box<dyn DynamicInternTable>>,
}
```

Public surface (`core-relations/src/base_values/mod.rs:71-123`):

- `register_type<P: BaseValue>() -> BaseValueId` — inserts `P` into the registry, allocates a typed `BaseInternTable<P>`
- `get_ty<P: BaseValue>() -> BaseValueId` — id lookup
- `get_ty_by_id(TypeId) -> BaseValueId` — runtime-id lookup
- `get<P: BaseValue>(p: P) -> Value` — intern (or inline-box if `P::MAY_UNBOX`)
- `unwrap<P: BaseValue>(v: Value) -> P` — extract (or unbox)

Internal: `DynamicInternTable: Any + DynClone + Send + Sync` (`core-relations/src/base_values/mod.rs:126`); each concrete table is `BaseInternTable<P> = InternTable<P, Value>`.

### 3.2 Registered `BaseValue` impls

Generated from `grep -rn "impl BaseValue for"` excluding `target/` and `egglog-bridge-duckdb/`:

| Type | File:line | `MAY_UNBOX` | Encoding |
|---|---|---|---|
| `u8`, `u16`, `u32`, `i8`, `i16`, `i32` | `core-relations/src/base_values/unboxed.rs:31` (macro) | true | Inline cast (`Value::new(x as u32)`) |
| `bool` | `core-relations/src/base_values/unboxed.rs:33` | true | `Value::new(1)` for true, `Value::new(0)` for false |
| `()` | `core-relations/src/base_values/unboxed.rs:54` | true | Always `Value::new(0)` |
| `u64`, `i64`, `usize`, `isize` | `core-relations/src/base_values/unboxed.rs:105` (macro) | true | Inline iff top bit clear (`x & VAL_MASK == 0`); otherwise intern with `+ VAL_OFFSET` |
| `String` | `core-relations/src/base_values/mod.rs:44` | false | Pure intern table |
| `&'static str` | `core-relations/src/base_values/mod.rs:45` | false | Pure intern table |
| `num::Rational64` | `core-relations/src/base_values/mod.rs:46` | false | Pure intern table |
| `ResolvedFunction` (the `(UnstableFn …)` representation) | `src/sort/fn.rs:408` | false | Pure intern table |

User-defined sorts also implement `BaseValue` via `src/sort/bigint.rs`, `bigrat.rs`, etc. — checked: those use a private `Z` / `Q` newtype + `impl BaseSortInner`; the actual `impl BaseValue` likely lives there too. (Not explored in detail; the per-type list above is the canonical core-relations set.)

### 3.3 Proposed `BaseValuePool` trait sketch

The plan proposes splitting `BaseValuePool` out of `Backend` so `Backend` stays `dyn`-compatible. The generic-per-T API of `BaseValues` needs Any-dispatch on the trait:

```rust
pub trait BaseValuePool: Send + Sync {
    fn register_type_dyn(&mut self, type_id: TypeId, factory: ...) -> BaseValueId;
    fn get_ty_by_type_id(&self, ty: TypeId) -> BaseValueId;
    fn get_dyn(&self, ty: BaseValueId, raw: &dyn Any) -> Value;
    fn unwrap_dyn(&self, ty: BaseValueId, val: Value) -> Box<dyn Any>;
}
// Then a free helper layer for the generic-T sugar:
fn pool_get<T: BaseValue>(p: &dyn BaseValuePool, x: T) -> Value { ... }
fn pool_unwrap<T: BaseValue>(p: &dyn BaseValuePool, v: Value) -> T { ... }
```

**Verification this works**:

- `BaseValues` already uses `Box<dyn DynamicInternTable>` internally (`core-relations/src/base_values/mod.rs:68`). The Any-based dispatch is already there.
- The `MAY_UNBOX` fast-path can stay generic-over-T (the inline-box check is purely `T::try_box(&self) -> Option<Value>`, no pool touched) and only the intern-fallback path needs dyn dispatch. **Achievable.**

**Bridge impl**: forwards to the existing `BaseValues`. **DuckDB impl**: builds its own `BaseValues`-shaped structure (or reuses `core-relations::BaseValues` directly, since that crate is backend-neutral). The plan recommends re-using `BaseValues` directly on DuckDB — confirmed feasible.

### 3.4 Containers (out of scope)

Per the updated scope: **containers are NOT implemented on the DuckDB backend.** Reference: `core-relations/src/containers/mod.rs:70` (`pub struct ContainerValues`) — future reference only.

Container call sites in `src/` that need a "containers not supported on duckdb" error path (or to be silently skipped on the duckdb branch):

| Site | Operation | Trigger |
|---|---|---|
| `src/lib.rs:1920-1925` | `EGraph::value_to_container<T>` | Public API method; users calling on duckdb need a runtime error |
| `src/lib.rs:1927-1932` | `EGraph::container_to_value<T>` | Same — runtime error |
| `src/serialize.rs:347-365` | `container_values()` + `inner_values` during serialize | Skip serialize-of-container-cells if container sorts not present (already an option) |
| `src/extract.rs:324` | container expansion during cycle detection | Returns empty if no containers registered — naturally degrades |
| `src/extract.rs:364` | container expansion in `eclass_for` | Same |
| `src/extract.rs:527` | container expansion in `extract_term` | Same |
| `src/extract.rs:534-535` | `reconstruct_termdag_container` | Naturally unused if no container values exist |
| `src/sort/vec.rs:250,255,272,304,309` | Vec primitives' `apply` | Vec sort registration errors before primitives run on duckdb |
| `src/sort/multiset.rs:355,360,379,411` | MultiSet primitives' `apply` | Same |
| `src/sort/map.rs`, `set.rs`, `pair.rs`, `fn.rs` | similar primitives | Same |
| `src/lib.rs:1929` (`with_execution_state` for `register_val`) | `container_to_value` body | DuckDB returns error |

**Trigger point** for the error: sort registration. The duckdb backend's `Backend::register_sort` (or equivalent) returns `Err("container sorts not supported on duckdb backend")` when it sees `ContainerSortImpl<T>`. After that, `container_values()` is never called (the user gets the error before any rule runs). Pair sort (proof mode) is the kept exception.

---

## Section 4 — `with_execution_state` analysis (Phase 0.4)

### 4.1 The 4 top-level callers

| # | File:line | What it captures | What it stages | Replacement trait method |
|---|---|---|---|---|
| 1 | `src/lib.rs:1611` | `&mut ExecutionState es` | Per-row: `table_action.insert(es, row.iter().copied())` for a parsed-file's worth of facts | `Backend::insert_rows(FunctionId, &[Vec<Value>])` |
| 2 | `src/lib.rs:1618` | `&mut ExecutionState es` | Per-row: `table_action.lookup(es, row)` (for constructor inserts that may allocate fresh ids) | `Backend::lookup_or_insert_rows(FunctionId, &[Vec<Value>])` (returns the allocated/looked-up output values; or split into two methods) |
| 3 | `src/lib.rs:1929` | `&mut ExecutionState state` | `container_values.register_val::<T>(x, state)` returning a fresh container `Value` | **N/A on duckdb.** On bridge: `Backend::register_container<C: ContainerValue>(c: C) -> Value`. DuckDB impl returns `Err`. |
| 4 | `src/scheduler.rs:225` | `&mut ExecutionState state` | Per matched rule: filter matches via the scheduler, then for each kept match call `table_action.insert(state, ...)` on the "decided" match table | `Backend::stage_match_insertions(FunctionId, &[Vec<Value>])` — same as #1 in effect (multi-row staged insert); the surrounding loop and `state` capture are an artifact of the bridge's transaction model |

### 4.2 Why `FnOnce(&mut ExecutionState) -> R`, not `&mut ExecutionState`

`ExecutionState<'_>` borrows from `Database` for the lifetime of the callback (`egglog-bridge/src/lib.rs:807-808` delegates to `Database::with_execution_state`). The borrow checker forbids handing back a long-lived `&mut`. The closure-based API stages many updates then drops, releasing the borrow.

### 4.3 Proposed trait method signatures

The plan's recommendation is to remove `with_execution_state` from the trait surface entirely. The 4 sites become:

```rust
// Sites #1 and #4 — bulk insertion (no ID allocation needed; lookup-on-existing-key)
fn insert_rows(&mut self, table: FunctionId, rows: &[Vec<Value>]);

// Site #2 — constructor lookup-or-insert (returns the looked-up / freshly-allocated output)
// On the bridge this maps to TableAction::lookup; on DuckDB to an INSERT with RETURNING or a
// SELECT-then-INSERT with sequence allocation.
fn lookup_or_insert(&mut self, table: FunctionId, key: &[Value]) -> Value;
// (Or, batched: fn lookup_or_insert_rows(&mut self, table: FunctionId, keys: &[Vec<Value>]) -> Vec<Value>;)

// Site #3 — container registration. Bridge-only; DuckDB returns Err.
fn register_container<C: ContainerValue>(&mut self, c: C) -> Result<Value>;
// Because this is generic-over-C and dyn Backend can't take generic methods, this becomes a method
// on BaseValuePool/ContainerPool sub-trait — or a runtime-dispatched dyn variant.
```

### 4.4 Container-side specifics (N/A on duckdb)

Site #3 (`src/lib.rs:1929`) registers a Rust container value (e.g. a `Vec<Value>` for `Vec` sort) and obtains a `Value` id. Bridge: maintains an intern table per container type; reuses the id if the value has been seen before. DuckDB: doesn't apply because users can't declare container sorts. The trait method exists in the trait but DuckDB impl returns `Err("containers not supported on duckdb")`.

### 4.5 What about `ExecutionState` inside user primitives?

Separate from the 4 top-level sites, **`ExecutionState` is also threaded into every `ExternalFunction::apply(&self, &mut ExecutionState, &[Value]) -> Option<Value>`** (`src/lib.rs:113`, `src/sort/fn.rs:346,435,450`, `src/scheduler.rs:107,316`, `src/prelude.rs:323,442`, `src/typechecking.rs:181`). This is the primitive-VM bridge: a primitive's body may call `state.base_values().get(...)`, `state.container_values()...`, or `state.call_external_func(...)`.

For DuckDB, primitives run inside DuckDB's SQL via a VScalar UDF, and they cannot synchronously call back into DuckDB. Per the plan's recommendation (Phase 1.5α), `Backend::supports_inline_table_lookups() -> bool` is a flag; `rust_rule` / `query` are runtime errors when backend is DuckDB. **Out of scope for this inventory** — flagged as a known incompatibility.

---

## Section 5 — Rule compilation IR (Phase 0.5)

### 5.1 Bridge IR (callback-based)

Key types in `egglog-bridge/src/rule.rs`:

```rust
// rule.rs:38
pub struct Variable { pub id: VariableId, pub name: Option<Box<str>> }

// rule.rs:58
pub enum QueryEntry {
    Var(Variable),
    Const { val: Value, ty: ColumnTy },
}

// rule.rs:75
pub enum Function {
    Table(FunctionId),
    Prim(ExternalFunctionId),
}

// rule.rs:101 — opaque, callback-based
struct Query {
    uf_table: TableId,
    id_counter: CounterId,
    ts_counter: CounterId,
    rule_id: RuleId,
    vars: DenseIdMap<VariableId, VarInfo>,
    atoms: Vec<(TableId, Vec<QueryEntry>, SchemaMath)>,
    add_rule: Vec<BuildRuleCallback>,  // closure chain
    sole_focus: Option<usize>,
    seminaive: bool,
    plan_strategy: PlanStrategy,
}

// rule.rs:122
pub struct RuleBuilder<'a> { egraph: &'a mut EGraph, desc: Arc<str>, query: Query }
```

`RuleBuilder` methods (rule.rs:160-688): `new_var`, `new_var_named`, `query_table`, `query_prim`, `call_external_func`, `subsume`, `lookup`, `union`, `set`, `remove`, `panic`, `build`.

**Semantics**: each builder call appends a callback to `query.add_rule`. On `build()`, the chain is materialized into a `core_relations::CachedPlan`.

### 5.2 DuckDB IR (data-driven)

```rust
// egglog-bridge-duckdb/src/lib.rs:387
pub enum Term {
    Var(String),
    Lit(Literal),
    Prim(String, Vec<Term>),
    FuncCall(String, Vec<Term>),
}

// lib.rs:403
pub enum Atom {
    Func   { name: String, args: Vec<Term> },        // function-table body atom
    Filter (Term),                                    // pure predicate
    Bind   { var: String, expr: Term },               // (= var (primitive ...)) pattern
}

// lib.rs:421
pub enum Action {
    Insert  { name: String, args: Vec<Term> },
    Delete  { name: String, key_args: Vec<Term> },
    LetCtor { var: String, name: String, args: Vec<Term> },
    LetExpr { var: String, expr: Term },
    Panic   { msg: String },
}

// lib.rs:454
pub struct Rule {
    pub name: String,
    pub ruleset: String,
    pub body: Vec<Atom>,
    pub actions: Vec<Action>,
}
```

**Bridge-only operations** (no DuckDB equivalent today):

- `subsume` action (no `subsumed` column on DuckDB rows; degraded silently to a no-op or error)
- `MergeFn::Function` and `MergeFn::Primitive` for complex merges (DuckDB has `Custom` merge expressions via `MergeMode` — partly supported, not 1:1)
- `union` action — DuckDB has it indirectly via native-UF; semantics differ subtly

**DuckDB-only operations**: none above the line; native-UF call lowering is an internal optimization, not an IR construct.

### 5.3 Proposed `RuleIr`

A neutral IR that both backends compile from. The plan's draft (refined here based on what the existing IRs require):

```rust
// In egglog-backend-trait crate
pub struct RuleIr {
    pub name: String,
    pub ruleset: String,        // empty string = default
    pub seminaive: bool,
    pub vars: DenseIdMap<VarId, VarInfo>,  // id -> (ty, optional name)
    pub atoms: Vec<RuleAtom>,
    pub actions: Vec<RuleAction>,
}

pub struct VarInfo {
    pub ty: ColumnTy,
    pub name: Option<String>,
}

pub enum RuleAtom {
    /// Body atom against a function/relation table.
    Table {
        func: FunctionId,
        args: Vec<RuleEntry>,
        /// None = match either; Some(false) = only un-subsumed
        is_subsumed: Option<bool>,
    },
    /// Body atom that is a primitive constraint or computation.
    Prim {
        func: ExternalFunctionId,
        args: Vec<RuleEntry>,
        ret_ty: ColumnTy,
    },
    /// Body-level `(= var (primitive ...))` extension. Maps to DuckDB's `Atom::Bind`.
    Bind {
        var: VarId,
        source: BindSource,
    },
    /// Pure filter (e.g. `(< x 5)`, `(!= a b)`).
    Filter {
        prim: ExternalFunctionId,
        args: Vec<RuleEntry>,
    },
}

pub enum BindSource {
    Prim   { func: ExternalFunctionId, args: Vec<RuleEntry>, ret_ty: ColumnTy },
    Entry  (RuleEntry),
}

pub enum RuleEntry {
    Var   (VarId),
    Const { val: Value, ty: ColumnTy },
}

pub enum RuleAction {
    Let {
        var: VarId,
        source: LetSource,
    },
    /// `(set (f a b) c)` — set the function output.
    Set {
        func: FunctionId,
        args: Vec<RuleEntry>,   // includes output as last arg
    },
    /// Insert a row.
    Insert {
        func: FunctionId,
        args: Vec<RuleEntry>,
    },
    /// `(subsume (f a b))` — bridge-only. DuckDB impl: error or no-op (degrade).
    Subsume {
        func: FunctionId,
        key:  Vec<RuleEntry>,
    },
    /// `(delete (f a b))` action.
    Remove {
        func: FunctionId,
        key:  Vec<RuleEntry>,
    },
    /// `(union x y)`.
    Union {
        l: RuleEntry,
        r: RuleEntry,
    },
    /// `(panic msg)`.
    Panic {
        msg: String,
    },
}

pub enum LetSource {
    /// `(let v (f a b))` — function call, panic if not present.
    Lookup {
        func: FunctionId,
        args: Vec<RuleEntry>,
        panic_msg: String,
    },
    /// `(let v (prim a b))`.
    CallPrim {
        func: ExternalFunctionId,
        args: Vec<RuleEntry>,
        ret_ty: ColumnTy,
    },
    /// `(let v x)` — alias.
    Entry (RuleEntry),
}
```

### 5.4 Per-construct support matrix

| Construct | Bridge | DuckDB today | Notes |
|---|---|---|---|
| `RuleAtom::Table { is_subsumed: None }` | ✅ | ✅ | |
| `RuleAtom::Table { is_subsumed: Some(false) }` | ✅ | ⚠ | DuckDB has no `subsumed` column; degrades to `None` semantics. Document. |
| `RuleAtom::Prim` | ✅ | ✅ | |
| `RuleAtom::Bind` | ✅ (via `lookup` + new_var) | ✅ | DuckDB has `Atom::Bind` |
| `RuleAtom::Filter` | ✅ (via `query_prim` with bool result) | ✅ (via `Atom::Filter`) | |
| `RuleAction::Let::Lookup` | ✅ | ✅ (via `LetCtor` or expression in materialized match table) | |
| `RuleAction::Let::CallPrim` | ✅ | ✅ (via `LetExpr`) | |
| `RuleAction::Set` | ✅ | ⚠ | DuckDB does `Insert` with merge-mode; semantics depend on `FunctionInfo::merge` |
| `RuleAction::Insert` | ✅ | ✅ | |
| `RuleAction::Subsume` | ✅ | ❌ | DuckDB error or degrade |
| `RuleAction::Remove` | ✅ | ✅ (`Action::Delete`) | |
| `RuleAction::Union` | ✅ | ✅ (via native-UF when enabled, else via congruence) | |
| `RuleAction::Panic` | ✅ | ✅ | |

### 5.5 Minimal IR covering the test suite — recommendation

The current `tests/*.egg` suite covers virtually all egglog constructs, but only a few "stress" the corners (e.g. complex merge functions live in `complex-merge-prim.egg`, `complex-merge-func.egg`; subsumption lives in `subsume.egg`, `subsume-relation.egg`). The plan suggests instrumenting `add_rule` to dump rule shapes; that's a useful future Phase-0 follow-on but is not strictly needed for this inventory.

**The IR sketched in 5.3 is sufficient for the bridge's full surface.** For DuckDB, the unsupported constructs (`Subsume`, complex `MergeFn::Function`/`Primitive`) are caught at IR translation time and surface as clean errors.

---

## Section 6 — Test impact preview (Phase 0.6)

The refactor's Commit 14 flips `tests/files.rs` so DuckDB runs go through the unified pipeline. Predicting newly-failing tests:

### 6.1 `(extract …)` tests

**17 test files** call `(extract …)`:

```
container-rebuild.egg, extract-vec-bench.egg, fibonacci-demand.egg,
intersection.egg, hardboiled_conv1d_32.egg, hardboiled_conv1d_128.egg,
proof-extract-cost.egg, python_array_optimize.egg, factoring-multisets.egg,
repro-typecheck-term-encoding.egg, interval.egg, map.egg,
repro-738-fn-sort.egg, uf-extraction.egg, taylor51.egg,
stresstest_large_expr.egg, use-at-in-string.egg
```

**Current behavior** (`tests/files.rs:42-48`):

```rust
// Skip extraction outputs. --duckdb mode silently
// ignores `(extract …)` commands (no extraction
// pipeline yet), so this keeps the shared snapshot
// comparable across all backends.
CommandOutput::ExtractBest(..) => None,
CommandOutput::ExtractVariants(..) => None,
```

The shared snapshot strips extract output entirely. Once the unified pipeline lights up, **extract output will be re-included** and snapshots may diverge in two ways:

1. **Cost ties broken differently**: when multiple equivalent terms exist with the same cost, the DuckDB pipeline may pick a different one (depends on row iteration order). Many of these files have stable single-best extract output, but `container-rebuild.egg`, `repro-738-fn-sort.egg`, and `eggcc-extraction.egg` (cross-referenced from `shared_snapshot_eggcc_extraction.snap`) are likely tie-prone.
2. **Container sorts in extract**: `container-rebuild.egg`, `extract-vec-bench.egg`, `hardboiled_conv1d_{32,128}.egg`, `python_array_optimize.egg`, `factoring-multisets.egg`, `map.egg` all use container sorts. These files will be **excluded from the DuckDB pipeline** under the updated scope (see 6.3).

Existing snapshots that already show extract output (sanity check): `shared_snapshot_uf_extraction.snap` shows `((Bar 1) (Foo 1) (UF_Expr 2))` — that's `(print-size)` output, not extract. `shared_snapshot_eggcc_extraction.snap` and `shared_snapshot_proof_extract_cost.snap` may contain extract content; both come from container-sort files.

**Prediction**: 6–8 files (the non-container, extract-using ones — `fibonacci-demand.egg`, `intersection.egg`, `proof-extract-cost.egg` if non-container, `interval.egg`, `taylor51.egg`, `stresstest_large_expr.egg`, `use-at-in-string.egg`, `repro-typecheck-term-encoding.egg`) will need either snapshot updates or a re-confirmation that extract output is identical across backends. Strategy: run Commit 14, see what diverges, snapshot-bless or harden.

### 6.2 `(prove-exists …)` tests

**Zero test files** under `tests/` invoke `(prove-exists …)` directly (verified by `grep -rln "(prove-exists" tests/`). The construct is exercised via the `proof_testing` mode in the test harness (`tests/files.rs:489,496`) and via `tests/integration_test.rs:127` (`prove_exists_reports_query_mismatch`).

In `proof_testing` mode the harness auto-generates `(prove-exists …)` commands per the `desugar.rs` flow (`src/ast/desugar.rs:212-224`). The DuckDB path is gated on `file_supports_proofs` and the static skip list (`tests/files.rs:424-430`); proof-testing on DuckDB is not currently wired (per the harness). When wired, the proof-extraction body (`src/proofs/proof_extraction.rs:55`) uses `for_each_while` — straightforward to support on DuckDB if the function table can be iterated.

**Prediction**: `prove-exists` only enters the picture if/when proof-testing on DuckDB is enabled; that's already out of scope for the current `duckdb_proofs_supported` branch which does NOT enable `proof_testing` (`tests/files.rs:471-477`).

### 6.3 Container-sort tests (must skip on DuckDB)

Searched with `grep -rln "(Vec \|(Set \|(Map \|(MultiSet "`. **29 files** declare a container sort:

| File | Container kinds |
|---|---|
| `container-rebuild.egg` | Vec |
| `eggcc-extraction.egg` | Vec |
| `container-fail.egg` | Vec (already fail-test) |
| `hardboiled_conv1d_32.egg` | Vec |
| `extract-vec-bench.egg` | Vec |
| `eggcc-2mm.egg` | Set |
| `python_array_optimize.egg` | Vec (many) |
| `repro-new-backend-python-vec.egg` | Vec |
| `repro-665-set-union.egg` | Set |
| `factoring-multisets.egg` | MultiSet |
| `hardboiled_conv1d_128.egg` | Vec |
| `map.egg` | Map |
| `repro-vec-unequal.egg` | Vec |
| `tricky-type-checking.egg` | Map |
| `vec.egg` | Vec |
| `repro-querybug3.egg` | (container) |
| `stresstest_large_expr.egg` | (container) |
| `type-constraints-tests.egg` | (container) |
| `web-demo/typeinfer.egg` | Set |
| `web-demo/fusion.egg` | (container) |
| `web-demo/lambda.egg` | (container) |
| `web-demo/set.egg` | Set |
| `web-demo/datatypes.egg` | (container) |
| `web-demo/eqsat-basic-multiset.egg` | MultiSet |
| `web-demo/multiset.egg` | MultiSet |
| `fail-typecheck/repro-containers-disallowed.egg` | (fail test) |
| `fail-typecheck/ungrounded-3.egg` | (fail test) |
| `fail-typecheck/ungrounded-4.egg` | (fail test) |
| `tests/integration_test.rs` | uses container sorts inline |

**Recommended skip mechanism**: extend the existing scan loop in `tests/files.rs:436-440`. Today it scans for `"(push"`/`"(pop"`; add a regex scan for `(Vec `, `(Set `, `(Map `, `(MultiSet ` token boundaries. The 29 files become duckdb-skipped. Pair is fine (it's already supported and not in this list).

The existing duckdb static skip list (`tests/files.rs:424-430`) has 5 files; adding the container-sort skip adds ~29 more (some already in the skip list, e.g. `eggcc-2mm.egg`). Net DuckDB-supported file count after Commit 14: roughly **103 - 29 - 5 ≈ 70 files**, give or take.

### 6.4 Other potential failures at Commit 14

- **Files using `(push)/(pop)`**: already skipped (`tests/files.rs:437-440`), unaffected.
- **Files using `(rust-rule)`/`(query)`**: per Phase 1.5α, DuckDB-incompatible. These are not commonly used inside `tests/*.egg`; mostly exercised in `src/prelude.rs` unit tests.
- **Serialize tests**: `src/serialize.rs:347-365` touches `container_values()`. Serializing a container-free egraph is fine; serializing a container egraph triggers the sort-registration error before serialization.
- **Subsumption tests** (`subsume.egg`, `subsume-relation.egg`): already in the static skip list. Confirmed.

---

## Open questions for the next agent

1. **`Value` widening**: should the preflight commit widen `Value` to `u64`, or rely on `i64`'s intern-table fallback? Recommendation: widen to `u64`. Cost is a single grep-and-replace across `core-relations` + recompile of `egglog-bridge` and `egglog-bridge-duckdb`.
2. **`MergeFn::Function`/`Primitive` on DuckDB**: DuckDB already has a `MergeMode::Custom` path. Is it semantically equivalent to bridge's `MergeFn::Function`? Worth a focused investigation in Phase 2.C10 before committing.
3. **`Subsume` semantics on DuckDB**: the plan says "degrade to no-op + document". A safer alternative is to add a `subsumed BOOLEAN` column to DuckDB tables; cost is one column-per-function and a small WHERE-clause change. Decision deferred.
4. **`RustRuleRhs` / `prelude.rs::TableAction` in user primitives**: Phase 1.5α picks "DuckDB returns error". Verify no production-critical user-primitive path depends on this.
5. **Rule-shape instrumentation**: low priority, but a 30-line `add_rule` debug-dump would confirm the proposed `RuleIr` covers all real-world rules before committing the refactor. Recommended as Phase-0 follow-on if Phase 2.C5 (RuleIr emission) starts to grow.
6. **`for_each_while` lifetime on DuckDB**: the bridge passes a borrowed `&[Value]` inside `FunctionRow`. On DuckDB, each row needs a per-row buffer behind the borrow. Confirm the slice can outlive the callback's single invocation (it can — that's the standard cursor pattern); minor implementation detail.
