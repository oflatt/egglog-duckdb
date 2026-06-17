//! # egglog-bridge-feldera
//!
//! A Feldera/DBSP-backed executor for egglog's resolved IR, behind the
//! [`egglog_backend_trait::Backend`] interface.
//!
//! ## Milestone 1 — bounded per-iteration stepping
//!
//! - Register relations and *non-recursive* single-/double-join rules.
//! - Run them so that `(run 1)` and `(run 3)` produce **different, bounded**
//!   results — N rounds of extension, **NOT** full saturation in one call.
//! - Read results back through `for_each` / `lookup_id` / `table_size`.
//!
//! ## Milestone 2 — union-find + rebuild (the research crux)
//!
//! The backend now faithfully RUNS multi-ruleset term-encoded programs that do
//! unions + rebuild, matching the reference backend's per-function tuple counts
//! (see `tests/rebuild_proof.rs`). Three new pieces of machinery:
//!
//! - **Ruleset-scoped execution.** `run_rules(&[RuleId])` runs an arbitrary
//!   *subset* of rules. A DBSP circuit is static once built, so we build and
//!   cache one circuit per distinct sorted rule subset (keyed in
//!   [`EGraph::circuits`]). This lets the frontend schedule the term encoder's
//!   `(saturate single_parent)` / `(saturate path_compress)` /
//!   `(saturate uf_index)` rulesets in order.
//! - **Retraction-rebuild.** `(delete …)` / the trait `remove` becomes a
//!   negative-weight DBSP diff. The `@uf` rewrites (delete `a→b`, set `a→c`)
//!   are retraction + insertion; the host folds the integrated insert/delete
//!   diff streams into the mirror as `(old ∪ inserts) \ deletes`.
//! - **Merge recognition.** `:merge (ordering-min old new)` (the term encoder's
//!   `@uff` uf-index) is recognized DuckDB-style and implemented as a
//!   lattice-min upsert at fold time. `Old` / `New` / relation modes too.
//!
//! ## The per-iteration model (load-bearing)
//!
//! The brief's central correction: egglog's `(run N)` is N bounded rounds with
//! rebuild between them, so the backend must do **ONE egglog iteration per
//! `run_rules` call**, not saturate. We therefore compile a **non-recursive**
//! circuit (NO `recursive` scope): one `transaction()` is one round of rule
//! application over the *current* relation contents. The frontend's existing
//! loop drives `(run N)` / `(saturate R)` by calling `run_rules` repeatedly.
//! See `compile.rs` for the circuit shape and `run_rules` below for the
//! host-side feedback that takes the next hop.

use std::any::Any;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use egglog_backend_trait::{
    Backend, BaseValueId, BaseValuePool, ColumnTy, ContainerPool, ExternalFunction,
    ExternalFunctionId, FunctionConfig, FunctionId, FunctionRow, IterationReport, QueryEntry,
    ReportLevel, RuleBuilderOps, RuleId, Value,
};
use egglog_core_relations::Database;
use egglog_numeric_id::NumericId;
use hashbrown::{HashMap, HashSet};

mod base_values;
pub mod circuit_rebuild;
pub mod compile;
pub mod dbsp_join;
mod external_func;
mod interpret;
pub mod rebuild_circuit;
mod rule_builder;
mod uf;

use base_values::base_values_as_pool_mut;
use compile::{pack_row, row_col, unpack_row, MergeMode, Row, RuleIr};
use external_func::ExternalFuncRegistry;

// ---------------------------------------------------------------------------
// Container pool stub (milestone 1 has no containers; mirror DuckDB's stub)
// ---------------------------------------------------------------------------

/// Zero-sized [`ContainerPool`] stub. Milestone 1 does not support container
/// sorts (PLAN Phase 2+); all accessors are empty and registration errors.
pub(crate) struct FelderaContainerPool;

impl ContainerPool for FelderaContainerPool {
    fn has_container_type(&self, _type_id: std::any::TypeId) -> bool {
        false
    }
    fn enabled(&self) -> bool {
        false
    }
    fn get_dyn(&self, _ty: std::any::TypeId, _val: Value) -> Option<Box<dyn Any + Send + Sync>> {
        None
    }
    fn register_val_dyn(
        &mut self,
        _ty: std::any::TypeId,
        _value: Box<dyn Any + Send + Sync>,
    ) -> Result<Value> {
        Err(anyhow::anyhow!(
            "containers not supported on the Feldera backend (milestone 1)"
        ))
    }
    fn for_each_dyn(&self, _ty: std::any::TypeId, _f: &mut dyn FnMut(Value, &dyn Any)) {}
    fn size(&self, _ty: std::any::TypeId) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Shared primitive engine (Stage C)
// ---------------------------------------------------------------------------

/// A shareable handle to a [`Database`] used purely as the primitive engine, so
/// that the `Send + 'static` DBSP circuit closures can evaluate value-computing
/// body primitives ON-CIRCUIT (Stage C of #23), instead of forcing a host
/// nested-loop fallback.
///
/// The handle wraps `Arc<Mutex<Database>>`. The wrapped `Database` is a **clone**
/// of [`EGraph::db`]; crucially, a cloned `Database` *shares* its base-value
/// intern tables (`InternTable` clones the inner `Arc<ConcurrentVec>` /
/// `Arc<Mutex<HashTable>>`), so a pure primitive evaluated through this clone
/// interns its result into the SAME tables the host `Value`s came from — the
/// returned handle is bit-for-bit identical to a host evaluation. Pure prims are
/// stateless beyond interning, so re-evaluating them on-circuit is idempotent.
///
/// The Feldera backend is single-threaded (DBSP runs one worker synchronously
/// during `step()`), so the `Mutex` is uncontended and cannot deadlock: the host
/// never holds the lock while the circuit steps (it locks, evaluates, unlocks).
/// The `unsafe impl Send + Sync` mirrors `EGraph`'s own assertion (the egraph is
/// only ever driven from a single thread); a `Database`'s `Rc`-free internals are
/// `Send`-safe under that single-thread invariant.
#[derive(Clone)]
pub struct PrimEngine(Arc<Mutex<Database>>);

// SAFETY: see `EGraph`'s `unsafe impl Send/Sync` below — the egraph (and hence
// this handle) is only ever used from a single thread; DBSP runs its worker
// synchronously on that same thread during `step()`.
unsafe impl Send for PrimEngine {}
unsafe impl Sync for PrimEngine {}

impl PrimEngine {
    /// Wrap a `Database` as a shared primitive-engine handle.
    pub(crate) fn new(db: Database) -> Self {
        PrimEngine(Arc::new(Mutex::new(db)))
    }

    /// Evaluate primitive `id` on `args` through the shared engine, returning
    /// the interned result handle (or `None` if the prim "fails", e.g. `!=` of
    /// equal args — which prunes the match). Used by the on-circuit call-prim
    /// node; reuses the exact host eval path (`call_external_func`) so prim
    /// semantics are never reimplemented.
    pub(crate) fn eval(&self, id: ExternalFunctionId, args: &[Value]) -> Option<Value> {
        self.0
            .lock()
            .unwrap()
            .with_execution_state(|st| st.call_external_func(id, args))
    }
}

// ---------------------------------------------------------------------------
// Relation metadata
// ---------------------------------------------------------------------------

/// What we remember about each registered relation/function.
struct RelationInfo {
    name: String,
    /// Number of columns (including the output column for functions).
    pub(crate) arity: usize,
    /// The column types (one per column), used by `dbsp_join::plan_join` to
    /// type-gate prim inlining: rep-arithmetic (`ordering-min/max`, `<`) is only
    /// valid on `ColumnTy::Id` columns (rep IS the union-find id) and on bool
    /// columns; on other base values the rep is an intern handle whose order
    /// differs from the logical value.
    pub(crate) schema: Vec<ColumnTy>,
    /// True for functions/constructors that have an output column.
    has_output: bool,
    /// How functional-dependency conflicts are resolved at flush time. For a
    /// function the key is the input columns (`arity - 1`) and the output column
    /// is resolved per this mode; for a relation it is [`MergeMode::Relation`]
    /// (whole row is the key, nothing to resolve).
    merge: MergeMode,
    /// True iff this function uses identity-on-miss lookup semantics
    /// (`DefaultVal::Identity`): an action-position lookup of an absent key
    /// resolves to the key itself, with no row inserted. Used by the
    /// canonicalize-at-creation encoding for the flat UF-index `@UF_Sf`.
    /// Only valid for a single-key function whose key and output share a type.
    identity_on_miss: bool,
}

// ---------------------------------------------------------------------------
// EGraph
// ---------------------------------------------------------------------------

/// The Feldera/DBSP-backed egraph.
pub struct EGraph {
    relations: Vec<RelationInfo>,
    /// Rule slots; `None` = freed.
    pub(crate) rules: Vec<Option<RuleIr>>,
    /// Rust-side materialized mirror: the accumulated contents of each
    /// relation, kept in sync with the circuit's integrated output after each
    /// transaction. This is what `for_each` / `lookup_id` / `table_size` read.
    ///
    /// Each function's row set is held behind an [`std::rc::Rc`] so that the
    /// per-iteration start-of-call *read snapshot* (`interpret::run_iteration`)
    /// can be taken in O(#functions) by cloning the `Rc` handles instead of
    /// deep-cloning every row. Mutations go through `Rc::make_mut`, which
    /// copy-on-writes only the (few) functions actually changed this call while
    /// the snapshot is alive — turning the old O(state) per-call read clone into
    /// O(changed-state).
    pub(crate) mirror: HashMap<FunctionId, std::rc::Rc<HashSet<Row>>>,
    /// A core-relations [`Database`] used purely as the **base-value /
    /// primitive engine**. It owns the [`egglog_core_relations::BaseValues`]
    /// registry (so `Value`s are bit-for-bit identical to the reference
    /// backend) AND the registered external functions. The Feldera frontend
    /// path (Milestone 3) needs primitives to actually *evaluate* — the default
    /// scheduler splits every user rule into a query rule whose head is a
    /// `call_external_func(collect_matches, …)`, so primitive invocation is
    /// mandatory. We invoke primitives host-side via
    /// [`Database::with_execution_state`], not inside the DBSP circuit (DBSP map
    /// closures are `Send + 'static` and cannot borrow the database).
    db: Database,
    /// Shared primitive-engine handle (Stage C of #23), built lazily the first
    /// time a persistent circuit needs to evaluate a value-computing body prim
    /// on-circuit. A clone of [`EGraph::db`] taken after all primitives are
    /// registered; shares interning with `db` so on-circuit prim results are
    /// bit-identical to host evaluation. See [`PrimEngine`]. `None` until first
    /// use; reset to `None` whenever a primitive is (de)registered so the next
    /// build picks up a fresh clone with the current prim table.
    prim_engine: Option<PrimEngine>,
    /// Names of PURE primitives, keyed by [`ExternalFunctionId`] rep. Recorded
    /// by the frontend (`set_pure_prim_name`) so `dbsp_join::plan_join` knows a
    /// body prim is pure (safe to re-evaluate on-circuit) and can lower it to a
    /// call-prim node. Impure/IO prims are absent here and stay ineligible.
    pub(crate) pure_prim_names: HashMap<u32, String>,
    container_pool: FelderaContainerPool,
    external_funcs: ExternalFuncRegistry,
    /// Monotonic fresh-id counter for `fresh_id` / `add_term`.
    next_id: u32,
    report_level: ReportLevel,
    /// Diagnostics: how many rule firings have run their body join on DBSP
    /// (the M4 dbsp-join path) vs. the host interpreter fallback. Used to
    /// characterize the frontier honestly (see `dbsp_join_stats`).
    pub(crate) dbsp_rule_runs: u64,
    pub(crate) host_rule_runs: u64,
    /// Stage-3 (migration) flag: when set (env `FELDERA_CIRCUIT_REBUILD=1`),
    /// rebuild rulesets are routed onto the persistent congruence-shuttle
    /// circuit (`rebuild_circuit`) instead of the host interpreter. Off by
    /// default — the interpreter stays the oracle.
    pub(crate) circuit_rebuild: bool,
    /// Diagnostics: number of `run_rules` calls served by the circuit-rebuild
    /// path vs. the interpreter, when the flag is on.
    pub(crate) circuit_rebuild_runs: u64,
    /// Persistent rebuild-circuit cache (Stage 1 persistence for Stage 3): the
    /// congruence-shuttle circuit is built ONCE and fed only per-call DELTAS of
    /// the raw view rows / union edges across `run_rules` calls, so rebuild cost
    /// is O(delta) not O(state). `None` until the first circuit-rebuild call.
    pub(crate) rebuild_cache: Option<circuit_rebuild::RebuildCache>,
    /// The persistent per-rule DBSP join circuit ([`dbsp_join::PersistentJoin`]),
    /// built lazily per rule. This is the ONLY body-join path (#23 Stage C
    /// complete): every atom-bearing rule runs its join on its persistent circuit
    /// fed signed deltas; the circuit's integral does the seminaive bookkeeping
    /// and handles retraction natively (signed weights), uniformly for user AND
    /// `@uf` rules — no recognition. The host nested-loop + `seen` fallback was
    /// retired; a genuinely-ineligible (impure) rule panics.
    pub(crate) persistent: HashMap<usize, dbsp_join::PersistentJoin>,
    /// FUSED per-RULESET DBSP join circuit, keyed by the sorted live rule-index
    /// list of the `run_rules` call. Collapses the ruleset's R per-rule
    /// transactions into ONE transaction per call (the dominant fixed
    /// per-transaction clocking cost — see [`dbsp_join::FusedJoin`]). The
    /// per-rule [`EGraph::persistent`] map is retained only for the unit test and
    /// the atom-less path; the fused circuit is the production join path.
    pub(crate) fused: HashMap<Vec<usize>, dbsp_join::FusedJoin>,
    /// Per-FUSED-circuit, per-body-relation last-fed row set (keyed by the same
    /// sorted rule-index list as [`EGraph::fused`]). The fused circuit shares one
    /// input per relation across all its rules, so the fed view is per-circuit,
    /// not per-rule. Same `Rc`-snapshot fast-path as [`EGraph::fed`].
    pub(crate) fed_fused: HashMap<Vec<usize>, HashMap<FunctionId, std::rc::Rc<HashSet<Row>>>>,
    /// Rule indices for ATOM-LESS rules (`(rule () …)` / `eval_actions` /
    /// `eval_resolved_expr`) that have already fired their single unconditional
    /// binding. An atom-less rule has no body relation to drive a join, so its
    /// (trivially-satisfied) empty binding fires exactly once — tracked here so it
    /// never re-fires on subsequent iterations (the seminaive "seen" for the lone
    /// empty binding). Handled on the persistent path WITHOUT a nested-loop.
    pub(crate) atomless_fired: HashSet<usize>,
    /// Per-rule, per-body-relation last-fed row set: each `run_rules` pushes only
    /// the `+/-` diff vs the start-of-call read view into that rule's persistent
    /// circuit, keeping its integral equal to the read view.
    ///
    /// Stored as the start-of-call `Rc` snapshot handle, so the per-rule fed-diff
    /// can `Rc::ptr_eq` the new read view against it: an unchanged function shares
    /// the same `Rc` and is skipped in O(1) instead of an O(state) set diff.
    pub(crate) fed: HashMap<usize, HashMap<FunctionId, std::rc::Rc<HashSet<Row>>>>,
    /// Temporary profiling accumulators (gated by env `FELDERA_PROFILE`).
    pub(crate) prof_read_clone: std::time::Duration,
    pub(crate) prof_read_rows: u64,
    pub(crate) prof_fed_diff: std::time::Duration,
    pub(crate) prof_circuit_step: std::time::Duration,
    /// Diagnostics: number of DBSP `transaction()`s clocked (gated
    /// `FELDERA_PROFILE`). With fused per-ruleset circuits this is ~1 per
    /// `run_rules` call, vs ~R (one per rule) before fusion.
    pub(crate) prof_transactions: u64,
    pub(crate) prof_apply: std::time::Duration,
    pub(crate) prof_merge: std::time::Duration,
    pub(crate) prof_change: std::time::Duration,
    /// `--native-uf --feldera`: drive PR #782's UF-backed-table encoding through
    /// Feldera's fast HOST-PASS rebuild (instead of the onchange-driven rebuild
    /// rules whose `view ⋈ @UF_Sf` arrangement is the integral that regressed the
    /// relational fast-rebuild on DBSP — the ~24% / transaction-count win).
    ///
    /// The #782 term encoder emits a UF-backed `:impl displaced-union-find`
    /// function `@UF_Sf (S) S` per eq-sort (via [`Backend::add_uf_function`])
    /// plus a `@UFChange_S` onchange relation and `@rebuild_rule*` /
    /// `@uf_change_drain_rule*` maintenance rules. When this is on we honour the
    /// `add_uf_function` request (a real `UfTable` + find-or-self canon-prim),
    /// route union writes `(set (@UF_Sf lhs) rhs)` into the in-core UF, suppress
    /// the maintenance rules by name, and re-canonicalize view rows host-side —
    /// keeping the `@rebuild_rule*` OUT of the fused DBSP circuit. Must be
    /// enabled (via [`EGraph::enable_native_uf`]) before any UF function is
    /// registered. Pure backend interception — the encoder is unchanged.
    pub(crate) native_uf_enabled: bool,
    /// Per-eq-sort native union-find, keyed by the [`FunctionId`] of the
    /// `@UF_Sf` UF-backed function (the [`Backend::add_uf_function`] handle).
    /// Reads (`find_ro`) and union ingestion both go through this. Single-
    /// threaded host, so a plain `UfTable` (no `Arc<Mutex>`) suffices.
    pub(crate) native_ufs: HashMap<FunctionId, uf::UfTable>,
    /// Maps the `@canon_S` find-or-self primitive's [`ExternalFunctionId`]
    /// (returned by [`Backend::add_uf_function`] and bound by the frontend to
    /// the canon-prim name) to its UF function id. A `BodyOp::Prim` / `HeadOp::
    /// Call` on this id is answered host-side from the matching [`native_ufs`]
    /// entry (`find_ro`) instead of through the `Database` external func.
    pub(crate) native_uf_canon_prim: HashMap<ExternalFunctionId, FunctionId>,
    /// `--fast-rebuild` (RELATIONAL, without `--native-uf`): drive the fused DBSP
    /// circuit's two-substep δuf rebuild that drops the always-empty
    /// `δview ⋈ uf_old` term. OR'd with the `FELDERA_DELTA_REBUILD` env var at
    /// circuit-build time. No-op under native-UF (the rebuild rules are taken out
    /// of the DBSP circuit and the host-side `view ⋈ δuf` delta runs by default).
    /// Off by default.
    pub(crate) fast_rebuild: bool,
    /// Reverse index for the native-UF delta rebuild: per VIEW function (the
    /// function a `@rebuild_rule*` re-inserts into), maps an eq-sort column
    /// *value* to the set of view rows that hold that value in some eq-sort
    /// column. Lets the rebuild pass enumerate "rows touching displaced id `v`"
    /// without scanning every row. Maintained at the centralized mirror-apply
    /// block in `interpret::run_iteration` (the only place view rows change
    /// during saturation), so when the next iteration's rebuild reads it the
    /// index matches that iteration's start-of-call snapshot. Empty unless
    /// `native_uf_enabled` is on. Keyed by `u32` (the stored rep), values are
    /// full rows so the rebuild can rebind every view column.
    pub(crate) native_uf_rev_index: HashMap<FunctionId, HashMap<u32, HashSet<Row>>>,
    /// The set of eq-sort column indices per VIEW function (which columns are
    /// UF-canonicalized, derived once from the `@rebuild_rule*` `col_uf`
    /// mapping). Tells the index maintainer which columns of a written/removed
    /// row to index. Populated on the first native-UF rebuild pass per view.
    pub(crate) native_uf_view_cols: HashMap<FunctionId, Vec<usize>>,
    /// Native-UF delta rebuild: the carried δuf — ids displaced since the rebuild
    /// pass last consumed this UF func (the `@UF_Sf` function id), ACCUMULATED
    /// across iterations. `native_uf_drain_all` APPENDS each round's
    /// newly-displaced ids here at the end of every `run_rules` call (drain runs
    /// AFTER the rebuild pass); a LATER iteration's rebuild pass consumes and
    /// clears it (`native_uf_take_displaced`). Accumulation is load-bearing: the
    /// union and the rebuild ruleset that canonicalizes over it are SEPARATE
    /// `run_rules` calls, so an overwrite-with-empty would lose a displaced set
    /// before the rebuild ever read it (issue: merge-during-rebuild). Carrying it
    /// across the boundary also lets a merge-triggered cascade converge.
    pub(crate) native_uf_prev_displaced: HashMap<FunctionId, Vec<i64>>,
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl EGraph {
    /// Construct a fresh Feldera-backed egraph.
    pub fn new() -> Self {
        EGraph {
            relations: Vec::new(),
            rules: Vec::new(),
            mirror: HashMap::new(),
            db: Database::new(),
            prim_engine: None,
            pure_prim_names: HashMap::new(),
            container_pool: FelderaContainerPool,
            external_funcs: ExternalFuncRegistry::default(),
            // Start at 1 so id 0 stays available as a "null"/padding sentinel
            // (rows are 0-padded; reserving 0 avoids colliding a real id with
            // padding in the uniform Row representation).
            next_id: 1,
            report_level: ReportLevel::default(),
            dbsp_rule_runs: 0,
            host_rule_runs: 0,
            circuit_rebuild: std::env::var("FELDERA_CIRCUIT_REBUILD").is_ok(),
            circuit_rebuild_runs: 0,
            rebuild_cache: None,
            persistent: HashMap::new(),
            fused: HashMap::new(),
            fed_fused: HashMap::new(),
            atomless_fired: HashSet::new(),
            fed: HashMap::new(),
            prof_read_clone: std::time::Duration::ZERO,
            prof_read_rows: 0,
            prof_fed_diff: std::time::Duration::ZERO,
            prof_circuit_step: std::time::Duration::ZERO,
            prof_transactions: 0,
            prof_apply: std::time::Duration::ZERO,
            prof_merge: std::time::Duration::ZERO,
            prof_change: std::time::Duration::ZERO,
            native_uf_enabled: false,
            native_ufs: HashMap::new(),
            native_uf_canon_prim: HashMap::new(),
            fast_rebuild: false,
            native_uf_rev_index: HashMap::new(),
            native_uf_view_cols: HashMap::new(),
            native_uf_prev_displaced: HashMap::new(),
        }
    }

    /// Turn on the native union-find path (`--native-uf --feldera`). Must be
    /// called before any UF function is registered: [`Backend::add_uf_function`]
    /// checks this flag to decide whether to honour the PR #782 UF-backed-table
    /// request (a real in-core [`uf::UfTable`] + find-or-self canon-prim) or to
    /// bail (the default). Mirrors the DuckDB / FlowLog backends' `enable_native_uf`.
    pub fn enable_native_uf(&mut self) {
        self.native_uf_enabled = true;
    }

    /// Turn on the RELATIONAL δuf fast-rebuild (`--fast-rebuild`): the fused DBSP
    /// circuit's two-substep split that drops the always-empty `δview ⋈ uf_old`
    /// rebuild term (sound under canonicalize-at-creation). Only meaningful
    /// WITHOUT native UF — under native-UF the rebuild rules are taken out of the
    /// DBSP circuit entirely and the host-side `view ⋈ δuf` delta rebuild runs by
    /// default, so this flag is a no-op there. Equivalent to the
    /// `FELDERA_DELTA_REBUILD` env var. May be called any time before `run_rules`.
    pub fn enable_fast_rebuild(&mut self) {
        self.fast_rebuild = true;
    }

    /// Read-only native-UF find for the `@UF_Sf` function `uf_func`. Returns the
    /// class leader of `x` (or `x` itself if `x` has never been unioned /
    /// `uf_func` is not a native UF). Single hash lookup (eager-flatten UF).
    pub(crate) fn native_uf_find(&self, uf_func: FunctionId, x: u32) -> u32 {
        match self.native_ufs.get(&uf_func) {
            Some(uf) => uf.find_ro(x as i64) as u32,
            None => x,
        }
    }

    /// Apply all queued unions to every native UF (called once per `run_rules`
    /// after head writes land). After this, every UF is flat: `find_ro` is O(1)
    /// and consistent with the union assertions ingested this iteration.
    /// Returns the total number of ids whose canonical changed (the "real
    /// change" signal the outer saturate loop needs, since the relational UF's
    /// `@UF_S` / flat-index churn is no longer produced).
    pub(crate) fn native_uf_drain_all(&mut self) -> usize {
        for uf in self.native_ufs.values_mut() {
            uf.drain_pending();
        }
        let mut displaced = 0;
        // Under native-UF the rebuild is ALWAYS the delta path (`view ⋈ δuf`),
        // so stash this round's displaced ids per UF func for a LATER iteration's
        // rebuild to consume (the δuf carried across the boundary).
        //
        // ACCUMULATE rather than overwrite: a union does not necessarily land in
        // the same `run_rules` call that runs the rebuild ruleset over it. The
        // frontend schedules the user ruleset and the `@rebuilding` ruleset as
        // SEPARATE `run_rules` calls, and `native_uf_drain_all` runs at the end of
        // EVERY call — so an overwrite would clobber a still-unconsumed displaced
        // set with an empty Vec the very next (union-free) iteration, before the
        // rebuild pass ever read it (issue: merge-during-rebuild). We instead
        // append each round's newly-displaced ids to the per-UF stash and let the
        // rebuild pass DRAIN it on consumption (`native_uf_take_displaced`), so the
        // δuf survives intervening iterations and a cascade converges across calls.
        let stash = self.native_uf_enabled;
        for (&func, uf) in self.native_ufs.iter_mut() {
            displaced += uf.displaced_len();
            let drained = uf.drain_displaced();
            if stash && !drained.is_empty() {
                self.native_uf_prev_displaced
                    .entry(func)
                    .or_default()
                    .extend(drained);
            }
        }
        displaced
    }

    /// Take (and clear) the carried displaced-id set for UF `func` — the δuf the
    /// native-UF rebuild pass consumes for one `@rebuild_rule*` view. Accumulated
    /// by [`native_uf_drain_all`] across however many iterations separate the
    /// union from the rebuild ruleset's `run_rules` call; cleared here on
    /// consumption so each displaced id drives exactly one rebuild scan. Returns
    /// an empty slice when nothing is pending.
    pub(crate) fn native_uf_take_displaced(&mut self, func: FunctionId) -> Vec<i64> {
        self.native_uf_prev_displaced
            .remove(&func)
            .unwrap_or_default()
    }

    /// Diagnostics: `(dbsp_rule_runs, host_rule_runs)` — the number of rule
    /// firings whose body join ran on DBSP vs. on the host interpreter
    /// fallback since construction. Lets tests / surveys report exactly which
    /// fraction of work ran genuinely on DBSP.
    pub fn dbsp_join_stats(&self) -> (u64, u64) {
        (self.dbsp_rule_runs, self.host_rule_runs)
    }

    /// Record a primitive's user-visible egglog name. Mirrors the duckdb
    /// `set_external_func_name` side-channel: the frontend's typechecker calls
    /// this after `register_external_func` so `dbsp_join::plan_join` can
    /// recognize the generic `!=` guard by name and make the surrounding rule
    /// (congruence / rebuild / `@uf_update`) DBSP-eligible. Unlike the duckdb
    /// path this is purely informational — feldera never renames prims for
    /// evaluation; it only consults the name to decide join eligibility.
    pub fn set_external_func_name(&mut self, id: ExternalFunctionId, name: String) {
        self.external_funcs.set_name(id, name);
    }

    /// Diagnostics: number of `run_rules` calls served by the Stage-3
    /// persistent rebuild circuit (`FELDERA_CIRCUIT_REBUILD=1`) rather than the
    /// host interpreter. Zero when the flag is off or no rebuild ruleset ran.
    pub fn circuit_rebuild_runs(&self) -> u64 {
        self.circuit_rebuild_runs
    }

    pub(crate) fn info(&self, f: FunctionId) -> &RelationInfo {
        self.relations
            .get(f.rep() as usize)
            .unwrap_or_else(|| panic!("FunctionId({}) not registered", f.rep()))
    }

    /// The type of column `col` of function `f`, if known. Used by
    /// `dbsp_join::plan_join` to type-gate prim inlining (rep-arithmetic is only
    /// valid on `Id` and bool columns).
    pub(crate) fn col_ty(&self, f: FunctionId, col: usize) -> Option<ColumnTy> {
        self.relations
            .get(f.rep() as usize)
            .and_then(|r| r.schema.get(col).copied())
    }

    /// The [`BaseValueId`] of the `bool` base type, if it has been registered.
    /// Lets `dbsp_join::plan_join` recognize bool-typed columns (whose distinct
    /// reps make equality / `bool-!=` / `or` rep-arithmetic valid).
    pub(crate) fn bool_bvid(&self) -> Option<BaseValueId> {
        self.bvid_of::<bool>()
    }

    /// The [`BaseValueId`] of the `i64` base type, if it has been registered.
    /// `i64` interning is injective, so rep-equality matches value equality:
    /// `dbsp_join::plan_join` may inline `=`/`!=` (but never ordering) on these
    /// columns. (See the `f64` exception in `plan_join`'s docs.)
    pub(crate) fn i64_bvid(&self) -> Option<BaseValueId> {
        self.bvid_of::<i64>()
    }

    /// The [`BaseValueId`] of the `String` base type, if it has been registered.
    /// Like `i64`, string interning is injective so rep-equality matches value
    /// equality and `=`/`!=` may be inlined.
    pub(crate) fn string_bvid(&self) -> Option<BaseValueId> {
        self.bvid_of::<String>()
    }

    /// The [`BaseValueId`] of base type `T`, if it has been registered.
    fn bvid_of<T: 'static>(&self) -> Option<BaseValueId> {
        let bvs = self.db.base_values();
        let id = std::any::TypeId::of::<T>();
        bvs.has_ty_by_id(id).then(|| bvs.get_ty_by_id(id))
    }

    /// Schema changed (relation/rule added/removed). No cached state to clear in
    /// the host-interpreter execution model; kept as a hook (and so the rule
    /// builder's `invalidate_circuit()` call site stays meaningful).
    fn invalidate_circuit(&mut self) {}

    /// Insert a single row into the Rust mirror.
    fn mirror_insert(&mut self, f: FunctionId, row: Row) {
        self.native_uf_index_insert(f, &row);
        std::rc::Rc::make_mut(self.mirror.entry(f).or_default()).insert(row);
    }

    /// `fast_rebuild` reverse-index maintenance: record that `row` (of view
    /// function `f`) holds each of its eq-sort column values, so the rebuild
    /// pass can find it from a displaced id. No-op unless `fast_rebuild` is on
    /// and `f` is a recognized view (its eq-sort columns are known). Idempotent
    /// (a `HashSet` of rows per value). Cheap: a few inserts per row.
    pub(crate) fn native_uf_index_insert(&mut self, f: FunctionId, row: &Row) {
        if !self.native_uf_enabled {
            return;
        }
        let Some(cols) = self.native_uf_view_cols.get(&f) else {
            return;
        };
        let cols: Vec<usize> = cols.clone();
        let entry = self.native_uf_rev_index.entry(f).or_default();
        for ci in cols {
            if ci < row.len() {
                let v = row[ci];
                entry.entry(v).or_default().insert(row.clone());
            }
        }
    }

    /// `fast_rebuild` reverse-index maintenance: drop `row` (of view `f`) from
    /// every eq-sort-value bucket it was registered under. Mirror of
    /// [`native_uf_index_insert`]; call BEFORE the row leaves the mirror.
    pub(crate) fn native_uf_index_remove(&mut self, f: FunctionId, row: &Row) {
        if !self.native_uf_enabled {
            return;
        }
        let Some(cols) = self.native_uf_view_cols.get(&f) else {
            return;
        };
        let Some(entry) = self.native_uf_rev_index.get_mut(&f) else {
            return;
        };
        for &ci in cols {
            if ci < row.len() {
                let v = row[ci];
                if let Some(bucket) = entry.get_mut(&v) {
                    bucket.remove(row);
                    if bucket.is_empty() {
                        entry.remove(&v);
                    }
                }
            }
        }
    }

    /// `fast_rebuild`: register view function `f`'s eq-sort columns and build its
    /// reverse index from the current mirror, ONCE. Called at the start of each
    /// `run_iteration` for every `@rebuild_rule*`'s view (idempotent after the
    /// first call). Seeding from the mirror here captures rows that entered the
    /// mirror outside the per-iteration apply block (base facts / `add_term` /
    /// seed inserts) — after this the index is maintained incrementally at every
    /// mirror write, so it always reflects the live mirror.
    pub(crate) fn native_uf_seed_view_index(&mut self, f: FunctionId, eq_cols: &[usize]) {
        if !self.native_uf_enabled {
            return;
        }
        if self.native_uf_view_cols.contains_key(&f) {
            return;
        }
        self.native_uf_view_cols.insert(f, eq_cols.to_vec());
        let rows: Vec<Row> = match self.mirror.get(&f) {
            Some(set) => set.iter().cloned().collect(),
            None => Vec::new(),
        };
        for row in rows {
            self.native_uf_index_insert(f, &row);
        }
    }

    /// Resolve a function's merge mode (for FD-conflict resolution).
    fn merge_mode(&self, f: FunctionId) -> MergeMode {
        self.info(f).merge
    }

    /// Evaluate a primitive through the embedded `Database` (the inherent
    /// counterpart of the [`Backend::eval_prim`] trait method). Both the host
    /// interpreter and the DBSP-join path call this; neither reaches into
    /// `self.db` directly.
    pub(crate) fn eval_prim_internal(
        &self,
        id: ExternalFunctionId,
        args: &[Value],
    ) -> Option<Value> {
        self.db
            .with_execution_state(|st| st.call_external_func(id, args))
    }

    /// A shareable clone of the primitive engine, built lazily on first use
    /// (after all primitives are registered). The returned [`PrimEngine`] is
    /// `Send + 'static` and can be captured into DBSP circuit closures so they
    /// evaluate pure value-computing body prims ON-CIRCUIT (Stage C). The clone
    /// shares interning with `self.db`, so on-circuit results are bit-identical
    /// to host evaluation.
    pub(crate) fn prim_engine(&mut self) -> PrimEngine {
        if self.prim_engine.is_none() {
            self.prim_engine = Some(PrimEngine::new(self.db.clone()));
        }
        self.prim_engine.as_ref().unwrap().clone()
    }

    /// Record a PURE primitive's user-visible name (Stage C). The frontend's
    /// typechecker calls this for every pure prim so `dbsp_join::plan_join` can
    /// lower an arbitrary pure value prim to an on-circuit call-prim node. Impure
    /// prims are never recorded here and stay host-ineligible.
    pub fn set_pure_prim_name(&mut self, id: ExternalFunctionId, name: String) {
        self.external_funcs.set_name(id, name.clone());
        self.pure_prim_names.insert(id.rep(), name);
    }

    /// Whether `id` names a primitive the frontend marked PURE (safe to
    /// re-evaluate on-circuit). See [`EGraph::set_pure_prim_name`].
    pub(crate) fn is_pure_prim(&self, id: ExternalFunctionId) -> bool {
        self.pure_prim_names.contains_key(&id.rep())
    }

    /// Allocate a fresh id (used by the interpreter's eq-sort constructor
    /// hash-cons; same counter the trait's `fresh_id` advances).
    pub(crate) fn fresh_id_internal(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Resolve functional-dependency conflicts in a relation's mirror set: for
    /// each key (input columns) keep a single output column chosen by the merge
    /// mode. Relations (whole-row key) are left untouched. Returns whether any
    /// row was actually changed/collapsed.
    ///
    /// INCREMENTAL: the pre-call mirror is already FD-resolved (this runs every
    /// `run_rules` call), so only the keys whose rows were `set` this call (in
    /// `keys`) can newly conflict. We re-resolve just those keys instead of
    /// rebuilding the whole function — O(touched-keys), not O(state). The
    /// deterministic fold order (sort) is preserved PER KEY, which gives the same
    /// chosen output as the old whole-set sort+fold (the fold is independent
    /// across distinct keys).
    fn resolve_merge(&mut self, f: FunctionId, keys: &HashSet<Vec<u32>>) -> bool {
        let arity = self.info(f).arity;
        let merge = self.merge_mode(f);
        if !matches!(merge, MergeMode::Old | MergeMode::New | MergeMode::Min)
            || arity == 0
            || keys.is_empty()
        {
            return false;
        }
        let Some(set) = self.mirror.get(&f) else {
            return false;
        };
        let inputs_len = arity - 1;
        // Gather the candidate rows for the touched keys only.
        let mut by_key: HashMap<&[u32], Vec<&Row>> = HashMap::new();
        for row in set.iter() {
            let key: Vec<u32> = (0..inputs_len).map(|i| row_col(row, i)).collect();
            if keys.contains(&key) {
                by_key.entry(&row[..inputs_len]).or_default().push(row);
            }
        }
        // Resolve each touched key; collect the rows to remove and the winner to
        // insert. Only keys with >1 candidate row can change.
        let mut new_rows: Vec<Row> = Vec::new();
        let mut drop_rows: HashSet<Row> = HashSet::new();
        for (_key, mut cands) in by_key {
            if cands.len() < 2 {
                continue;
            }
            // Deterministic fold order (mirror is a HashSet — arbitrary order).
            cands.sort();
            let mut chosen = row_col(cands[0], inputs_len);
            for row in &cands[1..] {
                let out = row_col(row, inputs_len);
                chosen = match merge {
                    MergeMode::Old => chosen,
                    MergeMode::New => out,
                    MergeMode::Min => chosen.min(out),
                    MergeMode::Relation => unreachable!(),
                };
            }
            // The winning row.
            let mut winner: Vec<u32> = cands[0][..inputs_len].to_vec();
            winner.push(chosen);
            let winner: Row = winner.into_boxed_slice();
            for row in cands {
                if **row != *winner {
                    drop_rows.insert((*row).clone());
                }
            }
            new_rows.push(winner);
        }
        if drop_rows.is_empty() {
            return false;
        }
        let set = std::rc::Rc::make_mut(self.mirror.get_mut(&f).unwrap());
        for r in &drop_rows {
            set.remove(r);
        }
        for r in &new_rows {
            set.insert(r.clone());
        }
        // Keep the `fast_rebuild` reverse index consistent with the merge's
        // retract-losers / insert-winner edits (no-op unless `fast_rebuild` is
        // on and `f` is a view).
        for r in &drop_rows {
            self.native_uf_index_remove(f, r);
        }
        for r in &new_rows {
            self.native_uf_index_insert(f, r);
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Send + Sync (single-threaded use; same posture as the DuckDB backend)
// ---------------------------------------------------------------------------
//
// DBSP's `CircuitHandle` holds an `Rc`-based `RootCircuit`, so `EGraph` is not
// auto-`Send`/`Sync`. The `Backend` trait requires both. As with the DuckDB
// backend, the egraph is only ever driven from a single thread, so we assert
// the bounds. Concurrent multi-thread use would violate this and is not a
// supported configuration.
unsafe impl Send for EGraph {}
unsafe impl Sync for EGraph {}

impl Drop for EGraph {
    // The `FELDERA_PROFILE` breakdown is a temporary stderr diagnostic gated
    // behind an env var (never on in normal runs); `eprintln!` is intentional.
    #[allow(clippy::disallowed_macros)]
    fn drop(&mut self) {
        // DIAGNOSTIC ONLY (gated `FELDERA_STATS`): final cumulative split of
        // rule firings (body join on DBSP vs host nested-loop fallback). Used by
        // suite-wide surveys; never on in normal runs.
        if std::env::var("FELDERA_STATS").is_ok() {
            eprintln!(
                "[FELDERA_STATS] dbsp_runs={} host_runs={}",
                self.dbsp_rule_runs, self.host_rule_runs,
            );
        }
        if std::env::var("FELDERA_PROFILE").is_ok() {
            eprintln!(
                "[PROF] read_clone={:.2}s (rows_total={}) fed_diff={:.2}s circuit_step={:.2}s transactions={}",
                self.prof_read_clone.as_secs_f64(),
                self.prof_read_rows,
                self.prof_fed_diff.as_secs_f64(),
                self.prof_circuit_step.as_secs_f64(),
                self.prof_transactions,
            );
            eprintln!(
                "[PROF] apply_writes={:.2}s resolve_merge={:.2}s change_detect={:.2}s (folded into apply/merge)",
                self.prof_apply.as_secs_f64(),
                self.prof_merge.as_secs_f64(),
                self.prof_change.as_secs_f64(),
            );
            eprintln!(
                "[PROF] dbsp_runs={} host_runs={} (host rebuild/congruence join is the residual bottleneck)",
                self.dbsp_rule_runs, self.host_rule_runs,
            );
            use std::sync::atomic::Ordering;
            let feed = crate::dbsp_join::PROF_FEED_NS.load(Ordering::Relaxed) as f64 / 1e9;
            let step = crate::dbsp_join::PROF_STEP_NS.load(Ordering::Relaxed) as f64 / 1e9;
            let read = crate::dbsp_join::PROF_READ_NS.load(Ordering::Relaxed) as f64 / 1e9;
            let calls = crate::dbsp_join::PROF_STEP_CALLS.load(Ordering::Relaxed);
            eprintln!(
                "[PROF-PHASE] feed={feed:.2}s step(transaction)={step:.2}s read(consolidate)={read:.2}s nonempty_steps={calls}",
            );
        }
    }
}

// ---------------------------------------------------------------------------
// impl Backend
// ---------------------------------------------------------------------------

impl Backend for EGraph {
    // -- table lifecycle ----------------------------------------------------

    fn add_table(&mut self, config: FunctionConfig) -> FunctionId {
        let id = FunctionId::new(self.relations.len() as u32);
        let arity = config.schema.len();
        assert!(
            arity <= compile::MAX_ARITY,
            "Feldera backend (milestone 1) supports relations of arity <= {} (got {} for `{}`)",
            compile::MAX_ARITY,
            arity,
            config.name
        );
        // Relation vs function: a table is a **relation** (whole row is the key,
        // no output column to merge) iff it is nullary OR its last column is
        // `Unit` — this is the term encoder's view-table pattern
        // `(function @XView (...) Unit :merge old)` AND ordinary relations.
        // Otherwise the last column is a function OUTPUT, resolved by the merge
        // mode. This mirrors the DuckDB backend's Unit-detection
        // (`backend_impl.rs` ~593) — NOT `DefaultVal`, which is `Fail` for every
        // custom function regardless of whether it has a real output column.
        let output_is_unit = config.schema.last().is_some_and(|t| match t {
            ColumnTy::Base(bv) => {
                let bvs = self.db.base_values();
                bvs.has_ty_by_id(std::any::TypeId::of::<()>())
                    && *bv == bvs.get_ty_by_id(std::any::TypeId::of::<()>())
            }
            _ => false,
        });
        let has_output = arity > 0 && !output_is_unit;
        // Recognize the merge mode (FD-conflict resolution). Mirrors the DuckDB
        // backend's stopgap recognition:
        //   - `AssertEq` / `Old`  => keep the old value
        //   - `New`               => keep the new value
        //   - `UnionId`           => lattice-min (the union-find leader)
        //   - `Primitive(_)`      => lattice-min (the term encoder's `@uff`
        //     `:merge (ordering-min …)` and user `:merge (min old new)`). The
        //     only complex merge the rebuild / tractable programs need.
        //   - `Function`/`Const`  => fall back to `Old`.
        // A relation (no output column, or Unit output) needs no FD resolution.
        use egglog_backend_trait::MergeFn;
        let merge = if !has_output {
            MergeMode::Relation
        } else {
            match &config.merge {
                MergeFn::AssertEq | MergeFn::Old => MergeMode::Old,
                MergeFn::New => MergeMode::New,
                MergeFn::UnionId => MergeMode::Min,
                MergeFn::Primitive(_, _) => MergeMode::Min,
                MergeFn::Function(_, _) | MergeFn::Const(_) => MergeMode::Old,
            }
        };
        // Identity-on-miss ("lookup-or-self"): an action-position lookup of an
        // absent key resolves to the key itself, inserting no row. Only valid for
        // a single-key function (2 columns) whose key and output share a type —
        // the term encoder's flat UF-index `@UF_Sf`.
        let identity_on_miss = matches!(config.default, egglog_backend_trait::DefaultVal::Identity);
        if identity_on_miss {
            assert_eq!(
                arity, 2,
                "DefaultVal::Identity (`{}`) expects a single key column (2-column key->output function)",
                config.name
            );
        }
        self.relations.push(RelationInfo {
            name: config.name,
            arity,
            schema: config.schema.clone(),
            has_output,
            merge,
            identity_on_miss,
        });
        self.mirror.insert(id, std::rc::Rc::new(HashSet::new()));
        self.invalidate_circuit();
        id
    }

    fn add_uf_function(
        &mut self,
        name: String,
        _onchange: Option<FunctionId>,
        proof: Option<egglog_backend_trait::UfProofConfig>,
    ) -> Result<(FunctionId, ExternalFunctionId)> {
        // Only the `--native-uf --feldera` path supports PR #782's UF-backed
        // function. With the flag off the relational UF encoding is used (no
        // `add_uf_function` calls), so a bail here is unreachable in practice.
        if !self.native_uf_enabled {
            anyhow::bail!(
                "the Feldera backend only supports `:impl displaced-union-find` \
                 functions under `--native-uf` (it drives PR #782's UF-backed \
                 encoding through a host-pass rebuild)"
            );
        }
        // Proof mode is a later step (TERM mode only for now): a provenance-
        // tracking UF would need the `@UFChange_S` proof column composed in a
        // leader-change callback, which the host-pass rebuild does not run.
        if proof.is_some() {
            anyhow::bail!(
                "the Feldera backend does not yet support proof-mode native-UF \
                 functions (`--native-uf` is TERM mode only on Feldera)"
            );
        }

        // Register the UF function as a real relation: schema `(S) S` (arity 2,
        // two eq-sort id columns, output column, `Min` merge — the union-find
        // leader). The mirror is never populated by writes (union `set`s are
        // intercepted into the in-core UF), but the relation must exist so its
        // FunctionId resolves in `info` / `lookup_id` (the extractor's
        // `find_canonical` reads it).
        let id = FunctionId::new(self.relations.len() as u32);
        self.relations.push(RelationInfo {
            name,
            arity: 2,
            schema: vec![ColumnTy::Id, ColumnTy::Id],
            has_output: true,
            merge: MergeMode::Min,
            identity_on_miss: false,
        });
        self.mirror.insert(id, std::rc::Rc::new(HashSet::new()));
        self.native_ufs.insert(id, uf::UfTable::new());

        // The find-or-self canon-prim. A real, freeable `ExternalFunctionId`,
        // but the interpreter intercepts calls to it (see `native_uf_canon_prim`)
        // and answers `find_ro` from the in-core UF — the registered stub is
        // never actually invoked through the `Database`.
        let canon = self.register_external_func(Box::new(external_func::CanonStub));
        self.native_uf_canon_prim.insert(canon, id);
        self.invalidate_circuit();
        Ok((id, canon))
    }

    fn table_size(&self, table: FunctionId) -> usize {
        self.mirror.get(&table).map(|s| s.len()).unwrap_or(0)
    }

    fn approx_table_size(&self, table: FunctionId) -> usize {
        self.table_size(table)
    }

    // -- iteration ----------------------------------------------------------

    fn for_each(&self, table: FunctionId, f: &mut dyn for<'r> FnMut(FunctionRow<'r>)) {
        self.for_each_while(table, &mut |row| {
            f(row);
            true
        });
    }

    fn for_each_while(
        &self,
        table: FunctionId,
        f: &mut dyn for<'r> FnMut(FunctionRow<'r>) -> bool,
    ) {
        let arity = self.info(table).arity;
        let Some(set) = self.mirror.get(&table) else {
            return;
        };
        for row in set.iter() {
            let vals = unpack_row(row, arity);
            let frow = FunctionRow {
                vals: &vals,
                subsumed: false,
            };
            if !f(frow) {
                break;
            }
        }
    }

    // -- direct access ------------------------------------------------------

    fn lookup_id(&self, func: FunctionId, key: &[Value]) -> Option<Value> {
        // Native-UF find route: the `@UF_Sf` function's rows are not
        // materialized (unions live in the in-core UF), so a mirror scan would
        // always miss. Answer from the UF instead: `find_ro(x)` is the class
        // leader, or `x` itself when unrecorded. This is the extractor's
        // `find_canonical` path (`backend.lookup_id`).
        if let Some(uf) = self.native_ufs.get(&func) {
            if key.len() == 1 {
                return Some(Value::new(uf.find_ro(key[0].rep() as i64) as u32));
            }
            return None;
        }
        let info = self.info(func);
        if !info.has_output {
            return None;
        }
        let inputs_len = info.arity - 1;
        if key.len() != inputs_len {
            return None;
        }
        let set = self.mirror.get(&func)?;
        for row in set.iter() {
            if (0..inputs_len).all(|i| compile::row_col(row, i) == key[i].rep()) {
                return Some(Value::new(compile::row_col(row, inputs_len)));
            }
        }
        None
    }

    fn add_values(&mut self, values: Box<dyn Iterator<Item = (FunctionId, Vec<Value>)> + '_>) {
        for (func, row) in values {
            let arity = self.info(func).arity;
            assert_eq!(row.len(), arity, "add_values: row arity mismatch");
            self.mirror_insert(func, pack_row(&row));
        }
        // Seed inserts are reflected in the mirror immediately and pushed into
        // the circuit at the next `run_rules`; no separate flush needed.
        self.invalidate_circuit();
    }

    fn add_term(&mut self, func: FunctionId, inputs: &[Value]) -> Value {
        // Allocate a fresh id and store `(inputs..., fresh_id)`.
        let id = self.fresh_id();
        let mut full = inputs.to_vec();
        full.push(id);
        let arity = self.info(func).arity;
        assert_eq!(full.len(), arity, "add_term: arity mismatch");
        self.mirror_insert(func, pack_row(&full));
        self.invalidate_circuit();
        id
    }

    fn insert_rows(&mut self, table: FunctionId, rows: &[Vec<Value>]) {
        let arity = self.info(table).arity;
        for row in rows {
            assert_eq!(row.len(), arity, "insert_rows: row arity mismatch");
            self.mirror_insert(table, pack_row(row));
        }
        self.invalidate_circuit();
    }

    fn lookup_constructor_rows(&mut self, table: FunctionId, rows: &[Vec<Value>]) {
        for row in rows {
            if self.lookup_id(table, row).is_none() {
                let _ = self.add_term(table, row);
            }
        }
    }

    fn get_canon_repr(&self, val: Value, _ty: ColumnTy) -> Value {
        // Milestone 1 has no union-find; canonicalization is the identity.
        val
    }

    fn fresh_id(&mut self) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        Value::new(id)
    }

    fn clear_table(&mut self, func: FunctionId) {
        if let Some(set) = self.mirror.get_mut(&func) {
            std::rc::Rc::make_mut(set).clear();
        }
        // Drop the cleared function's `fast_rebuild` reverse-index entry too, so
        // it is re-seeded from the (now-empty) mirror on the next iteration.
        self.native_uf_rev_index.remove(&func);
        // No per-rule `seen` to forget: each persistent rule's circuit will
        // diff the now-cleared mirror against its last-fed view at the next
        // `run_rules`, naturally retracting the dropped rows from its integral
        // (and re-adding them if the table is repopulated). The `fed` snapshot is
        // intentionally left in place so that diff is computed correctly.
        self.invalidate_circuit();
    }

    fn base_values(&self) -> &egglog_core_relations::BaseValues {
        self.db.base_values()
    }

    fn with_execution_state_dyn(
        &self,
        _f: &mut dyn FnMut(&mut egglog_backend_trait::ExecutionState<'_>),
    ) {
        // Bridge-only escape hatch; not supported (same as DuckDB).
        unimplemented!("with_execution_state is not supported on the Feldera backend")
    }

    fn action_registry_any(&self) -> &(dyn Any + Send + Sync) {
        unimplemented!("action_registry is not supported on the Feldera backend")
    }

    // -- rule management ----------------------------------------------------

    fn new_rule<'a>(&'a mut self, desc: &str, _seminaive: bool) -> Box<dyn RuleBuilderOps + 'a> {
        // Seminaive is subsumed by DBSP's incremental join (PLAN §3.2); the
        // flag is accepted for parity and ignored.
        Box::new(rule_builder::FelderaRuleBuilder::new(self, desc))
    }

    fn free_rule(&mut self, id: RuleId) {
        if let Some(slot) = self.rules.get_mut(id.rep() as usize) {
            *slot = None;
            let i = id.rep() as usize;
            // Drop all per-rule persistent state so a reused rule index (egglog's
            // `eval_actions` builds + frees a fresh rule every command) starts
            // clean — otherwise a stale `atomless_fired` / circuit would suppress
            // the new rule's firing.
            self.persistent.remove(&i);
            self.fed.remove(&i);
            self.atomless_fired.remove(&i);
            // Drop any FUSED circuit whose ruleset includes this freed rule, so
            // a reused index never reuses a stale fused circuit.
            self.fused.retain(|key, _| !key.contains(&i));
            self.fed_fused.retain(|key, _| !key.contains(&i));
            self.invalidate_circuit();
        }
    }

    fn run_rules(&mut self, rules: &[RuleId]) -> Result<IterationReport> {
        // ONE egglog iteration = one round of rule application over the current
        // relations, running ONLY the rules in `rules`. The frontend calls this
        // N times for `(run N)` and with different rule subsets to schedule
        // distinct rulesets.
        //
        // Milestone 3 executes rules with a **host-side interpreter**
        // (`interpret.rs`): a nested-loop join of the rule's table body atoms
        // over the current mirror, primitive body atoms / RHS lookups / RHS
        // primitive calls evaluated against the embedded `Database`, then the
        // head actions applied. This is one bounded hop (the round reads the
        // mirror as it was at entry — semi-naive-equivalent for one iteration),
        // matching the per-iteration model M1/M2 established with DBSP.
        if rules.is_empty() {
            return Ok(IterationReport::default());
        }

        let live: Vec<usize> = rules
            .iter()
            .map(|r| r.rep() as usize)
            .filter(|&i| self.rules.get(i).map(|s| s.is_some()).unwrap_or(false))
            .collect();
        if live.is_empty() {
            return Ok(IterationReport::default());
        }

        // Stage-3: when `FELDERA_CIRCUIT_REBUILD=1` and this call is a pure
        // rebuild ruleset, run the whole rebuild fixpoint on the persistent
        // congruence-shuttle circuit instead of the interpreter. Recognition is
        // conservative — anything unrecognized falls back to the interpreter, so
        // this can never regress a program, only accelerate rebuild.
        if self.circuit_rebuild {
            if let Some(roles) = circuit_rebuild::recognize(self, &live) {
                self.circuit_rebuild_runs += 1;
                let changed = circuit_rebuild::run_rebuild(self, &roles)?;
                let mut report = IterationReport::default();
                report.rule_set_report.changed = changed;
                return Ok(report);
            }
        }

        let changed = interpret::run_iteration(self, &live)?;

        let mut report = IterationReport::default();
        report.rule_set_report.changed = changed;
        Ok(report)
    }

    fn flush_updates(&mut self) -> bool {
        // Seed inserts land in the mirror immediately; there is no separate
        // staged-update queue distinct from `run_rules`.
        false
    }

    // -- primitives ---------------------------------------------------------

    fn register_external_func(
        &mut self,
        func: Box<dyn ExternalFunction + 'static>,
    ) -> ExternalFunctionId {
        // Register into the embedded Database so the primitive is *invokable*
        // through `Database::with_execution_state` during rule firing (the
        // frontend's query rules call `collect_matches` this way). The local
        // registry mirrors the same id for name tracking + panic sentinels;
        // both advance in lockstep so the ids stay aligned.
        let func2 = dyn_clone::clone_box(&*func);
        let id = self.db.add_external_function(func);
        self.external_funcs.add_func_at(id, func2);
        // The cached prim-engine clone (if any) predates this registration;
        // drop it so the next on-circuit eval rebuilds a clone holding this prim.
        self.prim_engine = None;
        id
    }

    fn free_external_func(&mut self, func: ExternalFunctionId) {
        self.db.free_external_function(func);
        self.external_funcs.free(func);
        self.pure_prim_names.remove(&func.rep());
        self.prim_engine = None;
    }

    fn new_panic(&mut self, message: String) -> ExternalFunctionId {
        // A panic sentinel needs an id aligned with the Database's external-func
        // table (the frontend references it via `call_external_func`). Register
        // a real panicking `ExternalFunction` so invoking it surfaces the
        // message, and mirror it in the local registry.
        let panic_fn = external_func::PanicFunc::new(message.clone());
        let id = self.db.add_external_function(Box::new(panic_fn));
        self.external_funcs.add_panic_at(id, message);
        self.prim_engine = None;
        id
    }

    fn eval_prim(&self, id: ExternalFunctionId, args: &[Value]) -> Option<Value> {
        // The primitive lives in the embedded `Database`; invoke it through an
        // execution state. This is the single backend-agnostic entry point the
        // interpreter and the DBSP-join path both use to evaluate primitives,
        // so neither has to reach into `self.db` directly.
        self.db
            .with_execution_state(|st| st.call_external_func(id, args))
    }

    // -- typed value handles ------------------------------------------------

    fn base_value_pool(&self) -> &dyn BaseValuePool {
        base_values::base_values_as_pool(self.db.base_values())
    }

    fn base_value_pool_mut(&mut self) -> &mut dyn BaseValuePool {
        base_values_as_pool_mut(self.db.base_values_mut())
    }

    fn container_pool(&self) -> &dyn ContainerPool {
        &self.container_pool
    }

    fn container_pool_mut(&mut self) -> &mut dyn ContainerPool {
        &mut self.container_pool
    }

    fn base_value_constant_dyn(&self, value: Value, ty: BaseValueId) -> QueryEntry {
        QueryEntry::Const {
            val: value,
            ty: ColumnTy::Base(ty),
        }
    }

    // -- capability flags ---------------------------------------------------

    fn supports_inline_table_lookups(&self) -> bool {
        // Reads cannot reenter the circuit mid-rule (same as DuckDB).
        false
    }

    fn supports_subsumption(&self) -> bool {
        false
    }

    fn supports_complex_merge(&self) -> bool {
        false
    }

    fn supports_containers(&self) -> bool {
        false
    }

    // -- diagnostics --------------------------------------------------------

    fn set_report_level(&mut self, level: ReportLevel) {
        self.report_level = level;
    }

    fn dump_debug_info(&self) {
        for (i, info) in self.relations.iter().enumerate() {
            let f = FunctionId::new(i as u32);
            let n = self.table_size(f);
            if n == 0 {
                continue;
            }
            log::info!("== Feldera relation `{}` ({} rows) ==", info.name, n);
        }
    }

    // -- cloning ------------------------------------------------------------

    fn clone_boxed(&self) -> Box<dyn Backend> {
        // Push/pop snapshot support is a later milestone (PLAN Phase 5): a
        // built DBSP circuit cannot be cloned, but the *state* (mirror + rule
        // IR + relation metadata) can be replayed into a fresh circuit. Not
        // needed for milestone 1.
        unimplemented!(
            "Feldera backend clone_boxed (push/pop) is deferred to PLAN Phase 5 \
             (snapshot-and-replay into a fresh circuit)"
        )
    }

    // -- bridge-only escape hatch ------------------------------------------

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
