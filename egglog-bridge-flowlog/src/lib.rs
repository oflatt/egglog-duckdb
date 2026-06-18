//! # egglog-bridge-flowlog
//!
//! A FlowLog (Differential-Dataflow)-backed executor for egglog's resolved IR,
//! behind the [`egglog_backend_trait::Backend`] interface. This is the FlowLog
//! analog of the Feldera/DBSP backend's Milestone 1.
//!
//! ## Milestone 1 — bounded per-iteration `(run N)` stepping on real flowlog-rs
//!
//! egglog's `(run N)` applies a ruleset **N times with bounded extension per
//! round** — a transitive-closure rule extends **N hops, NOT to full closure**.
//! This backend proves that bounded behavior runs *through the `Backend` trait*
//! on a **live, in-process flowlog-rs `DatalogIncrementalEngine`** (Differential
//! Dataflow), and matches the reference backend (`egglog_bridge::EGraph`)
//! round-for-round.
//!
//! The load-bearing mapping: **one `run_rules` call = one flowlog `commit()` =
//! one hop.** The bundled `transitive_step.dl` is **non-recursive**
//! (`hop(x,z) :- path(x,y), edge(y,z).`), so a single `commit()` performs
//! exactly one round of the join over the freshly-staged `path` delta. The host
//! folds each epoch's `hop` deltas into a Rust-side materialized mirror and
//! re-stages the new `path` rows for the next round (the PLAN §4 design-A
//! mirror + the proven Feldera host-feedback loop). N calls = N hops, bounded.
//!
//! ## The build-time-fixed `.dl` (and the FlowLog crux)
//!
//! flowlog compiles `.dl` -> Rust at BUILD time (`build.rs`), but egglog defines
//! rules at RUNTIME. For the M1 PROOF a build-time-fixed `.dl` is acceptable
//! (per the brief). `run_rules` therefore **recognizes** the canonical
//! transitive-closure-step rule shape the frontend builds and routes it to the
//! bundled engine. Runtime rule installation (the FlowLog crux — the analog of
//! Feldera's static-circuit-rebuild risk, but harder) is investigated in
//! ../MILESTONE1.md and deferred to M2.
//!
//! ## Read-back through the trait
//!
//! `for_each` / `for_each_while` / `lookup_id` / `table_size` read the
//! materialized mirror, refreshed from `commit()`'s per-epoch deltas.

use std::any::Any;

use anyhow::Result;
use egglog_backend_trait::{
    Backend, BaseValueId, BaseValuePool, ColumnTy, ContainerPool, DefaultVal, ExternalFunction,
    ExternalFunctionId, FunctionConfig, FunctionId, FunctionRow, IterationReport, MergeFn,
    QueryEntry, ReportLevel, RuleBuilderOps, RuleId, Value,
};
use egglog_core_relations::Database;
use egglog_numeric_id::NumericId;
use hashbrown::{HashMap, HashSet};

mod base_values;
pub mod codegen;
pub mod compile;
pub mod dd_native;
mod engine;
mod external_func;
pub mod interpret;
mod rule_builder;
pub mod subprocess;
mod uf;

use base_values::base_values_as_pool_mut;
use compile::{pack_row, row_col, unpack_row, BodyOp, HeadOp, MergeMode, Row, RuleIr, Slot};
use external_func::ExternalFuncRegistry;

// ---------------------------------------------------------------------------
// Container pool stub (milestone 1 has no containers; mirror DuckDB/Feldera)
// ---------------------------------------------------------------------------

/// Zero-sized [`ContainerPool`] stub. Milestone 1 does not support container
/// sorts; all accessors are empty and registration errors.
pub(crate) struct FlowlogContainerPool;

impl ContainerPool for FlowlogContainerPool {
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
            "containers not supported on the FlowLog backend (milestone 1)"
        ))
    }
    fn for_each_dyn(&self, _ty: std::any::TypeId, _f: &mut dyn FnMut(Value, &dyn Any)) {}
    fn size(&self, _ty: std::any::TypeId) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Relation metadata
// ---------------------------------------------------------------------------

/// What we remember about each registered relation/function.
pub(crate) struct RelationInfo {
    #[allow(dead_code)]
    name: String,
    /// Number of columns (including the output column for functions).
    pub(crate) arity: usize,
    /// True for functions/constructors that have an output column.
    #[allow(dead_code)]
    has_output: bool,
    /// How functional-dependency conflicts are resolved.
    pub(crate) merge: MergeMode,
    /// True iff this function uses identity-on-miss lookup semantics
    /// (`DefaultVal::Identity`): an action-position lookup of an absent key
    /// resolves to the key itself, with no row inserted. Used by the
    /// canonicalize-at-creation encoding for the flat UF-index `@UF_Sf`.
    pub(crate) identity_on_miss: bool,
}

// ---------------------------------------------------------------------------
// EGraph
// ---------------------------------------------------------------------------

/// How `run_rules` executes the recognized step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecMode {
    /// M1: drive a build-time-fixed in-process flowlog engine (recognized
    /// transitive-closure step only; used by `run_n_proof`).
    InProcess,
    /// M2: translate the recognized step to `.dl`, compile a driver crate
    /// (cached by rule-set hash), and drive it as a subprocess over a pipe
    /// (used by `run_n_shellout_proof`).
    ShellOut,
    /// M3 (the frontend default): the in-process iteration driver
    /// ([`interpret::run_iteration`]) runs general multi-atom bodies +
    /// primitives + head actions. Every body table-atom join runs on the
    /// in-process Differential-Dataflow dataflow ([`dd_native`]); the primitive
    /// tail + head actions are applied host-side. This is what real `.egg` files
    /// run on. There is NO host nested-loop fallback — an unsupported rule shape
    /// panics with a reason.
    Interpret,
}

/// The FlowLog-backed egraph.
pub struct EGraph {
    relations: Vec<RelationInfo>,
    /// Rule slots; `None` = freed.
    pub(crate) rules: Vec<Option<RuleIr>>,
    /// Rust-side materialized mirror: the accumulated contents of each
    /// relation, kept in sync with the flowlog engine's per-epoch `commit()`
    /// deltas. This is what `for_each` / `lookup_id` / `table_size` read.
    ///
    /// The set per function is shared by `Rc` so the per-iteration read
    /// snapshot (see `interpret::run_iteration`) is O(#functions), not
    /// O(state): mutations copy-on-write via `Rc::make_mut` only the functions
    /// actually changed while a snapshot is alive (the Feldera flatness model).
    pub(crate) mirror: HashMap<FunctionId, std::rc::Rc<HashSet<Row>>>,
    /// The live, in-process flowlog-rs `DatalogIncrementalEngine`, wrapped so
    /// the trait code stays free of the generated-symbol details. `None` until
    /// the first transitive-closure-step `run_rules`, which is where we know
    /// which `FunctionId`s play the `edge` / `path` / head roles.
    flow: Option<engine::FlowEngine>,
    /// Execution mode: in-process flowlog engine (M1) or the M2 shell-out
    /// driver subprocess (runtime rule installation via codegen + compile).
    mode: ExecMode,
    /// The M2 shell-out driver: a subprocess that embeds a flowlog engine
    /// compiled at runtime from the rule IR. `None` until the first
    /// `run_rules` under [`ExecMode::ShellOut`] (where the rule is known).
    driver: Option<subprocess::DriverHandle>,
    /// Per-round feedback buffer for the shell-out path: the new `path` rows
    /// derived last round, to re-stage as the next round's `insert path` delta
    /// (the bounded host-feedback loop, mirrored across the pipe).
    pending_path: Vec<(i32, i32)>,
    /// A core-relations [`Database`] used purely as the base-value / primitive
    /// engine, so `Value`s are bit-for-bit identical to the reference backend.
    db: Database,
    container_pool: FlowlogContainerPool,
    pub(crate) external_funcs: ExternalFuncRegistry,
    /// Monotonic fresh-id counter for `fresh_id` / `add_term`.
    pub(crate) next_id: u32,
    report_level: ReportLevel,
    /// Diagnostics: rule firings whose body join ran on the DD engine.
    pub(crate) dd_rule_runs: u64,
    /// Atom-less rules (`(rule () …)`) fire ONCE; an entry here marks a rule
    /// index as already fired. The DD dataflow has no input relation to drive an
    /// atom-less body, so this fired-marker is the one piece of seminaive
    /// bookkeeping the DD path reuses (see `interpret::dd_native_bindings`).
    /// `free_rule` removes the entry so a re-installed rule can fire again.
    pub(crate) seen: HashMap<usize, ()>,
    /// Per-rule persistent DD join, built once (lazily) and stepped each
    /// iteration with only the per-relation delta (the Feldera `persistent` analog).
    pub(crate) dd_native: HashMap<usize, dd_native::PersistentDdJoin>,
    /// Per-rule, per-function last-fed row snapshot `Rc`, for computing the
    /// signed delta vs what the persistent DD join was last fed (Feldera `fed`).
    pub(crate) dd_native_fed: HashMap<usize, HashMap<FunctionId, std::rc::Rc<HashSet<Row>>>>,
    /// Per-RULESET fused DD join (one shared timely worker + one dataflow for the
    /// whole ruleset), keyed by the sorted live rule-index list — the FlowLog
    /// analog of feldera's `FusedJoin`. This is the perf-critical join path;
    /// `dd_native` (per-rule) is retained only for the bridge-level unit tests.
    pub(crate) dd_fused: HashMap<Vec<usize>, dd_native::FusedDdJoin>,
    /// Per-ruleset, per-function last-fed row snapshot `Rc`, for computing the
    /// signed delta fed into the FUSED join's shared inputs (the fused `fed`).
    pub(crate) dd_fused_fed: HashMap<Vec<usize>, HashMap<FunctionId, std::rc::Rc<HashSet<Row>>>>,
    /// `--native-uf --flowlog`: drive PR #782's UF-backed-table encoding through
    /// FlowLog's fast HOST-PASS rebuild (instead of the onchange-driven rebuild
    /// rules that would run on the DD dataflow and regress the ~2.1× win).
    ///
    /// The #782 term encoder emits a UF-backed `:impl displaced-union-find`
    /// function `@UF_Sf (S) S` per eq-sort (via [`Backend::add_uf_function`])
    /// plus a `@UFChange_S` onchange relation and `@rebuild_rule*` /
    /// `@uf_change_drain_rule*` maintenance rules. When this is on we honour the
    /// `add_uf_function` request (a real `UfTable` + find-or-self canon-prim),
    /// route union writes `(set (@UF_Sf lhs) rhs)` into the in-core UF, suppress
    /// the maintenance rules by name, and re-canonicalize view rows host-side.
    /// Must be enabled (via [`EGraph::enable_native_uf`]) before any UF function
    /// is registered. Pure backend interception — the encoder is unchanged.
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
    /// Every id ever ingested into a native UF (per `@UF_Sf` function), so the
    /// iteration-boundary drain can bound its displaced-set bookkeeping without
    /// exposing the `UfTable`'s internal node map.
    pub(crate) native_uf_members: HashMap<FunctionId, HashSet<u32>>,
    /// The `--fast-rebuild` axis. Two distinct roles depending on `--native-uf`:
    ///   * RELATIONAL (without `--native-uf`): drive the fused DD worker's δUF
    ///     rebuild that drops the always-empty `δview ⋈ uf_old` term. OR'd with
    ///     the `FLOWLOG_DELTA_REBUILD` env var at build time.
    ///   * NATIVE-UF: select the CUSTOM rebuild (the off-engine host-pass
    ///     reverse-index δuf scan, `interpret::native_uf_rebuild_envs`) instead of
    ///     the PURE-ENGINE rebuild (the DD-engine `view ⋈ @DispΔ` join). The host
    ///     UfTable is shared by both — only the REBUILD differs.
    ///
    /// Off by default.
    pub(crate) fast_rebuild: bool,
    /// Native-UF delta rebuild: the set of ids whose canonical changed during the
    /// PREVIOUS iteration's `drain_pending` (per `@UF_Sf` function). Stashed by
    /// `native_uf_drain_all` instead of discarded, then fed into the synthetic
    /// `@DispΔ` displaced-ids relation at the start of the NEXT `run_iteration`
    /// (before the read snapshot) so the DD-engine rebuild's `view ⋈ δdisplaced`
    /// join sees the prior round's displaced ids (the rebuild runs BEFORE the
    /// drain). See [`native_uf_disp_rel`] / `interpret::sync_displaced_relations`.
    pub(crate) native_uf_displaced_prev: HashMap<FunctionId, Vec<i64>>,
    /// Native-UF rebuild: set in `run_iteration`/`native_uf_rebuild_envs` when
    /// this call's `@rebuilding` ruleset consumed [`native_uf_displaced_prev`]
    /// (the PURE-ENGINE `view ⋈ δdisplaced` DD join consumed the fed `@DispΔ`
    /// rows, or the CUSTOM host-pass scoped scan read the displaced sets). The
    /// iteration-boundary drain reads it to RESET `native_uf_displaced_prev`
    /// exactly once per rebuild round, so the next round carries only ids
    /// displaced AFTER this rebuild (one UF backs many views, all fused into the
    /// one rebuild `run_iteration`). Cleared each drain.
    pub(crate) native_uf_rebuild_ran: bool,
    /// CUSTOM rebuild (`--native-uf --fast-rebuild`): reverse index per VIEW
    /// function — eq-sort id value → the view rows that hold it in an eq-sort
    /// column. Maintained incrementally as view rows enter/leave the mirror (see
    /// `index_insert_row` / `index_remove_row`). Built lazily on the first
    /// rebuild scan for a view func (a one-time full scan that also seeds this
    /// index); thereafter the scoped scan looks up only the displaced ids' rows
    /// here. Absence of an entry means "not yet built" → fall back to a full scan
    /// that builds it. Maintained ONLY on the custom path (empty on PURE-ENGINE).
    pub(crate) native_uf_rev_index: HashMap<FunctionId, HashMap<u32, HashSet<Row>>>,
    /// CUSTOM rebuild (`--native-uf --fast-rebuild`): the eq-sort columns of a
    /// VIEW function, as `(column index, UF func)` pairs (the `col_uf` mapping
    /// `native_uf_rebuild_envs` derives from a `@rebuild_rule`). Cached on the
    /// first rebuild scan so the index-maintenance hooks know which columns of a
    /// view row to index. Populated ONLY on the custom path.
    pub(crate) native_uf_view_cols: HashMap<FunctionId, Vec<(usize, FunctionId)>>,
    /// PURE-ENGINE rebuild (`--native-uf`, no `--fast-rebuild`): per `@UF_Sf`
    /// function, the [`FunctionId`] of a synthetic arity-1 "displaced ids"
    /// relation `@DispΔ`. The PURE-ENGINE rebuild feeds it each iteration and
    /// joins the view against it on the DD dataflow (`rebuild_rule_dd_ir`,
    /// `sync_displaced_relations`); the CUSTOM path leaves it empty.
    /// It is the native analog of the relational `@UF_Sf` flat-index relation
    /// the plain-`--flowlog` rebuild joins the view against: instead of routing
    /// the rebuild to a host pass, we feed this relation's mirror — at the start
    /// of each `run_iteration`, before the read snapshot — with the previous
    /// round's displaced ids ([`native_uf_displaced_prev`]), and rewrite each
    /// `@rebuild_rule` to join the view against it (the empty `@UFChange_S`
    /// onchange atom is stripped, exactly the DuckDB host-pass rewrite, but the
    /// surviving `view ⋈ δdisplaced` join then runs on the DD dataflow,
    /// seminaive on the fed δ). The `@canon_S` guard / head finds stay native
    /// (host-side prim tail), so the JOIN moves onto DD while the O(1) find
    /// stays in-core — no full re-scan, no eqsat blowup. Registered in
    /// [`Backend::add_uf_function`].
    pub(crate) native_uf_disp_rel: HashMap<FunctionId, FunctionId>,
    /// Cache of the DD-rewritten `@rebuild_rule` IR, keyed by rule index: the
    /// onchange atom stripped and a `@DispΔ(eqsort_col)` atom appended, so the
    /// rule lowers to `view ⋈ δdisplaced` on the fused DD worker. Built lazily
    /// the first time a rebuild rule is routed (the source IR never changes).
    pub(crate) native_uf_rebuild_dd_ir: HashMap<usize, RuleIr>,
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for EGraph {
    fn drop(&mut self) {
        // Step-0 profile dump (gated FLOWLOG_DD_PROF): #workers, #InputSessions,
        // and the worker.step vs host-side prim/delta split. No-op otherwise.
        dd_native::dd_prof_dump();
        // Per-ruleset profile dump (gated FLOWLOG_DD_RULESET_PROF): attribute
        // DD wall time to the ruleset NAME, sorted descending. No-op otherwise.
        dd_native::dd_ruleset_prof_dump();
    }
}

impl EGraph {
    /// Construct a fresh FlowLog-backed egraph (M1 in-process mode).
    pub fn new() -> Self {
        Self::with_mode(ExecMode::InProcess)
    }

    /// Construct a fresh FlowLog-backed egraph driven by the M2 shell-out
    /// runtime-codegen path (compile a driver subprocess from the rule IR).
    pub fn new_shellout() -> Self {
        Self::with_mode(ExecMode::ShellOut)
    }

    /// Construct a fresh FlowLog-backed egraph driven by the M3 iteration driver
    /// (general bodies + primitives + head actions; every body table-atom join
    /// runs on the in-process Differential-Dataflow dataflow). This is the
    /// constructor the egglog frontend uses (`EGraph::with_flowlog_backend`).
    pub fn new_interpret() -> Self {
        Self::with_mode(ExecMode::Interpret)
    }

    /// Construct with an explicit execution mode.
    pub fn with_mode(mode: ExecMode) -> Self {
        EGraph {
            relations: Vec::new(),
            rules: Vec::new(),
            mirror: HashMap::new(),
            flow: None,
            mode,
            driver: None,
            pending_path: Vec::new(),
            db: Database::new(),
            container_pool: FlowlogContainerPool,
            external_funcs: ExternalFuncRegistry::default(),
            // Start at 1 so id 0 stays a "null"/padding sentinel.
            next_id: 1,
            report_level: ReportLevel::default(),
            dd_rule_runs: 0,
            seen: HashMap::new(),
            dd_native: HashMap::new(),
            dd_native_fed: HashMap::new(),
            dd_fused: HashMap::new(),
            dd_fused_fed: HashMap::new(),
            native_uf_enabled: false,
            native_ufs: HashMap::new(),
            native_uf_canon_prim: HashMap::new(),
            native_uf_members: HashMap::new(),
            fast_rebuild: false,
            native_uf_displaced_prev: HashMap::new(),
            native_uf_rebuild_ran: false,
            native_uf_rev_index: HashMap::new(),
            native_uf_view_cols: HashMap::new(),
            native_uf_disp_rel: HashMap::new(),
            native_uf_rebuild_dd_ir: HashMap::new(),
        }
    }

    /// Turn on the native union-find path (`--native-uf --flowlog`). Must be
    /// called before any UF function is registered: [`Backend::add_uf_function`]
    /// checks this flag to decide whether to honour the PR #782 UF-backed-table
    /// request (a real in-core [`uf::UfTable`] + find-or-self canon-prim) or to
    /// bail (the default). Mirrors the DuckDB backend's `enable_native_uf`.
    pub fn enable_native_uf(&mut self) {
        self.native_uf_enabled = true;
    }

    /// Turn on `--fast-rebuild`. Two roles depending on `--native-uf`:
    ///   * RELATIONAL (no native UF): the fused DD worker's substep split that
    ///     drops the always-empty `δview ⋈ uf_old` rebuild term (sound under
    ///     canonicalize-at-creation). Equivalent to the `FLOWLOG_DELTA_REBUILD`
    ///     env var.
    ///   * NATIVE-UF: select the CUSTOM rebuild (the off-engine host-pass
    ///     reverse-index δuf scan) over the PURE-ENGINE rebuild (the DD-engine
    ///     `view ⋈ @DispΔ` join). The host UfTable is shared by both.
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
        if self.native_uf_enabled {
            // The native-UF DD rebuild is the δdisplaced-driven path, so stash
            // this round's displaced ids for the next round's `@DispΔ` feed.
            // If the `@rebuilding` ruleset ran earlier in THIS `run_iteration`
            // (so the `view ⋈ δdisplaced` DD join consumed the displaced sets),
            // RESET them now — every view sharing each UF has had its chance, so
            // the next round should see only ids displaced AFTER this rebuild
            // (the per-round reset that bounds the `@DispΔ` relation). The rebuild
            // itself enqueues no unions, so this clear precedes appending this
            // call's (typically empty) displaced ids.
            if self.native_uf_rebuild_ran {
                self.native_uf_displaced_prev.clear();
                self.native_uf_rebuild_ran = false;
            }
            // STASH this round's displaced ids per UF func for the `@DispΔ` feed.
            // We APPEND (not overwrite): a `run_iteration` that enqueues no unions
            // still reaches this drain, and the rebuild ruleset runs in a SEPARATE
            // `run_iteration` than the union-enqueuing user rules — so the
            // displaced ids must accumulate across the intervening (possibly
            // empty) drains until a rebuild consumes them.
            for (&func, uf) in self.native_ufs.iter_mut() {
                let d = uf.drain_displaced();
                displaced += d.len();
                if !d.is_empty() {
                    self.native_uf_displaced_prev
                        .entry(func)
                        .or_default()
                        .extend(d);
                }
            }
        } else {
            for uf in self.native_ufs.values_mut() {
                displaced += uf.displaced_len();
                let _ = uf.drain_displaced();
            }
        }
        displaced
    }

    pub(crate) fn info(&self, f: FunctionId) -> &RelationInfo {
        self.relations
            .get(f.rep() as usize)
            .unwrap_or_else(|| panic!("FunctionId({}) not registered", f.rep()))
    }

    /// Relation name for `f` (used by the per-ruleset profiler to detect `@uf`
    /// body atoms — the union-find tables are named `@UF_<sort>` / `@UF_<sort>f`).
    pub(crate) fn relation_name(&self, f: FunctionId) -> &str {
        &self.info(f).name
    }

    /// Insert a single row into the Rust mirror.
    fn mirror_insert(&mut self, f: FunctionId, row: Row) {
        // CUSTOM rebuild only: keep the reverse index in sync (no-op unless `f`'s
        // index is built). `index_insert_row` itself gates on `native_uf_enabled`
        // + a built index, so it is cheap when off; the explicit `fast_rebuild`
        // guard keeps the PURE-ENGINE path from ever building an index.
        if self.native_uf_enabled && self.fast_rebuild {
            self.index_insert_row(f, &row);
        }
        std::rc::Rc::make_mut(self.mirror.entry(f).or_default()).insert(row);
    }

    /// CUSTOM rebuild (`--native-uf --fast-rebuild`) reverse-index maintenance:
    /// record that `row` (now in the mirror of view func `f`) holds each of its
    /// eq-sort column values, so a later scoped rebuild can find it by a displaced
    /// id. No-op unless native-UF is on AND `f`'s eq-sort columns have been
    /// learned (i.e. the index for `f` was already built by the first full-scan
    /// rebuild). Indexing is keyed only on the eq-sort columns
    /// (`native_uf_view_cols`), so non-UF columns never bloat the index.
    pub(crate) fn index_insert_row(&mut self, f: FunctionId, row: &[u32]) {
        if !self.native_uf_enabled {
            return;
        }
        let Some(cols) = self.native_uf_view_cols.get(&f) else {
            return;
        };
        // Skip if the index for `f` is not yet built (first rebuild does the
        // full scan + seed). `native_uf_view_cols` is set together with the
        // index, so this is belt-and-suspenders.
        let Some(idx) = self.native_uf_rev_index.get_mut(&f) else {
            return;
        };
        for &(ci, _uf) in cols {
            if let Some(&v) = row.get(ci) {
                idx.entry(v)
                    .or_default()
                    .insert(row.to_vec().into_boxed_slice());
            }
        }
    }

    /// CUSTOM rebuild reverse-index maintenance: drop `row` from view func `f`'s
    /// reverse index (it just left the mirror). Mirror of
    /// [`index_insert_row`](Self::index_insert_row).
    pub(crate) fn index_remove_row(&mut self, f: FunctionId, row: &[u32]) {
        if !self.native_uf_enabled {
            return;
        }
        let Some(cols) = self.native_uf_view_cols.get(&f) else {
            return;
        };
        let Some(idx) = self.native_uf_rev_index.get_mut(&f) else {
            return;
        };
        for &(ci, _uf) in cols {
            if let Some(&v) = row.get(ci) {
                if let Some(rows) = idx.get_mut(&v) {
                    rows.remove(row);
                    if rows.is_empty() {
                        idx.remove(&v);
                    }
                }
            }
        }
    }

    /// Inherent accessor for the embedded [`BaseValues`] registry (the frontend
    /// extraction path threads `&BaseValues` through `reconstruct_termdag_base`).
    pub fn base_values_inner(&self) -> &egglog_core_relations::BaseValues {
        self.db.base_values()
    }

    /// Diagnostics: the number of rule firings whose body table-atom join ran on
    /// the in-process Differential-Dataflow dataflow. Every atom-bearing rule
    /// runs there (no host fallback); lets a test assert the join genuinely ran
    /// on DD.
    pub fn flowlog_dd_rule_runs(&self) -> u64 {
        self.dd_rule_runs
    }

    /// The functional-dependency merge mode of a function (from `add_table`).
    pub(crate) fn merge_mode(&self, f: FunctionId) -> MergeMode {
        self.info(f).merge
    }

    /// Evaluate a primitive through the embedded `Database` (the base-value /
    /// primitive engine). Both the host interpreter and the DD-join path's
    /// host-side primitive tail call this, so `Value`s are bit-for-bit
    /// identical to the reference backend. This is the inherent counterpart of
    /// Feldera M4's `eval_prim` trait method — flowlog keeps it inherent to
    /// preserve its zero-trait-change posture (no `egglog-backend-trait` edit).
    pub(crate) fn eval_prim_internal(
        &self,
        id: ExternalFunctionId,
        args: &[Value],
    ) -> Option<Value> {
        self.db
            .with_execution_state(|st| st.call_external_func(id, args))
    }

    /// Allocate a fresh id (the interpreter's eq-sort constructor hash-cons uses
    /// the same counter the trait's `fresh_id` advances).
    pub(crate) fn fresh_id_internal(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Resolve functional-dependency conflicts in a function's mirror set: for
    /// each key (input columns) keep a single output column chosen by the merge
    /// mode. Relations (whole-row key) are left untouched. Returns whether any
    /// row was actually changed/collapsed.
    ///
    /// INCREMENTAL (ported from Feldera's flatness work): the pre-call mirror is
    /// already FD-resolved (this runs every `run_rules` call), so only the keys
    /// whose rows were `set` this call (in `keys`) can newly conflict. We
    /// re-resolve just those keys instead of rebuilding the whole function —
    /// O(touched-keys), not O(state). The deterministic fold order (sort) is
    /// preserved PER KEY, which gives the same chosen output as the old
    /// whole-set sort+fold (the fold is independent across distinct keys), so
    /// `Old`/`New` conflicts arriving in the same iteration resolve stably.
    pub(crate) fn resolve_merge(&mut self, f: FunctionId, keys: &HashSet<Vec<u32>>) -> bool {
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
        {
            let set = std::rc::Rc::make_mut(self.mirror.get_mut(&f).unwrap());
            for r in &drop_rows {
                set.remove(r);
            }
            for r in &new_rows {
                set.insert(r.clone());
            }
        }
        // CUSTOM rebuild only: reflect the FD collapse in the reverse index (the
        // mutation block above borrowed `self.mirror` exclusively, so update the
        // index after it drops). No-op on PURE-ENGINE / when native-UF is off.
        if self.native_uf_enabled && self.fast_rebuild {
            for r in &drop_rows {
                self.index_remove_row(f, r);
            }
            for r in &new_rows {
                self.index_insert_row(f, r);
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Send + Sync (single-threaded use; same posture as DuckDB/Feldera)
// ---------------------------------------------------------------------------
//
// The flowlog engine owns a timely worker `JoinHandle` and channel endpoints,
// which are not all auto-`Sync`. As with the sibling backends, the egraph is
// only ever driven from a single thread, so we assert the bounds the trait
// requires. Concurrent multi-thread use is unsupported.
unsafe impl Send for EGraph {}
unsafe impl Sync for EGraph {}

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
            "FlowLog backend supports relations of arity <= {} (got {} for `{}`)",
            compile::MAX_ARITY,
            arity,
            config.name
        );
        // Relation vs function: a table is a **relation** (whole row is the key,
        // no output column to merge) iff it is nullary OR its last column is
        // `Unit` — the term encoder's view-table pattern
        // `(function @XView (...) Unit :merge old)` AND ordinary relations.
        // Otherwise the last column is a function OUTPUT, resolved by the merge
        // mode. This mirrors the DuckDB/Feldera backends' Unit-detection — NOT
        // `DefaultVal`, which is `Fail` for every custom function regardless of
        // whether it has a real output column. (This is the fix that took the
        // Feldera survey from 38 to 59 passing files.)
        let output_is_unit = config.schema.last().is_some_and(|t| match t {
            ColumnTy::Base(bv) => {
                let bvs = self.db.base_values();
                bvs.has_ty_by_id(std::any::TypeId::of::<()>())
                    && *bv == bvs.get_ty_by_id(std::any::TypeId::of::<()>())
            }
            _ => false,
        });
        let has_output = arity > 0 && !output_is_unit;
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
        let identity_on_miss = matches!(config.default, DefaultVal::Identity);
        self.relations.push(RelationInfo {
            name: config.name,
            arity,
            has_output,
            merge,
            identity_on_miss,
        });
        self.mirror.insert(id, std::rc::Rc::new(HashSet::new()));
        id
    }

    fn add_uf_function(
        &mut self,
        name: String,
        _onchange: Option<FunctionId>,
        proof: Option<egglog_backend_trait::UfProofConfig>,
    ) -> Result<(FunctionId, ExternalFunctionId)> {
        // Only the `--native-uf --flowlog` path supports PR #782's UF-backed
        // function. With the flag off the relational UF encoding is used (no
        // `add_uf_function` calls), so a bail here is unreachable in practice.
        if !self.native_uf_enabled {
            anyhow::bail!(
                "the FlowLog backend only supports `:impl displaced-union-find` \
                 functions under `--native-uf` (it drives PR #782's UF-backed \
                 encoding through either a DD-engine `view ⋈ δdisplaced` rebuild \
                 or, under `--fast-rebuild`, a custom host-pass rebuild)"
            );
        }
        // Proof mode is a later step (TERM mode only for now): a provenance-
        // tracking UF would need the `@UFChange_S` proof column composed in a
        // leader-change callback, which neither native-UF rebuild runs.
        if proof.is_some() {
            anyhow::bail!(
                "the FlowLog backend does not yet support proof-mode native-UF \
                 functions (`--native-uf` is TERM mode only on FlowLog)"
            );
        }

        // Register the UF function as a real relation: schema `(S) S` (arity 2,
        // output column, `Min` merge — the union-find leader). The mirror is
        // never populated by writes (union `set`s are intercepted into the
        // in-core UF), but the relation must exist so its FunctionId resolves in
        // `info` / `lookup_id` (the extractor's `find_canonical` reads it).
        let id = FunctionId::new(self.relations.len() as u32);
        self.relations.push(RelationInfo {
            name,
            arity: 2,
            has_output: true,
            merge: MergeMode::Min,
            identity_on_miss: false,
        });
        self.mirror.insert(id, std::rc::Rc::new(HashSet::new()));
        self.native_ufs.insert(id, uf::UfTable::new());
        self.native_uf_members.insert(id, HashSet::new());

        // Synthetic arity-1 "displaced ids" relation `@DispΔ` for the PURE-ENGINE
        // rebuild (the native analog of the relational `@UF_Sf` flat index the
        // plain-`--flowlog` rebuild joins against). On the PURE-ENGINE path it is
        // fed each iteration — before the read snapshot — with the previous
        // round's displaced ids, and the rewritten `@rebuild_rule` joins the
        // view's eq-sort column against it on the fused DD worker
        // (`view ⋈ δdisplaced`). On the CUSTOM (`--fast-rebuild`) path it stays
        // empty. Registered unconditionally (harmless when unused): plain relation
        // (whole-row key), never read back through the trait.
        let disp_id = FunctionId::new(self.relations.len() as u32);
        self.relations.push(RelationInfo {
            name: format!("@Disp\u{0394}_{}", id.rep()),
            arity: 1,
            has_output: false,
            merge: MergeMode::Relation,
            identity_on_miss: false,
        });
        self.mirror
            .insert(disp_id, std::rc::Rc::new(HashSet::new()));
        self.native_uf_disp_rel.insert(id, disp_id);

        // The find-or-self canon-prim. A real, freeable `ExternalFunctionId`,
        // but the interpreter intercepts calls to it (see `native_uf_canon_prim`)
        // and answers `find_ro` from the in-core UF — the registered stub is
        // never actually invoked through the `Database`.
        let canon = self.register_external_func(Box::new(external_func::CanonStub));
        self.native_uf_canon_prim.insert(canon, id);
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
            if (0..inputs_len).all(|i| row_col(row, i) == key[i].rep()) {
                return Some(Value::new(row_col(row, inputs_len)));
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
    }

    fn add_term(&mut self, func: FunctionId, inputs: &[Value]) -> Value {
        let id = self.fresh_id();
        let mut full = inputs.to_vec();
        full.push(id);
        let arity = self.info(func).arity;
        assert_eq!(full.len(), arity, "add_term: arity mismatch");
        self.mirror_insert(func, pack_row(&full));
        id
    }

    fn insert_rows(&mut self, table: FunctionId, rows: &[Vec<Value>]) {
        let arity = self.info(table).arity;
        for row in rows {
            assert_eq!(row.len(), arity, "insert_rows: row arity mismatch");
            self.mirror_insert(table, pack_row(row));
        }
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
        // CUSTOM rebuild only: drop the func's reverse index too (the index for
        // an emptied func has no live rows). Clearing to an empty map keeps it
        // "built" so the incremental hooks stay active; the next inserts
        // repopulate it. No-op on PURE-ENGINE.
        if self.native_uf_enabled && self.fast_rebuild {
            if let Some(idx) = self.native_uf_rev_index.get_mut(&func) {
                idx.clear();
            }
        }
        // The DD `fed` snapshots (`dd_native_fed` / `dd_fused_fed`) are diffed
        // against the live mirror each iteration, so clearing the mirror is
        // picked up as a retraction delta automatically — no extra bookkeeping
        // needed here.
    }

    fn base_values(&self) -> &egglog_core_relations::BaseValues {
        self.db.base_values()
    }

    fn with_execution_state_dyn(
        &self,
        _f: &mut dyn FnMut(&mut egglog_backend_trait::ExecutionState<'_>),
    ) {
        unimplemented!("with_execution_state is not supported on the FlowLog backend")
    }

    fn action_registry_any(&self) -> &(dyn Any + Send + Sync) {
        unimplemented!("action_registry is not supported on the FlowLog backend")
    }

    // -- rule management ----------------------------------------------------

    fn new_rule<'a>(&'a mut self, desc: &str, _seminaive: bool) -> Box<dyn RuleBuilderOps + 'a> {
        // Seminaive is native to differential dataflow; the flag is accepted
        // for parity and ignored.
        Box::new(rule_builder::FlowlogRuleBuilder::new(self, desc))
    }

    fn free_rule(&mut self, id: RuleId) {
        if let Some(slot) = self.rules.get_mut(id.rep() as usize) {
            *slot = None;
            let i = id.rep() as usize;
            self.seen.remove(&i);
            self.dd_native.remove(&i);
            self.dd_native_fed.remove(&i);
            // Any fused ruleset that included this rule is now stale: drop it so
            // it is rebuilt (without the freed rule) on the next `run_rules`.
            self.dd_fused.retain(|key, _| !key.contains(&i));
            self.dd_fused_fed.retain(|key, _| !key.contains(&i));
        }
    }

    fn run_rules(&mut self, rules: &[RuleId]) -> Result<IterationReport> {
        // ONE egglog iteration = ONE flowlog `commit()` = ONE transitive-
        // closure hop. The frontend calls this N times for `(run N)`.
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

        let changed = match self.mode {
            ExecMode::Interpret => interpret::run_iteration(self, &live)?,
            ExecMode::InProcess | ExecMode::ShellOut => self.run_one_hop(&live)?,
        };

        let mut report = IterationReport::default();
        report.rule_set_report.changed = changed;
        Ok(report)
    }

    fn flush_updates(&mut self) -> bool {
        // Seed inserts land in the mirror immediately; flowlog staging happens
        // inside `run_rules` (one commit per call). No separate flush.
        false
    }

    // -- primitives ---------------------------------------------------------

    fn register_external_func(
        &mut self,
        func: Box<dyn ExternalFunction + 'static>,
    ) -> ExternalFunctionId {
        let func2 = dyn_clone::clone_box(&*func);
        let id = self.db.add_external_function(func);
        self.external_funcs.add_func_at(id, func2);
        id
    }

    fn free_external_func(&mut self, func: ExternalFunctionId) {
        self.db.free_external_function(func);
        self.external_funcs.free(func);
    }

    fn new_panic(&mut self, message: String) -> ExternalFunctionId {
        let panic_fn = external_func::PanicFunc::new(message.clone());
        let id = self.db.add_external_function(Box::new(panic_fn));
        self.external_funcs.add_panic_at(id, message);
        id
    }

    fn eval_prim(&self, id: ExternalFunctionId, args: &[Value]) -> Option<Value> {
        // Delegate to the inherent stopgap (`eval_prim_internal`) that the
        // host-side primitive tail already uses, so primitive evaluation
        // stays bit-for-bit identical to the reference backend. The merged
        // `Backend` trait (from the Feldera milestones) now requires this entry
        // point; flowlog satisfies it without changing its internal posture.
        self.eval_prim_internal(id, args)
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
            log::info!("== FlowLog relation `{}` ({} rows) ==", info.name, n);
        }
    }

    // -- cloning ------------------------------------------------------------

    fn clone_boxed(&self) -> Box<dyn Backend> {
        // Push/pop snapshot support is a later milestone: a running flowlog
        // dataflow can't be cloned, but the *state* (mirror + rule IR +
        // relation metadata) can be replayed into a fresh engine. Not needed
        // for milestone 1.
        unimplemented!(
            "FlowLog backend clone_boxed (push/pop) is deferred (snapshot-and-replay \
             into a fresh engine)"
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

// ---------------------------------------------------------------------------
// The flowlog-driven per-iteration hop
// ---------------------------------------------------------------------------

/// The roles played by the three `FunctionId`s in a transitive-closure-step
/// rule `head(x, z) :- path(x, y), edge(y, z)`, recognized from the rule IR.
struct StepShape {
    /// The body atom whose middle (join) column also appears in the *other*
    /// body atom's first column — i.e. the "left/path" relation contributing
    /// the head's first column.
    path: FunctionId,
    /// The "right/edge" relation contributing the head's last column.
    edge: FunctionId,
    /// The head relation (where new rows are derived). In egglog's
    /// transitive-closure step this equals `path`.
    head: FunctionId,
    /// Column index in `path` rows of the head's first variable (`x`).
    path_x: usize,
    /// Column index in `path` rows of the shared join variable (`y`).
    path_y: usize,
    /// Column index in `edge` rows of the shared join variable (`y`).
    edge_y: usize,
    /// Column index in `edge` rows of the head's last variable (`z`).
    edge_z: usize,
    /// The (constant) value to write in any non-(x,z) head columns, by column
    /// index — e.g. the fixed FD value of a `(x,y) -> value` table.
    head_consts: Vec<(usize, u32)>,
    /// Which head column carries `x` and which carries `z`.
    head_x: usize,
    head_z: usize,
    /// Head arity.
    head_arity: usize,
}

impl EGraph {
    /// Run exactly one transitive-closure hop through the bundled flowlog
    /// incremental engine: stage this round's new `path` rows, `commit()` once,
    /// fold the resulting `hop` deltas into the mirror. Returns whether the
    /// mirror changed.
    fn run_one_hop(&mut self, live: &[usize]) -> Result<bool> {
        match self.mode {
            ExecMode::InProcess => self.run_one_hop_inprocess(live),
            ExecMode::ShellOut => self.run_one_hop_shellout(live),
            ExecMode::Interpret => {
                unreachable!("Interpret mode routes through interpret::run_iteration")
            }
        }
    }

    /// M1 in-process path: drive the build-time-fixed flowlog engine.
    fn run_one_hop_inprocess(&mut self, live: &[usize]) -> Result<bool> {
        // Exactly one rule per `run_rules`: the transitive-closure step. (The
        // frontend's `(run N)` loop calls this with the single user rule N
        // times.)
        let shape = self.recognize_step(live)?;

        // Lazily build the engine and do the initial seeding hop. On the first
        // call we feed ALL current `edge` and `path` rows; the non-recursive
        // join's first commit yields the 1-hop extension. On subsequent calls
        // we feed only the previous round's NEW path rows so each commit is
        // exactly one further hop (bounded extension, not saturation).
        if self.flow.is_none() {
            self.flow = Some(engine::FlowEngine::new());
            // Seed all edges (they never change across rounds in this proof).
            let edge_rows = self.engine_rows(shape.edge, shape.edge_y, shape.edge_z);
            // Seed all current path rows as the first delta.
            let path_rows = self.engine_rows(shape.path, shape.path_x, shape.path_y);
            let new_hops = {
                let flow = self.flow.as_mut().unwrap();
                flow.insert_edge(&edge_rows);
                flow.insert_path(&path_rows);
                flow.commit_hop()
            };
            return Ok(self.fold_hops(&shape, &new_hops));
        }

        // Subsequent rounds: re-stage the rows derived LAST round as the new
        // `path` delta (tracked in the engine wrapper), commit one hop, fold.
        let to_feed = self.flow.as_mut().unwrap().take_pending_path();
        if to_feed.is_empty() {
            // Nothing new to extend with: no further hop is possible.
            return Ok(false);
        }
        let new_hops = {
            let flow = self.flow.as_mut().unwrap();
            flow.insert_path(&to_feed);
            flow.commit_hop()
        };
        Ok(self.fold_hops(&shape, &new_hops))
    }

    /// M2 shell-out path: translate the runtime rule to `.dl`, compile (or
    /// reuse a cached) driver subprocess, and drive ONE bounded hop over the
    /// pipe. Same host-feedback loop as the in-process path, but the engine
    /// lives in a subprocess compiled from the rule defined at runtime.
    fn run_one_hop_shellout(&mut self, live: &[usize]) -> Result<bool> {
        let shape = self.recognize_step(live)?;

        // First call: emit the `.dl` from the runtime rule, build/cache + spawn
        // the driver, then stage all current `edge` + `path` rows and commit
        // the first (1-hop) epoch.
        if self.driver.is_none() {
            let dl = codegen::emit_dl();
            let mut handle = subprocess::DriverHandle::build_or_cached(&dl)?;
            handle.spawn()?;
            self.driver = Some(handle);

            let edge_rows = self.engine_rows(shape.edge, shape.edge_y, shape.edge_z);
            let path_rows = self.engine_rows(shape.path, shape.path_x, shape.path_y);
            let new_hops = {
                let drv = self.driver.as_mut().unwrap();
                for (a, b) in &edge_rows {
                    drv.insert(codegen::REL_EDGE, *a, *b)?;
                }
                for (a, b) in &path_rows {
                    drv.insert(codegen::REL_PATH, *a, *b)?;
                }
                drv.commit()?
            };
            return Ok(self.fold_hops_shellout(&shape, &new_hops));
        }

        // Subsequent rounds: re-stage last round's NEW path rows as the next
        // `insert path` delta, commit one further hop, fold.
        let to_feed = std::mem::take(&mut self.pending_path);
        if to_feed.is_empty() {
            return Ok(false);
        }
        let new_hops = {
            let drv = self.driver.as_mut().unwrap();
            for (a, b) in &to_feed {
                drv.insert(codegen::REL_PATH, *a, *b)?;
            }
            drv.commit()?
        };
        Ok(self.fold_hops_shellout(&shape, &new_hops))
    }

    /// Fold the shell-out driver's `hop` deltas into the mirror and stage the
    /// new `path` rows for next round. Mirrors `fold_hops` but writes the
    /// feedback buffer on the egraph (the subprocess is stateless across
    /// rounds w.r.t. host feedback).
    fn fold_hops_shellout(&mut self, shape: &StepShape, hops: &[(i32, i32, i32)]) -> bool {
        let mut changed = false;
        let mut new_path_feed: Vec<(i32, i32)> = Vec::new();
        for &(x, z, diff) in hops {
            if diff <= 0 {
                continue;
            }
            let mut full = vec![0u32; shape.head_arity];
            for &(ci, cv) in &shape.head_consts {
                full[ci] = cv;
            }
            full[shape.head_x] = x as u32;
            full[shape.head_z] = z as u32;
            let row: Row = full.into_boxed_slice();
            let set = std::rc::Rc::make_mut(self.mirror.entry(shape.head).or_default());
            if set.insert(row) {
                changed = true;
                new_path_feed.push((x, z));
            }
        }
        self.pending_path = new_path_feed;
        changed
    }

    /// Collect the `(a, b)` projection of relation `f`'s mirror rows at the
    /// given two column indices, as engine tuples.
    fn engine_rows(&self, f: FunctionId, ca: usize, cb: usize) -> Vec<(i32, i32)> {
        let mut out = Vec::new();
        if let Some(set) = self.mirror.get(&f) {
            for row in set.iter() {
                out.push((row_col(row, ca) as i32, row_col(row, cb) as i32));
            }
        }
        out
    }

    /// Fold this epoch's `hop` deltas `(x, z)` into the head-relation mirror.
    /// Records the genuinely-new head rows so the *next* round can feed them
    /// back to the engine as the next `path` delta. Returns whether the mirror
    /// gained any rows.
    fn fold_hops(&mut self, shape: &StepShape, hops: &[(i32, i32, i32)]) -> bool {
        let mut changed = false;
        let mut new_path_feed: Vec<(i32, i32)> = Vec::new();
        for &(x, z, diff) in hops {
            if diff <= 0 {
                // M1 is monotone (no retraction); ignore non-positive diffs.
                continue;
            }
            // Build the full head row: x and z in their columns, constants
            // elsewhere.
            let mut full = vec![0u32; shape.head_arity];
            for &(ci, cv) in &shape.head_consts {
                full[ci] = cv;
            }
            full[shape.head_x] = x as u32;
            full[shape.head_z] = z as u32;
            let row: Row = full.into_boxed_slice();
            let set = std::rc::Rc::make_mut(self.mirror.entry(shape.head).or_default());
            if set.insert(row) {
                changed = true;
                // The head IS the path relation in a transitive-closure step,
                // so a new head row is a new path row to extend with next round.
                // Feed its (x, y=z) projection — for path the join column is
                // path_y, and the new row's path_y position holds `z`.
                new_path_feed.push((x, z));
            }
        }
        if let Some(flow) = self.flow.as_mut() {
            flow.set_pending_path(new_path_feed);
        }
        changed
    }

    /// Recognize the transitive-closure-step shape from the single live rule.
    fn recognize_step(&self, live: &[usize]) -> Result<StepShape> {
        if live.len() != 1 {
            return Err(anyhow::anyhow!(
                "FlowLog backend (M1) runs exactly one rule per `run_rules` \
                 (got {}); multi-rule rulesets are an M2 feature.",
                live.len()
            ));
        }
        let ir = self.rules[live[0]]
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("FlowLog backend: rule slot freed"))?;

        // Body: exactly two table atoms.
        let mut atoms = Vec::new();
        for op in &ir.body {
            match op {
                BodyOp::Atom(a) => atoms.push(a),
                BodyOp::Prim { .. } => {
                    return Err(anyhow::anyhow!(
                        "FlowLog backend (M1): primitive body atoms are not \
                         supported (rule `{}`)",
                        ir.name
                    ))
                }
            }
        }
        if atoms.len() != 2 {
            return Err(anyhow::anyhow!(
                "FlowLog backend (M1) supports a two-atom join body \
                 (transitive-closure step); rule `{}` has {} body atoms.",
                ir.name,
                atoms.len()
            ));
        }
        // Head: exactly one `set`.
        let set_head = ir.head.iter().find_map(|h| match h {
            HeadOp::Set { func, slots } => Some((*func, slots)),
            _ => None,
        });
        let (head_func, head_slots) = set_head.ok_or_else(|| {
            anyhow::anyhow!("FlowLog backend (M1): rule `{}` has no `set` head", ir.name)
        })?;

        // Identify head variables (x = first, z = last) and constant columns.
        let head_vars: Vec<Option<u32>> = head_slots
            .iter()
            .map(|s| match s {
                Slot::Var(v) => Some(*v),
                Slot::Const(_) => None,
            })
            .collect();
        let head_var_positions: Vec<usize> = head_vars
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|_| i))
            .collect();
        if head_var_positions.len() != 2 {
            return Err(anyhow::anyhow!(
                "FlowLog backend (M1): head must bind exactly two variables \
                 (x, z); rule `{}`",
                ir.name
            ));
        }
        let head_x = head_var_positions[0];
        let head_z = head_var_positions[1];
        let var_x = head_vars[head_x].unwrap();
        let var_z = head_vars[head_z].unwrap();
        let head_consts: Vec<(usize, u32)> = head_slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Slot::Const(c) => Some((i, *c)),
                Slot::Var(_) => None,
            })
            .collect();
        let head_arity = head_slots.len();

        // Find which atom carries `x` (-> path) and which carries `z` (-> edge).
        let atom_has = |a: &compile::BodyAtom, var: u32| -> Option<usize> {
            a.slots
                .iter()
                .position(|s| matches!(s, Slot::Var(v) if *v == var))
        };
        // path atom contains x; edge atom contains z.
        let (path_atom, edge_atom) =
            if atom_has(atoms[0], var_x).is_some() && atom_has(atoms[1], var_z).is_some() {
                (atoms[0], atoms[1])
            } else if atom_has(atoms[1], var_x).is_some() && atom_has(atoms[0], var_z).is_some() {
                (atoms[1], atoms[0])
            } else {
                return Err(anyhow::anyhow!(
                    "FlowLog backend (M1): could not match head vars to body atoms \
                 in rule `{}` (expected transitive-closure-step shape)",
                    ir.name
                ));
            };

        let path_x = atom_has(path_atom, var_x).unwrap();
        let edge_z = atom_has(edge_atom, var_z).unwrap();

        // The shared join variable `y` is the var that appears in BOTH atoms
        // and is neither x nor z.
        let path_vars: HashSet<u32> = path_atom
            .slots
            .iter()
            .filter_map(|s| match s {
                Slot::Var(v) => Some(*v),
                _ => None,
            })
            .collect();
        let mut join_var = None;
        for s in &edge_atom.slots {
            if let Slot::Var(v) = s {
                if *v != var_x && *v != var_z && path_vars.contains(v) {
                    join_var = Some(*v);
                    break;
                }
            }
        }
        let join_var = join_var.ok_or_else(|| {
            anyhow::anyhow!(
                "FlowLog backend (M1): no shared join variable between body \
                 atoms in rule `{}`",
                ir.name
            )
        })?;
        let path_y = atom_has(path_atom, join_var).unwrap();
        let edge_y = atom_has(edge_atom, join_var).unwrap();

        Ok(StepShape {
            path: path_atom.func,
            edge: edge_atom.func,
            head: head_func,
            path_x,
            path_y,
            edge_y,
            edge_z,
            head_consts,
            head_x,
            head_z,
            head_arity,
        })
    }
}
