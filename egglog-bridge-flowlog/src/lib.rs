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
mod engine;
mod external_func;
mod rule_builder;
pub mod subprocess;

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
struct RelationInfo {
    #[allow(dead_code)]
    name: String,
    /// Number of columns (including the output column for functions).
    arity: usize,
    /// True for functions/constructors that have an output column.
    has_output: bool,
    /// How functional-dependency conflicts are resolved (M2+; recognized now).
    #[allow(dead_code)]
    merge: MergeMode,
}

// ---------------------------------------------------------------------------
// EGraph
// ---------------------------------------------------------------------------

/// How `run_rules` executes the recognized step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecMode {
    /// M1: drive a build-time-fixed in-process flowlog engine.
    InProcess,
    /// M2: translate the runtime rule to `.dl`, compile a driver crate
    /// (cached by rule-set hash), and drive it as a subprocess over a pipe.
    ShellOut,
}

/// The FlowLog-backed egraph.
pub struct EGraph {
    relations: Vec<RelationInfo>,
    /// Rule slots; `None` = freed.
    rules: Vec<Option<RuleIr>>,
    /// Rust-side materialized mirror: the accumulated contents of each
    /// relation, kept in sync with the flowlog engine's per-epoch `commit()`
    /// deltas. This is what `for_each` / `lookup_id` / `table_size` read.
    mirror: HashMap<FunctionId, HashSet<Row>>,
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
    next_id: u32,
    report_level: ReportLevel,
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
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
        }
    }

    fn info(&self, f: FunctionId) -> &RelationInfo {
        self.relations
            .get(f.rep() as usize)
            .unwrap_or_else(|| panic!("FunctionId({}) not registered", f.rep()))
    }

    /// Insert a single row into the Rust mirror.
    fn mirror_insert(&mut self, f: FunctionId, row: Row) {
        self.mirror.entry(f).or_default().insert(row);
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
        let has_output =
            arity > 0 && matches!(config.default, DefaultVal::FreshId | DefaultVal::Const(_));
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
        self.relations.push(RelationInfo {
            name: config.name,
            arity,
            has_output,
            merge,
        });
        self.mirror.insert(id, HashSet::new());
        id
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
            set.clear();
        }
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

        let changed = self.run_one_hop(&live)?;

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
            let set = self.mirror.entry(shape.head).or_default();
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
            let set = self.mirror.entry(shape.head).or_default();
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
