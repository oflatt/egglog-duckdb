//! An implementation of egglog-style queries on top of core-relations.
//!
//! This module translates a well-typed egglog-esque query into the abstractions
//! from the `core-relations` crate. The main higher-level functionality that it
//! implements are seminaive evaluation, default values, and merge functions.
//!
//! This crate is essentially involved in desugaring: it elaborates the encoding
//! of core egglog functionality, but it does not implement algorithms for
//! joins, union-finds, etc.

use std::{
    fmt::Debug,
    hash::Hash,
    iter, mem,
    ops::{Index, IndexMut},
    sync::{Arc, Mutex},
};

use crate::core_relations::{
    BaseValue, BaseValues, ColumnId, Constraint, ContainerValue, ContainerValues, CounterId,
    Database, DisplacedTable, DisplacedTableWithProvenance, ExecutionState, ExternalFunction,
    ExternalFunctionId, LeaderChange, MergeVal, Offset, PlanStrategy, ProofReason, ProofStep,
    SortedWritesTable, TableId, TaggedRowBuffer, Value, WrappedTable, make_external_func,
};
use crate::numeric_id::{DenseIdMap, DenseIdMapWithReuse, NumericId, define_id};
use egglog_core_relations as core_relations;
use egglog_numeric_id as numeric_id;
use egglog_reports::{IterationReport, ReportLevel, RuleSetReport};
use hashbrown::HashMap;
use indexmap::IndexSet;
use log::info;
use once_cell::sync::Lazy;
use smallvec::SmallVec;
use web_time::{Duration, Instant};

mod backend_impl;
pub mod macros;
pub(crate) mod rule;
#[cfg(test)]
mod tests;

pub use rule::{Function, RuleBuilder};

// Re-export the basic id and config types from the neutral trait crate so
// existing callers (and the frontend in `src/lib.rs`) continue to find them
// at `egglog_bridge::FunctionId` / `egglog_bridge::ColumnTy` / etc.
pub use egglog_backend_trait::{
    ColumnTy, DefaultVal, FunctionConfig, FunctionId, FunctionRow, MergeFn, QueryEntry, RuleId,
    Variable, VariableId,
};

use thiserror::Error;

/// A live registry of action handles for use by typed primitives.
///
/// Maps table name to [`TableAction`] (plus the shared [`UnionAction`]
/// and the default-panic external-function id) and is owned by the
/// bridge `EGraph`. The state wrappers (`PureState`/`ReadState`/
/// `WriteState`/`FullState`) live in the `egglog` crate; they read
/// from this registry at invoke time to back name-indexed action
/// methods. Held by the bridge `EGraph` inside an `Arc<RwLock<_>>`.
#[derive(Clone)]
pub struct ActionRegistry {
    table_actions: hashbrown::HashMap<String, TableAction>,
    union_action: UnionAction,
    default_panic_id: ExternalFunctionId,
}

impl ActionRegistry {
    pub(crate) fn new(union_action: UnionAction, default_panic_id: ExternalFunctionId) -> Self {
        Self {
            table_actions: hashbrown::HashMap::new(),
            union_action,
            default_panic_id,
        }
    }

    pub(crate) fn register_table(&mut self, name: String, action: TableAction) {
        self.table_actions.insert(name, action);
    }

    /// Look up the [`TableAction`] for a table by name, or `None` if
    /// no table with that name has been registered.
    pub fn lookup_table(&self, name: &str) -> Option<&TableAction> {
        self.table_actions.get(name)
    }

    /// Snapshot the registered table names and their current row counts.
    pub fn table_sizes(&self, state: &ExecutionState) -> Vec<(&str, usize)> {
        self.table_actions
            .iter()
            .map(|(name, action)| (name.as_str(), action.row_count(state)))
            .collect()
    }

    /// The shared [`UnionAction`] for this EGraph's union-find.
    pub fn union_action(&self) -> &UnionAction {
        &self.union_action
    }

    /// The default panic external function id, used by the egglog
    /// crate's `ActionView::panic`.
    pub fn default_panic_id(&self) -> ExternalFunctionId {
        self.default_panic_id
    }
}

define_id!(pub(crate) Timestamp, u32, "An abstract timestamp used to track execution of egglog rules");
impl Timestamp {
    fn to_value(self) -> Value {
        Value::new(self.rep())
    }
}

/// The state associated with an egglog program.
#[derive(Clone)]
pub struct EGraph {
    db: Database,
    uf_table: TableId,
    id_counter: CounterId,
    timestamp_counter: CounterId,
    rules: DenseIdMapWithReuse<RuleId, RuleInfo>,
    funcs: DenseIdMap<FunctionId, FunctionInfo>,
    panic_message: SideChannel<String>,
    /// This is a cache of all the different panic messages that we may use while executing rules
    /// against the EGraph. Oftentimes, these messages are generated dynamically: keeping this map
    /// around allows us to cache external function ids with repeat panic messages and they can
    /// also serve as a debugging tool in the case that the number of panic messages grows without
    /// bound.
    panic_funcs: HashMap<String, ExternalFunctionId>,
    report_level: ReportLevel,
    /// Live registry of name-indexed action handles. Shared (via
    /// `Arc<RwLock<_>>`) with state wrappers and primitive callbacks
    /// in the egglog crate so name-indexed action methods on
    /// [`WriteState`] / [`FullState`] can resolve table actions at
    /// invoke time. Mutated in place from [`add_table`](EGraph::add_table).
    action_registry: Arc<std::sync::RwLock<ActionRegistry>>,
    /// `--nativerb` (term-encoding native-UF, non-proof, native bridge only):
    /// maps a per-sort `@UF_Sf` UF-backed function to the set of `@<F>View`
    /// view functions that must be re-canonicalized against it by the ENGINE's
    /// native table rebuild (`apply_rebuild`), replacing the encoding-level
    /// `@rebuild_rule*` rules. Populated by
    /// [`register_nativerb_view`](EGraph::register_nativerb_view). Empty (and
    /// the rebuild loop is the unchanged `$uf`-only path) when the flag is off.
    nativerb_views: HashMap<FunctionId, Vec<FunctionId>>,
    /// `--native-merge` (term-encoding native-UF, non-proof, native bridge): maps
    /// an FD-keyed constructor `@<F>View` view function to the per-sort `@UF_Sf`
    /// UF-backed function that owns its eclass (OUTPUT) column. Populated by
    /// [`register_native_merge_view`](EGraph::register_native_merge_view), the
    /// uniform contract the dataflow/SQL backends use to route the FD-conflict
    /// congruence union. On the bridge the routing is already baked into the
    /// view's [`MergeFn::UnionIntoUf`] merge (which names the same `@UF_Sf`); this
    /// map records the association so the registration can assert the view's merge
    /// actually targets the registered UF. Empty when the flag is off.
    native_merge_uf: HashMap<FunctionId, FunctionId>,
    /// `--native-merge`: the UF-backed function each [`MergeFn::UnionIntoUf`]
    /// view's merge actually stages its FD-conflict union into, captured at
    /// `add_table` time. Used to assert the merge target matches the UF the
    /// frontend later registers via
    /// [`register_native_merge_view`](EGraph::register_native_merge_view). The
    /// view's [`FunctionId`] is its position in `funcs` (the value returned by the
    /// `add_table` that consumed the `UnionIntoUf` merge). Empty when the flag is
    /// off.
    native_merge_view_target: HashMap<FunctionId, FunctionId>,
}

pub type Result<T> = std::result::Result<T, anyhow::Error>;

#[derive(Error, Debug)]
pub enum FunctionConfigError {
    #[error("Merge functions cannot call UF-backed function {0}")]
    MergeUsesUf(String),
    #[error("TableAction does not support UF-backed function {0}")]
    TableActionUsesUf(String),
}

/// A callback that runs every time the leader of an equivalence class changes in
/// a UF-backed function's union-find. See [`LeaderChange`] for the details
/// passed.
pub type LeaderChangeCallback =
    Box<dyn Fn(&mut ExecutionState, LeaderChange) + Send + Sync + 'static>;

/// Configuration for a UF-backed function (see [`EGraph::add_uf_function`]).
pub struct UfFunctionConfig {
    pub name: String,
    pub on_leader_change: Option<LeaderChangeCallback>,
    /// Tables the `on_leader_change` callback reads from.
    pub read_deps: Vec<TableId>,
    /// Tables the `on_leader_change` callback writes to.
    pub write_deps: Vec<TableId>,
    /// When `Some(_)`, back the function with a [`DisplacedTableWithProvenance`]
    /// (4-column writes carrying a per-edge proof) instead of the plain
    /// [`DisplacedTable`] (3-column writes), and build the leader-change
    /// callback internally so it can compose the onchange relation's proof
    /// column via the provenance variant's `get_proof`. When set, `on_leader_change`
    /// is ignored. Used by native-UF proof mode.
    pub proof_wiring: Option<UfProofWiring>,
}

/// Proof-mode wiring for a UF-backed function. The leader-change callback uses
/// these to compose a proof that `displaced_leader = new_leader` and write it
/// into the onchange relation.
#[derive(Copy, Clone)]
pub struct UfProofWiring {
    /// The onchange relation (now `(S S S S S Proof)`): the callback writes
    /// `(write_lhs write_rhs lhs_leader rhs_leader new_leader proof)`.
    pub onchange: FunctionId,
    /// `Trans` proof constructor: `(Trans Proof Proof) -> Proof`.
    pub trans: FunctionId,
    /// `Sym` proof constructor: `(Sym Proof) -> Proof`.
    pub sym: FunctionId,
}

impl Default for EGraph {
    fn default() -> Self {
        let mut db = Database::new();
        let uf_table = db.add_table_named(
            DisplacedTable::default(),
            "$uf".into(),
            iter::empty(),
            iter::empty(),
        );
        let id_counter = db.add_counter();
        let ts_counter = db.add_counter();
        // Start the timestamp counter at 1.
        db.inc_counter(ts_counter);

        // Register a default panic external function so the typed
        // state wrappers' `panic()` method has an id to call. This
        // also seeds `panic_funcs` so a later `new_panic` with the
        // same message reuses the id.
        let panic_message: SideChannel<String> = Default::default();
        let mut panic_funcs: HashMap<String, ExternalFunctionId> = Default::default();
        let default_panic_msg = "primitive panicked".to_string();
        let default_panic_id = db.add_external_function(Box::new(Panic(
            default_panic_msg.clone(),
            panic_message.clone(),
        )));
        panic_funcs.insert(default_panic_msg, default_panic_id);

        let union_action = UnionAction {
            table: uf_table,
            timestamp: ts_counter,
        };
        let action_registry = Arc::new(std::sync::RwLock::new(ActionRegistry::new(
            union_action,
            default_panic_id,
        )));

        Self {
            db,
            uf_table,
            id_counter,
            timestamp_counter: ts_counter,
            rules: Default::default(),
            funcs: Default::default(),
            panic_message,
            panic_funcs,
            report_level: Default::default(),
            action_registry,
            nativerb_views: Default::default(),
            native_merge_uf: Default::default(),
            native_merge_view_target: Default::default(),
        }
    }
}

impl EGraph {
    fn next_ts(&self) -> Timestamp {
        Timestamp::from_usize(self.db.read_counter(self.timestamp_counter))
    }

    fn inc_ts(&mut self) {
        self.db.inc_counter(self.timestamp_counter);
    }

    /// Get a mutable reference to the underlying table of base values for this
    /// `EGraph`.
    pub fn base_values_mut(&mut self) -> &mut BaseValues {
        self.db.base_values_mut()
    }

    /// Get a mutable reference to the underlying table of containers for this
    /// `EGraph`.
    pub fn container_values_mut(&mut self) -> &mut ContainerValues {
        self.db.container_values_mut()
    }

    /// Get a reference to the underlying table of containers for this `EGraph`.
    pub fn container_values(&self) -> &ContainerValues {
        self.db.container_values()
    }

    /// Intern the given container value into the EGraph.
    pub fn get_container_value<C: ContainerValue>(&mut self, val: C) -> Value {
        self.register_container_ty::<C>();
        self.db
            .with_execution_state(|state| state.clone().container_values().register_val(val, state))
    }

    /// Register the given [`ContainerValue`] type with this EGraph.
    ///
    /// The given container will use the EGraph's union-find to manage rebuilding and the merging
    /// of containers with a common id.
    pub fn register_container_ty<C: ContainerValue>(&mut self) {
        let uf_table = self.uf_table;
        let ts_counter = self.timestamp_counter;
        self.db.container_values_mut().register_type::<C>(
            self.id_counter,
            move |state, old, new| {
                if old != new {
                    let next_ts = Value::from_usize(state.read_counter(ts_counter));
                    state.stage_insert(uf_table, &[old, new, next_ts]);
                    std::cmp::min(old, new)
                } else {
                    old
                }
            },
        );
    }

    /// Get a reference to the underlying table of base values for this `EGraph`.
    pub fn base_values(&self) -> &BaseValues {
        self.db.base_values()
    }

    /// Create a [`QueryEntry`] for a base value.
    pub fn base_value_constant<T>(&self, x: T) -> QueryEntry
    where
        T: BaseValue,
    {
        QueryEntry::Const {
            val: self.base_values().get(x),
            ty: ColumnTy::Base(self.base_values().get_ty::<T>()),
        }
    }

    /// Register a low-level external function. The callback receives a
    /// raw `&mut ExecutionState`.
    ///
    /// # Seminaive-safety trust boundary
    ///
    /// Like [`EGraph::with_execution_state`], this is a raw escape —
    /// the registered function has unrestricted access and is not
    /// tracked by the per-context validity system. Prefer building
    /// primitives via the higher-level `egglog::Primitive` /
    /// `egglog::EGraph::add_primitive` API, which enforces #772's
    /// seminaive-safety contract.
    pub fn register_external_func(
        &mut self,
        func: Box<dyn ExternalFunction + 'static>,
    ) -> ExternalFunctionId {
        self.db.add_external_function(func)
    }

    pub fn free_external_func(&mut self, func: ExternalFunctionId) {
        self.db.free_external_function(func)
    }

    /// Generate a fresh id.
    pub fn fresh_id(&mut self) -> Value {
        Value::from_usize(self.db.inc_counter(self.id_counter))
    }

    /// Look up the canonical value for `val` in the union-find.
    ///
    /// If the value has never been inserted into the union-find, `val` is returned.
    fn get_canon_in_uf(&self, val: Value) -> Value {
        let table = self.db.get_table(self.uf_table);
        let row = table.get_row(&[val]);
        row.map(|row| row.vals[1]).unwrap_or(val)
    }

    /// Get the canonical representation for `val` based on type.
    ///
    /// For [`ColumnTy::Id`], it looks up the union find; otherwise,
    /// it returns the value itself.
    pub fn get_canon_repr(&self, val: Value, ty: ColumnTy) -> Value {
        match ty {
            ColumnTy::Id => self.get_canon_in_uf(val),
            ColumnTy::Base(_) => val,
        }
    }

    /// Load the given values into the database.
    ///
    /// # Panics
    /// This method panics if the values do not match the arity of the function.
    ///
    /// NB: this is not an efficient interface for bulk loading. We should add
    /// one that allows us to pass through a series of RowBuffers before
    /// incrementing the timestamp.
    pub fn add_values(&mut self, values: impl IntoIterator<Item = (FunctionId, Vec<Value>)>) {
        let mut extended_row = Vec::<Value>::new();
        let mut bufs = DenseIdMap::default();
        for (func, row) in values.into_iter() {
            let table_info = &self.funcs[func];
            let schema_math = table_info.schema_math();
            let table_id = table_info.table;
            extended_row.extend_from_slice(&row);
            schema_math.write_table_row(
                &mut extended_row,
                RowVals {
                    timestamp: self.next_ts().to_value(),
                    subsume: schema_math.subsume.then_some(NOT_SUBSUMED),
                    ret_val: None, // already filled in.
                },
            );
            let buf = bufs.get_or_insert(table_id, || self.db.new_buffer(table_id));
            buf.stage_insert(&extended_row);
            extended_row.clear();
        }
        // Flush the buffers.
        mem::drop(bufs);
        self.flush_updates();
    }

    /// A term-oriented means of adding data to the database: hand back a "term
    /// id" for the given function and keys for the function.
    ///
    /// # Panics
    /// This method panics if the values do not match the arity of the function.
    pub fn add_term(&mut self, func: FunctionId, inputs: &[Value]) -> Value {
        let info = &self.funcs[func];
        let schema_math = info.schema_math();
        let mut extended_row = Vec::new();
        extended_row.extend_from_slice(inputs);
        let res = self.fresh_id();
        schema_math.write_table_row(
            &mut extended_row,
            RowVals {
                timestamp: self.next_ts().to_value(),
                ret_val: Some(res),
                subsume: schema_math.subsume.then_some(NOT_SUBSUMED),
            },
        );
        extended_row[schema_math.ret_val_col()] = res;
        let table_id = self.funcs[func].table;
        self.db.new_buffer(table_id).stage_insert(&extended_row);
        self.flush_updates();
        self.get_canon_in_uf(res)
    }

    /// Lookup the id associated with a function `func` and the given arguments
    /// (`key`).
    pub fn lookup_id(&self, func: FunctionId, key: &[Value]) -> Option<Value> {
        let info = &self.funcs[func];
        let schema_math = info.schema_math();
        let table_id = info.table;
        let table = self.db.get_table(table_id);
        let row = table.get_row(key)?;
        Some(row.vals[schema_math.ret_val_col()])
    }

    pub fn approx_table_size(&self, table: FunctionId) -> usize {
        self.db.estimate_size(self.funcs[table].table, None)
    }

    pub fn table_size(&self, table: FunctionId) -> usize {
        self.db.get_table(self.funcs[table].table).len()
    }

    /// Remove every row from the given function's backing table.
    ///
    /// This is the bulk counterpart to staging a `remove` for every key in the
    /// table: the underlying `Database::clear_table` drops the row buffer in
    /// O(1)-in-row-count time and bumps the table's major generation, which
    /// lazily invalidates any cached subsets or indexes a later reader might
    /// consult. Any rows staged for this table by an in-flight
    /// `MutationBuffer` are dropped along with the table contents.
    ///
    /// Callers that have staged inserts/removes for *other* tables that they
    /// want flushed first should call [`EGraph::flush_updates`] before
    /// clearing.
    pub fn clear_table(&mut self, func: FunctionId) {
        let table_id = self.funcs[func].table;
        self.db.clear_table(table_id);
    }

    /// Read the contents of the given function.
    ///
    /// The callback `f` is called with each row and its subsumption status.
    pub fn for_each(&self, table: FunctionId, mut f: impl FnMut(FunctionRow<'_>)) {
        self.for_each_while(table, |row| {
            f(row);
            true
        });
    }

    /// Iterate over the rows of a function table, calling `f` on each row. If `f` returns `false`
    /// the function returns early and stops reading rows from the table.
    pub fn for_each_while(&self, table: FunctionId, mut f: impl FnMut(FunctionRow<'_>) -> bool) {
        let info = &self.funcs[table];
        let table = self.funcs[table].table;
        let schema_math = info.schema_math();
        let imp = self.db.get_table(table);
        let all = imp.all();
        let mut cur = Offset::new(0);
        let mut buf = TaggedRowBuffer::new(imp.spec().arity());
        // This somewhat awkward iteration strategy is forced on us by the `scan_bounded` API. We
        // should look into ways to avoid this cludge where the loop body effectively must be
        // repeated at the end. The obvious and idiomatic ways to do this all require
        // `dyn`-compatibility on `Table` or dynamic dispatch per row.
        macro_rules! drain_buf {
            ($buf:expr) => {
                for (_, row) in $buf.non_stale() {
                    let subsumed =
                        schema_math.subsume && row[schema_math.subsume_col()] == SUBSUMED;
                    if !f(FunctionRow {
                        vals: &row[0..schema_math.func_cols],
                        subsumed,
                    }) {
                        return;
                    }
                }
                $buf.clear();
            };
        }
        while let Some(next) = imp.scan_bounded(all.as_ref(), cur, 32, &mut buf) {
            drain_buf!(buf);
            cur = next;
        }
        drain_buf!(buf);
    }

    /// A basic method for dumping the state of the database to `log::info!`.
    ///
    /// For large tables, this is unlikely to give particularly useful output.
    pub fn dump_debug_info(&self) {
        info!("=== View Tables ===");
        for (id, info) in self.funcs.iter() {
            let table = self.db.get_table(info.table);
            self.scan_table(table, |row| {
                info!(
                    "View Table {name} / {id:?} / {table:?}: {row:?}",
                    name = info.name,
                    table = info.table
                )
            });
        }
    }

    /// A helper for scanning the entries in a table.
    fn scan_table(&self, table: &WrappedTable, mut f: impl FnMut(&[Value])) {
        const BATCH_SIZE: usize = 128;
        let all = table.all();
        let mut cur = Offset::new(0);
        let mut out = TaggedRowBuffer::new(table.spec().arity());
        while let Some(next) = table.scan_bounded(all.as_ref(), cur, BATCH_SIZE, &mut out) {
            out.non_stale().for_each(|(_, row)| f(row));
            out.clear();
            cur = next;
        }
        out.non_stale().for_each(|(_, row)| f(row));
    }

    /// Peek at the [`FunctionId`] that the next [`EGraph::add_table`] call will
    /// return, WITHOUT registering anything.
    ///
    /// `add_table` pushes the new `FunctionInfo` at `self.funcs.next_id()` and
    /// asserts the returned id equals that value (see the `debug_assert_eq!` at
    /// the end of `add_table`), so this is the id the next table will be given.
    ///
    /// Used for "knot-tying": the term-encoder's self-referential union-find
    /// function `@UF_Sf` needs its OWN id while building its `:merge` (a
    /// `MergeFn::TableInsert` into itself for the recursive parent-union). The
    /// caller peeks the id here immediately before `add_table` so the merge can
    /// name the table being declared. Note that the corresponding `TableId`
    /// (`Database::next_table_id`) is consumed in lock-step by the same
    /// `add_table`, so the peeked id stays valid as long as no other table is
    /// added in between.
    pub fn peek_next_function_id(&self) -> FunctionId {
        self.funcs.next_id()
    }

    /// Register a function in this EGraph.
    pub fn add_table(&mut self, config: FunctionConfig) -> FunctionId {
        let FunctionConfig {
            schema,
            default,
            merge,
            name,
            can_subsume,
        } = config;
        assert!(
            !schema.is_empty(),
            "must have at least one column in schema"
        );
        let to_rebuild: Vec<ColumnId> = schema
            .iter()
            .enumerate()
            .filter(|(_, ty)| matches!(ty, ColumnTy::Id))
            .map(|(i, _)| ColumnId::from_usize(i))
            .collect();
        // The number of value (return) columns is determined by the merge: a `Columns` merge has
        // one entry per value column; every other (scalar) merge applies to a single value column.
        let n_vals = match &merge {
            MergeFn::Columns(cols) => cols.len(),
            _ => 1,
        };
        assert!(
            schema.len() >= n_vals,
            "function {name} has fewer columns ({}) than value columns ({n_vals})",
            schema.len()
        );
        let n_keys = schema.len() - n_vals;
        let schema_math = SchemaMath {
            subsume: can_subsume,
            n_keys,
            func_cols: schema.len(),
        };
        let n_args = schema_math.num_keys();
        let n_cols = schema_math.table_columns();
        let next_func_id = self.funcs.next_id();
        // `--native-merge`: remember which per-sort `@UF_Sf` this view's
        // FD-conflict congruence union targets (named by the `UnionIntoUf` merge),
        // so a later `register_native_merge_view` can assert the registered UF
        // matches the one already baked into the merge.
        let native_merge_target = match &merge {
            MergeFn::UnionIntoUf(uf_func) => Some(*uf_func),
            // Proof-mode tuple-output view: the UF target is named by the `col0`
            // `UnionIntoUfWithProof` inside the top-level `Columns`.
            MergeFn::Columns(cols) => cols.iter().find_map(|c| match c {
                MergeFn::UnionIntoUf(uf_func) => Some(*uf_func),
                MergeFn::UnionIntoUfWithProof { uf, .. } => Some(*uf),
                _ => None,
            }),
            _ => None,
        };
        // Knot-tying for a SELF-REFERENTIAL merge (the term-encoder's
        // `@UF_Sf : (S) -> S`, whose `:merge` does a `MergeFn::TableInsert` into
        // ITS OWN table for the recursive parent-union): the merge resolution
        // (`merge_fn_fill_deps` / `merge_fn_to_callback` → `TableAction::new`)
        // reads `self.funcs[self_id].table` etc., so the new `FunctionInfo` must
        // already be in `self.funcs` BEFORE the merge is built. We therefore
        // reserve the table id (deterministic — the next `add_table_named` will
        // assign exactly this id) and push the `FunctionInfo` up front, then
        // build the merge and create the backing table. Tables whose merge does
        // not self-reference resolve identically either way; this ordering is a
        // strict superset.
        let name: Arc<str> = name.into();
        let table_id = self.db.next_table_id();
        let res = self.funcs.push(FunctionInfo {
            table: table_id,
            schema: schema.clone(),
            n_keys,
            incremental_rebuild_rules: Default::default(),
            nonincremental_rebuild_rule: RuleId::new(!0),
            default_val: default,
            can_subsume,
            name: name.clone(),
            kind: FunctionKind::Table,
            uf_rebuild_rule: None,
        });
        debug_assert_eq!(res, next_func_id);

        let mut read_deps = IndexSet::<TableId>::new();
        let mut write_deps = IndexSet::<TableId>::new();
        merge_fn_fill_deps(&merge, self, &mut read_deps, &mut write_deps);
        let merge_fn = merge_fn_to_callback(&merge, schema_math, &name, self);
        let table = SortedWritesTable::new(
            n_args,
            n_cols,
            Some(ColumnId::from_usize(schema.len())),
            to_rebuild,
            merge_fn,
        );
        let assigned_table_id = self.db.add_table_named(
            table,
            name.clone(),
            read_deps.iter().copied(),
            write_deps.iter().copied(),
        );
        debug_assert_eq!(assigned_table_id, table_id);
        if let Some(uf_func) = native_merge_target {
            self.native_merge_view_target.insert(res, uf_func);
        }
        let incremental_rebuild_rules = self.incremental_rebuild_rules(res, &schema);
        let nonincremental_rebuild_rule = self.nonincremental_rebuild(res, &schema);
        let info = &mut self.funcs[res];
        info.incremental_rebuild_rules = incremental_rebuild_rules;
        info.nonincremental_rebuild_rule = nonincremental_rebuild_rule;
        let action = TableAction::new(self, res);
        let table_name = self.funcs[res].name.to_string();
        self.action_registry
            .write()
            .unwrap()
            .register_table(table_name, action);
        res
    }

    /// The backing [`TableId`] for a function. Used to register a UF-backed
    /// function's `on_leader_change` callback as a write-dependency.
    pub fn table_id(&self, table: FunctionId) -> TableId {
        self.funcs[table].table
    }

    /// `--nativerb`: register a `@<F>View` view function (`view_func`) to be
    /// re-canonicalized by the ENGINE's native table rebuild against the
    /// per-sort `@UF_Sf` UF-backed function (`uf_func`), instead of the
    /// encoding-level `@rebuild_rule*` rules. The frontend calls this for every
    /// view function once all functions are registered; `rebuild()` then drives
    /// `apply_rebuild(uf, &views, ts)` per registered UF to a joint fixpoint.
    /// `uf_func` must be a UF-backed function (`FunctionKind::Uf`).
    pub fn register_nativerb_view(&mut self, uf_func: FunctionId, view_func: FunctionId) {
        assert!(
            self.funcs[uf_func].is_uf(),
            "register_nativerb_view: {uf_func:?} is not a UF-backed function"
        );
        self.nativerb_views
            .entry(uf_func)
            .or_default()
            .push(view_func);
    }

    /// `--native-merge` (term-encoding native-UF, non-proof, native bridge):
    /// associate an FD-keyed constructor `@<F>View` view function (`view_func`)
    /// with the per-sort `@UF_Sf` UF-backed function (`uf_func`) that owns its
    /// eclass (OUTPUT) column. This is the uniform contract the dataflow/SQL
    /// backends use to route the FD-conflict congruence union into the right
    /// union-find. On the native bridge the routing is already baked into the
    /// view's [`MergeFn::UnionIntoUf`] merge (which stages the union directly into
    /// this `@UF_Sf`), so this method records the association and ASSERTS the
    /// view's merge actually targets the registered UF — a defensive consistency
    /// check that the frontend's declare-time UF resolution and its
    /// schedule-time registration agree. `uf_func` must be a UF-backed function.
    pub fn register_native_merge_view(&mut self, uf_func: FunctionId, view_func: FunctionId) {
        assert!(
            self.funcs[uf_func].is_uf(),
            "register_native_merge_view: {uf_func:?} is not a UF-backed function"
        );
        if let Some(target) = self.native_merge_view_target.get(&view_func) {
            assert_eq!(
                *target, uf_func,
                "register_native_merge_view: view {view_func:?} merge targets UF {target:?} \
                 but is being registered against UF {uf_func:?}"
            );
        }
        self.native_merge_uf.insert(view_func, uf_func);
    }

    /// Register a UF-backed function in this EGraph.
    ///
    /// UF Functions record the series of "leader changes" in the e-graph, where
    /// the leader is always 'up to date' with the latest value. It has two
    /// columns: one key (the "displaced" value), mapping to its current leader.
    /// Queries of such a function, coupled with semi-naive evaluation, allow for
    /// rules to "listen" on 'leader changes' in the underlying union-find. This
    /// makes it more efficient, if less convenient, than a standard
    /// quadratic-sized equivalence relation.
    ///
    /// The returned external function canonicalizes its single argument against
    /// the underlying union-find (find-or-self).
    ///
    /// Values in this union find are "coherent" with those of the underlying
    /// e-graph union-find: rebuilding the e-graph also forwards global unions
    /// into these union-finds. What makes these useful is that they can store
    /// _more_ equalities that are themselves "visible" to egglog rules.
    pub fn add_uf_function(
        &mut self,
        config: UfFunctionConfig,
    ) -> Result<(FunctionId, ExternalFunctionId)> {
        let UfFunctionConfig {
            name,
            on_leader_change,
            read_deps,
            write_deps,
            proof_wiring,
        } = config;
        let name: Arc<str> = name.into();
        // In proof mode the function is backed by a `DisplacedTableWithProvenance`
        // (4-column writes carrying a per-edge proof, exposing `get_proof`); in
        // term mode the plain `DisplacedTable` (3-column writes) is used.
        let table_id = if let Some(wiring) = proof_wiring {
            // TableActions to intern the composed proof and write the onchange row.
            let onchange_action = TableAction::new(self, wiring.onchange);
            let trans_action = TableAction::new(self, wiring.trans);
            let sym_action = TableAction::new(self, wiring.sym);
            let onchange_tid = self.table_id(wiring.onchange);
            let trans_tid = self.table_id(wiring.trans);
            let sym_tid = self.table_id(wiring.sym);
            // The leader-change callback both reads (hash-cons dedup via
            // `predict_val`'s `get_row`) and writes (`stage_insert`) the onchange
            // and Trans/Sym tables. They must therefore be both read- and
            // write-dependencies: the write dep pre-allocates a mutation buffer
            // (so writes don't touch the moved-out table info), and the read dep
            // forces them into an earlier merge stratum (so they stay in the
            // database — readable — while this UF table is merging). Without the
            // read dep, the Trans/Sym tables (which are also written by ordinary
            // proof rules) can be merged in the same stratum as the UF table and
            // get moved out from under the callback's `get_row`.
            let mut read_deps = read_deps;
            read_deps.push(onchange_tid);
            read_deps.push(trans_tid);
            read_deps.push(sym_tid);
            let mut write_deps = write_deps;
            write_deps.push(onchange_tid);
            write_deps.push(trans_tid);
            write_deps.push(sym_tid);

            // The provenance table reconstructs the proof path `displaced ->
            // new_leader` itself (it can't be read back from the database during
            // the table's own merge) and hands it to the callback, which folds
            // it into a single `Proof` term and records the onchange row.
            let table = DisplacedTableWithProvenance::with_leader_change_callback(
                move |state: &mut ExecutionState, change: LeaderChange, steps: &[ProofStep]| {
                    let new_leader = change.new_leader();
                    let proof = compose_uf_proof(state, &trans_action, &sym_action, steps);
                    onchange_action.lookup_or_insert(
                        state,
                        &[
                            change.write_lhs,
                            change.write_rhs,
                            change.lhs_leader,
                            change.rhs_leader,
                            new_leader,
                            proof,
                        ],
                    );
                },
            );
            self.db.add_table_named(
                table,
                name.clone(),
                read_deps.iter().copied(),
                write_deps.iter().copied(),
            )
        } else {
            let mut table = DisplacedTable::default();
            if let Some(callback) = on_leader_change {
                table.set_leader_change_callback(move |state, change| callback(state, change));
            }
            self.db.add_table_named(
                table,
                name.clone(),
                read_deps.iter().copied(),
                write_deps.iter().copied(),
            )
        };
        let canon =
            self.register_external_func(Box::new(make_external_func(move |state, vals| {
                let [val] = vals else {
                    panic!("uf canonicalizer expected 1 value, got {vals:?}")
                };
                let table = state.get_table(table_id);
                let canon = table
                    .get_row_column(&[*val], ColumnId::new(1))
                    .unwrap_or(*val);
                Some(canon)
            })));
        // In proof mode the UF function is `(S S) Proof`: writes carry the
        // per-edge proof as a third column (`set (@UF_Sf a b) proof` →
        // `[a, b, proof, ts]`), which the provenance table consumes. In term
        // mode it is the usual `(S) S` (writes `[a, b, ts]`).
        let schema = if proof_wiring.is_some() {
            vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id]
        } else {
            vec![ColumnTy::Id, ColumnTy::Id]
        };
        // UF-backed functions are single-value (keyed on the displaced id; the leader, plus an
        // optional proof column, are the value columns). Mirror the single-output convention.
        let n_keys = schema.len() - 1;
        let res = self.funcs.push(FunctionInfo {
            table: table_id,
            schema,
            n_keys,
            incremental_rebuild_rules: Vec::new(),
            nonincremental_rebuild_rule: RuleId::new(!0),
            default_val: DefaultVal::Fail,
            can_subsume: false,
            name,
            kind: FunctionKind::Uf,
            uf_rebuild_rule: None,
        });
        let uf_rebuild_rule = self.uf_rebuild_rule(res);
        self.funcs[res].uf_rebuild_rule = Some(uf_rebuild_rule);
        Ok((res, canon))
    }

    /// Build a rule that forwards global uf_table unions into a UF-backed table
    /// so it stays consistent with the main union-find.
    fn uf_rebuild_rule(&mut self, table: FunctionId) -> RuleId {
        let uf_table = self.uf_table;
        // Proof-mode UF functions are `(S S) Proof` (3 columns): the set must
        // supply a proof column. Forwarded global e-graph unions don't carry an
        // egglog-level proof here (the term encoder routes unions through
        // `(set @UF_Sf ...)` directly, so this rule is normally inert), so use a
        // placeholder proof value.
        let proof_col = self.funcs[table].schema.len() == 3;
        let mut rb = self.new_rule(&format!("uf rebuild {table:?}"), true);
        let lhs: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let rhs: QueryEntry = rb.new_var(ColumnTy::Id).into();
        rb.add_atom_with_timestamp_and_func(uf_table, None, None, &[lhs.clone(), rhs.clone()]);
        if proof_col {
            let placeholder = QueryEntry::Const {
                val: Value::new(0),
                ty: ColumnTy::Id,
            };
            rb.set(table, &[lhs, rhs, placeholder]);
        } else {
            rb.set(table, &[lhs, rhs]);
        }
        rb.build()
    }

    fn uf_rebuild_rules(&self) -> Vec<RuleId> {
        self.funcs
            .iter()
            .filter_map(|(_, info)| info.uf_rebuild_rule)
            .collect()
    }

    /// A handle to the live [`ActionRegistry`] for this EGraph.
    /// The handle is shared (`Arc<RwLock<_>>`); cloning the outer
    /// `Arc` does not duplicate the underlying registry. Used by the
    /// egglog crate's primitive machinery to thread the registry into
    /// state wrappers at invoke time.
    pub fn action_registry(&self) -> &Arc<std::sync::RwLock<ActionRegistry>> {
        &self.action_registry
    }

    /// Run the given rules, returning whether the database changed.
    ///
    /// If the given rules are malformed, this method can return an error.
    pub fn run_rules(&mut self, rules: &[RuleId]) -> Result<IterationReport> {
        self.run_rules_inner(rules)
    }

    /// Total rows across the union-find tables that drive a rebuild: the global
    /// `$uf`, plus (under `--nativerb`) every registered per-sort `@UF_Sf`. In
    /// term-encoding native-UF mode unions go through `@UF_Sf`, not `$uf`, so
    /// the rebuild trigger must watch those tables too.
    fn rebuild_uf_size(&self) -> usize {
        let mut total = self.db.get_table(self.uf_table).len();
        for uf_func in self.nativerb_views.keys() {
            total += self.db.get_table(self.funcs[*uf_func].table).len();
        }
        total
    }

    fn run_rules_inner(&mut self, rules: &[RuleId]) -> Result<IterationReport> {
        let ts = self.next_ts();

        let uf_size_before = self.rebuild_uf_size();
        let rule_set_report =
            run_rules_impl(&mut self.db, &mut self.rules, rules, ts, self.report_level)?;
        if let Some(message) = self.panic_message.lock().unwrap().take() {
            return Err(PanicError(message).into());
        }

        let mut iteration_report = IterationReport {
            rule_set_report,
            rebuild_time: Duration::ZERO,
        };
        let uf_size_after = self.rebuild_uf_size();
        if uf_size_before == uf_size_after {
            // No new unions: skip the full rebuild but still advance the
            // timestamp so that seminaive evaluation sees a fresh epoch.
            // Rebuilding is only necessary when new unions have been made because ids may need to be updated.
            // Adding terms doesn't necessarily touch the union-find, only doing a union between existing ids does.
            self.inc_ts();
            return Ok(iteration_report);
        }

        let rebuild_timer = Instant::now();
        self.rebuild()?;
        iteration_report.rebuild_time = rebuild_timer.elapsed();

        if let Some(message) = self.panic_message.lock().unwrap().take() {
            return Err(PanicError(message).into());
        }

        Ok(iteration_report)
    }

    fn rebuild(&mut self) -> Result<()> {
        let uf_rules = self.uf_rebuild_rules();
        let do_parallel = rayon::current_num_threads() > 1;
        if self.db.get_table(self.uf_table).rebuilder(&[]).is_some() {
            // The UF implementation supports "native"  rebuilding.
            let mut tables = Vec::with_capacity(self.funcs.next_id().index());
            for (_, func) in self.funcs.iter() {
                if func.is_uf() {
                    continue;
                }
                tables.push(func.table);
            }
            loop {
                // Order matters here: we need to rebuild containers first and then rebuild the
                // tables. Why?
                //
                // Say we have a sort that can map to and from a vector containing only itself:
                // (sort X)
                // (function to-vec (X) (Vec X) :no-merge)
                // (constructor from-vec (Vec X) X)
                // (constructor Num (i64) X)
                // (constructor Add (X X) X)
                //
                // Along with rules:
                // (rule ((= x (Num i))) ((set (to-vec x) (vec-of x))))
                // (rule ((= x (Add i j))) ((set (to-vec x) (vec-of x))))
                // (rule ((= x (from-vec v))) ((set (to-vec x) v))
                // (rewrite (Add (Num i) (Num j)) (Num (+ i j)))
                //
                // These rules, while redundant, should be safe. However, if we rebuild tables
                // before containers some schedules can cause us to violate the `:no-merge`
                // directive, which asserts that all values written for a key are equal.
                //
                // Suppose we start off with x1=(Num 1), x2=(Num 3), and x3=(Add (Num 1) (Num 2)) as
                // expressions, with `to-vec` and `from-vec` entries for all three expressions.
                // We'll call (to-vec xi) vi for all i.
                //
                // Now suppose we run the `rewrite` above: now, x3 = x2. But v3 will only equal v2
                // _after_ we rebuild the `Vec` container. That means that if we rebuild `to-vec`
                // we will collapse the the rows for x3 and x2, but then fail to merge v3 and v2
                // because they are not (yet) equal.
                //
                // Rebuilding containers first will find that v3 and v2 are equal, and the rest of
                // the rules can proceed.
                let container_rebuild = self.db.rebuild_containers(self.uf_table);
                let next_ts = self.next_ts().to_value();
                let table_rebuild = self.db.apply_rebuild(self.uf_table, &tables, next_ts);
                // Container rebuild can make a parent row newly matchable without
                // changing the row's stored id. Re-timestamp those parents so
                // seminaive sees the newly enabled match on the next pass.
                let dirty_ids: Vec<Value> = container_rebuild.dirty_ids().iter().copied().collect();
                let refreshed_rows = self
                    .db
                    .refresh_rows_for_values(&tables, &dirty_ids, next_ts);
                // Forward any global e-graph unions into the UF-backed tables so
                // they stay coherent with the main union-find.
                let uf_rebuild = if uf_rules.is_empty() {
                    false
                } else {
                    let uf_ts = self.next_ts();
                    run_rules_impl(
                        &mut self.db,
                        &mut self.rules,
                        &uf_rules,
                        uf_ts,
                        ReportLevel::TimeOnly,
                    )?
                    .changed
                };
                // `--nativerb`: re-canonicalize the registered `@<F>View` view
                // tables against each per-sort `@UF_Sf` using the ENGINE's
                // native table rebuild, replacing the encoding-level
                // `@rebuild_rule*` rules. This runs inside the same saturating
                // rebuild loop as the `$uf` rebuild so the two reach a JOINT
                // fixpoint: congruence (still emitted as `@congruence_rule*`)
                // issues unions on `@UF_Sf` from outside (in the `@rebuilding`
                // ruleset), and the surrounding `(saturate @rebuilding)` re-runs
                // congruence whenever this engine rebuild collapses a view row.
                // Container rebuild first (same ordering rationale as above), in
                // case a container holds eq-sort ids of this sort.
                let nativerb_changed = self.nativerb_rebuild_pass()?;
                self.inc_ts();
                if !table_rebuild
                    && !refreshed_rows
                    && !container_rebuild.changed()
                    && !uf_rebuild
                    && !nativerb_changed
                {
                    break;
                }
            }
            return Ok(());
        }
        if do_parallel {
            return self.rebuild_parallel();
        }
        let start = Instant::now();

        // The database changed. Rebuild. New entries should land after the given rules.
        let mut changed = true;
        while changed {
            changed = false;
            // We need to iterate rebuilding to a fixed point. Future scans
            // should look only at the latest updates.
            self.inc_ts();
            let ts = self.next_ts();
            for (_, info) in self.funcs.iter_mut() {
                if info.is_uf() {
                    continue;
                }
                let last_rebuilt_at = self.rules[info.nonincremental_rebuild_rule].last_run_at;
                let table_size = self.db.estimate_size(info.table, None);
                let uf_size = self.db.estimate_size(
                    self.uf_table,
                    Some(Constraint::GeConst {
                        col: ColumnId::new(2),
                        val: last_rebuilt_at.to_value(),
                    }),
                );
                if incremental_rebuild(uf_size, table_size, false) {
                    marker_incremental_rebuild(|| -> Result<()> {
                        // Run each of the incremental rules serially.
                        //
                        // This is to avoid recanonicalizing the same row multiple
                        // times.
                        for rule in &info.incremental_rebuild_rules {
                            changed |= run_rules_impl(
                                &mut self.db,
                                &mut self.rules,
                                &[*rule],
                                ts,
                                ReportLevel::TimeOnly,
                            )?
                            .changed;
                        }
                        // Reset the rule we did not run. These two should be equivalent.
                        self.rules[info.nonincremental_rebuild_rule].last_run_at = ts;
                        Ok(())
                    })?;
                } else {
                    marker_nonincremental_rebuild(|| -> Result<()> {
                        changed |= run_rules_impl(
                            &mut self.db,
                            &mut self.rules,
                            &[info.nonincremental_rebuild_rule],
                            ts,
                            ReportLevel::TimeOnly,
                        )?
                        .changed;
                        for rule in &info.incremental_rebuild_rules {
                            self.rules[*rule].last_run_at = ts;
                        }
                        Ok(())
                    })?;
                }
            }
            if !uf_rules.is_empty() {
                changed |= run_rules_impl(
                    &mut self.db,
                    &mut self.rules,
                    &uf_rules,
                    ts,
                    ReportLevel::TimeOnly,
                )?
                .changed;
            }
        }
        log::info!("rebuild took {:?}", start.elapsed());
        Ok(())
    }

    /// `--nativerb`: re-canonicalize every registered `@<F>View` view table
    /// against its per-sort `@UF_Sf` using the engine's native table rebuild.
    /// Returns whether anything changed (a view row was re-canonicalized or
    /// collapsed by `:merge`), so the caller's rebuild loop can iterate to a
    /// fixpoint. A no-op (returns `false`) when no UF is registered.
    fn nativerb_rebuild_pass(&mut self) -> Result<bool> {
        if self.nativerb_views.is_empty() {
            return Ok(false);
        }
        // Snapshot the registry so we can take `&mut self.db` in the loop.
        let entries: Vec<(FunctionId, Vec<TableId>)> = self
            .nativerb_views
            .iter()
            .map(|(uf, views)| {
                (
                    *uf,
                    views
                        .iter()
                        .map(|v| self.funcs[*v].table)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let mut changed = false;
        for (uf_func, view_tables) in &entries {
            let uf_table = self.funcs[*uf_func].table;
            // Container rebuild first (same ordering rationale as the `$uf`
            // path): a container value holding eq-sort ids of this sort must be
            // re-canonicalized before the view rows that key on it.
            let container_rebuild = self.db.rebuild_containers(uf_table);
            let next_ts = self.next_ts().to_value();
            let table_rebuild = self.db.apply_rebuild(uf_table, view_tables, next_ts);
            let dirty_ids: Vec<Value> = container_rebuild.dirty_ids().iter().copied().collect();
            let refreshed_rows = self
                .db
                .refresh_rows_for_values(view_tables, &dirty_ids, next_ts);
            changed |= table_rebuild || refreshed_rows || container_rebuild.changed();
        }
        Ok(changed)
    }

    /// A variant of `rebuild` that attempts to combine rebuild rules into
    /// larger rulesets to increase parallelism. This kind of preprocessing can
    /// slow processing down in a single-threaded setting, so it is only used
    /// when the number of active threads is greater than 1.
    fn rebuild_parallel(&mut self) -> Result<()> {
        let start = Instant::now();
        let uf_rules = self.uf_rebuild_rules();
        #[derive(Default)]
        struct RebuildState {
            nonincremental: Vec<FunctionId>,
            incremental: DenseIdMap<usize, SmallVec<[FunctionId; 2]>>,
        }

        impl RebuildState {
            fn clear(&mut self) {
                self.nonincremental.clear();
                self.incremental.iter_mut().for_each(|(_, v)| v.clear());
            }
        }

        let mut changed = true;
        let mut state = RebuildState::default();
        let mut scratch = Vec::new();
        while changed {
            changed = false;
            state.clear();
            self.inc_ts();
            // First, figure out which functions will be rebuilt nonincrementally,
            // vs. incrementally. Group them together.
            for (func, info) in self.funcs.iter_mut() {
                if info.is_uf() {
                    continue;
                }
                let last_rebuilt_at = self.rules[info.nonincremental_rebuild_rule].last_run_at;
                let table_size = self.db.estimate_size(info.table, None);
                let uf_size = self.db.estimate_size(
                    self.uf_table,
                    Some(Constraint::GeConst {
                        col: ColumnId::new(2),
                        val: last_rebuilt_at.to_value(),
                    }),
                );
                if incremental_rebuild(uf_size, table_size, true) {
                    for (i, _) in info.incremental_rebuild_rules.iter().enumerate() {
                        state.incremental.get_or_default(i).push(func);
                    }
                } else {
                    state.nonincremental.push(func);
                }
            }
            let ts = self.next_ts();
            for func in state.nonincremental.iter().copied() {
                scratch.push(self.funcs[func].nonincremental_rebuild_rule);
                for rule in &self.funcs[func].incremental_rebuild_rules {
                    self.rules[*rule].last_run_at = ts;
                }
            }
            changed |= run_rules_impl(
                &mut self.db,
                &mut self.rules,
                &scratch,
                ts,
                ReportLevel::TimeOnly,
            )?
            .changed;
            scratch.clear();
            let ts = self.next_ts();
            for (i, funcs) in state.incremental.iter() {
                for func in funcs.iter().copied() {
                    let info = &mut self.funcs[func];
                    scratch.push(info.incremental_rebuild_rules[i]);
                    self.rules[info.nonincremental_rebuild_rule].last_run_at = ts;
                }
                changed |= run_rules_impl(
                    &mut self.db,
                    &mut self.rules,
                    &scratch,
                    ts,
                    ReportLevel::TimeOnly,
                )?
                .changed;
                scratch.clear();
            }
            if !uf_rules.is_empty() {
                changed |= run_rules_impl(
                    &mut self.db,
                    &mut self.rules,
                    &uf_rules,
                    ts,
                    ReportLevel::TimeOnly,
                )?
                .changed;
            }
        }
        log::info!("rebuild took {:?}", start.elapsed());
        Ok(())
    }

    fn incremental_rebuild_rules(&mut self, table: FunctionId, schema: &[ColumnTy]) -> Vec<RuleId> {
        schema
            .iter()
            .enumerate()
            .filter_map(|(i, ty)| match ty {
                ColumnTy::Id => {
                    Some(self.incremental_rebuild_rule(table, schema, ColumnId::from_usize(i)))
                }
                ColumnTy::Base(_) => None,
            })
            .collect()
    }

    fn incremental_rebuild_rule(
        &mut self,
        table: FunctionId,
        schema: &[ColumnTy],
        col: ColumnId,
    ) -> RuleId {
        let subsume = self.funcs[table].can_subsume;
        let table_id = self.funcs[table].table;
        let uf_table = self.uf_table;
        // Two atoms, one binding a whole tuple, one binding a displaced column
        let mut rb = self.new_rule(&format!("incremental rebuild {table:?}, {col:?}"), true);
        rb.set_plan_strategy(PlanStrategy::MinCover);
        let mut vars = Vec::<QueryEntry>::with_capacity(schema.len());
        for ty in schema {
            vars.push(rb.new_var(*ty).into());
        }
        let canon_val: QueryEntry = rb.new_var(ColumnTy::Id).into();
        let subsume_var = subsume.then(|| rb.new_var(ColumnTy::Id));
        rb.add_atom_with_timestamp_and_func(
            table_id,
            Some(table),
            subsume_var.clone().map(QueryEntry::from),
            &vars,
        );
        rb.add_atom_with_timestamp_and_func(
            uf_table,
            None,
            None,
            &[vars[col.index()].clone(), canon_val.clone()],
        );
        rb.set_focus(1); // Set the uf atom as the sole focus.

        // Now canonicalize the entire row.
        let mut canon = Vec::<QueryEntry>::with_capacity(schema.len());
        for (i, (var, ty)) in vars.iter().zip(schema.iter()).enumerate() {
            canon.push(if i == col.index() {
                canon_val.clone()
            } else if let ColumnTy::Id = ty {
                rb.lookup_uf(var.clone()).unwrap().into()
            } else {
                var.clone()
            })
        }

        // Remove the old row and insert the new one.
        rb.rebuild_row(table, &vars, &canon, subsume_var);
        rb.build()
    }

    fn nonincremental_rebuild(&mut self, table: FunctionId, schema: &[ColumnTy]) -> RuleId {
        let can_subsume = self.funcs[table].can_subsume;
        let table_id = self.funcs[table].table;
        let mut rb = self.new_rule(&format!("nonincremental rebuild {table:?}"), false);
        rb.set_plan_strategy(PlanStrategy::MinCover);
        let mut vars = Vec::<QueryEntry>::with_capacity(schema.len());
        for ty in schema {
            vars.push(rb.new_var(*ty).into());
        }
        let subsume_var = can_subsume.then(|| rb.new_var(ColumnTy::Id));
        rb.add_atom_with_timestamp_and_func(
            table_id,
            Some(table),
            subsume_var.clone().map(QueryEntry::from),
            &vars,
        );
        let mut lhs = SmallVec::<[QueryEntry; 4]>::new();
        let mut rhs = SmallVec::<[QueryEntry; 4]>::new();
        let mut canon = Vec::<QueryEntry>::with_capacity(schema.len());
        for (var, ty) in vars.iter().zip(schema.iter()) {
            canon.push(if let ColumnTy::Id = ty {
                lhs.push(var.clone());
                let canon_var = QueryEntry::from(rb.lookup_uf(var.clone()).unwrap());
                rhs.push(canon_var.clone());
                canon_var
            } else {
                var.clone()
            })
        }
        rb.check_for_update(&lhs, &rhs).unwrap();
        rb.rebuild_row(table, &vars, &canon, subsume_var);
        rb.build()
    }

    /// Gives the user a handle to the underlying ExecutionState. Useful for staging updates
    /// to the database.
    ///
    /// The staged updates are not immediately reflected in the EGraph, so you may want to
    /// manually flush the updates using [`EGraph::flush_updates`].
    ///
    /// # Seminaive-safety trust boundary
    ///
    /// This method hands out a raw `&mut ExecutionState`, which bypasses
    /// the typed state wrappers (`PureState`, `WriteState`, `ReadState`,
    /// `FullState`) that the egglog crate uses to enforce #772's
    /// seminaive-safety model. Treat it as top-level / global-action
    /// context: appropriate for one-shot database manipulation from
    /// outside any rule, not for use inside primitive implementations.
    pub fn with_execution_state<R>(&self, f: impl FnOnce(&mut ExecutionState<'_>) -> R) -> R {
        self.db.with_execution_state(f)
    }

    /// Flush the pending update buffers to the EGraph.
    /// Returns `true` if the database is updated.
    pub fn flush_updates(&mut self) -> bool {
        let uf_size_before = self.rebuild_uf_size();
        let updated = self.db.merge_all();
        self.inc_ts();
        let uf_size_after = self.rebuild_uf_size();
        if uf_size_before != uf_size_after {
            // Rebuilding is only necessary when new unions have been made because ids may need to be updated.
            // Adding terms doesn't necessarily touch the union-find, only doing a union between existing ids does.
            self.rebuild().unwrap();
        }
        updated
    }

    pub fn set_report_level(&mut self, level: ReportLevel) {
        self.report_level = level;
    }
}

#[derive(Clone)]
struct RuleInfo {
    last_run_at: Timestamp,
    query: rule::Query,
    cached_plan: Option<CachedPlanInfo>,
    desc: Arc<str>,
}

#[derive(Clone)]
struct CachedPlanInfo {
    plan: Arc<core_relations::CachedPlan>,
    /// A mapping from index into a [`rule::Query`]'s atoms to the atoms in the underlying cached
    /// plan.
    atom_mapping: Vec<core_relations::AtomId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FunctionKind {
    Table,
    Uf,
}

#[derive(Clone)]
struct FunctionInfo {
    table: TableId,
    schema: Vec<ColumnTy>,
    /// The number of key (input) columns. The remaining columns of `schema` are value/return
    /// columns (one for most functions, more for tuple-output functions).
    n_keys: usize,
    incremental_rebuild_rules: Vec<RuleId>,
    nonincremental_rebuild_rule: RuleId,
    default_val: DefaultVal,
    can_subsume: bool,
    name: Arc<str>,
    kind: FunctionKind,
    uf_rebuild_rule: Option<RuleId>,
}

impl FunctionInfo {
    fn ret_ty(&self) -> ColumnTy {
        self.schema.last().copied().unwrap()
    }

    fn is_uf(&self) -> bool {
        matches!(self.kind, FunctionKind::Uf)
    }

    fn schema_math(&self) -> SchemaMath {
        SchemaMath {
            subsume: self.can_subsume,
            n_keys: self.n_keys,
            func_cols: self.schema.len(),
        }
    }
}

// `DefaultVal` and `MergeFn` are now defined in `egglog-backend-trait`; the
// re-export above brings them back into this module's path. The bridge-only
// helpers (`fill_deps`, `to_callback`, `resolve`) live as free functions
// here so that they can keep using `EGraph` internals.
fn merge_fn_fill_deps(
    merge: &MergeFn,
    egraph: &EGraph,
    read_deps: &mut IndexSet<TableId>,
    write_deps: &mut IndexSet<TableId>,
) {
    use MergeFn::*;
    match merge {
        Primitive(_, args) => {
            args.iter()
                .for_each(|arg| merge_fn_fill_deps(arg, egraph, read_deps, write_deps));
            write_deps.insert(egraph.uf_table);
        }
        Function(func, args) => {
            assert!(
                !egraph.funcs[*func].is_uf(),
                "{}",
                FunctionConfigError::MergeUsesUf(egraph.funcs[*func].name.to_string())
            );
            read_deps.insert(egraph.funcs[*func].table);
            write_deps.insert(egraph.funcs[*func].table);
            args.iter()
                .for_each(|arg| merge_fn_fill_deps(arg, egraph, read_deps, write_deps));
        }
        UnionId => {
            write_deps.insert(egraph.uf_table);
        }
        UnionIntoUf(uf_func) => {
            // `--native-merge`: the FD-conflict congruence union is staged into
            // the named per-sort `@UF_Sf` (a UF-backed function), so that table —
            // not the global `$uf` — is the write dependency.
            write_deps.insert(egraph.funcs[*uf_func].table);
        }
        UnionIntoParentTable { parent_table, .. } => {
            // `--native-merge` relational path: the FD-conflict congruence edge is
            // written into the per-sort relational `@UF_S` parent table, so that
            // table is the write dependency.
            write_deps.insert(egraph.funcs[*parent_table].table);
        }
        UnionIntoUfWithProof { uf, trans, sym, .. } => {
            // `--native-merge` PROOF mode: the proof-carrying congruence edge is
            // staged into the proof-mode `@UF_Sf`, and composing its edge proof
            // hash-conses `Trans` / `Sym` rows. All three are write deps; the
            // Trans/Sym tables are also read deps (the `lookup_or_insert`
            // hash-cons dedup reads them — same dependency shape as the
            // leader-change callback in `add_uf_function`).
            write_deps.insert(egraph.funcs[*uf].table);
            write_deps.insert(egraph.funcs[*trans].table);
            write_deps.insert(egraph.funcs[*sym].table);
            read_deps.insert(egraph.funcs[*trans].table);
            read_deps.insert(egraph.funcs[*sym].table);
        }
        UnionIntoParentTableWithProof { .. } => {
            // The 2-table proof-congruence merge is FlowLog/Feldera-only (it reads a
            // proof side-table and writes the relational `@UF_S`); the native bridge
            // never builds or resolves it (it uses the A2 tuple proof view). So it
            // has no dependency footprint here.
            panic!(
                "MergeFn::UnionIntoParentTableWithProof is a single-output-backend \
                 (FlowLog/Feldera) merge and is not resolved on the native bridge"
            )
        }
        EclassMinProof { .. } => {}
        Columns(cols) => {
            cols.iter()
                .for_each(|col| merge_fn_fill_deps(col, egraph, read_deps, write_deps));
        }
        TableInsert(func, args) => {
            // The side write makes the target table a write-dependency, so its
            // buffer is pre-allocated during batched merges (the whole reason
            // this exists rather than a by-name primitive insert). Ported from
            // PR #933.
            write_deps.insert(egraph.funcs[*func].table);
            args.iter()
                .for_each(|arg| merge_fn_fill_deps(arg, egraph, read_deps, write_deps));
        }
        Seq(items) => {
            items
                .iter()
                .for_each(|item| merge_fn_fill_deps(item, egraph, read_deps, write_deps));
        }
        Construct(func, args, value_args) => {
            // A nested constructor mint inside the merge reads (hash-cons dedup)
            // and writes the target view table. Ported from PR #933.
            read_deps.insert(egraph.funcs[*func].table);
            write_deps.insert(egraph.funcs[*func].table);
            args.iter()
                .chain(value_args.iter())
                .for_each(|arg| merge_fn_fill_deps(arg, egraph, read_deps, write_deps));
        }
        IfEq { a, b, then, els } => {
            // Both branches may run (the condition is data-dependent), so union
            // the deps of every sub-expression.
            for m in [a, b, then, els] {
                merge_fn_fill_deps(m, egraph, read_deps, write_deps);
            }
        }
        AssertEq | Old | New | OldCol(..) | NewCol(..) | KeyCol(..) | Const(..) => {}
    }
}

fn merge_fn_to_callback(
    merge: &MergeFn,
    schema_math: SchemaMath,
    function_name: &str,
    egraph: &mut EGraph,
) -> Box<core_relations::MergeFn> {
    let resolved = merge_fn_resolve_columns(merge, function_name, egraph);
    assert_eq!(
        resolved.len(),
        schema_math.n_vals(),
        "merge for {function_name} must have one entry per value column"
    );

    // Fast path: the overwhelmingly common single-output function. Avoids the per-column loop
    // and the `SmallVec` of merged values, keeping the hot path identical to the pre-multi-value
    // form. A single-output merge always resolves to a one-element `Vec` (a bare scalar merge or
    // `Columns` of length 1), so this branch is exhaustive for `n_vals == 1`.
    if let [resolved] = resolved.as_slice() {
        let resolved = resolved.clone();
        let ret_val_col = schema_math.ret_val_col();
        return Box::new(move |state, cur, new, out| {
            let timestamp = new[schema_math.ts_col()];

            let mut changed = false;

            let ret_val = {
                let out = resolved.run(state, cur, new, schema_math.n_keys, 0, timestamp);
                changed |= cur[ret_val_col] != out;
                out
            };

            let subsume = schema_math.subsume.then(|| {
                let cur = cur[schema_math.subsume_col()];
                let new = new[schema_math.subsume_col()];
                let out = combine_subsumed(cur, new);
                changed |= cur != out;
                out
            });
            if changed {
                out.extend_from_slice(new);
                schema_math.write_table_row(
                    out,
                    RowVals {
                        timestamp,
                        subsume,
                        ret_val: Some(ret_val),
                    },
                );
            }

            changed
        });
    }

    Box::new(move |state, cur, new, out| {
        let timestamp = new[schema_math.ts_col()];

        let mut changed = false;

        // Compute each merged value column. `cur` and `new` are full rows, so a tuple-output
        // column's merge may reference any output column via `OldCol`/`NewCol`.
        let mut merged_vals = SmallVec::<[Value; 4]>::new();
        for (i, col_merge) in resolved.iter().enumerate() {
            let out_val = col_merge.run(state, cur, new, schema_math.n_keys, i, timestamp);
            changed |= cur[schema_math.val_col(i)] != out_val;
            merged_vals.push(out_val);
        }

        let subsume = schema_math.subsume.then(|| {
            let cur = cur[schema_math.subsume_col()];
            let new = new[schema_math.subsume_col()];
            let out = combine_subsumed(cur, new);
            changed |= cur != out;
            out
        });
        if changed {
            out.extend_from_slice(new);
            for (i, val) in merged_vals.iter().enumerate() {
                out[schema_math.val_col(i)] = *val;
            }
            schema_math.write_table_row(
                out,
                RowVals {
                    timestamp,
                    subsume,
                    ret_val: None,
                },
            );
        }

        changed
    })
}

/// Resolve a merge into one [`ResolvedMergeFn`] per value column. Single-value merges (the common
/// case) resolve to a one-element vector; [`MergeFn::Columns`] resolves to one entry per column.
fn merge_fn_resolve_columns(
    merge: &MergeFn,
    function_name: &str,
    egraph: &mut EGraph,
) -> Vec<ResolvedMergeFn> {
    match merge {
        MergeFn::Columns(cols) => cols
            .iter()
            .map(|col| merge_fn_resolve(col, function_name, egraph))
            .collect(),
        other => vec![merge_fn_resolve(other, function_name, egraph)],
    }
}

fn merge_fn_resolve(merge: &MergeFn, function_name: &str, egraph: &mut EGraph) -> ResolvedMergeFn {
    match merge {
        MergeFn::Const(v) => ResolvedMergeFn::Const(*v),
        MergeFn::Old => ResolvedMergeFn::Old,
        MergeFn::New => ResolvedMergeFn::New,
        MergeFn::OldCol(i) => ResolvedMergeFn::OldCol(*i),
        MergeFn::NewCol(i) => ResolvedMergeFn::NewCol(*i),
        MergeFn::KeyCol(i) => ResolvedMergeFn::KeyCol(*i),
        MergeFn::Columns(_) => {
            panic!("nested Columns merge is not supported (Columns must be top-level)")
        }
        MergeFn::AssertEq => ResolvedMergeFn::AssertEq {
            panic: egraph.new_panic(format!(
                "Illegal merge attempted for function {function_name}"
            )),
        },
        MergeFn::UnionId => ResolvedMergeFn::UnionId {
            uf_table: egraph.uf_table,
        },
        // `--native-merge`: stage the FD-conflict congruence union directly into
        // the named per-sort `@UF_Sf`'s union-find (the table that owns this
        // view's eclass column) instead of the global `$uf`. Resolves to the same
        // `ResolvedMergeFn::UnionId` runtime form — only the target table differs
        // — so the staged `[cur, new, ts]` lands in the per-sort UF, whose leader
        // change drives the view's relational `@rebuild_rule*` re-canonicalization.
        MergeFn::UnionIntoUf(uf_func) => ResolvedMergeFn::UnionId {
            uf_table: egraph.funcs[*uf_func].table,
        },
        // `--native-merge` relational path: stage the FD-conflict congruence edge
        // as a row into the per-sort relational `@UF_S` parent table (a plain
        // 2-key `(S S) -> Unit` function with `:merge old`), matching the
        // rule-encoded `(set (@UF_S larger smaller) ())`. We capture the parent
        // table's schema math so the runtime can shape the `[larger, smaller,
        // unit, ts]` row.
        MergeFn::UnionIntoParentTable { parent_table, unit } => {
            let info = &egraph.funcs[*parent_table];
            ResolvedMergeFn::UnionIntoParentTable {
                parent_table: info.table,
                parent_math: info.schema_math(),
                unit: *unit,
            }
        }
        // `--native-merge` PROOF mode (`col0`): stage a proof-carrying congruence
        // edge into the proof-mode `@UF_Sf`. The `Trans`/`Sym` proof constructors
        // are interned via `TableAction`s, mirroring `compose_uf_proof`.
        MergeFn::UnionIntoUfWithProof {
            uf,
            trans,
            sym,
            eclass_col,
            proof_col,
        } => ResolvedMergeFn::UnionIntoUfWithProof {
            uf_table: egraph.funcs[*uf].table,
            trans: TableAction::new(egraph, *trans),
            sym: TableAction::new(egraph, *sym),
            eclass_col: *eclass_col,
            proof_col: *proof_col,
        },
        // `--native-merge` PROOF mode on a single-output backend (FlowLog/Feldera):
        // the 2-table proof-congruence merge is resolved by that backend's host
        // interpreter, never by the native bridge.
        MergeFn::UnionIntoParentTableWithProof { .. } => {
            panic!(
                "MergeFn::UnionIntoParentTableWithProof is a single-output-backend \
                 (FlowLog/Feldera) merge and is not resolved on the native bridge"
            )
        }
        // `--native-merge` PROOF mode (`col1`): proof of the surviving min eclass.
        MergeFn::EclassMinProof {
            eclass_col,
            proof_col,
        } => ResolvedMergeFn::EclassMinProof {
            eclass_col: *eclass_col,
            proof_col: *proof_col,
        },
        // NB: The primitive and function-based merge functions heap allocate a single callback
        // for each layer of nesting. This introduces a bit of overhead, particularly for cases
        // that look like `(f old new)` or `(f new old)`. We could special-case common cases in
        // this function if that overhead shows up.
        MergeFn::Primitive(prim, args) => ResolvedMergeFn::Primitive {
            prim: *prim,
            args: args
                .iter()
                .map(|arg| merge_fn_resolve(arg, function_name, egraph))
                .collect::<Vec<_>>(),
            panic: egraph.new_panic(format!(
                "Merge function for {function_name} primitive call failed"
            )),
        },
        MergeFn::Function(func, args) => {
            let func_info = &egraph.funcs[*func];
            assert_eq!(
                func_info.schema.len(),
                args.len() + 1,
                "Merge function for {function_name} must match function arity for {}",
                func_info.name
            );
            let identity_on_miss = matches!(func_info.default_val, DefaultVal::Identity);
            ResolvedMergeFn::Function {
                func: TableAction::new(egraph, *func),
                panic: egraph.new_panic(format!(
                    "Lookup on {} failed in the merge function for {function_name}",
                    func_info.name
                )),
                args: args
                    .iter()
                    .map(|arg| merge_fn_resolve(arg, function_name, egraph))
                    .collect::<Vec<_>>(),
                identity_on_miss,
            }
        }
        // `:merge`-multiple-actions variants ported from PR #933.
        MergeFn::TableInsert(func, args) => ResolvedMergeFn::TableInsert {
            table: TableAction::new(egraph, *func),
            args: args
                .iter()
                .map(|arg| merge_fn_resolve(arg, function_name, egraph))
                .collect::<Vec<_>>(),
        },
        MergeFn::Seq(items) => ResolvedMergeFn::Seq(
            items
                .iter()
                .map(|item| merge_fn_resolve(item, function_name, egraph))
                .collect::<Vec<_>>(),
        ),
        MergeFn::Construct(func, args, value_args) => {
            let func_info = &egraph.funcs[*func];
            let num_values = func_info.schema.len() - func_info.n_keys;
            debug_assert_eq!(
                func_info.schema.len(),
                args.len() + num_values,
                "Construct for {function_name}: key arity must be schema minus value columns for {}",
                func_info.name
            );
            debug_assert_eq!(
                value_args.len() + 1,
                num_values,
                "Construct for {function_name}: value_args must fill every value column \
                 except the minted output for {}",
                func_info.name
            );
            ResolvedMergeFn::Construct {
                table: TableAction::new(egraph, *func),
                args: args
                    .iter()
                    .map(|arg| merge_fn_resolve(arg, function_name, egraph))
                    .collect::<Vec<_>>(),
                value_args: value_args
                    .iter()
                    .map(|arg| merge_fn_resolve(arg, function_name, egraph))
                    .collect::<Vec<_>>(),
            }
        }
        MergeFn::IfEq { a, b, then, els } => ResolvedMergeFn::IfEq {
            a: Box::new(merge_fn_resolve(a, function_name, egraph)),
            b: Box::new(merge_fn_resolve(b, function_name, egraph)),
            then: Box::new(merge_fn_resolve(then, function_name, egraph)),
            els: Box::new(merge_fn_resolve(els, function_name, egraph)),
        },
    }
}

/// This enum is taking the place of a
/// `Box<dyn Fn(&mut ExecutionState, Value, Value, Value) -> Value + Send + Sync>`
/// to avoid extra boxes. It stores the data needed to run a `MergeFn` without
/// holding onto any references, so it can be `move`d inside the `core_relations::MergeFn`.
#[derive(Clone)]
enum ResolvedMergeFn {
    Const(Value),
    Old,
    New,
    OldCol(usize),
    NewCol(usize),
    KeyCol(usize),
    AssertEq {
        panic: ExternalFunctionId,
    },
    UnionId {
        uf_table: TableId,
    },
    /// `--native-merge` relational path (no `--native-uf`): write the FD-conflict
    /// congruence edge as a `[larger, smaller, unit, ts]` row into the relational
    /// `@UF_S` parent table. See [`MergeFn::UnionIntoParentTable`].
    UnionIntoParentTable {
        parent_table: TableId,
        parent_math: SchemaMath,
        unit: Value,
    },
    /// `--native-merge` PROOF mode, `col0` (eclass): stage a proof-carrying
    /// congruence edge into the proof-mode `@UF_Sf` and return the min eclass.
    /// See [`MergeFn::UnionIntoUfWithProof`].
    UnionIntoUfWithProof {
        uf_table: TableId,
        trans: TableAction,
        sym: TableAction,
        eclass_col: usize,
        proof_col: usize,
    },
    /// `--native-merge` PROOF mode, `col1` (proof): the term proof of the
    /// surviving (min) eclass. See [`MergeFn::EclassMinProof`].
    EclassMinProof {
        eclass_col: usize,
        proof_col: usize,
    },
    Primitive {
        prim: ExternalFunctionId,
        args: Vec<ResolvedMergeFn>,
        panic: ExternalFunctionId,
    },
    Function {
        func: TableAction,
        args: Vec<ResolvedMergeFn>,
        panic: ExternalFunctionId,
        /// When the looked-up function has a [`DefaultVal::Identity`] default
        /// (the term-encoder's `@UF_Sf` flat union-find index), a missing key
        /// returns the (single) lookup argument unchanged ("find-or-self")
        /// instead of panicking. This expresses the canonicalize-at-creation
        /// `(__UF_Sf x)` lookup inside a `:merge`-multiple-actions body
        /// (PR #933 term-build lowering).
        identity_on_miss: bool,
    },
    /// `(set (f ...) v)` inside a merge: insert a full row into `table`,
    /// respecting that table's own merge. Ported from PR #933.
    TableInsert {
        table: TableAction,
        args: Vec<ResolvedMergeFn>,
    },
    /// Run each item in order for its effects; return the value of the last.
    /// Ported from PR #933.
    Seq(Vec<ResolvedMergeFn>),
    /// Mint a pair-valued constructor inside a merge (FreshId output column,
    /// `value_args` for the remaining value columns) and return its output
    /// e-class. Ported from PR #933.
    Construct {
        table: TableAction,
        args: Vec<ResolvedMergeFn>,
        value_args: Vec<ResolvedMergeFn>,
    },
    /// Conditional: if `a`'s value equals `b`'s value, run `then`; else run `els`.
    /// Returns the value of whichever branch ran. Used to guard a term-building
    /// merge `Seq` on `old != new`. See [`MergeFn::IfEq`].
    IfEq {
        a: Box<ResolvedMergeFn>,
        b: Box<ResolvedMergeFn>,
        then: Box<ResolvedMergeFn>,
        els: Box<ResolvedMergeFn>,
    },
}

impl ResolvedMergeFn {
    /// Compute the merged value for value column `self_col`.
    ///
    /// `cur` and `new` are the full conflicting rows. `n_keys` is the number of key columns, so
    /// value column `i` lives at `cur[n_keys + i]`. `Old`/`New` refer to `self_col`'s value, while
    /// `OldCol`/`NewCol` reference an explicit column — this lets a tuple-output column's merge
    /// read any other output column.
    fn run(
        &self,
        state: &mut ExecutionState,
        cur: &[Value],
        new: &[Value],
        n_keys: usize,
        self_col: usize,
        ts: Value,
    ) -> Value {
        match self {
            ResolvedMergeFn::Const(v) => *v,
            ResolvedMergeFn::Old => cur[n_keys + self_col],
            ResolvedMergeFn::New => new[n_keys + self_col],
            ResolvedMergeFn::OldCol(i) => cur[n_keys + i],
            ResolvedMergeFn::NewCol(i) => new[n_keys + i],
            // Key (input) column `i`. The two FD-conflicting rows share the same
            // key, so `cur[i] == new[i]`; read `cur`.
            ResolvedMergeFn::KeyCol(i) => cur[*i],
            ResolvedMergeFn::AssertEq { panic } => {
                let (cur, new) = (cur[n_keys + self_col], new[n_keys + self_col]);
                if cur != new {
                    let res = state.call_external_func(*panic, &[]);
                    assert_eq!(res, None);
                }
                cur
            }
            ResolvedMergeFn::UnionId { uf_table } => {
                let (cur, new) = (cur[n_keys + self_col], new[n_keys + self_col]);
                if cur != new {
                    state.stage_insert(*uf_table, &[cur, new, ts]);
                    // We pick the minimum when unioning. This matches the original egglog
                    // behavior. THIS MUST MATCH THE UNION-FIND IMPLEMENTATION!
                    std::cmp::min(cur, new)
                } else {
                    cur
                }
            }
            ResolvedMergeFn::UnionIntoParentTable {
                parent_table,
                parent_math,
                unit,
            } => {
                let (cur, new) = (cur[n_keys + self_col], new[n_keys + self_col]);
                if cur != new {
                    // Write the union edge into the relational `@UF_S` parent
                    // table, EXACTLY as the rule-encoded `union()` helper does:
                    // `(set (@UF_S larger smaller) ())`, where the key is the
                    // larger id and the value is the smaller id. The relational
                    // maintenance rulesets (singleparent / path_compress /
                    // uf_function_index) then propagate this into `@UF_Sf` and the
                    // rule-based rebuild re-canonicalizes the views. Keeping the
                    // min as the surviving eclass matches the union-by-min the
                    // `:merge (ordering-min old new)` index relies on.
                    let larger = std::cmp::max(cur, new);
                    let smaller = std::cmp::min(cur, new);
                    // Row layout for the 2-key `(S S) -> Unit` parent table:
                    // `[larger, smaller, unit, ts]` (no subsume column).
                    let mut row: SmallVec<[Value; 4]> = smallvec::smallvec![larger, smaller];
                    parent_math.write_table_row(
                        &mut row,
                        RowVals {
                            timestamp: ts,
                            subsume: None,
                            ret_val: Some(*unit),
                        },
                    );
                    state.stage_insert(*parent_table, &row);
                    smaller
                } else {
                    cur
                }
            }
            ResolvedMergeFn::UnionIntoUfWithProof {
                uf_table,
                trans,
                sym,
                eclass_col,
                proof_col,
            } => {
                let cur_eclass = cur[n_keys + *eclass_col];
                let new_eclass = new[n_keys + *eclass_col];
                if cur_eclass == new_eclass {
                    return cur_eclass;
                }
                let cur_proof = cur[n_keys + *proof_col];
                let new_proof = new[n_keys + *proof_col];
                // Match the rule-encoded `@congruence_rule`'s orientation EXACTLY:
                // it writes `(set (@UF_Sf larger smaller) (Trans larger_pf (Sym
                // smaller_pf)))`, where `larger = (ordering-max old new)` and the
                // edge proof carries the larger row's proof first. The proof-mode
                // UF's `get_proof` consumes the per-edge proof verbatim for a
                // forward step (`larger -> smaller` IS the leader-change
                // direction, since the UF tie-break makes `smaller` the leader),
                // so the stored proof must be the un-Sym'd `Trans(larger, Sym(smaller))`.
                let (larger, smaller, larger_pf, smaller_pf) = if cur_eclass > new_eclass {
                    (cur_eclass, new_eclass, cur_proof, new_proof)
                } else {
                    (new_eclass, cur_eclass, new_proof, cur_proof)
                };
                // edge_proof = Trans(larger_pf, Sym(smaller_pf)), proving
                // `larger = smaller` (larger = f(children) = smaller).
                let sym_smaller = sym
                    .lookup_or_insert(state, &[smaller_pf])
                    .expect("Sym constructor lookup_or_insert failed in congruence merge");
                let edge_proof = trans
                    .lookup_or_insert(state, &[larger_pf, sym_smaller])
                    .expect("Trans constructor lookup_or_insert failed in congruence merge");
                // Proof-mode UF writes are 4-column `[lhs, rhs, proof, ts]`.
                state.stage_insert(*uf_table, &[larger, smaller, edge_proof, ts]);
                smaller
            }
            ResolvedMergeFn::EclassMinProof {
                eclass_col,
                proof_col,
            } => {
                // The proof of the surviving (min) eclass: matches the eclass kept
                // by `UnionIntoUfWithProof` (which returns `min`).
                if cur[n_keys + *eclass_col] <= new[n_keys + *eclass_col] {
                    cur[n_keys + *proof_col]
                } else {
                    new[n_keys + *proof_col]
                }
            }
            // NB: The primitive and function-based merge functions heap allocate a single callback
            // for each layer of nesting. This introduces a bit of overhead, particularly for cases
            // that look like `(f old new)` or `(f new old)`. We could special-case common cases in
            // this function if that overhead shows up.
            ResolvedMergeFn::Primitive { prim, args, panic } => {
                let args = args
                    .iter()
                    .map(|arg| arg.run(state, cur, new, n_keys, self_col, ts))
                    .collect::<Vec<_>>();

                match state.call_external_func(*prim, &args) {
                    Some(result) => result,
                    None => {
                        let res = state.call_external_func(*panic, &[]);
                        assert_eq!(res, None);
                        cur[n_keys + self_col]
                    }
                }
            }
            ResolvedMergeFn::Function {
                func,
                args,
                panic,
                identity_on_miss,
            } => {
                // see github.com/egraphs-good/egglog/pull/287
                //
                // The `cur == new` short-circuit is only valid when this
                // `Function` IS the merge resolving `self_col` (the classic
                // `(f old new)` value-fold case). For an `identity_on_miss`
                // canon lookup `(__UF_Sf x)` nested inside a `Construct`/`Seq`
                // (PR #933 term build), the lookup must run unconditionally —
                // `cur`/`new` here are the conflicting FD rows, not this lookup's
                // argument.
                if !identity_on_miss && cur[n_keys + self_col] == new[n_keys + self_col] {
                    return cur[n_keys + self_col];
                }

                let args = args
                    .iter()
                    .map(|arg| arg.run(state, cur, new, n_keys, self_col, ts))
                    .collect::<Vec<_>>();

                // Merge functions dispatch to another function that may be
                // a constructor (mint fresh id on miss) or a custom function
                // (return `None` → panic). `lookup_or_insert` preserves
                // both behaviors; the pure-read `lookup` would skip
                // constructor minting.
                func.lookup_or_insert(state, &args).unwrap_or_else(|| {
                    if *identity_on_miss {
                        // Find-or-self: a missing key in an `Identity`-default
                        // `@UF_Sf` index returns the (single) lookup argument
                        // unchanged, NOT a panic — the canonicalize-at-creation
                        // `(__UF_Sf x)` semantics.
                        debug_assert_eq!(
                            args.len(),
                            1,
                            "identity_on_miss canon lookup must be single-arg"
                        );
                        args[0]
                    } else {
                        let res = state.call_external_func(*panic, &[]);
                        assert_eq!(res, None);
                        cur[n_keys + self_col]
                    }
                })
            }
            // `:merge`-multiple-actions runtime (ported from PR #933).
            ResolvedMergeFn::TableInsert { table, args } => {
                let row = args
                    .iter()
                    .map(|arg| arg.run(state, cur, new, n_keys, self_col, ts))
                    .collect::<Vec<_>>();
                // Insert respects the target table's own merge; the timestamp
                // column is appended by `TableAction::insert`.
                table.insert(state, row.into_iter());
                // Return value is discarded by the enclosing `Seq`.
                cur[n_keys + self_col]
            }
            ResolvedMergeFn::Seq(items) => {
                let mut result = cur[n_keys + self_col];
                for item in items {
                    result = item.run(state, cur, new, n_keys, self_col, ts);
                }
                result
            }
            ResolvedMergeFn::Construct {
                table,
                args,
                value_args,
            } => {
                let key = args
                    .iter()
                    .map(|arg| arg.run(state, cur, new, n_keys, self_col, ts))
                    .collect::<SmallVec<[Value; 4]>>();
                let vals = value_args
                    .iter()
                    .map(|arg| arg.run(state, cur, new, n_keys, self_col, ts))
                    .collect::<SmallVec<[Value; 4]>>();
                // Constructor: always mints on miss, so this is `Some`.
                table
                    .lookup_or_insert_multi(state, &key, &vals)
                    .unwrap_or(cur[n_keys + self_col])
            }
            ResolvedMergeFn::IfEq { a, b, then, els } => {
                // Evaluate the (side-effect-free) condition operands first, then
                // run ONLY the taken branch — so a guarded term-building `Seq`
                // mints nothing when the values are already equal.
                let av = a.run(state, cur, new, n_keys, self_col, ts);
                let bv = b.run(state, cur, new, n_keys, self_col, ts);
                if av == bv {
                    then.run(state, cur, new, n_keys, self_col, ts)
                } else {
                    els.run(state, cur, new, n_keys, self_col, ts)
                }
            }
        }
    }
}

/// This is an intern-able struct that holds all the data needed
/// to do table operations with an [`ExecutionState`], assuming
/// that the [`FunctionId`] for the table is known ahead of time.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct TableAction {
    table: TableId,
    table_math: SchemaMath,
    default: Option<MergeVal>,
    timestamp: CounterId,
}

impl TableAction {
    /// Create a new `TableAction` to be used later.
    /// This requires access to the `egglog_bridge::EGraph`.
    pub fn new(egraph: &EGraph, func: FunctionId) -> TableAction {
        let func_info = &egraph.funcs[func];
        assert!(
            !func_info.is_uf(),
            "{}",
            FunctionConfigError::TableActionUsesUf(func_info.name.to_string())
        );
        TableAction {
            table: func_info.table,
            table_math: func_info.schema_math(),
            default: match &func_info.default_val {
                DefaultVal::FreshId => Some(MergeVal::Counter(egraph.id_counter)),
                // `Identity` is consumed only by the rule-compilation lookup
                // path (see `lookup_with_subsumed`); the `TableAction` exec path
                // never looks up an Identity-default function, so fall back to
                // plain `lookup` semantics here.
                DefaultVal::Fail | DefaultVal::Identity => None,
                DefaultVal::Const(val) => Some(MergeVal::Constant(*val)),
            },
            timestamp: egraph.timestamp_counter,
        }
    }

    /// Look up a row and return its return-value column, or `None` if the
    /// key is not present. **This is a pure read**: it never inserts a row,
    /// regardless of the table's configured [`DefaultVal`].
    ///
    /// For the lookup-or-insert behavior that mints fresh eclass IDs for
    /// constructors, use [`TableAction::lookup_or_insert`].
    pub fn lookup(&self, state: &ExecutionState, key: &[Value]) -> Option<Value> {
        state
            .get_table(self.table)
            .get_row(key)
            .map(|row| row.vals[self.table_math.ret_val_col()])
    }

    /// Return the current number of rows in this table.
    pub fn row_count(&self, state: &ExecutionState) -> usize {
        state.get_table(self.table).len()
    }

    /// Look up a row, inserting the configured default value if absent.
    /// For constructor tables this mints a fresh eclass ID; for custom
    /// functions (no default) this behaves identically to
    /// [`TableAction::lookup`].
    ///
    /// This is a write operation — only safe in action contexts. See
    /// issue #772.
    pub fn lookup_or_insert(&self, state: &mut ExecutionState, key: &[Value]) -> Option<Value> {
        match self.default {
            Some(default) => {
                let timestamp =
                    MergeVal::Constant(Value::from_usize(state.read_counter(self.timestamp)));
                let mut merge_vals = SmallVec::<[MergeVal; 3]>::new();
                // Build just the non-key portion (single value + ts + subsume) of the row.
                // `lookup_or_insert` is only used for single-value functions/constructors.
                SchemaMath {
                    n_keys: 0,
                    func_cols: 1,
                    ..self.table_math
                }
                .write_table_row(
                    &mut merge_vals,
                    RowVals {
                        timestamp,
                        subsume: self
                            .table_math
                            .subsume
                            .then_some(MergeVal::Constant(NOT_SUBSUMED)),
                        ret_val: Some(default),
                    },
                );
                Some(
                    state.predict_val(self.table, key, merge_vals.iter().copied())
                        [self.table_math.ret_val_col()],
                )
            }
            None => self.lookup(state, key),
        }
    }

    /// Multi-value variant of [`TableAction::lookup_or_insert`] for a pair-valued
    /// constructor `(children) -> (output, extra...)`: the first value column
    /// (`output`) is minted (the configured `FreshId` default) and the rest are
    /// written from `provided_vals` (e.g. the proof). Returns the minted `output`.
    ///
    /// Idempotent: an already-present key returns its existing `output` and writes
    /// nothing, so it can be evaluated more than once and yield the same id. This is
    /// a write operation, only safe in action/merge contexts. Ported from PR #933.
    pub fn lookup_or_insert_multi(
        &self,
        state: &mut ExecutionState,
        key: &[Value],
        provided_vals: &[Value],
    ) -> Option<Value> {
        match self.default {
            Some(default) => {
                debug_assert_eq!(
                    self.table_math.n_vals(),
                    1 + provided_vals.len(),
                    "lookup_or_insert_multi: provided_vals must fill every value \
                     column except the minted first one"
                );
                let timestamp =
                    MergeVal::Constant(Value::from_usize(state.read_counter(self.timestamp)));
                // Non-key columns, in order: [output (minted), provided.., ts, subsume?].
                let mut merge_vals = SmallVec::<[MergeVal; 4]>::new();
                merge_vals.push(default);
                merge_vals.extend(provided_vals.iter().map(|v| MergeVal::Constant(*v)));
                merge_vals.push(timestamp);
                if self.table_math.subsume {
                    merge_vals.push(MergeVal::Constant(NOT_SUBSUMED));
                }
                // The first value column (the minted output) is at `ret_val_col()`.
                Some(
                    state.predict_val(self.table, key, merge_vals.iter().copied())
                        [self.table_math.ret_val_col()],
                )
            }
            None => self.lookup(state, key),
        }
    }

    /// Insert a row into this table.
    pub fn insert(&self, state: &mut ExecutionState, row: impl Iterator<Item = Value>) {
        let ts = Value::from_usize(state.read_counter(self.timestamp));
        let mut scratch = row.collect::<SmallVec<[_; 8]>>();
        self.table_math.write_table_row(
            &mut scratch,
            RowVals {
                timestamp: ts,
                subsume: self.table_math.subsume.then_some(NOT_SUBSUMED),
                ret_val: None,
            },
        );
        state.stage_insert(self.table, &scratch);
    }

    /// Delete a row from this table.
    pub fn remove(&self, state: &mut ExecutionState, key: &[Value]) {
        state.stage_remove(self.table, key);
    }

    /// Subsume a row in this table.
    pub fn subsume(&self, state: &mut ExecutionState, key: impl Iterator<Item = Value>) {
        let ts = Value::from_usize(state.read_counter(self.timestamp));
        let mut scratch = key.collect::<SmallVec<[_; 8]>>();

        let ret_val = self.lookup(state, &scratch).expect("subsume lookup failed");

        self.table_math.write_table_row(
            &mut scratch,
            RowVals {
                timestamp: ts,
                subsume: Some(SUBSUMED),
                ret_val: Some(ret_val),
            },
        );
        state.stage_insert(self.table, &scratch);
    }
}

/// A variant of `TableAction` for the union-find.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnionAction {
    table: TableId,
    timestamp: CounterId,
}

impl UnionAction {
    /// Create a new `UnionAction` to be used later.
    /// This requires access to the `egglog_bridge::EGraph`.
    pub fn new(egraph: &EGraph) -> UnionAction {
        UnionAction {
            table: egraph.uf_table,
            timestamp: egraph.timestamp_counter,
        }
    }

    /// Union two values.
    pub fn union(&self, state: &mut ExecutionState, x: Value, y: Value) {
        let ts = Value::from_usize(state.read_counter(self.timestamp));
        state.stage_insert(self.table, &[x, y, ts]);
    }
}

/// Compose a `Proof` term for a native-UF leader change from a reconstructed
/// proof path (`steps`, produced by the union-find's provenance graph).
///
/// Folds the path into a single `Proof` value: each step contributes its
/// per-edge proof (the value stored in the write's proof column), wrapped in
/// `Sym` for a backward step, and consecutive steps are chained with `Trans`.
/// The `Trans`/`Sym` constructor rows are minted/hash-consed via
/// `TableAction::lookup_or_insert`, so the resulting proof is a real, interned
/// term visible to egglog rules.
fn compose_uf_proof(
    state: &mut ExecutionState,
    trans_action: &TableAction,
    sym_action: &TableAction,
    steps: &[ProofStep],
) -> Value {
    let mut acc: Option<Value> = None;
    for step in steps {
        // Each step's per-edge proof proves `write_lhs = write_rhs`.
        let step_proof = match step.reason {
            ProofReason::Forward(p) => p,
            ProofReason::Backward(p) => sym_action
                .lookup_or_insert(state, &[p])
                .expect("Sym constructor lookup_or_insert failed"),
        };
        acc = Some(match acc {
            None => step_proof,
            Some(prev) => trans_action
                .lookup_or_insert(state, &[prev, step_proof])
                .expect("Trans constructor lookup_or_insert failed"),
        });
    }
    acc.expect("get_proof returned an empty path for a real leader change")
}

fn run_rules_impl(
    db: &mut Database,
    rule_info: &mut DenseIdMapWithReuse<RuleId, RuleInfo>,
    rules: &[RuleId],
    next_ts: Timestamp,
    report_level: ReportLevel,
) -> Result<RuleSetReport> {
    for rule in rules {
        let info = &mut rule_info[*rule];
        if info.cached_plan.is_none() {
            info.cached_plan = Some(info.query.build_cached_plan(db, &info.desc)?);
        }
    }
    let mut rsb = db.new_rule_set();
    for rule in rules {
        let info = &mut rule_info[*rule];
        let cached_plan = info.cached_plan.as_ref().unwrap();
        info.query
            .add_rules_from_cached(&mut rsb, info.last_run_at, cached_plan);
        info.last_run_at = next_ts;
    }
    let ruleset = rsb.build();
    Ok(db.run_rule_set(&ruleset, report_level))
}

// These markers are just used to make it easy to distinguish time spent in
// incremental vs. nonincremental rebuilds in time-based profiles.

#[inline(never)]
fn marker_incremental_rebuild<R>(f: impl FnOnce() -> R) -> R {
    f()
}

#[inline(never)]
fn marker_nonincremental_rebuild<R>(f: impl FnOnce() -> R) -> R {
    f()
}

/// A useful type definition for external functions that need to pass data
/// to outside code, such as `Panic`.
pub type SideChannel<T> = Arc<Mutex<Option<T>>>;

/// An external function used to grab a value out of the database matching a
/// particular query.
//
// TODO: once we have parallelism wired in, we'll want to replace this with a
// more efficient solution (e.g. one based on crossbeam or arcswap).
/// This is a variant on [`Panic`] that avoids eager construction of the panic message.
///
/// The main thing this is used for is to avoid constructing the panic message ahead of time during
/// a call to [`RuleBuilder::call_external_func`]; these panic messages are often quite rare and
/// may never need to be constructed at all. Furthermore, a closure to produce the panic message in
/// most cases need only close over a few cheap-to-clone values.
///
/// The downside of this, and why we do not use it everywhere, is that there's no natural "key"
/// that we can use to cache duplicate panic messages. We would need a more complex API to support
/// both and fully replace our use of `Panic`.
struct LazyPanic<F>(Arc<Lazy<String, F>>, SideChannel<String>);

impl<F: FnOnce() -> String + Send> ExternalFunction for LazyPanic<F> {
    fn invoke(&self, state: &mut core_relations::ExecutionState, args: &[Value]) -> Option<Value> {
        assert!(args.is_empty());
        state.trigger_early_stop();
        let mut guard = self.1.lock().unwrap();
        if guard.is_none() {
            *guard = Some(Lazy::force(&self.0).clone());
        }
        None
    }
}

impl<F> Clone for LazyPanic<F> {
    fn clone(&self) -> Self {
        LazyPanic(self.0.clone(), self.1.clone())
    }
}

/// An external function used to store a message when a panic occurs.
//
// TODO: once we have parallelism wired in, we'll want to replace this with a
// more efficient solution (e.g. one based on crossbeam or arcswap).
#[derive(Clone)]
struct Panic(String, SideChannel<String>);

impl EGraph {
    /// Create a new `ExternalFunction` that panics with the given message.
    pub fn new_panic(&mut self, message: String) -> ExternalFunctionId {
        *self
            .panic_funcs
            .entry(message.to_string())
            .or_insert_with(|| {
                let panic = Panic(message, self.panic_message.clone());
                self.db.add_external_function(Box::new(panic))
            })
    }

    pub fn new_panic_lazy(
        &mut self,
        message: impl FnOnce() -> String + Send + 'static,
    ) -> ExternalFunctionId {
        let lazy = Lazy::new(message);
        let panic = LazyPanic(Arc::new(lazy), self.panic_message.clone());
        self.db.add_external_function(Box::new(panic))
    }
}

impl ExternalFunction for Panic {
    fn invoke(&self, state: &mut core_relations::ExecutionState, args: &[Value]) -> Option<Value> {
        // TODO (egglog feature): change this to support interpolating panic messages
        assert!(args.is_empty());

        state.trigger_early_stop();
        let mut guard = self.1.lock().unwrap();
        if guard.is_none() {
            *guard = Some(self.0.clone());
        }
        None
    }
}

/// Heuristic for deciding whether to do an incremental or nonincremental
/// rebuild for a given table.
fn incremental_rebuild(uf_size: usize, table_size: usize, parallel: bool) -> bool {
    if parallel {
        uf_size <= (table_size / 16)
    } else {
        uf_size <= (table_size / 8)
    }
}

pub(crate) const SUBSUMED: Value = Value::new_const(1);
pub(crate) const NOT_SUBSUMED: Value = Value::new_const(0);
fn combine_subsumed(v1: Value, v2: Value) -> Value {
    std::cmp::max(v1, v2)
}

/// A struct helping with some calculations of where some information is stored at the
/// core-relations Table level for a given function.
///
/// Functions can have multiple "output columns" in the underlying core-relations layer depending
/// on whether different features are enabled. Roughly, tables are laid out as:
///
/// > `[key0, ..., keyn, value0, ..., valuem, timestamp, subsume?]`
///
/// Where there are `n+1` key columns, `m+1` value (return) columns, and columns marked with a
/// question mark are optional, depending on the egraph and table-level configuration.
///
/// Most functions have a single value column (`m == 0`); tuple-output (multi-value) functions
/// have more than one. The functional dependency is always from the key columns to the value
/// columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SchemaMath {
    /// Whether or not the table is enabled for subsumption.
    subsume: bool,
    /// The number of key (input) columns.
    n_keys: usize,
    /// The number of columns in the function (keys plus all value/return columns).
    func_cols: usize,
}

/// A struct containing possible non-key portions of a table row. To be used with
/// [`SchemaMath::write_table_row`].
///
/// This is not to be confused with [`FunctionRow`], which is higher-level and for public uses.
struct RowVals<T> {
    /// The timestamp for the row.
    timestamp: T,
    /// The subsumption tag for the row. Only relevant if the table has subsumption enabled.
    subsume: Option<T>,
    /// The return value of the row. Return values are mandatory but callers may have already
    /// filled it in.
    ret_val: Option<T>,
}

// `FunctionRow` is now defined in `egglog-backend-trait`; the re-export at
// the top of this module brings it back into `egglog_bridge::FunctionRow`.

impl SchemaMath {
    fn write_table_row<T: Clone>(
        &self,
        row: &mut impl HasResizeWith<T>,
        RowVals {
            timestamp,
            subsume,
            ret_val,
        }: RowVals<T>,
    ) {
        row.resize_with(self.table_columns(), || timestamp.clone());
        row[self.ts_col()] = timestamp;
        if let Some(ret_val) = ret_val {
            row[self.ret_val_col()] = ret_val;
        }
        if let Some(subsume) = subsume {
            row[self.subsume_col()] = subsume;
        } else {
            assert!(
                !self.subsume,
                "subsume flag must be provided if subsumption is enabled"
            );
        }
    }

    fn num_keys(&self) -> usize {
        self.n_keys
    }

    /// The number of value (return) columns.
    fn n_vals(&self) -> usize {
        self.func_cols - self.n_keys
    }

    fn table_columns(&self) -> usize {
        self.func_cols + 1 /* timestamp */ + if self.subsume { 1 } else { 0 }
    }

    /// The column index of the `i`th value (return) column.
    fn val_col(&self, i: usize) -> usize {
        self.n_keys + i
    }

    /// The first value column. For single-output functions this is *the* return value; for
    /// tuple-output functions it is value column 0.
    fn ret_val_col(&self) -> usize {
        self.n_keys
    }

    fn ts_col(&self) -> usize {
        self.func_cols
    }

    #[track_caller]
    fn subsume_col(&self) -> usize {
        assert!(self.subsume);
        self.func_cols + 1
    }
}

#[derive(Error, Debug)]
#[error("Panic: {0}")]
struct PanicError(String);

/// Basic ad-hoc polymorphism around `resize_with` in order to get [`SchemaMath::write_table_row`]
/// to work with both `Vec` and `SmallVec`.
trait HasResizeWith<T>:
    AsMut<[T]> + AsRef<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    fn resize_with<F>(&mut self, new_size: usize, f: F)
    where
        F: FnMut() -> T;
}

impl<T> HasResizeWith<T> for Vec<T> {
    fn resize_with<F>(&mut self, new_size: usize, f: F)
    where
        F: FnMut() -> T,
    {
        self.resize_with(new_size, f);
    }
}

impl<T, A: smallvec::Array<Item = T>> HasResizeWith<T> for SmallVec<A> {
    fn resize_with<F>(&mut self, new_size: usize, f: F)
    where
        F: FnMut() -> T,
    {
        self.resize_with(new_size, f);
    }
}
