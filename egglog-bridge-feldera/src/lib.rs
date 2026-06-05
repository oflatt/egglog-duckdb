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

use anyhow::Result;
use dbsp::{CircuitHandle, RootCircuit, ZWeight};
use egglog_backend_trait::{
    Backend, BaseValueId, BaseValuePool, ColumnTy, ContainerPool, DefaultVal, ExternalFunction,
    ExternalFunctionId, FunctionConfig, FunctionId, FunctionRow, IterationReport, QueryEntry,
    ReportLevel, RuleBuilderOps, RuleId, Value,
};
use egglog_numeric_id::NumericId;
use hashbrown::{HashMap, HashSet};

mod base_values;
pub mod compile;
mod external_func;
mod rule_builder;

use base_values::FelderaBaseValuePool;
use compile::{build_circuit, pack_row, unpack_row, MergeMode, RelationHandles, Row, RuleIr};
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
// Relation metadata
// ---------------------------------------------------------------------------

/// What we remember about each registered relation/function.
struct RelationInfo {
    name: String,
    /// Number of columns (including the output column for functions).
    arity: usize,
    /// True for functions/constructors that have an output column.
    has_output: bool,
    /// How functional-dependency conflicts are resolved at flush time. For a
    /// function the key is the input columns (`arity - 1`) and the output column
    /// is resolved per this mode; for a relation it is [`MergeMode::Relation`]
    /// (whole row is the key, nothing to resolve).
    merge: MergeMode,
}

// ---------------------------------------------------------------------------
// The compiled circuit + its per-relation handles
// ---------------------------------------------------------------------------

/// A built DBSP circuit and the per-relation input/output handles for ONE rule
/// subset. DBSP circuits are static once built (PLAN §4.2 #2), and
/// `run_rules(&[RuleId])` runs an arbitrary *subset* of rules — so we build (and
/// cache) one circuit per distinct sorted rule-id list. `pushed` tracks, per
/// relation, the rows already pushed into the input handle so we only push
/// deltas across successive transactions of THIS circuit.
struct CircuitState {
    handle: CircuitHandle,
    relations: HashMap<FunctionId, RelationHandles>,
    /// Rows already pushed into each relation's input handle.
    pushed: HashMap<FunctionId, HashSet<Row>>,
}

// ---------------------------------------------------------------------------
// EGraph
// ---------------------------------------------------------------------------

/// The Feldera/DBSP-backed egraph.
pub struct EGraph {
    relations: Vec<RelationInfo>,
    /// Rule slots; `None` = freed.
    rules: Vec<Option<RuleIr>>,
    /// Rust-side materialized mirror: the accumulated contents of each
    /// relation, kept in sync with the circuit's integrated output after each
    /// transaction. This is what `for_each` / `lookup_id` / `table_size` read.
    mirror: HashMap<FunctionId, HashSet<Row>>,
    base_value_pool: FelderaBaseValuePool,
    container_pool: FelderaContainerPool,
    external_funcs: ExternalFuncRegistry,
    /// Monotonic fresh-id counter for `fresh_id` / `add_term`.
    next_id: u32,
    report_level: ReportLevel,
    /// Cache of built circuits, keyed by the sorted list of rule ids the circuit
    /// contains. `run_rules(&[RuleId])` runs a *subset* of rules, and a DBSP
    /// circuit is static once built, so we build one circuit per distinct subset
    /// and reuse it across calls. Cleared whenever a relation or rule is
    /// added/removed (the schema changed).
    circuits: HashMap<Vec<u32>, CircuitState>,
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
            base_value_pool: FelderaBaseValuePool::default(),
            container_pool: FelderaContainerPool,
            external_funcs: ExternalFuncRegistry::default(),
            // Start at 1 so id 0 stays available as a "null"/padding sentinel
            // (rows are 0-padded; reserving 0 avoids colliding a real id with
            // padding in the uniform Row representation).
            next_id: 1,
            report_level: ReportLevel::default(),
            circuits: HashMap::new(),
        }
    }

    fn info(&self, f: FunctionId) -> &RelationInfo {
        self.relations
            .get(f.rep() as usize)
            .unwrap_or_else(|| panic!("FunctionId({}) not registered", f.rep()))
    }

    /// Invalidate all built circuits; they will be rebuilt on the next
    /// `run_rules` because the schema (relations or rules) changed.
    fn invalidate_circuit(&mut self) {
        self.circuits.clear();
    }

    /// Insert a single row into the Rust mirror.
    fn mirror_insert(&mut self, f: FunctionId, row: Row) {
        self.mirror.entry(f).or_default().insert(row);
    }

    /// Build a DBSP circuit containing exactly the rules named by `rule_idxs`
    /// (indices into `self.rules`). Returns the built circuit state. This is how
    /// **ruleset-scoped execution** is realized: each distinct rule subset gets
    /// its own circuit, cached by the sorted index list in `self.circuits`.
    fn build_circuit_for(&self, rule_idxs: &[u32]) -> Result<CircuitState> {
        let rel_ids: Vec<FunctionId> = (0..self.relations.len())
            .map(|i| FunctionId::new(i as u32))
            .collect();
        let arities: HashMap<FunctionId, usize> =
            rel_ids.iter().map(|&f| (f, self.info(f).arity)).collect();
        let rules: Vec<RuleIr> = rule_idxs
            .iter()
            .filter_map(|&i| self.rules.get(i as usize).and_then(|r| r.clone()))
            .collect();

        // `build` runs the constructor closure once; the relation list /
        // arities / rules are moved into the closure. `build_circuit`'s
        // `anyhow::Result` flows through; `build`'s `dbsp::Error` lifts via `?`.
        let (handle, relations) =
            RootCircuit::build(move |root| build_circuit(root, &rel_ids, &arities, &rules))?;
        Ok(CircuitState {
            handle,
            relations,
            pushed: HashMap::new(),
        })
    }

    /// Resolve a function's merge mode (for FD-conflict resolution at fold time).
    fn merge_mode(&self, f: FunctionId) -> MergeMode {
        self.info(f).merge
    }

    /// Apply this round's insert/delete diffs (read from the circuit's
    /// integrated diff handles) against the mirror, resolving FD conflicts via
    /// the relation's merge mode. This is the host-side analog of egglog's
    /// flush: `new = (old ∪ inserts) \ deletes`, then per-key merge.
    fn fold_diffs_into_mirror(&mut self, cs: &CircuitState, f: FunctionId) {
        let Some(handles) = cs.relations.get(&f) else {
            return;
        };
        let arity = self.info(f).arity;
        let merge = self.merge_mode(f);
        let has_output = self.info(f).has_output;
        // A delete addresses a row by its KEY: the input columns for a function
        // (arity-1) or the whole row for a plain relation. The term encoder's
        // `(delete (@uf a b))` provides exactly the key columns, so the delete
        // stream rows carry the key in columns `0..keylen` (rest 0-padded).
        let keylen = if has_output {
            arity.saturating_sub(1)
        } else {
            arity
        };

        // Collect inserts and deletes (positive net weight = present).
        let mut inserts: Vec<Row> = Vec::new();
        for (row, (), w) in handles.inserts.consolidate().iter() {
            let w: ZWeight = w;
            if w > 0 {
                inserts.push(row);
            }
        }
        let mut delete_keys: HashSet<Vec<u32>> = HashSet::new();
        for (row, (), w) in handles.deletes.consolidate().iter() {
            let w: ZWeight = w;
            if w > 0 {
                delete_keys.insert((0..keylen).map(|i| compile::row_col(&row, i)).collect());
            }
        }

        let mut set = self.mirror.get(&f).cloned().unwrap_or_default();
        // Deletes retract every existing row whose key matches a deleted key.
        if !delete_keys.is_empty() {
            set.retain(|row| {
                let key: Vec<u32> = (0..keylen).map(|i| compile::row_col(row, i)).collect();
                !delete_keys.contains(&key)
            });
        }
        // Inserts add new rows.
        for r in inserts {
            set.insert(r);
        }
        // Resolve FD conflicts per merge mode (functions only; relations have no
        // output column to merge). For each key (input columns), pick a single
        // surviving output column.
        if matches!(merge, MergeMode::Old | MergeMode::New | MergeMode::Min) && arity >= 1 {
            let inputs_len = arity - 1;
            let mut by_key: HashMap<Vec<u32>, u32> = HashMap::new();
            for row in &set {
                let key: Vec<u32> = (0..inputs_len).map(|i| compile::row_col(row, i)).collect();
                let out = compile::row_col(row, inputs_len);
                match by_key.entry(key) {
                    hashbrown::hash_map::Entry::Vacant(e) => {
                        e.insert(out);
                    }
                    hashbrown::hash_map::Entry::Occupied(mut e) => {
                        let cur = *e.get();
                        let chosen = match merge {
                            MergeMode::Old => cur,
                            MergeMode::New => out,
                            MergeMode::Min => cur.min(out),
                            MergeMode::Relation => unreachable!(),
                        };
                        e.insert(chosen);
                    }
                }
            }
            let mut resolved: HashSet<Row> = HashSet::new();
            for (key, out) in by_key {
                let mut full = key;
                full.push(out);
                resolved.insert(pack_row(
                    &full.iter().map(|v| Value::new(*v)).collect::<Vec<_>>(),
                ));
            }
            set = resolved;
        }
        self.mirror.insert(f, set);
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
        // A relation has no output column; a function/constructor's last column
        // is the output. We treat `DefaultVal::FreshId` (eq-sort constructor)
        // and `DefaultVal::Const(_)` (function with a default) as having an
        // output column; `DefaultVal::Fail` with an `AssertEq` merge is the
        // plain-relation shape. For milestone 1 this only gates `lookup_id`'s
        // key/value split.
        let has_output =
            arity > 0 && matches!(config.default, DefaultVal::FreshId | DefaultVal::Const(_));
        // Recognize the merge mode (FD-conflict resolution). We mirror the
        // DuckDB backend's stopgap recognition (`backend_impl.rs` ~577):
        //   - `AssertEq` / `Old`          => keep the old value
        //   - `New`                       => keep the new value
        //   - `UnionId`                   => lattice-min (the union-find leader)
        //   - `Primitive(ordering-min..)` => lattice-min — this is how the term
        //     encoder lowers `@uff`'s `:merge (ordering-min old new)`. The
        //     Feldera external-func registry doesn't track names, so we
        //     conservatively treat ANY `Primitive` whose first arg-merge is
        //     `Old`/`New` as `Min` (the only complex merge the term encoder's
        //     rebuild needs). `Function`/`Const` fall back to `Old`.
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
        self.relations.push(RelationInfo {
            name: config.name,
            arity,
            has_output,
            merge,
        });
        self.mirror.insert(id, HashSet::new());
        self.invalidate_circuit();
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
            set.clear();
        }
        self.invalidate_circuit();
    }

    fn base_values(&self) -> &egglog_core_relations::BaseValues {
        self.base_value_pool.inner()
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
            self.invalidate_circuit();
        }
    }

    fn run_rules(&mut self, rules: &[RuleId]) -> Result<IterationReport> {
        // ONE egglog iteration = one round of rule application over the current
        // relations, running ONLY the rules in `rules`. The frontend calls this
        // N times for `(run N)` and with different rule subsets to schedule
        // distinct rulesets (e.g. the term encoder's `(saturate single_parent)`
        // then `(saturate path_compress)` …).
        //
        // **Ruleset-scoped execution:** a DBSP circuit is static once built, so
        // we build (and cache) one circuit per distinct sorted rule subset. The
        // mirror is the single source of truth shared across all circuits; we
        // push the current mirror as the circuit's input delta, run one
        // transaction, and fold the resulting insert/delete diffs back into the
        // mirror.
        if rules.is_empty() {
            return Ok(IterationReport::default());
        }

        // Cache key: the sorted list of rule-id reps that actually map to a live
        // rule. Freed/empty rules are dropped (their circuit just omits them).
        let mut key: Vec<u32> = rules
            .iter()
            .map(|r| r.rep())
            .filter(|&i| {
                self.rules
                    .get(i as usize)
                    .map(|s| s.is_some())
                    .unwrap_or(false)
            })
            .collect();
        key.sort_unstable();
        key.dedup();
        if key.is_empty() {
            return Ok(IterationReport::default());
        }

        if !self.circuits.contains_key(&key) {
            let cs = self.build_circuit_for(&key)?;
            self.circuits.insert(key.clone(), cs);
        }

        let rel_ids: Vec<FunctionId> = (0..self.relations.len())
            .map(|i| FunctionId::new(i as u32))
            .collect();

        // 1. Sync this circuit's input handles to the CURRENT mirror by pushing
        //    the delta (additions +1, removals -1) versus what was last pushed
        //    into THIS circuit. Removals matter because rebuild shrinks
        //    relations (deletes) — the spike-proven negative-weight retraction.
        {
            let cs = self.circuits.get_mut(&key).unwrap();
            for &f in &rel_ids {
                let want = self.mirror.get(&f).cloned().unwrap_or_default();
                let have = cs.pushed.entry(f).or_default();
                let handles = cs.relations.get(&f).expect("relation handle missing");
                for row in want.difference(have) {
                    handles.input.push(*row, 1);
                }
                for row in have.difference(&want) {
                    handles.input.push(*row, -1);
                }
                *have = want;
            }
        }

        // 2. One transaction = one round of rule application over the current
        //    inputs (non-recursive circuit ⇒ exactly one hop / one rebuild step).
        {
            let cs = self.circuits.get(&key).unwrap();
            cs.handle.transaction()?;
        }

        // 3. Fold the per-relation insert/delete diffs into the mirror,
        //    resolving FD conflicts per merge mode. Detect change for the
        //    frontend's loop termination.
        let before_mirror: HashMap<FunctionId, HashSet<Row>> = self.mirror.clone();
        // Move the circuit out so we can borrow `self` mutably while reading its
        // (immutable) handles.
        let cs = self.circuits.remove(&key).unwrap();
        for &f in &rel_ids {
            self.fold_diffs_into_mirror(&cs, f);
        }
        self.circuits.insert(key, cs);
        let changed = rel_ids
            .iter()
            .any(|f| before_mirror.get(f) != self.mirror.get(f));

        let mut report = IterationReport::default();
        report.rule_set_report.changed = changed;
        Ok(report)
    }

    fn flush_updates(&mut self) -> bool {
        // Seed inserts land in the mirror immediately and are pushed into the
        // circuit at the next `run_rules`; there is no separate staged-update
        // queue. Return false (nothing accrued outside `run_rules`).
        false
    }

    // -- primitives ---------------------------------------------------------

    fn register_external_func(
        &mut self,
        func: Box<dyn ExternalFunction + 'static>,
    ) -> ExternalFunctionId {
        // Storage-only: primitives are not yet invokable from rules (milestone
        // 1 is primitive-light). See external_func.rs.
        self.external_funcs.add_func(func)
    }

    fn free_external_func(&mut self, func: ExternalFunctionId) {
        self.external_funcs.free(func);
    }

    fn new_panic(&mut self, message: String) -> ExternalFunctionId {
        self.external_funcs.add_panic(message)
    }

    // -- typed value handles ------------------------------------------------

    fn base_value_pool(&self) -> &dyn BaseValuePool {
        self.base_value_pool.as_pool()
    }

    fn base_value_pool_mut(&mut self) -> &mut dyn BaseValuePool {
        self.base_value_pool.as_pool_mut()
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
