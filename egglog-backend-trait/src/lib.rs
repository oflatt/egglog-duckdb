//! # egglog-backend-trait
//!
//! A backend-agnostic interface to an egglog egraph. This crate exposes the
//! [`Backend`] trait, along with companion traits ([`RuleBuilderOps`],
//! [`BaseValuePool`], [`ContainerPool`]), so that the frontend `EGraph` in
//! the top-level `egglog` crate can drive either the in-memory reference
//! backend (`egglog_bridge::EGraph`) or the DuckDB-backed backend
//! (`egglog_bridge_duckdb::EGraph`) through a single dyn-compatible API.
//!
//! ## Design principles
//!
//! - **Minimal IR changes.** This crate intentionally does NOT introduce a new
//!   neutral rule IR. [`RuleBuilderOps`] mirrors `egglog_bridge::RuleBuilder`
//!   one-for-one. The reference backend's `RuleBuilderOps` impl is a trivial
//!   passthrough; the DuckDB backend's impl accumulates calls into its
//!   internal data IR and submits them to the existing `compile_rule`
//!   pipeline on `build()`. Frontend code (`BackendRule` in
//!   `src/lib.rs::EGraph`) is unchanged in shape.
//! - **Basic id and config types live here.** As of Phase 2 Commit 3,
//!   `FunctionId`, `RuleId`, `ColumnTy`, `FunctionRow`, `FunctionConfig`,
//!   `MergeFn`, `DefaultVal`, `QueryEntry`, `Variable`, and `VariableId` are
//!   defined in this crate. `egglog-bridge` re-exports them so existing
//!   callers continue to work. `Value`, `BaseValueId`, `ContainerValueId`,
//!   `ExecutionState`, `ExternalFunction`, and `ExternalFunctionId` remain in
//!   `egglog-core-relations` (already a neutral crate) and are re-exported
//!   here for caller convenience.
//! - **`Backend` is `Send + Sync` and dyn-compatible.** Methods that need
//!   `T: BaseValue` or `C: ContainerValue` are factored onto
//!   [`BaseValuePool`] and [`ContainerPool`], which expose Any-based dynamic
//!   dispatch. A small set of generic helper functions (see the bottom of
//!   this module) reintroduce the per-`T` sugar on top of those dyn traits.
//! - **Cloning.** Backends must be cloneable via [`Backend::clone_boxed`].
//!   The reference backend already derives `Clone`; DuckDB will need a
//!   bespoke `clone_boxed` (e.g. database snapshot or replay buffer) when
//!   it implements this trait.
//!
//! ## What is intentionally NOT in this trait
//!
//! - `with_execution_state`. The four callers in `src/` are migrated to
//!   dedicated trait methods in a follow-up commit; this keeps `Backend`
//!   dyn-compatible and avoids leaking the lifetime semantics of
//!   `ExecutionState`.
//! - `TableAction` / `UnionAction`. Under the minimal-change posture these
//!   remain inherent methods on the bridge. The frontend keeps using them
//!   directly. Backends that don't support them (DuckDB) error at the
//!   primitive-registration call sites that touch them. The
//!   [`Backend::supports_inline_table_lookups`] capability flag gates this.
//! - A neutral rule IR. [`RuleBuilderOps`] is the seam instead.

use std::any::{Any, TypeId};

use anyhow::Result;

use egglog_numeric_id::define_id;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------
//
// Types that live in neutral lower-level crates and are re-exported here for
// caller convenience.

pub use egglog_core_relations::{
    BaseValue, BaseValueId, BaseValues, ContainerValue, ContainerValueId, DynamicInternTable,
    ExecutionState, ExternalFunction, ExternalFunctionId, Value,
};

pub use egglog_reports::{IterationReport, ReportLevel};

// ---------------------------------------------------------------------------
// Basic id types
// ---------------------------------------------------------------------------
//
// Each backend interprets these handles in its own internal map. The trait
// promises only "we return one to you, you give it back". The bridge's
// internal `TableId` / `core_relations::RuleId` are separate types not
// surfaced through the trait.

define_id!(pub RuleId, u32, "An egglog-style rule");
define_id!(pub FunctionId, u32, "An id representing an egglog function");

// ---------------------------------------------------------------------------
// ColumnTy
// ---------------------------------------------------------------------------

/// The type of a column (or `QueryEntry`): either an eq-sort / container id,
/// or a base value (with the [`BaseValueId`] identifying which base type).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum ColumnTy {
    Id,
    Base(BaseValueId),
}

// ---------------------------------------------------------------------------
// FunctionConfig / DefaultVal / MergeFn
// ---------------------------------------------------------------------------

/// Properties of a function added to an [`Backend`].
pub struct FunctionConfig {
    /// The function's schema. The last column in the schema is the return type.
    pub schema: Vec<ColumnTy>,
    /// The behavior of the function when lookups are made on keys not currently present.
    pub default: DefaultVal,
    /// How to resolve FD conflicts for the function.
    pub merge: MergeFn,
    /// The function's name
    pub name: String,
    /// Whether or not subsumption is enabled for this function.
    pub can_subsume: bool,
}

/// Proof-mode configuration for a UF-backed function (see
/// [`Backend::add_uf_function`]).
///
/// When present, the UF function is backed by a provenance-tracking
/// union-find: writes carry a per-edge proof column, and on each leader change
/// the callback composes a proof that the displaced leader equals the new
/// leader (reconstructed from the union-find's proof graph) and writes it into
/// the onchange relation's trailing proof column. The composition interns
/// `Trans` / `Sym` proof-constructor rows, so their backend [`FunctionId`]s are
/// required here.
#[derive(Copy, Clone)]
pub struct UfProofConfig {
    /// `Trans` proof constructor: `(Trans Proof Proof) -> Proof`.
    pub trans: FunctionId,
    /// `Sym` proof constructor: `(Sym Proof) -> Proof`.
    pub sym: FunctionId,
}

/// How defaults are computed for the given function.
#[derive(Copy, Clone)]
pub enum DefaultVal {
    /// Generate a fresh UF id.
    FreshId,
    /// Cause an egglog-level panic if a lookup fails.
    Fail,
    /// Insert a constant of some kind.
    Const(Value),
    /// Identity-on-miss ("lookup-or-self"): a failing lookup returns the
    /// (single) lookup key unchanged, and does NOT insert a row. Only valid
    /// for single-key functions whose key and return column share a type
    /// (e.g. the term-encoder's flat union-find index `@UF_Sf`). Used by the
    /// canonicalize-at-creation encoding to express a `find` against the
    /// frozen UF_old table.
    Identity,
}

/// How to resolve FD conflicts for a table.
#[derive(Clone)]
pub enum MergeFn {
    /// Panic if the old and new values don't match.
    AssertEq,
    /// Use congruence to resolve FD conflicts: union the two colliding output
    /// ids into the GLOBAL union-find (`$uf`). The standard congruence merge.
    UnionId,
    /// `--native-merge`: like [`MergeFn::UnionId`], but union the two colliding
    /// output ids directly into the named per-sort UF-backed function's
    /// union-find (its `@UF_Sf`) instead of the global `$uf`. Used by an FD-keyed
    /// constructor view `(children) -> eclass` so the FD-conflict congruence
    /// edge lands in exactly the union-find that owns the view's eclass column,
    /// whose leader changes the view's relational rebuild then re-canonicalizes.
    /// The `FunctionId` names that UF-backed function (the same association the
    /// uniform [`Backend::register_native_merge_view`] contract records). Only
    /// the native bridge resolves this variant; the dataflow/SQL backends route
    /// the union via their own host-pass `native_merge_uf` association.
    UnionIntoUf(FunctionId),
    /// `--native-merge` WITHOUT `--native-uf` (the RELATIONAL union-find path):
    /// like [`MergeFn::UnionIntoUf`], but the colliding output ids are written as
    /// a union EDGE into the per-sort RELATIONAL parent table `@UF_S`
    /// (`(S S) -> Unit :merge old`) — exactly the row the rule-encoded
    /// `union()` helper writes: `(set (@UF_S larger smaller) ())`, where
    /// `larger = max(cur, new)` and `smaller = min(cur, new)`. The merge returns
    /// `min(cur, new)`. The relational maintenance rulesets
    /// (singleparent / path_compress / uf_function_index) then propagate the edge
    /// into `@UF_Sf`, and the view's relational `@rebuild_rule*` re-canonicalizes —
    /// i.e. the rule-based rebuild absorbs the congruence cascade in batched
    /// seminaive delta passes (no native-UF onchange re-scan blowup).
    ///
    /// Fields:
    /// - `parent_table`: the relational `@UF_S` parent table to write the edge into.
    /// - `unit`: the interned `Unit` value for the parent table's value column.
    ///
    /// Native bridge only (relational native-merge). The dataflow/SQL backends
    /// keep their `--native-uf`-driven native-merge path.
    UnionIntoParentTable {
        parent_table: FunctionId,
        unit: Value,
    },
    /// `--native-merge` in PROOF mode: the proof-carrying counterpart of
    /// [`MergeFn::UnionIntoUf`]. Used by the `col0` (eclass) merge of a
    /// tuple-output FD-keyed constructor view `(children) -> (eclass, proof)`.
    ///
    /// On an FD conflict (same children, two `(eclass, proof)` output rows) it
    /// stages a proof-carrying congruence edge into the named proof-mode
    /// per-sort `@UF_Sf` (a `DisplacedTableWithProvenance`, 4-column writes
    /// `[lhs, rhs, proof, ts]`), then returns the surviving (min) eclass. The
    /// staged edge proof is composed as `Trans(larger_proof, Sym(smaller_proof))`
    /// over the two rows' proof columns, EXACTLY matching the orientation of the
    /// rule-encoded `@congruence_rule` it replaces (which writes
    /// `(set (@UF_Sf larger smaller) (Trans larger_pf (Sym smaller_pf)))`).
    ///
    /// Fields:
    /// - `uf`: the proof-mode `@UF_Sf` UF-backed function to union into.
    /// - `trans` / `sym`: the `Trans` / `Sym` proof constructors used to compose
    ///   the edge proof.
    /// - `eclass_col`: the value-column index of the eclass (this column's own
    ///   index; the merge resolving `col0`).
    /// - `proof_col`: the value-column index of the per-row term proof
    ///   (`eclass = f(children)`).
    ///
    /// Only the native bridge resolves this variant; the dataflow/SQL backends
    /// have no proof-mode native-UF and treat it as unsupported (the proof half
    /// of native-merge is bridge-only).
    UnionIntoUfWithProof {
        uf: FunctionId,
        trans: FunctionId,
        sym: FunctionId,
        eclass_col: usize,
        proof_col: usize,
    },
    /// `--native-merge` in PROOF mode: the `col1` (proof) merge companion of
    /// [`MergeFn::UnionIntoUfWithProof`]. Returns the term proof
    /// (`eclass = f(children)`) of the SURVIVING (min) eclass — i.e. the proof of
    /// whichever row has the smaller eclass value, matching the eclass kept by
    /// the UF leader tie-break (`min`). This expresses the select-by-min that the
    /// scalar `Old`/`New` leaves cannot.
    ///
    /// `eclass_col` / `proof_col` are the value-column indices of the eclass and
    /// the proof, respectively. Bridge-only, like its companion.
    EclassMinProof { eclass_col: usize, proof_col: usize },
    /// The output of a merge is determined by applying the given ExternalFunction to the result
    /// of the argument merge functions.
    Primitive(ExternalFunctionId, Vec<MergeFn>),
    /// The output of a merge is determined by looking up the value for the given function and the
    /// given arguments in the egraph.
    Function(FunctionId, Vec<MergeFn>),
    /// Always return the old value for *this* value column (the column this merge resolves).
    Old,
    /// Always return the new value for *this* value column.
    New,
    /// The old value of value column `i`. Used by tuple-output (multi-value) merges, where a
    /// column's merge may reference any output column of the OLD row. For a single-output
    /// function, `OldCol(0)` is equivalent to [`MergeFn::Old`].
    OldCol(usize),
    /// The new value of value column `i`. The multi-value counterpart of [`MergeFn::New`].
    NewCol(usize),
    /// Always overwrite the new value for the given function with a constant. This is more useful
    /// as a "base case" in a more complicated merge function (e.g. one that clamps a value between
    /// 1 and 100) than it is as a standalone merge function.
    Const(Value),
    /// A merge for a tuple-output (multi-value) function: one [`MergeFn`] per value (return)
    /// column. Each inner `MergeFn` produces that column's merged value and may reference any
    /// output column via [`MergeFn::OldCol`] / [`MergeFn::NewCol`]. Its length determines the
    /// number of value columns of the function (`n_vals`); `n_keys = schema.len() - n_vals`.
    ///
    /// `Columns` must appear only at the top level: a nested `Columns` (e.g. as an argument to a
    /// [`MergeFn::Primitive`] or inside another `Columns`) is an error. A single-output function
    /// uses a bare scalar merge (`Old`/`New`/`UnionId`/…), which is exactly a `Columns` of
    /// length 1.
    Columns(Vec<MergeFn>),
    /// Insert a row into the given function's table (the `args` evaluate to the
    /// full row), respecting that table's own merge. Returns the OLD value of the
    /// resolving column (its return is meant to be discarded inside a
    /// [`MergeFn::Seq`]). Models a `(set (f ...) v)` action inside a merge;
    /// declares `f`'s table as a write-dependency so the side write is safe during
    /// batched merges. Ported from PR #933 (`:merge`-multiple-actions). Bridge-only
    /// — the dataflow/SQL backends treat it as unsupported.
    TableInsert(FunctionId, Vec<MergeFn>),
    /// Evaluate each merge function in order (for their effects) and return the
    /// value of the last one. Models an action-block merge: the leading entries
    /// are effects (e.g. [`MergeFn::TableInsert`] / [`MergeFn::Construct`]) and the
    /// last is the value. Used to lower a term-building custom `:merge` body into a
    /// single native merge. Ported from PR #933. Bridge-only.
    Seq(Vec<MergeFn>),
    /// Mint a pair-valued constructor inside a merge and return its output e-class.
    /// `args` evaluate to the key (children); the first value column (the output) is
    /// minted (the table's `FreshId` default) and the remaining value columns are
    /// written from `value_args` (e.g. the term proof). Used to express a nested
    /// constructor application inside a term-building custom `:merge` body. Ported
    /// from PR #933 (FD proof encoding). Bridge-only.
    Construct(FunctionId, Vec<MergeFn>, Vec<MergeFn>),
    /// Conditional merge: evaluate `a` and `b`; if their values are EQUAL run and
    /// return `then`, otherwise run and return `els`. Used to guard a term-building
    /// custom `:merge` body (a [`MergeFn::Seq`]) so it only mints helper terms when
    /// the function's old/new values genuinely differ — mirroring the rule-encoded
    /// `@merge_rule`'s `(!= old new)` body guard. When `a == b` the rule-encoded
    /// merge rule never fires, so no helper term is created; this variant reproduces
    /// that exactly (return the already-equal value, no side effects). Bridge-only.
    IfEq {
        a: Box<MergeFn>,
        b: Box<MergeFn>,
        then: Box<MergeFn>,
        els: Box<MergeFn>,
    },
}

// ---------------------------------------------------------------------------
// FunctionRow
// ---------------------------------------------------------------------------

/// A struct representing the content of a row in a function table
#[derive(Clone, Debug)]
pub struct FunctionRow<'a> {
    pub vals: &'a [Value],
    pub subsumed: bool,
}

// ---------------------------------------------------------------------------
// Variable / VariableId / QueryEntry
// ---------------------------------------------------------------------------

define_id!(pub VariableId, u32, "A variable in an egglog query");

/// A variable in a rule body / RHS, with an optional display name for
/// debugging.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Variable {
    pub id: VariableId,
    pub name: Option<Box<str>>,
}

impl Variable {
    /// Construct an unnamed variable from a [`VariableId`].
    pub fn from_id(id: VariableId) -> Self {
        Variable { id, name: None }
    }
}

/// A reference in a rule body or RHS: either a variable or a typed constant.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum QueryEntry {
    Var(Variable),
    Const {
        val: Value,
        // Constants can have a type plumbed through, particularly if they
        // correspond to a base value constant in egglog.
        ty: ColumnTy,
    },
}

impl From<Variable> for QueryEntry {
    fn from(var: Variable) -> Self {
        QueryEntry::Var(var)
    }
}

// ---------------------------------------------------------------------------
// The `Backend` trait
// ---------------------------------------------------------------------------

/// A backend that drives an egglog egraph.
///
/// Implementations: `egglog_bridge::EGraph` (reference, in-memory) and
/// `egglog_bridge_duckdb::EGraph` (DuckDB-backed). The frontend `EGraph` in
/// the `egglog` crate holds a `Box<dyn Backend>` and dispatches all state
/// access through this trait.
///
/// ## Method correspondence
///
/// Each method's doc comment names the inherent method on
/// `egglog_bridge::EGraph` that the bridge's `impl Backend` wraps. The
/// reference backend's impl is meant to be one line per method
/// (`fn foo(&self, ...) -> X { self.foo(...) }`). The DuckDB backend's
/// translations are documented in `docs/backend_trait_design.md`.
///
/// ## Dyn-compatibility
///
/// All methods are object-safe. Methods that would otherwise be generic over
/// `T: BaseValue` or `C: ContainerValue` live on the
/// [`BaseValuePool`] / [`ContainerPool`] sub-traits, which use Any-based
/// dynamic dispatch internally.
///
/// ## Bridge-only escape hatch via [`Backend::as_any`]
///
/// A handful of frontend constructs — most notably `TableAction::new` /
/// `UnionAction::new` used inside `rust_rule` and `query` primitive
/// `apply()` bodies in `src/prelude.rs` — depend on bridge-specific state
/// (`egglog_bridge::EGraph`) that does not fit cleanly behind the trait
/// surface. The Phase 1 design (see `docs/backend_trait_design.md`,
/// "How `TableAction` and `UnionAction` are handled") explicitly keeps
/// these types as bridge inherent methods rather than lifting them onto
/// the trait.
///
/// To let callers reach the bridge-specific state when the backend field
/// is `Box<dyn Backend>`, the trait exposes an `Any`-based downcast pair:
///
/// ```ignore
/// // Inside a primitive that needs TableAction:
/// let bridge = self
///     .backend
///     .as_any()
///     .downcast_ref::<egglog_bridge::EGraph>()
///     .expect("table actions are bridge-only");
/// let action = egglog_bridge::TableAction::new(bridge, func_id);
/// ```
///
/// The DuckDB backend's `as_any` returns a different concrete type
/// (`egglog_bridge_duckdb::EGraph`); the downcast to
/// `&egglog_bridge::EGraph` therefore fails on DuckDB. This is correct: the
/// primitives that go through this path (`rust_rule` / `query` table-lookup
/// helpers) are gated by [`Backend::supports_inline_table_lookups`] and
/// are documented as unsupported on DuckDB in v1.
///
/// Using `&dyn Any` keeps this trait crate from having to name
/// `egglog_bridge::EGraph` directly, which would re-introduce the
/// `egglog-bridge ↔ egglog-backend-trait` dependency cycle that Phase 2
/// Commit 3 deliberately broke.
pub trait Backend: Send + Sync {
    // -- table lifecycle ----------------------------------------------------

    /// Register a function/relation/constructor and return its handle.
    ///
    /// Wraps `egglog_bridge::EGraph::add_table`.
    fn add_table(&mut self, config: FunctionConfig) -> FunctionId;

    /// Peek the [`FunctionId`] the next [`Backend::add_table`] call will return,
    /// WITHOUT registering anything. Used for "knot-tying" a self-referential
    /// `:merge`: the relational native-merge union-find function `@UF_Sf` needs
    /// its OWN id while building its merge (a `MergeFn::TableInsert` into itself
    /// for the recursive parent-union). The caller peeks the id immediately before
    /// `add_table`.
    ///
    /// Bridge-only (this whole path is `--native-merge` on the native bridge). The
    /// default panics: the dataflow/SQL backends never declare a self-referential
    /// merge, matching the inert handling of the other bridge-only merge variants
    /// (`Seq` / `TableInsert` / `IfEq`).
    fn peek_next_function_id(&self) -> FunctionId {
        panic!("peek_next_function_id is only supported on the native bridge backend")
    }

    /// `--nativerb` (native bridge, term-encoding native-UF, non-proof): register
    /// a `@<F>View` view function (`view_func`) to be re-canonicalized by the
    /// engine's native table rebuild against the per-sort `@UF_Sf` UF-backed
    /// function (`uf_func`), replacing the encoding-level `@rebuild_rule*` rules.
    /// The frontend calls this once per view function after all functions are
    /// registered. Default no-op: only the native bridge supports a native table
    /// rebuild; the dataflow/SQL backends drive their own host-pass rebuild.
    fn register_nativerb_view(&mut self, _uf_func: FunctionId, _view_func: FunctionId) {}

    /// `--native-merge` (term-encoding native-UF, non-proof): associate an
    /// FD-keyed constructor view (`view_func`, keyed `(children) -> eclass`) with
    /// the union-find (`uf_func`) that owns its eclass (OUTPUT) column. The view's
    /// `MergeFn::UnionId` merge then injects a union into this UF on an FD conflict
    /// (same children, two eclasses) — congruence done INLINE at insert time
    /// instead of by a self-join rule. The frontend calls this before each run for
    /// each not-yet-registered native-merge view. Default no-op: only backends
    /// that inject the union on FD-conflict (currently FlowLog) implement it.
    fn register_native_merge_view(&mut self, _uf_func: FunctionId, _view_func: FunctionId) {}

    /// Register a union-find-backed function (see upstream PR #782).
    ///
    /// The function has schema `(S) S` for an EqSort `S` and records leader
    /// changes in the underlying union-find. When `onchange` is `Some(rel)`,
    /// each leader change is recorded into the relation `rel` (5 columns:
    /// `write_lhs write_rhs lhs_leader rhs_leader new_leader`, looked-up-or-
    /// inserted to mint a fresh id, since relations are constructor-backed).
    /// Returns the function's handle and the [`ExternalFunctionId`] of the
    /// canonicalizer primitive (find-or-self against the union-find).
    ///
    /// When `proof` is `Some(_)`, the function is backed by a
    /// provenance-tracking union-find: the onchange relation gains a trailing
    /// proof column (so it is `(S S S S S Proof)`), and the leader-change
    /// callback composes that proof from the union-find's proof graph (see
    /// [`UfProofConfig`]).
    ///
    /// Wraps `egglog_bridge::EGraph::add_uf_function`. Backends that do not
    /// support UF-backed functions (DuckDB, Feldera, FlowLog) return an error.
    fn add_uf_function(
        &mut self,
        name: String,
        onchange: Option<FunctionId>,
        proof: Option<UfProofConfig>,
    ) -> Result<(FunctionId, ExternalFunctionId)>;

    /// Number of rows currently in the given function's table.
    ///
    /// Wraps `egglog_bridge::EGraph::table_size`.
    fn table_size(&self, table: FunctionId) -> usize;

    /// Approximate size; backends may return a fast estimate.
    ///
    /// Wraps `egglog_bridge::EGraph::approx_table_size`.
    fn approx_table_size(&self, table: FunctionId) -> usize;

    // -- iteration ----------------------------------------------------------

    /// Iterate over every row in `table`, calling `f` on each.
    ///
    /// Wraps `egglog_bridge::EGraph::for_each`.
    ///
    /// The closure's `FunctionRow` is borrowed from a transient per-call
    /// buffer (a `TaggedRowBuffer` in the bridge; a row cursor in DuckDB).
    /// The HRTB lifetime `for<'r>` reflects that the row reference is
    /// scoped to the closure invocation, not the outer `&self` borrow.
    fn for_each(&self, table: FunctionId, f: &mut dyn for<'r> FnMut(FunctionRow<'r>)) {
        // The default implementation cannot be written here without a wrapper
        // because the closure types differ. Implementations should override
        // this to avoid the boolean threading overhead; the no-default
        // alternative would be to require both methods. We mark this as
        // mandatory below to keep impls explicit.
        let _ = (table, f);
        unimplemented!(
            "Backend impls must override for_each; the default exists only \
             to satisfy dyn-compatibility lint chains."
        )
    }

    /// Iterate over rows in `table`, stopping early when `f` returns `false`.
    ///
    /// Wraps `egglog_bridge::EGraph::for_each_while`.
    ///
    /// See [`Backend::for_each`] for why the closure uses an HRTB.
    fn for_each_while(&self, table: FunctionId, f: &mut dyn for<'r> FnMut(FunctionRow<'r>) -> bool);

    // -- direct access ------------------------------------------------------

    /// Look up the output value associated with `key` in `func`.
    ///
    /// Wraps `egglog_bridge::EGraph::lookup_id`.
    fn lookup_id(&self, func: FunctionId, key: &[Value]) -> Option<Value>;

    /// Bulk-insert one or many rows across one or many functions.
    ///
    /// Wraps `egglog_bridge::EGraph::add_values`. (The boxed iterator is
    /// used to keep this method dyn-compatible.)
    fn add_values(&mut self, values: Box<dyn Iterator<Item = (FunctionId, Vec<Value>)> + '_>);

    /// Add a term-shaped row: stage `(inputs ... fresh_id)` and return the
    /// canonical id of the freshly allocated output.
    ///
    /// Wraps `egglog_bridge::EGraph::add_term`. On DuckDB this maps to an
    /// `INSERT ... RETURNING` against the function's table.
    fn add_term(&mut self, func: FunctionId, inputs: &[Value]) -> Value;

    /// Stage a batch of complete rows into `table`.
    ///
    /// Each entry in `rows` is the full row for the table (key columns
    /// followed by the value column, where applicable). The rows are staged
    /// against a single internal execution state; callers should invoke
    /// [`Backend::flush_updates`] afterward to merge them into the database.
    ///
    /// Unlike [`Backend::add_values`], this is scoped to a single table and
    /// does not call `flush_updates` itself, which makes it suitable for
    /// scenarios that interleave many table-level insertions before a final
    /// flush (e.g. parsing an input file, scheduler match dispatch).
    ///
    /// The bridge's impl wraps `with_execution_state` around a loop of
    /// `TableAction::insert` calls.
    fn insert_rows(&mut self, table: FunctionId, rows: &[Vec<Value>]);

    /// Stage a batch of constructor-style lookup-or-insert rows.
    ///
    /// Each entry in `rows` is a key for the constructor `table` (without the
    /// output column); the bridge calls `TableAction::lookup` per row, which
    /// either returns the existing output or allocates a fresh id. As with
    /// [`Backend::insert_rows`], callers should invoke
    /// [`Backend::flush_updates`] afterward.
    ///
    /// The bridge's impl wraps `with_execution_state` around a loop of
    /// `TableAction::lookup` calls.
    fn lookup_constructor_rows(&mut self, table: FunctionId, rows: &[Vec<Value>]);

    /// Get the canonical representative of `val` according to `ty`.
    ///
    /// For `ColumnTy::Id` this is the union-find canonicalization. For
    /// `ColumnTy::Base(_)` it returns `val` unchanged.
    ///
    /// Wraps `egglog_bridge::EGraph::get_canon_repr`.
    fn get_canon_repr(&self, val: Value, ty: ColumnTy) -> Value;

    /// Allocate a fresh egraph id (counter increment, returned as a `Value`).
    ///
    /// Wraps `egglog_bridge::EGraph::fresh_id`.
    fn fresh_id(&mut self) -> Value;

    /// Remove every row from the given function's table.
    ///
    /// Wraps `egglog_bridge::EGraph::clear_table`.
    fn clear_table(&mut self, func: FunctionId);

    /// Access the backend's [`BaseValues`] registry directly.
    ///
    /// Wraps `egglog_bridge::EGraph::base_values`. This returns the concrete
    /// `egglog_core_relations::BaseValues` (re-exported here) so callers can
    /// use the generic-over-`T` `get::<T>` / `unwrap::<T>` sugar that the
    /// dyn-friendly [`BaseValuePool`] does not expose.
    fn base_values(&self) -> &BaseValues;

    /// Run `f` against a fresh execution state, dyn-compatible form.
    ///
    /// Wraps `egglog_bridge::EGraph::with_execution_state`. The generic,
    /// result-returning ergonomic wrapper is provided as the inherent
    /// [`Backend::with_execution_state`] method on `dyn Backend`; this
    /// erased form exists so the trait stays object-safe.
    fn with_execution_state_dyn(&self, f: &mut dyn FnMut(&mut ExecutionState<'_>));

    /// Type-erased access to the backend's action registry handle.
    ///
    /// Wraps `egglog_bridge::EGraph::action_registry`, which returns a
    /// `&Arc<RwLock<ActionRegistry>>`. `ActionRegistry` is defined in the
    /// `egglog-bridge` crate, which depends on this crate; naming it here
    /// would re-introduce the dependency cycle that Phase 2 deliberately
    /// broke. So the object-safe form erases the handle to `&dyn Any`, and
    /// the generic inherent [`Backend::action_registry`] downcasts it back
    /// to the caller-inferred concrete registry type.
    ///
    /// The `&dyn Any` is expected to be an `Arc<RwLock<ActionRegistry>>`.
    /// Backends without an action registry (e.g. duckdb) panic.
    fn action_registry_any(&self) -> &(dyn Any + Send + Sync);

    // -- rule management ----------------------------------------------------

    /// Begin building a new rule. Returns a builder whose lifetime is tied to
    /// `&mut self`. Callers populate the builder via [`RuleBuilderOps`] and
    /// finalize with [`RuleBuilderOps::build`].
    ///
    /// Wraps `egglog_bridge::EGraph::new_rule`.
    fn new_rule<'a>(&'a mut self, desc: &str, seminaive: bool) -> Box<dyn RuleBuilderOps + 'a>;

    /// Drop a registered rule. The handle becomes invalid.
    ///
    /// Wraps `egglog_bridge::EGraph::free_rule`.
    fn free_rule(&mut self, id: RuleId);

    /// Run one iteration of the given rule set. Returns timing and change
    /// counts; the database may have been modified even if `changed` is
    /// false (e.g. timestamps advanced).
    ///
    /// Wraps `egglog_bridge::EGraph::run_rules`.
    fn run_rules(&mut self, rules: &[RuleId]) -> Result<IterationReport>;

    /// Drain staged inserts and run a rebuild pass if the UF changed.
    /// Returns whether the database changed.
    ///
    /// Wraps `egglog_bridge::EGraph::flush_updates`.
    fn flush_updates(&mut self) -> bool;

    // -- primitives ---------------------------------------------------------

    /// Register a user-defined primitive (`ExternalFunction`).
    ///
    /// Wraps `egglog_bridge::EGraph::register_external_func`.
    ///
    /// On DuckDB, primitives that synchronously call back into table state
    /// (e.g. `TableAction::lookup` in the apply body) are not supported in
    /// v1. See [`Backend::supports_inline_table_lookups`].
    fn register_external_func(
        &mut self,
        func: Box<dyn ExternalFunction + 'static>,
    ) -> ExternalFunctionId;

    /// Drop a user-defined primitive.
    ///
    /// Wraps `egglog_bridge::EGraph::free_external_func`.
    fn free_external_func(&mut self, func: ExternalFunctionId);

    /// Register a deferred-panic primitive that stops rule execution when
    /// invoked. The returned id can be used as a fallback target in
    /// `lookup_with_fallback` and similar.
    ///
    /// Wraps `egglog_bridge::EGraph::new_panic`.
    fn new_panic(&mut self, message: String) -> ExternalFunctionId;

    /// Evaluate a registered primitive on the given (already-interned) argument
    /// `Value`s, returning its result, or `None` if the primitive fails
    /// (e.g. `!=` of equal arguments, or a guard that does not hold).
    ///
    /// This is the **backend-agnostic primitive entry point**. It lets a
    /// backend evaluate a primitive *inline* (e.g. inside a DBSP map/filter
    /// operator, or a host-side join loop) without owning or borrowing a
    /// `core_relations::ExecutionState`, which is the type
    /// [`ExternalFunction::invoke`] requires.
    ///
    /// ## Contract
    ///
    /// - `args` are interned `Value`s in the backend's value space (same
    ///   representation `for_each` / `lookup_id` use).
    /// - The result, if any, is an interned `Value` in that same space.
    /// - For backends that evaluate primitives against an execution state
    ///   (reference, feldera), this is `with_execution_state(|st|
    ///   st.call_external_func(id, args))`.
    /// - Pure primitives (comparisons, arithmetic, string ops) do not touch
    ///   the database, so they evaluate identically regardless of which
    ///   backend hosts them. Primitives that reenter table state are only
    ///   well-defined on backends whose
    ///   [`Backend::supports_inline_table_lookups`] is `true`.
    fn eval_prim(&self, id: ExternalFunctionId, args: &[Value]) -> Option<Value>;

    // -- typed value handles (sub-traits) -----------------------------------

    /// Access the backend's [`BaseValuePool`] for typed base-value queries.
    ///
    /// Wraps `egglog_bridge::EGraph::base_values` (returns `&BaseValues`,
    /// which implements this sub-trait either directly or via a thin shim
    /// the bridge provides).
    fn base_value_pool(&self) -> &dyn BaseValuePool;

    /// Mutable access to the [`BaseValuePool`]. Used to register new
    /// `BaseValue` types.
    ///
    /// Wraps `egglog_bridge::EGraph::base_values_mut`.
    fn base_value_pool_mut(&mut self) -> &mut dyn BaseValuePool;

    /// Access the backend's [`ContainerPool`].
    ///
    /// Wraps `egglog_bridge::EGraph::container_values`.
    ///
    /// On DuckDB this returns an empty stub (see [`ContainerPool`]'s docs).
    /// All container-using egglog programs are gated out of DuckDB by the
    /// existing `program_supports_proofs` check, so the stub is never
    /// reached in practice.
    fn container_pool(&self) -> &dyn ContainerPool;

    /// Mutable access to the [`ContainerPool`].
    ///
    /// Wraps `egglog_bridge::EGraph::container_values_mut`.
    fn container_pool_mut(&mut self) -> &mut dyn ContainerPool;

    /// Build a [`QueryEntry`] constant for a base value of dynamic type.
    ///
    /// `value` is the interned `Value`; `ty` is the base-value-type id
    /// returned by `BaseValuePool::register_type` /
    /// `BaseValuePool::get_ty_by_type_id`.
    ///
    /// Wraps `egglog_bridge::EGraph::base_value_constant`. (The bridge's
    /// inherent method is generic over `T: BaseValue`; the dyn-friendly form
    /// takes the already-interned `Value` and the `BaseValueId`. A generic
    /// helper that wraps this is provided below
    /// (`base_value_constant<T>`).)
    fn base_value_constant_dyn(&self, value: Value, ty: BaseValueId) -> QueryEntry;

    // -- capability flags ---------------------------------------------------

    /// Whether this backend's user-defined primitives can synchronously
    /// call back into table state during their `apply()` body.
    ///
    /// Reference backend: `true`. DuckDB: `false` (primitives run inside
    /// DuckDB VScalar UDFs and cannot reenter the database).
    ///
    /// Callers that need `rust_rule` / `query` should gate on this flag.
    fn supports_inline_table_lookups(&self) -> bool;

    /// Whether this backend supports the `subsume` action and the
    /// `is_subsumed` filter on table atoms.
    ///
    /// Reference backend: `true`. DuckDB: `false` in v1 — the trait
    /// surfaces `subsume` on [`RuleBuilderOps`] but the DuckDB impl returns
    /// an error when called.
    fn supports_subsumption(&self) -> bool;

    /// Whether this backend supports `MergeFn::Function` and
    /// `MergeFn::Primitive`.
    ///
    /// Reference backend: `true`. DuckDB: `false` in v1.
    fn supports_complex_merge(&self) -> bool;

    /// Whether this backend can run a pure VALUE-FOLD custom `:merge` NATIVELY in
    /// its FD-conflict resolver — a fold of primitives / literals over `old`/`new`
    /// that yields a plain value (no e-node creation, no cross-table writes), e.g.
    /// `(from-string (to-string (* new old)))`. Gates
    /// `EncodeContext::native_value_fold_merge`: when `true` such a merge is lowered
    /// to a `MergeFn` tree and run host-side; when `false` it is rule-encoded as
    /// `@merge_rule` / `@merge_cleanup`.
    ///
    /// Defaults to [`Backend::supports_complex_merge`] so the bridge/duckdb/feldera
    /// behavior is unchanged. The flowlog backend overrides this to `true` (it
    /// evaluates the value-fold tree natively) while keeping `supports_complex_merge`
    /// / [`Backend::supports_term_build_merge`] `false`.
    fn supports_value_fold_merge(&self) -> bool {
        self.supports_complex_merge()
    }

    /// Whether this backend can run a TERM-BUILDING custom `:merge` NATIVELY — a
    /// merge body that mints e-nodes via constructor calls (a `MergeFn::Seq` of
    /// `Construct` / `TableInsert` / union variants), e.g.
    /// `(C2 (C1 old new) (C2 old new))`. Gates
    /// `EncodeContext::native_term_build_merge`.
    ///
    /// Defaults to [`Backend::supports_complex_merge`] so the bridge/duckdb/feldera
    /// behavior is unchanged. (The dataflow backends keep this `false`; term-build
    /// native merge is a later increment.)
    fn supports_term_build_merge(&self) -> bool {
        self.supports_complex_merge()
    }

    /// Whether this backend supports `Vec` / `Set` / `Map` / `MultiSet`
    /// container sorts.
    ///
    /// Reference backend: `true`. DuckDB: `false` (container sorts are
    /// excluded from DuckDB test combos by the existing `supports_proofs`
    /// gate; see `docs/backend_trait_inventory.md` Section 6.3).
    fn supports_containers(&self) -> bool;

    /// Whether this backend exposes the in-memory `ActionRegistry` /
    /// `ExecutionState` that registry-backed primitives
    /// (`ReadPrim`/`WritePrim`/`FullPrim`) dispatch through. The reference
    /// (bridge) backend wraps each such primitive in a `RegistryPrimWrapper`
    /// that clones `action_registry()` at registration time; the dataflow /
    /// SQL backends (`action_registry_any()` = `unimplemented!()`) cannot, so
    /// the egglog frontend registers a registry-free placeholder/snapshot
    /// wrapper instead (see `register_registry_primitive` in
    /// `typechecking.rs`).
    ///
    /// Reference backend: `true`. DuckDB / FlowLog / Feldera: `false`.
    fn supports_action_registry(&self) -> bool;

    // -- diagnostics --------------------------------------------------------

    /// Set the verbosity of the per-rule-iteration timing report.
    ///
    /// Wraps `egglog_bridge::EGraph::set_report_level`.
    fn set_report_level(&mut self, level: ReportLevel);

    /// Dump the database state to the `log::info!` channel (debug only).
    ///
    /// Wraps `egglog_bridge::EGraph::dump_debug_info`.
    fn dump_debug_info(&self);

    // -- cloning ------------------------------------------------------------

    /// Produce a deep clone of this backend.
    ///
    /// The frontend uses this for push/pop snapshot support. The reference
    /// backend derives `Clone`, so its impl is a one-liner
    /// (`Box::new(self.clone())`). The DuckDB backend will need a bespoke
    /// implementation (database snapshot / replay buffer); see
    /// `docs/backend_trait_design.md` for the chosen strategy.
    fn clone_boxed(&self) -> Box<dyn Backend>;

    // -- bridge-only escape hatch ------------------------------------------

    /// Return `&self` as `&dyn Any`, enabling callers to downcast to the
    /// concrete backend type (e.g. `&egglog_bridge::EGraph`).
    ///
    /// This is the supported path for invoking bridge-only inherent
    /// methods such as `TableAction::new` / `UnionAction::new` from
    /// frontend code whose `backend` field is a `Box<dyn Backend>`. See the
    /// trait-level documentation for the rationale (Any-based downcast
    /// keeps the trait crate free of `egglog-bridge` dependencies, avoiding
    /// a dependency cycle).
    ///
    /// Implementations are expected to be one-liners: `fn as_any(&self) ->
    /// &dyn Any { self }`. The trait can't supply a default body because
    /// `dyn Backend` is not `Sized` and `&dyn Any` requires
    /// `Self: 'static + Sized` — but every concrete backend type satisfies
    /// both, so each impl trivially provides the body.
    fn as_any(&self) -> &dyn Any;

    /// Mutable counterpart of [`Backend::as_any`]. Implementations are
    /// expected to return `self`.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl Clone for Box<dyn Backend> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

impl dyn Backend {
    /// Run `f` against a fresh execution state and return its result.
    ///
    /// Ergonomic, generic-over-`R` wrapper around the object-safe
    /// [`Backend::with_execution_state_dyn`]. The result is threaded back
    /// out through a captured slot rather than `Box<dyn Any>`, so `R` does
    /// not need to be `'static`. Mirrors
    /// `egglog_bridge::EGraph::with_execution_state`.
    pub fn with_execution_state<R>(&self, f: impl FnOnce(&mut ExecutionState<'_>) -> R) -> R {
        let mut f = Some(f);
        let mut out: Option<R> = None;
        self.with_execution_state_dyn(&mut |es| {
            let f = f.take().expect("with_execution_state closure called once");
            out = Some(f(es));
        });
        out.expect("with_execution_state_dyn must invoke its closure exactly once")
    }

    /// A handle to the live action registry for this backend.
    ///
    /// Generic ergonomic wrapper around [`Backend::action_registry_any`].
    /// `R` is the concrete registry type (`egglog_bridge::ActionRegistry`),
    /// inferred at the call site so this crate never has to name it (which
    /// would re-introduce the `egglog-bridge -> egglog-backend-trait`
    /// dependency cycle). Mirrors `egglog_bridge::EGraph::action_registry`.
    ///
    /// # Panics
    ///
    /// Panics if the backend has no action registry (e.g. duckdb) or if the
    /// erased handle is not an `Arc<RwLock<R>>`.
    pub fn action_registry<R: Any + Send + Sync>(&self) -> &std::sync::Arc<std::sync::RwLock<R>> {
        self.action_registry_any()
            .downcast_ref::<std::sync::Arc<std::sync::RwLock<R>>>()
            .expect("action_registry: backend has no action registry of the requested type")
    }
}

// ---------------------------------------------------------------------------
// `RuleBuilderOps` — mirrors `egglog_bridge::RuleBuilder` one-for-one
// ---------------------------------------------------------------------------

/// A lazily-computed failure message for RHS lookups / external calls.
///
/// The message is only built if the lookup or call actually fails at
/// runtime. Deferring it matters because the usual message form is
/// `format!("{span}: ...")`, and formatting a `Span` calls
/// `SrcFile::get_location`, which scans the source-file prefix. Building it
/// eagerly for every action in every rule is `O(rules * file)` — quadratic
/// when many generated rules (term/proof encoding) are compiled against a
/// large source file. Boxed so the trait stays dyn-compatible.
pub type PanicMsg = Box<dyn FnOnce() -> String + Send>;

/// Operations on an in-progress rule.
///
/// This trait mirrors the public methods on
/// `egglog_bridge::RuleBuilder`. The bridge's impl is a trivial newtype
/// passthrough; the DuckDB impl accumulates calls into its internal
/// `duck::Rule` data IR and submits to `compile_rule` on
/// [`RuleBuilderOps::build`].
///
/// ## Variable creation
///
/// `new_var` and `new_var_named` allocate new variables. The returned
/// `QueryEntry` can be passed back to body atoms / actions. Each `QueryEntry`
/// carries its [`ColumnTy`] for runtime arity / type checking.
///
/// ## Unsupported on DuckDB
///
/// - [`RuleBuilderOps::subsume`][]: error.
/// - Complex `MergeFn::Function` / `MergeFn::Primitive` referenced via
///   `query_prim`: error at `build()` time when the rule references such a
///   merge.
pub trait RuleBuilderOps {
    /// Bind a new variable of the given type.
    ///
    /// Wraps `RuleBuilder::new_var`.
    fn new_var(&mut self, ty: ColumnTy) -> QueryEntry;

    /// Bind a new variable of the given type with a display name (eases
    /// debugging).
    ///
    /// Wraps `RuleBuilder::new_var_named`.
    fn new_var_named(&mut self, ty: ColumnTy, name: &str) -> QueryEntry;

    /// Add a table body atom. The final entry is the function's return
    /// value. When `is_subsumed` is `Some`, the atom is constrained to rows
    /// with that subsumption bit.
    ///
    /// Wraps `RuleBuilder::query_table`.
    fn query_table(
        &mut self,
        func: FunctionId,
        entries: &[QueryEntry],
        is_subsumed: Option<bool>,
    ) -> Result<()>;

    /// Add a primitive body atom. The final entry is the return value.
    ///
    /// Wraps `RuleBuilder::query_prim`.
    fn query_prim(
        &mut self,
        func: ExternalFunctionId,
        entries: &[QueryEntry],
        ret_ty: ColumnTy,
    ) -> Result<()>;

    /// Call an external function in the RHS, panicking with `panic_msg` on
    /// failure. Returns the result variable.
    ///
    /// Wraps `RuleBuilder::call_external_func`. `panic_msg` is a [`PanicMsg`]
    /// closure, evaluated only if the call actually fails.
    fn call_external_func(
        &mut self,
        func: ExternalFunctionId,
        args: &[QueryEntry],
        ret_ty: ColumnTy,
        panic_msg: PanicMsg,
    ) -> QueryEntry;

    /// RHS: look up the value of `func(entries)`, with the function's
    /// configured default behavior on miss.
    ///
    /// Wraps `RuleBuilder::lookup`. `panic_msg` is a [`PanicMsg`] closure
    /// (evaluated only on miss when the function is `DefaultVal::Fail`).
    fn lookup(
        &mut self,
        func: FunctionId,
        entries: &[QueryEntry],
        panic_msg: PanicMsg,
    ) -> QueryEntry;

    /// RHS: subsume the row keyed by `entries` in `func`.
    ///
    /// Wraps `RuleBuilder::subsume`.
    ///
    /// **DuckDB**: errors at this call site. Programs that need subsume
    /// must use the reference backend in v1.
    fn subsume(&mut self, func: FunctionId, entries: &[QueryEntry]) -> Result<()>;

    /// RHS: set `func(entries[..n-1])` to `entries[n-1]`.
    ///
    /// Wraps `RuleBuilder::set`.
    fn set(&mut self, func: FunctionId, entries: &[QueryEntry]);

    /// RHS: remove the row keyed by `entries` from `func`.
    ///
    /// Wraps `RuleBuilder::remove`.
    fn remove(&mut self, func: FunctionId, entries: &[QueryEntry]);

    /// RHS: merge two values in the union-find.
    ///
    /// Wraps `RuleBuilder::union`.
    fn union(&mut self, l: QueryEntry, r: QueryEntry);

    /// RHS: panic with the given message.
    ///
    /// Wraps `RuleBuilder::panic`.
    fn panic(&mut self, message: String);

    /// Register a deferred-panic external function on the underlying egraph
    /// and return its id. Used (e.g. by `unstable-fn`) to bake a panic id
    /// into a wrapped function's call site.
    ///
    /// Wraps `RuleBuilder::new_panic`. Default panics for backends that do
    /// not support it; the bridge forwards to its `RuleBuilder`.
    fn new_panic(&mut self, _message: String) -> ExternalFunctionId {
        unimplemented!("new_panic is not supported on this backend")
    }

    /// Skip tree-decomposition during query planning for this rule
    /// (the `:no-decomp` option / `--no-decomp`). Default no-op for
    /// backends that don't decompose (e.g. duckdb); the bridge
    /// forwards it to its `RuleBuilder`.
    fn set_no_decomp(&mut self, _no_decomp: bool) {}

    /// Exclude the given function's table from being a seminaive delta focus
    /// for this rule: the rule never fires on *new* rows of this table joined
    /// against *old* rows of the others. Used by the term-encoding
    /// fast-rebuild (`--fast-rebuild` on the native bridge) to drop the
    /// always-empty `δview ⋈ uf_old` variant of the `view ⋈ @UF_Sf` rebuild
    /// rule (excludes the view table), keeping only `view ⋈ δuf`. Bit-exact
    /// with the full rebuild under canonicalize-at-creation.
    ///
    /// Default no-op: the dataflow/SQL backends implement their own
    /// fast-rebuild (`enable_fast_rebuild()`) and never receive these encoding
    /// rebuild rules in fast-rebuild mode; only the bridge forwards this to its
    /// `RuleBuilder`.
    fn set_focus_exclude_table(&mut self, _func: FunctionId) {}

    /// Finalize the rule. Returns the registered [`RuleId`].
    ///
    /// Wraps `RuleBuilder::build`. The DuckDB impl hands the accumulated
    /// `duck::Rule` IR off to `compile_rule` and inserts it into the
    /// backend's rule registry. If any accumulated call referenced a
    /// feature the backend does not support (e.g. subsume on DuckDB), this
    /// is where the error surfaces.
    fn build(self: Box<Self>) -> Result<RuleId>;

    /// Consume the builder and check whether its accumulated body
    /// matches at least one assignment in the current database.
    /// Actions accumulated in the builder are ignored. The bridge
    /// backend's default implementation panics; the DuckDB backend
    /// overrides this to compile the body to SQL and run it.
    /// Used by `(check …)` on the DuckDB path; the bridge keeps its
    /// existing external-func side-channel approach in `check_facts`.
    fn build_check(self: Box<Self>) -> Result<bool> {
        Err(anyhow::anyhow!(
            "build_check is not implemented for this backend"
        ))
    }

    /// Rename an already-registered primitive on the backend so that
    /// the per-id name reflects the call-site's concrete type. Used
    /// to disambiguate built-in primitives whose egglog name is
    /// overloaded across sorts (e.g. `^` is XOR for i64 and POWER for
    /// f64; `+` for String is concat). The bridge backend doesn't
    /// need this (it dispatches via the typed `ExternalFunction`
    /// directly), so the default is a no-op. The duck rule-builder
    /// routes the call through the egraph it already holds mutably.
    fn rename_prim(&mut self, _id: ExternalFunctionId, _name: String) {}
}

// ---------------------------------------------------------------------------
// `BaseValuePool` — dyn-compatible base-value registry
// ---------------------------------------------------------------------------

/// A registry for base-value types, exposed through dynamic dispatch so it
/// fits inside `dyn Backend`.
///
/// Generic-over-`T: BaseValue` helpers are provided at the bottom of this
/// module (`pool_get<T>`, `pool_unwrap<T>`, `pool_register_type<T>`,
/// `pool_get_ty<T>`); they wrap the dyn methods below.
///
/// The bridge's impl forwards to `egglog_core_relations::BaseValues`
/// directly. The DuckDB impl maintains its own `BaseValues`-shaped registry
/// in-process; entries for `i64`/`f64`/`bool`/`String`/`()` are encoded
/// inline into SQL columns where possible, and exotic types fall back to
/// an in-memory intern table identical to the bridge's.
pub trait BaseValuePool: Send + Sync {
    /// Register a new base-value type using its `TypeId` plus a factory
    /// that constructs a fresh typed intern table for the type.
    ///
    /// The factory is invoked at most once: if a type with the same
    /// `TypeId` is already registered, the existing [`BaseValueId`] is
    /// returned and the factory is dropped.
    ///
    /// Callers typically use [`pool_register_type`], the generic-over-`T`
    /// helper below, which constructs the appropriate factory and threads
    /// it through.
    ///
    /// Wraps `BaseValues::register_type_dyn` on the bridge side.
    fn register_type_dyn(
        &mut self,
        type_id: TypeId,
        factory: Box<dyn FnOnce() -> Box<dyn DynamicInternTable>>,
    ) -> BaseValueId;

    /// Look up the `BaseValueId` for a registered base-value type by its
    /// Rust `TypeId`.
    ///
    /// Wraps `BaseValues::get_ty_by_id`.
    fn get_ty_by_type_id(&self, type_id: TypeId) -> BaseValueId;

    /// Intern an opaque (already-boxed) base value, returning its `Value`
    /// handle. The `Box<dyn Any>` must contain a value of the type
    /// previously registered as `ty`.
    ///
    /// Wraps the dyn dispatch over `BaseValues::get<P>`. Concrete-`T`
    /// callers should prefer `pool_get<T>` below, which special-cases
    /// `T::MAY_UNBOX` and avoids boxing.
    fn intern_dyn(&self, ty: BaseValueId, value: Box<dyn Any + Send + Sync>) -> Value;

    /// Extract a base value of the registered type `ty` from `val`. The
    /// returned `Box<dyn Any>` holds a value of the same type that was
    /// previously interned with `intern_dyn`.
    ///
    /// Wraps the dyn dispatch over `BaseValues::unwrap<P>`. Concrete-`T`
    /// callers should prefer `pool_unwrap<T>` below.
    fn unwrap_dyn(&self, ty: BaseValueId, val: Value) -> Box<dyn Any + Send + Sync>;

    /// True iff a base-value type with the given `TypeId` is registered.
    fn has_ty(&self, type_id: TypeId) -> bool;
}

/// Generic helper: register a `T: BaseValue` with the pool and return its
/// [`BaseValueId`]. Equivalent to `BaseValues::register_type::<T>`.
///
/// This sits outside the trait so the trait remains dyn-compatible. It
/// builds the typed intern-table factory closure that
/// [`BaseValuePool::register_type_dyn`] requires.
pub fn pool_register_type<T: BaseValue>(pool: &mut dyn BaseValuePool) -> BaseValueId {
    pool.register_type_dyn(
        TypeId::of::<T>(),
        Box::new(|| egglog_core_relations::new_intern_table::<T>()),
    )
}

/// Generic helper: look up the `BaseValueId` for `T: BaseValue`.
pub fn pool_get_ty<T: BaseValue>(pool: &dyn BaseValuePool) -> BaseValueId {
    pool.get_ty_by_type_id(TypeId::of::<T>())
}

/// Generic helper: intern a typed base value into the pool, returning its
/// `Value`. Mirrors `BaseValues::get<T>`. Honors `T::MAY_UNBOX` for
/// inline-encodable types.
pub fn pool_get<T: BaseValue>(pool: &dyn BaseValuePool, value: T) -> Value {
    if T::MAY_UNBOX
        && let Some(v) = value.try_box()
    {
        return v;
    }
    let ty = pool_get_ty::<T>(pool);
    pool.intern_dyn(ty, Box::new(value))
}

/// Generic helper: extract a typed base value of `T` from a `Value`.
/// Mirrors `BaseValues::unwrap<T>`. Honors `T::MAY_UNBOX`.
///
/// # Panics
///
/// Panics if `val` does not correspond to a value of type `T` previously
/// interned in `pool` (matching the bridge's existing semantics).
pub fn pool_unwrap<T: BaseValue>(pool: &dyn BaseValuePool, val: Value) -> T {
    if T::MAY_UNBOX
        && let Some(p) = T::try_unbox(val)
    {
        return p;
    }
    let ty = pool_get_ty::<T>(pool);
    let boxed = pool.unwrap_dyn(ty, val);
    *boxed
        .downcast::<T>()
        .expect("BaseValuePool::unwrap_dyn returned wrong type")
}

// ---------------------------------------------------------------------------
// `ContainerPool` — stub-friendly container registry
// ---------------------------------------------------------------------------

/// A registry for container values (`Vec` / `Set` / `Map` / `MultiSet` /
/// user-defined `ContainerValue` impls).
///
/// ## Two implementations
///
/// - **Reference backend**: delegates to
///   `egglog_core_relations::ContainerValues`. All methods succeed; the
///   pool participates in the EGraph's rebuild loop.
/// - **DuckDB backend**: an empty stub. All accessor methods return `None`
///   / empty iterators; all mutators return errors. This is safe because
///   the term-encoding gate (`program_supports_proofs` in
///   `src/proofs/proof_encoding_helpers.rs`) excludes every container-using
///   program from DuckDB's test combos. The stub's error path is a
///   defensive measure for programmer error, never a routine code path.
///
/// ## Why not just generic-over-`C`?
///
/// `register_val<C>` / `for_each<C>` are generic over `C: ContainerValue`,
/// which is incompatible with `dyn Backend`. We expose Any-based dynamic
/// dispatch here; concrete-`C` sugar is implemented as free functions in a
/// future commit (the call sites in `src/lib.rs:1924`, `serialize.rs`,
/// `extract.rs` will each adopt the dyn API directly because they already
/// thread a `TypeId` through).
pub trait ContainerPool: Send + Sync {
    /// True iff a container type with the given `TypeId` is registered.
    fn has_container_type(&self, type_id: TypeId) -> bool;

    /// True iff this backend supports containers at all.
    ///
    /// Reference: `true`. DuckDB: `false`. Provided as a convenience so
    /// callers don't need to consult [`Backend::supports_containers`].
    fn enabled(&self) -> bool;

    /// Look up the container value associated with `val`. Returns `None`
    /// when no entry is registered.
    ///
    /// The returned `Box<dyn Any>` holds an instance of the container's
    /// registered Rust type.
    ///
    /// Wraps `ContainerValues::get_val`. On DuckDB always returns `None`.
    fn get_dyn(&self, ty: TypeId, val: Value) -> Option<Box<dyn Any + Send + Sync>>;

    /// Register a Rust container value, returning a fresh `Value` handle.
    ///
    /// On DuckDB this returns an error (containers unsupported).
    fn register_val_dyn(&mut self, ty: TypeId, value: Box<dyn Any + Send + Sync>) -> Result<Value>;

    /// Iterate (id, container) pairs for all registered values of a
    /// container type. The callback receives the container as a
    /// `&dyn Any` of the registered concrete type.
    ///
    /// Wraps `ContainerValues::for_each<C>`. On DuckDB this is a no-op
    /// (the stub registry is empty).
    fn for_each_dyn(&self, ty: TypeId, f: &mut dyn FnMut(Value, &dyn Any));

    /// Number of registered values of the given container type.
    fn size(&self, ty: TypeId) -> usize;
}

/// Generic helper: register a `C: ContainerValue` with the pool, returning
/// its `Value`. On DuckDB this returns an error.
pub fn container_register_val<C: ContainerValue>(
    pool: &mut dyn ContainerPool,
    value: C,
) -> Result<Value> {
    pool.register_val_dyn(TypeId::of::<C>(), Box::new(value))
}

// ---------------------------------------------------------------------------
// Trait sanity checks
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time check that `Backend` is dyn-compatible.
    #[allow(dead_code)]
    fn assert_dyn_backend(_: &dyn Backend) {}

    /// Compile-time check that `RuleBuilderOps` is dyn-compatible.
    #[allow(dead_code)]
    fn assert_dyn_rule_builder(_: &mut dyn RuleBuilderOps) {}

    /// Compile-time check that `BaseValuePool` is dyn-compatible.
    #[allow(dead_code)]
    fn assert_dyn_base_pool(_: &dyn BaseValuePool) {}

    /// Compile-time check that `ContainerPool` is dyn-compatible.
    #[allow(dead_code)]
    fn assert_dyn_container_pool(_: &dyn ContainerPool) {}

    /// Compile-time check that `Box<dyn Backend>` is `Send + Sync` and
    /// `Clone`.
    #[allow(dead_code)]
    fn assert_box_backend_send_sync_clone(b: Box<dyn Backend>) -> Box<dyn Backend> {
        fn require_send_sync<T: Send + Sync>() {}
        require_send_sync::<Box<dyn Backend>>();
        b.clone()
    }
}
