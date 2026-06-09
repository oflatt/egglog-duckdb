//! DBSP-backed body join for the Feldera backend (Milestone 4).
//!
//! This module runs a rule's **relational body join on DBSP's dataflow
//! engine** — the paper's core technical contribution — rather than on the
//! host-side nested-loop interpreter (`interpret.rs`, retained as the
//! correctness oracle / fallback).
//!
//! ## What runs on DBSP
//!
//! For a DBSP-eligible rule (see [`plan_join`]), [`run_join`] builds a
//! **non-recursive** DBSP circuit that computes the join of the body's table
//! atoms (multi-atom, left-deep), with `!=` guards applied as DBSP `filter`
//! operators *inside* the dataflow. One `transaction()` performs exactly one
//! round of the join over the current relation contents (the per-iteration
//! model M1/M2 proved). The circuit's output is a z-set of **binding rows**:
//! one fixed-width binding row ([`BindRow`], a 32-wide [`TupRow`]) per
//! satisfying assignment, holding the rule's body variables in a fixed
//! canonical order (see [`JOIN_WIDTH`]).
//!
//! ## What stays on the host (the frontier)
//!
//! The join *output* (binding rows) is read back and handed to the existing
//! head-application machinery (`interpret::run_iteration`), which evaluates
//! value-computing primitives (via the backend-agnostic
//! [`crate::EGraph::eval_prim_internal`]) and applies `set` / `delete` /
//! `lookup` / `union` head actions + FD-merge resolution. DBSP map/filter
//! closures are `Send + 'static` and cannot borrow the primitive engine, so
//! primitive *value computation* and head writes remain host-side. The join
//! itself — the expensive, paper-relevant part — runs on DBSP.
//!
//! ## Eligibility / the row-width cap
//!
//! DBSP rows must be fixed-arity `DBData` (rkyv-archivable); we use a uniform
//! 32-wide [`TupRow`] (see [`JOIN_WIDTH`]). A rule is DBSP-eligible iff:
//!   - its body has at least one table atom;
//!   - every table atom has arity <= [`JOIN_WIDTH`];
//!   - the rule's body uses <= [`JOIN_WIDTH`] distinct variables (binding row);
//!   - its only body *primitives* are a recognized set of PURE prims (`!=`,
//!     `bool-!=`, `or`, `guard`, `ordering-min/max`, recognized by name) whose
//!     operands sit on columns supporting the relevant rep-arithmetic: ORDERING
//!     prims (`ordering-min/max`) require `Id` columns; EQUALITY prims (`!=` /
//!     `bool-!=` / `or` / `guard`) also accept the injectively-interned base
//!     types `bool` / `i64` / `string` (rep-equality ⇔ value-equality). These
//!     are inlined into the join — value prims as symbolic expressions, guards
//!     as DBSP `filter`s — so the `@congruence` / `@rebuild_cleanup` /
//!     base-value `@merge` rules run their body join on-circuit. Any other body
//!     prim, an ordering prim on a non-`Id` column, or any prim touching `f64`
//!     (NaN breaks rep-equality ⇔ value-`!=`) or another unvetted base value,
//!     forces the host fallback.
//!
//! Rules that are not eligible fall back to the host interpreter; `run_rules`
//! reports the split so the milestone can characterize the frontier honestly.

use anyhow::{anyhow, Result};
use dbsp::{CircuitHandle, OrdZSet, OutputHandle, RootCircuit, Stream, ZSetHandle, ZWeight};
use egglog_backend_trait::{ColumnTy, ExternalFunctionId, FunctionId, Value};
use egglog_numeric_id::NumericId;
use hashbrown::{HashMap, HashSet};

use crate::compile::{BodyOp, RuleIr, Slot};
use crate::{EGraph, PrimEngine};

/// Max distinct body variables / atom columns the DBSP join supports (the
/// fixed-arity `DBData` row width).
///
/// Stage C widens this from 10 (dbsp's stock [`dbsp::utils::Tup10`]) to 32 via a
/// custom [`TupRow`] declared with dbsp's `declare_tuples!` macro, because the
/// persistent circuit is now the ONLY join path: it must cover the WIDEST
/// feldera-supported rule. A suite-wide survey put the peak at ~23 distinct body
/// variables (e.g. `cykjson` `check_facts`, the `const-prop` value rules — the
/// latter also materialize on-circuit prim outputs into binding columns), so 32
/// gives headroom. Cost: a binding row is `32 * 4 = 128` bytes (vs 40 at width
/// 10); rows flow through the join z-sets, so this trades memory for coverage.
/// Pulling in dbsp's DBData derive stack (paste/rkyv/size_of/serde/derive_more)
/// is the one-time dependency price for a row wider than `Tup10`.
pub const JOIN_WIDTH: usize = 32;

// Declare a 32-wide `DBData` tuple via dbsp's own macro, so the generated trait
// impls (rkyv `Archive`, `SizeOf`, `MulByRef`, `HasZero`, …) match dbsp's stock
// tuples exactly. The derive stack the macro expands to (paste/derive_more/
// serde/rkyv/size_of) is pinned in Cargo.toml to dbsp 0.150's versions.
// The macro binds its generic type params as value-position idents (`T1`..),
// which trips `non_snake_case`; the names are dbsp's macro contract, not ours.
#[allow(non_snake_case)]
mod tup_row {
    dbsp::declare_tuples! {
        TupRow<
            T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16,
            T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32
        >,
    }
}
pub(crate) use tup_row::TupRow;

/// Destructure a [`TupRow`] / build one — the two operations that touch all 32
/// positional fields. Centralized here so the field list is written once.
macro_rules! with_row_fields {
    // Build a TupRow from a `[u32; 32]` array `$a`.
    (build $a:expr) => {{
        let a = $a;
        TupRow(
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
            a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25],
            a[26], a[27], a[28], a[29], a[30], a[31],
        )
    }};
    // Destructure a `&TupRow` `$r` into a `[u32; 32]` array.
    (read $r:expr) => {{
        let TupRow(
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18,
            a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
        ) = $r;
        [
            *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12, *a13, *a14, *a15,
            *a16, *a17, *a18, *a19, *a20, *a21, *a22, *a23, *a24, *a25, *a26, *a27, *a28, *a29,
            *a30, *a31,
        ]
    }};
}

/// A fixed-width binding row flowing through the DBSP circuit: `bind[i]` is the
/// value of the rule's `i`-th canonical body variable (0 if not yet bound).
type BindRow = TupRow<
    u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32,
    u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32,
>;

/// A fixed-width relation row pushed into the circuit's input z-sets.
type RelRow = BindRow;

/// Build a fixed-width row from a slice of column values (0-padded to
/// [`JOIN_WIDTH`]). Slices longer than [`JOIN_WIDTH`] are rejected upstream by
/// [`plan_join`].
fn pack_row(vals: &[u32]) -> BindRow {
    let mut a = [0u32; JOIN_WIDTH];
    for (i, v) in vals.iter().enumerate() {
        a[i] = *v;
    }
    arr_to_row(a)
}

/// Pack a fixed `[u32; JOIN_WIDTH]` array into the row tuple.
#[inline]
fn arr_to_row(a: [u32; JOIN_WIDTH]) -> BindRow {
    with_row_fields!(build a)
}

/// Read column `i` of a fixed-width row.
#[inline]
fn get_col(r: &BindRow, i: usize) -> u32 {
    let a = with_row_fields!(read r);
    a[i]
}

// Names of the PURE prims that `plan_join` inlines into the persistent DBSP
// join (Stage B). Recognition is by name only (no `@uf`/rebuild rule-name
// recognition). See `set_external_func_name` for where these are registered.
/// `!=` guard (pure `u32` inequality on the interned rep).
const NEQ_NAME: &str = "!=";
/// `bool-!=` value prim: produces the bool `a != b`.
const BOOL_NE_NAME: &str = "bool-!=";
/// `or` value prim: produces the disjunction of its bool operands.
const OR_NAME: &str = "or";
/// `guard` prim: prunes the match unless its bool operand is true.
const GUARD_NAME: &str = "guard";
/// `ordering-min` value prim: min of two reps.
const ORD_MIN_NAME: &str = "ordering-min";
/// `ordering-max` value prim: max of two reps.
const ORD_MAX_NAME: &str = "ordering-max";

/// Rep-arithmetic kind of an atom-bound column, gating which inlined prim
/// operations are correct on its interned `u32` rep. See `plan_join`'s
/// "Correctness gating" docs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RepKind {
    /// `Id` (union-find) column: rep order = id order, so ordering AND equality
    /// are both valid.
    Ordering,
    /// bool / i64 / string column: injective interning makes equality / `!=`
    /// valid, but rep order ≠ value order so ordering is NOT inlinable.
    Equality,
}

/// A pure VALUE expression over a binding row, built from inlined value prims.
/// Evaluates to a `u32` (an interned rep). Only constructed when every leaf is
/// a binding-row column whose type makes rep-arithmetic provably correct (an
/// `Id`/union-find column, or — for equality only — a bool column).
#[derive(Clone, Debug)]
enum PureExpr {
    /// A binding-row column (an atom-bound variable).
    Col(usize),
    /// A literal rep.
    Const(u32),
    /// `ordering-min(a, b)` = numeric min of the two reps.
    Min(Box<PureExpr>, Box<PureExpr>),
    /// `ordering-max(a, b)` = numeric max of the two reps.
    Max(Box<PureExpr>, Box<PureExpr>),
}

/// A pure BOOLEAN condition over a binding row, built from inlined guard/bool
/// prims. Lowered to a DBSP `filter` once all its leaf columns are bound.
#[derive(Clone, Debug)]
enum Cond {
    /// `a != b` on reps (from `!=` and `bool-!=`).
    Ne(PureExpr, PureExpr),
    /// `a == b` on reps (from an `(= (ordering-* a b) c)` assert).
    Eq(PureExpr, PureExpr),
    /// Disjunction (from `or`).
    Or(Vec<Cond>),
}

impl PureExpr {
    /// Evaluate against a binding row.
    fn eval(&self, row: &BindRow) -> u32 {
        match self {
            PureExpr::Col(c) => get_col(row, *c),
            PureExpr::Const(v) => *v,
            PureExpr::Min(a, b) => a.eval(row).min(b.eval(row)),
            PureExpr::Max(a, b) => a.eval(row).max(b.eval(row)),
        }
    }
    /// Binding-row columns this expression reads.
    fn cols(&self, out: &mut Vec<usize>) {
        match self {
            PureExpr::Col(c) => out.push(*c),
            PureExpr::Const(_) => {}
            PureExpr::Min(a, b) | PureExpr::Max(a, b) => {
                a.cols(out);
                b.cols(out);
            }
        }
    }
}

impl Cond {
    fn eval(&self, row: &BindRow) -> bool {
        match self {
            Cond::Ne(a, b) => a.eval(row) != b.eval(row),
            Cond::Eq(a, b) => a.eval(row) == b.eval(row),
            Cond::Or(cs) => cs.iter().any(|c| c.eval(row)),
        }
    }
    fn cols(&self, out: &mut Vec<usize>) {
        match self {
            Cond::Ne(a, b) | Cond::Eq(a, b) => {
                a.cols(out);
                b.cols(out);
            }
            Cond::Or(cs) => cs.iter().for_each(|c| c.cols(out)),
        }
    }
}

/// How a [`PrimStep`] reads one of its primitive arguments from the binding row.
#[derive(Clone, Debug)]
enum ArgSrc {
    /// Read binding-row column `usize`.
    Col(usize),
    /// A literal rep.
    Const(u32),
}

/// What a [`PrimStep`] does with the primitive's result.
#[derive(Clone, Debug)]
enum PrimRet {
    /// Bind the result into this (freshly-allocated) binding-row column.
    Bind(usize),
    /// Assert the result equals binding-row column `usize` (prune the row if
    /// not).
    AssertCol(usize),
    /// Assert the result equals this literal rep.
    AssertConst(u32),
}

/// An ON-CIRCUIT call-prim step (Stage C of #23): re-evaluate a PURE body
/// primitive through the shared [`PrimEngine`] under a lock, materializing the
/// interned result into the binding row (or pruning the row if the prim fails /
/// the asserted return value mismatches). Applied as a DBSP `flat_map` after the
/// join, in body order, so a step may read a column an earlier step produced.
///
/// This is the general path for ANY pure value prim (`+`, `-`, `>`, `int-div`,
/// `string-concat`, `i64-to-string`, …) — semantics are never reimplemented; the
/// real prim runs on the same engine the host uses, interning into shared tables
/// so the result handle is bit-identical to a host evaluation.
#[derive(Clone, Debug)]
struct PrimStep {
    id: ExternalFunctionId,
    args: Vec<ArgSrc>,
    ret: PrimRet,
}

/// The analysis of a DBSP-eligible rule body: canonical variable order, the
/// table atoms, and the boolean guards inlined from pure body prims.
pub struct JoinPlan {
    /// Canonical body-variable order: `var_order[i]` is the variable id placed
    /// at binding-row column `i`.
    var_order: Vec<u32>,
    /// var id -> its binding-row column index.
    var_col: HashMap<u32, usize>,
    /// The body table atoms (in emission order).
    atoms: Vec<AtomPlan>,
    /// Boolean guards inlined from the body's pure prims, applied as DBSP
    /// `filter`s once every binding-row column they read is bound. Used by the
    /// in-join fast path (rules whose only prims are recognized rep-comparisons).
    guards: Vec<Cond>,
    /// ON-CIRCUIT call-prim steps (Stage C): present iff the rule has a genuine
    /// pure VALUE prim (one not in the recognized rep-comparison set). When
    /// non-empty, `guards` is empty and the WHOLE prim chain is lowered to these
    /// engine-evaluated steps, applied (as `flat_map`s) after the join in body
    /// order. Mutually exclusive with `guards`: a rule takes either the in-join
    /// rep fast path (no lock) or the call-prim path (locks per row), never both.
    steps: Vec<PrimStep>,
}

/// One table atom in the plan.
struct AtomPlan {
    func: FunctionId,
    /// Per-column slot (variable or constant).
    slots: Vec<Slot>,
}

impl JoinPlan {
    /// The number of canonical body variables (binding-row width in use).
    pub fn n_vars(&self) -> usize {
        self.var_order.len()
    }

    /// The canonical variable order (column i holds variable `var_order[i]`).
    pub fn var_order(&self) -> &[u32] {
        &self.var_order
    }
}

/// Decide whether `rule` can run its body join on DBSP, and if so return the
/// [`JoinPlan`]. Returns `None` (host fallback) when any eligibility condition
/// fails.
///
/// `allow_prims` controls whether a body PURE prim (`!=`, `bool-!=`, `or`,
/// `guard`, `ordering-min/max`) keeps the rule eligible by being inlined into
/// the join (Stage B). The **persistent** circuit (`persistent_bindings`)
/// drives the transaction lifecycle manually and is bit-exact for these inlined
/// prims, so it passes `true`; this is what makes the `@congruence` /
/// `@rebuild_cleanup` rules DBSP-eligible. The non-persistent seminaive path
/// (`seminaive_bindings`, `run_join_seminaive`) is *not* bit-exact for them, so
/// it passes `false` and those rules stay on the correct host path. Rules whose
/// bodies have no prims are unaffected either way.
///
/// ## Correctness gating (rep-arithmetic validity)
///
/// Inlined prim closures operate on the INTERNED `u32` rep, not the logical
/// value. `ordering-min/max` and `<`/`==`/`!=` on the rep are only valid when
/// the rep order/equality matches the logical one:
///   - `ColumnTy::Id` columns: the rep IS the union-find id, so min/max = the
///     leader choice and equality is identity — VALID.
///   - bool columns: distinct logical values get distinct interned reps, so
///     equality / `!=` are VALID (but ordering is meaningless — we never inline
///     ordering on bool).
///   - i64 / string columns: interning is injective (distinct values get
///     distinct reps, and `intern(a) == intern(b)` iff `a == b`), so equality /
///     `!=` are VALID. Ordering is NOT — the rep is a handle whose order ≠ value
///     order — so we never inline `<`/`ordering-min/max` on them.
///   - f64 columns: an interned `NaN` rep equals itself, but IEEE says
///     `NaN != NaN`, so rep-equality does NOT match value-`!=`. f64 is therefore
///     EXCLUDED from equality inlining and stays on the host.
///   - any other base value (BigInt/BigRat/…): conservatively INVALID. Any prim
///     instance touching such a column leaves the whole rule on the host
///     (graceful fallback).
pub fn plan_join(eg: &EGraph, rule: &RuleIr, allow_prims: bool) -> Option<JoinPlan> {
    // DIAGNOSTIC ONLY (gated `FELDERA_DBG_PLAN`): name the rejected rule and the
    // blocking reason so a suite-wide survey can aggregate host-fallback causes.
    // Purely env-gated; never affects default behavior. Only logs the
    // prim-allowing (persistent) call so each rejection is reported once.
    macro_rules! reject {
        ($($reason:tt)*) => {{
            if allow_prims && std::env::var("FELDERA_DBG_PLAN").is_ok() {
                #[allow(clippy::disallowed_macros)]
                {
                    eprintln!("[DBG_PLAN] reject rule={:?} reason={}", rule.name, format!($($reason)*));
                }
            }
            return None;
        }};
    }

    let mut atoms: Vec<AtomPlan> = Vec::new();
    let mut guards: Vec<Cond> = Vec::new();
    let mut var_order: Vec<u32> = Vec::new();
    let mut var_col: HashMap<u32, usize> = HashMap::new();
    // Each atom-bound variable's column rep-arithmetic kind:
    //   `Ordering` (Id columns): rep order = id order, so ordering AND equality
    //     are both valid.
    //   `Equality` (bool / i64 / string): interning is injective so equality /
    //     `!=` are valid, but rep order ≠ value order so ordering is NOT inlined.
    //   absent (`None`): f64 (NaN breaks rep-equality ↔ value-`!=`) or any other
    //     base value — never inlinable; any prim use leaves the rule on host.
    let mut var_kind: HashMap<u32, RepKind> = HashMap::new();
    // Symbolic definitions for prim-produced (not atom-bound) variables.
    let mut pure_vals: HashMap<u32, PureExpr> = HashMap::new();
    let mut pure_conds: HashMap<u32, Cond> = HashMap::new();

    let see_var = |v: u32, var_order: &mut Vec<u32>, var_col: &mut HashMap<u32, usize>| {
        if !var_col.contains_key(&v) {
            var_col.insert(v, var_order.len());
            var_order.push(v);
        }
    };

    let bool_bvid = eg.bool_bvid();
    let i64_bvid = eg.i64_bvid();
    let string_bvid = eg.string_bvid();
    // Classify a function column's type into the rep-arithmetic kind, or `None`
    // if it cannot be inlined at all (f64 / other base values). NOTE: matching
    // i64/string POSITIVELY (rather than excluding f64) is deliberately
    // conservative — any base type we have not vetted as injective-with-matching
    // equality (BigInt, BigRat, future types) falls through to `None` and the
    // rule stays on the host.
    let col_kind = |f: FunctionId, col: usize| -> Option<RepKind> {
        match eg.col_ty(f, col) {
            Some(ColumnTy::Id) => Some(RepKind::Ordering),
            Some(ColumnTy::Base(bv))
                if Some(bv) == bool_bvid || Some(bv) == i64_bvid || Some(bv) == string_bvid =>
            {
                Some(RepKind::Equality)
            }
            _ => None,
        }
    };

    // Resolve a prim-operand slot to a `PureExpr` value, or `None` if it cannot
    // be safely inlined (unbound var, or a column whose rep-arithmetic kind does
    // not support the requested operation).
    let val_of = |s: &Slot,
                  pure_vals: &HashMap<u32, PureExpr>,
                  var_col: &HashMap<u32, usize>,
                  var_kind: &HashMap<u32, RepKind>,
                  need_ordering: bool|
     -> Option<PureExpr> {
        match s {
            Slot::Const(c) => Some(PureExpr::Const(*c)),
            Slot::Var(v) => {
                if let Some(e) = pure_vals.get(v) {
                    return Some(e.clone());
                }
                let &col = var_col.get(v)?;
                match var_kind.get(v) {
                    // Id column: ordering + equality both valid.
                    Some(RepKind::Ordering) => Some(PureExpr::Col(col)),
                    // bool / i64 / string column: equality only (rep order is
                    // meaningless / handle-order, so never inline ordering).
                    Some(RepKind::Equality) if !need_ordering => Some(PureExpr::Col(col)),
                    _ => None,
                }
            }
        }
    };

    // Names recognized by the in-join rep fast path (no engine lock).
    let is_recognized = |name: Option<&str>| {
        matches!(
            name,
            Some(NEQ_NAME)
                | Some(BOOL_NE_NAME)
                | Some(OR_NAME)
                | Some(GUARD_NAME)
                | Some(ORD_MIN_NAME)
                | Some(ORD_MAX_NAME)
        )
    };

    // Decide the prim-lowering MODE for this rule (Stage C). If every body prim
    // is a recognized rep-comparison, take the in-join fast path (PureExpr/Cond
    // guards, no lock). If ANY body prim is a genuine pure VALUE prim, lower the
    // WHOLE prim chain to on-circuit call-prim steps (engine-evaluated). If any
    // body prim is impure (not registered as pure) the rule is ineligible — it
    // must stay on the host / panic, never re-evaluated on-circuit.
    let mut use_calls = false;
    for op in &rule.body {
        if let BodyOp::Prim { id, .. } = op {
            if !allow_prims {
                return None;
            }
            if !eg.is_pure_prim(*id) {
                reject!(
                    "impure body prim {:?} (not eligible on-circuit)",
                    eg.external_funcs.name(*id)
                );
            }
            if !is_recognized(eg.external_funcs.name(*id)) {
                use_calls = true;
            }
        }
    }

    // Steps emitted for the call-prim path (Stage C). Empty unless `use_calls`.
    let mut steps: Vec<PrimStep> = Vec::new();
    // Resolve a call-prim argument slot to an `ArgSrc` (a binding-row column or a
    // const). The var MUST already have a column (bound by an atom or an earlier
    // call-prim's `Bind`); otherwise the rule is ineligible.
    let arg_src = |s: &Slot, var_col: &HashMap<u32, usize>| -> Option<ArgSrc> {
        match s {
            Slot::Const(c) => Some(ArgSrc::Const(*c)),
            Slot::Var(v) => var_col.get(v).map(|&c| ArgSrc::Col(c)),
        }
    };

    // Call-prim path: pre-allocate a binding column for EVERY atom-bound variable
    // before lowering any prim, regardless of body order. A prim whose `ret` is a
    // variable an atom ALSO binds must become an `AssertCol` (the join binds the
    // column; the step then asserts the prim result equals it) — never a `Bind`,
    // which would overwrite the atom's binding. Body order would otherwise make a
    // prim appearing before its consuming atom (the `(predicate (+ x 1))` shape)
    // mis-lower to `Bind`, silently dropping the atom's join constraint.
    if use_calls {
        for op in &rule.body {
            if let BodyOp::Atom(atom) = op {
                for s in &atom.slots {
                    if let Slot::Var(v) = s {
                        see_var(*v, &mut var_order, &mut var_col);
                    }
                }
            }
        }
    }

    for op in &rule.body {
        match op {
            BodyOp::Atom(atom) => {
                if atom.slots.len() > JOIN_WIDTH {
                    reject!(
                        "atom arity {} > JOIN_WIDTH {}",
                        atom.slots.len(),
                        JOIN_WIDTH
                    );
                }
                for (col, s) in atom.slots.iter().enumerate() {
                    if let Slot::Var(v) = s {
                        see_var(*v, &mut var_order, &mut var_col);
                        // Record the column's rep-arithmetic kind. (If a var is
                        // bound by several atoms, any `Ordering` occurrence wins
                        // — they must agree by the type system anyway.)
                        if let Some(k) = col_kind(atom.func, col) {
                            var_kind
                                .entry(*v)
                                .and_modify(|e| {
                                    if k == RepKind::Ordering {
                                        *e = RepKind::Ordering;
                                    }
                                })
                                .or_insert(k);
                        }
                    }
                }
                atoms.push(AtomPlan {
                    func: atom.func,
                    slots: atom.slots.clone(),
                });
            }
            BodyOp::Prim { id, args, ret } => {
                if !allow_prims {
                    return None;
                }
                // Stage C call-prim path: this rule has a genuine value prim, so
                // EVERY prim (recognized or not) is lowered to an on-circuit
                // engine call. Args read prior binding columns; the result either
                // binds a fresh column (`ret` a new var) or asserts equality with
                // an already-bound column/const. Pure prims are idempotent, so
                // re-evaluating here is bit-identical to the host.
                if use_calls {
                    let mut srcs: Vec<ArgSrc> = Vec::with_capacity(args.len());
                    for a in args {
                        match arg_src(a, &var_col) {
                            Some(s) => srcs.push(s),
                            None => reject!("call-prim {:?} arg unbound", eg.external_funcs.name(*id)),
                        }
                    }
                    let ret = match ret {
                        // `ret` already bound (by an atom or an earlier step):
                        // assert the prim result equals it.
                        Slot::Var(rv) if var_col.contains_key(rv) => {
                            PrimRet::AssertCol(var_col[rv])
                        }
                        // `ret` is a fresh var: allocate a column and bind it. The
                        // result may then be read by a later step, an atom (rare),
                        // or escape to the head (the whole point of a value prim).
                        Slot::Var(rv) => {
                            see_var(*rv, &mut var_order, &mut var_col);
                            PrimRet::Bind(var_col[rv])
                        }
                        Slot::Const(c) => PrimRet::AssertConst(*c),
                    };
                    steps.push(PrimStep {
                        id: *id,
                        args: srcs,
                        ret,
                    });
                    continue;
                }
                let name = eg.external_funcs.name(*id);
                match name {
                    Some(NEQ_NAME) | Some(BOOL_NE_NAME) => {
                        // `a != b` on reps. `!=` returns unit (a guard); `bool-!=`
                        // returns a bool the `or`/`guard` chain consumes.
                        if args.len() != 2 {
                            reject!("prim {} bad-arity {}", name.unwrap_or("?"), args.len());
                        }
                        let a = match val_of(&args[0], &pure_vals, &var_col, &var_kind, false) {
                            Some(e) => e,
                            None => {
                                reject!("prim {} operand0 not-rep-comparable", name.unwrap_or("?"))
                            }
                        };
                        let b = match val_of(&args[1], &pure_vals, &var_col, &var_kind, false) {
                            Some(e) => e,
                            None => {
                                reject!("prim {} operand1 not-rep-comparable", name.unwrap_or("?"))
                            }
                        };
                        let cond = Cond::Ne(a, b);
                        if name == Some(NEQ_NAME) {
                            // Unit guard: prune unless a != b.
                            guards.push(cond);
                        } else if let Slot::Var(rv) = ret {
                            // Bind the bool result symbolically for `or`/`guard`.
                            pure_conds.insert(*rv, cond);
                        } else {
                            reject!("prim bool-!= ret not a var");
                        }
                    }
                    Some(OR_NAME) => {
                        // Disjunction of bool operands (each a prior `bool-!=`).
                        let mut cs = Vec::with_capacity(args.len());
                        for s in args {
                            match s {
                                Slot::Var(v) => match pure_conds.get(v) {
                                    Some(c) => cs.push(c.clone()),
                                    None => reject!("prim or operand not a bound bool-cond"),
                                },
                                Slot::Const(_) => reject!("prim or const operand"),
                            }
                        }
                        if let Slot::Var(rv) = ret {
                            pure_conds.insert(*rv, Cond::Or(cs));
                        } else {
                            reject!("prim or ret not a var");
                        }
                    }
                    Some(GUARD_NAME) => {
                        // Prune unless the bool operand is true.
                        if args.len() != 1 {
                            reject!("prim guard bad-arity {}", args.len());
                        }
                        match &args[0] {
                            Slot::Var(v) => match pure_conds.get(v) {
                                Some(c) => guards.push(c.clone()),
                                None => reject!("prim guard operand not a bound bool-cond"),
                            },
                            Slot::Const(_) => reject!("prim guard const operand"),
                        }
                    }
                    Some(ORD_MIN_NAME) | Some(ORD_MAX_NAME) => {
                        // Value-producing ordering prim on reps — ONLY valid on
                        // Id columns (rep order = id order = leader choice).
                        if args.len() != 2 {
                            reject!("prim {} bad-arity {}", name.unwrap_or("?"), args.len());
                        }
                        let a = match val_of(&args[0], &pure_vals, &var_col, &var_kind, true) {
                            Some(e) => e,
                            None => {
                                reject!("prim {} operand0 not-Id-ordering", name.unwrap_or("?"))
                            }
                        };
                        let b = match val_of(&args[1], &pure_vals, &var_col, &var_kind, true) {
                            Some(e) => e,
                            None => {
                                reject!("prim {} operand1 not-Id-ordering", name.unwrap_or("?"))
                            }
                        };
                        let expr = if name == Some(ORD_MIN_NAME) {
                            PureExpr::Min(Box::new(a), Box::new(b))
                        } else {
                            PureExpr::Max(Box::new(a), Box::new(b))
                        };
                        match ret {
                            // `ret` already bound by an atom (or another prim
                            // value): an equality assert `expr == ret`.
                            Slot::Var(rv) if var_col.contains_key(rv) => {
                                let rexpr = match val_of(ret, &pure_vals, &var_col, &var_kind, true)
                                {
                                    Some(e) => e,
                                    None => {
                                        reject!("prim {} ret not-Id-ordering", name.unwrap_or("?"))
                                    }
                                };
                                guards.push(Cond::Eq(expr, rexpr));
                            }
                            Slot::Var(rv) if pure_vals.contains_key(rv) => {
                                let rexpr = pure_vals.get(rv).unwrap().clone();
                                guards.push(Cond::Eq(expr, rexpr));
                            }
                            // `ret` is a fresh var: bind it symbolically so later
                            // prims can read it. (If it were needed by an atom or
                            // the head we'd have to materialize it into the row;
                            // none of the recognized rules do this, and an atom
                            // binding a value-prim output is not produced by the
                            // term encoder.)
                            Slot::Var(rv) => {
                                pure_vals.insert(*rv, expr);
                            }
                            Slot::Const(c) => {
                                guards.push(Cond::Eq(expr, PureExpr::Const(*c)));
                            }
                        }
                    }
                    other => {
                        // An unrecognized / non-inlinable body prim: leave the
                        // whole rule on the host nested-loop join.
                        reject!("unrecognized-prim {}", other.unwrap_or("<unnamed>"));
                    }
                }
            }
        }
    }

    // A prim-produced var must not escape to the head as a materialized value
    // (we keep such vars symbolic, never in the binding row). The recognized
    // rules only use prim outputs inside the body prim chain, so verify the head
    // never reads one — if it does, fall back to the host to stay correct.
    for hv in head_vars(rule) {
        if (pure_vals.contains_key(&hv) || pure_conds.contains_key(&hv))
            && !var_col.contains_key(&hv)
        {
            reject!("prim-output var escapes to head");
        }
    }

    if atoms.is_empty() {
        reject!("no body atoms (constant-only / prim-only body)");
    }
    if var_order.len() > JOIN_WIDTH {
        reject!(
            "too many vars {} > JOIN_WIDTH {}",
            var_order.len(),
            JOIN_WIDTH
        );
    }

    Some(JoinPlan {
        var_order,
        var_col,
        atoms,
        guards,
        steps,
    })
}

/// Variables read by a rule's head actions (so we can verify no symbolic
/// prim-output var escapes the body).
fn head_vars(rule: &RuleIr) -> Vec<u32> {
    use crate::compile::HeadOp;
    let mut out = Vec::new();
    let push_slot = |s: &Slot, out: &mut Vec<u32>| {
        if let Slot::Var(v) = s {
            out.push(*v);
        }
    };
    for op in &rule.head {
        match op {
            HeadOp::Set { slots, .. }
            | HeadOp::Remove { slots, .. }
            | HeadOp::Subsume { slots, .. } => {
                for s in slots {
                    push_slot(s, &mut out);
                }
            }
            HeadOp::Lookup { args, ret, .. } => {
                for s in args {
                    push_slot(s, &mut out);
                }
                out.push(*ret);
            }
            HeadOp::Call { args, ret, .. } => {
                for s in args {
                    push_slot(s, &mut out);
                }
                out.push(*ret);
            }
            HeadOp::Union { l, r } => {
                push_slot(l, &mut out);
                push_slot(r, &mut out);
            }
            HeadOp::Panic(_) => {}
        }
    }
    out
}

/// Per-relation input handle for the join circuit.
struct RelHandles {
    input: ZSetHandle<RelRow>,
}

/// Run the body join of `plan` on DBSP over the current mirror (full relations
/// in every atom occurrence), returning one binding row per satisfying
/// assignment. Retained for the M4 non-incremental proof; the seminaive driver
/// uses [`run_join_seminaive`].
pub fn run_join(eg: &EGraph, plan: &JoinPlan) -> Result<Vec<Vec<u32>>> {
    // No delta atom: every occurrence reads the full mirror.
    run_join_with(eg, plan, None)
}

/// Run the body join with **one atom occurrence restricted to its relation's
/// delta** and every other occurrence over the full relation — one term of the
/// seminaive union (see `interpret::seminaive_bindings`). `delta_atom_ord` is
/// the occurrence's index within `plan.atoms` (plan atoms are in body order).
pub fn run_join_seminaive(
    eg: &EGraph,
    plan: &JoinPlan,
    delta_atom_ord: usize,
    delta: &HashMap<FunctionId, HashSet<crate::compile::Row>>,
) -> Result<Vec<Vec<u32>>> {
    run_join_with(eg, plan, Some((delta_atom_ord, delta)))
}

/// Core join runner. Builds a fresh non-recursive circuit with **one input
/// stream per atom occurrence** (so the same relation appearing in multiple
/// atoms can be fed different row sets — full vs. delta — for seminaive), pushes
/// each occurrence's rows, runs one `transaction()`, and reads the consolidated
/// binding rows. When `delta` is `Some((ord, d))`, occurrence `ord` is fed
/// `d[func]` (its delta) and all others the full mirror.
fn run_join_with(
    eg: &EGraph,
    plan: &JoinPlan,
    delta: Option<(usize, &HashMap<FunctionId, HashSet<crate::compile::Row>>)>,
) -> Result<Vec<Vec<u32>>> {
    let n_atoms = plan.atoms.len();

    // Snapshot the rows feeding each atom occurrence (fixed-width, 0-padded).
    let mut snapshot: Vec<Vec<RelRow>> = Vec::with_capacity(n_atoms);
    for (ord, a) in plan.atoms.iter().enumerate() {
        let rows: Vec<RelRow> = match delta {
            Some((delta_ord, d)) if delta_ord == ord => d
                .get(&a.func)
                .map(|set| set.iter().map(|row| pack_row(row)).collect())
                .unwrap_or_default(),
            _ => eg
                .mirror
                .get(&a.func)
                .map(|set| set.iter().map(|row| pack_row(row)).collect())
                .unwrap_or_default(),
        };
        snapshot.push(rows);
    }

    // Clone the plan pieces the circuit closure needs (it must be `'static`).
    let atoms: Vec<Vec<Slot>> = plan.atoms.iter().map(|a| a.slots.clone()).collect();
    let var_col = plan.var_col.clone();
    let guards = plan.guards.clone();
    let n_vars = plan.var_order.len();

    let (handle, (inputs, output)) = RootCircuit::build(move |root| {
        let mut inputs: Vec<RelHandles> = Vec::with_capacity(n_atoms);
        let mut streams: Vec<Stream<RootCircuit, OrdZSet<RelRow>>> = Vec::with_capacity(n_atoms);
        for _ in 0..n_atoms {
            let (stream, input) = root.add_input_zset::<RelRow>();
            inputs.push(RelHandles { input });
            streams.push(stream);
        }

        let out = build_join_stream(&streams, &atoms, &var_col, &guards, n_vars)?;
        Ok((inputs, out.output()))
    })?;

    // Push each occurrence's snapshot as the circuit input delta (all +1).
    for (ord, rows) in snapshot.iter().enumerate() {
        for row in rows {
            inputs[ord].input.push(*row, 1);
        }
    }

    // One round.
    handle.transaction()?;

    // Read the consolidated binding rows (positive weight = present).
    let consolidated = output.consolidate();
    let mut bindings: Vec<Vec<u32>> = Vec::new();
    for (row, (), w) in consolidated.iter() {
        let w: ZWeight = w;
        if w > 0 {
            bindings.push((0..n_vars).map(|i| get_col(&row, i)).collect());
        }
    }
    Ok(bindings)
}

/// Build the DBSP stream that produces the join's binding rows.
///
/// Left-deep join: start from the first atom's bindings, then for each
/// subsequent atom join on the variables already bound (shared variables). The
/// binding row carries all canonical variables bound so far (others 0). `!=`
/// guards are applied as `filter`s as soon as both operands are bound.
fn build_join_stream(
    streams: &[Stream<RootCircuit, OrdZSet<RelRow>>],
    atoms: &[Vec<Slot>],
    var_col: &HashMap<u32, usize>,
    guards: &[Cond],
    n_vars: usize,
) -> Result<Stream<RootCircuit, OrdZSet<BindRow>>> {
    // `bound`: which canonical variable columns are filled after each atom.
    let mut bound: Vec<bool> = vec![false; n_vars];
    // Track which guards have already been applied (each is applied once, as
    // soon as every binding-row column it reads is bound).
    let mut applied: Vec<bool> = vec![false; guards.len()];

    // Initialize from the first atom: map each row to a binding row.
    let slots0 = &atoms[0];
    let s0 = streams
        .first()
        .ok_or_else(|| anyhow!("join has no atoms"))?
        .clone();
    let vc0 = var_col.clone();
    let slots0c = slots0.clone();
    // A row matches atom 0 if its constant columns agree and repeated variables
    // within the atom agree; bind the variables into the canonical row.
    let mut cur = s0.flat_map(move |r: &RelRow| match bind_atom(r, &slots0c, &vc0) {
        Some(b) => vec![b],
        None => vec![],
    });
    mark_bound(slots0, var_col, &mut bound);
    // Apply any guards now satisfiable.
    cur = apply_guards(cur, guards, &bound, &mut applied);

    // Join successive atoms.
    for (i, slots) in atoms.iter().enumerate().skip(1) {
        let s = streams[i].clone();

        // Shared variables = atom variables already bound.
        let shared: Vec<u32> = atom_vars(slots)
            .into_iter()
            .filter(|v| var_col.get(v).map(|&c| bound[c]).unwrap_or(false))
            .collect();

        let vc = var_col.clone();
        let slotsc = slots.clone();

        if shared.is_empty() {
            // No shared bound variable: cartesian product. Index both sides by
            // a unit key so `join` produces the full cross product.
            let bound_now = bound.clone();
            let left = cur.map_index(|b: &BindRow| ((), *b));
            let vc2 = vc.clone();
            let right = s.map_index(move |r: &RelRow| ((), *r));
            cur = left
                .join(&right, move |_k, b: &BindRow, r: &RelRow| {
                    merge_atom_into(b, r, &slotsc, &vc2, &bound_now)
                })
                .flat_map(|o: &Option<BindRow>| match o {
                    Some(b) => vec![*b],
                    None => vec![],
                });
        } else {
            // Hash-join on the shared variable columns.
            let shared_cols_left: Vec<usize> = shared.iter().map(|v| var_col[v]).collect();
            let scl = shared_cols_left.clone();
            let left = cur.map_index(move |b: &BindRow| (join_key(b, &scl, get_col), *b));

            // For the right side, the key is read from the row's columns that
            // correspond to the shared variables (the atom's slot positions).
            let shared_atom_cols: Vec<usize> = shared
                .iter()
                .map(|v| {
                    slots
                        .iter()
                        .position(|s| matches!(s, Slot::Var(x) if x == v))
                        .expect("shared var present in atom")
                })
                .collect();
            let sac = shared_atom_cols.clone();
            let right = s.map_index(move |r: &RelRow| (join_key(r, &sac, get_col), *r));

            let bound_now = bound.clone();
            let vc2 = vc.clone();
            cur = left
                .join(&right, move |_k, b: &BindRow, r: &RelRow| {
                    merge_atom_into(b, r, &slotsc, &vc2, &bound_now)
                })
                .flat_map(|o: &Option<BindRow>| match o {
                    Some(b) => vec![*b],
                    None => vec![],
                });
        }

        mark_bound(slots, var_col, &mut bound);
        cur = apply_guards(cur, guards, &bound, &mut applied);
    }

    Ok(cur)
}

/// Apply the [`PrimStep`]s (Stage C call-prims) to the joined binding stream as
/// a chain of `flat_map`s, in body order. Each step evaluates its pure prim
/// through the shared [`PrimEngine`] (locking the engine per row), materializes
/// the interned result into the binding row, or prunes the row (the prim failed,
/// or an asserted return value mismatched). A `flat_map` returning `vec![]`
/// drops the row; `vec![row]` keeps it.
fn apply_steps(
    mut cur: Stream<RootCircuit, OrdZSet<BindRow>>,
    steps: &[PrimStep],
    engine: &PrimEngine,
) -> Stream<RootCircuit, OrdZSet<BindRow>> {
    for step in steps {
        let step = step.clone();
        let engine = engine.clone();
        cur = cur.flat_map(move |row: &BindRow| {
            // Gather the prim's argument reps from the binding row / consts.
            let argv: Vec<Value> = step
                .args
                .iter()
                .map(|a| match a {
                    ArgSrc::Col(c) => Value::new(get_col(row, *c)),
                    ArgSrc::Const(v) => Value::new(*v),
                })
                .collect();
            // Re-evaluate the real prim under the engine lock. `None` ⇒ the prim
            // failed (e.g. `!=` of equal args, `guard` of false) ⇒ prune.
            let Some(result) = engine.eval(step.id, &argv) else {
                return Vec::new();
            };
            match step.ret {
                PrimRet::Bind(col) => vec![set_col(*row, col, result.rep())],
                PrimRet::AssertCol(col) => {
                    if get_col(row, col) == result.rep() {
                        vec![*row]
                    } else {
                        Vec::new()
                    }
                }
                PrimRet::AssertConst(c) => {
                    if c == result.rep() {
                        vec![*row]
                    } else {
                        Vec::new()
                    }
                }
            }
        });
    }
    cur
}

/// Variables appearing in an atom (in column order, may repeat).
fn atom_vars(slots: &[Slot]) -> Vec<u32> {
    let mut out = Vec::new();
    for s in slots {
        if let Slot::Var(v) = s {
            if !out.contains(v) {
                out.push(*v);
            }
        }
    }
    out
}

/// Mark the canonical columns of an atom's variables as bound.
fn mark_bound(slots: &[Slot], var_col: &HashMap<u32, usize>, bound: &mut [bool]) {
    for s in slots {
        if let Slot::Var(v) = s {
            if let Some(&c) = var_col.get(v) {
                bound[c] = true;
            }
        }
    }
}

/// Build a `u64`/`u128`-free join key from selected columns. DBSP requires the
/// key be `DBData` (`Vec<u32>` is not), so the key reuses the fixed-width row
/// tuple with the selected columns packed into the low slots (others 0).
fn join_key<R>(r: &R, cols: &[usize], get: fn(&R, usize) -> u32) -> BindRow {
    let mut a = [0u32; JOIN_WIDTH];
    for (i, &c) in cols.iter().enumerate() {
        a[i] = get(r, c);
    }
    arr_to_row(a)
}

/// Match the first atom's row against its slots and produce the initial binding
/// row, or `None` if a constant / repeated-variable constraint fails.
fn bind_atom(r: &RelRow, slots: &[Slot], var_col: &HashMap<u32, usize>) -> Option<BindRow> {
    let mut out = [0u32; JOIN_WIDTH];
    // Track values bound to each canonical var within this row to enforce
    // repeated-variable equality.
    let mut local: HashMap<u32, u32> = HashMap::new();
    for (i, s) in slots.iter().enumerate() {
        let val = get_col(r, i);
        match s {
            Slot::Const(c) => {
                if *c != val {
                    return None;
                }
            }
            Slot::Var(v) => {
                if let Some(&prev) = local.get(v) {
                    if prev != val {
                        return None;
                    }
                } else {
                    local.insert(*v, val);
                    out[var_col[v]] = val;
                }
            }
        }
    }
    Some(arr_to_row(out))
}

/// Merge atom row `r` into binding `b` (already-bound columns must agree with
/// the row; previously-unbound atom variables are written). Returns `None` if a
/// constant / shared-variable / repeated-variable constraint fails.
fn merge_atom_into(
    b: &BindRow,
    r: &RelRow,
    slots: &[Slot],
    var_col: &HashMap<u32, usize>,
    bound: &[bool],
) -> Option<BindRow> {
    let mut out = *b;
    let mut local: HashMap<u32, u32> = HashMap::new();
    for (i, s) in slots.iter().enumerate() {
        let val = get_col(r, i);
        match s {
            Slot::Const(c) => {
                if *c != val {
                    return None;
                }
            }
            Slot::Var(v) => {
                // Repeated variable within this atom must agree.
                if let Some(&prev) = local.get(v) {
                    if prev != val {
                        return None;
                    }
                    continue;
                }
                local.insert(*v, val);
                let c = var_col[v];
                if bound[c] {
                    // Already bound by a previous atom: must agree.
                    if get_col(&out, c) != val {
                        return None;
                    }
                } else {
                    out = set_col(out, c, val);
                }
            }
        }
    }
    Some(out)
}

#[inline]
fn set_col(r: BindRow, i: usize, v: u32) -> BindRow {
    let mut a = with_row_fields!(read &r);
    a[i] = v;
    arr_to_row(a)
}

/// Apply every not-yet-applied guard whose binding-row columns are all bound,
/// as a `filter` on the binding stream. Each guard is applied exactly once (the
/// `applied` flags persist across atoms).
fn apply_guards(
    stream: Stream<RootCircuit, OrdZSet<BindRow>>,
    guards: &[Cond],
    bound: &[bool],
    applied: &mut [bool],
) -> Stream<RootCircuit, OrdZSet<BindRow>> {
    let mut cur = stream;
    for (i, g) in guards.iter().enumerate() {
        if applied[i] {
            continue;
        }
        let mut cols = Vec::new();
        g.cols(&mut cols);
        // Only apply once every column the guard reads is bound.
        if !cols.iter().all(|&c| bound[c]) {
            continue;
        }
        applied[i] = true;
        let g = g.clone();
        cur = cur.filter(move |row: &BindRow| g.eval(row));
    }
    cur
}

// ===========================================================================
// PersistentJoin — the persistent, delta-fed body join (Stage A of #23)
// ===========================================================================
//
// Unlike [`run_join_with`] (which builds a fresh circuit and pushes the FULL
// relations every call — O(state)), `PersistentJoin` builds the circuit ONCE
// and is fed only per-transaction RELATION DELTAS. DBSP's `join` is incremental
// and maintains the integrals of its inputs internally, so feeding `δR` across
// transactions yields the full seminaive join (`δR⋈S + R⋈δS + δR⋈δS`)
// automatically — no manual per-delta-atom loop, and no host `seen` set. The
// circuit's integrals ARE the seminaive bookkeeping.
//
// The output is read WITHOUT `integrate()`, so each `step` returns exactly the
// *binding delta* produced by that transaction: positive-weight rows are new
// satisfying assignments, negative-weight rows are assignments retracted because
// a body row was retracted (deletion at the transaction boundary — handled
// natively by DBSP's signed weights).

/// A persistent, delta-fed body join for one rule. Built once via
/// [`PersistentJoin::build`]; driven across iterations via [`PersistentJoin::step`].
pub struct PersistentJoin {
    handle: CircuitHandle,
    /// One input handle per atom occurrence (in `plan.atoms` order).
    inputs: Vec<ZSetHandle<RelRow>>,
    /// The INTEGRATED (accumulated) distinct binding set. We read the full
    /// accumulated set each step and diff it against [`PersistentJoin::prev`]
    /// in Rust to get the new matches. Reading non-integrated per-transaction
    /// deltas under-reports on large single-transaction batches (a DBSP
    /// delta-read quirk), so we mirror the proven `rebuild_circuit` pattern of
    /// reading the integral and diffing host-side.
    output: OutputHandle<OrdZSet<BindRow>>,
    /// `func` → the atom-occurrence indices that read it, so a relation's delta
    /// is fanned out to every occurrence (correct for self-joins).
    occ_of_func: HashMap<FunctionId, Vec<usize>>,
    /// Number of canonical body variables (binding-row width in use).
    n_vars: usize,
}

impl PersistentJoin {
    /// Build the persistent circuit for `plan` ONCE. The circuit retains the
    /// integral of every body relation across transactions.
    ///
    /// `engine` is the shared primitive engine (Stage C): captured into the
    /// circuit's call-prim closures so value-computing body prims evaluate
    /// on-circuit. It is only used if `plan.steps` is non-empty (a rule with a
    /// genuine value prim); the in-join rep fast path never touches it.
    pub fn build(plan: &JoinPlan, engine: &PrimEngine) -> Result<PersistentJoin> {
        let n_atoms = plan.atoms.len();
        let atoms: Vec<Vec<Slot>> = plan.atoms.iter().map(|a| a.slots.clone()).collect();
        let var_col = plan.var_col.clone();
        let guards = plan.guards.clone();
        let steps = plan.steps.clone();
        let engine = engine.clone();
        let n_vars = plan.var_order.len();

        let (handle, (inputs, output)) = RootCircuit::build(move |root| {
            let mut inputs: Vec<ZSetHandle<RelRow>> = Vec::with_capacity(n_atoms);
            let mut streams: Vec<Stream<RootCircuit, OrdZSet<RelRow>>> =
                Vec::with_capacity(n_atoms);
            for _ in 0..n_atoms {
                let (stream, input) = root.add_input_zset::<RelRow>();
                inputs.push(input);
                // `.distinct()` makes each input set-semantic (weights 0/1),
                // matching egglog relations.
                streams.push(stream.distinct());
            }
            let out = build_join_stream(&streams, &atoms, &var_col, &guards, n_vars)?;
            // Stage C: apply on-circuit call-prim steps (value prims) after the
            // join, evaluating the real prim through the shared engine.
            let out = apply_steps(out, &steps, &engine);
            // Non-integrated, distinct binding stream: each `step()` yields this
            // tick's binding DELTA directly. (We drive the circuit with `step()`,
            // not `transaction()`: a transaction is a *sequence* of steps for one
            // logical tick, and the non-integrated output handle only reflects the
            // last internal step — which silently truncates large batches.)
            Ok((inputs, out.distinct().output()))
        })?;

        let mut occ_of_func: HashMap<FunctionId, Vec<usize>> = HashMap::new();
        for (ord, a) in plan.atoms.iter().enumerate() {
            occ_of_func.entry(a.func).or_default().push(ord);
        }

        Ok(PersistentJoin {
            handle,
            inputs,
            output,
            occ_of_func,
            n_vars,
        })
    }

    /// Feed one round of relation deltas and run a single transaction, returning
    /// the resulting binding delta as `(binding_row, weight)` pairs. `deltas`
    /// maps a body relation to its `±`-weighted changed rows since the previous
    /// `step`. Relations not in this rule's body are ignored; a relation read by
    /// several atoms has its delta fanned out to each occurrence.
    pub fn step(
        &mut self,
        deltas: &HashMap<FunctionId, Vec<(Vec<u32>, ZWeight)>>,
    ) -> Result<Vec<(Vec<u32>, ZWeight)>> {
        let mut pushed_any = false;
        for (func, rows) in deltas {
            if let Some(occs) = self.occ_of_func.get(func) {
                for &ord in occs {
                    for (row, w) in rows {
                        self.inputs[ord].push(pack_row(row), *w);
                        pushed_any = true;
                    }
                }
            }
        }
        // No input change ⇒ no new bindings; skip the transaction entirely. This
        // short-circuits the many no-op rebuild-saturation re-runs.
        if !pushed_any {
            return Ok(Vec::new());
        }

        // Drive the transaction lifecycle manually and ACCUMULATE the
        // non-integrated output across all commit steps. A `transaction()`
        // processes a batch over several internal steps and the non-integrated
        // handle only retains the *last* step's delta — silently truncating
        // large batches (the `@uf` rebuild bug). Summing per-step deltas recovers
        // the complete tick delta at O(delta) cost (no full-integral re-read).
        self.handle.start_transaction()?;
        self.handle.start_commit_transaction()?;
        let mut acc: HashMap<Vec<u32>, ZWeight> = HashMap::new();
        while !self.handle.is_commit_complete() {
            self.handle.step()?;
            for (row, (), w) in self.output.consolidate().iter() {
                let w: ZWeight = w;
                if w != 0 {
                    let key: Vec<u32> = (0..self.n_vars).map(|i| get_col(&row, i)).collect();
                    *acc.entry(key).or_insert(0) += w;
                }
            }
        }
        Ok(acc.into_iter().filter(|(_, w)| *w != 0).collect())
    }
}

#[cfg(test)]
mod persistent_tests {
    use super::*;
    use egglog_numeric_id::NumericId;

    /// Build a 2-atom self-join plan `R(x,y), R(y,z)` (transitive-closure hop)
    /// over a single relation, without going through `plan_join` (which needs a
    /// full `RuleIr`). Same-module access to the private `JoinPlan` fields.
    fn tc_plan(func: FunctionId) -> JoinPlan {
        let mut var_col = HashMap::new();
        var_col.insert(0u32, 0usize); // x
        var_col.insert(1u32, 1usize); // y
        var_col.insert(2u32, 2usize); // z
        JoinPlan {
            var_order: vec![0, 1, 2],
            var_col,
            atoms: vec![
                AtomPlan {
                    func,
                    slots: vec![Slot::Var(0), Slot::Var(1)], // R(x, y)
                },
                AtomPlan {
                    func,
                    slots: vec![Slot::Var(1), Slot::Var(2)], // R(y, z)
                },
            ],
            guards: vec![],
            steps: vec![],
        }
    }

    /// An engine with no prims registered — the TC plan has no `steps`, so the
    /// engine is never invoked; this just satisfies `build`'s signature.
    fn empty_engine() -> PrimEngine {
        PrimEngine::new(egglog_core_relations::Database::new())
    }

    fn delta(
        func: FunctionId,
        rows: &[(&[u32], ZWeight)],
    ) -> HashMap<FunctionId, Vec<(Vec<u32>, ZWeight)>> {
        let mut m = HashMap::new();
        m.insert(func, rows.iter().map(|(r, w)| (r.to_vec(), *w)).collect());
        m
    }

    /// The load-bearing property: after seeding the relation, a SECOND step fed
    /// only the new edge produces ONLY the new bindings — i.e. the join is
    /// incremental/seminaive (O(delta)), not a full re-evaluation.
    #[test]
    fn persistent_join_is_incremental() {
        let f = FunctionId::new(0);
        let plan = tc_plan(f);
        let mut pj = PersistentJoin::build(&plan, &empty_engine()).expect("build persistent join");

        // Step 1: seed edges (1,2) and (2,3). The only TC hop is x=1,y=2,z=3.
        let out1 = pj
            .step(&delta(f, &[(&[1, 2], 1), (&[2, 3], 1)]))
            .expect("step 1");
        assert_eq!(out1, vec![(vec![1, 2, 3], 1)], "first hop");

        // Step 2: add only the NEW edge (3,4). The incremental join must emit
        // ONLY the new binding x=2,y=3,z=4 — not re-derive (1,2,3).
        let out2 = pj.step(&delta(f, &[(&[3, 4], 1)])).expect("step 2");
        assert_eq!(out2, vec![(vec![2, 3, 4], 1)], "only the new hop");

        // Step 3: retract edge (2,3). The binding (1,2,3) used it as R(y,z) and
        // (2,3,4) used it as R(x,y); both retract (negative weight) — deletion at
        // the transaction boundary, handled by signed weights.
        let mut out3 = pj.step(&delta(f, &[(&[2, 3], -1)])).expect("step 3");
        out3.sort();
        assert_eq!(
            out3,
            vec![(vec![1, 2, 3], -1), (vec![2, 3, 4], -1)],
            "retraction propagates"
        );
    }
}
