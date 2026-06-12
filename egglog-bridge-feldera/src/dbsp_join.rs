//! DBSP-backed body join for the Feldera backend (#23 Stage C complete).
//!
//! This module runs a rule's **relational body join on DBSP's dataflow engine**
//! — the paper's core technical contribution. The persistent per-rule circuit
//! ([`PersistentJoin`]) is the ONLY join path: there is no host nested-loop
//! fallback. A genuinely-ineligible rule PANICS (see [`plan_join`]).
//!
//! ## What runs on DBSP
//!
//! [`PersistentJoin`] builds a **non-recursive** DBSP circuit ONCE per rule that
//! computes the join of the body's table atoms (multi-atom, left-deep), with
//! `!=` guards applied as DBSP `filter` operators *inside* the dataflow. It is
//! fed only the per-relation signed DELTA each iteration; the incremental join +
//! integral do the seminaive bookkeeping and retraction natively. The circuit's
//! output is a z-set of **binding rows** (a fixed-width [`Row`] bucket — 8, 16 or
//! 32 columns, chosen per ruleset) holding the rule's body variables in a fixed
//! canonical order.
//!
//! ## Pure body prims run ON-CIRCUIT
//!
//! Value-computing body prims are evaluated on-circuit through the shared
//! [`crate::PrimEngine`] (an `Arc<Mutex<Database>>` captured into the circuit
//! closures), so the whole body join — guards AND value prims — runs on DBSP.
//! Recognized rep-comparison prims are inlined without a lock; everything else
//! is evaluated by re-running the REAL prim under the engine lock (the call-prim
//! path). Only head application (`set`/`delete`/`lookup`/`union` + FD-merge)
//! stays host-side (`interpret::run_iteration` / `apply_head`).
//!
//! ## Eligibility / the row-width cap
//!
//! DBSP rows must be fixed-arity `DBData` (rkyv-archivable); we use a fixed-width
//! [`Row`] bucket per ruleset (the smallest of 8/16/32 columns that fits — see
//! [`pick_width`]). A rule is eligible (does not panic) iff:
//!   - it has at least one table atom (atom-less rules are fired once by the
//!     caller, not planned here);
//!   - every table atom has arity <= [`JOIN_WIDTH`] (the widest bucket, 32);
//!   - the rule's body uses <= [`JOIN_WIDTH`] distinct variables (binding row);
//!   - every body primitive is PURE. A recognized rep-comparison prim (`!=`,
//!     `bool-!=`, `or`, `guard`, `ordering-min/max`) is inlined when its operand
//!     columns support the relevant rep-arithmetic (ORDERING prims require `Id`
//!     columns; EQUALITY prims also accept the injectively-interned base types
//!     `bool`/`i64`/`string`). A pure prim that is NOT rep-inlinable
//!     (base-value `ordering-min/max`, f64 `!=`, or any unrecognized value prim
//!     like `+`/`int-div`/`string-concat`) is NOT rejected — it ROUTES to the
//!     on-circuit call-prim engine, which evaluates the real prim semantics on
//!     actual values. Only an IMPURE prim (or a shape above the row-width cap)
//!     makes the rule ineligible — and that PANICS.

use anyhow::{anyhow, Result};
use dbsp::{CircuitHandle, OrdZSet, OutputHandle, RootCircuit, Stream, ZSetHandle, ZWeight};
use egglog_backend_trait::{ColumnTy, ExternalFunctionId, FunctionId, Value};
use egglog_numeric_id::NumericId;
use hashbrown::HashMap;

use crate::compile::{BodyOp, RuleIr, Slot};
use crate::{EGraph, PrimEngine};

// ---------------------------------------------------------------------------
// PROF phase attribution (gated FELDERA_PROFILE). Static nanosecond counters
// for the three phases of `FusedJoin::step`: input-feed, transaction-step loop,
// and per-rule output consolidate/read. Read+printed by the EGraph Drop.
// ---------------------------------------------------------------------------
use std::sync::atomic::{AtomicU64, Ordering};
pub(crate) static PROF_FEED_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_STEP_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_READ_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_STEP_CALLS: AtomicU64 = AtomicU64::new(0);
#[inline]
fn add_ns(c: &AtomicU64, d: std::time::Duration) {
    c.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
}

/// Hard upper bound on distinct body variables / atom columns the DBSP join
/// supports (the WIDEST fixed-arity `DBData` row bucket).
///
/// The circuit row is no longer a single fixed width: the binding/relation row
/// is one of several [`declare_tuples!`]-generated buckets (8 / 16 / 32 columns),
/// and [`FusedJoin::build`] picks the SMALLEST bucket that fits the ruleset's
/// actual max binding-row width (see [`pick_width`]). Since every DBSP operator
/// (join, map_index, integral consolidation, z-set hashing/ordering) copies,
/// hashes and compares the FULL fixed-width row, a ruleset whose rules bind only
/// a handful of variables (the common case: math-microbenchmark's hot ruleset
/// peaks at 7 vars) flows through 8-wide (32-byte) rows instead of 32-wide
/// (128-byte) ones — 4x less per-row memory traffic across the whole circuit.
///
/// The widest bucket is still 32: a suite-wide survey put the peak among PASSING
/// programs at ~23 distinct body variables (`cykjson` `check_facts`, the
/// `const-prop` value rules). A rule wider than 32 PANICS as ineligible
/// (`plan_join`'s "too many vars" reject); the only suite program that trips it
/// is `luminal-llama`'s `@rebuild_rule34` (35 vars), which is independently
/// unsupported. Widening to 48 regressed the N=11 run to OOM, so 32 stays the
/// ceiling. The DBData derive stack (paste/rkyv/size_of/serde/derive_more) is
/// pinned in Cargo.toml to dbsp 0.150's versions.
pub const JOIN_WIDTH: usize = MAX_WIDTH;

/// The width buckets available, smallest first. A ruleset is lowered into the
/// smallest bucket >= its max binding-row width.
const WIDTH_BUCKETS: [usize; 3] = [8, 16, 32];
/// The widest bucket (the eligibility cap).
const MAX_WIDTH: usize = 32;

/// Pick the smallest width bucket that holds `needed` columns. Returns
/// [`MAX_WIDTH`] if `needed` exceeds every bucket (the caller — `plan_join` —
/// has already rejected `needed > MAX_WIDTH`).
fn pick_width(needed: usize) -> usize {
    WIDTH_BUCKETS
        .iter()
        .copied()
        .find(|&w| w >= needed)
        .unwrap_or(MAX_WIDTH)
}

// Declare the bucket `DBData` tuples via dbsp's own macro, so the generated
// trait impls (rkyv `Archive`, `SizeOf`, `MulByRef`, `HasZero`, …) match dbsp's
// stock tuples exactly. The macro binds its generic type params as value-
// position idents (`T1`..), which trips `non_snake_case`; the names are dbsp's
// macro contract, not ours.
#[allow(non_snake_case)]
mod tup_row {
    dbsp::declare_tuples! {
        TupRow8<T1, T2, T3, T4, T5, T6, T7, T8>,
        TupRow16<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16>,
        TupRow32<
            T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16,
            T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32
        >,
    }
}
pub(crate) use tup_row::{TupRow16, TupRow32, TupRow8};

/// A fixed-width binding/relation row flowing through the DBSP circuit, abstract
/// over the bucket width so every circuit-building function is monomorphized per
/// chosen width. `bind[i]` is the value of the rule's `i`-th canonical body
/// variable (0 if not yet bound / unused).
///
/// The bounds are exactly what DBSP's stream operators require of a z-set element
/// (`DBData`) plus `Copy` (the join/map_index closures dereference-copy rows).
pub trait Row: dbsp::DBData + Copy {
    /// Number of `u32` columns in this row.
    const WIDTH: usize;
    /// Build a row from a slice of column values (0-padded to [`Row::WIDTH`]).
    /// The slice is `<= WIDTH` (enforced by `plan_join`'s width cap + bucket
    /// pick).
    fn pack(vals: &[u32]) -> Self;
    /// Read column `i`.
    fn get(&self, i: usize) -> u32;
    /// Return a copy of the row with column `i` set to `v`.
    fn set(self, i: usize, v: u32) -> Self;
}

/// Generate the [`Row`] impl for a `TupRow{N}` bucket of width `$w` with the
/// positional field idents `$f`. Build/read touch every field positionally so
/// the compiler sees a fixed-size struct (no array indexing through a `Vec`).
macro_rules! impl_row {
    ($ty:ident, $w:expr, $($f:tt),+) => {
        impl Row for $ty<$( impl_row!(@u32 $f) ),+> {
            const WIDTH: usize = $w;
            #[inline]
            fn pack(vals: &[u32]) -> Self {
                let mut a = [0u32; $w];
                for (i, v) in vals.iter().enumerate() {
                    a[i] = *v;
                }
                let mut k = 0usize;
                $( let $f = { let x = a[k]; k += 1; x }; )+
                let _ = k;
                $ty($($f),+)
            }
            #[inline]
            fn get(&self, i: usize) -> u32 {
                let $ty($($f),+) = self;
                let mut k = 0usize;
                $( if i == k { return *$f; } k += 1; )+
                let _ = k;
                0
            }
            #[inline]
            fn set(mut self, i: usize, v: u32) -> Self {
                let $ty($($f),+) = &mut self;
                let mut k = 0usize;
                $( if i == k { *$f = v; return self; } k += 1; )+
                let _ = k;
                self
            }
        }
    };
    (@u32 $f:tt) => { u32 };
}

impl_row!(TupRow8, 8, a0, a1, a2, a3, a4, a5, a6, a7);
impl_row!(
    TupRow16, 16, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15
);
impl_row!(
    TupRow32, 32, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,
    a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31
);

/// The width-8 binding/relation row (32 bytes).
type Row8 = TupRow8<u32, u32, u32, u32, u32, u32, u32, u32>;
/// The width-16 binding/relation row (64 bytes).
type Row16 =
    TupRow16<u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32>;
/// The width-32 binding/relation row (128 bytes) — the eligibility-cap bucket.
type Row32 = TupRow32<
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
>;

/// Build a fixed-width row from a slice of column values (0-padded). Slices
/// longer than the bucket width are rejected upstream by [`plan_join`].
#[inline]
fn pack_row<R: Row>(vals: &[u32]) -> R {
    R::pack(vals)
}

/// Read column `i` of a fixed-width row.
#[inline]
fn get_col<R: Row>(r: &R, i: usize) -> u32 {
    r.get(i)
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
    fn eval<R: Row>(&self, row: &R) -> u32 {
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
    fn eval<R: Row>(&self, row: &R) -> bool {
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

/// Build the body-join [`JoinPlan`] for `rule` on the persistent DBSP engine, or
/// return `Err(reason)` if the rule is genuinely ineligible (the caller then
/// panics — the persistent circuit is the ONLY join path, there is no host
/// fallback). Ineligibility means an IMPURE body prim, an arity/var count above
/// the fixed [`JOIN_WIDTH`] row, or a structural shape the lowering cannot
/// express. PURE prims are never a rejection cause: rep-comparable ones inline
/// into the join (the no-lock fast path), all others (base-value
/// `ordering-min/max`, f64 `!=`, arbitrary value prims like `+`/`int-div`) route
/// to the on-circuit call-prim path that re-evaluates the REAL prim on actual
/// values through the shared engine. Atom-less rules are handled by the caller
/// before this is reached.
///
/// ## Correctness gating (rep-arithmetic validity)
///
/// Inlined prim closures operate on the INTERNED `u32` rep, not the logical
/// value. `ordering-min/max` and `<`/`==`/`!=` on the rep are only valid when
/// the rep order/equality matches the logical one:
///   - `ColumnTy::Id` columns: the rep IS the union-find id, so min/max = the
///     leader choice and equality is identity — VALID (inlined, no lock).
///   - bool columns: distinct logical values get distinct interned reps, so
///     equality / `!=` are VALID (but ordering is meaningless — we never inline
///     ordering on bool).
///   - i64 / string columns: interning is injective (distinct values get
///     distinct reps, and `intern(a) == intern(b)` iff `a == b`), so equality /
///     `!=` are VALID. Ordering is NOT — the rep is a handle whose order ≠ value
///     order — so base-value `ordering-min/max` ROUTES to the call-prim engine.
///   - f64 columns: an interned `NaN` rep equals itself, but IEEE says
///     `NaN != NaN`, so rep-equality does NOT match value-`!=`. f64 `!=`
///     therefore ROUTES to the call-prim engine (which evaluates real IEEE `!=`).
///   - any other base value (BigInt/BigRat/…): conservatively non-rep-inlinable;
///     any recognized prim touching such a column also routes to the call-prim
///     engine, which evaluates the real prim semantics on the actual values.
pub fn plan_join(eg: &EGraph, rule: &RuleIr) -> Result<JoinPlan, String> {
    // DIAGNOSTIC ONLY (gated `FELDERA_DBG_PLAN`): name the rejected rule and the
    // blocking reason. Purely env-gated; never affects behavior. The reason is
    // also returned so the caller's panic carries it.
    macro_rules! reject {
        ($($reason:tt)*) => {{
            let reason = format!($($reason)*);
            if std::env::var("FELDERA_DBG_PLAN").is_ok() {
                #[allow(clippy::disallowed_macros)]
                {
                    eprintln!("[DBG_PLAN] reject rule={:?} reason={}", rule.name, reason);
                }
            }
            return Err(reason);
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

    // PRE-PASS (Stage C generalization): scan the body's table atoms to populate
    // `var_col` / `var_kind` BEFORE deciding the prim-lowering mode, so the mode
    // decision can consult each operand's rep-arithmetic kind. (The full atom
    // processing below re-walks atoms; this pre-pass only fills the maps the mode
    // decision reads — it does not push `atoms`, so the canonical work is not
    // duplicated.) Variable column order from this pre-pass is the canonical body
    // order, the same order the main loop would assign.
    for op in &rule.body {
        if let BodyOp::Atom(atom) = op {
            for (col, s) in atom.slots.iter().enumerate() {
                if let Slot::Var(v) = s {
                    see_var(*v, &mut var_order, &mut var_col);
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
        }
    }

    // Decide the prim-lowering MODE for this rule (Stage C). Three outcomes:
    //   - in-join rep fast path (no lock): every body prim is a recognized
    //     rep-comparison (`!=`/`bool-!=`/`or`/`guard`/`ordering-min/max`) AND
    //     every operand is rep-inlinable on its column kind (Id for ordering,
    //     Id/bool/i64/string for equality);
    //   - on-circuit call-prim path (locks per row, evaluates the REAL prim on
    //     actual values): ANY body prim is either unrecognized (a genuine value
    //     prim like `+`/`int-div`/`string-concat`) OR recognized-but-not-
    //     rep-inlinable (base-value `ordering-min/max`, f64 `!=`, …). The shared
    //     engine handles these correctly (IEEE NaN for f64 `!=`, value-order for
    //     base-value ordering), so we route there INSTEAD of rejecting to host;
    //   - ineligible (host / panic): a body prim is impure (not registered pure).
    //
    // To decide rep-inlinability we track a lightweight `RepKind` for prim-output
    // vars too: `bool-!=`/`or`/`guard` produce a bool (Equality), and
    // `ordering-min/max` propagate their operand kind. If any recognized prim's
    // operand cannot be resolved to an inlinable kind, the WHOLE rule takes the
    // call-prim path (never a host fallback for a pure prim).
    let mut use_calls = false;
    // Operand-kind probe for the mode decision (does not mutate plan state).
    let mut prim_kind: HashMap<u32, Option<RepKind>> = HashMap::new();
    let operand_kind = |s: &Slot,
                        prim_kind: &HashMap<u32, Option<RepKind>>,
                        var_kind: &HashMap<u32, RepKind>|
     -> Option<RepKind> {
        match s {
            // A const is rep-inlinable for equality (its rep is itself); treat as
            // Equality (the weakest kind that still permits `!=`/`==`).
            Slot::Const(_) => Some(RepKind::Equality),
            Slot::Var(v) => {
                if let Some(k) = prim_kind.get(v) {
                    return *k;
                }
                var_kind.get(v).copied()
            }
        }
    };
    for op in &rule.body {
        if let BodyOp::Prim { id, args, ret } = op {
            let name = eg.external_funcs.name(*id);
            if !is_recognized(name) {
                // An UNrecognized prim goes to the on-circuit call-prim engine,
                // which RE-EVALUATES the real prim — so it must be PURE
                // (idempotent under re-evaluation). Recognized rep-comparison
                // prims (`!=`/`bool-!=`/`or`/`guard`/`ordering-min/max`) are
                // known-pure by construction and inlined by name, so they do NOT
                // require the frontend's pure-prim registration (the bridge-level
                // tests build rules without it).
                if !eg.is_pure_prim(*id) {
                    reject!("impure body prim {:?} (not eligible on-circuit)", name);
                }
                use_calls = true;
                continue;
            }
            // Recognized prim: check whether it is rep-inlinable on its operands.
            let need_ordering = matches!(name, Some(ORD_MIN_NAME) | Some(ORD_MAX_NAME));
            let mut inlinable = true;
            let mut out_kind = Some(RepKind::Equality);
            for a in args {
                match operand_kind(a, &prim_kind, &var_kind) {
                    Some(RepKind::Ordering) => {
                        if need_ordering {
                            out_kind = Some(RepKind::Ordering);
                        }
                    }
                    Some(RepKind::Equality) => {
                        if need_ordering {
                            // base-value ordering: rep order != value order ⇒
                            // not rep-inlinable. Route to the call-prim engine.
                            inlinable = false;
                        }
                    }
                    // f64 / unvetted base value (or an unbound operand): not
                    // rep-inlinable ⇒ call-prim engine.
                    None => inlinable = false,
                }
            }
            if !inlinable {
                use_calls = true;
            }
            // Record the recognized prim's output kind for downstream operands
            // (e.g. `or(bool-!=, bool-!=)`). Value prims bind a var.
            if let Slot::Var(rv) = ret {
                prim_kind.insert(*rv, out_kind);
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

    // Note: the call-prim path needs a binding column for EVERY atom-bound
    // variable BEFORE lowering any prim (so a prim whose `ret` is also an
    // atom-bound var lowers to `AssertCol`, not a `Bind` that would clobber the
    // atom's join constraint). The PRE-PASS above already populated `var_col` for
    // all atom-bound vars in canonical body order, so this holds for both modes.

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
                            None => {
                                reject!("call-prim {:?} arg unbound", eg.external_funcs.name(*id))
                            }
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

    Ok(JoinPlan {
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

/// Build the DBSP stream that produces the join's binding rows.
///
/// Left-deep join: start from the first atom's bindings, then for each
/// subsequent atom join on the variables already bound (shared variables). The
/// binding row carries all canonical variables bound so far (others 0). `!=`
/// guards are applied as `filter`s as soon as both operands are bound.
fn build_join_stream<R: Row>(
    streams: &[Stream<RootCircuit, OrdZSet<R>>],
    atoms: &[Vec<Slot>],
    var_col: &HashMap<u32, usize>,
    guards: &[Cond],
    n_vars: usize,
) -> Result<Stream<RootCircuit, OrdZSet<R>>> {
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
    let mut cur = s0.flat_map(move |r: &R| match bind_atom::<R>(r, &slots0c, &vc0) {
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
            let left = cur.map_index(|b: &R| ((), *b));
            let vc2 = vc.clone();
            let right = s.map_index(move |r: &R| ((), *r));
            cur = left
                .join(&right, move |_k, b: &R, r: &R| {
                    merge_atom_into::<R>(b, r, &slotsc, &vc2, &bound_now)
                })
                .flat_map(|o: &Option<R>| match o {
                    Some(b) => vec![*b],
                    None => vec![],
                });
        } else {
            // Hash-join on the shared variable columns.
            let shared_cols_left: Vec<usize> = shared.iter().map(|v| var_col[v]).collect();
            let scl = shared_cols_left.clone();
            let left = cur.map_index(move |b: &R| (join_key::<R>(b, &scl), *b));

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
            let right = s.map_index(move |r: &R| (join_key::<R>(r, &sac), *r));

            let bound_now = bound.clone();
            let vc2 = vc.clone();
            cur = left
                .join(&right, move |_k, b: &R, r: &R| {
                    merge_atom_into::<R>(b, r, &slotsc, &vc2, &bound_now)
                })
                .flat_map(|o: &Option<R>| match o {
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
fn apply_steps<R: Row>(
    mut cur: Stream<RootCircuit, OrdZSet<R>>,
    steps: &[PrimStep],
    engine: &PrimEngine,
) -> Stream<RootCircuit, OrdZSet<R>> {
    for step in steps {
        let step = step.clone();
        let engine = engine.clone();
        cur = cur.flat_map(move |row: &R| {
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
fn join_key<R: Row>(r: &R, cols: &[usize]) -> R {
    let mut a = [0u32; MAX_WIDTH];
    for (i, &c) in cols.iter().enumerate() {
        a[i] = r.get(c);
    }
    R::pack(&a[..R::WIDTH])
}

/// Match the first atom's row against its slots and produce the initial binding
/// row, or `None` if a constant / repeated-variable constraint fails.
fn bind_atom<R: Row>(r: &R, slots: &[Slot], var_col: &HashMap<u32, usize>) -> Option<R> {
    let mut out = [0u32; MAX_WIDTH];
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
    Some(R::pack(&out[..R::WIDTH]))
}

/// Merge atom row `r` into binding `b` (already-bound columns must agree with
/// the row; previously-unbound atom variables are written). Returns `None` if a
/// constant / shared-variable / repeated-variable constraint fails.
fn merge_atom_into<R: Row>(
    b: &R,
    r: &R,
    slots: &[Slot],
    var_col: &HashMap<u32, usize>,
    bound: &[bool],
) -> Option<R> {
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
fn set_col<R: Row>(r: R, i: usize, v: u32) -> R {
    r.set(i, v)
}

/// Apply every not-yet-applied guard whose binding-row columns are all bound,
/// as a `filter` on the binding stream. Each guard is applied exactly once (the
/// `applied` flags persist across atoms).
fn apply_guards<R: Row>(
    stream: Stream<RootCircuit, OrdZSet<R>>,
    guards: &[Cond],
    bound: &[bool],
    applied: &mut [bool],
) -> Stream<RootCircuit, OrdZSet<R>> {
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
        cur = cur.filter(move |row: &R| g.eval(row));
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
    inputs: Vec<ZSetHandle<PjRow>>,
    /// The INTEGRATED (accumulated) distinct binding set. We read the full
    /// accumulated set each step and diff it against [`PersistentJoin::prev`]
    /// in Rust to get the new matches. Reading non-integrated per-transaction
    /// deltas under-reports on large single-transaction batches (a DBSP
    /// delta-read quirk), so we mirror the proven `rebuild_circuit` pattern of
    /// reading the integral and diffing host-side.
    output: OutputHandle<OrdZSet<PjRow>>,
    /// `func` → the atom-occurrence indices that read it, so a relation's delta
    /// is fanned out to every occurrence (correct for self-joins).
    occ_of_func: HashMap<FunctionId, Vec<usize>>,
    /// Number of canonical body variables (binding-row width in use).
    n_vars: usize,
}

/// `PersistentJoin` is the (test-only) per-rule circuit; it is pinned to the
/// widest bucket since it is not on the perf-critical fused path.
type PjRow = Row32;

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
            let mut inputs: Vec<ZSetHandle<PjRow>> = Vec::with_capacity(n_atoms);
            let mut streams: Vec<Stream<RootCircuit, OrdZSet<PjRow>>> =
                Vec::with_capacity(n_atoms);
            for _ in 0..n_atoms {
                let (stream, input) = root.add_input_zset::<PjRow>();
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

// ===========================================================================
// FusedJoin — ONE circuit per RULESET (transaction-count reduction)
// ===========================================================================
//
// `PersistentJoin` builds one circuit PER RULE, and each rule's `step()` runs
// its OWN DBSP `transaction()`. Profiling (`FELDERA_PROFILE`) showed the fixed
// per-transaction circuit-clocking cost (~6.8ms, independent of delta size)
// dominates: with R rules × C `run_rules` calls that is R·C transactions.
//
// `FusedJoin` collapses this to ONE transaction per `run_rules` call by building
// a SINGLE circuit for the whole ruleset: every distinct body relation gets ONE
// shared input z-set, each rule's join is a parallel sub-stream reading those
// shared streams, and each rule keeps its own `OutputHandle<OrdZSet<BindRow>>`
// (so per-rule bindings route to that rule's head unchanged). One `step()`
// pushes every relation delta into the shared inputs ONCE and clocks ONE
// transaction, amortizing the fixed clocking cost across all R rules.
//
// Semantics are identical to R separate `PersistentJoin`s: DBSP's incremental
// join maintains each input relation's integral, and a shared input feeding K
// sub-streams is mathematically the same as K inputs each fed the same delta.
// The output is read per-rule, non-integrated, accumulated across commit steps
// exactly as `PersistentJoin::step` does.

/// The per-rule lowering inside a [`FusedJoinImpl`]: the rule's index (for
/// routing bindings to its head), its canonical var order, and its DBSP output
/// handle.
struct FusedRuleW<R: Row> {
    idx: usize,
    /// Canonical body-variable order (column i holds variable `var_order[i]`).
    var_order: Vec<u32>,
    /// This rule's binding-delta output handle (non-integrated, distinct).
    output: OutputHandle<OrdZSet<R>>,
    /// Number of canonical body variables (binding-row width in use).
    n_vars: usize,
}

/// A fused, delta-fed body join for a WHOLE ruleset, MONOMORPHIZED to a fixed
/// row-width bucket `R`. Built once via [`FusedJoinImpl::build`]; driven via
/// [`FusedJoinImpl::step`] with a SINGLE `transaction()` per call.
pub struct FusedJoinImpl<R: Row> {
    handle: CircuitHandle,
    /// One shared input handle per distinct body relation across all rules.
    inputs: HashMap<FunctionId, ZSetHandle<R>>,
    /// The fused rules, in build order. Each carries its own output handle.
    rules: Vec<FusedRuleW<R>>,
}

/// A fused body join for a whole ruleset, width-dispatched: the row carried
/// through the circuit is the SMALLEST bucket ([`WIDTH_BUCKETS`]) that fits the
/// ruleset's max binding-row width, so narrow rulesets pay 4x less per-row
/// memory/hash/compare cost across every DBSP operator (the integrals are large,
/// and every join/map_index/consolidate touches the full fixed-width row).
pub enum FusedJoin {
    W8(FusedJoinImpl<Row8>),
    W16(FusedJoinImpl<Row16>),
    W32(FusedJoinImpl<Row32>),
}

impl FusedJoin {
    /// Build ONE circuit for the whole ruleset, dispatching to the smallest
    /// width bucket that fits. `plans` pairs each rule's index with its
    /// [`JoinPlan`]; `engine` is the shared primitive engine.
    pub fn build(plans: &[(usize, JoinPlan)], engine: &PrimEngine) -> Result<FusedJoin> {
        let needed = plans
            .iter()
            .map(|(_, p)| p.var_order.len())
            .max()
            .unwrap_or(0);
        // A/B knob: force the widest bucket to reproduce the pre-narrowing
        // single-width behavior for back-to-back comparison under concurrent
        // machine load.
        let needed = if std::env::var("FELDERA_FORCE_W32").is_ok() {
            MAX_WIDTH
        } else {
            needed
        };
        Ok(match pick_width(needed) {
            8 => FusedJoin::W8(FusedJoinImpl::build(plans, engine)?),
            16 => FusedJoin::W16(FusedJoinImpl::build(plans, engine)?),
            _ => FusedJoin::W32(FusedJoinImpl::build(plans, engine)?),
        })
    }

    /// The rule indices this fused circuit serves (build order).
    pub fn rule_indices(&self) -> Vec<usize> {
        match self {
            FusedJoin::W8(f) => f.rule_indices(),
            FusedJoin::W16(f) => f.rule_indices(),
            FusedJoin::W32(f) => f.rule_indices(),
        }
    }

    /// The canonical var order for the fused rule at build position `pos`.
    pub fn var_order_at(&self, pos: usize) -> &[u32] {
        match self {
            FusedJoin::W8(f) => f.var_order_at(pos),
            FusedJoin::W16(f) => f.var_order_at(pos),
            FusedJoin::W32(f) => f.var_order_at(pos),
        }
    }

    /// Feed one round of relation deltas and run a SINGLE transaction, returning
    /// per-rule binding deltas. Dispatches to the chosen-width impl.
    pub fn step(
        &mut self,
        deltas: &HashMap<FunctionId, Vec<(Vec<u32>, ZWeight)>>,
    ) -> Result<Vec<Vec<(Vec<u32>, ZWeight)>>> {
        match self {
            FusedJoin::W8(f) => f.step(deltas),
            FusedJoin::W16(f) => f.step(deltas),
            FusedJoin::W32(f) => f.step(deltas),
        }
    }
}

impl<R: Row> FusedJoinImpl<R> {
    /// Build ONE circuit for the whole ruleset. `plans` pairs each rule's index
    /// with its [`JoinPlan`]; `engine` is the shared primitive engine captured
    /// into call-prim closures (only used by rules with genuine value prims).
    fn build(plans: &[(usize, JoinPlan)], engine: &PrimEngine) -> Result<FusedJoinImpl<R>> {
        // Snapshot the per-rule plan data the circuit closure needs (owned, so
        // the `move` closure is `'static`).
        struct RulePlan {
            idx: usize,
            atoms: Vec<Vec<Slot>>,
            atom_funcs: Vec<FunctionId>,
            var_col: HashMap<u32, usize>,
            var_order: Vec<u32>,
            guards: Vec<Cond>,
            steps: Vec<PrimStep>,
            n_vars: usize,
        }
        let rule_plans: Vec<RulePlan> = plans
            .iter()
            .map(|(idx, plan)| RulePlan {
                idx: *idx,
                atoms: plan.atoms.iter().map(|a| a.slots.clone()).collect(),
                atom_funcs: plan.atoms.iter().map(|a| a.func).collect(),
                var_col: plan.var_col.clone(),
                var_order: plan.var_order.clone(),
                guards: plan.guards.clone(),
                steps: plan.steps.clone(),
                n_vars: plan.var_order.len(),
            })
            .collect();

        // Distinct body relations across all rules → one shared input each.
        let mut funcs: Vec<FunctionId> = Vec::new();
        for rp in &rule_plans {
            for &f in &rp.atom_funcs {
                if !funcs.contains(&f) {
                    funcs.push(f);
                }
            }
        }
        let engine = engine.clone();

        let (handle, (input_vec, rule_outs)) = RootCircuit::build(move |root| {
            // Build ONE distinct'd stream per relation, shared by every atom
            // occurrence (in every rule) that reads it.
            let mut input_vec: Vec<(FunctionId, ZSetHandle<R>)> = Vec::with_capacity(funcs.len());
            let mut rel_stream: HashMap<FunctionId, Stream<RootCircuit, OrdZSet<R>>> =
                HashMap::new();
            // PERF (#23): the per-step input delta is already set-semantic — it
            // is built from a `HashSet` set-difference vs the fed view
            // (`interpret::fused_bindings`), so each row appears at most once with
            // weight ±1 (+1 only when newly present, -1 only when newly absent).
            // The input integral therefore stays 0/1 per row WITHOUT `.distinct()`,
            // making the input distinct (a full integral + per-key consolidation
            // every tick, over the LARGE relation integrals on a fast-growing
            // egraph) pure overhead. Dropping it cut N=10 circuit_step ~7.5% with
            // bit-exact output. Set `FELDERA_KEEP_INPUT_DISTINCT` to restore it.
            let keep_input_distinct = std::env::var("FELDERA_KEEP_INPUT_DISTINCT").is_ok();
            for &f in &funcs {
                let (stream, input) = root.add_input_zset::<R>();
                input_vec.push((f, input));
                let s = if keep_input_distinct {
                    stream.distinct()
                } else {
                    stream
                };
                rel_stream.insert(f, s);
            }

            // For each rule, assemble its per-atom stream vector from the shared
            // relation streams (cloning a stream is just a handle copy in DBSP),
            // build its join + steps, and expose its own output handle.
            let mut rule_outs: Vec<(usize, Vec<u32>, usize, OutputHandle<OrdZSet<R>>)> =
                Vec::with_capacity(rule_plans.len());
            for rp in &rule_plans {
                let streams: Vec<Stream<RootCircuit, OrdZSet<R>>> = rp
                    .atom_funcs
                    .iter()
                    .map(|f| rel_stream[f].clone())
                    .collect();
                let out =
                    build_join_stream(&streams, &rp.atoms, &rp.var_col, &rp.guards, rp.n_vars)?;
                let out = apply_steps(out, &rp.steps, &engine);
                // PERF (#23): the output `.distinct()` is also redundant here.
                // `FusedJoin::step` accumulates each rule's binding deltas into a
                // per-key weight map and the env consumer
                // (`interpret::fused_bindings`) inspects only the SIGN of the net
                // weight (>0 ⇒ one env, ≤0 ⇒ a retraction, net-zero already
                // filtered). distinct would clamp the binding multiplicity to
                // {0,1}, but since only the sign is observed and net-zero rows are
                // dropped by the accumulator, the clamp is unobservable. Dropping
                // it cut another ~10% off N=10 circuit_step, bit-exact. Set
                // `FELDERA_KEEP_OUTPUT_DISTINCT` to restore it.
                let keep_output_distinct = std::env::var("FELDERA_KEEP_OUTPUT_DISTINCT").is_ok();
                let out = if keep_output_distinct {
                    out.distinct()
                } else {
                    out
                };
                rule_outs.push((rp.idx, rp.var_order.clone(), rp.n_vars, out.output()));
            }
            Ok((input_vec, rule_outs))
        })?;

        let inputs: HashMap<FunctionId, ZSetHandle<R>> = input_vec.into_iter().collect();
        let rules: Vec<FusedRuleW<R>> = rule_outs
            .into_iter()
            .map(|(idx, var_order, n_vars, output)| FusedRuleW {
                idx,
                var_order,
                output,
                n_vars,
            })
            .collect();

        Ok(FusedJoinImpl {
            handle,
            inputs,
            rules,
        })
    }

    /// The rule indices this fused circuit serves (build order).
    fn rule_indices(&self) -> Vec<usize> {
        self.rules.iter().map(|r| r.idx).collect()
    }

    /// The canonical var order for the fused rule at build position `pos`.
    fn var_order_at(&self, pos: usize) -> &[u32] {
        &self.rules[pos].var_order
    }

    /// Feed one round of relation deltas into the SHARED inputs and run a SINGLE
    /// transaction, returning per-rule binding deltas. The outer `Vec` is in the
    /// same order as [`FusedJoinImpl::rule_indices`]; each inner `Vec` is that
    /// rule's `(binding_row, weight)` pairs (positive = new, negative = retracted).
    fn step(
        &mut self,
        deltas: &HashMap<FunctionId, Vec<(Vec<u32>, ZWeight)>>,
    ) -> Result<Vec<Vec<(Vec<u32>, ZWeight)>>> {
        let prof = std::env::var("FELDERA_PROFILE").is_ok();
        let t_feed = std::time::Instant::now();
        let mut pushed_any = false;
        for (func, rows) in deltas {
            if let Some(input) = self.inputs.get(func) {
                for (row, w) in rows {
                    input.push(pack_row(row), *w);
                    pushed_any = true;
                }
            }
        }
        if !pushed_any {
            return Ok(vec![Vec::new(); self.rules.len()]);
        }
        if prof {
            add_ns(&PROF_FEED_NS, t_feed.elapsed());
            PROF_STEP_CALLS.fetch_add(1, Ordering::Relaxed);
        }

        // ONE transaction for the WHOLE ruleset. Accumulate each rule's
        // non-integrated output across commit steps (mirrors PersistentJoin::step:
        // the non-integrated handle only retains the last internal step's delta).
        self.handle.start_transaction()?;
        self.handle.start_commit_transaction()?;
        let mut accs: Vec<HashMap<Vec<u32>, ZWeight>> = vec![HashMap::new(); self.rules.len()];
        while !self.handle.is_commit_complete() {
            let t_step = std::time::Instant::now();
            self.handle.step()?;
            if prof {
                add_ns(&PROF_STEP_NS, t_step.elapsed());
            }
            let t_read = std::time::Instant::now();
            for (ri, rule) in self.rules.iter().enumerate() {
                for (row, (), w) in rule.output.consolidate().iter() {
                    let w: ZWeight = w;
                    if w != 0 {
                        let key: Vec<u32> = (0..rule.n_vars).map(|i| get_col(&row, i)).collect();
                        *accs[ri].entry(key).or_insert(0) += w;
                    }
                }
            }
            if prof {
                add_ns(&PROF_READ_NS, t_read.elapsed());
            }
        }
        Ok(accs
            .into_iter()
            .map(|acc| acc.into_iter().filter(|(_, w)| *w != 0).collect())
            .collect())
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
