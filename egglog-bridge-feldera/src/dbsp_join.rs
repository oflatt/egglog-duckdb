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
//! one fixed-width binding row ([`BindRow`], a dbsp [`Tup10`]) per satisfying
//! assignment, holding the rule's body variables in a fixed canonical order.
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
//! [`Tup10`] (see [`JOIN_WIDTH`]). A rule is DBSP-eligible iff:
//!   - its body has at least one table atom;
//!   - every table atom has arity <= [`JOIN_WIDTH`];
//!   - the rule's body uses <= [`JOIN_WIDTH`] distinct variables (binding row);
//!   - its only body *primitives* are `!=` guards (recognized by name), which
//!     lower to a pure-`u32`-inequality DBSP filter — every other body prim
//!     (value-computing, ordering guards on typed base values) forces the
//!     host fallback, because evaluating it needs the primitive engine the
//!     DBSP closure cannot hold.
//!
//! Rules that are not eligible fall back to the host interpreter; `run_rules`
//! reports the split so the milestone can characterize the frontier honestly.

use anyhow::{anyhow, Result};
use dbsp::utils::Tup10;
use dbsp::{CircuitHandle, OrdZSet, OutputHandle, RootCircuit, Stream, ZSetHandle, ZWeight};
use egglog_backend_trait::FunctionId;
use hashbrown::{HashMap, HashSet};

use crate::compile::{BodyOp, RuleIr, Slot};
use crate::EGraph;

/// Max distinct body variables / atom columns the DBSP join supports (the
/// fixed-arity `DBData` row width). Width 10 uses dbsp's stock [`Tup10`]: it
/// covers every var-rejected rule observed (the math-microbenchmark
/// distributivity rewrites peak at 9 distinct body variables) with one column of
/// margin, at zero extra dependency cost (a wider custom row would need to pull
/// in dbsp's full DBData derive stack — paste/rkyv/size_of/serde/derive_more —
/// and re-pin them to dbsp 0.150). The rebuild/congruence/`@uf` rules are *not*
/// rejected by this cap (they carry value-computing body prims — `ordering-max`,
/// `next_ts`, `bool-!=`, ordering guards — and are rejected by the prim check in
/// [`plan_join`]); widening the row does not make them DBSP-eligible.
pub const JOIN_WIDTH: usize = 10;

/// A fixed-width binding row flowing through the DBSP circuit: `bind[i]` is the
/// value of the rule's `i`-th canonical body variable (0 if not yet bound).
type BindRow = Tup10<u32, u32, u32, u32, u32, u32, u32, u32, u32, u32>;

/// A fixed-width relation row pushed into the circuit's input z-sets.
type RelRow = Tup10<u32, u32, u32, u32, u32, u32, u32, u32, u32, u32>;

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
    Tup10(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9])
}

/// Read column `i` of a fixed-width row.
#[inline]
fn get_col(r: &BindRow, i: usize) -> u32 {
    let Tup10(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9) = r;
    match i {
        0 => *a0,
        1 => *a1,
        2 => *a2,
        3 => *a3,
        4 => *a4,
        5 => *a5,
        6 => *a6,
        7 => *a7,
        8 => *a8,
        9 => *a9,
        _ => panic!("col index {i} out of range"),
    }
}

/// The name the frontend records for the `!=` predicate (via `rename_prim`).
const NEQ_NAME: &str = "!=";

/// A `!=` guard recognized in a rule body: the two operand slots.
struct NeqGuard {
    a: Slot,
    b: Slot,
}

/// The analysis of a DBSP-eligible rule body: canonical variable order, the
/// table atoms, and the `!=` guards.
pub struct JoinPlan {
    /// Canonical body-variable order: `var_order[i]` is the variable id placed
    /// at binding-row column `i`.
    var_order: Vec<u32>,
    /// var id -> its binding-row column index.
    var_col: HashMap<u32, usize>,
    /// The body table atoms (in emission order).
    atoms: Vec<AtomPlan>,
    /// `!=` guards to apply as DBSP filters once both operands are bound.
    neq: Vec<NeqGuard>,
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
pub fn plan_join(eg: &EGraph, rule: &RuleIr) -> Option<JoinPlan> {
    let mut atoms: Vec<AtomPlan> = Vec::new();
    let mut neq: Vec<NeqGuard> = Vec::new();
    let mut var_order: Vec<u32> = Vec::new();
    let mut var_col: HashMap<u32, usize> = HashMap::new();

    let see_var = |v: u32, var_order: &mut Vec<u32>, var_col: &mut HashMap<u32, usize>| {
        if !var_col.contains_key(&v) {
            var_col.insert(v, var_order.len());
            var_order.push(v);
        }
    };

    for op in &rule.body {
        match op {
            BodyOp::Atom(atom) => {
                if atom.slots.len() > JOIN_WIDTH {
                    return None; // atom too wide for the fixed row
                }
                for s in &atom.slots {
                    if let Slot::Var(v) = s {
                        see_var(*v, &mut var_order, &mut var_col);
                    }
                }
                atoms.push(AtomPlan {
                    func: atom.func,
                    slots: atom.slots.clone(),
                });
            }
            BodyOp::Prim { id, args, ret } => {
                // Only `!=` guards are DBSP-eligible (pure u32 inequality). Any
                // other body primitive needs the primitive engine inside the
                // join, which a `Send + 'static` DBSP closure cannot hold.
                if eg.external_funcs.name(*id) != Some(NEQ_NAME) {
                    return None;
                }
                // `!=` is encoded as `query_prim([a, b, ret_unit])`: two
                // operands and a unit return slot. The operands must be
                // variables already bound by a preceding atom, or constants.
                if args.len() != 2 {
                    return None;
                }
                // The return slot of `!=` is unit; it neither binds nor guards.
                let _ = ret;
                for s in args {
                    if let Slot::Var(v) = s {
                        // A `!=` over a variable not bound by any table atom is
                        // not something the join can evaluate.
                        if !var_col.contains_key(v) {
                            return None;
                        }
                    }
                }
                neq.push(NeqGuard {
                    a: args[0].clone(),
                    b: args[1].clone(),
                });
            }
        }
    }

    if atoms.is_empty() {
        return None; // nothing to join (constant-only / prim-only body)
    }
    if var_order.len() > JOIN_WIDTH {
        return None; // too many distinct body variables for the fixed row
    }

    Some(JoinPlan {
        var_order,
        var_col,
        atoms,
        neq,
    })
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
    let neq = plan
        .neq
        .iter()
        .map(|g| (g.a.clone(), g.b.clone()))
        .collect::<Vec<_>>();
    let n_vars = plan.var_order.len();

    let (handle, (inputs, output)) = RootCircuit::build(move |root| {
        let mut inputs: Vec<RelHandles> = Vec::with_capacity(n_atoms);
        let mut streams: Vec<Stream<RootCircuit, OrdZSet<RelRow>>> = Vec::with_capacity(n_atoms);
        for _ in 0..n_atoms {
            let (stream, input) = root.add_input_zset::<RelRow>();
            inputs.push(RelHandles { input });
            streams.push(stream);
        }

        let out = build_join_stream(&streams, &atoms, &var_col, &neq, n_vars)?;
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
    neq: &[(Slot, Slot)],
    n_vars: usize,
) -> Result<Stream<RootCircuit, OrdZSet<BindRow>>> {
    // `bound`: which canonical variable columns are filled after each atom.
    let mut bound: Vec<bool> = vec![false; n_vars];

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
    // Apply any `!=` guards now satisfiable.
    cur = apply_neq(cur, neq, var_col, &bound);

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
        cur = apply_neq(cur, neq, var_col, &bound);
    }

    Ok(cur)
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
fn set_col(mut r: BindRow, i: usize, v: u32) -> BindRow {
    match i {
        0 => r.0 = v,
        1 => r.1 = v,
        2 => r.2 = v,
        3 => r.3 = v,
        4 => r.4 = v,
        5 => r.5 = v,
        6 => r.6 = v,
        7 => r.7 = v,
        8 => r.8 = v,
        9 => r.9 = v,
        _ => panic!("set_col index {i} out of range"),
    }
    r
}

/// Apply every `!=` guard whose operands are both resolvable (bound vars or
/// constants) as a `filter` on the binding stream.
fn apply_neq(
    stream: Stream<RootCircuit, OrdZSet<BindRow>>,
    neq: &[(Slot, Slot)],
    var_col: &HashMap<u32, usize>,
    bound: &[bool],
) -> Stream<RootCircuit, OrdZSet<BindRow>> {
    let mut cur = stream;
    for (a, b) in neq {
        // Only apply once both operands are available.
        let ra = resolvable(a, var_col, bound);
        let rb = resolvable(b, var_col, bound);
        if !(ra && rb) {
            continue;
        }
        let a = a.clone();
        let b = b.clone();
        let vc = var_col.clone();
        cur = cur.filter(move |row: &BindRow| {
            let av = slot_val(&a, row, &vc);
            let bv = slot_val(&b, row, &vc);
            av != bv
        });
    }
    cur
}

/// Whether a slot's value is available in the binding row (bound var) or is a
/// constant.
fn resolvable(s: &Slot, var_col: &HashMap<u32, usize>, bound: &[bool]) -> bool {
    match s {
        Slot::Const(_) => true,
        Slot::Var(v) => var_col.get(v).map(|&c| bound[c]).unwrap_or(false),
    }
}

/// Read a slot's value out of a binding row.
fn slot_val(s: &Slot, row: &BindRow, var_col: &HashMap<u32, usize>) -> u32 {
    match s {
        Slot::Const(c) => *c,
        Slot::Var(v) => get_col(row, var_col[v]),
    }
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
    pub fn build(plan: &JoinPlan) -> Result<PersistentJoin> {
        let n_atoms = plan.atoms.len();
        let atoms: Vec<Vec<Slot>> = plan.atoms.iter().map(|a| a.slots.clone()).collect();
        let var_col = plan.var_col.clone();
        let neq = plan
            .neq
            .iter()
            .map(|g| (g.a.clone(), g.b.clone()))
            .collect::<Vec<_>>();
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
            let out = build_join_stream(&streams, &atoms, &var_col, &neq, n_vars)?;
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
            neq: vec![],
        }
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
        let mut pj = PersistentJoin::build(&plan).expect("build persistent join");

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
