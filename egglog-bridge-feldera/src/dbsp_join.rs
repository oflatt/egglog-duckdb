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
//! one [`Tup8`] per satisfying assignment, holding the rule's body variables
//! in a fixed canonical order.
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
//! ## Eligibility / the Tup8 cap
//!
//! DBSP rows must be fixed-arity `DBData` (rkyv-archivable); we use the same
//! uniform [`Tup8`] the M1/M2 circuits used. A rule is DBSP-eligible iff:
//!   - its body has at least one table atom;
//!   - every table atom has arity <= 8;
//!   - the rule's body uses <= 8 distinct variables (canonical binding row);
//!   - its only body *primitives* are `!=` guards (recognized by name), which
//!     lower to a pure-`u32`-inequality DBSP filter — every other body prim
//!     (value-computing, ordering guards on typed base values) forces the
//!     host fallback, because evaluating it needs the primitive engine the
//!     DBSP closure cannot hold.
//!
//! Rules that are not eligible fall back to the host interpreter; `run_rules`
//! reports the split so the milestone can characterize the frontier honestly.

use anyhow::{anyhow, Result};
use dbsp::utils::Tup8;
use dbsp::{OrdZSet, RootCircuit, Stream, ZSetHandle, ZWeight};
use egglog_backend_trait::FunctionId;
use hashbrown::{HashMap, HashSet};

use crate::compile::{BodyOp, RuleIr, Slot};
use crate::EGraph;

/// Max distinct body variables / atom columns the DBSP join supports (the
/// fixed-arity `DBData` row width).
pub const JOIN_WIDTH: usize = 8;

/// A fixed-width binding row flowing through the DBSP circuit: `bind[i]` is the
/// value of the rule's `i`-th canonical body variable (0 if not yet bound).
type BindRow = Tup8<u32, u32, u32, u32, u32, u32, u32, u32>;

/// A fixed-width relation row pushed into the circuit's input z-sets.
type RelRow = Tup8<u32, u32, u32, u32, u32, u32, u32, u32>;

fn pack8(vals: &[u32]) -> Tup8<u32, u32, u32, u32, u32, u32, u32, u32> {
    let mut a = [0u32; JOIN_WIDTH];
    for (i, v) in vals.iter().enumerate() {
        a[i] = *v;
    }
    Tup8(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
}

#[inline]
fn col8(r: &Tup8<u32, u32, u32, u32, u32, u32, u32, u32>, i: usize) -> u32 {
    let Tup8(a0, a1, a2, a3, a4, a5, a6, a7) = r;
    match i {
        0 => *a0,
        1 => *a1,
        2 => *a2,
        3 => *a3,
        4 => *a4,
        5 => *a5,
        6 => *a6,
        7 => *a7,
        _ => panic!("col8 index {i} out of range"),
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

/// Run the body join of `plan` on DBSP over the current mirror, returning one
/// binding row per satisfying assignment. Each returned `Vec<u32>` has length
/// `plan.n_vars()` and is indexed by the canonical variable order.
///
/// This builds a fresh non-recursive circuit, pushes the current mirror as the
/// circuit input, runs exactly one `transaction()` (one round), reads the
/// consolidated join output, and drops the circuit. Building per call keeps the
/// DBSP join a pure function of the current relation contents (the
/// per-iteration model) without per-subset circuit-cache bookkeeping.
pub fn run_join(eg: &EGraph, plan: &JoinPlan) -> Result<Vec<Vec<u32>>> {
    // The set of distinct relations referenced by the body atoms.
    let rel_ids: Vec<FunctionId> = {
        let mut seen = HashSet::new();
        let mut v = Vec::new();
        for a in &plan.atoms {
            if seen.insert(a.func) {
                v.push(a.func);
            }
        }
        v
    };

    // Snapshot each relation's rows (fixed-width, 0-padded) up front; the
    // circuit-builder closure captures them by move.
    let mut snapshot: HashMap<FunctionId, Vec<RelRow>> = HashMap::new();
    for &f in &rel_ids {
        let rows: Vec<RelRow> = eg
            .mirror
            .get(&f)
            .map(|set| set.iter().map(|row| pack8(row)).collect())
            .unwrap_or_default();
        snapshot.insert(f, rows);
    }

    // Clone the plan pieces the circuit closure needs (it must be `'static`).
    let atoms: Vec<(FunctionId, Vec<Slot>)> = plan
        .atoms
        .iter()
        .map(|a| (a.func, a.slots.clone()))
        .collect();
    let var_col = plan.var_col.clone();
    let neq = plan
        .neq
        .iter()
        .map(|g| (g.a.clone(), g.b.clone()))
        .collect::<Vec<_>>();
    let n_vars = plan.var_order.len();
    let rel_ids_c = rel_ids.clone();

    let (handle, (mut inputs, output)) = RootCircuit::build(move |root| {
        let mut inputs: HashMap<FunctionId, RelHandles> = HashMap::new();
        let mut streams: HashMap<FunctionId, Stream<RootCircuit, OrdZSet<RelRow>>> = HashMap::new();
        for &f in &rel_ids_c {
            let (stream, input) = root.add_input_zset::<RelRow>();
            inputs.insert(f, RelHandles { input });
            streams.insert(f, stream);
        }

        let out = build_join_stream(&streams, &atoms, &var_col, &neq, n_vars)?;
        Ok((inputs, out.output()))
    })?;

    // Push each relation's snapshot as the circuit input delta (all +1).
    for &f in &rel_ids {
        let h = inputs.get_mut(&f).expect("relation handle");
        for row in &snapshot[&f] {
            h.input.push(*row, 1);
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
            bindings.push((0..n_vars).map(|i| col8(&row, i)).collect());
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
    streams: &HashMap<FunctionId, Stream<RootCircuit, OrdZSet<RelRow>>>,
    atoms: &[(FunctionId, Vec<Slot>)],
    var_col: &HashMap<u32, usize>,
    neq: &[(Slot, Slot)],
    n_vars: usize,
) -> Result<Stream<RootCircuit, OrdZSet<BindRow>>> {
    // `bound`: which canonical variable columns are filled after each atom.
    let mut bound: Vec<bool> = vec![false; n_vars];

    // Initialize from the first atom: map each row to a binding row.
    let (f0, slots0) = &atoms[0];
    let s0 = streams
        .get(f0)
        .ok_or_else(|| anyhow!("join atom references unregistered relation"))?
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
    for (f, slots) in &atoms[1..] {
        let s = streams
            .get(f)
            .ok_or_else(|| anyhow!("join atom references unregistered relation"))?
            .clone();

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
            let left = cur.map_index(move |b: &BindRow| (join_key(b, &scl, col8), *b));

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
            let right = s.map_index(move |r: &RelRow| (join_key(r, &sac, col8), *r));

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

/// Build a `u64`/`u128`-free join key from selected columns (packed into a
/// `Vec<u32>` for arbitrary arity — DBSP requires the key be `DBData`, and
/// `Vec<u32>` is not, so we pack up to 4 columns into a `Tup4`-style tuple via
/// a fixed `[u32; 8]`-backed `Tup8`). For simplicity and to stay within
/// `DBData`, the key is a `Tup8` with the selected columns in the low slots.
fn join_key<R>(r: &R, cols: &[usize], get: fn(&R, usize) -> u32) -> BindRow {
    let mut a = [0u32; JOIN_WIDTH];
    for (i, &c) in cols.iter().enumerate() {
        a[i] = get(r, c);
    }
    Tup8(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
}

/// Match the first atom's row against its slots and produce the initial binding
/// row, or `None` if a constant / repeated-variable constraint fails.
fn bind_atom(r: &RelRow, slots: &[Slot], var_col: &HashMap<u32, usize>) -> Option<BindRow> {
    let mut out = [0u32; JOIN_WIDTH];
    // Track values bound to each canonical var within this row to enforce
    // repeated-variable equality.
    let mut local: HashMap<u32, u32> = HashMap::new();
    for (i, s) in slots.iter().enumerate() {
        let col = col8(r, i);
        match s {
            Slot::Const(c) => {
                if *c != col {
                    return None;
                }
            }
            Slot::Var(v) => {
                if let Some(&prev) = local.get(v) {
                    if prev != col {
                        return None;
                    }
                } else {
                    local.insert(*v, col);
                    out[var_col[v]] = col;
                }
            }
        }
    }
    Some(Tup8(
        out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
    ))
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
        let col = col8(r, i);
        match s {
            Slot::Const(c) => {
                if *c != col {
                    return None;
                }
            }
            Slot::Var(v) => {
                // Repeated variable within this atom must agree.
                if let Some(&prev) = local.get(v) {
                    if prev != col {
                        return None;
                    }
                    continue;
                }
                local.insert(*v, col);
                let c = var_col[v];
                if bound[c] {
                    // Already bound by a previous atom: must agree.
                    if col8(&out, c) != col {
                        return None;
                    }
                } else {
                    out = set_col8(out, c, col);
                }
            }
        }
    }
    Some(out)
}

#[inline]
fn set_col8(mut r: BindRow, i: usize, v: u32) -> BindRow {
    match i {
        0 => r.0 = v,
        1 => r.1 = v,
        2 => r.2 = v,
        3 => r.3 = v,
        4 => r.4 = v,
        5 => r.5 = v,
        6 => r.6 = v,
        7 => r.7 = v,
        _ => panic!("set_col8 index {i} out of range"),
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
        Slot::Var(v) => col8(row, var_col[v]),
    }
}
