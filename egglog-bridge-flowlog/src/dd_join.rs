//! The relational **table-atom join on the Differential-Dataflow engine** for
//! the FlowLog backend (Milestone 3) — the FlowLog analog of Feldera M4's
//! `dbsp_join.rs`.
//!
//! ## The split (the milestone mandate)
//!
//! For a DD-eligible rule, the rule's **table-atom join** runs on a
//! Differential-Dataflow engine in a subprocess compiled from a runtime-emitted
//! `.dl` (the engine, BY CONSTRUCTION, since the shell-out compiles
//! `.dl -> flowlog -> DD`). The join's binding rows come back over the pipe, and
//! the **primitive tail + head actions are applied HOST-side** (in
//! [`crate::interpret`]). This mirrors Feldera M4 exactly: table join on the
//! engine, primitive tail host-side.
//!
//! Body **primitives** (`!=` guards, value-computing prims like `+`) are NOT
//! lowered into the `.dl`: the host interpreter re-runs every `BodyOp::Prim`
//! over the join bindings (see `interpret::run_iteration`'s DD branch), so all
//! guards/computations apply host-side with bit-for-bit primitive parity. The
//! DD `.dl` therefore only computes the **relational join of the table atoms**,
//! which is the expensive, paper-relevant part — and the part the mandate wants
//! on the engine.
//!
//! ## Non-recursive = one bounded iteration
//!
//! The emitted `.dl` is a single non-recursive rule `out(vars..) :- r0(..),
//! r1(..), ...`. One `run_rules` call stages the whole pre-iteration read view
//! of each body relation, `commit()`s **once**, and reads back the binding
//! rows. One commit = one egglog iteration (the M1/M2 per-iteration model). The
//! subprocess + its compiled binary are cached by the `.dl` hash and reused.
//!
//! ## Eligibility / the column cap
//!
//! A rule's join runs on DD iff its body has ≥1 table atom and the join is
//! within the engine's fixed row width (`MAX_JOIN_VARS`, the `int32`-tuple cap).
//! Wider joins fall back to the host nested-loop interpreter (the oracle).

use anyhow::Result;
use egglog_backend_trait::FunctionId;
use hashbrown::{HashMap, HashSet};

use crate::compile::{row_col, BodyOp, Row, RuleIr, Slot};
use crate::subprocess::DriverHandle;
use crate::EGraph;

/// Maximum number of distinct body variables a DD join can carry (the output
/// `.dl` relation's column count). Generous; wider rules fall back to host.
pub const MAX_JOIN_VARS: usize = 16;

/// A planned DD join: the canonical body-variable order, the table atoms, and
/// the per-atom column→variable wiring.
pub struct JoinPlan {
    /// `var_order[i]` is the variable id placed in output column `i`.
    var_order: Vec<u32>,
    /// Inverse of `var_order`: variable id → output column index.
    var_col: HashMap<u32, usize>,
    /// The table atoms, in body order.
    atoms: Vec<PlanAtom>,
}

/// One table atom in the join plan.
struct PlanAtom {
    func: FunctionId,
    /// One entry per column: the relation's column references this slot.
    slots: Vec<Slot>,
}

impl JoinPlan {
    /// Number of output columns (== distinct body variables).
    pub fn width(&self) -> usize {
        self.var_order.len()
    }

    /// The canonical variable order (output column `i` holds `var_order[i]`).
    pub fn var_order(&self) -> &[u32] {
        &self.var_order
    }
}

/// Decide whether `rule`'s table-atom join can run on DD. Returns the plan if
/// so, `None` to fall back to the host interpreter.
pub fn plan_join(_eg: &EGraph, rule: &RuleIr) -> Option<JoinPlan> {
    let mut var_order: Vec<u32> = Vec::new();
    let mut var_col: HashMap<u32, usize> = HashMap::new();
    let mut atoms: Vec<PlanAtom> = Vec::new();

    let see_var = |v: u32, var_order: &mut Vec<u32>, var_col: &mut HashMap<u32, usize>| {
        if !var_col.contains_key(&v) {
            var_col.insert(v, var_order.len());
            var_order.push(v);
        }
    };

    for op in &rule.body {
        match op {
            BodyOp::Atom(atom) => {
                for s in &atom.slots {
                    if let Slot::Var(v) = s {
                        see_var(*v, &mut var_order, &mut var_col);
                    }
                }
                atoms.push(PlanAtom {
                    func: atom.func,
                    slots: atom.slots.clone(),
                });
            }
            // Body primitives stay host-side (re-run by the interpreter over the
            // join bindings); they don't affect join planning. A primitive may
            // bind a fresh var that is NOT a join output column — that's fine,
            // the host evaluates it after the join.
            BodyOp::Prim { .. } => {}
        }
    }

    // Need at least one table atom to have a relational join to run on DD.
    if atoms.is_empty() {
        return None;
    }
    // Respect the engine row width.
    if var_order.len() > MAX_JOIN_VARS {
        return None;
    }

    Some(JoinPlan {
        var_order,
        var_col,
        atoms,
    })
}

/// Run `plan`'s join on the DD engine: emit the join `.dl`, build/cache + spawn
/// the driver subprocess, stage the pre-iteration read view of each body
/// relation, `commit()` once, and read back the binding rows (one `Vec<u32>`
/// per satisfying assignment, body variables in `plan.var_order()` order).
pub fn run_join(
    eg: &mut EGraph,
    plan: &JoinPlan,
    read: &HashMap<FunctionId, Vec<Row>>,
) -> Result<Vec<Vec<u32>>> {
    // Full×…×full: every atom occurrence ranges over its relation's full read
    // view. (Retained for the M3 `dd_join_proof`; the seminaive driver uses
    // `run_join_seminaive`.)
    let atom_rows: Vec<Vec<&Row>> = plan
        .atoms
        .iter()
        .map(|atom| {
            read.get(&atom.func)
                .map(|v| v.iter().collect())
                .unwrap_or_default()
        })
        .collect();
    run_join_with(eg, plan, &atom_rows)
}

/// Seminaive variant: the atom **occurrence** at index `delta_ord` is fed only
/// its relation's *delta* rows, every other occurrence the full read view. This
/// is one term of the seminaive union — the relational join still runs on the
/// Differential-Dataflow engine, now restricted to bindings that touch a
/// newly-derived fact in occurrence `delta_ord`.
pub fn run_join_seminaive(
    eg: &mut EGraph,
    plan: &JoinPlan,
    delta_ord: usize,
    read: &HashMap<FunctionId, Vec<Row>>,
    delta: &HashMap<FunctionId, HashSet<Row>>,
) -> Result<Vec<Vec<u32>>> {
    let empty_full: Vec<Row> = Vec::new();
    let empty_delta: HashSet<Row> = HashSet::new();
    let atom_rows: Vec<Vec<&Row>> = plan
        .atoms
        .iter()
        .enumerate()
        .map(|(ord, atom)| {
            if ord == delta_ord {
                delta
                    .get(&atom.func)
                    .unwrap_or(&empty_delta)
                    .iter()
                    .collect()
            } else {
                read.get(&atom.func).unwrap_or(&empty_full).iter().collect()
            }
        })
        .collect();
    run_join_with(eg, plan, &atom_rows)
}

/// Core: stage the supplied per-atom-occurrence row sets onto the DD engine,
/// `commit()` once, and read back the binding rows (one `Vec<u32>` per
/// satisfying assignment, body variables in `plan.var_order()` order). Each plan
/// atom occurrence becomes engine relation `r{idx}`, staged independently — so
/// the same FunctionId appearing in two atoms can be fed different row sets
/// (required for the seminaive per-occurrence delta).
fn run_join_with(
    eg: &mut EGraph,
    plan: &JoinPlan,
    atom_rows: &[Vec<&Row>],
) -> Result<Vec<Vec<u32>>> {
    let dl = emit_join_dl(plan);
    let main_rs = emit_join_main(plan);

    // Cache one warm driver per distinct join `.dl` on this egraph.
    let driver = match eg.dd_drivers.get_mut(&dl) {
        Some(d) => d,
        None => {
            let mut handle = DriverHandle::build_or_cached_with(&dl, &main_rs)?;
            handle.spawn()?;
            eg.dd_drivers.insert(dl.clone(), handle);
            eg.dd_drivers.get_mut(&dl).unwrap()
        }
    };

    // The engine is non-recursive and recomputed from scratch each call (we
    // restage and read the full join output), so clear any previously-staged
    // rows first — a fresh `clear` resets the engine's relations to empty.
    driver.send_clear()?;

    // Stage the supplied rows for each atom occurrence into engine relation
    // `r{idx}`, mapping the relation's columns to the `.dl` relation columns.
    for (idx, atom) in plan.atoms.iter().enumerate() {
        let arity = atom.slots.len();
        for row in &atom_rows[idx] {
            let cols: Vec<i32> = (0..arity).map(|c| row_col(row, c) as i32).collect();
            driver.insert_row(idx, &cols)?;
        }
    }

    let binding_rows = driver.commit_rows()?;
    // Each returned row has `plan.width()` columns in var_order order.
    let w = plan.width();
    let mut out = Vec::with_capacity(binding_rows.len());
    for r in binding_rows {
        if r.len() == w {
            out.push(r.iter().map(|&c| c as u32).collect());
        }
    }
    Ok(out)
}

/// Emit the join `.dl`: one command-input relation per table atom, plus the
/// non-recursive join rule projecting the body variables, and `.output out`.
fn emit_join_dl(plan: &JoinPlan) -> String {
    let mut s = String::new();
    s.push_str("// AUTO-GENERATED at runtime by egglog-bridge-flowlog (M3 dd-join).\n");
    s.push_str("// One non-recursive multi-atom join = one bounded egglog iteration.\n");
    s.push_str("// Table-atom join runs on Differential Dataflow; prim tail is host-side.\n\n");

    for (idx, atom) in plan.atoms.iter().enumerate() {
        let arity = atom.slots.len();
        let cols: Vec<String> = (0..arity).map(|c| format!("c{c}: int32")).collect();
        s.push_str(&format!(".decl r{idx}({})\n", cols.join(", ")));
        s.push_str(&format!(
            ".input r{idx}(IO=\"command\", delimiter=\",\")\n\n"
        ));
    }

    // Output relation: one column per distinct body variable, var_order order.
    let out_cols: Vec<String> = (0..plan.width()).map(|i| format!("v{i}: int32")).collect();
    s.push_str(&format!(".decl out({})\n", out_cols.join(", ")));

    // Build the join rule. Each atom's column becomes either a variable name
    // `vN` (for a Var slot, named by its output-column index) or a constant
    // literal (for a Const slot). Repeated variables across atoms express the
    // join; repeated within an atom express a self-constraint.
    let head_args: Vec<String> = (0..plan.width()).map(|i| format!("v{i}")).collect();
    let mut body_atoms: Vec<String> = Vec::new();
    for (idx, atom) in plan.atoms.iter().enumerate() {
        let args: Vec<String> = atom
            .slots
            .iter()
            .map(|sl| match sl {
                Slot::Var(v) => format!("v{}", plan.var_col[v]),
                Slot::Const(c) => (*c as i32).to_string(),
            })
            .collect();
        body_atoms.push(format!("r{idx}({})", args.join(", ")));
    }
    s.push_str(&format!(
        "out({}) :- {}.\n",
        head_args.join(", "),
        body_atoms.join(", ")
    ));
    s.push_str(".output out\n");
    s
}

/// Emit the driver `main.rs` for the generalized dd-join protocol:
///   - `clear`                  -> reset all staged relations to empty
///   - `ins <rel_idx> <c..>`    -> stage an insert row into relation `r<idx>`
///   - `commit`                 -> step one epoch; emit `row <c..>` per `out`
///     tuple (net positive multiplicity), then `ok`
///   - `quit`                   -> exit 0
fn emit_join_main(plan: &JoinPlan) -> String {
    let n_rel = plan.atoms.len();
    let w = plan.width();

    // Per-relation insert dispatch: `r{idx}` has arity = atom.slots.len(); the
    // generated tuple alias is `(i32, ...)`. We build the tuple from the parsed
    // columns and call `engine.insert_r{idx}`.
    let mut insert_arms = String::new();
    for (idx, atom) in plan.atoms.iter().enumerate() {
        let arity = atom.slots.len();
        let tup = build_tuple_expr(arity);
        insert_arms.push_str(&format!(
            "            {idx} => {{ engine.insert_r{idx}(vec![{tup}]); }}\n"
        ));
    }

    // `out` tuple destructure for read-back: emit `row c0 c1 ...`.
    let out_fields: Vec<String> = (0..w).map(|i| format!("f{i}")).collect();
    let out_pat = if w == 1 {
        // Arity-1 tuple alias is a bare scalar, not a 1-tuple.
        "f0".to_string()
    } else {
        format!("({})", out_fields.join(", "))
    };
    let row_fmt = vec!["{}"; w].join(" ");
    let row_args = out_fields.join(", ");

    format!(
        r#"//! AUTO-GENERATED dd-join driver (M3): table-atom join on Differential
//! Dataflow. One `commit` = one bounded egglog iteration; the host applies the
//! primitive tail + head actions.

#[allow(clippy::all)]
#[allow(dead_code)]
#[allow(unused)]
mod gen {{
    include!(concat!(env!("OUT_DIR"), "/program.rs"));
}}

use gen::DatalogIncrementalEngine;
use std::io::{{BufRead, Write}};

fn main() {{
    let mut engine = DatalogIncrementalEngine::new(1);
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {{
        let line = match line {{ Ok(l) => l, Err(_) => break }};
        let line = line.trim();
        if line.is_empty() {{ continue; }}
        let mut it = line.split_whitespace();
        let cmd = it.next().unwrap_or("");
        match cmd {{
            "clear" => {{
                // Reset the engine to a fresh empty state. The non-recursive
                // join is recomputed from scratch each iteration from the
                // restaged read view, so a brand-new engine is the cleanest
                // reset and keeps results deterministic.
                engine = DatalogIncrementalEngine::new(1);
                let _ = writeln!(out, "ok");
                let _ = out.flush();
            }}
            "ins" => {{
                let rel: usize = it.next().and_then(|t| t.parse().ok()).unwrap_or(usize::MAX);
                let cols: Vec<i32> = it.filter_map(|t| t.parse::<i32>().ok()).collect();
                let _ = &cols;
                match rel {{
{insert_arms}                    _ => {{ let _ = writeln!(out, "err unknown rel {{rel}}"); let _ = out.flush(); }}
                }}
            }}
            "commit" => {{
                let results = engine.commit();
                for (t, d) in results.out.into_iter() {{
                    if d <= 0 {{ continue; }}
                    let {out_pat} = t;
                    let _ = writeln!(out, "row {row_fmt}", {row_args});
                }}
                let _ = writeln!(out, "ok");
                let _ = out.flush();
            }}
            "quit" => break,
            other => {{ let _ = writeln!(out, "err unknown command {{other}}"); let _ = out.flush(); }}
        }}
    }}
    let _ = n_rel_marker();
}}

// Keep the relation count referenced so codegen stays in sync ({n_rel} rels).
fn n_rel_marker() -> usize {{ {n_rel} }}
"#,
    )
}

/// Build the tuple-construction expression for an arity-`n` relation's
/// `insert_r{idx}(vec![ ... ])` from the parsed `cols: Vec<i32>`. Arity-1 is a
/// bare scalar (the tuple alias degenerates), else an `(c0, c1, ...)` tuple.
fn build_tuple_expr(arity: usize) -> String {
    if arity == 1 {
        "cols[0]".to_string()
    } else {
        let elems: Vec<String> = (0..arity).map(|i| format!("cols[{i}]")).collect();
        format!("({})", elems.join(", "))
    }
}
