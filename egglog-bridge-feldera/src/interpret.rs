//! Host-side rule interpreter for the Feldera backend (Milestone 3).
//!
//! One `run_rules` call = **one bounded egglog iteration**. The interpreter:
//!
//! 1. snapshots the relation mirror (the read view for this iteration — all
//!    rules see the same pre-iteration state, matching egglog's semi-naive
//!    "match against the old database, then apply" model for a single hop);
//! 2. for each rule, runs a **nested-loop join** over the body's table atoms,
//!    threading a variable→value binding environment, evaluating primitive body
//!    atoms (`BodyOp::Prim`) against the embedded `Database` (guards prune,
//!    bindings extend);
//! 3. for every surviving binding, executes the head ops in order — `set` /
//!    `delete` / `subsume` writes, RHS `lookup` (eq-sort constructor: create on
//!    miss), RHS primitive `call` (with side effects + result binding),
//!    `union`, `panic`;
//! 4. applies all collected writes/removes to the mirror and resolves
//!    functional-dependency conflicts per each touched function's merge mode.
//!
//! Primitives are invoked through `Database::with_execution_state`, so they see
//! the same interned base `Value`s the frontend created — giving the Feldera
//! backend bit-for-bit value parity with the reference backend.

use anyhow::{anyhow, Result};
use egglog_backend_trait::{FunctionId, Value};
use egglog_numeric_id::NumericId;
use hashbrown::{HashMap, HashSet};

use crate::compile::{slot_lookup, BodyOp, HeadOp, Row, RuleIr, Slot};
use crate::EGraph;

/// Binding environment: variable id → bound `u32` value.
type Env = HashMap<u32, u32>;

/// A pending write to apply after all matches are computed.
enum Write {
    /// Insert/overwrite a full row.
    Set(FunctionId, Row),
    /// Retract by key (the slots address inputs for a function, whole row for a
    /// relation).
    Remove(FunctionId, Vec<u32>),
}

/// Run one bounded iteration of `rule_idxs` against the egraph. Returns whether
/// the mirror changed.
pub fn run_iteration(eg: &mut EGraph, rule_idxs: &[usize]) -> Result<bool> {
    // Snapshot the read view: rules match against the pre-iteration mirror.
    let read: HashMap<FunctionId, Vec<Row>> = eg
        .mirror
        .iter()
        .map(|(f, set)| (*f, set.iter().cloned().collect()))
        .collect();

    let mut writes: Vec<Write> = Vec::new();
    let mut touched: HashSet<FunctionId> = HashSet::new();

    // Collect each rule's IR up front (clone to avoid borrow conflicts while we
    // also mutate the db / mirror via lookups).
    let rules: Vec<RuleIr> = rule_idxs
        .iter()
        .filter_map(|&i| eg.rules.get(i).and_then(|r| r.clone()))
        .collect();

    for rule in &rules {
        // Enumerate all body matches as a list of binding environments.
        let mut envs: Vec<Env> = vec![Env::new()];
        for op in &rule.body {
            envs = step_body(eg, &read, op, envs)?;
            if envs.is_empty() {
                break;
            }
        }
        // Apply the head for each surviving binding.
        for mut env in envs {
            apply_head(eg, &rule.head, &mut env, &mut writes, &mut touched)?;
        }
    }

    // Apply collected writes to the mirror.
    for w in writes {
        match w {
            Write::Set(f, row) => {
                eg.mirror.entry(f).or_default().insert(row);
            }
            Write::Remove(f, key) => {
                let keylen = key.len();
                if let Some(set) = eg.mirror.get_mut(&f) {
                    set.retain(|row| {
                        (0..keylen).any(|i| crate::compile::row_col(row, i) != key[i])
                    });
                }
            }
        }
    }

    // Resolve FD conflicts on every function that a head action wrote to (a
    // `set` can introduce two rows sharing a key that must be merged per the
    // function's merge mode).
    for &f in &touched {
        eg.resolve_merge(f);
    }

    // Detect change for the frontend's saturation loop: any content delta on
    // ANY function versus the pre-iteration read view (covers head writes,
    // merges that collapse rows, AND eq-sort constructor rows created by
    // `lookup_or_create`, which write the mirror directly outside `touched`).
    let mut changed = false;
    let all_funcs: HashSet<FunctionId> = read
        .keys()
        .copied()
        .chain(eg.mirror.keys().copied())
        .collect();
    for f in all_funcs {
        let after = eg.mirror.get(&f);
        let before = read.get(&f);
        let same = match (before, after) {
            (Some(b), Some(a)) => b.len() == a.len() && b.iter().all(|r| a.contains(r)),
            (None, Some(a)) => a.is_empty(),
            (Some(b), None) => b.is_empty(),
            (None, None) => true,
        };
        if !same {
            changed = true;
            break;
        }
    }
    Ok(changed)
}

/// Extend each binding env by matching `op` against the read view (table atom)
/// or evaluating it (primitive). Returns the new list of envs.
fn step_body(
    eg: &mut EGraph,
    read: &HashMap<FunctionId, Vec<Row>>,
    op: &BodyOp,
    envs: Vec<Env>,
) -> Result<Vec<Env>> {
    match op {
        BodyOp::Atom(atom) => {
            let rows = read.get(&atom.func).map(|v| v.as_slice()).unwrap_or(&[]);
            // Determine the JOIN KEY: columns whose slot is a variable that is
            // already bound in the incoming envs (so the same var must agree
            // between the env and the row). If non-empty, hash-join on it
            // instead of a full cartesian scan. (Const-slot columns are
            // constraints, applied during `match_atom` regardless.)
            let bound_in_env = |v: u32| envs.first().map(|e| e.contains_key(&v)).unwrap_or(false);
            let key_cols: Vec<usize> = atom
                .slots
                .iter()
                .enumerate()
                .filter_map(|(i, s)| match s {
                    Slot::Var(v) if bound_in_env(*v) => Some(i),
                    _ => None,
                })
                .collect();
            // Which env var supplies each key column.
            let key_vars: Vec<u32> = key_cols
                .iter()
                .map(|&i| match &atom.slots[i] {
                    Slot::Var(v) => *v,
                    _ => unreachable!(),
                })
                .collect();

            let mut out = Vec::new();
            if key_cols.is_empty() {
                // No shared bound variable: cartesian (correct for the first
                // atom / a fresh body).
                for env in &envs {
                    for row in rows {
                        if let Some(next) = match_atom(&atom.slots, row, env) {
                            out.push(next);
                        }
                    }
                }
            } else {
                // Index rows by the key columns, then probe per env.
                let mut index: HashMap<Vec<u32>, Vec<&Row>> = HashMap::new();
                for row in rows {
                    let key: Vec<u32> = key_cols.iter().map(|&i| row[i]).collect();
                    index.entry(key).or_default().push(row);
                }
                for env in &envs {
                    let key: Vec<u32> = key_vars.iter().map(|v| env[v]).collect();
                    if let Some(cands) = index.get(&key) {
                        for row in cands {
                            if let Some(next) = match_atom(&atom.slots, row, env) {
                                out.push(next);
                            }
                        }
                    }
                }
            }
            Ok(out)
        }
        BodyOp::Prim { id, args, ret } => {
            let mut out = Vec::new();
            for env in envs {
                // Resolve args; an unbound arg means this primitive can't fire
                // for this binding (shouldn't happen for well-formed rules).
                let resolved: Option<Vec<Value>> = args
                    .iter()
                    .map(|s| slot_lookup(s, &|v| env.get(&v).copied()).map(Value::new))
                    .collect();
                let Some(argv) = resolved else { continue };
                let result = eg
                    .db
                    .with_execution_state(|st| st.call_external_func(*id, &argv));
                let Some(result) = result else {
                    // Primitive failed (e.g. `!=` of equal args) — prune.
                    continue;
                };
                // The `ret` slot binds the result or asserts equality.
                match ret {
                    Slot::Var(v) => {
                        let mut next = env.clone();
                        match next.get(v) {
                            Some(&existing) if existing != result.rep() => continue,
                            _ => {
                                next.insert(*v, result.rep());
                            }
                        }
                        out.push(next);
                    }
                    Slot::Const(c) => {
                        if *c == result.rep() {
                            out.push(env);
                        }
                    }
                }
            }
            Ok(out)
        }
    }
}

/// Try to unify `slots` with `row` under `env`. Returns the extended env on
/// success, `None` if a bound var / constant conflicts.
fn match_atom(slots: &[Slot], row: &Row, env: &Env) -> Option<Env> {
    let mut next = env.clone();
    for (i, s) in slots.iter().enumerate() {
        let col = crate::compile::row_col(row, i);
        match s {
            Slot::Const(c) => {
                if *c != col {
                    return None;
                }
            }
            Slot::Var(v) => match next.get(v) {
                Some(&bound) if bound != col => return None,
                _ => {
                    next.insert(*v, col);
                }
            },
        }
    }
    Some(next)
}

/// Execute the head ops for one binding, accumulating writes.
fn apply_head(
    eg: &mut EGraph,
    head: &[HeadOp],
    env: &mut Env,
    writes: &mut Vec<Write>,
    touched: &mut HashSet<FunctionId>,
) -> Result<()> {
    for op in head {
        match op {
            HeadOp::Set { func, slots } => {
                let row = build_row(slots, env)?;
                touched.insert(*func);
                writes.push(Write::Set(*func, row));
            }
            HeadOp::Remove { func, slots } => {
                let key: Vec<u32> = slots
                    .iter()
                    .map(|s| resolve(s, env))
                    .collect::<Result<_>>()?;
                touched.insert(*func);
                writes.push(Write::Remove(*func, key));
            }
            HeadOp::Subsume { func, .. } => {
                // Subsumption is not tracked; treat as a no-op (the row stays
                // present). `supports_subsumption()` is false, so the frontend
                // never relies on subsumed-row filtering on this backend.
                touched.insert(*func);
            }
            HeadOp::Lookup { func, args, ret } => {
                let key: Vec<Value> = args
                    .iter()
                    .map(|s| resolve(s, env).map(Value::new))
                    .collect::<Result<_>>()?;
                let val = lookup_or_create(eg, *func, &key);
                env.insert(*ret, val.rep());
            }
            HeadOp::Call { id, args, ret } => {
                let argv: Vec<Value> = args
                    .iter()
                    .map(|s| resolve(s, env).map(Value::new))
                    .collect::<Result<_>>()?;
                let result = eg
                    .db
                    .with_execution_state(|st| st.call_external_func(*id, &argv));
                if let Some(v) = result {
                    env.insert(*ret, v.rep());
                }
                // A `None` result (primitive failure) in an action is a no-op
                // here; the binding simply has no value for `ret`. Real failures
                // surface as panics through `PanicFunc`.
            }
            HeadOp::Union { .. } => {
                // Term-encoding mode emits unions as `(set (@uf …))` writes, not
                // trait `union` calls (mirrors the DuckDB backend). A direct
                // `union` therefore never reaches a tractable program; ignore.
            }
            HeadOp::Panic(msg) => {
                return Err(anyhow!("{msg}"));
            }
        }
    }
    Ok(())
}

/// Build a full row from head-action slots under `env`.
fn build_row(slots: &[Slot], env: &Env) -> Result<Row> {
    let vals: Vec<u32> = slots
        .iter()
        .map(|s| resolve(s, env))
        .collect::<Result<_>>()?;
    Ok(vals.into_boxed_slice())
}

/// Resolve a slot to a concrete value, erroring if it is an unbound variable.
fn resolve(s: &Slot, env: &Env) -> Result<u32> {
    slot_lookup(s, &|v| env.get(&v).copied())
        .ok_or_else(|| anyhow!("unbound variable {s:?} in rule head"))
}

/// Look up the output of `func` for input `key`. If absent, create the row with
/// a fresh id (eq-sort constructor semantics — mirrors `add_term`). The created
/// row is written directly into the mirror so subsequent lookups in the same
/// iteration see it (hash-cons).
fn lookup_or_create(eg: &mut EGraph, func: FunctionId, key: &[Value]) -> Value {
    let info = eg.info(func);
    let inputs_len = info.arity.saturating_sub(1);
    if let Some(set) = eg.mirror.get(&func) {
        for row in set.iter() {
            if (0..inputs_len).all(|i| crate::compile::row_col(row, i) == key[i].rep()) {
                return Value::new(crate::compile::row_col(row, inputs_len));
            }
        }
    }
    // Not present: allocate a fresh id and insert the row.
    let id = eg.fresh_id_internal();
    let mut full: Vec<u32> = key.iter().map(|v| v.rep()).collect();
    full.push(id);
    eg.mirror
        .entry(func)
        .or_default()
        .insert(full.into_boxed_slice());
    Value::new(id)
}
