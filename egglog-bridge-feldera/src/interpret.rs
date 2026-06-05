//! Host-side rule interpreter for the Feldera backend (Milestones 3–5).
//!
//! One `run_rules` call = **one bounded egglog iteration**. The interpreter:
//!
//! 1. snapshots the relation mirror (the read view for this iteration — all
//!    rules see the same pre-iteration state, matching egglog's semi-naive
//!    "match against the old database, then apply" model for a single hop);
//! 2. for each rule, runs a **seminaive join** over the body's table atoms (see
//!    "Seminaive incrementality" below), threading a variable→value binding
//!    environment, evaluating primitive body atoms (`BodyOp::Prim`) against the
//!    embedded `Database` (guards prune, bindings extend);
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
//!
//! ## Seminaive incrementality (Milestone 5 — the headline result)
//!
//! Each iteration fires a rule only on bindings that involve **at least one
//! newly-derived fact** (the *delta*), exactly mirroring egglog's seminaive
//! evaluation — and exactly the incremental view maintenance DBSP exists for. A
//! rule that already fired on the existing facts does NOT re-fire, so a fixpoint
//! (`changed == false`) is actually reached instead of oscillating forever
//! against a cleanup rule.
//!
//! The delta of rule `r` over relation `f` is `read[f] \ seen[r][f]`, where
//! `seen[r][f]` is the snapshot `r` has already matched (per-rule, see
//! `EGraph::seen`). The body join is run as the standard seminaive union over
//! atom positions:
//!
//! ```text
//! Δbindings(r) = ⋃_j  A_1(full) ⋈ … ⋈ A_j(delta) ⋈ … ⋈ A_k(full)
//! ```
//!
//! i.e. for each table-atom position `j`, atom `j` ranges over only the delta
//! rows while the others range over the full relation; the union over `j` is the
//! set of all bindings touching ≥1 new fact. After `r` has generated its
//! bindings, `seen[r][f]` is advanced to the *start-of-iteration* snapshot
//! (`read[f]`) — never the post-write mirror — so a row that is deleted and
//! later re-added reappears in `r`'s delta and re-fires (load-bearing for
//! rebuild's rule-driven retraction).

use anyhow::{anyhow, Result};
use egglog_backend_trait::{FunctionId, Value};
use egglog_numeric_id::NumericId;
use hashbrown::{HashMap, HashSet};

use crate::compile::{slot_lookup, BodyOp, HeadOp, Row, RuleIr, Slot};
use crate::dbsp_join;
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
    // Iteration-scoped `key -> output` index for `lookup_or_create` (eq-sort
    // constructor hash-cons). Built lazily per function so repeated lookups in
    // one iteration are O(1) instead of rescanning the growing mirror each time.
    let mut lookup_index: HashMap<FunctionId, HashMap<Box<[u32]>, u32>> = HashMap::new();

    // Collect each rule's index + IR up front (clone to avoid borrow conflicts
    // while we also mutate the db / mirror via lookups). The index lets us
    // advance the per-rule seminaive `seen` snapshot after the rule fires.
    let rules: Vec<(usize, RuleIr)> = rule_idxs
        .iter()
        .filter_map(|&i| eg.rules.get(i).and_then(|r| r.clone()).map(|r| (i, r)))
        .collect();

    // Shared start-of-iteration snapshots, built lazily once per function and
    // reused (by refcount) for every rule's `seen` advance this call.
    let mut shared_snapshot: HashMap<FunctionId, std::rc::Rc<HashSet<Row>>> = HashMap::new();
    for (idx, rule) in &rules {
        // The relations this rule's body reads, and the seminaive delta of each
        // (rows present now that this rule has not yet matched against).
        let body_funcs: Vec<FunctionId> = rule
            .body
            .iter()
            .filter_map(|op| match op {
                BodyOp::Atom(a) => Some(a.func),
                BodyOp::Prim { .. } => None,
            })
            .collect();
        let delta: HashMap<FunctionId, HashSet<Row>> = body_funcs
            .iter()
            .map(|&f| {
                let cur = read.get(&f).map(|v| v.as_slice()).unwrap_or(&[]);
                let seen = eg.seen.get(idx).and_then(|m| m.get(&f));
                let d: HashSet<Row> = cur
                    .iter()
                    .filter(|r| seen.map(|s| !s.contains(*r)).unwrap_or(true))
                    .cloned()
                    .collect();
                (f, d)
            })
            .collect();

        // The seminaive binding set: union over table-atom positions of the join
        // with exactly that atom restricted to the delta. Each variant is run
        // through the DBSP engine (if eligible) or the host nested-loop fallback.
        let bindings = seminaive_bindings(eg, &read, &delta, rule)?;

        for mut env in bindings {
            apply_head(
                eg,
                &rule.head,
                &mut env,
                &mut writes,
                &mut touched,
                &mut lookup_index,
            )?;
        }

        // Advance this rule's seen snapshot to the start-of-iteration read view
        // (NOT the post-write mirror): the rule has now matched everything
        // currently present. A row deleted+readded later reappears in the delta.
        // Build each touched function's snapshot once (shared by Rc across all
        // rules in this call), then bump the refcount into this rule's seen.
        for &f in &body_funcs {
            if !shared_snapshot.contains_key(&f) {
                let cur: HashSet<Row> = read
                    .get(&f)
                    .map(|v| v.iter().cloned().collect())
                    .unwrap_or_default();
                shared_snapshot.insert(f, std::rc::Rc::new(cur));
            }
        }
        let entry = eg.seen.entry(*idx).or_default();
        for &f in &body_funcs {
            entry.insert(f, std::rc::Rc::clone(&shared_snapshot[&f]));
        }
    }

    // Apply collected writes to the mirror.
    //
    // Removes are BATCHED per function: a `Write::Remove(f, key)` retracts every
    // row of `f` whose key columns equal `key`. Applying each remove with its own
    // `set.retain` scan is O(|removes| · |state|) — quadratic, and the dominant
    // per-iteration cost during rebuild (which retracts many rows at once). We
    // instead collect all retraction keys per function into a hash set keyed by
    // the function's key arity, then do a SINGLE `retain` pass per touched
    // function: O(|state|) total regardless of how many rows are retracted.
    //
    // Removes are applied FIRST (batched), then Sets — preserving the term
    // encoder's `(@uf)` "delete old leader row, set new leader row" ordering
    // (delete-then-set: the set must win). A Set whose key was also retracted
    // this iteration is re-inserted afterward, which is the intended result.
    let mut removes_by_func: HashMap<FunctionId, (usize, HashSet<Box<[u32]>>)> = HashMap::new();
    let mut sets: Vec<(FunctionId, Row)> = Vec::new();
    for w in writes {
        match w {
            Write::Set(f, row) => {
                sets.push((f, row));
            }
            Write::Remove(f, key) => {
                let entry = removes_by_func
                    .entry(f)
                    .or_insert_with(|| (key.len(), HashSet::new()));
                entry.1.insert(key.into_boxed_slice());
            }
        }
    }
    for (f, (keylen, keys)) in removes_by_func {
        if let Some(set) = eg.mirror.get_mut(&f) {
            set.retain(|row| {
                let k: Box<[u32]> = (0..keylen)
                    .map(|i| crate::compile::row_col(row, i))
                    .collect();
                !keys.contains(&k)
            });
        }
    }
    for (f, row) in sets {
        eg.mirror.entry(f).or_default().insert(row);
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
            Ok(step_atom(&atom.slots, rows, envs))
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
                let result = eg.eval_prim_internal(*id, &argv);
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

/// Join a table atom (given by its `slots`) against an explicit `rows` set,
/// extending each incoming env. Hash-joins on already-bound shared columns when
/// possible, otherwise scans. This is the core of both the full-relation match
/// (`step_body`) and the seminaive delta match (`seminaive_bindings`).
fn step_atom(slots: &[Slot], rows: &[Row], envs: Vec<Env>) -> Vec<Env> {
    // JOIN KEY: columns whose slot is a variable already bound in the incoming
    // envs. If non-empty, hash-join on it instead of a full cartesian scan.
    let bound_in_env = |v: u32| envs.first().map(|e| e.contains_key(&v)).unwrap_or(false);
    let key_cols: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter_map(|(i, s)| match s {
            Slot::Var(v) if bound_in_env(*v) => Some(i),
            _ => None,
        })
        .collect();
    let key_vars: Vec<u32> = key_cols
        .iter()
        .map(|&i| match &slots[i] {
            Slot::Var(v) => *v,
            _ => unreachable!(),
        })
        .collect();

    let mut out = Vec::new();
    if key_cols.is_empty() {
        // No shared bound variable: cartesian (correct for the first atom).
        for env in &envs {
            for row in rows {
                if let Some(next) = match_atom(slots, row, env) {
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
                    if let Some(next) = match_atom(slots, row, env) {
                        out.push(next);
                    }
                }
            }
        }
    }
    out
}

/// Compute the **seminaive** binding set for one rule: the union, over each
/// table-atom position `j`, of the body join with atom `j` restricted to its
/// relation's *delta* (rows new to this rule) and every other atom ranging over
/// the full relation. The union is exactly the set of bindings touching at
/// least one newly-derived fact — egglog's seminaive semantics, and DBSP's
/// incremental view maintenance.
///
/// Each delta-atom variant is run through the DBSP dataflow engine when the
/// rule is DBSP-eligible (`dbsp_join`), otherwise through the host nested-loop
/// fallback (the oracle). Body primitives (`!=` guards, value-computing prims)
/// are applied host-side over the produced bindings. Bindings are deduplicated
/// across the `j`-variants (a binding with new facts in two positions appears
/// in two variants).
fn seminaive_bindings(
    eg: &mut EGraph,
    read: &HashMap<FunctionId, Vec<Row>>,
    delta: &HashMap<FunctionId, HashSet<Row>>,
    rule: &RuleIr,
) -> Result<Vec<Env>> {
    // Positions of the table atoms within the body op list.
    let atom_positions: Vec<usize> = rule
        .body
        .iter()
        .enumerate()
        .filter_map(|(i, op)| matches!(op, BodyOp::Atom(_)).then_some(i))
        .collect();

    if atom_positions.is_empty() {
        // Prim-only / constant body: no table atoms to delta over. There is no
        // notion of "new fact" here; evaluate it once (it is bounded and
        // idempotent — these bodies are rare and not on the saturation hot path).
        let mut envs: Vec<Env> = vec![Env::new()];
        for op in &rule.body {
            envs = step_body(eg, read, op, envs)?;
            if envs.is_empty() {
                break;
            }
        }
        return Ok(envs);
    }

    // If NO body relation has any delta rows, the rule cannot produce a new
    // binding this iteration — skip it entirely (the seminaive win).
    let any_delta = atom_positions.iter().any(|&p| {
        if let BodyOp::Atom(a) = &rule.body[p] {
            delta.get(&a.func).map(|d| !d.is_empty()).unwrap_or(false)
        } else {
            false
        }
    });
    if !any_delta {
        return Ok(Vec::new());
    }

    // Try the DBSP-eligible whole-body join once per delta-atom variant.
    let plan = dbsp_join::plan_join(eg, rule);

    let mut seen_bindings: HashSet<Vec<(u32, u32)>> = HashSet::new();
    let mut out: Vec<Env> = Vec::new();

    for (atom_ord, &delta_pos) in atom_positions.iter().enumerate() {
        // Skip a variant whose delta atom has no new rows (it contributes
        // nothing and would just rescan the full relation pointlessly).
        let delta_func = match &rule.body[delta_pos] {
            BodyOp::Atom(a) => a.func,
            _ => unreachable!(),
        };
        if delta.get(&delta_func).map(|d| d.is_empty()).unwrap_or(true) {
            continue;
        }

        let variant_envs = if let Some(plan) = &plan {
            // DBSP path: run the relational join with this atom occurrence fed
            // the delta rows and the others the full relation. `atom_ord` is the
            // occurrence's index within the plan's atom list (plan atoms are in
            // body order, matching `atom_positions`).
            eg.dbsp_rule_runs += 1;
            let bindings = dbsp_join::run_join_seminaive(eg, plan, atom_ord, delta)?;
            let var_order = plan.var_order().to_vec();
            let mut envs: Vec<Env> = Vec::new();
            for bind in bindings {
                let mut env: Env = Env::new();
                for (i, &v) in var_order.iter().enumerate() {
                    env.insert(v, bind[i]);
                }
                // Re-run body primitives host-side (value-computing prims bind
                // fresh vars; `!=` re-filters identically). Table atoms / `!=`
                // were already handled inside the DBSP join.
                let mut es: Vec<Env> = vec![env];
                for op in &rule.body {
                    if let BodyOp::Prim { .. } = op {
                        es = step_body(eg, read, op, es)?;
                    }
                }
                envs.extend(es);
            }
            envs
        } else {
            // Host fallback: nested-loop body scan where the atom at
            // `delta_pos` ranges over the delta rows, all others over `read`.
            eg.host_rule_runs += 1;
            let mut envs: Vec<Env> = vec![Env::new()];
            for (pos, op) in rule.body.iter().enumerate() {
                match op {
                    BodyOp::Atom(atom) => {
                        let rows: Vec<Row> = if pos == delta_pos {
                            delta
                                .get(&atom.func)
                                .map(|d| d.iter().cloned().collect())
                                .unwrap_or_default()
                        } else {
                            read.get(&atom.func).cloned().unwrap_or_default()
                        };
                        envs = step_atom(&atom.slots, &rows, envs);
                    }
                    BodyOp::Prim { .. } => {
                        envs = step_body(eg, read, op, envs)?;
                    }
                }
                if envs.is_empty() {
                    break;
                }
            }
            envs
        };

        // Deduplicate across the delta-position variants.
        for env in variant_envs {
            let mut key: Vec<(u32, u32)> = env.iter().map(|(&k, &v)| (k, v)).collect();
            key.sort_unstable();
            if seen_bindings.insert(key) {
                out.push(env);
            }
        }
    }

    Ok(out)
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
    lookup_index: &mut HashMap<FunctionId, HashMap<Box<[u32]>, u32>>,
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
                let val = lookup_or_create(eg, *func, &key, lookup_index);
                env.insert(*ret, val.rep());
            }
            HeadOp::Call { id, args, ret } => {
                let argv: Vec<Value> = args
                    .iter()
                    .map(|s| resolve(s, env).map(Value::new))
                    .collect::<Result<_>>()?;
                let result = eg.eval_prim_internal(*id, &argv);
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
///
/// `index` is an iteration-scoped `key -> output` map per function. A naive
/// implementation rescans the whole (growing) mirror set on every call, which is
/// O(bindings · state) per iteration — a super-linear blowup on eqsat workloads
/// that create many terms per round. We instead build the index lazily once per
/// function (O(state) the first time it is touched this iteration) and keep it
/// updated as new rows are hash-consed, so each subsequent lookup is O(1).
fn lookup_or_create(
    eg: &mut EGraph,
    func: FunctionId,
    key: &[Value],
    index: &mut HashMap<FunctionId, HashMap<Box<[u32]>, u32>>,
) -> Value {
    let info = eg.info(func);
    let inputs_len = info.arity.saturating_sub(1);
    // Lazily populate the key->output index for this function from the mirror.
    let idx = index.entry(func).or_insert_with(|| {
        let mut m: HashMap<Box<[u32]>, u32> = HashMap::new();
        if let Some(set) = eg.mirror.get(&func) {
            for row in set.iter() {
                let k: Box<[u32]> = (0..inputs_len)
                    .map(|i| crate::compile::row_col(row, i))
                    .collect();
                m.insert(k, crate::compile::row_col(row, inputs_len));
            }
        }
        m
    });
    let k: Box<[u32]> = key.iter().map(|v| v.rep()).collect();
    if let Some(&out) = idx.get(&k) {
        return Value::new(out);
    }
    // Not present: allocate a fresh id, insert the row, and update the index.
    let id = eg.fresh_id_internal();
    idx.insert(k, id);
    let mut full: Vec<u32> = key.iter().map(|v| v.rep()).collect();
    full.push(id);
    eg.mirror
        .entry(func)
        .or_default()
        .insert(full.into_boxed_slice());
    Value::new(id)
}
