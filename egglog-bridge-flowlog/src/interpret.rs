//! Host-side iteration driver for the FlowLog backend.
//!
//! One `run_rules` call = **one bounded egglog iteration**. The body join runs
//! on the in-process, build-once, epoch-driven raw differential-dataflow
//! dataflow (`crate::dd_native`); this module owns the orchestration around it:
//!
//! 1. snapshot the relation mirror (the read view for this iteration — all rules
//!    match against the same pre-iteration state, egglog's semi-naive "match the
//!    old database, then apply" model for a single hop);
//! 2. for each rule, drive its persistent DD join with the per-relation signed
//!    delta vs. what that join was last fed, then re-run any body primitives
//!    (`!=` guards, value-computing prims) host-side over the produced bindings;
//! 3. for every surviving binding, execute the head ops in order — `set` /
//!    `remove` / `subsume` writes, RHS `lookup` (eq-sort constructor: create on
//!    miss), RHS primitive `call`, `union`, `panic`;
//! 4. apply all collected writes/removes to the mirror and resolve
//!    functional-dependency conflicts per each touched function's merge mode.
//!
//! ## The engine split (mirrors Feldera Stage C)
//!
//! The relational table-atom join is the ONLY thing on the engine, and it is the
//! ONLY join path — there is no host nested-loop fallback. Any rule the DD plan
//! cannot lower (a binding row exceeding the fixed width cap [`dd_native::W`], or
//! any shape `plan_join` rejects) `panic!`s with a specific reason. The primitive
//! tail + head actions are applied HOST-side here, exactly mirroring Feldera's
//! "table join on the engine, prim tail host-side" split.
//!
//! Primitives are invoked through `Database::with_execution_state`, so they see
//! the same interned base `Value`s the frontend created — giving the FlowLog
//! backend bit-for-bit value parity with the reference backend (no `eval_prim`
//! trait change — flowlog keeps its zero-trait-change posture).

use anyhow::{anyhow, Result};
use egglog_backend_trait::{FunctionId, Value};
use egglog_numeric_id::NumericId;
use hashbrown::{HashMap, HashSet};

use crate::compile::{row_col, slot_lookup, BodyOp, HeadOp, Row, RuleIr, Slot};
use crate::EGraph;

/// Binding environment: variable id → bound `u32` value.
pub(crate) type Env = HashMap<u32, u32>;

/// A pending write to apply after all matches are computed.
enum Write {
    /// Insert/overwrite a full row.
    Set(FunctionId, Row),
    /// Retract by key (the slots address inputs for a function, whole row for a
    /// relation).
    Remove(FunctionId, Vec<u32>),
}

/// One bounded egglog iteration with the body join running on the in-process,
/// build-once, epoch-driven raw differential-dataflow dataflow
/// (`crate::dd_native`). This is the FlowLog analog of the Feldera backend's
/// `interpret::run_iteration` + `persistent_bindings`.
///
/// Per rule: compute the signed `+/-` delta of each body relation vs the rows
/// last fed to that rule's persistent DD join, `step` the join (which feeds ONLY
/// the delta into never-cleared InputSessions — genuinely incremental), turn the
/// positive binding deltas into envs, re-run body prims host-side (value prims /
/// guards the engine keeps off-circuit), then apply head actions. Writes +
/// FD-merge are applied so results are bit-exact.
pub fn run_iteration(eg: &mut EGraph, rule_idxs: &[usize]) -> Result<bool> {
    // Snapshot the read view: rules match against the pre-iteration mirror.
    //
    // The snapshot shares each function's row set by `Rc` rather than
    // deep-cloning every row: this is O(#functions), not O(state). Mutations to
    // the mirror this call (head writes, hash-cons in `lookup_or_create`, merge
    // resolution) go through `Rc::make_mut`, which copy-on-writes only the
    // functions actually changed while this snapshot is alive — so `read` keeps
    // the start-of-call contents and rules all match the pre-iteration state.
    let read: HashMap<FunctionId, std::rc::Rc<HashSet<Row>>> = eg
        .mirror
        .iter()
        .map(|(f, set)| (*f, std::rc::Rc::clone(set)))
        .collect();

    // Snapshot the fresh-id counter: any hash-cons (`lookup_or_create`) this
    // call advances it, the O(1) signal that a new term row was created.
    let next_id_at_start = eg.next_id;

    let mut writes: Vec<Write> = Vec::new();
    let mut touched: HashSet<FunctionId> = HashSet::new();
    // Iteration-scoped `key -> output` index for `lookup_or_create` (eq-sort
    // constructor hash-cons). Built lazily per function so repeated lookups in
    // one iteration are O(1) instead of rescanning the growing mirror each time.
    let mut lookup_index: HashMap<FunctionId, HashMap<Box<[u32]>, u32>> = HashMap::new();

    let rules: Vec<(usize, RuleIr)> = rule_idxs
        .iter()
        .filter_map(|&i| eg.rules.get(i).and_then(|r| r.clone()).map(|r| (i, r)))
        .collect();

    for (idx, rule) in &rules {
        let envs = dd_native_bindings(eg, &read, *idx, rule)?;
        for mut env in envs {
            apply_head(
                eg,
                &rule.head,
                &mut env,
                &mut writes,
                &mut touched,
                &mut lookup_index,
            )?;
        }
    }

    // Apply collected writes to the mirror.
    //
    // Removes are BATCHED per function: applying each `Write::Remove` with its
    // own `set.retain` scan is O(|removes| · |state|) — quadratic. We collect all
    // retraction keys per function into a hash set, then do a SINGLE `retain`
    // pass per touched function: O(|state|) total. Removes are applied FIRST
    // (batched), then Sets — preserving the term encoder's `(@uf)` "delete old
    // leader, set new leader" delete-then-set ordering.
    //
    // `changed` is computed INCREMENTALLY as writes land (O(delta)), not via a
    // full before/after content compare. A hash-cons in `lookup_or_create`
    // always allocates a fresh id, so any term created this call advances
    // `next_id` — that alone is a real mirror change.
    let mut changed = eg.next_id != next_id_at_start;
    let mut removes_by_func: HashMap<FunctionId, (usize, HashSet<Box<[u32]>>)> = HashMap::new();
    let mut sets: Vec<(FunctionId, Row)> = Vec::new();
    for w in writes {
        match w {
            Write::Set(f, row) => sets.push((f, row)),
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
            let before_len = set.len();
            std::rc::Rc::make_mut(set).retain(|row| {
                let k: Box<[u32]> = (0..keylen).map(|i| row_col(row, i)).collect();
                !keys.contains(&k)
            });
            changed |= set.len() != before_len;
        }
    }
    let mut touched_keys: HashMap<FunctionId, HashSet<Vec<u32>>> = HashMap::new();
    for (f, row) in sets {
        let inputs_len = eg.info(f).arity.saturating_sub(1);
        let key: Vec<u32> = (0..inputs_len).map(|i| row_col(&row, i)).collect();
        let inserted = std::rc::Rc::make_mut(eg.mirror.entry(f).or_default()).insert(row);
        changed |= inserted;
        touched_keys.entry(f).or_default().insert(key);
    }

    // Resolve FD conflicts on every function a head action wrote to (a `set`
    // can introduce two rows sharing a key that must merge per the merge mode).
    let empty_keys: HashSet<Vec<u32>> = HashSet::new();
    for &f in &touched {
        let keys = touched_keys.get(&f).unwrap_or(&empty_keys);
        changed |= eg.resolve_merge(f, keys);
    }
    Ok(changed)
}

/// Run rule `idx`'s body join on its persistent in-process DD dataflow, returning
/// the positive binding deltas as envs ready for head application. Atom-less
/// rules fire their single unconditional binding once (tracked via `seen`).
fn dd_native_bindings(
    eg: &mut EGraph,
    read: &HashMap<FunctionId, std::rc::Rc<HashSet<Row>>>,
    idx: usize,
    rule: &RuleIr,
) -> Result<Vec<Env>> {
    use crate::dd_native;

    let has_atoms = rule.body.iter().any(|op| matches!(op, BodyOp::Atom(_)));
    if !has_atoms {
        // Atom-less rule (`(rule () …)`): fire once. Reuse `seen` as a fired-set
        // marker (presence of an entry for this rule index = already fired). The
        // DD dataflow has no input relation to drive an atom-less body, so this
        // is the one rule shape evaluated purely host-side (no table join).
        if eg.seen.contains_key(&idx) {
            return Ok(Vec::new());
        }
        eg.seen.insert(idx, ());
        let mut envs: Vec<Env> = vec![Env::new()];
        for op in &rule.body {
            envs = step_prim(eg, op, envs)?;
            if envs.is_empty() {
                break;
            }
        }
        return Ok(envs);
    }

    // Plan + build the persistent DD join once. There is NO host fallback —
    // any shape `plan_join` rejects (a binding row exceeding the fixed width
    // cap `dd_native::W`, an over-wide atom arity) `panic!`s with the reason,
    // mirroring the Feldera backend's Stage-C "ineligible ⇒ panic" posture.
    let plan = match dd_native::plan_join(rule) {
        Ok(p) => p,
        Err(reason) => panic!(
            "FlowLog DD join cannot lower rule {:?}: {reason} \
             (no host fallback; the DD dataflow is the only join path)",
            rule.name
        ),
    };
    let var_order = plan.var_order().to_vec();
    if !eg.dd_native.contains_key(&idx) {
        let pj = dd_native::PersistentDdJoin::build(&plan)?;
        eg.dd_native.insert(idx, pj);
    }

    let body_funcs: Vec<FunctionId> = rule
        .body
        .iter()
        .filter_map(|op| match op {
            BodyOp::Atom(a) => Some(a.func),
            BodyOp::Prim { .. } => None,
        })
        .collect();

    // Signed delta vs the rows last fed to THIS rule's join (Feldera `fed`).
    let empty_set: std::rc::Rc<HashSet<Row>> = std::rc::Rc::new(HashSet::new());
    let mut delta: HashMap<FunctionId, Vec<(Vec<u32>, isize)>> = HashMap::new();
    {
        let fed = eg.dd_native_fed.entry(idx).or_default();
        for &f in &body_funcs {
            let cur = read.get(&f).cloned().unwrap_or_else(|| empty_set.clone());
            let prev = fed.entry(f).or_insert_with(|| empty_set.clone());
            if std::rc::Rc::ptr_eq(&cur, prev) {
                *prev = cur;
                continue;
            }
            let mut rows: Vec<(Vec<u32>, isize)> = Vec::new();
            for r in cur.iter() {
                if !prev.contains(r) {
                    rows.push((r.to_vec(), 1));
                }
            }
            for r in prev.iter() {
                if !cur.contains(r) {
                    rows.push((r.to_vec(), -1));
                }
            }
            if !rows.is_empty() {
                delta.insert(f, rows);
            }
            *prev = cur;
        }
    }

    eg.dd_rule_runs += 1;
    let bindings = {
        let pj = eg
            .dd_native
            .get_mut(&idx)
            .expect("persistent dd join present");
        pj.step(&delta)?
    };

    // Turn positive binding deltas into envs; re-run body prims host-side over
    // them (value prims bind fresh vars; `!=`/guard prims re-filter). Negative
    // weights are integral bookkeeping (a body row retracted) — egglog heads are
    // monotone-fire, so we do NOT re-fire the head on disappearance.
    let mut out: Vec<Env> = Vec::new();
    for (bind, w) in &bindings {
        if *w <= 0 {
            continue;
        }
        let mut env: Env = Env::new();
        for (i, &v) in var_order.iter().enumerate() {
            env.insert(v, bind[i]);
        }
        let mut es: Vec<Env> = vec![env];
        for op in &rule.body {
            if let BodyOp::Prim { .. } = op {
                es = step_prim(eg, op, es)?;
            }
        }
        out.extend(es);
    }
    Ok(out)
}

/// Evaluate a primitive body op over each binding env, returning the new list of
/// envs. A value-computing prim binds (or checks) its return var; a guard prim
/// (`!=`) that fails prunes the env. Table atoms are NOT handled here — they run
/// on the DD dataflow; this is only the host-side primitive tail.
pub(crate) fn step_prim(eg: &mut EGraph, op: &BodyOp, envs: Vec<Env>) -> Result<Vec<Env>> {
    let BodyOp::Prim { id, args, ret } = op else {
        unreachable!("step_prim called on a non-primitive body op");
    };
    let mut out = Vec::new();
    for env in envs {
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
                // A `None` result (primitive failure) in an action is a no-op;
                // real failures surface as panics through `PanicFunc`.
            }
            HeadOp::Union { .. } => {
                // Term-encoding mode emits unions as `(set (@uf …))` writes, not
                // trait `union` calls (mirrors the DuckDB/Feldera backends). A
                // direct `union` therefore never reaches a tractable program.
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
fn lookup_or_create(
    eg: &mut EGraph,
    func: FunctionId,
    key: &[Value],
    index: &mut HashMap<FunctionId, HashMap<Box<[u32]>, u32>>,
) -> Value {
    let info = eg.info(func);
    let inputs_len = info.arity.saturating_sub(1);
    // Lazily build the key->output index for this function from the mirror so
    // repeated lookups within one iteration are O(1) instead of O(state) scans.
    let idx = index.entry(func).or_insert_with(|| {
        let mut m: HashMap<Box<[u32]>, u32> = HashMap::new();
        if let Some(set) = eg.mirror.get(&func) {
            for row in set.iter() {
                let k: Box<[u32]> = (0..inputs_len).map(|i| row_col(row, i)).collect();
                m.insert(k, row_col(row, inputs_len));
            }
        }
        m
    });
    let k: Box<[u32]> = key.iter().map(|v| v.rep()).collect();
    if let Some(&out) = idx.get(&k) {
        return Value::new(out);
    }
    let id = eg.fresh_id_internal();
    idx.insert(k, id);
    let mut full: Vec<u32> = key.iter().map(|v| v.rep()).collect();
    full.push(id);
    let row: Row = full.into_boxed_slice();
    std::rc::Rc::make_mut(eg.mirror.entry(func).or_default()).insert(row);
    Value::new(id)
}
