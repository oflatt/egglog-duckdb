//! Per-iteration driver for the Feldera backend (#23 Stage C complete).
//!
//! One `run_rules` call = **one bounded egglog iteration**. The driver:
//!
//! 1. snapshots the relation mirror (the read view for this iteration — all
//!    rules see the same pre-iteration state, matching egglog's semi-naive
//!    "match against the old database, then apply" model for a single hop);
//! 2. for each rule, runs its body join on the rule's **persistent DBSP
//!    circuit** ([`crate::dbsp_join::PersistentJoin`]) — the ONLY join path. The
//!    circuit is fed the per-relation signed delta vs the rows last fed to it;
//!    its incremental join + integral do the seminaive bookkeeping and handle
//!    retraction natively (signed weights), so there is no host nested-loop and
//!    no per-rule host `seen` set. Pure value-computing body prims are evaluated
//!    ON-CIRCUIT through the shared [`crate::PrimEngine`] (rep-comparable ones
//!    are inlined as `filter`/`map`; everything else — base-value
//!    `ordering-min/max`, f64 `!=`, `+`/`int-div`/`string-concat`, … — runs the
//!    REAL prim on actual values via the call-prim path). Atom-less rules
//!    (`(rule () …)` / `eval_actions`) fire their single unconditional binding
//!    once. A genuinely-ineligible rule (an IMPURE body prim, or a shape above
//!    the [`crate::dbsp_join::JOIN_WIDTH`] row cap) PANICS — there is no host
//!    fallback;
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
use dbsp::ZWeight;
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
    let prof = std::env::var("FELDERA_PROFILE").is_ok();
    let t_read = std::time::Instant::now();
    // Snapshot the read view: rules match against the pre-iteration mirror.
    //
    // The snapshot shares each function's row set by `Rc` rather than deep-cloning
    // every row: this is O(#functions), not O(state). Mutations to the mirror this
    // call (head writes, hash-cons in `lookup_or_create`, merge resolution) go
    // through `Rc::make_mut`, which copy-on-writes only the functions actually
    // changed while this snapshot is alive — so `read` keeps the start-of-call
    // contents and rules all match the pre-iteration state, exactly as before, but
    // without paying for a full-mirror clone every call.
    let read: HashMap<FunctionId, std::rc::Rc<HashSet<Row>>> = eg
        .mirror
        .iter()
        .map(|(f, set)| (*f, std::rc::Rc::clone(set)))
        .collect();
    if prof {
        let n: usize = read.values().map(|v| v.len()).sum();
        eg.prof_read_clone += t_read.elapsed();
        eg.prof_read_rows += n as u64;
    }

    // Snapshot the fresh-id counter: any hash-cons (`lookup_or_create`) this call
    // advances it, which is the O(1) signal that a new term row was created.
    let next_id_at_start = eg.next_id;

    let mut writes: Vec<Write> = Vec::new();
    let mut touched: HashSet<FunctionId> = HashSet::new();
    // Iteration-scoped `key -> output` index for `lookup_or_create` (eq-sort
    // constructor hash-cons). Built lazily per function so repeated lookups in
    // one iteration are O(1) instead of rescanning the growing mirror each time.
    let mut lookup_index: HashMap<FunctionId, HashMap<Box<[u32]>, u32>> = HashMap::new();

    // Collect each rule's index + IR up front (clone to avoid borrow conflicts
    // while we also mutate the db / mirror via lookups).
    let rules: Vec<(usize, RuleIr)> = rule_idxs
        .iter()
        .filter_map(|&i| eg.rules.get(i).and_then(|r| r.clone()).map(|r| (i, r)))
        .collect();

    // The persistent DBSP circuit is the ONLY join path (the host nested-loop +
    // `seen` fallback was retired): every rule runs its body join on a per-rule
    // persistent circuit fed signed deltas, which does the seminaive bookkeeping
    // and retraction natively (its integral IS the `seen`). Atom-less rules fire
    // their single unconditional binding once. A genuinely-ineligible rule (an
    // IMPURE body prim, or a shape above the row-width cap) PANICS — there is no
    // longer a graceful fallback.
    // Partition the ruleset into ATOM-LESS rules (fired once via the dedicated
    // helper) and ATOM-BEARING rules (joined on the FUSED per-ruleset circuit,
    // ONE transaction for the whole set — see `fused_bindings`).
    let mut atom_rules: Vec<(usize, &RuleIr)> = Vec::new();
    for (idx, rule) in &rules {
        let has_atoms = rule.body.iter().any(|op| matches!(op, BodyOp::Atom(_)));
        if has_atoms {
            atom_rules.push((*idx, rule));
        } else {
            let envs = atomless_bindings(eg, *idx, rule)?;
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
    }

    // Run all atom-bearing rules' joins in ONE fused transaction, then apply
    // each rule's head over its bindings.
    let fused = fused_bindings(eg, &read, &atom_rules)?;
    for ((_idx, rule), envs) in atom_rules.iter().zip(fused) {
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
    let t_apply = std::time::Instant::now();
    // `changed` is computed INCREMENTALLY as writes land (O(delta)), not via a
    // full before/after content compare of every function (O(state)). A hash-cons
    // in `lookup_or_create` always allocates a fresh id, so any term created this
    // call advances `next_id` — that alone is a real mirror change.
    let mut changed = eg.next_id != next_id_at_start;
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
            let before_len = set.len();
            std::rc::Rc::make_mut(set).retain(|row| {
                let k: Box<[u32]> = (0..keylen)
                    .map(|i| crate::compile::row_col(row, i))
                    .collect();
                !keys.contains(&k)
            });
            // A retraction that actually removed a row is a real change.
            changed |= set.len() != before_len;
        }
    }
    // Per-function set of input keys touched by a `set` this call — the only
    // keys that can newly conflict and need merge resolution.
    let mut touched_keys: HashMap<FunctionId, HashSet<Vec<u32>>> = HashMap::new();
    for (f, row) in sets {
        let inputs_len = eg.info(f).arity.saturating_sub(1);
        let key: Vec<u32> = (0..inputs_len)
            .map(|i| crate::compile::row_col(&row, i))
            .collect();
        // `insert` returns true iff the row was genuinely new (set a row that
        // already exists ⇒ no content change, so don't flag `changed`).
        let inserted = std::rc::Rc::make_mut(eg.mirror.entry(f).or_default()).insert(row);
        changed |= inserted;
        touched_keys.entry(f).or_default().insert(key);
    }

    if prof {
        eg.prof_apply += t_apply.elapsed();
    }

    // Resolve FD conflicts on every function that a head action wrote to (a
    // `set` can introduce two rows sharing a key that must be merged per the
    // function's merge mode). Resolution is INCREMENTAL — only the keys whose
    // rows were touched this call can have a new conflict, and `resolve_merge`
    // reports whether it actually collapsed/changed any row (another real
    // change source for the saturation loop).
    let t_merge = std::time::Instant::now();
    let empty_keys: HashSet<Vec<u32>> = HashSet::new();
    for &f in &touched {
        let keys = touched_keys.get(&f).unwrap_or(&empty_keys);
        changed |= eg.resolve_merge(f, keys);
    }
    if prof {
        eg.prof_merge += t_merge.elapsed();
    }

    // `change_detect` is now folded incrementally into apply/merge (O(delta));
    // there is no separate full before/after compare to time.
    Ok(changed)
}

/// Evaluate one body PRIMITIVE op against the current binding envs, extending /
/// pruning each. Used ONLY by the atom-less rule path (`persistent_bindings`):
/// an atom-less rule has no table atoms, so its body is exclusively primitives
/// (a guard prunes, a value prim binds a var the head reads). A body `Atom` is
/// unreachable here — the persistent DBSP circuit handles all atom-bearing
/// rules.
fn eval_body_prim(eg: &mut EGraph, op: &BodyOp, envs: Vec<Env>) -> Result<Vec<Env>> {
    let BodyOp::Prim { id, args, ret } = op else {
        unreachable!("atom-less body must contain only primitives");
    };
    let mut out = Vec::new();
    for env in envs {
        // Resolve args; an unbound arg means this primitive can't fire for this
        // binding (shouldn't happen for well-formed rules).
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

/// Fire an ATOM-LESS rule (`(rule () …)` / `eval_actions` / `eval_resolved_expr`)
/// once: there is no body relation to drive a join, so the (trivially-satisfied)
/// empty binding fires EXACTLY ONCE. Evaluate the body prims once over the empty
/// env (a guard may prune; a value prim binds a var the head reads) and record
/// that the rule has fired so it never re-fires. NOT a join.
fn atomless_bindings(eg: &mut EGraph, idx: usize, rule: &RuleIr) -> Result<Vec<Env>> {
    if eg.atomless_fired.contains(&idx) {
        return Ok(Vec::new());
    }
    // An impure body prim here is still safe (it runs exactly once), so we do
    // not gate on purity for the atom-less case.
    let mut envs: Vec<Env> = vec![Env::new()];
    for op in &rule.body {
        envs = eval_body_prim(eg, op, envs)?;
        if envs.is_empty() {
            break;
        }
    }
    eg.atomless_fired.insert(idx);
    Ok(envs)
}

/// Run ALL atom-bearing rules of a ruleset on ONE FUSED DBSP circuit (#23
/// transaction-count fix): a single circuit per ruleset, one shared input
/// z-set per body relation, each rule a parallel join sub-stream with its own
/// output handle — clocked in a SINGLE `transaction()` per `run_rules` call.
/// This collapses the ruleset's R per-rule transactions (the dominant fixed
/// per-transaction clocking cost) into one.
///
/// Returns one env vector per rule, in the SAME order as `atom_rules`. Semantics
/// are identical to running each rule's join on its own persistent circuit: the
/// fused circuit's shared-input integrals do the seminaive bookkeeping and
/// retraction (signed weights); positive binding weights are new matches,
/// negatives are integral bookkeeping (no head re-fire). A genuinely-ineligible
/// rule (an IMPURE body prim, or a shape above the row-width cap) PANICS — there
/// is no host fallback.
// The `FELDERA_DEBUG_COUNTS` block below is an env-gated stderr diagnostic (off
// in normal runs); `eprintln!` is intentional.
#[allow(clippy::disallowed_macros)]
fn fused_bindings(
    eg: &mut EGraph,
    read: &HashMap<FunctionId, std::rc::Rc<HashSet<Row>>>,
    atom_rules: &[(usize, &RuleIr)],
) -> Result<Vec<Vec<Env>>> {
    if atom_rules.is_empty() {
        return Ok(Vec::new());
    }

    // The fused circuit is keyed by the sorted rule-index list of this ruleset.
    let key: Vec<usize> = {
        let mut k: Vec<usize> = atom_rules.iter().map(|(i, _)| *i).collect();
        k.sort_unstable();
        k
    };

    // Build the fused circuit once for this ruleset. Plan every rule's join; an
    // ineligible rule PANICS (the persistent circuit is the only join path).
    if !eg.fused.contains_key(&key) {
        let mut plans: Vec<(usize, dbsp_join::JoinPlan)> = Vec::with_capacity(atom_rules.len());
        for (idx, rule) in atom_rules {
            let plan = match dbsp_join::plan_join(eg, rule) {
                Ok(p) => p,
                Err(reason) => panic!(
                    "unsupported on feldera persistent engine: {reason} (rule {:?})",
                    rule.name
                ),
            };
            plans.push((*idx, plan));
        }
        let engine = eg.prim_engine();
        let fj = dbsp_join::FusedJoin::build(&plans, &engine)?;
        eg.fused.insert(key.clone(), fj);
    }

    // The fused circuit shares one input per relation across all its rules, so
    // the body relations are the UNION of every rule's body relations.
    let mut body_funcs: Vec<FunctionId> = Vec::new();
    for (_idx, rule) in atom_rules {
        for op in &rule.body {
            if let BodyOp::Atom(a) = op {
                if !body_funcs.contains(&a.func) {
                    body_funcs.push(a.func);
                }
            }
        }
    }

    // Compute the +/- delta vs the last-fed view per body relation (one fed view
    // per fused circuit), and advance it to the current read view. Same
    // `Rc`-ptr-eq fast path as the per-rule code: an untouched function is
    // skipped in O(1).
    let prof = std::env::var("FELDERA_PROFILE").is_ok();
    let diag = std::env::var("FELDERA_DEBUG_COUNTS").is_ok();
    let t_diff = std::time::Instant::now();
    let mut delta: HashMap<FunctionId, Vec<(Vec<u32>, ZWeight)>> = HashMap::new();
    let mut added_delta: HashMap<FunctionId, HashSet<Row>> = HashMap::new();
    let empty_set: std::rc::Rc<HashSet<Row>> = std::rc::Rc::new(HashSet::new());
    {
        let fed = eg.fed_fused.entry(key.clone()).or_default();
        for &f in &body_funcs {
            let cur = read.get(&f).cloned().unwrap_or_else(|| empty_set.clone());
            let prev = fed.entry(f).or_insert_with(|| empty_set.clone());
            if std::rc::Rc::ptr_eq(&cur, prev) {
                *prev = cur;
                continue;
            }
            let mut rows: Vec<(Vec<u32>, ZWeight)> = Vec::new();
            let mut added: HashSet<Row> = HashSet::new();
            for r in cur.iter() {
                if !prev.contains(r) {
                    rows.push((r.to_vec(), 1));
                    if diag {
                        added.insert(r.clone());
                    }
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
            if !added.is_empty() {
                added_delta.insert(f, added);
            }
            *prev = cur;
        }
    }
    if prof {
        eg.prof_fed_diff += t_diff.elapsed();
    }

    eg.dbsp_rule_runs += atom_rules.len() as u64;
    let t_step = std::time::Instant::now();
    let per_rule = {
        let fj = eg.fused.get_mut(&key).expect("fused circuit present");
        fj.step(&delta)?
    };
    if prof {
        eg.prof_circuit_step += t_step.elapsed();
        // ONE transaction per fused step (and only when a delta was pushed).
        if !delta.is_empty() {
            eg.prof_transactions += 1;
        }
    }

    // Route each fused rule's bindings back to its env vector. The fused circuit
    // reports rules in the SAME order as the `plans`/`atom_rules` slice it was
    // built from (its `rule_indices()` mirror that), and `atom_rules` here is the
    // same slice in the same order, so positions line up directly.
    debug_assert_eq!(
        eg.fused.get(&key).map(|f| f.rule_indices()),
        Some(atom_rules.iter().map(|(i, _)| *i).collect::<Vec<_>>()),
        "fused rule order must match atom_rules order"
    );
    let mut out: Vec<Vec<Env>> = Vec::with_capacity(atom_rules.len());
    for (pos, (_idx, rule)) in atom_rules.iter().enumerate() {
        let var_order = eg
            .fused
            .get(&key)
            .expect("fused circuit present")
            .var_order_at(pos)
            .to_vec();
        let bindings = &per_rule[pos];
        let mut envs: Vec<Env> = Vec::new();
        let mut neg_count = 0usize;
        for (bind, w) in bindings {
            if *w <= 0 {
                neg_count += 1;
                continue;
            }
            let mut env: Env = Env::new();
            for (i, &v) in var_order.iter().enumerate() {
                env.insert(v, bind[i]);
            }
            envs.push(env);
        }
        if diag {
            let added_n: usize = added_delta.values().map(|s| s.len()).sum();
            eprintln!(
                "[CNT] rule={} added={} pos_emit={} neg_emit={}",
                rule.name,
                added_n,
                envs.len(),
                neg_count
            );
        }
        out.push(envs);
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
    let identity_on_miss = info.identity_on_miss;
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
    // Identity-on-miss ("lookup-or-self"): a missing key resolves to the key
    // itself and inserts no row. Used by the canonicalize-at-creation encoding
    // for the flat UF-index `@UF_Sf`, where this is exactly `find_UFold(x)=x`
    // for an id with no recorded leader.
    if identity_on_miss {
        debug_assert_eq!(key.len(), 1, "identity-on-miss expects a single key column");
        return key[0];
    }
    // Not present: allocate a fresh id, insert the row, and update the index.
    let id = eg.fresh_id_internal();
    idx.insert(k, id);
    let mut full: Vec<u32> = key.iter().map(|v| v.rep()).collect();
    full.push(id);
    std::rc::Rc::make_mut(eg.mirror.entry(func).or_default()).insert(full.into_boxed_slice());
    Value::new(id)
}
