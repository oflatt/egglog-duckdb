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

    // Compute every rule's binding envs FIRST (so the whole atom-bearing ruleset
    // runs on ONE fused DD worker in a single epoch — `fused_bindings`), THEN
    // apply head actions in the original rule firing order. Atom-less rules
    // (`(rule () …)`) have no input relation to drive the DD dataflow, so they
    // stay host-side (fire once); they are computed inline below.
    let envs_by_rule = fused_bindings(eg, &read, &rules)?;

    for ((idx, rule), envs) in rules.iter().zip(envs_by_rule.into_iter()) {
        let _ = idx;
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

/// Derive a stable ruleset LABEL for a `run_rules` call from the rules it runs,
/// for `FLOWLOG_DD_RULESET_PROF`. The backend trait does not carry the ruleset
/// name down to `run_rules`, but the term encoder gives each maintenance rule a
/// `fresh()`-suffixed name with a stable PREFIX that identifies its ruleset
/// (`uf_update*` / `singleparent*` / `uf_function_index*` / `congruence_rule*` /
/// `rebuild_rule*` / `merge_rule*` / `merge_cleanup*` / `delete_rule*`); user
/// rewrite rules are named by their full s-expression text. We map each rule
/// name to its category, then label the call by the categories present (all
/// rules in one `run_rules` call belong to one egglog ruleset).
/// Map a single rule NAME to its bucket label. Maintenance rules emitted by the
/// term encoder (proof_encoding.rs) carry a stable, `fresh()`-suffixed name;
/// most are `@`-prefixed (`@uf_update`, `@congruence_rule`, …) but
/// `singleparent@uf_update` is not. Match the identifying substring so the
/// leading `@` and trailing digits don't matter. Order: most specific first
/// (`singleparent` and the `_subsume` variant before their bare forms;
/// `uf_function_index` before `uf_update`).
pub(crate) fn rule_category(name: &str) -> &'static str {
    const MAINT: &[(&str, &str)] = &[
        ("singleparent", "single_parent"),
        ("uf_function_index", "uf_function_index"),
        ("uf_update", "path_compress/uf_update"),
        ("delete_rule_subsume", "delete_subsume"),
        ("delete_rule", "delete_subsume"),
        // `@congruence_rule` and `@rebuild_rule` are both in the term encoder's
        // `rebuilding` ruleset and run in ONE fused `run_rules` call; split them
        // into distinct buckets so the native-UF-addressable cost
        // (canonicalization, which joins function rows against `@uf`) can be
        // separated from congruence detection (which stays relational under a
        // native UF).
        ("congruence_rule", "congruence"),
        ("rebuild_rule", "canonicalize"),
        ("merge_cleanup", "rebuilding_cleanup"),
        ("merge_rule", "merge_rule"),
    ];
    for (needle, label) in MAINT {
        if name.contains(needle) {
            return label;
        }
    }
    if name.starts_with("eval_actions") {
        return "eval_actions";
    }
    "<user>"
}

/// Compute every rule's binding envs in ONE fused pass: the whole atom-bearing
/// ruleset's body joins run on a SINGLE shared timely worker
/// ([`dd_native::FusedDdJoin`]) clocked once this iteration, then each rule's
/// host-side prim tail is re-run over its own bindings. Atom-less rules
/// (`(rule () …)`) have no input relation to drive the DD dataflow, so they are
/// fired once host-side. Returns a `Vec<Vec<Env>>` parallel to `rules` (same
/// order), ready for `apply_head`.
fn fused_bindings(
    eg: &mut EGraph,
    read: &HashMap<FunctionId, std::rc::Rc<HashSet<Row>>>,
    rules: &[(usize, RuleIr)],
) -> Result<Vec<Vec<Env>>> {
    use crate::dd_native;

    let prof = dd_native::prof_enabled();
    let rs_prof = dd_native::ruleset_prof_enabled();
    // Per-ruleset attribution: the wall clock for the WHOLE atom-bearing path
    // this call, plus the per-bucket nanos we accumulate below. The call's time
    // is later apportioned across the rule CATEGORIES present (so `@rebuild_rule`
    // and `@congruence_rule`, fused into one timely step, land in distinct
    // `canonicalize`/`congruence` buckets).
    let rs_t_total = std::time::Instant::now();
    let mut rs_delta_ns: u64 = 0;
    let mut rs_prim_ns: u64 = 0;
    let mut rs_feed_ns: u64 = 0;
    let mut rs_step_ns: u64 = 0;
    let mut rs_delta_rows: u64 = 0;
    let mut out: Vec<Vec<Env>> = vec![Vec::new(); rules.len()];

    // Partition: atom-bearing rules drive the fused DD worker; atom-less rules
    // fire once host-side. Record each atom-bearing rule's POSITION in `rules` so
    // we can scatter the fused output back into `out` in the caller's order.
    let mut atom_positions: Vec<usize> = Vec::new();
    let mut atom_rule_idxs: Vec<usize> = Vec::new();
    for (pos, (idx, rule)) in rules.iter().enumerate() {
        let has_atoms = rule.body.iter().any(|op| matches!(op, BodyOp::Atom(_)));
        if has_atoms {
            atom_positions.push(pos);
            atom_rule_idxs.push(*idx);
        } else {
            // Atom-less rule: fire once (presence in `seen` = already fired).
            if eg.seen.contains_key(idx) {
                continue;
            }
            eg.seen.insert(*idx, ());
            let mut envs: Vec<Env> = vec![Env::new()];
            for op in &rule.body {
                envs = step_prim(eg, op, envs)?;
                if envs.is_empty() {
                    break;
                }
            }
            out[pos] = envs;
        }
    }

    if atom_positions.is_empty() {
        return Ok(out);
    }

    // The fused join is keyed by the SORTED atom-bearing rule-index list (the
    // ruleset identity), exactly like feldera's `FusedJoin`. Build it ONCE
    // (lazily) per distinct ruleset, planning each rule. Any shape `plan_join`
    // rejects PANICS (no host fallback; the DD dataflow is the only join path).
    let mut key: Vec<usize> = atom_rule_idxs.clone();
    key.sort_unstable();

    if !eg.dd_fused.contains_key(&key) {
        // Plan in the SAME order as `atom_positions` so the fused build order
        // matches our scatter order (the fused join preserves plan order).
        let mut plans: Vec<(usize, dd_native::JoinPlan)> = Vec::with_capacity(atom_positions.len());
        for (&pos, &idx) in atom_positions.iter().zip(atom_rule_idxs.iter()) {
            let rule = &rules[pos].1;
            let plan = match dd_native::plan_join(rule) {
                Ok(p) => p,
                Err(reason) => panic!(
                    "FlowLog DD join cannot lower rule {:?}: {reason} \
                     (no host fallback; the DD dataflow is the only join path)",
                    rule.name
                ),
            };
            plans.push((idx, plan));
        }
        let fused = dd_native::FusedDdJoin::build(&plans)?;
        eg.dd_fused.insert(key.clone(), fused);
    }

    // The fused join's internal rule order (its build order = `atom_positions`
    // order). Map each fused output slot back to the caller `rules` position and
    // capture each rule's canonical var order.
    let (fused_rule_idxs, fused_body_funcs): (Vec<usize>, Vec<Vec<FunctionId>>) = {
        let fused = eg.dd_fused.get(&key).expect("fused join present");
        (
            fused.rule_indices(),
            (0..fused.rule_indices().len())
                .map(|p| fused.rule_body_funcs(p).to_vec())
                .collect(),
        )
    };
    // The fused build order equals `atom_rule_idxs` (we built it that way), so
    // map fused position -> caller `rules` position via `atom_positions`.
    debug_assert_eq!(fused_rule_idxs, atom_rule_idxs, "fused build order");

    // Each atom-bearing rule's canonical var order (for env reconstruction).
    let var_orders: Vec<Vec<u32>> = atom_positions
        .iter()
        .map(|&pos| {
            let rule = &rules[pos].1;
            dd_native::plan_join(rule)
                .expect("plan re-derivable")
                .var_order()
                .to_vec()
        })
        .collect();

    // Distinct relations across the whole ruleset → ONE combined signed delta map
    // fed into the fused worker's SHARED inputs. The `fed` snapshot is per-ruleset
    // (the fused join's identity), diffed against the live mirror like the per-rule
    // `dd_native_fed`.
    let t_delta = std::time::Instant::now();
    let empty_set: std::rc::Rc<HashSet<Row>> = std::rc::Rc::new(HashSet::new());
    let mut all_funcs: Vec<FunctionId> = Vec::new();
    for bf in &fused_body_funcs {
        for &f in bf {
            if !all_funcs.contains(&f) {
                all_funcs.push(f);
            }
        }
    }
    let mut delta: HashMap<FunctionId, Vec<(Vec<u32>, isize)>> = HashMap::new();
    {
        let fed = eg.dd_fused_fed.entry(key.clone()).or_default();
        for &f in &all_funcs {
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
                rs_delta_rows += rows.len() as u64;
                delta.insert(f, rows);
            }
            *prev = cur;
        }
    }
    let delta_elapsed = t_delta.elapsed().as_nanos() as u64;
    if prof {
        dd_native::PROF_DELTA_NS.fetch_add(delta_elapsed, std::sync::atomic::Ordering::Relaxed);
    }
    rs_delta_ns += delta_elapsed;

    // ONE step of the shared worker for the WHOLE ruleset. `step` updates the
    // global PROF_FEED_NS / PROF_STEP_NS counters when profiling is on (which it
    // is whenever either env var is set), so snapshot them before/after to split
    // this ruleset's feed vs worker_step time.
    use std::sync::atomic::Ordering as ProfOrd;
    let feed_before = dd_native::PROF_FEED_NS.load(ProfOrd::Relaxed);
    let step_before = dd_native::PROF_STEP_NS.load(ProfOrd::Relaxed);
    eg.dd_rule_runs += atom_positions.len() as u64;
    let per_rule_bindings = {
        let fused = eg.dd_fused.get_mut(&key).expect("fused join present");
        fused.step(&delta)?
    };
    if rs_prof {
        rs_feed_ns += dd_native::PROF_FEED_NS
            .load(ProfOrd::Relaxed)
            .wrapping_sub(feed_before);
        rs_step_ns += dd_native::PROF_STEP_NS
            .load(ProfOrd::Relaxed)
            .wrapping_sub(step_before);
    }

    // Per-rule positive-binding count = the fused join's workload proxy for each
    // rule (used below to apportion the single fused worker_step across the rule
    // CATEGORIES present in this call). Length is parallel to `atom_positions`.
    let mut rs_pos_bindings: Vec<u64> = vec![0; atom_positions.len()];

    // Turn each rule's positive binding deltas into envs; re-run its body prims
    // host-side. Negative weights are integral bookkeeping (a body row retracted)
    // — egglog heads are monotone-fire, so we do NOT re-fire on disappearance.
    let t_prim = std::time::Instant::now();
    for (fpos, bindings) in per_rule_bindings.into_iter().enumerate() {
        let caller_pos = atom_positions[fpos];
        let rule = &rules[caller_pos].1;
        let var_order = &var_orders[fpos];
        let mut envs: Vec<Env> = Vec::new();
        for (bind, w) in &bindings {
            if *w <= 0 {
                continue;
            }
            rs_pos_bindings[fpos] += 1;
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
            envs.extend(es);
        }
        out[caller_pos] = envs;
    }
    let prim_elapsed = t_prim.elapsed().as_nanos() as u64;
    if prof {
        dd_native::PROF_PRIM_NS.fetch_add(prim_elapsed, std::sync::atomic::Ordering::Relaxed);
    }
    rs_prim_ns += prim_elapsed;

    if rs_prof {
        let rs_total_ns = rs_t_total.elapsed().as_nanos() as u64;
        // Several rule CATEGORIES (e.g. `canonicalize` = `@rebuild_rule` and
        // `congruence` = `@congruence_rule`) run in ONE fused timely step, so
        // there is no per-category wall clock. Apportion this call's measured
        // buckets across the categories present, weighted by each category's
        // share of POSITIVE output bindings (the join workload it produced). If
        // no rule produced output this call, fall back to an even split by rule
        // count so the call's wall time is still attributed.
        let mut cat_w: HashMap<&'static str, u64> = HashMap::new();
        let mut cat_n: HashMap<&'static str, u64> = HashMap::new();
        // Cross-check: nanos of (apportioned) worker_step attributable to rules
        // whose BODY reads a `@uf` table (`UF_*` relation) — the native-UF-
        // addressable fraction proxy.
        let mut uf_body_w: u64 = 0;
        for (fpos, &pos) in atom_positions.iter().enumerate() {
            let rule = &rules[pos].1;
            let cat = rule_category(&rule.name);
            let w = rs_pos_bindings[fpos];
            *cat_w.entry(cat).or_default() += w;
            *cat_n.entry(cat).or_default() += 1;
            let reads_uf = rule.body.iter().any(
                |op| matches!(op, BodyOp::Atom(a) if eg.relation_name(a.func).contains("UF_")),
            );
            if reads_uf {
                uf_body_w += w;
            }
        }
        let total_w: u64 = cat_w.values().sum();
        let total_n: u64 = cat_n.values().sum();
        // Worker_step nanos apportioned to UF-body-reading rules.
        let uf_step_ns = if total_w > 0 {
            (rs_step_ns as u128 * uf_body_w as u128 / total_w as u128) as u64
        } else {
            0
        };
        dd_native::ruleset_uf_body_record(uf_step_ns, rs_step_ns);
        for (cat, &w) in &cat_w {
            let n = cat_n[cat];
            // share by binding workload, else by rule count.
            let (num, den): (u128, u128) = if total_w > 0 {
                (w as u128, total_w as u128)
            } else {
                (n as u128, total_n.max(1) as u128)
            };
            let part = |v: u64| (v as u128 * num / den) as u64;
            dd_native::ruleset_prof_record(
                cat,
                part(rs_total_ns),
                part(rs_step_ns),
                part(rs_feed_ns),
                part(rs_prim_ns),
                part(rs_delta_ns),
                part(rs_delta_rows),
            );
        }
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
