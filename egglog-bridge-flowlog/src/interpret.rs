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
    // The native-UF `+fastrb` axis selects between TWO rebuild implementations
    // (the host UfTable is shared; only the REBUILD differs):
    //
    //   * PURE-ENGINE rebuild (`--native-uf`, NO `--fast-rebuild`): the DD-engine
    //     `view ⋈ @DispΔ(δdisplaced)` join. `sync_displaced_relations` feeds the
    //     `@DispΔ` relation, `rebuild_rule_dd_ir` rewrites each `@rebuild_rule` to
    //     join it, and the join runs on the fused DD worker — no host pass, no
    //     reverse index.
    //   * CUSTOM rebuild (`--native-uf --fast-rebuild`): the off-engine host-pass
    //     reverse-index scan (`native_uf_rebuild_envs`), δuf-only (the
    //     `δview ⋈ uf_old` probe dropped). No `@DispΔ` rules injected, no DD-engine
    //     rebuild; the reverse index + supporting state are maintained ONLY here.
    let native_pure = eg.native_uf_enabled && !eg.fast_rebuild;
    let native_custom = eg.native_uf_enabled && eg.fast_rebuild;

    // PURE-ENGINE rebuild only — feed the synthetic `@DispΔ` displaced-ids
    // relations BEFORE the read snapshot, so this iteration's fused DD worker
    // sees the previous round's displaced ids as the δ that drives
    // `view ⋈ δdisplaced` re-canonicalization. Each `@DispΔ` relation's mirror
    // is set to EXACTLY the current `native_uf_displaced_prev[uf]` (cleared
    // otherwise), so the fused `fed`-diff yields: this round's ids as +1, last
    // round's as -1. Positive-weight bindings re-canonicalize the matching view
    // rows (the `@canon_S` guard rejects already-canonical rows host-side, so an
    // over-inclusive feed is still bit-exact); negative weights are ignored
    // (heads are monotone-fire). The join runs on the DD engine — keeping
    // seminaive incrementality (no full view re-scan). The O(1) find stays
    // native (host-side `@canon_S`). Must precede the `read` snapshot: `read`
    // shares the mirror `Rc`, so a later mutation copy-on-writes away from it.
    // (On the custom host-pass path the `@DispΔ` relations stay empty.)
    if native_pure {
        sync_displaced_relations(eg);
    }

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

    // Under `--native-uf --flowlog` we drive PR #782's UF-backed encoding through
    // ONE of two rebuild implementations, selected by `--fast-rebuild`. The
    // maintenance rules are handled per path:
    //
    //   * `@uf_change_drain_rule*` (the `@uf_change_drain` ruleset): DROPPED
    //     entirely on BOTH paths. The `@UFChange_S` onchange relation is never
    //     populated (the leader-change callback is never invoked on FlowLog —
    //     unions route into the in-core UF instead), so its drain matches nothing.
    //   * `@rebuild_rule*` (`canonicalize`, the `@rebuilding` ruleset):
    //       - PURE-ENGINE (`!fast_rebuild`): rewritten to the DD-engine form
    //         (`rebuild_rule_dd_ir`: the empty `@UFChange_S` onchange atom replaced
    //         by a `@DispΔ(eqsort_col)` atom) so it lowers to `view ⋈ δdisplaced`
    //         and runs on the fused DD worker — seminaive on the fed displaced δ.
    //       - CUSTOM (`fast_rebuild`): intercepted host-side in `fused_bindings`
    //         (`native_uf_rebuild_envs`, the reverse-index δuf scan) so it never
    //         drives the DD dataflow — see the interception there.
    //   * `@rebuild_dview_probe*` (the FULL native-UF rebuild's `δview ⋈ uf_old`
    //     probe): run on DD under PURE-ENGINE (it is a pure view-scan body with the
    //     `@canon_S` guard, which the DD dataflow runs directly); DROPPED under
    //     CUSTOM (δuf-only — the optimisation that elides this
    //     empty-under-canon-at-creation δview term). The encoder emits this rule
    //     for EVERY backend (its `fast_rebuild` flag is bridge-only), so the
    //     +nuf / +nuf+fastrb distinction is the backend's job here.
    let drop_rule = |name: &str| -> bool {
        eg.native_uf_enabled
            && (is_uf_drain_rule(name) || (eg.fast_rebuild && is_uf_dview_probe_rule(name)))
    };
    let mut rules: Vec<(usize, RuleIr)> = rule_idxs
        .iter()
        .filter_map(|&i| eg.rules.get(i).and_then(|r| r.clone()).map(|r| (i, r)))
        .filter(|(_, r)| !drop_rule(&r.name))
        .collect();
    // PURE-ENGINE only: rewrite each `@rebuild_rule` to the DD-engine form (strip
    // the empty `@UFChange_S` onchange atom, append a `@DispΔ(eqsort_col)` atom)
    // so it lowers to `view ⋈ δdisplaced` on the fused worker. Cached per rule idx
    // (the source IR never changes). Done HERE, before `fused_bindings`, so the
    // rewritten body flows through planning, the fused-join build key, and the
    // delta machinery uniformly. On the CUSTOM path the rule is left intact and
    // intercepted host-side instead (`native_uf_rebuild_envs` in `fused_bindings`).
    if native_pure {
        for (idx, rule) in rules.iter_mut() {
            if is_uf_rebuild_rule(&rule.name) {
                *rule = rebuild_rule_dd_ir(eg, *idx, rule);
                // Mark that the `@rebuilding` ruleset ran this iteration (the DD
                // `view ⋈ δdisplaced` join consumes the fed displaced ids), so
                // the iteration-boundary drain resets `native_uf_displaced_prev`.
                eg.native_uf_rebuild_ran = true;
            }
        }
    }

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
    // CUSTOM rebuild only: maintain the per-view reverse index as rows leave /
    // enter the mirror, so `native_uf_rebuild_envs` can scope its scan to the
    // displaced ids. The index hooks no-op unless a view func's index is built,
    // so capture the removed rows for index updates only on the custom path —
    // avoids a clone on the pure-engine path (and when native-UF is off).
    let track_index = native_custom;
    for (f, (keylen, keys)) in removes_by_func {
        let mut removed: Vec<Row> = Vec::new();
        if let Some(set) = eg.mirror.get_mut(&f) {
            let before_len = set.len();
            std::rc::Rc::make_mut(set).retain(|row| {
                let k: Box<[u32]> = (0..keylen).map(|i| row_col(row, i)).collect();
                let keep = !keys.contains(&k);
                if !keep && track_index {
                    removed.push(row.clone());
                }
                keep
            });
            changed |= set.len() != before_len;
        }
        if track_index {
            for row in &removed {
                eg.index_remove_row(f, row);
            }
        }
    }
    let mut touched_keys: HashMap<FunctionId, HashSet<Vec<u32>>> = HashMap::new();
    for (f, row) in sets {
        let inputs_len = eg.info(f).arity.saturating_sub(1);
        let key: Vec<u32> = (0..inputs_len).map(|i| row_col(&row, i)).collect();
        if track_index {
            eg.index_insert_row(f, &row);
        }
        let inserted = std::rc::Rc::make_mut(eg.mirror.entry(f).or_default()).insert(row.clone());
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

    // The FULL native-UF rebuild's `δview ⋈ uf_old` seminaive probe is handled
    // per path with NO host-side probe in either case: PURE-ENGINE runs the
    // `@rebuild_dview_probe*` rule on the DD engine (a pure view-scan body with
    // the `@canon_S` guard, fired on δview by seminaive); CUSTOM (`fast_rebuild`)
    // is δuf-only and drops the probe rule at the `drop_rule` filter above.

    // Native-UF drain at the iteration boundary: apply this call's enqueued
    // unions (from intercepted `(set (@UF_Sf lhs) rhs)` head actions) to every
    // in-core UF. After this, every UF is flat, so the NEXT iteration's
    // `find_ro` reads — and the rebuild's `@canon_S` finds (DD-engine on the pure
    // path, host-pass on the custom path) — see fresh leaders. A union that
    // actually merged two classes displaces ids; surface that as a real change so
    // the outer saturate loop keeps iterating (the relational path's signal was
    // `@UF_S` / flat-index churn, not produced here).
    if eg.native_uf_enabled {
        let displaced = eg.native_uf_drain_all();
        if displaced > 0 {
            changed = true;
        }
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
        // PR #782 native-UF drain (the `@uf_change_drain` ruleset). Must be
        // matched BEFORE `uf_update` (substring overlap is none, but keep the
        // most-specific UF rules grouped). Dropped under `--native-uf`.
        ("uf_change_drain", "uf_change_drain"),
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

/// True for PR #782's `@uf_change_drain_rule*` drain rules (the
/// `@uf_change_drain` ruleset). Under `--native-uf` these are DROPPED: the
/// `@UFChange_S` onchange relation they drain is never populated (the
/// leader-change callback is never invoked on FlowLog — unions route into the
/// in-core UF instead), so the drain matches nothing anyway. Matched by the
/// `fresh()`-suffixed name's stable prefix.
pub(crate) fn is_uf_drain_rule(name: &str) -> bool {
    rule_category(name) == "uf_change_drain"
}

/// True for the encoding's `@rebuild_dview_probe*` rule (the FULL native-UF
/// rebuild's `δview ⋈ uf_old` probe term, emitted for the bridge). Its body is a
/// pure view scan with the `@canon_S` guard prim (evaluated host-side in the
/// fused worker's prim tail, like every value prim), so the DD dataflow runs it
/// directly: under `--native-uf` alone it runs on the engine (full rebuild),
/// and `--native-uf --fast-rebuild` DROPS it from the engine path (the
/// optimisation that elides this empty-under-canon-at-creation δview term). (It
/// is NOT a `@rebuild_rule*`, so `rule_category` classifies it as `<user>` —
/// match it explicitly by name.)
pub(crate) fn is_uf_dview_probe_rule(name: &str) -> bool {
    name.contains("rebuild_dview_probe")
}

/// True for PR #782's `@rebuild_rule*` canonicalization rules (the
/// `@rebuilding` ruleset). Routed per the `--fast-rebuild` axis: PURE-ENGINE
/// (`!fast_rebuild`) rewrites them to the DD-engine form (`rebuild_rule_dd_ir`:
/// the always-empty `@UFChange_S` onchange atom replaced by a
/// `@DispΔ(eqsort_col)` atom) so they lower to `view ⋈ δdisplaced` on the DD
/// dataflow; CUSTOM (`fast_rebuild`) intercepts them host-side
/// (`native_uf_rebuild_envs`, the reverse-index δuf scan) so they never drive
/// the DD dataflow.
pub(crate) fn is_uf_rebuild_rule(name: &str) -> bool {
    rule_category(name) == "canonicalize"
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
        let _ = idx;
        // CUSTOM rebuild (`--native-uf --fast-rebuild`): intercept PR #782's
        // `@rebuild_rule*` (`canonicalize`) rules host-side. `@UFChange_S` is
        // empty (the DD join would produce nothing anyway) and finds go through
        // the in-core UF, so produce this rule's binding envs from the
        // reverse-indexed view scan (`native_uf_rebuild_envs`) — DON'T route it
        // to the fused DD worker (nor re-run its prim tail — the leader env
        // already encodes the changed-row filter the guard expresses). On the
        // PURE-ENGINE path the rule was already rewritten to `view ⋈ δdisplaced`
        // (`rebuild_rule_dd_ir`) and flows through the atom-bearing path below.
        if eg.native_uf_enabled && eg.fast_rebuild && is_uf_rebuild_rule(&rule.name) {
            out[pos] = native_uf_rebuild_envs(eg, read, rule)?;
            continue;
        }
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
        // Phase-2 transient-UF rebuild (gated `EGGLOG_CANON_AT_CREATION`): a
        // transient input is a `@UF_Sf` flat index (identity-on-miss, only set
        // under the flag) read by a `@rebuild_rule`. Keeping its integral at zero
        // makes the rebuild join compute only `view_all ⋈ δuf`. Empty otherwise,
        // so user-rule joins and the flag-off path are byte-for-byte unchanged.
        let mut transient_funcs: HashSet<FunctionId> = HashSet::new();
        for &pos in &atom_positions {
            let rule = &rules[pos].1;
            if rule_category(&rule.name) != "canonicalize" {
                continue;
            }
            for op in &rule.body {
                if let BodyOp::Atom(a) = op {
                    if eg.info(a.func).identity_on_miss {
                        transient_funcs.insert(a.func);
                    }
                }
            }
        }
        // RELATIONAL fast-rebuild (`--fast-rebuild` without `--native-uf`):
        // engage the δUF-driven substep split (drop the empty `δview⋈uf_old`
        // term) when the backend config flag OR the `FLOWLOG_DELTA_REBUILD` env
        // var is set. (Under native-UF the rebuild rules run host-side, not on
        // the DD join, so this is a no-op there.)
        let delta_rebuild = eg.fast_rebuild || dd_native::delta_rebuild_enabled();
        // `--wcoj`: enable the worst-case-optimal triangle delta query. The
        // build detects the triangle shape per rule; non-triangle rules in the
        // ruleset keep the binary `.join` chain (hybrid). Off ⇒ byte-identical
        // to the pre-WCOJ build.
        let wcoj = eg.wcoj_enabled;
        let fused =
            dd_native::FusedDdJoin::build(&plans, &transient_funcs, delta_rebuild, wcoj, false)?;
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

/// Feed each `@UF_Sf` function's synthetic `@DispΔ` displaced-ids relation with
/// EXACTLY the previous round's displaced ids (`native_uf_displaced_prev`), so
/// the DD-engine rebuild's `view ⋈ δdisplaced` join fires the matching view rows
/// THIS iteration. Called at the top of `run_iteration`, BEFORE the read
/// snapshot, so the snapshot (which shares the mirror `Rc`) sees the fed rows.
///
/// Setting each relation's mirror to exactly the current displaced set (and
/// clearing it otherwise) makes the fused `fed`-diff present this round's ids as
/// a `+1` delta and last round's as a `-1` delta — the seminaive driver of
/// re-canonicalization. This is the native analog of the relational `@UF_Sf`
/// flat-index relation the plain-`--flowlog` rebuild joins the view against; the
/// only difference is `@DispΔ` carries just the displaced ids (the in-core UF
/// supplies the leaders via the `@canon_S` guard/head), keeping the O(1) find
/// native while the JOIN runs on DD.
///
/// Idempotence: the `@rebuild_rule` head retracts the stale row and re-inserts
/// the canonical one; the `@canon_S` guard rejects rows already canonical. So an
/// over-inclusive displaced feed is still bit-exact — exactly the prior host
/// pass's scoped-scan guarantee.
fn sync_displaced_relations(eg: &mut EGraph) {
    if !eg.native_uf_enabled {
        return;
    }
    // Snapshot the per-UF displaced ids first (immutable borrow), then mutate the
    // mirror. `native_uf_displaced_prev` is the previous round's displaced set,
    // stashed by `native_uf_drain_all` (the SAME source the host pass consumed).
    let updates: Vec<(FunctionId, Vec<i64>)> = eg
        .native_uf_disp_rel
        .iter()
        .map(|(&uf, &disp_rel)| {
            let ids = eg
                .native_uf_displaced_prev
                .get(&uf)
                .cloned()
                .unwrap_or_default();
            (disp_rel, ids)
        })
        .collect();
    for (disp_rel, ids) in updates {
        let set = std::rc::Rc::make_mut(eg.mirror.entry(disp_rel).or_default());
        // Reset to exactly this round's displaced ids: the fed-diff then yields
        // the right +/- delta. (A row is the single-column `[id]`.)
        set.clear();
        for id in ids {
            set.insert(vec![id as u32].into_boxed_slice());
        }
    }
}

/// Rewrite one PR #782 `@rebuild_rule*` (`canonicalize`) rule into the DD-engine
/// form, mirroring DuckDB's `rewrite_native_uf_rule`: strip the always-empty
/// `@UFChange_S` onchange body atom and replace it with a `@DispΔ(eqsort_col)`
/// atom over the synthetic displaced-ids relation, leaving the view atom + the
/// `@canon_S` guard prims. The result lowers to `view ⋈ δdisplaced` on the fused
/// DD worker (`plan_join` accepts it; the guard runs in the host prim tail).
///
/// The source rule shape is
/// ```text
/// (rule ((@UFChange_S _wl_ _wr_ _ll_ _rl_ _nl_ disp_)   ; ALWAYS empty under native-UF
///        (@View c0_ .. cn_)
///        (= cj disp_)                                    ; cj = the view eq-sort col
///        (guard (or (bool-!= ci (@canon_S ci)) ..)))
///       ((@View (@canon_S c0_) .. (@canon_S cn_) ())     ; head re-canonicalize
///        (delete (@View c0_ .. cn_))))
/// ```
/// `(= cj disp_)` is encoded by the onchange's `disp_` slot and the view's `cj`
/// slot sharing one variable, so `cj` survives as a view column after the
/// onchange atom is dropped. We append `@DispΔ(cj)` joining the view on `cj` —
/// `cj`'s UF is the one the head's `@canon_S` call on `cj` canonicalizes against,
/// so `@DispΔ` is that UF func's displaced relation.
///
/// Cached by rule index in `native_uf_rebuild_dd_ir` (the source IR is fixed).
fn rebuild_rule_dd_ir(eg: &mut EGraph, idx: usize, rule: &RuleIr) -> RuleIr {
    if let Some(cached) = eg.native_uf_rebuild_dd_ir.get(&idx) {
        return cached.clone();
    }

    // The view = the head `set` (the canonicalized re-insert) target.
    let view_func = rule.head.iter().find_map(|op| match op {
        HeadOp::Set { func, .. } => Some(*func),
        _ => None,
    });
    // Map each eq-sort body var (a `@canon_S` head-call arg) → its UF func, so
    // we know which displaced relation each eq-sort column joins against.
    let mut var_uf: HashMap<u32, FunctionId> = HashMap::new();
    for op in &rule.head {
        if let HeadOp::Call { id, args, .. } = op {
            if let Some(&uf) = eg.native_uf_canon_prim.get(id) {
                if let Some(Slot::Var(v)) = args.first() {
                    var_uf.entry(*v).or_insert(uf);
                }
            }
        }
    }

    // Vars bound by the (empty) onchange atom — any body atom that is NOT the
    // view and NOT itself a native UF / displaced relation.
    let is_view = |f: FunctionId| Some(f) == view_func;
    let mut onchange_vars: HashSet<u32> = HashSet::new();
    let mut other_vars: HashSet<u32> = HashSet::new();
    for op in &rule.body {
        if let BodyOp::Atom(a) = op {
            let target = if is_view(a.func) {
                &mut other_vars
            } else {
                &mut onchange_vars
            };
            for s in &a.slots {
                if let Slot::Var(v) = s {
                    target.insert(*v);
                }
            }
        }
    }
    // A var is onchange-only if no surviving (view) atom binds it.
    let onchange_only: HashSet<u32> = onchange_vars.difference(&other_vars).copied().collect();

    // The displaced column the dropped onchange constrained (`= cj disp_`): the
    // view eq-sort var that the onchange atom ALSO bound (its `disp_` slot was
    // unified with this view column `cj`). The encoder emits one `@rebuild_rule`
    // per eq-sort view column, so exactly one of the view's eq-sort columns is
    // the onchange's `disp_` — pick THAT one (NOT merely the first eq-sort
    // column, which for a multi-eq-sort view would join `@DispΔ` on the wrong
    // column and starve the rebuild → under-derivation).
    let eqsort_var = rule.body.iter().find_map(|op| match op {
        BodyOp::Atom(a) if is_view(a.func) => a.slots.iter().find_map(|s| match s {
            Slot::Var(v) if var_uf.contains_key(v) && onchange_vars.contains(v) => Some(*v),
            _ => None,
        }),
        _ => None,
    });

    let mut new_body: Vec<BodyOp> = Vec::new();
    for op in &rule.body {
        match op {
            // Drop the onchange (non-view) table atom.
            BodyOp::Atom(a) if !is_view(a.func) => {}
            // Drop prims that reference an onchange-only var (a dangling
            // `(= cj disp_)`-style binding); keep the `@canon_S` guard prims
            // (they reference the surviving view eq-sort columns).
            BodyOp::Prim { args, ret, .. }
                if args
                    .iter()
                    .chain(std::iter::once(ret))
                    .any(|s| matches!(s, Slot::Var(v) if onchange_only.contains(v))) => {}
            other => new_body.push(other.clone()),
        }
    }

    // Append the synthetic `@DispΔ(eqsort_var)` atom so the rule becomes
    // `view ⋈ δdisplaced` (joined on the view's eq-sort column). If we cannot
    // identify the eq-sort var / UF / displaced relation, leave the body as the
    // pure view scan (still correct: the `@canon_S` guard re-canonicalizes; only
    // the displaced-scoping speedup is lost — the dview-probe path).
    if let Some(ev) = eqsort_var {
        if let Some(uf) = var_uf.get(&ev) {
            if let Some(&disp_rel) = eg.native_uf_disp_rel.get(uf) {
                new_body.push(BodyOp::Atom(crate::compile::BodyAtom {
                    func: disp_rel,
                    slots: vec![Slot::Var(ev)],
                }));
            }
        }
    }

    let rewritten = RuleIr {
        name: rule.name.clone(),
        body: new_body,
        head: rule.head.clone(),
    };
    eg.native_uf_rebuild_dd_ir.insert(idx, rewritten.clone());
    rewritten
}

/// CUSTOM rebuild (`--native-uf --fast-rebuild`): the off-engine host-pass
/// rebuild for one PR #782 `@rebuild_rule*` (`canonicalize`) rule. This is the
/// reverse-indexed, δuf-only scoped scan — the specialized "custom rebuild" the
/// `+fastrb` axis selects (the PURE-ENGINE path runs the DD-engine
/// `view ⋈ @DispΔ` join instead; see `rebuild_rule_dd_ir`).
///
/// The relational rebuild rule is
/// ```text
/// (rule ((@UFChange_S _wl_ _wr_ _ll_ _rl_ _nl_ disp_)
///        (@CView c0_ .. cn_)
///        (= cj disp_)
///        (guard (or (bool-!= ci (@canon_S ci)) ..)))
///       ((@CView (@canon_S c0_) .. (@canon_S cn_) ())  ; canonicalized re-set
///        (delete (@CView c0_ .. cn_))))                ; retract the stale row
/// ```
/// Under `--native-uf` the `@UFChange_S` onchange relation is empty (the
/// leader-change callback never runs on FlowLog; unions route into the in-core
/// UF), so the relational join produces nothing. We instead DRIVE THE REBUILD
/// FROM A SCOPED VIEW SCAN: for every view row touching a previous-round
/// displaced id (the host-side `view ⋈ δuf`) whose canonical form differs (some
/// eq-sort column's `find_ro` leader differs from the stored value), we emit ONE
/// binding env that binds the view's body vars. `apply_head` then runs the
/// rule's head VERBATIM — its `@canon_S` calls re-canonicalize each eq-sort
/// column from the in-core UF, the `set` re-inserts the canonical row, and the
/// `delete` retracts the stale one — reproducing the relational rebuild's
/// retract-old / insert-canonical writes bit-for-bit, but touching only changed
/// rows (the `guard` filter, applied here as the changed-row test). The
/// `δview ⋈ uf_old` term is DROPPED (δuf-only — that is the `--fast-rebuild`
/// optimisation), so new view rows are not re-probed (they are born canonical).
///
/// Recognition (the only difference from the committed `flowlog-native-uf`
/// pattern, which read `@UF_Sf` body atoms): #782 encodes the eq-sort columns
/// via `@canon_S` PRIMITIVE calls in the head, not relational atoms. So:
///   * the VIEW function is the one the head's `set` (`HeadOp::Set`) targets;
///   * its body atom (same func) gives the var→column mapping;
///   * each head `@canon_S` call (`HeadOp::Call` whose id is a native-UF canon
///     prim) names an eq-sort body var and its UF function.
fn native_uf_rebuild_envs(
    eg: &mut EGraph,
    read: &HashMap<FunctionId, std::rc::Rc<HashSet<Row>>>,
    rule: &RuleIr,
) -> Result<Vec<Env>> {
    // The view is the function the head's `set` writes to. (There is exactly one
    // such `set` in a `@rebuild_rule` — the canonicalized re-insert.)
    let view_func = rule
        .head
        .iter()
        .find_map(|op| match op {
            HeadOp::Set { func, .. } => Some(*func),
            _ => None,
        })
        .ok_or_else(|| anyhow!("native-UF rebuild: rule `{}` has no view `set`", rule.name))?;

    // The view's body atom (same func) gives the var → column index mapping.
    let view_atom = rule
        .body
        .iter()
        .find_map(|op| match op {
            BodyOp::Atom(a) if a.func == view_func => Some(a),
            _ => None,
        })
        .ok_or_else(|| {
            anyhow!(
                "native-UF rebuild: rule `{}` has no view body atom for the `set` target",
                rule.name
            )
        })?;
    // var -> column index in the view row (first occurrence wins, matching the
    // DD plan's binding order; the view's columns are distinct vars in practice).
    let mut var_col: HashMap<u32, usize> = HashMap::new();
    for (i, s) in view_atom.slots.iter().enumerate() {
        if let Slot::Var(v) = s {
            var_col.entry(*v).or_insert(i);
        }
    }

    // Each head `@canon_S` call names an eq-sort body var (its single arg) and
    // the UF function to canonicalize it against. Map view COLUMN -> UF func.
    let mut col_uf: Vec<(usize, FunctionId)> = Vec::new();
    for op in &rule.head {
        if let HeadOp::Call { id, args, .. } = op {
            if let Some(&uf_func) = eg.native_uf_canon_prim.get(id) {
                let Some(Slot::Var(av)) = args.first() else {
                    return Err(anyhow!(
                        "native-UF rebuild: `@canon_S` call in rule `{}` has no var arg",
                        rule.name
                    ));
                };
                let ci = *var_col.get(av).ok_or_else(|| {
                    anyhow!(
                        "native-UF rebuild: `@canon_S` arg not a view column in rule `{}`",
                        rule.name
                    )
                })?;
                col_uf.push((ci, uf_func));
            }
        }
    }

    let Some(set) = read.get(&view_func) else {
        return Ok(Vec::new());
    };

    // Scope the scan to view rows touching a previous-round displaced id (the
    // host-side `view ⋈ δuf`), instead of scanning every view row. The index is
    // built on the FIRST rebuild for this view func (a one-time full scan that
    // seeds it — the correctness fallback for the first iteration) and maintained
    // incrementally thereafter (see `index_insert_row` / `index_remove_row`).
    // When the index is already built we look up only the displaced ids' rows;
    // the per-row `find != cur` guard below stays the exactness check, so an
    // over-inclusive candidate set is still bit-exact.

    // Cache this view func's eq-sort columns so the index hooks know which
    // columns to track (set together with the index entry below).
    eg.native_uf_view_cols.insert(view_func, col_uf.clone());

    let index_built = eg.native_uf_rev_index.contains_key(&view_func);
    if !index_built {
        // First rebuild for this func: full scan over the snapshot, building
        // the reverse index as we go (so subsequent rounds can scope).
        let mut index: HashMap<u32, HashSet<Row>> = HashMap::new();
        let mut envs: Vec<Env> = Vec::new();
        for row in set.iter() {
            for &(ci, _uf) in &col_uf {
                if let Some(&v) = row.get(ci) {
                    index.entry(v).or_default().insert(row.clone());
                }
            }
            if row_needs_rebuild(eg, row, &col_uf) {
                envs.push(bind_view_env(row, &var_col));
            }
        }
        eg.native_uf_rev_index.insert(view_func, index);
        return Ok(envs);
    }

    // Index is built: gather the candidate rows for the displaced ids of
    // every UF func this view canonicalizes against (the previous round's
    // displaced set, stashed by `native_uf_drain_all`). Clone the rows out
    // so the immutable index borrow drops before the guard pass.
    let mut uf_funcs: Vec<FunctionId> = col_uf.iter().map(|&(_, uf)| uf).collect();
    uf_funcs.sort();
    uf_funcs.dedup();
    let mut candidates: HashSet<Row> = HashSet::new();
    if let Some(index) = eg.native_uf_rev_index.get(&view_func) {
        for uf in &uf_funcs {
            if let Some(disp) = eg.native_uf_displaced_prev.get(uf) {
                for &d in disp {
                    if let Some(rows) = index.get(&(d as u32)) {
                        for row in rows {
                            candidates.insert(row.clone());
                        }
                    }
                }
            }
        }
    }
    // Mark that a rebuild scan consumed the displaced sets THIS iteration.
    // ONE UF func backs MANY view funcs (every function with a column of
    // that eq-sort), and the encoder emits one `@rebuild_rule` per view —
    // all fused into THIS `run_iteration`. So the displaced sets must NOT be
    // cleared here (that would starve the views processed after this one);
    // instead `native_uf_drain_all` clears them once, at the iteration
    // boundary, after every rebuild rule in this call has consumed them.
    eg.native_uf_rebuild_ran = true;
    let mut envs: Vec<Env> = Vec::new();
    for row in &candidates {
        if row_needs_rebuild(eg, row, &col_uf) {
            envs.push(bind_view_env(row, &var_col));
        }
    }
    Ok(envs)
}

/// The `@rebuild_rule` guard `(or (bool-!= ci (@canon_S ci)))`: true iff some
/// eq-sort column's stored value is NOT its current UF leader (so the row is
/// stale and must be re-canonicalized). The exactness check for the custom
/// host-pass rebuild (`native_uf_rebuild_envs`).
fn row_needs_rebuild(eg: &EGraph, row: &Row, col_uf: &[(usize, FunctionId)]) -> bool {
    for &(ci, uf_func) in col_uf {
        let cur = row_col(row, ci);
        if eg.native_uf_find(uf_func, cur) != cur {
            return true;
        }
    }
    false
}

/// Bind every view body var to its column value. `apply_head` then runs the
/// `@rebuild_rule` head verbatim (its `@canon_S` calls compute the leaders).
fn bind_view_env(row: &Row, var_col: &HashMap<u32, usize>) -> Env {
    let mut env: Env = Env::new();
    for (&v, &ci) in var_col {
        env.insert(v, row_col(row, ci));
    }
    env
}

/// Evaluate a primitive body op over each binding env, returning the new list of
/// envs. A value-computing prim binds (or checks) its return var; a guard prim
/// (`!=`) that fails prunes the env. Table atoms are NOT handled here — they run
/// on the DD dataflow; this is only the host-side primitive tail.
pub(crate) fn step_prim(eg: &mut EGraph, op: &BodyOp, envs: Vec<Env>) -> Result<Vec<Env>> {
    let BodyOp::Prim { id, args, ret } = op else {
        unreachable!("step_prim called on a non-primitive body op");
    };
    // Native-UF canon-prim interception: `@canon_S` is a find-or-self primitive
    // bound to the in-core UF (see `native_uf_canon_prim`). Answer it host-side
    // (`find_ro`) instead of through the `Database` stub.
    let canon_uf = eg.native_uf_canon_prim.get(id).copied();
    let mut out = Vec::new();
    for env in envs {
        let resolved: Option<Vec<Value>> = args
            .iter()
            .map(|s| slot_lookup(s, &|v| env.get(&v).copied()).map(Value::new))
            .collect();
        let Some(argv) = resolved else { continue };
        let result = if let Some(uf_func) = canon_uf {
            Some(Value::new(eg.native_uf_find(uf_func, argv[0].rep())))
        } else {
            eg.eval_prim_internal(*id, &argv)
        };
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
                // Native-UF union ingestion: PR #782 writes a union as
                // `(set (@UF_Sf lhs) rhs)` — a SET on the UF FUNCTION id, NOT on
                // a relational parent. Route it into the in-core UF
                // (`enqueue_union(lhs, rhs)`; the UF picks the min leader) and
                // SUPPRESS the mirror write (the `@UF_Sf` table is never
                // materialized — finds go through the UF). Drained at the
                // iteration boundary.
                if eg.native_ufs.contains_key(func) {
                    debug_assert!(
                        row.len() >= 2,
                        "@UF_Sf union row must have at least (lhs, rhs)"
                    );
                    let (a, b) = (row[0], row[1]);
                    if let Some(uf) = eg.native_ufs.get_mut(func) {
                        uf.enqueue_union(a as i64, b as i64);
                    }
                    let mem = eg.native_uf_members.entry(*func).or_default();
                    mem.insert(a);
                    mem.insert(b);
                    continue;
                }
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
                // Native-UF canon-prim interception (head side): `@canon_S` in a
                // head action (e.g. canon-at-creation `(name (@canon_S a) ...)`)
                // is answered host-side from the in-core UF.
                let result = if let Some(&uf_func) = eg.native_uf_canon_prim.get(id) {
                    Some(Value::new(eg.native_uf_find(uf_func, argv[0].rep())))
                } else {
                    eg.eval_prim_internal(*id, &argv)
                };
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
    let identity_on_miss = info.identity_on_miss;
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
    // Identity-on-miss ("lookup-or-self"): a missing key resolves to the key
    // itself and inserts no row. Used by the canonicalize-at-creation encoding
    // for the flat UF-index `@UF_Sf`, where this is exactly `find_UFold(x)=x`
    // for an id with no recorded leader.
    if identity_on_miss {
        debug_assert_eq!(key.len(), 1, "identity-on-miss expects a single key column");
        return key[0];
    }
    let id = eg.fresh_id_internal();
    idx.insert(k, id);
    let mut full: Vec<u32> = key.iter().map(|v| v.rep()).collect();
    full.push(id);
    let row: Row = full.into_boxed_slice();
    // CUSTOM rebuild only: a hash-consed VIEW row is a new row this iteration, so
    // add it to the reverse index (no-op unless this view func's index is built)
    // so a later displaced-id scan can find it. The PURE-ENGINE path keeps no
    // reverse index, and its DD-engine dview probe (`@rebuild_dview_probe*`)
    // picks up the new row as δview on the NEXT iteration's fused `fed`-diff —
    // so neither path needs host-side δview bookkeeping here.
    if eg.native_uf_enabled && eg.fast_rebuild {
        eg.index_insert_row(func, &row);
    }
    std::rc::Rc::make_mut(eg.mirror.entry(func).or_default()).insert(row);
    Value::new(id)
}
