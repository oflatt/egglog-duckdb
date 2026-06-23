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

    // The native-UF `+fastrb` axis selects between TWO rebuild implementations
    // (the host `UfTable` is shared; only the REBUILD differs):
    //
    //   * PURE-ENGINE rebuild (`--native-uf`, NO `--fast-rebuild`): the on-circuit
    //     `view ⋈ @DispΔ(δdisplaced)` DBSP join. `sync_displaced_relations` feeds
    //     the `@DispΔ` relation (from the prior round's displaced ids),
    //     `rebuild_rule_dd_ir` rewrites each `@rebuild_rule` to join it (stripping
    //     the empty `@UFChange_S` onchange atom AND the impure `@canon_S` body
    //     guard prim), and the surviving pure relational join runs on the fused
    //     DBSP circuit — no host pass, no reverse index. THIS IS THE NEW DEFAULT.
    //   * CUSTOM rebuild (`--native-uf --fast-rebuild`): the off-circuit host-pass
    //     reverse-index δuf scan (`native_uf_rebuild_envs`). No `@DispΔ` rules
    //     injected; the reverse index + supporting state are maintained ONLY here.
    let native_pure = eg.native_uf_enabled && !eg.fast_rebuild;
    let native_custom = eg.native_uf_enabled && eg.fast_rebuild;

    // PURE-ENGINE rebuild only — feed the synthetic `@DispΔ` displaced-ids
    // relations BEFORE the read snapshot, so this iteration's fused DBSP circuit
    // sees the previous round's displaced ids as the δ that drives
    // `view ⋈ δdisplaced` re-canonicalization. Each `@DispΔ` relation's mirror is
    // set to EXACTLY the current `native_uf_prev_displaced[uf]` (cleared
    // otherwise); the fused `fed`-diff then yields this round's ids as +1 and last
    // round's as -1. Positive-weight bindings re-canonicalize the matching view
    // rows (the head's `@canon_S` calls run host-side, and re-inserting an
    // already-canonical row is a no-op, so an over-inclusive feed is still
    // bit-exact — the dropped `@canon_S` body guard's idempotence guarantee); the
    // join runs on the DBSP engine — keeping seminaive incrementality. The O(1)
    // find stays native. Must precede the `read` snapshot: `read` shares the
    // mirror `Rc`, so a later mutation copy-on-writes away from it. (On the custom
    // host-pass path the `@DispΔ` relations stay empty.)
    if native_pure {
        sync_displaced_relations(eg);
    }

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
    //
    // NATIVE UF (`--native-uf --feldera`): we drive PR #782's UF-backed encoding
    // through Feldera's HOST-PASS rebuild. Two classes of maintenance rule are
    // recognized by NAME (the feldera `RuleIr` carries no ruleset field, so we
    // key off the `symbol_gen.fresh(prefix)` family the encoder emits):
    //   * `@uf_change_drain_rule*` (the `@uf_change_drain` ruleset): DROPPED
    //     entirely. The host-pass owns onchange consumption; the `@UFChange_S`
    //     relation stays empty (the leader-change callback is never invoked on
    //     Feldera), so the drain matches nothing anyway.
    //   * `@rebuild_rule*` (`canonicalize`, the `@rebuilding` ruleset): taken
    //     OUT of the fused DBSP circuit (its `view ⋈ @UF_Sf` arrangement is the
    //     integral that regressed the relational fast-rebuild on DBSP — the
    //     ~24% / transaction-count win) and run as a host find-pass
    //     (`native_uf_rebuild_envs`), reading the in-core UF's O(1) `find_ro`.
    let native_uf = eg.native_uf_enabled;
    let mut atom_rules: Vec<(usize, &RuleIr)> = Vec::new();
    let mut rebuild_rules: Vec<(usize, &RuleIr)> = Vec::new();
    // PURE-ENGINE rebuild only: the DBSP-rewritten `@rebuild_rule` IRs (the
    // onchange atom + the `@canon_S` body guard prim stripped, a `@DispΔ` atom
    // appended). Owned here so each `&RuleIr` we push into `atom_rules` lives for
    // the whole call. Computed up front (cached per rule idx in the egraph).
    let mut pure_rebuild_ir: Vec<(usize, RuleIr)> = Vec::new();
    if native_pure {
        for (idx, rule) in &rules {
            if is_uf_rebuild_rule(&rule.name) {
                let dd = rebuild_rule_dd_ir(eg, *idx, rule);
                pure_rebuild_ir.push((*idx, dd));
                // The on-circuit `view ⋈ @DispΔ` join consumes the fed displaced
                // ids this iteration, so the iteration-boundary drain resets
                // `native_uf_prev_displaced`.
                eg.native_uf_rebuild_ran = true;
            }
        }
    }
    for (idx, rule) in &rules {
        if native_uf && is_uf_drain_rule(&rule.name) {
            continue;
        }
        // `@rebuild_dview_probe*` (the encoding's `δview ⋈ uf_old` probe, emitted
        // for the bridge's FULL native-UF rebuild): DROPPED entirely. Its body
        // carries the impure `@canon_S` guard prim, which the fused DBSP circuit
        // (`plan_join`) cannot host — it would PANIC. Under native-UF the
        // canon-at-creation invariant makes this δview term EMPTY (new view rows
        // are born canonical), so it is sound to drop on BOTH paths (the
        // PURE-ENGINE `view ⋈ @DispΔ` circuit join + the CUSTOM δuf host scan each
        // cover re-canonicalization without it). Matched by name BEFORE
        // `is_uf_rebuild_rule` (the probe is NOT a `@rebuild_rule*`).
        if native_uf && is_uf_dview_probe_rule(&rule.name) {
            continue;
        }
        if native_uf && is_uf_rebuild_rule(&rule.name) {
            // PURE-ENGINE: route the DBSP-rewritten `view ⋈ @DispΔ` form onto the
            // fused circuit (added to `atom_rules` below). CUSTOM: take it OUT of
            // the circuit and run the host find-pass (`native_uf_rebuild_envs`).
            if !native_pure {
                rebuild_rules.push((*idx, rule));
            }
            continue;
        }
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

    // PURE-ENGINE rebuild: add the DBSP-rewritten `view ⋈ @DispΔ` rules to the
    // fused circuit alongside the user / congruence rules. `pure_rebuild_ir` is
    // owned for the whole call, so its `&RuleIr` references are valid here. The
    // rewritten body is a pure relational join (`plan_join` accepts it), so it
    // joins on the DBSP engine; its head's `@canon_S` calls run host-side in
    // `apply_head` (the find stays native). Empty on the custom path.
    for (idx, rule) in &pure_rebuild_ir {
        atom_rules.push((*idx, rule));
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

    // CUSTOM rebuild only: make sure every `@rebuild_rule*`'s view has its
    // eq-sort columns registered and its reverse index seeded from the current
    // mirror (idempotent after the first iteration that sees the view). This
    // captures base-fact / seed rows that entered the mirror before any rebuild
    // ran; the index is then maintained incrementally at every mirror write.
    // Done before the rebuild pass so the scoped delta scan reads a complete
    // index. The reverse index exists ONLY on the custom host-pass path — the
    // pure-engine path runs the join on the circuit and keeps no index.
    if native_custom {
        for (_idx, rule) in &rebuild_rules {
            if let Ok(plan) = rebuild_rule_plan(eg, rule) {
                let eq_cols: Vec<usize> = plan.col_uf.iter().map(|&(ci, _)| ci).collect();
                eg.native_uf_seed_view_index(plan.view_func, &eq_cols);
            }
        }
    }

    // NATIVE-UF rebuild host-pass: run each `@rebuild_rule*` against the
    // start-of-call view rows, resolving each eq-sort column's leader via the
    // in-core UF's `find_ro` instead of a DBSP `view ⋈ @UF_Sf` join, then apply
    // the rule's head VERBATIM (its `@canon_S` calls re-canonicalize each column;
    // the `set` re-inserts the canonical row; the `delete` retracts the stale
    // one). Reads the native finds as of the START of this call — congruence's
    // unions, intercepted below as suppressed `set @UF_Sf` writes, are not
    // drained until after this pass, matching the relational semantics where the
    // rebuild rule reads the pre-iteration UF index.
    //
    // The carried δuf (`native_uf_prev_displaced`) is ACCUMULATED across however
    // many `run_rules` calls separate the union from this rebuild ruleset's call,
    // then DRAINED here on consumption. We snapshot it for the whole pass first
    // (so several rebuild rules sharing one UF func all see the same δuf), run
    // every rule against the snapshot, then clear each consumed UF exactly once —
    // so each displaced id drives one rebuild scan and the stash is empty for the
    // next round (a cascade re-fills it via the next `drain_all`).
    let consumed_ufs: HashSet<FunctionId> = rebuild_rules
        .iter()
        .filter_map(|(_, rule)| rebuild_rule_plan(eg, rule).ok())
        .flat_map(|plan| plan.col_uf.into_iter().map(|(_, uf)| uf))
        .collect();
    let displaced_snapshot: HashMap<FunctionId, Vec<i64>> = consumed_ufs
        .iter()
        .map(|&uf| {
            (
                uf,
                eg.native_uf_prev_displaced
                    .get(&uf)
                    .cloned()
                    .unwrap_or_default(),
            )
        })
        .collect();
    for (_idx, rule) in &rebuild_rules {
        let envs = native_uf_rebuild_envs(eg, &read, rule, &displaced_snapshot)?;
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
    // Drain the δuf we just consumed: each displaced id has now driven its rebuild
    // scan, so clear it (a later union re-accumulates a fresh set via `drain_all`).
    for uf in &consumed_ufs {
        let _ = eg.native_uf_take_displaced(*uf);
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
    // native-UF delta rebuild: rows dropped by the batched retract, per view
    // function, so we can drop them from the reverse index after the mirror edit
    // (the retain closure can't also borrow `eg`). Only collected for views.
    let mut removed_index_rows: Vec<(FunctionId, Row)> = Vec::new();
    let collect_removed = eg.native_uf_enabled;
    for (f, (keylen, keys)) in removes_by_func {
        let is_view = collect_removed && eg.native_uf_view_cols.contains_key(&f);
        if let Some(set) = eg.mirror.get_mut(&f) {
            let before_len = set.len();
            std::rc::Rc::make_mut(set).retain(|row| {
                let k: Box<[u32]> = (0..keylen)
                    .map(|i| crate::compile::row_col(row, i))
                    .collect();
                let keep = !keys.contains(&k);
                if !keep && is_view {
                    removed_index_rows.push((f, row.clone()));
                }
                keep
            });
            // A retraction that actually removed a row is a real change.
            changed |= set.len() != before_len;
        }
    }
    for (f, row) in &removed_index_rows {
        eg.native_uf_index_remove(*f, row);
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
        let inserted = std::rc::Rc::make_mut(eg.mirror.entry(f).or_default()).insert(row.clone());
        changed |= inserted;
        // CUSTOM rebuild only: index the genuinely-new row (no-op unless `f` is a
        // view — `native_uf_view_cols` is populated ONLY on the custom path, so
        // this is zero-cost on the pure-engine path). Re-inserting an existing row
        // is a no-op for the index too, so we only act on `inserted`.
        if inserted {
            eg.native_uf_index_insert(f, &row);
        }
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

    // The native-UF rebuild's `δview ⋈ uf_old` seminaive derivative is handled
    // per path with NO host-side probe in either case:
    //   * PURE-ENGINE (`--native-uf`): the rebuild runs ON the DBSP circuit
    //     (`view ⋈ @DispΔ`, the rewritten `@rebuild_rule` in `atom_rules`). New
    //     view rows are born canonical (canon-at-creation), so the `δview`
    //     derivative is empty; the circuit's seminaive join on the fed `@DispΔ` δ
    //     covers re-canonicalization without a host probe.
    //   * CUSTOM (`--native-uf --fast-rebuild`): δuf-only — the empty
    //     `δview ⋈ uf_old` term is exactly what fast-rebuild drops, so it is never
    //     probed (`native_uf_rebuild_envs` is the reverse-index δuf scan).

    // NATIVE-UF drain at the iteration boundary: apply this call's enqueued
    // unions (from intercepted `(set (@UF_Sf lhs) rhs)` head actions) to every
    // in-core UF. After this, every UF is flat, so the NEXT iteration's
    // `find_ro` reads — and the host-pass rebuild — see fresh leaders. A union
    // that actually merged two classes displaces ids; surface that as a real
    // change so the outer saturate loop keeps iterating (the relational path's
    // signal was `@UF_S` / flat-index churn, which we no longer produce).
    if native_uf {
        let displaced = eg.native_uf_drain_all();
        if displaced > 0 {
            changed = true;
        }
    }

    // `change_detect` is now folded incrementally into apply/merge (O(delta));
    // there is no separate full before/after compare to time.
    // Fold any proof-mode `Pair<sort, proof>` containers the head/on-circuit
    // prims interned in the prim engine's (deep-cloned) container store back
    // into `self.db`, the single store proof extraction reads. See
    // `EGraph::sync_prim_engine_containers`. Cheap when no container was
    // interned this iteration.
    eg.sync_prim_engine_containers();
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
    // Native-UF canon-prim interception: `@canon_S` is a find-or-self primitive
    // bound to the in-core UF (see `native_uf_canon_prim`). Answer it host-side
    // (`find_ro`) instead of through the `Database` stub.
    let canon_uf = eg.native_uf_canon_prim.get(id).copied();
    let mut out = Vec::new();
    for env in envs {
        // Resolve args; an unbound arg means this primitive can't fire for this
        // binding (shouldn't happen for well-formed rules).
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
        // δuf-driven rebuild (`FELDERA_DELTA_REBUILD`): the TRANSIENT body funcs
        // are the identity-on-miss `@UF_Sf` flat-index funcs (the canonicalize-
        // at-creation encoding's frozen UF index), collected ONLY from the
        // CANONICALIZE (`@rebuild_rule`) rules' bodies — exactly flowlog's
        // `rule_category(..) == "canonicalize"` gate. Restricting to rebuild
        // rules is load-bearing: a USER rewrite rule's instrumented body ALSO
        // reads `@UF_Sf` (canonicalize-at-creation lookups), so scanning every
        // rule would wrongly tag the user ruleset transient and split it (its
        // δview-driven matches would be dropped). The rebuild join `view ⋈
        // @UF_Sf` is driven from δuf alone (sub-step B); the δview⋈uf derivative
        // is dropped (empty by the eclass fix). Always computed; `FusedJoin`
        // ignores it unless the flag is set.
        let transient_funcs: HashSet<FunctionId> = atom_rules
            .iter()
            .filter(|(_, rule)| rule.name.contains("rebuild_rule"))
            .flat_map(|(_, rule)| rule.body.iter())
            .filter_map(|op| match op {
                BodyOp::Atom(a) if eg.info(a.func).identity_on_miss => Some(a.func),
                _ => None,
            })
            .collect();
        // RELATIONAL fast-rebuild (`--fast-rebuild` without `--native-uf`):
        // engage the δuf-driven two-substep rebuild (drop the empty `δview⋈uf_old`
        // term) when the backend config flag OR the `FELDERA_DELTA_REBUILD` env
        // var is set. (Under native-UF the rebuild rules are taken out of the
        // fused circuit entirely, so this is a no-op there.)
        let delta_rebuild = eg.fast_rebuild || dbsp_join::delta_rebuild_enabled();
        let engine = eg.prim_engine();
        let fj = dbsp_join::FusedJoin::build(&plans, &engine, &transient_funcs, delta_rebuild)?;
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

/// True for PR #782's `@uf_change_drain_rule*` drain rules (the
/// `@uf_change_drain` ruleset). Under `--native-uf` these are DROPPED: the
/// host-pass rebuild owns onchange consumption, and the `@UFChange_S` relation
/// they drain is never populated (the leader-change callback is never invoked
/// on Feldera). Matched by the `symbol_gen.fresh(prefix)` name family (the
/// feldera `RuleIr` carries no ruleset field, so we key off the rule NAME).
fn is_uf_drain_rule(name: &str) -> bool {
    name.contains("uf_change_drain")
}

/// True for PR #782's `@rebuild_rule*` canonicalization rules (the
/// `@rebuilding` ruleset). Under `--native-uf` these are taken OUT of the fused
/// DBSP circuit and run as a host find-pass (`native_uf_rebuild_envs`).
fn is_uf_rebuild_rule(name: &str) -> bool {
    name.contains("rebuild_rule")
}

/// True for the encoding's `@rebuild_dview_probe*` rule (the FULL native-UF
/// rebuild's `δview ⋈ uf_old` probe term, emitted for the bridge). Its body has
/// the impure `@canon_S` guard prim, which the fused DBSP circuit cannot host;
/// under `--native-uf` the canon-at-creation invariant makes the term empty, so
/// feldera DROPS the rule from the engine path on BOTH the pure and custom
/// rebuild paths (re-canonicalization is covered by the `view ⋈ @DispΔ` circuit
/// join / the δuf host scan respectively).
fn is_uf_dview_probe_rule(name: &str) -> bool {
    name.contains("rebuild_dview_probe")
}

/// Host-side native-UF rebuild for one PR #782 `@rebuild_rule*` (`canonicalize`)
/// rule under `--native-uf --feldera`.
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
/// Under `--native-uf` the `@UFChange_S` onchange relation is empty (the host-
/// pass owns onchange consumption; the leader-change callback never runs on
/// Feldera), so the relational join produces nothing AND, crucially, the
/// `view ⋈ @UF_Sf` arrangement that would otherwise be the integral on the
/// fused DBSP circuit is never built. We instead DRIVE THE REBUILD FROM A VIEW
/// SCAN: for every view row whose canonical form differs (some eq-sort column's
/// `find_ro` leader differs from the stored value), we emit ONE binding env that
/// binds the view's body vars. `apply_head` then runs the rule's head VERBATIM —
/// its `@canon_S` calls re-canonicalize each eq-sort column from the in-core UF,
/// the `set` re-inserts the canonical row, and the `delete` retracts the stale
/// one — reproducing the relational rebuild's retract-old / insert-canonical
/// writes bit-for-bit, but touching only changed rows (the `guard` filter,
/// applied here as the changed-row test).
///
/// Recognition (matching FlowLog's converged pattern): #782 encodes the eq-sort
/// columns via `@canon_S` PRIMITIVE calls in the head, not relational atoms. So:
///   * the VIEW function is the one the head's `set` (`HeadOp::Set`) targets;
///   * its body atom (same func) gives the var→column mapping;
///   * each head `@canon_S` call (`HeadOp::Call` whose id is a native-UF canon
///     prim) names an eq-sort body var and its UF function.
fn native_uf_rebuild_envs(
    eg: &EGraph,
    read: &HashMap<FunctionId, std::rc::Rc<HashSet<Row>>>,
    rule: &RuleIr,
    displaced_snapshot: &HashMap<FunctionId, Vec<i64>>,
) -> Result<Vec<Env>> {
    let plan = rebuild_rule_plan(eg, rule)?;
    let RebuildPlan {
        view_func,
        var_col,
        col_uf,
    } = &plan;

    let Some(set) = read.get(view_func) else {
        return Ok(Vec::new());
    };

    // NATIVE-UF DELTA REBUILD (always on under native-UF): scope the scan to
    // rows touching an id whose canonical changed in the PREVIOUS iteration —
    // the host-side `view ⋈ δuf`. The reverse index (`native_uf_rev_index`,
    // maintained at the apply block) maps an eq-sort value -> the rows holding
    // it; the carried displaced set (`native_uf_prev_displaced`) names the
    // values to look up. We still apply the per-row `find != cur` exactness
    // check, so this is bit-exact with the full scan (it only avoids visiting
    // rows that provably can't have changed: a row needs rebuild iff one of its
    // eq-sort cells holds a non-leader, and a cell only becomes a non-leader the
    // iteration its id is displaced — by then the row is in the index under that
    // id). Native-UF seeds the view's reverse index before this pass, so the
    // delta path is taken whenever the view has been registered; we fall through
    // to the full `set.iter()` scan only as a correctness fallback when the
    // view's index has not been seeded (e.g. an unrecognized view).
    if eg.native_uf_view_cols.contains_key(view_func) {
        let mut envs: Vec<Env> = Vec::new();
        // Dedup: a row may be reachable via several displaced columns / values.
        let mut seen: HashSet<&Row> = HashSet::new();
        if let Some(idx) = eg.native_uf_rev_index.get(view_func) {
            // The displaced ids of any UF this view canonicalizes against,
            // read from the pass-wide snapshot (accumulated across iterations by
            // `native_uf_drain_all`, drained on consumption after this pass).
            for &(_, uf_func) in col_uf {
                let Some(displaced) = displaced_snapshot.get(&uf_func) else {
                    continue;
                };
                for &d in displaced {
                    let d = d as u32;
                    let Some(rows) = idx.get(&d) else { continue };
                    for row in rows {
                        // The index can lag a row that was retracted out from
                        // under it within the same apply batch only if the row is
                        // still present in `read`; gate on actual membership to
                        // stay bit-exact with the full scan's `set.iter()`.
                        if !set.contains(row) || !seen.insert(row) {
                            continue;
                        }
                        if let Some(env) = rebuild_env_for_row(eg, row, var_col, col_uf) {
                            envs.push(env);
                        }
                    }
                }
            }
        }
        return Ok(envs);
    }

    let mut envs: Vec<Env> = Vec::new();
    for row in set.iter() {
        if let Some(env) = rebuild_env_for_row(eg, row, var_col, col_uf) {
            envs.push(env);
        }
    }
    Ok(envs)
}

/// The recognized shape of one `@rebuild_rule*` (`canonicalize`) rule: which
/// VIEW function it re-inserts into, the view's var→column mapping, and which
/// columns get canonicalized against which UF func. Derived once (see
/// `native_uf_rebuild_envs` / the `fast_rebuild` index maintenance).
pub(crate) struct RebuildPlan {
    pub(crate) view_func: FunctionId,
    pub(crate) var_col: HashMap<u32, usize>,
    pub(crate) col_uf: Vec<(usize, FunctionId)>,
}

/// Recognize one `@rebuild_rule*`'s view/columns (see `native_uf_rebuild_envs`).
pub(crate) fn rebuild_rule_plan(eg: &EGraph, rule: &RuleIr) -> Result<RebuildPlan> {
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
    // join's binding order; the view's columns are distinct vars in practice).
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
    Ok(RebuildPlan {
        view_func,
        var_col,
        col_uf,
    })
}

/// The per-row body of the rebuild scan, shared by the full and fast paths:
/// apply the `guard (or (bool-!= ci (@canon_S ci)))` (keep only rows where some
/// eq-sort column's leader differs from the stored value), and if it survives,
/// bind every view body var to its column value so `apply_head` can run the head
/// verbatim. Returns `None` for a row that is already fully canonical.
fn rebuild_env_for_row(
    eg: &EGraph,
    row: &Row,
    var_col: &HashMap<u32, usize>,
    col_uf: &[(usize, FunctionId)],
) -> Option<Env> {
    let mut changed = false;
    for &(ci, uf_func) in col_uf {
        let cur = crate::compile::row_col(row, ci);
        if eg.native_uf_find(uf_func, cur) != cur {
            changed = true;
            break;
        }
    }
    if !changed {
        return None;
    }
    let mut env: Env = Env::new();
    for (&v, &ci) in var_col {
        env.insert(v, crate::compile::row_col(row, ci));
    }
    Some(env)
}

/// PURE-ENGINE rebuild (`--native-uf`, no `--fast-rebuild`): feed each `@UF_Sf`
/// function's synthetic `@DispΔ` displaced-ids relation with EXACTLY the previous
/// round's displaced ids (`native_uf_prev_displaced`), so the on-circuit rebuild's
/// `view ⋈ @DispΔ` join fires the matching view rows THIS iteration. Called at the
/// top of `run_iteration`, BEFORE the read snapshot, so the snapshot (which shares
/// the mirror `Rc`) sees the fed rows.
///
/// Setting each relation's mirror to exactly the current displaced set (and
/// clearing it otherwise) makes the fused `fed`-diff present this round's ids as a
/// `+1` delta and last round's as a `-1` delta — the seminaive driver of
/// re-canonicalization on the DBSP engine. This is the native analog of the
/// relational `@UF_Sf` flat-index relation the plain-`--feldera` rebuild joins the
/// view against; the only difference is `@DispΔ` carries just the displaced ids
/// (the in-core UF supplies the leaders via the head's `@canon_S` calls), keeping
/// the O(1) find native while the JOIN runs on DBSP.
///
/// Idempotence: the rewritten `@rebuild_rule` head retracts the stale row and
/// re-inserts the canonical one (its `@canon_S` head calls run host-side); a row
/// already canonical re-inserts unchanged. So an over-inclusive displaced feed is
/// still bit-exact — exactly the prior host pass's scoped-scan guarantee, now
/// without the dropped `@canon_S` body guard.
fn sync_displaced_relations(eg: &mut EGraph) {
    if !eg.native_uf_enabled {
        return;
    }
    // Snapshot the per-UF displaced ids first (immutable borrow), then mutate the
    // mirror. `native_uf_prev_displaced` is the carried displaced set, stashed by
    // `native_uf_drain_all` (the SAME source the host pass consumed), reset once
    // per rebuild round at the drain boundary.
    let updates: Vec<(FunctionId, Vec<i64>)> = eg
        .native_uf_disp_rel
        .iter()
        .map(|(&uf, &disp_rel)| {
            let ids = eg
                .native_uf_prev_displaced
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

/// PURE-ENGINE rebuild: rewrite one PR #782 `@rebuild_rule*` (`canonicalize`) rule
/// into the on-circuit form, mirroring DuckDB/FlowLog: strip the always-empty
/// `@UFChange_S` onchange body atom AND the impure `@canon_S` body guard prim, and
/// append a `@DispΔ(eqsort_col)` atom over the synthetic displaced-ids relation,
/// leaving the view atom. The result is a PURE RELATIONAL join `view ⋈ @DispΔ` the
/// persistent DBSP engine accepts (`plan_join`) and runs seminaive on the fed δ.
///
/// Stripping the `@canon_S` body guard is the load-bearing difference from the
/// host pass: feldera evaluates body prims ON-circuit and REJECTS impure ones (the
/// `@canon_S` find-or-self is host-only, never registered pure), so leaving the
/// guard would make `plan_join` PANIC. The guard is sound to drop because the
/// head's `@canon_S` calls (run host-side in `apply_head`) re-canonicalize, and
/// re-inserting an already-canonical row is a no-op — so an over-inclusive join
/// (every view row whose eq-sort col is a displaced id, guarded or not) is still
/// bit-exact.
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
/// slot sharing one variable, so `cj` survives as a view column after the onchange
/// atom is dropped. We append `@DispΔ(cj)` joining the view on `cj` — `cj`'s UF is
/// the one the head's `@canon_S` call on `cj` canonicalizes against, so `@DispΔ` is
/// that UF func's displaced relation.
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
    // Map each eq-sort body var (a `@canon_S` head-call arg) → its UF func, so we
    // know which displaced relation each eq-sort column joins against.
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
    // view (and the view's own vars).
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
    // view eq-sort var the onchange atom ALSO bound (its `disp_` slot unified with
    // this view column `cj`). The encoder emits one `@rebuild_rule` per eq-sort
    // view column, so exactly one of the view's eq-sort columns is the onchange's
    // `disp_` — pick THAT one (NOT merely the first eq-sort column, which for a
    // multi-eq-sort view would join `@DispΔ` on the wrong column and starve the
    // rebuild → under-derivation).
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
            // Drop EVERY body prim: the `@canon_S` guard prim is impure (the DBSP
            // engine would reject it) and any dangling `(= cj disp_)`-style prim
            // referencing an onchange-only var is now unbound. The view ⋈ @DispΔ
            // join + the head's host-side `@canon_S` calls reproduce the rebuild
            // bit-exactly without it.
            BodyOp::Prim { .. } => {}
            other => new_body.push(other.clone()),
        }
    }
    let _ = onchange_only;

    // Append the synthetic `@DispΔ(eqsort_var)` atom so the rule becomes
    // `view ⋈ δdisplaced` (joined on the view's eq-sort column). If we cannot
    // identify the eq-sort var / UF / displaced relation, leave the body as the
    // pure view scan (still correct: the head's `@canon_S` calls re-canonicalize;
    // only the displaced-scoping speedup is lost).
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
                    let a = crate::compile::row_col(&row, 0) as i64;
                    let b = crate::compile::row_col(&row, 1) as i64;
                    if let Some(uf) = eg.native_ufs.get_mut(func) {
                        uf.enqueue_union(a, b);
                    }
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
                // head action (e.g. canon-at-creation `(name (@canon_S a) ...)`,
                // or the rebuild rule's `(set (@CView (@canon_S c0) ...))`) is
                // answered host-side from the in-core UF (`find_ro`).
                let result = if let Some(&uf_func) = eg.native_uf_canon_prim.get(id) {
                    Some(Value::new(eg.native_uf_find(uf_func, argv[0].rep())))
                } else {
                    // Route head prim calls through the SAME shared primitive
                    // engine the on-circuit body call-prims use, so any container
                    // (proof mode's `Pair<sort, proof>`) a head `pair` action
                    // interns is later visible to the body `pair-first` /
                    // `pair-second` reads. `eval_prim_internal` uses `self.db`
                    // directly, whose ContainerValues store is a DIFFERENT
                    // instance from the engine's clone — container ids interned in
                    // one are absent from the other (base values re-intern
                    // identically; containers do not). Sharing one engine keeps
                    // them consistent.
                    eg.prim_engine().eval(*id, &argv)
                };
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
    let row: Row = full.into_boxed_slice();
    // CUSTOM rebuild only: a hash-consed VIEW row is a new row this iteration, so
    // add it to the reverse index (no-op unless this view func's index is built —
    // `native_uf_view_cols` is populated ONLY on the custom path) so a later
    // displaced-id scan can find it. The PURE-ENGINE path keeps no reverse index;
    // its on-circuit rebuild picks up new rows born canonical (canon-at-creation),
    // so no host-side δview bookkeeping is needed.
    eg.native_uf_index_insert(func, &row);
    std::rc::Rc::make_mut(eg.mirror.entry(func).or_default()).insert(row);
    Value::new(id)
}
