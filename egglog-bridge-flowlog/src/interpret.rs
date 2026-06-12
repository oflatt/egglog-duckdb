//! Host-side rule interpreter for the FlowLog backend (Milestone 3).
//!
//! This is the FlowLog analog of the Feldera backend's Milestone-3 host
//! interpreter, and the **correctness oracle** for the engine path: it runs
//! real `.egg` programs (general multi-atom bodies, primitives, head actions)
//! to per-function tuple-count parity with the reference backend.
//!
//! One `run_rules` call = **one bounded egglog iteration**. The interpreter:
//!
//! 1. snapshots the relation mirror (the read view for this iteration — all
//!    rules match against the same pre-iteration state, egglog's semi-naive
//!    "match the old database, then apply" model for a single hop);
//! 2. for each rule, runs a **nested-loop join** over the body's table atoms,
//!    threading a variable→value binding environment, evaluating primitive body
//!    atoms (`BodyOp::Prim`) against the embedded `Database` (guards like `!=`
//!    prune; value-computing prims extend the env);
//! 3. for every surviving binding, executes the head ops in order — `set` /
//!    `remove` / `subsume` writes, RHS `lookup` (eq-sort constructor: create on
//!    miss), RHS primitive `call`, `union`, `panic`;
//! 4. applies all collected writes/removes to the mirror and resolves
//!    functional-dependency conflicts per each touched function's merge mode.
//!
//! ## The engine split (Milestone 3 mandate)
//!
//! The relational table-atom join for the **DD-eligible** rule class (table
//! atoms + `!=` guards, arity/width ≤ the engine row cap) runs on the
//! Differential-Dataflow engine in a subprocess (see [`crate::dd_join`]); the
//! primitive tail + head actions are applied HOST-side here, exactly mirroring
//! Feldera M4's "table join on the engine, prim tail host-side" split. Rules
//! outside the eligible class fall back entirely to this host nested-loop
//! interpreter (the oracle).
//!
//! Primitives are invoked through `Database::with_execution_state`, so they see
//! the same interned base `Value`s the frontend created — giving the FlowLog
//! backend bit-for-bit value parity with the reference backend (no
//! `eval_prim` trait change — flowlog keeps its zero-trait-change posture).

use anyhow::{anyhow, Result};
use egglog_backend_trait::{FunctionId, Value};
use egglog_numeric_id::NumericId;
use hashbrown::{HashMap, HashSet};

use crate::compile::{row_col, slot_lookup, BodyOp, HeadOp, Row, RuleIr, Slot};
use crate::dd_join;
use crate::EGraph;

/// Binding environment: variable id → bound `u32` value.
pub(crate) type Env = HashMap<u32, u32>;

/// A shared empty row set for atoms whose relation is absent from a snapshot —
/// `'static` so it satisfies any `step_atom` rows lifetime.
fn empty_rows() -> &'static HashSet<Row> {
    static EMPTY: std::sync::OnceLock<HashSet<Row>> = std::sync::OnceLock::new();
    EMPTY.get_or_init(HashSet::new)
}

/// Persistent, delta-maintained hash-join index store, owned by the [`EGraph`]
/// and living **across iterations**.
///
/// ## The asymptote problem this fixes
///
/// The host nested-loop join probes every non-delta body atom against the
/// `read[f]` start-of-iteration snapshot. Building a fresh `HashMap<key, rows>`
/// over that full relation every probe is `O(state)` per iteration — and the
/// relations grow monotonically during saturation, so the per-iteration join
/// cost grows with the state, producing the super-linear wall-clock blowup.
///
/// Instead we keep one hash-join index per `(FunctionId, key_cols)` that is
/// **maintained incrementally**: on every committed mirror change (head `set`
/// inserts, retraction removes, merge-resolution rewrites, and the
/// `lookup_or_create` hash-cons inserts) we patch the affected indices by the
/// `O(delta)` row diff rather than rebuilding. A probe is then an `O(1)` hash
/// lookup against an already-maintained index.
///
/// ## Correctness: the index must equal `read`, not the live mirror
///
/// The join probes the **start-of-iteration** `read` snapshot, NOT the live
/// mirror — mid-iteration `lookup_or_create` hash-cons rows are absent from
/// `read` even though they are already in the live mirror. So index maintenance
/// is **buffered**: every mirror mutation pushes its row diff into the per-func
/// `pending` log, and that log is flushed into the live indices only at the
/// **iteration boundary** (the first `index_for` of the next iteration, when the
/// `read` pointer moves), bringing each index from "equals the previous `read`"
/// to "equals the current `read`". Within an iteration the indices stay frozen
/// at the `read` contents, exactly matching what the old per-iteration cache
/// indexed.
///
/// ## Defense against unexpected mutations
///
/// Each function's index records the `Rc` pointer of the `read` snapshot it
/// currently reflects. When the `read` pointer moves, `index_for` replays the
/// buffered `pending` log and checks the resulting row count equals the new
/// `read`. If anything is off (an external mutation path — seed
/// `add_values`/`insert_rows`, `clear_table`, a mode that bypasses the buffered
/// hooks — changed the relation without a matching diff), it falls back to an
/// `O(state)` rebuild from `read[f]` for that one function. Correctness is never
/// at the mercy of complete hook coverage; the incremental path is purely an
/// optimization that the pointer check validates.
#[derive(Default)]
pub(crate) struct IndexStore {
    funcs: HashMap<FunctionId, FuncIndices>,
}

/// Per-function maintained indices plus the buffered, not-yet-applied diff.
#[derive(Default)]
struct FuncIndices {
    /// Pointer address of the `read` snapshot `Rc<HashSet<Row>>` the live
    /// `indices` currently reflect. `0` = never synced (force a rebuild).
    reflects_ptr: usize,
    /// Built indices keyed by join key columns: `key_cols -> key tuple -> rows`.
    indices: HashMap<Vec<usize>, HashMap<Vec<u32>, Vec<Row>>>,
    /// Ordered log of committed mirror mutations since the last sync, awaiting
    /// application: `(true, row)` = insert, `(false, row)` = remove. Replayed
    /// **in order** so an insert-then-remove of the same row (a `set` whose row
    /// merge-resolution later drops) nets out correctly.
    pending: Vec<(bool, Row)>,
}

impl FuncIndices {
    /// Patch every built index for a single inserted row.
    fn apply_insert(indices: &mut HashMap<Vec<usize>, HashMap<Vec<u32>, Vec<Row>>>, row: &Row) {
        for (key_cols, index) in indices.iter_mut() {
            let key: Vec<u32> = key_cols.iter().map(|&i| row[i]).collect();
            index.entry(key).or_default().push(row.clone());
        }
    }

    /// Patch every built index for a single removed row.
    fn apply_remove(indices: &mut HashMap<Vec<usize>, HashMap<Vec<u32>, Vec<Row>>>, row: &Row) {
        for (key_cols, index) in indices.iter_mut() {
            let key: Vec<u32> = key_cols.iter().map(|&i| row[i]).collect();
            if let Some(bucket) = index.get_mut(&key) {
                if let Some(pos) = bucket.iter().position(|r| r == row) {
                    bucket.swap_remove(pos);
                }
                if bucket.is_empty() {
                    index.remove(&key);
                }
            }
        }
    }

    /// Rebuild a single index over `rows` on `key_cols` (the `O(state)` path).
    fn build_index(rows: &HashSet<Row>, key_cols: &[usize]) -> HashMap<Vec<u32>, Vec<Row>> {
        let mut index: HashMap<Vec<u32>, Vec<Row>> = HashMap::new();
        for row in rows {
            let key: Vec<u32> = key_cols.iter().map(|&i| row[i]).collect();
            index.entry(key).or_default().push(row.clone());
        }
        index
    }
}

impl IndexStore {
    /// Record a committed insert of `row` into `func`'s mirror (buffered; applied
    /// in order at the next `index_for`).
    pub(crate) fn record_insert(&mut self, func: FunctionId, row: Row) {
        self.funcs.entry(func).or_default().pending.push((true, row));
    }

    /// Record a committed removal of `row` from `func`'s mirror (buffered).
    pub(crate) fn record_remove(&mut self, func: FunctionId, row: Row) {
        self.funcs
            .entry(func)
            .or_default()
            .pending
            .push((false, row));
    }

    /// Drop everything we know about `func` (e.g. `clear_table`): force a full
    /// rebuild on next use.
    pub(crate) fn forget(&mut self, func: FunctionId) {
        self.funcs.remove(&func);
    }

    /// Bring `func`'s indices into agreement with `read_set` (the current
    /// start-of-iteration snapshot, pointer `read_ptr`), then return the index
    /// for `key_cols`, building it on first request.
    ///
    /// Fast path: the buffered `pending` log is replayed in order and the result
    /// is validated to reflect `read_ptr` (its row count must equal `read_set`).
    /// If validation fails (an unexpected mutation bypassed the buffered hooks),
    /// every index is rebuilt from `read_set` (`O(state)`, rare).
    fn index_for(
        &mut self,
        func: FunctionId,
        read_ptr: usize,
        read_set: &HashSet<Row>,
        key_cols: &[usize],
    ) -> &HashMap<Vec<u32>, Vec<Row>> {
        let fi = self.funcs.entry(func).or_default();

        if fi.reflects_ptr != read_ptr {
            // The snapshot moved since we last synced. Try the incremental diff,
            // replaying the buffered mutation log IN ORDER.
            let pending = std::mem::take(&mut fi.pending);
            let consistent = fi.reflects_ptr != 0;
            if consistent {
                for (is_insert, row) in &pending {
                    if *is_insert {
                        FuncIndices::apply_insert(&mut fi.indices, row);
                    } else {
                        FuncIndices::apply_remove(&mut fi.indices, row);
                    }
                }
            }
            // Validate: the total row count across any one index must equal the
            // snapshot size; if not (or we had no prior anchor), rebuild.
            let rebuild = !consistent
                || fi
                    .indices
                    .values()
                    .next()
                    .map(|idx| idx.values().map(|v| v.len()).sum::<usize>() != read_set.len())
                    .unwrap_or(false);
            if rebuild {
                for (kc, index) in fi.indices.iter_mut() {
                    *index = FuncIndices::build_index(read_set, kc);
                }
            }
            fi.reflects_ptr = read_ptr;
        }

        if !fi.indices.contains_key(key_cols) {
            let index = FuncIndices::build_index(read_set, key_cols);
            fi.indices.insert(key_cols.to_vec(), index);
        }
        &fi.indices[key_cols]
    }
}

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
///
/// **Seminaive** (mirrors Feldera M5): each rule fires only against bindings
/// that touch at least one row NEW to that rule (`mirror[f] \ seen[r][f]`), so a
/// rule that already matched a set of facts does NOT re-fire — which is what
/// makes bounded `(saturate …)` loops converge instead of oscillating against
/// rule-driven cleanup/retraction. The relational table-atom join still runs on
/// the Differential-Dataflow engine for the DD-eligible class (now fed the
/// delta per atom-occurrence); the primitive tail + head actions are host-side.
pub fn run_iteration(eg: &mut EGraph, rule_idxs: &[usize]) -> Result<bool> {
    // Snapshot the read view: rules match against the pre-iteration mirror.
    //
    // The snapshot shares each function's row set by `Rc` rather than
    // deep-cloning every row: this is O(#functions), not O(state). Mutations to
    // the mirror this call (head writes, hash-cons in `lookup_or_create`, merge
    // resolution) go through `Rc::make_mut`, which copy-on-writes only the
    // functions actually changed while this snapshot is alive — so `read` keeps
    // the start-of-call contents and rules all match the pre-iteration state,
    // exactly as before, but without paying for a full-mirror clone every call.
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

    // Collect each rule's index + IR up front (clone to avoid borrow conflicts
    // while we also mutate the db / mirror via lookups). The index keys the
    // per-rule seminaive `seen` snapshot.
    let rules: Vec<(usize, RuleIr)> = rule_idxs
        .iter()
        .filter_map(|&i| eg.rules.get(i).and_then(|r| r.clone()).map(|r| (i, r)))
        .collect();

    // Shared start-of-iteration snapshots, built lazily once per function and
    // reused (by refcount) for every rule's `seen` advance this call.
    let mut shared_snapshot: HashMap<FunctionId, std::rc::Rc<HashSet<Row>>> = HashMap::new();
    for (idx, rule) in &rules {
        // The table relations this rule's body reads, and the seminaive delta of
        // each (rows present now that this rule has not yet matched against).
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
                let empty: HashSet<Row> = HashSet::new();
                let cur = read.get(&f).map(|v| &**v).unwrap_or(&empty);
                let seen = eg.seen.get(idx).and_then(|m| m.get(&f));
                // Fast path: if this rule's `seen[f]` snapshot is the SAME `Rc`
                // as the current read view, `f` is unchanged since the rule last
                // ran ⇒ empty delta, no O(state) scan (the proven fed-diff trick).
                if let (Some(cur_rc), Some(seen_rc)) = (read.get(&f), seen) {
                    if std::rc::Rc::ptr_eq(cur_rc, seen_rc) {
                        return (f, HashSet::new());
                    }
                }
                let d: HashSet<Row> = cur
                    .iter()
                    .filter(|r| seen.map(|s| !s.contains(*r)).unwrap_or(true))
                    .cloned()
                    .collect();
                (f, d)
            })
            .collect();

        // The seminaive binding set: union over table-atom positions of the join
        // with exactly that atom restricted to the delta. Each variant runs on
        // the DD engine (if eligible) or the host nested-loop fallback.
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

        // Advance this rule's seen snapshot to the START-OF-ITERATION read view
        // (NOT the post-write mirror): the rule has now matched everything
        // currently present. A row deleted+readded later reappears in the delta.
        // Build each touched function's snapshot once (shared by Rc across all
        // rules in this call), then bump the refcount into this rule's seen.
        for &f in &body_funcs {
            if !shared_snapshot.contains_key(&f) {
                // `read[f]` is already the start-of-call `Rc<HashSet<Row>>`;
                // reuse its handle directly as this rule's seen snapshot (no
                // rebuild). This also makes the next-iteration `Rc::ptr_eq`
                // delta short-circuit fire when `f` is untouched.
                let cur = read
                    .get(&f)
                    .map(std::rc::Rc::clone)
                    .unwrap_or_else(|| std::rc::Rc::new(HashSet::new()));
                shared_snapshot.insert(f, cur);
            }
        }
        let entry = eg.seen.entry(*idx).or_default();
        for &f in &body_funcs {
            entry.insert(f, std::rc::Rc::clone(&shared_snapshot[&f]));
        }
    }

    // Apply collected writes to the mirror.
    //
    // Removes are BATCHED per function: applying each `Write::Remove` with its
    // own `set.retain` scan is O(|removes| · |state|) — quadratic, and the
    // dominant per-iteration cost during rebuild (which retracts many rows at
    // once). We collect all retraction keys per function into a hash set, then
    // do a SINGLE `retain` pass per touched function: O(|state|) total. Removes
    // are applied FIRST (batched), then Sets — preserving the term encoder's
    // `(@uf)` "delete old leader, set new leader" delete-then-set ordering.
    //
    // `changed` is computed INCREMENTALLY as writes land (O(delta)), not via a
    // full before/after content compare of every function (O(state)). A
    // hash-cons in `lookup_or_create` always allocates a fresh id, so any term
    // created this call advances `next_id` — that alone is a real mirror change.
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
            // Collect the rows actually retracted so the persistent join index
            // can be patched by exactly this `O(delta)` diff (buffered; flushed
            // at the next iteration's `index_for`).
            let mut removed: Vec<Row> = Vec::new();
            std::rc::Rc::make_mut(set).retain(|row| {
                let k: Box<[u32]> = (0..keylen).map(|i| row_col(row, i)).collect();
                let keep = !keys.contains(&k);
                if !keep {
                    removed.push(row.clone());
                }
                keep
            });
            // A retraction that actually removed a row is a real change.
            changed |= set.len() != before_len;
            for row in removed {
                eg.index_store.record_remove(f, row);
            }
        }
    }
    // Per-function set of input keys touched by a `set` this call — the only
    // keys that can newly conflict and need merge resolution.
    let mut touched_keys: HashMap<FunctionId, HashSet<Vec<u32>>> = HashMap::new();
    for (f, row) in sets {
        let inputs_len = eg.info(f).arity.saturating_sub(1);
        let key: Vec<u32> = (0..inputs_len).map(|i| row_col(&row, i)).collect();
        // `insert` returns true iff the row was genuinely new (set a row that
        // already exists ⇒ no content change, so don't flag `changed`).
        let inserted = std::rc::Rc::make_mut(eg.mirror.entry(f).or_default()).insert(row.clone());
        changed |= inserted;
        // Only a genuinely-new row changes the relation contents; record it for
        // the persistent index (re-setting an existing row is a no-op there).
        if inserted {
            eg.index_store.record_insert(f, row);
        }
        touched_keys.entry(f).or_default().insert(key);
    }

    // Resolve FD conflicts on every function a head action wrote to (a `set`
    // can introduce two rows sharing a key that must merge per the merge mode).
    // Resolution is INCREMENTAL — only the keys whose rows were touched this
    // call can newly conflict, and `resolve_merge` reports whether it actually
    // collapsed/changed any row (another real change source for the loop).
    let empty_keys: HashSet<Vec<u32>> = HashSet::new();
    for &f in &touched {
        let keys = touched_keys.get(&f).unwrap_or(&empty_keys);
        changed |= eg.resolve_merge(f, keys);
    }

    // `change_detect` is now folded incrementally into apply/merge (O(delta));
    // there is no separate full before/after compare.
    Ok(changed)
}

/// Extend each binding env by matching `op` against the read view (table atom)
/// or evaluating it (primitive). Returns the new list of envs.
pub(crate) fn step_body(
    eg: &mut EGraph,
    read: &HashMap<FunctionId, std::rc::Rc<HashSet<Row>>>,
    op: &BodyOp,
    envs: Vec<Env>,
) -> Result<Vec<Env>> {
    match op {
        BodyOp::Atom(atom) => match read.get(&atom.func) {
            Some(rc) => {
                let ptr = std::rc::Rc::as_ptr(rc) as usize;
                Ok(step_atom(
                    &atom.slots,
                    rc,
                    envs,
                    Some((&mut eg.index_store, atom.func, ptr)),
                ))
            }
            None => Ok(step_atom(&atom.slots, empty_rows(), envs, None)),
        },
        BodyOp::Prim { id, args, ret } => {
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
    }
}

/// Match a single table atom (`slots`) against `rows` under each incoming env,
/// returning the extended envs. Shared by the full-scan body step
/// (`step_body`) and the seminaive delta match (`seminaive_bindings`): when
/// `rows` is a relation's delta slice, this performs the delta-restricted join
/// of that atom occurrence.
///
/// `index_src = Some((store, func, read_ptr))` routes the probe through the
/// **persistent, delta-maintained** [`IndexStore`]: the hash-join index over
/// `rows` (which MUST be the `read[func]` snapshot owned by the `Rc` whose
/// pointer is `read_ptr`) is maintained incrementally across iterations and
/// reused for every probe on the same `(func, key_cols)`. Pass `Some` ONLY for
/// the `read[f]` full-relation snapshot. Pass `None` for the transient per-call
/// delta slices (no stable identity; a one-shot index is built instead).
pub(crate) fn step_atom(
    slots: &[Slot],
    rows: &HashSet<Row>,
    envs: Vec<Env>,
    index_src: Option<(&mut IndexStore, FunctionId, usize)>,
) -> Vec<Env> {
    // JOIN KEY: columns whose slot is a variable already bound in the incoming
    // envs (the same var must agree between env and row). If non-empty,
    // hash-join on it instead of a full cartesian scan.
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
        for env in &envs {
            for row in rows {
                if let Some(next) = match_atom(slots, row, env) {
                    out.push(next);
                }
            }
        }
    } else if let Some((store, func, read_ptr)) = index_src {
        // Persistent delta-maintained full-relation hash-join: the index over
        // `read[func]` on `key_cols` is kept incrementally across iterations and
        // reused for every probe this iteration.
        let index = store.index_for(func, read_ptr, rows, &key_cols);
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
    } else {
        // Uncached one-shot hash-join (transient delta slice).
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
/// the full relation. This is exactly the set of bindings touching at least one
/// newly-derived fact — egglog's seminaive semantics.
///
/// For the DD-eligible class the relational join runs on the Differential-
/// Dataflow engine (`dd_join`); otherwise the host nested-loop fallback (the
/// oracle) runs it. Body primitives (`!=` guards, value-computing prims) are
/// applied host-side over the produced bindings. Bindings are deduplicated
/// across the `j`-variants (a binding new in two positions appears twice).
fn seminaive_bindings(
    eg: &mut EGraph,
    read: &HashMap<FunctionId, std::rc::Rc<HashSet<Row>>>,
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
        // Prim-only / constant body: no table atoms to delta over. Evaluate it
        // once (bounded + idempotent — rare, not on the saturation hot path).
        eg.host_rule_runs += 1;
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
    // binding this iteration — skip it entirely (the seminaive win that stops
    // the oscillation).
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

    // Plan the DD-eligible whole-body join once; reused per delta-atom variant.
    let plan = if eg.dd_enabled {
        dd_join::plan_join(eg, rule)
    } else {
        None
    };

    let mut seen_bindings: HashSet<Vec<(u32, u32)>> = HashSet::new();
    let mut out: Vec<Env> = Vec::new();

    for (atom_ord, &delta_pos) in atom_positions.iter().enumerate() {
        // Skip a variant whose delta atom has no new rows (it would just rescan
        // the full relation pointlessly).
        let delta_func = match &rule.body[delta_pos] {
            BodyOp::Atom(a) => a.func,
            _ => unreachable!(),
        };
        if delta.get(&delta_func).map(|d| d.is_empty()).unwrap_or(true) {
            continue;
        }

        let variant_envs = if let Some(plan) = &plan {
            // DD path: run the relational join on the engine with this atom
            // occurrence fed the delta rows and the others the full relation.
            eg.dd_rule_runs += 1;
            let bindings = dd_join::run_join_seminaive(eg, plan, atom_ord, read, delta)?;
            let var_order = plan.var_order().to_vec();
            let mut envs: Vec<Env> = Vec::new();
            for bind in bindings {
                let mut env: Env = Env::new();
                for (i, &v) in var_order.iter().enumerate() {
                    env.insert(v, bind[i]);
                }
                // Re-run body primitives host-side (value prims bind fresh vars;
                // `!=` re-filters identically). Table atoms were handled on DD.
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
            // Host fallback: nested-loop body scan. The atom evaluation order is
            // chosen to FLATTEN the join's asymptote: the **delta atom is matched
            // FIRST** (it ranges over the small seminaive delta), so every
            // remaining atom is then probed with join keys already bound by the
            // delta row — an `O(1)` hash lookup against the persistent index
            // instead of an `O(state)` full cartesian scan of a leading atom.
            //
            // Prims are applied **greedily** as soon as all their input vars are
            // bound (after each atom), preserving body-order semantics: the
            // binding set of a conjunctive body is order-independent, and a prim
            // is a pure function/filter of its (now-bound) inputs. A prim is
            // never run before its inputs exist, so it can't spuriously prune.
            eg.host_rule_runs += 1;

            // Atom evaluation order: delta atom first, then the rest in body
            // order.
            let mut atom_order: Vec<usize> = vec![delta_pos];
            atom_order.extend(atom_positions.iter().copied().filter(|&p| p != delta_pos));

            // Prim positions, tracked so each fires exactly once when ready.
            let prim_positions: Vec<usize> = rule
                .body
                .iter()
                .enumerate()
                .filter_map(|(i, op)| matches!(op, BodyOp::Prim { .. }).then_some(i))
                .collect();
            let mut prim_done = vec![false; prim_positions.len()];

            // Greedily apply every prim whose inputs are all currently bound.
            // Repeats until no progress (a value-prim may unlock another).
            let apply_ready_prims = |eg: &mut EGraph,
                                     envs: &mut Vec<Env>,
                                     prim_done: &mut [bool]|
             -> Result<()> {
                loop {
                    let mut progressed = false;
                    for (k, &pp) in prim_positions.iter().enumerate() {
                        if prim_done[k] || envs.is_empty() {
                            continue;
                        }
                        let BodyOp::Prim { args, .. } = &rule.body[pp] else {
                            continue;
                        };
                        // Ready iff every variable argument is bound in (the
                        // representative of) the current envs.
                        let ready = args.iter().all(|s| match s {
                            Slot::Var(v) => envs.first().map(|e| e.contains_key(v)).unwrap_or(true),
                            Slot::Const(_) => true,
                        });
                        if !ready {
                            continue;
                        }
                        *envs = step_body(eg, read, &rule.body[pp], std::mem::take(envs))?;
                        prim_done[k] = true;
                        progressed = true;
                    }
                    if !progressed {
                        break;
                    }
                }
                Ok(())
            };

            let mut envs: Vec<Env> = vec![Env::new()];
            // Prims with no var inputs (constant guards) can fire immediately.
            apply_ready_prims(eg, &mut envs, &mut prim_done)?;
            for &pos in &atom_order {
                if envs.is_empty() {
                    break;
                }
                let BodyOp::Atom(atom) = &rule.body[pos] else {
                    unreachable!()
                };
                if pos == delta_pos {
                    let rows = delta.get(&atom.func).unwrap_or_else(|| empty_rows());
                    envs = step_atom(&atom.slots, rows, envs, None);
                } else {
                    match read.get(&atom.func) {
                        Some(rc) => {
                            let ptr = std::rc::Rc::as_ptr(rc) as usize;
                            envs = step_atom(
                                &atom.slots,
                                rc,
                                envs,
                                Some((&mut eg.index_store, atom.func, ptr)),
                            );
                        }
                        None => {
                            envs = step_atom(&atom.slots, empty_rows(), envs, None);
                        }
                    }
                }
                // Fire any prim whose inputs the just-matched atom completed.
                apply_ready_prims(eg, &mut envs, &mut prim_done)?;
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
        let col = row_col(row, i);
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
    std::rc::Rc::make_mut(eg.mirror.entry(func).or_default()).insert(row.clone());
    // Buffer the hash-cons insert for the persistent join index. It is NOT
    // applied this iteration (the row is absent from `read`, the start-of-
    // iteration snapshot the join probes); it lands at the next iteration's
    // `index_for`, when `read` advances to include it.
    eg.index_store.record_insert(func, row);
    Value::new(id)
}
