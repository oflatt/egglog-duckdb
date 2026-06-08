//! Stage-3 integration: route the term encoder's rebuild rulesets onto the
//! persistent congruence-shuttle circuit ([`crate::rebuild_circuit`]) instead of
//! the host interpreter, behind the `FELDERA_CIRCUIT_REBUILD` flag.
//!
//! ## What this replaces
//!
//! The term encoder (`src/proofs/proof_encoding.rs`) emits, per eq-sort:
//!
//! * `@UF_<sort>(child, parent, _)` — union (parent) edges, a relation.
//! * `@UF_<sort>f(child) = leader`  — the uf-index function, `:merge ordering-min`.
//! * `@<ctor>View(args…, out)`      — one view table per constructor.
//!
//! and rebuild rulesets (`parent` / `single_parent` / `uf_function_index` /
//! `rebuilding` / `rebuilding_cleanup`) that the frontend saturates. The host
//! interpreter re-reads the full mirror on every `run_rules` call, which is
//! O(state) per call and blows up super-linearly (the migration's target bug).
//!
//! ## What this does instead
//!
//! When `run_rules` is invoked with a set of rules that are *all* recognized
//! rebuild rules, we run the **whole** rebuild fixpoint in one shot on the
//! persistent circuit:
//!
//! 1. Feed the circuit the current `@UF` edges (as unions) and every view-table
//!    row (as view rows, args packed).
//! 2. Run the §2.4 congruence shuttle to fixpoint.
//! 3. Read leaders + canonical views, and SYNTHESIZE the canonical mirror:
//!    `@UFf = {(x, leader(x)) : leader(x) defined}` and
//!    `@UF  = {(x, leader(x)) : leader(x) != x}` (child→leader edges, exactly
//!    what the interpreter's path-compressed `@UF` converges to), plus each view
//!    table rewritten to canonical outputs.
//!
//! Because the result is the *converged* rebuild, the FIRST such call applies
//! the full fixpoint and a SECOND call is a no-op (`changed=false`), so the
//! frontend's saturation loop terminates after one effective pass — replacing
//! the interpreter's many O(state) passes with one circuit fixpoint.
//!
//! ## Safety / fallback
//!
//! Recognition is conservative: if the rule set contains anything we don't
//! recognize as a pure rebuild rule, or the relation roles can't be resolved,
//! we return `None` and the caller falls back to the interpreter. So this can
//! never *regress* a program — at worst it declines to accelerate it.

use egglog_backend_trait::FunctionId;
use egglog_numeric_id::NumericId;
use hashbrown::{HashMap, HashSet};

use crate::compile::{row_col, BodyOp, HeadOp, RuleIr};
use crate::rebuild_circuit::{RebuildCircuit, NO_ARG};
use crate::EGraph;

/// The relation roles recognized for one eq-sort's rebuild.
#[derive(Default)]
pub struct RebuildRoles {
    /// The `@UF` parent relation(s) (child, parent, _).
    pub uf: HashSet<FunctionId>,
    /// The `@UFf` uf-index function(s) (child) -> leader.
    pub uff: HashSet<FunctionId>,
    /// The `@…View` view tables (args…, out).
    pub views: HashSet<FunctionId>,
}

/// Classify a single rebuild rule, contributing to `roles`. Returns `false` if
/// the rule is not a recognized rebuild rule.
///
/// Recognized term-mode rebuild rules (no proofs):
///   * uf_index:      body `(@uf a b _)`              head `set @uff(a)=b`
///   * path_compress: body `(@uf a b _)(@uf b c _)(!=)` head `delete/set @uf`
///   * single_parent: body `(@uf a b _)(@uff a c)(!=)` head `delete/set @uf,@uff`
///   * congruence:    body two `(@view args o1)(@view args o2)(!=)` head set @uf
///   * cleanup:       body `(@view args old)(@view args new)` head delete/set view
fn classify_rule(rule: &RuleIr, roles: &mut RebuildRoles) -> bool {
    // Collect body table atoms and head set/remove targets.
    let body_atoms: Vec<&crate::compile::BodyAtom> = rule
        .body
        .iter()
        .filter_map(|op| match op {
            BodyOp::Atom(a) => Some(a),
            BodyOp::Prim { .. } => None,
        })
        .collect();
    if body_atoms.is_empty() {
        return false;
    }

    let name = rule.name.as_str();
    // The encoder names rebuild rules predictably (see proof_encoding.rs).
    let is_uf_index = name.contains("uf_function_index");
    let is_path = !is_uf_index && (name.contains("uf_update") || name.starts_with("uf_update"));
    let is_single = name.contains("singleparent");
    let is_congru = name.contains("congruence_rule");
    let is_cleanup = name.contains("merge_cleanup") || name.contains("merge_rule");

    if is_uf_index {
        // body (@uf a b _) ; head set (@uff a) b
        roles.uf.insert(body_atoms[0].func);
        for op in &rule.head {
            if let HeadOp::Set { func, .. } = op {
                roles.uff.insert(*func);
            }
        }
        return true;
    }
    if is_path {
        // body (@uf a b _)(@uf b c _)
        roles.uf.insert(body_atoms[0].func);
        return true;
    }
    if is_single {
        // body (@uf a b _)(@uff a c)
        roles.uf.insert(body_atoms[0].func);
        if body_atoms.len() >= 2 {
            roles.uff.insert(body_atoms[1].func);
        }
        return true;
    }
    if is_congru {
        // body (@view args o1)(@view args o2) ; head set @uf
        roles.views.insert(body_atoms[0].func);
        for op in &rule.head {
            if let HeadOp::Set { func, .. } = op {
                roles.uf.insert(*func);
            }
        }
        return true;
    }
    if is_cleanup {
        roles.views.insert(body_atoms[0].func);
        return true;
    }
    false
}

/// Attempt to recognize `rule_idxs` as a set of rebuild rules. Returns `None`
/// (interpreter fallback) if any rule in this call is not a recognized rebuild
/// rule.
///
/// On success, returns the FULL set of rebuild roles learned across **every**
/// registered rebuild rule (not just this call's) — because rebuild relations
/// (`@UF` / `@UFf` / views) are persistent mirror state that any rebuild-ruleset
/// invocation must read in full, even when the frontend schedules only a subset
/// of the rebuild rules in a given `run_rules` call.
pub fn recognize(eg: &EGraph, rule_idxs: &[usize]) -> Option<RebuildRoles> {
    // First: every rule in THIS call must be a recognized rebuild rule.
    {
        let mut probe = RebuildRoles::default();
        for &i in rule_idxs {
            let rule = eg.rules.get(i).and_then(|r| r.as_ref())?;
            if !classify_rule(rule, &mut probe) {
                return None;
            }
        }
        if probe.uf.is_empty() && probe.views.is_empty() {
            return None;
        }
    }
    // Then: gather the FULL role set from all registered rebuild rules.
    let mut roles = RebuildRoles::default();
    for rule in eg.rules.iter().flatten() {
        let mut r = RebuildRoles::default();
        if classify_rule(rule, &mut r) {
            roles.uf.extend(r.uf);
            roles.uff.extend(r.uff);
            roles.views.extend(r.views);
        }
    }
    // The circuit canonicalizes up to TWO arguments (arity <= 3 view tables:
    // 2 args + output). Decline (interpreter fallback) if any recognized view
    // table is wider — never a regression, just no acceleration.
    for &v in &roles.views {
        let arity = eg.info(v).arity;
        if arity > 3 {
            return None;
        }
    }
    Some(roles)
}

/// The key columns of a `@UF` parent relation: 0 (child), 1 (parent).
const UF_CHILD: usize = 0;
const UF_PARENT: usize = 1;

/// Persistent rebuild-circuit cache: the congruence-shuttle circuit built ONCE
/// and the bookkeeping needed to feed it only per-call DELTAS (so rebuild is
/// O(delta), not O(state) — Stage-1 persistence in service of Stage 3).
pub struct RebuildCache {
    circuit: RebuildCircuit,
    /// Congruence/seed edges already pushed into the circuit's `union_in`
    /// (normalized lo<=hi), so each is shuttled exactly once over the lifetime.
    pushed_unions: std::collections::BTreeSet<(u64, u64)>,
    /// The view rows last fed to the circuit (so we push only the +/- diff).
    /// Keyed by `(table_tag, arg0, arg1, out)` real-id tuple.
    fed_views: HashSet<(u64, u64, u64, u64)>,
    /// The raw `@UF` edges last fed as seed unions (normalized lo<=hi).
    fed_uf_seed: std::collections::BTreeSet<(u64, u64)>,
    /// Stable per-view-table tag (assigned once; index in a sorted role list).
    tag_of: HashMap<FunctionId, u64>,
}

/// Run the entire rebuild fixpoint for `roles` on the PERSISTENT circuit, fed
/// only the per-call delta of raw view rows / `@UF` seed edges, and fold the
/// canonical result back into the mirror. Returns whether the mirror changed.
pub fn run_rebuild(eg: &mut EGraph, roles: &RebuildRoles) -> anyhow::Result<bool> {
    // Snapshot the pre-rebuild mirror for the change check.
    let before: HashMap<FunctionId, HashSet<crate::compile::Row>> = eg
        .mirror
        .iter()
        .filter(|(f, _)| {
            roles.uf.contains(*f) || roles.uff.contains(*f) || roles.views.contains(*f)
        })
        .map(|(f, s)| (*f, (**s).clone()))
        .collect();

    // Assign stable per-table tags (sorted for determinism) and lazily build the
    // persistent circuit + cache.
    let mut view_list: Vec<FunctionId> = roles.views.iter().copied().collect();
    view_list.sort_by_key(|f| f.rep());

    if eg.rebuild_cache.is_none() {
        let circuit = RebuildCircuit::build()?;
        let mut tag_of: HashMap<FunctionId, u64> = HashMap::new();
        for (i, &view) in view_list.iter().enumerate() {
            tag_of.insert(view, (1u64 << 40) + i as u64);
        }
        eg.rebuild_cache = Some(RebuildCache {
            circuit,
            pushed_unions: std::collections::BTreeSet::new(),
            fed_views: HashSet::new(),
            fed_uf_seed: std::collections::BTreeSet::new(),
            tag_of,
        });
    }
    // Any newly-appeared view tables get a fresh tag.
    {
        let cache = eg.rebuild_cache.as_mut().unwrap();
        for (i, &view) in view_list.iter().enumerate() {
            cache
                .tag_of
                .entry(view)
                .or_insert_with(|| (1u64 << 40) + i as u64);
        }
    }

    // Compute the CURRENT raw view-row set and @UF seed set from the mirror.
    let mut cur_views: HashSet<(u64, u64, u64, u64)> = HashSet::new();
    for &view in &view_list {
        let arity = eg.info(view).arity;
        if arity == 0 {
            continue;
        }
        let nargs = arity - 1;
        let tag = eg.rebuild_cache.as_ref().unwrap().tag_of[&view];
        if let Some(set) = eg.mirror.get(&view) {
            for row in set.iter() {
                let arg0 = row_col(row, 0) as u64;
                let arg1 = if nargs >= 2 {
                    row_col(row, 1) as u64
                } else {
                    NO_ARG
                };
                let out = row_col(row, nargs) as u64;
                cur_views.insert((tag, arg0, arg1, out));
            }
        }
    }
    let mut cur_uf_seed: std::collections::BTreeSet<(u64, u64)> = std::collections::BTreeSet::new();
    for &uf in &roles.uf {
        if let Some(set) = eg.mirror.get(&uf) {
            for row in set.iter() {
                let child = row_col(row, UF_CHILD) as u64;
                let parent = row_col(row, UF_PARENT) as u64;
                cur_uf_seed.insert((child.min(parent), child.max(parent)));
            }
        }
    }

    // Push only the DELTA to the persistent circuit.
    let canon_rows;
    let leaders;
    {
        let cache = eg.rebuild_cache.as_mut().unwrap();
        // View-row diff (+1 new, -1 retracted).
        for &(tag, a0, a1, out) in &cur_views {
            if !cache.fed_views.contains(&(tag, a0, a1, out)) {
                cache.circuit.push_view(tag, a0, a1, out, 1);
            }
        }
        for &(tag, a0, a1, out) in &cache.fed_views {
            if !cur_views.contains(&(tag, a0, a1, out)) {
                cache.circuit.push_view(tag, a0, a1, out, -1);
            }
        }
        cache.fed_views = cur_views;
        // @UF seed-union diff (additions only matter — unions are monotone; a
        // removed seed edge stays implied by the leaders it already produced,
        // matching the union-find's monotone "same class" semantics).
        for &(a, b) in &cur_uf_seed {
            if cache.fed_uf_seed.insert((a, b)) && cache.pushed_unions.insert((a, b)) {
                cache.circuit.push_union(a, b);
            }
        }

        let mut pushed = std::mem::take(&mut cache.pushed_unions);
        cache.circuit.run_to_fixpoint(&mut pushed)?;
        cache.pushed_unions = pushed;

        leaders = cache.circuit.read_leaders();
        canon_rows = cache.circuit.read_canon();
    }

    // Synthesize canonical @UFf: (x, leader(x)) for every x with a leader.
    for &uff in &roles.uff {
        let mut resolved: HashSet<crate::compile::Row> = HashSet::new();
        for (&x, &l) in &leaders {
            resolved.insert(vec![x as u32, l as u32].into_boxed_slice());
        }
        eg.mirror.insert(uff, std::rc::Rc::new(resolved));
    }

    // Synthesize canonical @UF: child->leader edge for each non-leader child.
    // This matches the interpreter's path-compressed @UF (every child points
    // directly at its leader; leaders have no edge).
    for &uf in &roles.uf {
        let arity = eg.info(uf).arity;
        let mut resolved: HashSet<crate::compile::Row> = HashSet::new();
        for (&x, &l) in &leaders {
            if x != l {
                let mut row = vec![x as u32, l as u32];
                // Pad the (unit/proof) output column(s) with 0.
                while row.len() < arity {
                    row.push(0);
                }
                resolved.insert(row.into_boxed_slice());
            }
        }
        eg.mirror.insert(uf, std::rc::Rc::new(resolved));
    }

    // Rewrite each view table to the circuit's canonical rows (args + output
    // both canonicalized, congruence-collapsed). Group canon_rows by table tag.
    let mut by_tag: HashMap<u64, Vec<(u64, u64, u64)>> = HashMap::new();
    for (tag, c0, c1, out) in &canon_rows {
        by_tag.entry(*tag).or_default().push((*c0, *c1, *out));
    }
    for &view in &view_list {
        let arity = eg.info(view).arity;
        if arity == 0 {
            continue;
        }
        let nargs = arity - 1;
        let tag = eg.rebuild_cache.as_ref().unwrap().tag_of[&view];
        let mut resolved: HashSet<crate::compile::Row> = HashSet::new();
        if let Some(rows) = by_tag.get(&tag) {
            for &(c0, c1, out) in rows {
                let mut full = vec![c0 as u32];
                if nargs >= 2 {
                    full.push(c1 as u32);
                }
                full.push(out as u32);
                resolved.insert(full.into_boxed_slice());
            }
        }
        eg.mirror.insert(view, std::rc::Rc::new(resolved));
    }

    // Forget per-rule seen snapshots for touched relations so later rulesets see
    // the canonicalized rows as fresh deltas (mirror the interpreter / clear).
    for per_rel in eg.seen.values_mut() {
        for &uf in &roles.uf {
            per_rel.remove(&uf);
        }
        for &uff in &roles.uff {
            per_rel.remove(&uff);
        }
        for &view in &view_list {
            per_rel.remove(&view);
        }
    }

    // Change check: did any touched relation's contents differ?
    let mut changed = false;
    for (f, b) in &before {
        let after = eg.mirror.get(f);
        let same = match after {
            Some(a) => a.len() == b.len() && b.iter().all(|r| a.contains(r)),
            None => b.is_empty(),
        };
        if !same {
            changed = true;
            break;
        }
    }
    Ok(changed)
}
