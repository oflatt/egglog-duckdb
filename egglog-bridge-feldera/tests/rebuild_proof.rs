//! Milestone-2 proof: union-find + rebuild, the research crux.
//!
//! egglog's rebuild is **already expressed as rules** by the term encoder: a
//! `@uf` parent table plus path-compress / single-parent / uf-index rulesets
//! with `(delete …)` / `(set …)` actions, run in a scheduled order to a fixed
//! point. This test hand-encodes exactly those term-mode rulesets (see
//! `src/proofs/proof_encoding.rs::declare_sort`) and drives them through the
//! `Backend` trait on BOTH the reference backend (`egglog_bridge::EGraph`) and
//! the Feldera/DBSP backend, then asserts the **per-function tuple counts
//! match** after rebuild — the milestone-2 success bar.
//!
//! ## The encoded program (term mode, no proofs)
//!
//!   @uf  : relation (child parent)                       -- the parent edges
//!   @uff : function (child) -> leader  :merge ordering-min -- the UF index
//!
//!   uf_index     : (@uf a b)                  -> (set (@uff a) b)
//!   path_compress: (@uf a b)(@uf b c)(!= b c) -> (delete (@uf a b))(set (@uf a c))
//!   single_parent: (@uf a b)(@uff a c)(!= b c)-> (delete (@uf a b))
//!                                                (set (@uf b c))(set (@uff b) c)
//!
//! Schedule (from `proof_encoding.rs::rebuild`):
//!   loop { saturate single_parent; saturate path_compress; saturate uf_index }
//!   until nothing changes.
//!
//! Seed: the union edges of a chain `4~3~2~1` as `@uf` rows
//! `(2,1) (3,2) (4,3)` (child=larger, parent=smaller — the encoder's
//! `(ordering-max ordering-min)` orientation). After rebuild every child must
//! resolve to leader `1`.

use std::collections::BTreeSet;

use egglog_backend_trait::{
    Backend, ColumnTy, DefaultVal, ExternalFunctionId, FunctionConfig, FunctionId, FunctionRow,
    MergeFn, QueryEntry, RuleId, Value,
};
use egglog_core_relations::make_external_func;
use egglog_numeric_id::NumericId;

/// Register `@uf` (Unit-output function over (child,parent)) and `@uff`
/// (1->1 function with `ordering-min` merge). Returns their ids plus the
/// `ordering-min` primitive id used by `@uff`'s merge.
fn add_tables(b: &mut dyn Backend) -> (FunctionId, FunctionId) {
    // @uf: matches the term encoder's `(function @uf (S S) Unit :merge old)` —
    // a function keyed by (child, parent) with a Unit output. We model Unit as
    // the constant 0, so the output column is always 0 and the key is the
    // (child, parent) pair. `remove(@uf, [a,b])` retracts the row keyed (a,b).
    let uf = b.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::Const(Value::new(0)),
        merge: MergeFn::Old,
        name: "@uf".to_string(),
        can_subsume: false,
    });
    // @uff: a function (child) -> leader, matching the term encoder's
    // `(function @uff (S) S :merge (ordering-min old new))`. We register an
    // `ordering-min` primitive and use it as a `MergeFn::Primitive([Old, New])`,
    // exactly as the frontend lowers `:merge (ordering-min old new)`. (We must
    // NOT use `MergeFn::UnionId` here: on the reference that invokes egglog's
    // *internal* union-find and canonicalizes Id columns, which is a different
    // semantics than the term encoder's data-level min and would diverge.)
    let min = make_external_func(|_state, args: &[Value]| {
        // args = [old, new]; return the numerically smaller representative.
        Some(if args[0].rep() <= args[1].rep() {
            args[0]
        } else {
            args[1]
        })
    });
    let min_id = b.register_external_func(Box::new(min));
    let uff = b.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::Const(Value::new(0)),
        merge: MergeFn::Primitive(min_id, vec![MergeFn::Old, MergeFn::New]),
        name: "@uff".to_string(),
        can_subsume: false,
    });
    (uf, uff)
}

/// Register the `!=` predicate primitive used by the rebuild guards. Returns its
/// id. The external returns `Some(())`-ish (a sentinel value) when its two
/// arguments differ, halting the rule otherwise — exactly egglog's `!=`.
fn add_neq(b: &mut dyn Backend) -> ExternalFunctionId {
    let neq = make_external_func(|_state, args: &[Value]| {
        if args.len() == 2 && args[0] != args[1] {
            // Return any value (the rule never reads it); reuse args[0].
            Some(args[0])
        } else {
            None
        }
    });
    b.register_external_func(Box::new(neq))
}

fn var(rb: &mut dyn egglog_backend_trait::RuleBuilderOps, name: &str) -> QueryEntry {
    rb.new_var_named(ColumnTy::Id, name)
}

/// The Unit output value for `@uf` (modeled as the constant 0).
fn unit() -> QueryEntry {
    QueryEntry::Const {
        val: Value::new(0),
        ty: ColumnTy::Id,
    }
}

/// uf_index: (@uf a b) -> (set (@uff a) b).
fn add_uf_index(b: &mut dyn Backend, uf: FunctionId, uff: FunctionId) -> RuleId {
    let mut rb = b.new_rule("uf_index", true);
    let a = var(rb.as_mut(), "a");
    let bb = var(rb.as_mut(), "b");
    let w = var(rb.as_mut(), "w"); // @uf's Unit output (wildcard)
    rb.query_table(uf, &[a.clone(), bb.clone(), w], None)
        .unwrap();
    rb.set(uff, &[a, bb]);
    rb.build().unwrap()
}

/// path_compress: (@uf a b)(@uf b c)(!= b c)
///   -> (delete (@uf a b))(set (@uf a c)).
fn add_path_compress(b: &mut dyn Backend, uf: FunctionId, neq: ExternalFunctionId) -> RuleId {
    let mut rb = b.new_rule("path_compress", true);
    let a = var(rb.as_mut(), "a");
    let bb = var(rb.as_mut(), "b");
    let c = var(rb.as_mut(), "c");
    let w1 = var(rb.as_mut(), "w1");
    let w2 = var(rb.as_mut(), "w2");
    rb.query_table(uf, &[a.clone(), bb.clone(), w1], None)
        .unwrap();
    rb.query_table(uf, &[bb.clone(), c.clone(), w2], None)
        .unwrap();
    // (!= b c): pass [b, c, ret]; ret is a fresh unused var. Name the primitive
    // so the Feldera backend can recognize it and lower it to a filter (the
    // bridge identifies prims by id and treats this name as a no-op).
    rb.rename_prim(neq, "!=".to_string());
    let ret = var(rb.as_mut(), "ne_ret");
    rb.query_prim(neq, &[bb.clone(), c.clone(), ret], ColumnTy::Id)
        .unwrap();
    rb.remove(uf, &[a.clone(), bb]); // key only (child, parent)
    rb.set(uf, &[a, c, unit()]);
    rb.build().unwrap()
}

/// single_parent: (@uf a b)(@uff a c)(!= b c)
///   -> (delete (@uf a b))(set (@uf b c))(set (@uff b) c).
fn add_single_parent(
    b: &mut dyn Backend,
    uf: FunctionId,
    uff: FunctionId,
    neq: ExternalFunctionId,
) -> RuleId {
    let mut rb = b.new_rule("single_parent", true);
    let a = var(rb.as_mut(), "a");
    let bb = var(rb.as_mut(), "b");
    let c = var(rb.as_mut(), "c");
    let w = var(rb.as_mut(), "w");
    rb.query_table(uf, &[a.clone(), bb.clone(), w], None)
        .unwrap();
    // (= c (@uff a)) is a function-table body atom binding output c.
    rb.query_table(uff, &[a.clone(), c.clone()], None).unwrap();
    rb.rename_prim(neq, "!=".to_string());
    let ret = var(rb.as_mut(), "ne_ret");
    rb.query_prim(neq, &[bb.clone(), c.clone(), ret], ColumnTy::Id)
        .unwrap();
    rb.remove(uf, &[a, bb.clone()]); // key only (child, parent)
    rb.set(uf, &[bb.clone(), c.clone(), unit()]);
    rb.set(uff, &[bb, c]);
    rb.build().unwrap()
}

/// Seed `@uf` with the given (child, parent) edges (Unit output = 0).
fn seed(b: &mut dyn Backend, uf: FunctionId, edges: &[(u32, u32)]) {
    let rows: Vec<Vec<Value>> = edges
        .iter()
        .map(|(c, p)| vec![Value::new(*c), Value::new(*p), Value::new(0)])
        .collect();
    b.insert_rows(uf, &rows);
    b.flush_updates();
}

/// Run `rule` repeatedly (one pass per call) until it stops changing the db,
/// flushing between passes. This is one `(saturate rule)`.
fn saturate(b: &mut dyn Backend, rule: RuleId) -> bool {
    let mut any = false;
    loop {
        let changed = b.run_rules(&[rule]).unwrap().changed();
        b.flush_updates();
        if changed {
            any = true;
        } else {
            break;
        }
    }
    any
}

/// The full rebuild schedule: loop the three saturated rulesets until a whole
/// pass makes no change. Mirrors `proof_encoding.rs::rebuild`.
fn rebuild(b: &mut dyn Backend, uf_index: RuleId, path_compress: RuleId, single_parent: RuleId) {
    loop {
        let mut changed = false;
        changed |= saturate(b, single_parent);
        changed |= saturate(b, path_compress);
        changed |= saturate(b, uf_index);
        if !changed {
            break;
        }
    }
}

/// Read a 2-column relation/function's rows as a set of (col0, col1) pairs.
fn read_pairs(b: &dyn Backend, f: FunctionId) -> BTreeSet<(u32, u32)> {
    let mut set = BTreeSet::new();
    b.for_each(f, &mut |row: FunctionRow<'_>| {
        set.insert((row.vals[0].rep(), row.vals[1].rep()));
    });
    set
}

/// Build the program on `b`, seed the given union edges, run rebuild, and
/// return the final (`@uf`, `@uff`) contents.
fn run_rebuild_program(
    b: &mut dyn Backend,
    edges: &[(u32, u32)],
) -> (BTreeSet<(u32, u32)>, BTreeSet<(u32, u32)>) {
    let (uf, uff) = add_tables(b);
    let neq = add_neq(b);
    let uf_index = add_uf_index(b, uf, uff);
    let path_compress = add_path_compress(b, uf, neq);
    let single_parent = add_single_parent(b, uf, uff, neq);

    seed(b, uf, edges);
    // Prime @uff from the seed edges (uf_index) before the first single_parent.
    saturate(b, uf_index);

    rebuild(b, uf_index, path_compress, single_parent);

    (read_pairs(b, uf), read_pairs(b, uff))
}

/// Run one rebuild scenario on both backends and assert per-function tuple
/// counts (and exact contents) match.
fn assert_rebuild_matches(label: &str, edges: &[(u32, u32)]) {
    let mut reference: Box<dyn Backend> = Box::new(egglog_bridge::EGraph::default());
    let mut feldera: Box<dyn Backend> = Box::new(egglog_bridge_feldera::EGraph::new());

    let (ref_uf, ref_uff) = run_rebuild_program(reference.as_mut(), edges);
    let (fel_uf, fel_uff) = run_rebuild_program(feldera.as_mut(), edges);

    use std::io::Write;
    let mut err = std::io::stderr();
    let _ = writeln!(
        err,
        "=== M2 rebuild scenario `{label}` (seed {edges:?}) ==="
    );
    let _ = writeln!(err, "  @uf  reference ({} rows) = {ref_uf:?}", ref_uf.len());
    let _ = writeln!(err, "  @uf  feldera   ({} rows) = {fel_uf:?}", fel_uf.len());
    let _ = writeln!(
        err,
        "  @uff reference ({} rows) = {ref_uff:?}",
        ref_uff.len()
    );
    let _ = writeln!(
        err,
        "  @uff feldera   ({} rows) = {fel_uff:?}",
        fel_uff.len()
    );

    assert_eq!(
        fel_uf.len(),
        ref_uf.len(),
        "[{label}] @uf tuple count must match reference"
    );
    assert_eq!(
        fel_uff.len(),
        ref_uff.len(),
        "[{label}] @uff tuple count must match reference"
    );
    assert_eq!(
        fel_uf, ref_uf,
        "[{label}] @uf contents must match reference"
    );
    assert_eq!(
        fel_uff, ref_uff,
        "[{label}] @uff contents must match reference"
    );
    // Semantic check: every child resolves to leader 1 after rebuild.
    for (child, leader) in &fel_uff {
        if *child != 1 {
            assert_eq!(
                *leader, 1,
                "[{label}] child {child} must resolve to leader 1"
            );
        }
    }
}

#[test]
fn rebuild_matches_reference_tuple_counts() {
    // A spread of union-find shapes, all driven through the same term-encoded
    // rebuild rulesets on both backends:
    assert_rebuild_matches("single-union", &[(2, 1)]);
    assert_rebuild_matches("2-chain", &[(2, 1), (3, 2)]);
    assert_rebuild_matches("3-chain", &[(2, 1), (3, 2), (4, 3)]);
    // Multi-parent collapse: node 3 starts with parents {1,2}; forces
    // single_parent to redirect (the (@uf a b)(@uff a c)(!= b c) rule), the
    // congruence-collapse half of rebuild — not just path compression.
    assert_rebuild_matches("diamond", &[(3, 1), (3, 2), (2, 1)]);
    // A wider star plus a chain, mixing both rules.
    assert_rebuild_matches("star+chain", &[(5, 1), (4, 1), (3, 2), (2, 1)]);
}
