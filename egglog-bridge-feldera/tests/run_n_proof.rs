//! Milestone-1 load-bearing proof.
//!
//! egglog's `(run N)` applies a ruleset N times **with bounded extension per
//! round** — a transitive-closure-style rule extends N hops, NOT to full
//! closure. This test proves the Feldera backend reproduces that bounded
//! behavior *behind the `Backend` trait* and that it **matches the reference
//! backend (`egglog_bridge::EGraph`)** on the same program:
//!
//!   - `(run 1)` and `(run 3)` produce DIFFERENT, bounded results.
//!   - Each round's result equals the reference backend's, round for round.
//!
//! Program (a single-join derivation, the milestone's target shape):
//!
//!   edge(x, y)                          -- seeded base relation
//!   path(x, y)                          -- seeded = a copy of edge
//!   path(x, z) :- path(x, y), edge(y, z).   -- one join, extends one hop/round
//!
//! Both relations are modeled as 3-column tables `(x, y) -> value` so the
//! reference backend (which splits the last column off as the FD value) and the
//! Feldera backend (which matches whole rows) agree: keys are (x, y), value is
//! a fixed constant. The join binds `y` across the two body atoms.

use std::collections::BTreeSet;

use egglog_backend_trait::{
    Backend, ColumnTy, DefaultVal, FunctionConfig, FunctionId, FunctionRow, MergeFn, QueryEntry,
    RuleId, Value,
};
use egglog_numeric_id::NumericId;

const VAL: u32 = 0; // fixed FD value for the (x,y) -> value tables.

/// Register `edge` and `path` as `(x, y) -> value` tables.
fn add_relations(b: &mut dyn Backend) -> (FunctionId, FunctionId) {
    let mk = |b: &mut dyn Backend, name: &str| {
        b.add_table(FunctionConfig {
            schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
            default: DefaultVal::Const(Value::new(VAL)),
            merge: MergeFn::AssertEq,
            name: name.to_string(),
            can_subsume: false,
        })
    };
    let edge = mk(b, "edge");
    let path = mk(b, "path");
    (edge, path)
}

/// Seed the given (x, y) pairs into both `edge` and `path`.
fn seed(b: &mut dyn Backend, edge: FunctionId, path: FunctionId, pairs: &[(u32, u32)]) {
    let rows: Vec<Vec<Value>> = pairs
        .iter()
        .map(|(x, y)| vec![Value::new(*x), Value::new(*y), Value::new(VAL)])
        .collect();
    b.insert_rows(edge, &rows);
    b.insert_rows(path, &rows);
    b.flush_updates();
}

/// Build the rule `path(x, z) :- path(x, y), edge(y, z)`.
fn add_join_rule(b: &mut dyn Backend, edge: FunctionId, path: FunctionId) -> RuleId {
    let mut rb = b.new_rule("transitive_step", true);
    let x = rb.new_var_named(ColumnTy::Id, "x");
    let y = rb.new_var_named(ColumnTy::Id, "y");
    let z = rb.new_var_named(ColumnTy::Id, "z");
    let val = QueryEntry::Const {
        val: Value::new(VAL),
        ty: ColumnTy::Id,
    };
    // body: path(x, y) , edge(y, z)
    rb.query_table(path, &[x.clone(), y.clone(), val.clone()], None)
        .unwrap();
    rb.query_table(edge, &[y.clone(), z.clone(), val.clone()], None)
        .unwrap();
    // head: path(x, z)
    rb.set(path, &[x, z, val]);
    rb.build().unwrap()
}

/// Read the (x, y) pairs of `path` out of a backend via `for_each`.
fn read_path(b: &dyn Backend, path: FunctionId) -> BTreeSet<(u32, u32)> {
    let mut set = BTreeSet::new();
    b.for_each(path, &mut |row: FunctionRow<'_>| {
        set.insert((row.vals[0].rep(), row.vals[1].rep()));
    });
    set
}

/// Run one egglog iteration (`run_rules` once) then `flush_updates`, mirroring
/// the frontend's per-round loop body.
fn run_one_round(b: &mut dyn Backend, rule: RuleId) {
    b.run_rules(&[rule]).unwrap();
    b.flush_updates();
}

/// Build the program on `b`, run `n` rounds, return `path`'s (x,y) pairs.
fn run_program(b: &mut dyn Backend, n: usize) -> BTreeSet<(u32, u32)> {
    let (edge, path) = add_relations(b);
    // A 4-node chain: 1->2->3->4 (ids 1..=4; 0 is reserved padding in Feldera).
    seed(b, edge, path, &[(1, 2), (2, 3), (3, 4)]);
    let rule = add_join_rule(b, edge, path);
    for _ in 0..n {
        run_one_round(b, rule);
    }
    read_path(b, path)
}

/// Expected `path` after `n` bounded rounds of `path(x,z):-path(x,y),edge(y,z)`
/// over the chain 1->2->3->4: round k adds all pairs reachable in <= k+1 hops.
fn expected(n: usize) -> BTreeSet<(u32, u32)> {
    // Base (round 0 contents): the 3 edges. Each round extends reachability by
    // one more hop. After n rounds, all pairs (x,y) with 1 <= y-x <= n+1.
    let mut set = BTreeSet::new();
    let max_hop = n + 1;
    for x in 1u32..=4 {
        for y in (x + 1)..=4 {
            if (y - x) as usize <= max_hop {
                set.insert((x, y));
            }
        }
    }
    set
}

#[test]
fn run1_vs_run3_bounded_and_matches_reference() {
    // Reference backend.
    let mut reference: Box<dyn Backend> = Box::new(egglog_bridge::EGraph::default());
    // Feldera backend.
    let mut feldera: Box<dyn Backend> = Box::new(egglog_bridge_feldera::EGraph::new());

    let ref_run1 = run_program(reference.as_mut(), 1);
    let fel_run1 = run_program(feldera.as_mut(), 1);

    // Fresh backends for the run-3 program (state is per-egraph).
    let mut reference3: Box<dyn Backend> = Box::new(egglog_bridge::EGraph::default());
    let mut feldera3: Box<dyn Backend> = Box::new(egglog_bridge_feldera::EGraph::new());
    let ref_run3 = run_program(reference3.as_mut(), 3);
    let fel_run3 = run_program(feldera3.as_mut(), 3);

    // Print the proof output via the stderr handle directly (the `eprintln!`
    // macro is disallowed project-wide in favor of `log`, but a test wants
    // visible output without a logger).
    use std::io::Write;
    let mut err = std::io::stderr();
    let _ = writeln!(err, "run(1) reference = {ref_run1:?}");
    let _ = writeln!(err, "run(1) feldera   = {fel_run1:?}");
    let _ = writeln!(err, "run(3) reference = {ref_run3:?}");
    let _ = writeln!(err, "run(3) feldera   = {fel_run3:?}");

    // 1. Feldera matches the reference backend, round for round.
    assert_eq!(fel_run1, ref_run1, "run(1): Feldera must match reference");
    assert_eq!(fel_run3, ref_run3, "run(3): Feldera must match reference");

    // 2. (run 1) and (run 3) are DIFFERENT and BOUNDED (not full closure).
    assert_ne!(
        fel_run1, fel_run3,
        "run(1) and run(3) must differ (bounded extension, not saturation)"
    );

    // 3. Concrete bounded expectations:
    //    run(1): one extra hop  -> {(1,2),(2,3),(3,4),(1,3),(2,4)}
    //    run(3): full closure   -> adds (1,4) as well.
    assert_eq!(fel_run1, expected(1), "run(1) bounded result");
    assert_eq!(fel_run3, expected(3), "run(3) result (full closure here)");

    // The (1,4) pair (3 hops) is present after run(3) but ABSENT after run(1).
    assert!(
        !fel_run1.contains(&(1, 4)),
        "run(1) must NOT reach 3-hop pair"
    );
    assert!(fel_run3.contains(&(1, 4)), "run(3) must reach 3-hop pair");
}

/// Milestone 4: the transitive-closure rule's body join must run on the DBSP
/// dataflow engine, NOT on the host interpreter fallback. We run the program on
/// a concrete Feldera `EGraph`, then read `dbsp_join_stats()`: the multi-atom
/// join `path(x,y), edge(y,z)` is DBSP-eligible, so every rule firing must be
/// counted on the DBSP path and zero on the host path.
#[test]
fn transitive_closure_join_runs_on_dbsp() {
    let mut feldera = egglog_bridge_feldera::EGraph::new();
    {
        let b: &mut dyn Backend = &mut feldera;
        let (edge, path) = add_relations(b);
        seed(b, edge, path, &[(1, 2), (2, 3), (3, 4)]);
        let rule = add_join_rule(b, edge, path);
        for _ in 0..3 {
            b.run_rules(&[rule]).unwrap();
            b.flush_updates();
        }
    }
    let (dbsp_runs, host_runs) = feldera.dbsp_join_stats();
    assert_eq!(
        host_runs, 0,
        "the 2-atom join must NOT fall back to the host interpreter"
    );
    assert!(
        dbsp_runs >= 3,
        "expected the join to fire on DBSP each round (got {dbsp_runs})"
    );
}
