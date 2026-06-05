//! Milestone-3 load-bearing proof: the relational **table-atom join runs on the
//! Differential-Dataflow engine** (a subprocess compiled from the rule's `.dl`),
//! not the host interpreter — the FlowLog analog of Feldera M4/M5's
//! `transitive_closure_join_runs_on_dbsp`.
//!
//! It drives the canonical transitive-closure rule
//! `path(x, z) :- path(x, y), edge(y, z)` through the interpret-mode FlowLog
//! backend with DD routing **enabled** (`enable_dd_join`), and asserts:
//!
//!   1. the result matches the reference backend round for round (bounded
//!      `(run 1)` ≠ `(run 3)`, the M1 bounded-iteration property);
//!   2. `flowlog_join_stats()` reports `host_rule_runs == 0` and
//!      `dd_rule_runs >= 3` — i.e. the 2-atom join ran ENTIRELY on the DD engine
//!      every round (seminaive: one delta-atom variant fires per round), never
//!      on the host nested-loop fallback.
//!
//! This test compiles a driver subprocess (cold `cargo build` of timely +
//! differential-dataflow). It is `#[ignore]` by default; run with:
//!
//!   EGGLOG_FLOWLOG_CACHE=/tmp/egglog-flowlog-m3cache \
//!   cargo test -p egglog-bridge-flowlog --release \
//!     --test dd_join_proof -- --ignored --nocapture

use std::collections::BTreeSet;

use egglog_backend_trait::{
    Backend, ColumnTy, DefaultVal, FunctionConfig, FunctionId, FunctionRow, MergeFn, QueryEntry,
    RuleId, Value,
};
use egglog_numeric_id::NumericId;

const VAL: u32 = 0;

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

fn seed(b: &mut dyn Backend, edge: FunctionId, path: FunctionId, pairs: &[(u32, u32)]) {
    let rows: Vec<Vec<Value>> = pairs
        .iter()
        .map(|(x, y)| vec![Value::new(*x), Value::new(*y), Value::new(VAL)])
        .collect();
    b.insert_rows(edge, &rows);
    b.insert_rows(path, &rows);
    b.flush_updates();
}

fn add_join_rule(b: &mut dyn Backend, edge: FunctionId, path: FunctionId) -> RuleId {
    let mut rb = b.new_rule("transitive_step", true);
    let x = rb.new_var_named(ColumnTy::Id, "x");
    let y = rb.new_var_named(ColumnTy::Id, "y");
    let z = rb.new_var_named(ColumnTy::Id, "z");
    let val = QueryEntry::Const {
        val: Value::new(VAL),
        ty: ColumnTy::Id,
    };
    rb.query_table(path, &[x.clone(), y.clone(), val.clone()], None)
        .unwrap();
    rb.query_table(edge, &[y.clone(), z.clone(), val.clone()], None)
        .unwrap();
    rb.set(path, &[x, z, val]);
    rb.build().unwrap()
}

fn read_path(b: &dyn Backend, path: FunctionId) -> BTreeSet<(u32, u32)> {
    let mut set = BTreeSet::new();
    b.for_each(path, &mut |row: FunctionRow<'_>| {
        set.insert((row.vals[0].rep(), row.vals[1].rep()));
    });
    set
}

fn run_program_reference(n: usize) -> BTreeSet<(u32, u32)> {
    let mut b: Box<dyn Backend> = Box::new(egglog_bridge::EGraph::default());
    let (edge, path) = add_relations(b.as_mut());
    seed(b.as_mut(), edge, path, &[(1, 2), (2, 3), (3, 4)]);
    let rule = add_join_rule(b.as_mut(), edge, path);
    for _ in 0..n {
        b.run_rules(&[rule]).unwrap();
        b.flush_updates();
    }
    read_path(b.as_ref(), path)
}

/// Run the program on the FlowLog interpret backend with DD routing ENABLED,
/// returning (path pairs, dd_rule_runs, host_rule_runs).
fn run_program_flowlog_dd(n: usize) -> (BTreeSet<(u32, u32)>, u64, u64) {
    let mut eg = egglog_bridge_flowlog::EGraph::new_interpret();
    eg.enable_dd_join();
    let (edge, path) = add_relations(&mut eg);
    seed(&mut eg, edge, path, &[(1, 2), (2, 3), (3, 4)]);
    let rule = add_join_rule(&mut eg, edge, path);
    for _ in 0..n {
        eg.run_rules(&[rule]).unwrap();
        eg.flush_updates();
    }
    let pairs = read_path(&eg, path);
    let (dd, host) = eg.flowlog_join_stats();
    (pairs, dd, host)
}

#[test]
#[ignore = "compiles a DD driver subprocess (cold cargo build); run with --ignored"]
fn transitive_closure_join_runs_on_dd() {
    let ref1 = run_program_reference(1);
    let ref3 = run_program_reference(3);
    let (flw1, dd1, host1) = run_program_flowlog_dd(1);
    let (flw3, dd3, host3) = run_program_flowlog_dd(3);

    use std::io::Write;
    let mut err = std::io::stderr();
    let _ = writeln!(err, "run(1) reference     = {ref1:?}");
    let _ = writeln!(
        err,
        "run(1) flowlog(dd)   = {flw1:?}  (dd_runs={dd1}, host_runs={host1})"
    );
    let _ = writeln!(err, "run(3) reference     = {ref3:?}");
    let _ = writeln!(
        err,
        "run(3) flowlog(dd)   = {flw3:?}  (dd_runs={dd3}, host_runs={host3})"
    );

    // 1. Matches the reference backend, round for round.
    assert_eq!(flw1, ref1, "run(1): FlowLog(DD) must match reference");
    assert_eq!(flw3, ref3, "run(3): FlowLog(DD) must match reference");

    // 2. Bounded extension, not saturation.
    assert_ne!(flw1, flw3, "run(1) and run(3) must differ (bounded)");
    assert!(
        !flw1.contains(&(1, 4)),
        "run(1) must NOT reach the 3-hop pair"
    );
    assert!(flw3.contains(&(1, 4)), "run(3) must reach the 3-hop pair");

    // 3. THE MANDATE: the relational join ran ENTIRELY on the DD engine, never
    //    on the host nested-loop fallback.
    assert_eq!(
        host3, 0,
        "no rule firing should fall back to the host interpreter"
    );
    assert!(
        dd3 >= 3,
        "the 2-atom join must run on DD every round (dd_runs={dd3})"
    );
}
