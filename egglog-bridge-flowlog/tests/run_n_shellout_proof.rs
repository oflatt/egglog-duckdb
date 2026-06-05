//! Milestone-2 load-bearing proof: **runtime rule installation via shell-out**.
//!
//! Same bounded `(run N)` semantics as M1, but the transitive-closure-step rule
//! is NOT backed by a build-time-fixed `.dl`. Instead it is installed AT RUNTIME
//! through the `Backend` trait (`RuleBuilderOps`), translated to `.dl` on the
//! fly, compiled into a thin **driver subprocess** via `cargo build` (cached by
//! rule-set hash), and driven across iterations over a stdin/stdout pipe — one
//! `commit` per `run_rules` call = one bounded hop. It must match the reference
//! backend (`egglog_bridge::EGraph`) round for round.
//!
//! This is M1's `run_n_proof` but via runtime codegen + shell-out compile +
//! subprocess (the M2 bar). The program is identical to M1's:
//!
//!   edge(x, y)                              -- seeded base relation
//!   path(x, y)                              -- seeded = a copy of edge
//!   path(x, z) :- path(x, y), edge(y, z).   -- one join, extends one hop/round

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

/// Build `path(x, z) :- path(x, y), edge(y, z)` AT RUNTIME through the trait.
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

fn run_one_round(b: &mut dyn Backend, rule: RuleId) {
    b.run_rules(&[rule]).unwrap();
    b.flush_updates();
}

fn run_program(b: &mut dyn Backend, n: usize) -> BTreeSet<(u32, u32)> {
    let (edge, path) = add_relations(b);
    seed(b, edge, path, &[(1, 2), (2, 3), (3, 4)]);
    let rule = add_join_rule(b, edge, path);
    for _ in 0..n {
        run_one_round(b, rule);
    }
    read_path(b, path)
}

fn expected(n: usize) -> BTreeSet<(u32, u32)> {
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

/// The M2 proof. Marked `#[ignore]` by default because the first run shells out
/// to `cargo build` (~tens of seconds cold; instant when the rule-set is
/// cached). Run with:
///   cargo test --release --manifest-path egglog-bridge-flowlog/Cargo.toml \
///     --test run_n_shellout_proof -- --ignored --nocapture
#[test]
#[ignore = "runtime shell-out compile (~tens of seconds cold); run with --ignored"]
fn run1_vs_run3_shellout_runtime_install_matches_reference() {
    // Reference backend (in-process bridge).
    let mut reference: Box<dyn Backend> = Box::new(egglog_bridge::EGraph::default());
    let mut reference3: Box<dyn Backend> = Box::new(egglog_bridge::EGraph::default());
    let ref_run1 = run_program(reference.as_mut(), 1);
    let ref_run3 = run_program(reference3.as_mut(), 3);

    // FlowLog backend in SHELL-OUT mode: rule installed at runtime -> codegen
    // -> cargo build -> subprocess driven over the pipe.
    let mut flowlog: Box<dyn Backend> = Box::new(egglog_bridge_flowlog::EGraph::new_shellout());
    let mut flowlog3: Box<dyn Backend> = Box::new(egglog_bridge_flowlog::EGraph::new_shellout());
    let flw_run1 = run_program(flowlog.as_mut(), 1);
    let flw_run3 = run_program(flowlog3.as_mut(), 3);

    use std::io::Write;
    let mut err = std::io::stderr();
    let _ = writeln!(err, "run(1) reference = {ref_run1:?}");
    let _ = writeln!(err, "run(1) flowlog(shellout) = {flw_run1:?}");
    let _ = writeln!(err, "run(3) reference = {ref_run3:?}");
    let _ = writeln!(err, "run(3) flowlog(shellout) = {flw_run3:?}");

    assert_eq!(
        flw_run1, ref_run1,
        "run(1): shell-out FlowLog must match reference"
    );
    assert_eq!(
        flw_run3, ref_run3,
        "run(3): shell-out FlowLog must match reference"
    );
    assert_ne!(
        flw_run1, flw_run3,
        "run(1) and run(3) must differ (bounded, not saturated)"
    );
    assert_eq!(flw_run1, expected(1), "run(1) bounded result");
    assert_eq!(flw_run3, expected(3), "run(3) result");
    assert!(
        !flw_run1.contains(&(1, 4)),
        "run(1) must NOT reach 3-hop pair"
    );
    assert!(flw_run3.contains(&(1, 4)), "run(3) must reach 3-hop pair");
}
