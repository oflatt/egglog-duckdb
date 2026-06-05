//! Spot-check: verify the feldera and flowlog backends extract the SAME
//! low-cost terms (and costs) as the reference backend for a few representative
//! extraction programs. This complements the shared-snapshot cost-parity check
//! in `tests/files.rs` by also comparing the extracted *terms*.
//!
//! Run with: `cargo test --release --test extract_parity_spotcheck -- --nocapture`

use egglog::{CommandOutput, EGraph};

/// Run `program` on the given backend and return the rendered `(extract …)`
/// results as `(cost, term-string)` pairs, in command order.
fn extract_results(make: impl Fn() -> EGraph, program: &str) -> Vec<(i64, String)> {
    let mut egraph = make();
    egraph.ensure_no_reserved_symbols(false);
    let outputs = egraph
        .parse_and_run_program(None, program)
        .expect("program should run");
    outputs
        .iter()
        .filter_map(|o| match o {
            CommandOutput::ExtractBest(termdag, cost, term) => {
                Some((*cost as i64, termdag.to_string(*term)))
            }
            _ => None,
        })
        .collect()
}

fn check_program(name: &str, program: &str) {
    let reference = extract_results(EGraph::new_with_term_encoding, program);
    assert!(
        !reference.is_empty(),
        "{name}: reference produced no extraction output"
    );
    let feldera = extract_results(
        || EGraph::with_feldera_backend().expect("feldera init"),
        program,
    );
    let flowlog = extract_results(
        || EGraph::with_flowlog_backend().expect("flowlog init"),
        program,
    );

    println!("==== {name} ====");
    for (i, ((rc, rt), ((fc, ft), (lc, lt)))) in reference
        .iter()
        .zip(feldera.iter().zip(flowlog.iter()))
        .enumerate()
    {
        println!("  extract #{i}:");
        println!("    reference: cost={rc} term={rt}");
        println!("    feldera:   cost={fc} term={ft}");
        println!("    flowlog:   cost={lc} term={lt}");
    }

    // Costs must match exactly across all three backends.
    let ref_costs: Vec<i64> = reference.iter().map(|(c, _)| *c).collect();
    let fel_costs: Vec<i64> = feldera.iter().map(|(c, _)| *c).collect();
    let flo_costs: Vec<i64> = flowlog.iter().map(|(c, _)| *c).collect();
    assert_eq!(ref_costs, fel_costs, "{name}: feldera cost mismatch");
    assert_eq!(ref_costs, flo_costs, "{name}: flowlog cost mismatch");
}

#[test]
fn spotcheck_math_simplify() {
    // 0*x = 0, x+0 = x. Best term for (Add (Num 1) (Mul (Num 0) (Num 5))) is (Num 1).
    let program = r#"
(datatype Math (Num i64) (Add Math Math) (Mul Math Math))
(let e (Add (Num 1) (Mul (Num 0) (Num 5))))
(rewrite (Mul (Num 0) x) (Num 0))
(rewrite (Add x (Num 0)) x)
(run 10)
(extract e)
"#;
    check_program("math_simplify", program);
}

#[test]
fn spotcheck_cost_driven() {
    // Custom :cost annotations force the cheaper constructor to win.
    let program = r#"
(datatype Thing
  (Cheap i64 :cost 1)
  (Expensive i64 :cost 100))
(union (Cheap 0) (Expensive 0))
(let t (Expensive 0))
(extract t)
"#;
    check_program("cost_driven", program);
}

#[test]
fn spotcheck_fib_demand() {
    let program = std::fs::read_to_string("tests/fibonacci-demand.egg")
        .expect("read tests/fibonacci-demand.egg");
    check_program("fibonacci_demand", &program);
}
