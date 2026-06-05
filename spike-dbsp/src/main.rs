//! Phase-0 confirmatory spike for an egglog-on-DBSP backend.
//!
//! Validates the load-bearing PLAN.md assumptions by driving the `dbsp`
//! crate *in-process, synchronously*, the way an egglog backend would.
//! (Pinned to dbsp 0.150.0 to match the installed rustc; the public API used
//! here is identical in the latest 0.305.0 — see Cargo.toml.)
//!
//!   1. Build a circuit with a RECURSIVE relation modelling egglog's
//!        path(x,y) :- edge(x,y);  path(x,z) :- path(x,y), edge(y,z)
//!      using a PLAIN 2-column `path` relation (NO weight/hop columns), which
//!      is exactly the egglog reachability shape. The recursive scope's
//!      built-in `distinct` is what makes this terminate even with cycles.
//!   2. Feed input DELTAS via an input ZSetHandle, call transaction(), and
//!      confirm the recursive scope runs to FIXPOINT inside that single
//!      transaction (the "one run_rules ruleset call = one saturating
//!      transaction()" adapter).
//!   3. Read the output relation back BETWEEN steps via an OutputHandle
//!      (egglog inspects the DB between schedule steps).
//!   4. RETRACTION: feed a negative-weight delta for an edge and confirm the
//!      closure SHRINKS correctly on the next step (egglog rebuild relies on
//!      delete-and-rewrite => negative weights).
//!
//! Run with:  cargo run --release --bin spike

use anyhow::Result;
// `Circuit` trait is in scope implicitly via RootCircuit's inherent methods in
// this dbsp version; `recursive`/`delta0` resolve without an explicit import.
use dbsp::{utils::Tup2, OrdZSet, OutputHandle, RootCircuit, Stream, ZSetHandle, ZWeight};
use std::collections::BTreeSet;

type Node = u64;
type Edge = Tup2<Node, Node>;

/// Build the reachability circuit.
///
/// Returns:
///   - `edge_in`:  push handle to feed +1/-1 edge deltas.
///   - `path_out`: output handle to read the consolidated `path` relation.
///
/// The whole recursive Datalog ruleset lives in ONE `recursive` scope, so a
/// single `step()` runs it to fixpoint.
fn build_circuit(
    root: &mut RootCircuit,
) -> Result<(ZSetHandle<Edge>, OutputHandle<OrdZSet<Edge>>)> {
    // `edge` is an input relation fed by the host via a push handle. This is
    // the egglog-backend shape: add_input_zset() -> (stream, handle).
    let (edges, edge_in): (Stream<RootCircuit, OrdZSet<Edge>>, ZSetHandle<Edge>) =
        root.add_input_zset::<Edge>();

    // path(x,z) is the recursive view. `recursive` builds a nested scope with
    // a z^-1 feedback edge and an automatic `distinct` on the body output:
    //     path = distinct( edge  UNION  (path JOIN edge) )
    // The automatic `distinct` is the termination guarantee: even on a cyclic
    // graph the set of (x,z) pairs is finite, so the weights stop changing and
    // the scope reaches a fixed point WITHIN one outer step().
    let path: Stream<RootCircuit, OrdZSet<Edge>> = root.recursive(
        |child: &_, path: Stream<_, OrdZSet<Edge>>| {
            // Import parent streams into the child scope via delta0.
            let edges = edges.delta0(child);

            // Base rule:  path(x,y) :- edge(x,y).
            let base = edges.clone();

            // Recursive rule: path(x,z) :- path(x,y), edge(y,z).
            // Index path by its middle column y, edge by its source column y,
            // join on y, project to (x,z).
            let path_by_mid = path.map_index(|Tup2(x, y)| (*y, *x)); // key=y, val=x
            let edge_by_src = edges.map_index(|Tup2(y, z)| (*y, *z)); // key=y, val=z
            let step = path_by_mid.join(&edge_by_src, |_y, x, z| Tup2(*x, *z));

            // path = base UNION step  (distinct is applied by `recursive`).
            Ok(base.plus(&step))
        },
    )?;

    // IMPORTANT: the recursive view's stream carries the PER-STEP DELTA
    // (DBSP recursion emits Δx = new - old between steps; this is why a
    // retraction shows up as negative-weight rows). egglog needs to read the
    // *accumulated* relation between steps, so we `integrate()` the delta
    // stream before exposing it: integrate() maintains the running sum, so
    // `consolidate()` on this handle returns the FULL current `path` relation.
    // (This integrated trace IS the Rust-side materialized mirror the backend
    // uses for point lookups.)
    Ok((edge_in, path.integrate().output()))
}

/// Snapshot the current consolidated `path` relation as a set of (x,z) pairs,
/// dropping any rows whose net weight is <= 0 (a materialized "mirror" of the
/// relation, which is what the backend keeps for point lookups).
fn read_paths(out: &OutputHandle<OrdZSet<Edge>>) -> BTreeSet<(Node, Node)> {
    let mut set = BTreeSet::new();
    out.consolidate()
        .iter()
        .for_each(|(Tup2(x, z), (), w): (Edge, (), ZWeight)| {
            if w > 0 {
                set.insert((x, z));
            }
        });
    set
}

fn fmt(set: &BTreeSet<(Node, Node)>) -> String {
    let v: Vec<String> = set.iter().map(|(a, b)| format!("{a}->{b}")).collect();
    format!("{{ {} }}  (n={})", v.join(", "), v.len())
}

fn main() -> Result<()> {
    // ---- Build the circuit once, up front (DBSP's static-graph model). ----
    // Note: we DON'T use a Generator source (the tutorial's approach); we use a
    // push handle so the host drives deltas synchronously, exactly like the
    // backend's insert_rows / retraction path. To make the recursive scope see
    // `edges`, the input stream is captured by the closure via build's env.
    let (circuit, (edge_in, path_out)) = RootCircuit::build(|root| {
        let handles = build_circuit(root).map_err(|e| anyhow::anyhow!(e))?;
        Ok(handles)
    })?;

    // =====================================================================
    // STEP 1: insert a chain  0 -> 1 -> 2 -> 3 -> 4  (acyclic).
    // Expect the FULL transitive closure after ONE step (proves fixpoint
    // inside a single step()).
    // =====================================================================
    for (a, b) in [(0u64, 1u64), (1, 2), (2, 3), (3, 4)] {
        edge_in.push(Tup2(a, b), 1); // +1 weight = insert
    }
    circuit.transaction()?; // one logical clock tick: absorb deltas, run recursive scope to fixpoint, commit
    let s1 = read_paths(&path_out);
    println!("STEP 1 (insert chain 0->1->2->3->4)");
    println!("  path = {}", fmt(&s1));
    let mut expected1 = BTreeSet::new();
    for x in 0..=4u64 {
        for z in (x + 1)..=4u64 {
            expected1.insert((x, z));
        }
    }
    assert_eq!(s1, expected1, "STEP 1: full transitive closure expected");
    println!("  OK: full closure (10 pairs) computed to fixpoint in ONE step\n");

    // =====================================================================
    // STEP 2: RETRACTION. Remove edge 1->2 (negative weight). The closure
    // must SHRINK: everything that only reached through 1->2 disappears.
    // This is what egglog rebuild's delete-and-rewrite relies on.
    // =====================================================================
    edge_in.push(Tup2(1u64, 2u64), -1); // -1 weight = retract
    circuit.transaction()?; // one logical clock tick: absorb deltas, run recursive scope to fixpoint, commit
    let s2 = read_paths(&path_out);
    println!("STEP 2 (retract edge 1->2)");
    println!("  path = {}", fmt(&s2));
    // Now two components: {0->1} and {2,3,4} fully connected.
    let mut expected2 = BTreeSet::new();
    expected2.insert((0u64, 1u64));
    for x in 2..=4u64 {
        for z in (x + 1)..=4u64 {
            expected2.insert((x, z));
        }
    }
    assert_eq!(s2, expected2, "STEP 2: closure must shrink after retraction");
    assert!(s2.len() < s1.len(), "closure should be smaller");
    println!("  OK: closure SHRANK correctly after retraction ({} -> {} pairs)\n", s1.len(), s2.len());

    // =====================================================================
    // STEP 3: CYCLE termination. Add 4->2 (and put 1->2 back) to form a
    // cycle 2->3->4->2. A plain path(x,y) relation + the recursive scope's
    // `distinct` must TERMINATE (unlike the tutorial's weighted version,
    // which diverges on cycles). This is the egglog-relevant case.
    // =====================================================================
    edge_in.push(Tup2(1u64, 2u64), 1); // restore 1->2
    edge_in.push(Tup2(4u64, 2u64), 1); // close the cycle 2->3->4->2
    circuit.transaction()?; // one logical clock tick: absorb deltas, run recursive scope to fixpoint, commit
    let s3 = read_paths(&path_out);
    println!("STEP 3 (restore 1->2, add 4->2 forming cycle 2->3->4->2)");
    println!("  path = {}", fmt(&s3));
    // Reachability: 0 and 1 reach everything; {2,3,4} are mutually reachable
    // (including self-loops 2->2, 3->3, 4->4 via the cycle).
    let cyc = [2u64, 3, 4];
    let mut expected3 = BTreeSet::new();
    for z in [1u64, 2, 3, 4] {
        expected3.insert((0u64, z));
    }
    for &z in &cyc {
        expected3.insert((1u64, z));
    }
    for &x in &cyc {
        for &z in &cyc {
            expected3.insert((x, z)); // includes self-loops
        }
    }
    assert_eq!(s3, expected3, "STEP 3: cyclic closure (with self-loops) expected");
    println!("  OK: cyclic graph reached FIXPOINT (distinct => termination), self-loops present\n");

    // =====================================================================
    // STEP 4: no-op step. The frontend's outer loop calls run_rules again
    // expecting `changed=false`. Confirm a step with no input delta leaves
    // the relation unchanged (the adapter reports no change => loop exits).
    // =====================================================================
    circuit.transaction()?; // one logical clock tick: absorb deltas, run recursive scope to fixpoint, commit
    let s4 = read_paths(&path_out);
    assert_eq!(s4, s3, "STEP 4: no input => relation unchanged");
    println!("STEP 4 (no input delta): path unchanged (n={}) => adapter reports changed=false\n", s4.len());

    println!("ALL ASSERTIONS PASSED");
    Ok(())
}
