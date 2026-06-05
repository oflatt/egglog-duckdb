//! Minimal probe: does a SINGLE recursive transitive-closure view persist its
//! contents across `transaction()` calls when fed one edge delta per
//! transaction? This isolates the persistence question from the rebuild's
//! non-monotone congruence feedback.
//!
//! We replicate `main.rs`'s working pattern exactly (one recursive `reach`
//! stream over `delta0(edges)`), but feed edges INCREMENTALLY — one new edge
//! per transaction — and check after each step that the FULL closure of all
//! edges fed so far is present. If old pairs disappear, the recursive view does
//! not persist across transactions (the rebuild blocker); if they remain, the
//! blocker is specific to the co-recursive non-monotone tuple.

use anyhow::Result;
use dbsp::{typed_batch::OrdZSet, utils::Tup2, OutputHandle, RootCircuit, Stream, ZSetHandle, ZWeight};
use std::collections::BTreeSet;

type Edge = Tup2<u64, u64>;

fn main() -> Result<()> {
    let (circuit, (edge_in, reach_out)): (
        _,
        (ZSetHandle<Edge>, OutputHandle<OrdZSet<Edge>>),
    ) = RootCircuit::build(|root| {
        let (edges, edge_in): (Stream<_, OrdZSet<Edge>>, ZSetHandle<Edge>) =
            root.add_input_zset::<Edge>();
        let reach = root
            .recursive(|child, reach: Stream<_, OrdZSet<Edge>>| {
                let edges = edges.delta0(child);
                // self-reach for every edge endpoint
                let self_r = edges
                    .flat_map(|Tup2(a, b): &Edge| vec![Tup2(*a, *a), Tup2(*b, *b)])
                    .distinct();
                let base = edges.clone();
                let r_by_mid = reach.map_index(|Tup2(x, y): &Edge| (*y, *x));
                let e_by_src = edges.map_index(|Tup2(y, z): &Edge| (*y, *z));
                let step = r_by_mid.join(&e_by_src, |_y, x, z| Tup2(*x, *z));
                Ok(self_r.plus(&base).plus(&step).distinct())
            })
            .unwrap();
        Ok((edge_in, reach.integrate().output()))
    })?;

    let read = |out: &OutputHandle<OrdZSet<Edge>>| -> BTreeSet<(u64, u64)> {
        let mut s = BTreeSet::new();
        for (Tup2(a, b), (), w) in out.consolidate().iter() {
            let w: ZWeight = w;
            if w > 0 {
                s.insert((a, b));
            }
        }
        s
    };

    // Feed a growing chain 1->2->3->...->N, ONE edge per transaction.
    let n = 8u64;
    let mut all_edges: Vec<(u64, u64)> = Vec::new();
    let mut ok = true;
    for k in 1..n {
        edge_in.push(Tup2(k, k + 1), 1);
        all_edges.push((k, k + 1));
        circuit.transaction()?;
        let got = read(&reach_out);
        // expected: transitive closure (with self-loops) of all_edges so far
        let mut exp: BTreeSet<(u64, u64)> = BTreeSet::new();
        for &(a, _) in &all_edges {
            exp.insert((a, a));
        }
        for &(_, b) in &all_edges {
            exp.insert((b, b));
        }
        // chain: i->j for all i<=j in [1..=k+1]
        for i in 1..=(k + 1) {
            for j in i..=(k + 1) {
                exp.insert((i, j));
            }
        }
        let match_ = got == exp;
        if !match_ {
            ok = false;
        }
        println!(
            "txn {k}: fed edge {k}->{}  reach_rows={:>3}  expected={:>3}  [{}]",
            k + 1,
            got.len(),
            exp.len(),
            if match_ { "OK" } else { "MISMATCH" }
        );
        if !match_ {
            let missing: Vec<_> = exp.difference(&got).collect();
            let extra: Vec<_> = got.difference(&exp).collect();
            println!("   missing: {:?}", missing);
            println!("   extra:   {:?}", extra);
        }
    }

    println!(
        "\nPERSISTENCE VERDICT: a single recursive transitive-closure view {} \
         across transactions when fed incremental edge deltas.",
        if ok {
            "PERSISTS correctly"
        } else {
            "does NOT persist (loses old pairs)"
        }
    );
    Ok(())
}
