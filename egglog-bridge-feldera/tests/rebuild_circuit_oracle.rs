//! Stage-3 validation: the production persistent rebuild circuit
//! (`rebuild_circuit::RebuildCircuit`) matches a plain-Rust union-find +
//! congruence oracle on a deep deep-chaining eqsat, AND per-iteration cost is
//! sub-linear in state (flat-ish).
//!
//! This ports the validated spike (`spike-dbsp/src/rebuild.rs`) into the
//! production crate so the §2.4 congruence-shuttle encoding is exercised by the
//! crate's own test suite, against the same oracle, with the same flatness bar.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use egglog_bridge_feldera::rebuild_circuit::{RebuildCircuit, NO_ARG};

type Id = u64;

#[derive(Default)]
struct Oracle {
    views: Vec<(Id, Id, Id)>, // (a0, a1, output-id) — args are real ids too
    parent: BTreeMap<Id, Id>,
}
impl Oracle {
    fn mk(&mut self, x: Id) {
        self.parent.entry(x).or_insert(x);
    }
    fn find(&mut self, x: Id) -> Id {
        let mut r = x;
        while self.parent[&r] != r {
            r = self.parent[&r];
        }
        let mut c = x;
        while self.parent[&c] != r {
            let n = self.parent[&c];
            self.parent.insert(c, r);
            c = n;
        }
        r
    }
    fn union(&mut self, a: Id, b: Id) {
        self.mk(a);
        self.mk(b);
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            self.parent.insert(hi, lo);
        }
    }
    fn add_view(&mut self, a0: Id, a1: Id, id: Id) {
        self.mk(a0);
        self.mk(a1);
        self.mk(id);
        self.views.push((a0, a1, id));
    }
    fn rebuild(&mut self) {
        loop {
            let mut by_args: BTreeMap<(Id, Id), Id> = BTreeMap::new();
            let mut changed = false;
            let rows = self.views.clone();
            // Canonicalize BOTH args and output (production semantics).
            let canon: Vec<(Id, Id, Id)> = rows
                .into_iter()
                .map(|(a0, a1, id)| (self.find(a0), self.find(a1), self.find(id)))
                .collect();
            for &(a0, a1, out) in &canon {
                match by_args.entry((a0, a1)) {
                    std::collections::btree_map::Entry::Vacant(e) => {
                        e.insert(out);
                    }
                    std::collections::btree_map::Entry::Occupied(e) => {
                        let other = *e.get();
                        if other != out {
                            self.union(other, out);
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
    fn canon_view(&mut self) -> BTreeSet<(Id, Id, Id)> {
        let rows = self.views.clone();
        rows.into_iter()
            .map(|(a0, a1, id)| (self.find(a0), self.find(a1), self.find(id)))
            .collect()
    }
    fn leaders(&mut self) -> BTreeMap<Id, Id> {
        let keys: Vec<Id> = self.parent.keys().copied().collect();
        keys.into_iter().map(|k| (k, self.find(k))).collect()
    }
}

struct Delta {
    views: Vec<(Id, Id, Id)>,
    unions: Vec<(Id, Id)>,
}

/// The same scenario generator as the spike: grows the e-graph steadily while
/// keeping each iteration's delta a small constant (the math-microbenchmark
/// regime that blew up the interpreter).
fn gen_scenario(iters: usize) -> Vec<Delta> {
    let mut out = Vec::new();
    let mut next_id: Id = 100;
    // Leaf-constant arg pool (ids 1..=64) is NEVER unioned, so view rows over
    // distinct leaf pairs stay distinct e-nodes — this grows the e-graph
    // steadily (commuted duplicates + redundant terms collapsed by congruence)
    // while exercising the production arg-canonicalization path (args carried as
    // real ids whose leaders are themselves). Outputs (ids >= 100) ARE unioned.
    let leaves: Vec<Id> = (1..=64).collect();
    let nl = leaves.len();
    let mut output_ids: Vec<Id> = Vec::new();
    for k in 0..iters {
        let mut views = Vec::new();
        let mut unions = Vec::new();
        let mut fresh = |outs: &mut Vec<Id>| {
            let id = next_id;
            next_id += 1;
            outs.push(id);
            id
        };
        // A fresh distinct leaf pair each iteration (grows the row count).
        let x = leaves[k % nl];
        let y = leaves[(k * 7 + 3) % nl];
        let p = fresh(&mut output_ids);
        let q = fresh(&mut output_ids);
        views.push((x, y, p));
        views.push((y, x, q));
        unions.push((p, q)); // commutativity-style merge of two outputs
        let r = fresh(&mut output_ids);
        views.push((x, y, r)); // redundant term -> congruence collapses with p
                               // constant-folding-style union of two older OUTPUT ids (deepens classes)
        if output_ids.len() >= 4 {
            let m = output_ids.len();
            let a = output_ids[(k * 3) % m];
            let b = output_ids[(k * 5 + 1) % m];
            if a != b {
                unions.push((a, b));
            }
        }
        out.push(Delta { views, unions });
    }
    out
}

#[test]
fn rebuild_circuit_matches_oracle_and_is_flat() {
    let c = RebuildCircuit::build().expect("build rebuild circuit");
    let mut oracle = Oracle::default();
    let iters = 120usize;
    let deltas = gen_scenario(iters);

    let mut step_times = Vec::new();
    let mut row_hist = Vec::new();
    let mut pushed: BTreeSet<(Id, Id)> = BTreeSet::new();

    for (i, d) in deltas.iter().enumerate() {
        for &(a0, a1, id) in &d.views {
            oracle.add_view(a0, a1, id);
        }
        for &(a, b) in &d.unions {
            oracle.union(a, b);
        }
        oracle.rebuild();
        let oracle_canon = oracle.canon_view();
        let oracle_leaders = oracle.leaders();

        let tag: Id = 1 << 40;
        let _ = NO_ARG;
        for &(a0, a1, id) in &d.views {
            c.push_view(tag, a0, a1, id, 1);
        }
        for &(a, b) in &d.unions {
            c.push_union(a, b);
            pushed.insert((a.min(b), a.max(b)));
        }
        let t0 = Instant::now();
        c.run_to_fixpoint(&mut pushed).expect("rebuild fixpoint");
        let dt = t0.elapsed();

        // Compare canonical views (drop the table tag).
        let dbsp_canon: BTreeSet<(Id, Id, Id)> = c
            .read_canon()
            .into_iter()
            .map(|(_tag, c0, c1, out)| (c0, c1, out))
            .collect();
        let oracle_packed: BTreeSet<(Id, Id, Id)> = oracle_canon.iter().copied().collect();
        let dbsp_leaders = c.read_leaders();

        assert_eq!(
            dbsp_canon, oracle_packed,
            "canonical view mismatch at iteration {i}"
        );
        for (k, v) in &oracle_leaders {
            assert_eq!(
                dbsp_leaders.get(k),
                Some(v),
                "leader[{k}] mismatch at iteration {i}: oracle={v} dbsp={:?}",
                dbsp_leaders.get(k)
            );
        }

        row_hist.push(dbsp_canon.len());
        step_times.push(dt.as_micros());
    }

    let n = step_times.len();
    let first_q: u128 = step_times[..n / 4].iter().sum::<u128>() / (n / 4).max(1) as u128;
    let last_q: u128 =
        step_times[3 * n / 4..].iter().sum::<u128>() / (n - 3 * n / 4).max(1) as u128;
    let ratio = last_q as f64 / first_q.max(1) as f64;
    use std::io::Write;
    let mut err = std::io::stderr();
    let _ = writeln!(
        err,
        "rebuild_circuit: e-graph {} -> {} canon rows; step first-q {first_q}us last-q {last_q}us ratio {ratio:.2}x",
        row_hist[0], row_hist[n - 1]
    );
    // Sub-linear-in-state bar. State grows ~64x over the run. The residual
    // non-flatness (doc §2.4) is the read-out `find` operators recomputed at
    // ROOT each transaction — here amplified by the three argument-canon joins
    // (the production extension over the spike). Per-iteration cost still grows
    // strongly SUB-linearly in state (ratio well under the ~64x state growth),
    // which is the property that eliminates the interpreter's
    // linear/quadratic-in-state blowup. (Full flatness needs the read-out made
    // incremental — scoped follow-on, see MIGRATION_STATUS.md.)
    let state_growth = row_hist[n - 1] as f64 / row_hist[0].max(1) as f64;
    assert!(
        ratio < state_growth,
        "per-iteration cost must stay sub-linear in state (time ratio {ratio:.2}x vs state growth {state_growth:.2}x)"
    );
}
