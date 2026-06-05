//! Feasibility SPIKE: egglog **rebuild** (union-find find + congruence,
//! delete-and-rewrite, NON-monotone) as DBSP recursive views in a **persistent
//! circuit fed input deltas across iterations**.
//!
//! This is the load-bearing unknown for "path A" of the Feldera plan
//! (`PLAN.md` §2.3 / §3.3): the Phase-0 spike (`main.rs`) only proved generic
//! recursive transitive closure + retraction on a toy `path` relation. It did
//! NOT model the actual rebuild that the term encoder emits
//! (`seminaive-encoding-add.egg`):
//!
//!   * `parent`/`single_parent` — union-find path union: given two parent
//!     edges, DELETE the stale edge and SET a new one (non-monotone).
//!   * `uf_function_index` — materialize `leader(x)` = the union-find
//!     representative of `x` (a `find`, here a min-reachable closure).
//!   * `congruence` — two view rows with equal args but different outputs
//!     ⟹ union the two outputs (insert a UF edge that feeds back into find).
//!   * `rebuild` — rewrite each view row to use the canonical (leader) output,
//!     DELETING the stale row (non-monotone); the rewrite can create a new
//!     arg-collision, re-triggering congruence ⟹ a JOINT relation+UF fixpoint.
//!
//! We model that joint fixpoint as ONE persistent DBSP `recursive` scope built
//! once. The host pushes only per-iteration DELTAS (new terms from user rules)
//! into input z-sets and calls `transaction()`; DBSP runs the rebuild fixpoint
//! internally and maintains the canonical relations incrementally.
//!
//! ## How find/leader is done WITHOUT a typed aggregate
//!
//! `leader(x)` = the minimum id reachable from `x` over undirected union edges.
//! Rather than fight the typed `aggregate(Min)` generics, we compute it with
//! pure join/antijoin (all `OrdZSet<Tup2>`), which is fully incremental:
//!
//! ```text
//!   reach(x,y)      :- x is a known node (reach(x,x)); reach(x,z),uf(z,y).   (closure, undirected)
//!   dominated(x,y)  :- reach(x,y), reach(x,z), z < y.    (y is not the minimum)
//!   leader(x,y)     :- reach(x,y), not dominated(x,y).   (the unique min)
//! ```
//!
//! `uf` is kept SYMMETRIC (both (a,b) and (b,a)) so reachability is over
//! connected components — exactly union-find's "same class" relation. This
//! sidesteps the encoder's explicit path-union deletes: instead of mutating
//! parent edges, the leader is recomputed from the (monotone, symmetric) union
//! relation, and the NON-monotone delete-and-rewrite shows up where it really
//! matters — the **view table**, whose stale rows are retracted via z-set
//! negative weights when a better (leader-canonical) row exists.
//!
//! ## What is non-monotone here (the crux being de-risked)
//!
//! `canon_view` retracts rows: when `view(a0,a1,id)` has `leader(id)=L != id`,
//! the canonical relation contains `(a0,a1,L)` and NOT `(a0,a1,id)`. As unions
//! accumulate, the canonical row for a given `(a0,a1)` MOVES (old id row
//! disappears, leader row appears) — a delete+insert inside the fixpoint. And
//! congruence reads `canon_view` to discover collisions, so the deletes feed
//! back into more unions: the joint non-monotone recursion the plan flagged as
//! the make-or-break risk.

use anyhow::Result;
use dbsp::{
    typed_batch::OrdZSet, utils::Tup2, utils::Tup3, CircuitHandle, OutputHandle, RootCircuit,
    Stream, ZSetHandle, ZWeight,
};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

type Id = u64;
type UfEdge = Tup2<Id, Id>; // (a, b): a and b are unioned (we keep both directions)
type View = Tup3<Id, Id, Id>; // (arg0, arg1, output-id)

type ZUf = OrdZSet<UfEdge>;
type ZView = OrdZSet<View>;
type ZPair = OrdZSet<Tup2<Id, Id>>;

/// Handles exposed by the persistent rebuild circuit.
struct Circuit {
    handle: CircuitHandle,
    /// Push (+1 insert / -1 retract) raw view rows produced by user rules.
    view_in: ZSetHandle<View>,
    /// Push (+1) union facts (a unioned with b). Symmetric closure is built
    /// inside the circuit.
    union_in: ZSetHandle<UfEdge>,
    /// The canonicalized view relation (each row uses leader outputs); this is
    /// what a reader would `for_each` over.
    canon_out: OutputHandle<ZView>,
    /// leader(x) = representative of x.
    leader_out: OutputHandle<OrdZSet<Tup2<Id, Id>>>,
    /// Congruence-discovered unions emitted this run (diagnostic).
    congru_out: OutputHandle<OrdZSet<UfEdge>>,
    /// Full reachability relation (diagnostic, for divergence debugging).
    reach_out: OutputHandle<OrdZSet<Tup2<Id, Id>>>,
    /// Accumulated raw union edges as the circuit sees them (diagnostic).
    uf_acc_out: OutputHandle<OrdZSet<UfEdge>>,
}

/// leader(x) = min y with reach(x,y). NON-monotone: a pair (x,y) is "dominated"
/// (not the minimum) if some (x,z) with z<y exists; antijoin removes those,
/// leaving the unique minimum per x.
///
/// A macro, not a fn: DBSP's typed operators (`map_index`/`join`/`antijoin`/
/// `distinct`) are implemented per concrete circuit-nesting depth, not for a
/// generic `C`, so this must be expanded both at the root scope and inside the
/// recursive scope (different concrete `Stream` types).
macro_rules! leaders_from_reach {
    ($reach:expr) => {{
        let reach = $reach;
        let reach_by_x = reach.map_index(|Tup2(x, y): &Tup2<Id, Id>| (*x, *y));
        // dominated(x,y): reach(x,y) ⋈ reach(x,z) with z<y
        let dominated = reach_by_x
            .join(&reach_by_x, |x, y, z| {
                if z < y {
                    Tup2(*x, *y)
                } else {
                    Tup2(u64::MAX, u64::MAX)
                }
            })
            .filter(|Tup2(x, _): &Tup2<Id, Id>| *x != u64::MAX)
            .distinct();
        // Key both by the full (x,y) pair; antijoin excludes dominated keys.
        let reach_keyed = reach
            .distinct()
            .map_index(|Tup2(x, y): &Tup2<Id, Id>| (Tup2(*x, *y), ()));
        let dom_keyed = dominated.map_index(|Tup2(x, y): &Tup2<Id, Id>| (Tup2(*x, *y), ()));
        reach_keyed
            .antijoin(&dom_keyed)
            .map(|(Tup2(x, y), ())| Tup2(*x, *y))
    }};
}

/// Build the persistent rebuild circuit ONCE.
fn build() -> Result<Circuit> {
    let (handle, (view_in, union_in, canon_out, leader_out, congru_out, reach_out, uf_acc_out)) =
        RootCircuit::build(|root| {
            let (raw_view, view_in): (Stream<_, ZView>, ZSetHandle<View>) =
                root.add_input_zset::<View>();
            let (raw_union, union_in): (Stream<_, ZUf>, ZSetHandle<UfEdge>) =
                root.add_input_zset::<UfEdge>();

            // The joint, non-monotone rebuild fixpoint lives in ONE recursive
            // scope with TWO mutually-recursive views:
            //   * `uf`        — accumulated union edges (symmetric), grows as
            //                   congruence discovers new equalities.
            //   * `canon`     — the canonicalized view relation (non-monotone:
            //                   rows move from id to leader as unions land).
            // Both are recomputed jointly to fixpoint each transaction.
            // ONE recursive scope carries TWO co-recursive streams jointly to
            // a single fixpoint per transaction (DBSP does not allow NESTING a
            // recursive scope inside another with the typed operators, so the
            // transitive `reach` closure cannot be its own inner scope — it is
            // a co-recursive view in the SAME scope, re-closed as `uf` grows):
            //   * `uf`    — accumulated union edges (grows via congruence).
            //   * `reach` — symmetric-transitive closure of `uf` (e-class
            //               membership); recomputed jointly with `uf`.
            // Because they iterate together, when congruence adds a uf edge,
            // `reach` extends, the leader (min) moves, the canonical view row
            // moves (NON-monotone delete+insert), which can expose a new arg
            // collision => more congruence — exactly the joint relation+UF
            // non-monotone fixpoint the plan flagged as the make-or-break risk.
            // §2.4 ENCODING (the fix the spike validates):
            //
            // The naive co-recursive `(uf, reach)` tuple with congruence feedback
            // does NOT persist its closure across transactions (older nodes lose
            // all reach rows -> divergence). But `persist_probe.rs` proves a
            // SINGLE recursive closure over a STABLE INPUT relation persists
            // perfectly. So:
            //   * `uf` is an ordinary INTEGRATED INPUT relation (seed unions +
            //     congruence edges fed back through the input handle by the host),
            //     NOT a co-recursive view.
            //   * `reach` is a SINGLE recursive closure over that `uf` input —
            //     exactly the shape that persists.
            //   * congruence is computed at ROOT from the converged reach and
            //     emitted as an output delta; the HOST pushes it back into
            //     `union_in` for the next transaction (rebuild is scheduled
            //     `(saturate)` anyway, so it converges over a few transactions,
            //     each O(delta)).
            let raw_view_acc = raw_view.integrate();
            let uf_acc = raw_union.integrate();
            // sym: symmetric closure of the accumulated union edges.
            let sym = uf_acc
                .flat_map(|Tup2(a, b): &UfEdge| vec![Tup2(*a, *b), Tup2(*b, *a)])
                .distinct();
            let sym_for_scope = sym.clone();
            // self-reach seeds for every node that appears anywhere.
            let self_reach0 = raw_view_acc
                .flat_map(|Tup3(a0, a1, id): &View| {
                    vec![Tup2(*a0, *a0), Tup2(*a1, *a1), Tup2(*id, *id)]
                })
                .plus(&sym.map(|Tup2(a, _): &UfEdge| Tup2(*a, *a)))
                .distinct();
            // reach = single recursive closure over the `uf`/sym INPUT (persists
            // across transactions, per persist_probe).
            let reach: Stream<_, ZPair> = root
                .recursive(move |child, reach: Stream<_, ZPair>| {
                    let self_reach = self_reach0.delta0(child);
                    let sym = sym_for_scope.delta0(child);
                    let reach_by_2nd = reach.map_index(|Tup2(x, y): &Tup2<Id, Id>| (*y, *x));
                    let sym_by_src = sym.map_index(|Tup2(y, z): &UfEdge| (*y, *z));
                    let step = reach_by_2nd.join(&sym_by_src, |_y, x, z| Tup2(*x, *z));
                    Ok(self_reach.plus(&step).distinct())
                })
                .unwrap();

            // Derive the read-out relations from the CONVERGED reach.
            let leader_pairs: Stream<_, ZPair> = leaders_from_reach!(&reach);
            let leader_of = leader_pairs.map_index(|Tup2(x, l): &Tup2<Id, Id>| (*x, *l));
            let canon = raw_view_acc
                .map_index(|Tup3(a0, a1, id): &View| (*id, Tup2(*a0, *a1)))
                .join(&leader_of, |_id, Tup2(a0, a1), l| Tup3(*a0, *a1, *l))
                .distinct();
            let canon_by_args = canon.map_index(|Tup3(a0, a1, out): &View| ((*a0, *a1), *out));
            let congru = canon_by_args
                .join(&canon_by_args, |_args, o1, o2| {
                    if o1 < o2 {
                        Tup2(*o1, *o2)
                    } else {
                        Tup2(u64::MAX, u64::MAX)
                    }
                })
                .filter(|Tup2(a, _): &UfEdge| *a != u64::MAX)
                .distinct();

            Ok((
                view_in,
                union_in,
                canon.integrate().output(),
                leader_pairs.integrate().output(),
                congru.integrate().output(),
                reach.integrate().output(),
                raw_union.integrate().output(),
            ))
        })?;

    Ok(Circuit {
        handle,
        view_in,
        union_in,
        canon_out,
        leader_out,
        congru_out,
        reach_out,
        uf_acc_out,
    })
}

fn read_view(out: &OutputHandle<ZView>) -> BTreeSet<(Id, Id, Id)> {
    let mut s = BTreeSet::new();
    for (Tup3(a, b, c), (), w) in out.consolidate().iter() {
        let w: ZWeight = w;
        if w > 0 {
            s.insert((a, b, c));
        }
    }
    s
}

fn read_leader(out: &OutputHandle<OrdZSet<Tup2<Id, Id>>>) -> BTreeMap<Id, Id> {
    let mut m = BTreeMap::new();
    for (Tup2(x, l), (), w) in out.consolidate().iter() {
        let w: ZWeight = w;
        if w > 0 {
            m.insert(x, l);
        }
    }
    m
}

fn count_pos(out: &OutputHandle<OrdZSet<Tup2<Id, Id>>>) -> usize {
    out.consolidate().iter().filter(|(_, (), w)| *w > 0).count()
}

// ===========================================================================
// Reference oracle: a plain union-find + congruence rebuild in Rust, run to
// fixpoint over the same inputs. The DBSP circuit must match it exactly.
// ===========================================================================
#[derive(Default)]
struct Oracle {
    // raw view rows as inserted (arg0,arg1,id)
    views: Vec<(Id, Id, Id)>,
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
        // path compress
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
            self.parent.insert(hi, lo); // min is the leader
        }
    }
    fn add_view(&mut self, a0: Id, a1: Id, id: Id) {
        self.mk(id);
        self.views.push((a0, a1, id));
    }
    fn rebuild(&mut self) {
        // congruence to fixpoint
        loop {
            let mut by_args: BTreeMap<(Id, Id), Id> = BTreeMap::new();
            let mut changed = false;
            let rows: Vec<(Id, Id, Id)> = self.views.clone();
            let canon: Vec<(Id, Id, Id)> = rows
                .into_iter()
                .map(|(a0, a1, id)| (a0, a1, self.find(id)))
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
        let rows: Vec<(Id, Id, Id)> = self.views.clone();
        rows.into_iter()
            .map(|(a0, a1, id)| (a0, a1, self.find(id)))
            .collect()
    }
    fn leaders(&mut self) -> BTreeMap<Id, Id> {
        let keys: Vec<Id> = self.parent.keys().copied().collect();
        keys.into_iter().map(|k| (k, self.find(k))).collect()
    }
}

/// One iteration's worth of input: new view rows + new direct unions.
struct Delta {
    views: Vec<(Id, Id, Id)>,
    unions: Vec<(Id, Id)>,
}

fn main() -> Result<()> {
    println!("=== SPIKE: rebuild (UF find + congruence, non-monotone) as persistent DBSP recursive views ===\n");

    let c = build()?;
    let mut oracle = Oracle::default();

    // -----------------------------------------------------------------------
    // A small eqsat that mirrors `seminaive-encoding-add.egg`:
    // start with (Add 1 2)=id10, then commutativity creates (Add 2 1)=id11
    // and unions id10≡id11. Congruence then must keep ONE canonical row per
    // arg pair. We extend it across several iterations to watch per-iteration
    // cost as the e-graph grows.
    // -----------------------------------------------------------------------

    // Iteration scenario generator: a chain of Add terms where each round adds
    // commuted duplicates that must be merged by congruence, plus a "constant
    // folding"-style union that collapses a fraction of ids. This grows the
    // e-graph while keeping each iteration's DELTA small and roughly constant.
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    let deltas = gen_scenario(iters);

    let mut next_iter_times = Vec::new();
    let mut total_rows_hist = Vec::new();
    // Congruence edges already pushed into `union_in` (normalized lo<=hi), so the
    // shuttle pushes each only once.
    let mut uf_pushed: BTreeSet<(Id, Id)> = BTreeSet::new();

    for (i, d) in deltas.iter().enumerate() {
        // Feed oracle.
        for &(a0, a1, id) in &d.views {
            oracle.add_view(a0, a1, id);
        }
        for &(a, b) in &d.unions {
            oracle.union(a, b);
        }
        oracle.rebuild();
        let oracle_canon = oracle.canon_view();
        let oracle_leaders = oracle.leaders();

        // Feed DBSP this iteration's view + seed-union deltas, then SATURATE the
        // rebuild via the §2.4 congruence shuttle: step once, read the congruence
        // output, push any NOT-yet-applied congruence edge into `union_in`, and
        // step again — until no new congruence edge appears. Each step is
        // O(delta) (only the new congruence edges flow). This is the host-side
        // feedback that keeps `reach` a single persisted recursive closure over a
        // STABLE union INPUT (which provably persists) while still letting
        // congruence grow the union relation.
        for &(a0, a1, id) in &d.views {
            c.view_in.push(Tup3(a0, a1, id), 1);
        }
        for &(a, b) in &d.unions {
            c.union_in.push(Tup2(a, b), 1);
            uf_pushed.insert((a.min(b), a.max(b)));
        }
        let t0 = Instant::now();
        c.handle.transaction()?;
        let mut shuttle_steps = 1usize;
        loop {
            // Read all congruence edges discovered so far and push the new ones.
            let mut new_edges: Vec<(Id, Id)> = Vec::new();
            for (Tup2(a, b), (), w) in c.congru_out.consolidate().iter() {
                if w > 0 {
                    let key = (a.min(b), a.max(b));
                    if uf_pushed.insert(key) {
                        new_edges.push((a, b));
                    }
                }
            }
            if new_edges.is_empty() {
                break;
            }
            for (a, b) in new_edges {
                c.union_in.push(Tup2(a, b), 1);
            }
            c.handle.transaction()?;
            shuttle_steps += 1;
            if shuttle_steps > 1000 {
                anyhow::bail!("congruence shuttle did not converge at iter {i}");
            }
        }
        let dt = t0.elapsed();
        let _ = shuttle_steps;

        let dbsp_canon = read_view(&c.canon_out);
        let dbsp_leaders = read_leader(&c.leader_out);
        let _congru = count_pos(&c.congru_out);

        let canon_ok = dbsp_canon == oracle_canon;
        // Compare only leaders for nodes the oracle knows (DBSP self-reach also
        // includes union endpoints; oracle.parent has all of them too).
        let leaders_ok = oracle_leaders
            .iter()
            .all(|(k, v)| dbsp_leaders.get(k) == Some(v));

        total_rows_hist.push(dbsp_canon.len());
        next_iter_times.push(dt.as_micros());

        let status = if canon_ok && leaders_ok { "OK " } else { "MISMATCH" };
        if i < 6 || i % 5 == 0 || !(canon_ok && leaders_ok) {
            println!(
                "iter {:>3}: delta(views={}, unions={})  canon_rows={:>4}  uf_nodes={:>4}  step={:>6}us  [{}]",
                i,
                d.views.len(),
                d.unions.len(),
                dbsp_canon.len(),
                dbsp_leaders.len(),
                dt.as_micros(),
                status
            );
        }
        if !(canon_ok && leaders_ok) {
            println!("  oracle canon ({}): {:?}", oracle_canon.len(), oracle_canon.iter().take(12).collect::<Vec<_>>());
            println!("  dbsp   canon ({}): {:?}", dbsp_canon.len(), dbsp_canon.iter().take(12).collect::<Vec<_>>());
            // show first leader divergence + the reachable set for that node
            for (k, v) in &oracle_leaders {
                if dbsp_leaders.get(k) != Some(v) {
                    println!("  leader[{}]: oracle={} dbsp={:?}", k, v, dbsp_leaders.get(k));
                    let mut reach_k: Vec<Id> = Vec::new();
                    for (Tup2(a, b), (), w) in c.reach_out.consolidate().iter() {
                        if w > 0 && a == *k {
                            reach_k.push(b);
                        }
                    }
                    reach_k.sort();
                    println!("  dbsp reach[{}] = {:?}", k, reach_k);
                    // also dump reach for the chain node and the target leader
                    for probe in [*v, dbsp_leaders.get(k).copied().unwrap_or(0)] {
                        let mut rp: Vec<Id> = Vec::new();
                        for (Tup2(a, b), (), w) in c.reach_out.consolidate().iter() {
                            if w > 0 && a == probe {
                                rp.push(b);
                            }
                        }
                        rp.sort();
                        println!("  dbsp reach[{}] = {:?}", probe, rp);
                    }
                    let mut ocls: Vec<Id> = oracle_leaders
                        .iter()
                        .filter(|(_, &lv)| lv == *v)
                        .map(|(&n, _)| n)
                        .collect();
                    ocls.sort();
                    println!("  oracle class of {} (leader {}): {:?}", k, v, ocls);
                    let mut ufedges: Vec<(Id, Id)> = Vec::new();
                    for (Tup2(a, b), (), w) in c.uf_acc_out.consolidate().iter() {
                        if w > 0 {
                            ufedges.push((a, b));
                        }
                    }
                    ufedges.sort();
                    println!("  dbsp accumulated raw_union edges: {:?}", ufedges);
                    println!(
                        "  this-iter delta unions fed: {:?}  views: {:?}",
                        d.unions, d.views
                    );
                    break;
                }
            }
            anyhow::bail!("DBSP diverged from oracle at iteration {i}");
        }
    }

    // -----------------------------------------------------------------------
    // Flatness analysis: is per-iteration cost O(delta) (flat) as state grows?
    // -----------------------------------------------------------------------
    println!("\n=== Per-iteration cost vs. e-graph size ===");
    println!("(deltas are ~constant-size each iteration; flat time => O(delta), the goal)\n");
    let n = next_iter_times.len();
    let first_q: u128 = next_iter_times[..n / 4].iter().sum::<u128>() / (n / 4).max(1) as u128;
    let last_q: u128 =
        next_iter_times[3 * n / 4..].iter().sum::<u128>() / (n - 3 * n / 4).max(1) as u128;
    println!("  e-graph grew from {} to {} canon rows", total_rows_hist[0], total_rows_hist[n - 1]);
    println!("  mean step time, first quarter: {first_q} us");
    println!("  mean step time, last  quarter: {last_q} us");
    let ratio = last_q as f64 / first_q.max(1) as f64;
    println!("  growth ratio (last/first): {ratio:.2}x");
    if ratio < 4.0 {
        println!("  => FLAT-ISH: per-iteration cost is roughly O(delta), not O(state). GOOD.");
    } else {
        println!("  => per-iteration cost grows with state (ratio >= 4x). Investigate.");
    }

    println!("\nALL ITERATIONS MATCHED THE ORACLE.");
    Ok(())
}

/// Generate a multi-iteration eqsat scenario whose e-graph GROWS steadily
/// (canonical rows grow ~linearly to the thousands) while each iteration's
/// input DELTA stays a small constant. This is the math-microbenchmark regime
/// that blew up the host interpreter (O(state) per iteration); we want to see
/// per-iteration DBSP cost stay roughly flat as total state climbs.
///
/// Each iteration k builds `Add` terms whose ARGS are outputs from EARLIER
/// iterations (so the e-graph deepens, like real eqsat), and adds:
///   * a commuted duplicate `(Add y x)` unioned with `(Add x y)` (commutativity),
///   * a redundant `(Add x y)` with a fresh id (congruence must collapse it),
///   * a "constant-folding"-style union of two older terms (more merges),
/// keeping the per-iteration delta at ~4 view rows + 2 unions regardless of k.
fn gen_scenario(iters: usize) -> Vec<Delta> {
    let mut out = Vec::new();
    let mut next_id: Id = 100;
    let mut all_ids: Vec<Id> = (1..=8).collect(); // leaf constants
    for k in 0..iters {
        let mut views = Vec::new();
        let mut unions = Vec::new();
        let mut fresh = |all: &mut Vec<Id>| {
            let id = next_id;
            next_id += 1;
            all.push(id);
            id
        };
        // Args drawn from the GROWING id pool (terms built over earlier terms),
        // but only a constant number of new rows per iteration.
        let n = all_ids.len();
        let x = all_ids[k % n];
        let y = all_ids[(k * 7 + 3) % n];
        let p = fresh(&mut all_ids);
        let q = fresh(&mut all_ids);
        views.push((x, y, p));
        views.push((y, x, q));
        unions.push((p, q)); // commutativity
                             // redundant (x,y) term with a fresh id -> congruence collapses it
        let r = fresh(&mut all_ids);
        views.push((x, y, r));
        // a term over the new outputs, to deepen the graph
        let s = fresh(&mut all_ids);
        views.push((p, q, s));
        // constant-folding-style union of two older ids (more component merges)
        if n >= 4 {
            let a = all_ids[(k * 3) % n];
            let b = all_ids[(k * 5 + 1) % n];
            if a != b {
                unions.push((a, b));
            }
        }
        out.push(Delta { views, unions });
    }
    out
}
