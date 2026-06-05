//! Persistent rebuild circuit (Stage 3 of the interpreter-deprecation
//! migration; see `docs/feldera_persistent_circuit_design.md`).
//!
//! This is the production port of the validated feasibility spike
//! (`spike-dbsp/src/rebuild.rs`): egglog **rebuild** (union-find find +
//! congruence, non-monotone delete-and-rewrite) expressed as DBSP recursive
//! views in a **persistent circuit fed input deltas across transactions**, with
//! the §2.4 **congruence-shuttle** encoding that is *required* for convergence
//! (the naive co-recursive `(uf, reach)` tuple silently loses closure across
//! transactions; routing congruence-discovered edges back through the INPUT
//! boundary fixes it — proven in the spike on all 120 iterations of a deep
//! deep-chaining eqsat).
//!
//! ## What the term encoder's rebuild actually is
//!
//! For each eq-sort, the proof/term encoder (`src/proofs/proof_encoding.rs`)
//! emits:
//!
//! * `@uf(a, b)` — a relation of union (parent) edges. `path_compress` /
//!   `single_parent` mutate it (delete stale edge, set new), `@uf_function_index`
//!   mirrors it into `@uff`.
//! * `@uff(a) = leader` — the uf-index function, `:merge (ordering-min …)`, i.e.
//!   `leader(a) = min` over the connected component of `a`.
//! * `@XView(args…, out)` — one view table per constructor; the `@rebuilding`
//!   ruleset's congruence rule unions the outputs of two view rows with equal
//!   args, and `@rebuilding_cleanup` rewrites each view row to its canonical
//!   (leader) output, deleting the stale row.
//!
//! The whole point of the migration is that this fixpoint, when scheduled by
//! the host interpreter, re-reads the full relation mirror on **every**
//! `run_rules` call — so it is O(state) per call and blows up super-linearly on
//! `math-microbenchmark` (`(run 11)` ≈ 56 s vs. `(run 9)` ≈ 4 s). The
//! persistent circuit feeds only per-transaction deltas, so the rebuild cost is
//! sub-linear in state.
//!
//! ## The encoding (mirrors the spike, generalized over view arity)
//!
//! `leader(x)` = the minimum id reachable from `x` over undirected union edges,
//! computed with pure join/antijoin (fully incremental, no typed `Min`):
//!
//! ```text
//!   reach(x,y)      :- self-reach; reach(x,z), sym(z,y).      (closure)
//!   dominated(x,y)  :- reach(x,y), reach(x,z), z < y.         (y not the min)
//!   leader(x,y)     :- reach(x,y), not dominated(x,y).        (the unique min)
//! ```
//!
//! `uf` is kept SYMMETRIC inside the circuit so reachability = connected
//! component. `reach` is a SINGLE recursive closure over the `uf` INPUT z-set
//! (the shape that provably persists across transactions). Congruence is
//! computed at root from the converged `reach` + canonical views and emitted as
//! an output delta; the host **shuttle** pushes new congruence edges back into
//! the `uf` input and re-steps until no new edge appears.
//!
//! ## Generality vs. the spike
//!
//! The spike fixed view arity at 2 with **leaf-constant** args (already
//! canonical). Production view-table arguments are themselves e-class ids that
//! get unioned, so congruence must be over *leader-canonicalized* arguments.
//! This circuit therefore carries each view row as `Tup4(table_tag, arg0, arg1,
//! out)` of **real ids** and canonicalizes BOTH arguments and the output via
//! `leader_of` joins before computing congruence — the production extension over
//! the spike. A `table_tag` distinguishes view tables so a single shared circuit
//! handles them all without cross-constructor congruence collisions. View tables
//! of arity > 3 (more than two arguments) are not handled here; the integration
//! (`circuit_rebuild.rs`) falls back to the interpreter for them. See
//! [`RebuildCircuit`].

use anyhow::Result;
use dbsp::{
    typed_batch::OrdZSet, utils::Tup2, utils::Tup4, CircuitHandle, OutputHandle, RootCircuit,
    Stream, ZSetHandle, ZWeight,
};

type Id = u64;
type UfEdge = Tup2<Id, Id>;
/// A view row flowing through the circuit, carrying REAL ids so the circuit can
/// canonicalize the arguments (not just the output) by the union-find leader —
/// the production extension over the spike (whose view args were leaf
/// constants, already canonical). Layout: `(table_tag, arg0, arg1, out)`.
///
/// `table_tag` distinguishes view tables so congruence never collides across
/// constructors. `arg1` is `NO_ARG` (a reserved sentinel) for arity-1
/// constructors. Constructors with >2 arguments are not handled here (the
/// integration falls back to the interpreter for them).
type View = Tup4<Id, Id, Id, Id>; // (table_tag, arg0, arg1, output-id)

/// Sentinel for "no second argument" (arity-1 constructors). Chosen above any
/// real egglog id (`u32` reps), and never a leader, so it is left unchanged by
/// the leader canonicalization (it maps to itself).
pub const NO_ARG: Id = u64::MAX - 1;

type ZUf = OrdZSet<UfEdge>;
type ZView = OrdZSet<View>;
type ZPair = OrdZSet<Tup2<Id, Id>>;
/// A canonical view row read out: (table_tag, arg0, arg1, out).
type CanonRow = (Id, Id, Id, Id);

/// Handles exposed by the persistent rebuild circuit (one per view table).
pub struct RebuildCircuit {
    handle: CircuitHandle,
    /// Push (+1 / -1) raw view rows produced by user rules / seeding.
    view_in: ZSetHandle<View>,
    /// Push (+1) union facts. Symmetric closure is built inside the circuit.
    union_in: ZSetHandle<UfEdge>,
    /// The canonicalized view relation (each row uses leader outputs).
    canon_out: OutputHandle<ZView>,
    /// leader(x) = representative of x.
    leader_out: OutputHandle<ZPair>,
    /// Congruence-discovered unions emitted this run (shuttled by the host).
    congru_out: OutputHandle<OrdZSet<UfEdge>>,
}

/// leader(x) = min y with reach(x,y). NON-monotone: a pair (x,y) is "dominated"
/// (not the minimum) if some (x,z) with z<y exists; antijoin removes those,
/// leaving the unique minimum per x.
///
/// A macro, not a fn: DBSP's typed operators are implemented per concrete
/// circuit-nesting depth, so this must be expanded at the root scope.
macro_rules! leaders_from_reach {
    ($reach:expr) => {{
        let reach = $reach;
        let reach_by_x = reach.map_index(|Tup2(x, y): &Tup2<Id, Id>| (*x, *y));
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
        let reach_keyed = reach
            .distinct()
            .map_index(|Tup2(x, y): &Tup2<Id, Id>| (Tup2(*x, *y), ()));
        let dom_keyed = dominated.map_index(|Tup2(x, y): &Tup2<Id, Id>| (Tup2(*x, *y), ()));
        reach_keyed
            .antijoin(&dom_keyed)
            .map(|(Tup2(x, y), ())| Tup2(*x, *y))
    }};
}

impl RebuildCircuit {
    /// Build the persistent rebuild circuit ONCE (for one view table). Mirrors
    /// `spike-dbsp/src/rebuild.rs::build` exactly — the §2.4 shuttle encoding.
    pub fn build() -> Result<RebuildCircuit> {
        let (handle, (view_in, union_in, canon_out, leader_out, congru_out)) =
            RootCircuit::build(|root| {
                let (raw_view, view_in): (Stream<_, ZView>, ZSetHandle<View>) =
                    root.add_input_zset::<View>();
                let (raw_union, union_in): (Stream<_, ZUf>, ZSetHandle<UfEdge>) =
                    root.add_input_zset::<UfEdge>();

                let raw_view_acc = raw_view.integrate();
                let uf_acc = raw_union.integrate();
                let sym = uf_acc
                    .flat_map(|Tup2(a, b): &UfEdge| vec![Tup2(*a, *b), Tup2(*b, *a)])
                    .distinct();
                let sym_for_scope = sym.clone();
                // Every real id appearing anywhere (args + output) needs a
                // self-reach seed so it has a leader (itself if in no union).
                // NO_ARG is a sentinel, not a real id — it self-reaches so the
                // arg-canon join leaves it unchanged.
                let self_reach0 = raw_view_acc
                    .flat_map(|Tup4(_tag, a0, a1, id): &View| {
                        vec![Tup2(*a0, *a0), Tup2(*a1, *a1), Tup2(*id, *id)]
                    })
                    .plus(&sym.map(|Tup2(a, _): &UfEdge| Tup2(*a, *a)))
                    .distinct();
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

                let leader_pairs: Stream<_, ZPair> = leaders_from_reach!(&reach);
                let leader_of = leader_pairs.map_index(|Tup2(x, l): &Tup2<Id, Id>| (*x, *l));
                // Canonicalize BOTH arguments AND the output to their leaders.
                // This is the production extension: view-table arguments are
                // themselves e-class ids that get unioned, so congruence must be
                // over leader-canonicalized arguments (the spike's args were leaf
                // constants and skipped this). Three successive joins canonicalize
                // arg0, arg1, then out; NO_ARG maps to itself (self-reach seed).
                let canon_a0 = raw_view_acc
                    .map_index(|Tup4(tag, a0, a1, id): &View| (*a0, Tup4(*tag, *a0, *a1, *id)))
                    .join(&leader_of, |_key, Tup4(tag, _a0, a1, id), l0| {
                        Tup4(*tag, *l0, *a1, *id)
                    });
                let canon_a1 = canon_a0
                    .map_index(|Tup4(tag, c0, a1, id): &View| (*a1, Tup4(*tag, *c0, *a1, *id)))
                    .join(&leader_of, |_key, Tup4(tag, c0, _a1, id), l1| {
                        Tup4(*tag, *c0, *l1, *id)
                    });
                let canon = canon_a1
                    .map_index(|Tup4(tag, c0, c1, id): &View| (*id, Tup4(*tag, *c0, *c1, *id)))
                    .join(&leader_of, |_key, Tup4(tag, c0, c1, _id), lo| {
                        Tup4(*tag, *c0, *c1, *lo)
                    })
                    .distinct();
                // congruence: same (tag, canon_arg0, canon_arg1), different
                // canonical output ⟹ union the outputs.
                let canon_by_args =
                    canon.map_index(|Tup4(tag, c0, c1, out): &View| ((*tag, *c0, *c1), *out));
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
                ))
            })?;

        Ok(RebuildCircuit {
            handle,
            view_in,
            union_in,
            canon_out,
            leader_out,
            congru_out,
        })
    }

    /// Push a +1/-1 view row `(table_tag, arg0, arg1, out)` with REAL ids.
    /// Use [`NO_ARG`] for `arg1` on arity-1 constructors.
    pub fn push_view(&self, table_tag: Id, arg0: Id, arg1: Id, out: Id, weight: ZWeight) {
        self.view_in.push(Tup4(table_tag, arg0, arg1, out), weight);
    }

    /// Push a +1 union edge (seed union from `@uf`).
    pub fn push_union(&self, a: Id, b: Id) {
        self.union_in.push(Tup2(a, b), 1);
    }

    /// Run the rebuild to fixpoint via the §2.4 congruence shuttle: step once,
    /// read the congruence output, push any not-yet-applied congruence edge into
    /// `union_in`, and step again — until no new congruence edge appears. Each
    /// step is O(delta). `already_pushed` tracks edges (normalized lo<=hi)
    /// pushed across the *lifetime* of the circuit so each is shuttled once.
    pub fn run_to_fixpoint(
        &self,
        already_pushed: &mut std::collections::BTreeSet<(Id, Id)>,
    ) -> Result<()> {
        self.handle.transaction()?;
        let mut steps = 1usize;
        loop {
            let mut new_edges: Vec<(Id, Id)> = Vec::new();
            for (Tup2(a, b), (), w) in self.congru_out.consolidate().iter() {
                if w > 0 {
                    let key = (a.min(b), a.max(b));
                    if already_pushed.insert(key) {
                        new_edges.push((a, b));
                    }
                }
            }
            if new_edges.is_empty() {
                break;
            }
            for (a, b) in new_edges {
                self.union_in.push(Tup2(a, b), 1);
            }
            self.handle.transaction()?;
            steps += 1;
            if steps > 100_000 {
                anyhow::bail!("congruence shuttle did not converge");
            }
        }
        Ok(())
    }

    /// Read the canonical view rows as `(table_tag, canon_arg0, canon_arg1,
    /// canon_out)` with positive weight.
    pub fn read_canon(&self) -> Vec<CanonRow> {
        let mut out = Vec::new();
        for (Tup4(tag, a, b, c), (), w) in self.canon_out.consolidate().iter() {
            let w: ZWeight = w;
            if w > 0 {
                out.push((tag, a, b, c));
            }
        }
        out
    }

    /// Read leader(x) for every known x.
    pub fn read_leaders(&self) -> std::collections::BTreeMap<Id, Id> {
        let mut m = std::collections::BTreeMap::new();
        for (Tup2(x, l), (), w) in self.leader_out.consolidate().iter() {
            let w: ZWeight = w;
            if w > 0 {
                m.insert(x, l);
            }
        }
        m
    }
}
