//! In-process, build-once, epoch-driven incremental body join on RAW
//! `differential-dataflow` + `timely` — the FlowLog analog of the Feldera
//! backend's `dbsp_join::PersistentJoin`.
//!
//! This is the ONLY join path for the FlowLog `Interpret` backend (driven by
//! [`crate::interpret::run_iteration`]); there is no host nested-loop fallback.
//! It panics (via the caller) on shapes it does not support — see `plan_join`.
//!
//! ## The architecture proven here (mirrors feldera/DBSP)
//!
//! For each atom-bearing rule we build ONE differential-dataflow dataflow, ONCE,
//! inside a single-threaded timely `Worker` we OWN (so we can `step` it across
//! host calls). Each body atom occurrence is sourced from a
//! `differential_dataflow::input::InputSession`; the rule's body join is a
//! left-deep chain of DD `.join`s, with `!=` guards and value-prims inlined in
//! `.flat_map`/`.filter`; the head binding rows flow out through `.inspect_batch`
//! into a shared `Rc<RefCell<Vec<(Row, isize)>>>` capture buffer.
//!
//! Each egglog iteration (= one epoch) the host feeds ONLY the per-relation
//! signed DELTA into the InputSessions (`+1` insert, `-1` retract), advances the
//! timely timestamp, `step_while`s the worker to that epoch's fixpoint, and
//! drains the capture buffer to get the OUTPUT binding deltas. The
//! InputSessions are NEVER cleared — the DD arrangements persist across epochs,
//! which is what makes the join genuinely incremental (epoch K does only
//! delta·integral work, not a full recompute) — the whole point of the design.
//!
//! ## Fixpoint structure
//!
//! We use EXTERNAL epoch-drive (the host loop advances epochs and feeds head
//! outputs back as the next epoch's inputs), NOT an in-dataflow `iterate()`
//! scope. This matches egglog's bounded `(run N)` fire->rebuild->repeat model and
//! sidesteps DD `iterate()`'s monotonicity constraints under retraction (a
//! rebuild RETRACTS non-canonical rows, which `iterate()` cannot express
//! cleanly). The dataflow itself is NON-recursive: one epoch = one bounded hop.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Result;
use differential_dataflow::input::InputSession;
use egglog_backend_trait::FunctionId;
use hashbrown::HashMap;
use timely::communication::allocator::thread::Thread;
use timely::communication::allocator::Allocator;
use timely::dataflow::operators::probe::Handle as ProbeHandle;
use timely::dataflow::operators::{Inspect, Probe};
use timely::worker::Worker;
use timely::WorkerConfig;

use crate::compile::{BodyOp, RuleIr, Slot};

/// Fixed binding-row width (DD `Data` needs a `Sized + Ord + Hash` type; an
/// array gives us that without dbsp's `declare_tuples!`). Mirrors feldera's
/// `JOIN_WIDTH = 32` but bumped to 48 to cover the widest rebuild rule the
/// flowlog test corpus generates: `luminal-llama`'s `@rebuild_rule34` uses 35
/// distinct body vars (a wide-arity congruence-closure rebuild). 48 covers
/// every reachable program with headroom; a rule exceeding this is reported as
/// a row-width-cap wall (raise `W` to extend coverage — it is purely a fixed
/// array size, costing `W * 4` bytes per binding row).
pub const W: usize = 48;

/// A fixed-width binding / relation row flowing through the DD dataflow:
/// `row[i]` is the value of canonical body variable `i` (0 if not yet bound).
///
/// A NEWTYPE over `[u32; W]` (rather than the bare array) because timely's
/// `ExchangeData` bound — required by DD `.join`/`.distinct` — is
/// `Serialize + Deserialize`, and `serde` only derives those for arrays up to
/// length 32. The hand-written serde impl (serialize as a fixed-length seq of
/// `W` `u32`s) lifts that cap so `W` can exceed 32 (the corpus needs 35). All
/// other derives (`Ord`/`Hash`/`Clone`/`Copy`) are auto for any array size.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Row([u32; W]);

/// The join key carried between DD `.join` stages: the shared bound columns
/// packed into a fixed-width array (others 0). Same newtype as [`Row`].
type Key = Row;

impl std::ops::Index<usize> for Row {
    type Output = u32;
    #[inline]
    fn index(&self, i: usize) -> &u32 {
        &self.0[i]
    }
}

impl std::ops::IndexMut<usize> for Row {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut u32 {
        &mut self.0[i]
    }
}

impl serde::Serialize for Row {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeTuple;
        // Fixed-length tuple of W u32s — bincode-friendly, no length prefix
        // needed (the deserializer knows W). Sidesteps serde's 32-array cap.
        let mut t = s.serialize_tuple(W)?;
        for v in &self.0 {
            t.serialize_element(v)?;
        }
        t.end()
    }
}

impl<'de> serde::Deserialize<'de> for Row {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Row, D::Error> {
        struct RowVisitor;
        impl<'de> serde::de::Visitor<'de> for RowVisitor {
            type Value = Row;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "a tuple of {W} u32s")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Row, A::Error> {
                let mut a = [0u32; W];
                for (i, slot) in a.iter_mut().enumerate() {
                    *slot = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(Row(a))
            }
        }
        d.deserialize_tuple(W, RowVisitor)
    }
}

fn empty_row() -> Row {
    Row([0u32; W])
}

/// SPIKE evidence flag: `FLOWLOG_DD_NATIVE_TRACE=1` prints per-epoch input/output
/// delta sizes to stderr (proof of incrementality + retraction). Off by default.
fn trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOWLOG_DD_NATIVE_TRACE").is_some())
}

/// A planned DD join: canonical body-variable order + the table atoms.
pub struct JoinPlan {
    /// `var_order[i]` is the variable id at binding-row column `i`.
    var_order: Vec<u32>,
    /// var id -> binding-row column index.
    var_col: HashMap<u32, usize>,
    /// Body table atoms in emission order.
    atoms: Vec<PlanAtom>,
}

struct PlanAtom {
    func: FunctionId,
    slots: Vec<Slot>,
}

impl JoinPlan {
    pub fn var_order(&self) -> &[u32] {
        &self.var_order
    }
}

/// Build the join plan for `rule`, or `Err(reason)` if the DD dataflow cannot
/// support its shape (the caller PANICS — there is no host fallback). Supported:
/// one or more table atoms, at most [`W`] distinct body vars, atom arity at most
/// [`W`]. Body prims (`!=` guards, value prims like `+`) are re-run host-side
/// over the bindings by the caller (the table-join-on-engine /
/// prim-tail-host-side split), so we accept them by leaving them to the host tail.
pub fn plan_join(rule: &RuleIr) -> Result<JoinPlan, String> {
    let mut var_order: Vec<u32> = Vec::new();
    let mut var_col: HashMap<u32, usize> = HashMap::new();
    let mut atoms: Vec<PlanAtom> = Vec::new();

    let see = |v: u32, var_order: &mut Vec<u32>, var_col: &mut HashMap<u32, usize>| {
        if !var_col.contains_key(&v) {
            var_col.insert(v, var_order.len());
            var_order.push(v);
        }
    };

    for op in &rule.body {
        match op {
            BodyOp::Atom(atom) => {
                if atom.slots.len() > W {
                    return Err(format!("atom arity {} > W {}", atom.slots.len(), W));
                }
                for s in &atom.slots {
                    if let Slot::Var(v) = s {
                        see(*v, &mut var_order, &mut var_col);
                    }
                }
                atoms.push(PlanAtom {
                    func: atom.func,
                    slots: atom.slots.clone(),
                });
            }
            // Body prims (e.g. `!=` guards, value prims like `+`) are re-run
            // host-side over the join bindings by the caller (the table-join-on-
            // engine / prim-tail-host-side split). They do not affect join
            // planning; a value prim may bind a fresh var the head reads.
            BodyOp::Prim { .. } => {}
        }
    }

    if atoms.is_empty() {
        return Err("no body table atoms (atom-less rule)".to_string());
    }
    if var_order.len() > W {
        return Err(format!("too many body vars {} > W {}", var_order.len(), W));
    }

    Ok(JoinPlan {
        var_order,
        var_col,
        atoms,
    })
}

/// The persistent, in-process DD body join for one rule. Built once; driven
/// across epochs by [`step`]. Owns its timely `Worker`, so it can be stepped
/// between host iterations without re-spawning threads.
pub struct PersistentDdJoin {
    worker: Worker,
    /// One input session per atom occurrence (in `plan.atoms` order).
    inputs: Vec<InputSession<u32, Row, isize>>,
    /// Probe on the output, to `step_while` until the epoch is fully processed.
    probe: ProbeHandle<u32>,
    /// Shared capture buffer: the output binding deltas of the most-recent
    /// epoch, drained by [`step`]. `inspect_batch` appends `(row, weight)`.
    captured: Rc<RefCell<Vec<(Row, isize)>>>,
    /// `func` -> atom-occurrence indices reading it (self-join fan-out).
    occ_of_func: HashMap<FunctionId, Vec<usize>>,
    /// Number of canonical body variables (binding-row width in use).
    n_vars: usize,
    /// Current epoch (monotonic; advanced once per [`step`]).
    epoch: u32,
}

impl PersistentDdJoin {
    /// Build the persistent DD dataflow for `plan` ONCE.
    pub fn build(plan: &JoinPlan) -> Result<PersistentDdJoin> {
        let alloc = Allocator::Thread(Thread::default());
        let mut worker = Worker::new(
            WorkerConfig::default(),
            alloc,
            Some(std::time::Instant::now()),
        );

        let n_atoms = plan.atoms.len();
        let atom_slots: Vec<Vec<Slot>> = plan.atoms.iter().map(|a| a.slots.clone()).collect();
        let var_col = plan.var_col.clone();
        let n_vars = plan.var_order.len();
        let captured: Rc<RefCell<Vec<(Row, isize)>>> = Rc::new(RefCell::new(Vec::new()));
        let captured_in = Rc::clone(&captured);
        let probe: ProbeHandle<u32> = ProbeHandle::new();
        let probe_in = probe.clone();

        let inputs = worker.dataflow::<u32, _, _>(|scope| {
            let mut inputs: Vec<InputSession<u32, Row, isize>> = Vec::with_capacity(n_atoms);
            let mut collections = Vec::with_capacity(n_atoms);
            for _ in 0..n_atoms {
                let mut session: InputSession<u32, Row, isize> = InputSession::new();
                let coll = session.to_collection(scope);
                inputs.push(session);
                collections.push(coll);
            }

            // Left-deep join. Start from atom 0's bindings (map each relation row
            // into a canonical binding row, dropping rows whose const / repeated-
            // var constraints fail). Then join successive atoms on shared bound
            // columns. `bound[c]` tracks which canonical columns are filled.
            let mut bound = vec![false; n_vars];

            let slots0 = atom_slots[0].clone();
            let vc0 = var_col.clone();
            let mut cur = collections[0]
                .clone()
                .flat_map(move |r: Row| bind_atom(&r, &slots0, &vc0));
            mark_bound(&atom_slots[0], &var_col, &mut bound);

            for i in 1..n_atoms {
                let slots = atom_slots[i].clone();
                // Shared variables = atom vars already bound by a previous atom.
                let shared: Vec<u32> = atom_vars(&slots)
                    .into_iter()
                    .filter(|v| var_col.get(v).map(|&c| bound[c]).unwrap_or(false))
                    .collect();

                // Left side: key the current bindings by the shared canonical
                // columns. Right side: key the atom rows by the matching slot
                // positions. `.join` then yields `(key, (bind, relrow))` pairs.
                let shared_cols_left: Vec<usize> = shared.iter().map(|v| var_col[v]).collect();
                let shared_atom_cols: Vec<usize> = shared
                    .iter()
                    .map(|v| {
                        slots
                            .iter()
                            .position(|s| matches!(s, Slot::Var(x) if x == v))
                            .expect("shared var present in atom")
                    })
                    .collect();

                let scl = shared_cols_left.clone();
                let left = cur.map(move |b: Row| (pack_key(&b, &scl), b));
                let sac = shared_atom_cols.clone();
                let right = collections[i]
                    .clone()
                    .map(move |r: Row| (pack_key(&r, &sac), r));

                let slotsc = slots.clone();
                let vc = var_col.clone();
                let bound_now = bound.clone();
                cur = left
                    .join(right)
                    .flat_map(move |(_k, (b, r)): (Key, (Row, Row))| {
                        merge_atom_into(&b, &r, &slotsc, &vc, &bound_now)
                    });

                mark_bound(&slots, &var_col, &mut bound);
            }

            // `.distinct()` makes the binding set set-semantic (weights ±1),
            // matching egglog relations. `.consolidate()` collapses any
            // multiplicities before capture.
            let out = cur
                .map(|b: Row| (b, ()))
                .distinct()
                .map(|(b, ())| b)
                .consolidate();

            // Capture the per-epoch output delta into the shared buffer via the
            // raw timely stream's batch inspector (DD `inspect_batch` gives us
            // `&[(Row, time, isize)]`). We DON'T integrate — each epoch we read
            // exactly the delta produced at that epoch's time.
            let cap = Rc::clone(&captured_in);
            out.inner
                .inspect_batch(move |_t, batch| {
                    let mut buf = cap.borrow_mut();
                    for (row, _time, w) in batch.iter() {
                        buf.push((*row, *w));
                    }
                })
                .probe_with(&probe_in);

            inputs
        });

        let mut occ_of_func: HashMap<FunctionId, Vec<usize>> = HashMap::new();
        for (ord, a) in plan.atoms.iter().enumerate() {
            occ_of_func.entry(a.func).or_default().push(ord);
        }

        Ok(PersistentDdJoin {
            worker,
            inputs,
            probe,
            captured,
            occ_of_func,
            n_vars,
            epoch: 0,
        })
    }

    /// Feed one epoch of signed relation deltas, advance the timestamp, run the
    /// worker to this epoch's fixpoint, and return the resulting binding delta as
    /// `(binding_row_as_var_order_vec, weight)`.
    ///
    /// CRUCIAL: the InputSessions are NEVER cleared — only the delta is pushed,
    /// so the DD arrangements persist and the join is genuinely incremental.
    pub fn step(
        &mut self,
        deltas: &HashMap<FunctionId, Vec<(Vec<u32>, isize)>>,
    ) -> Result<Vec<(Vec<u32>, isize)>> {
        let mut pushed = false;
        for (func, rows) in deltas {
            if let Some(occs) = self.occ_of_func.get(func) {
                for &ord in occs {
                    for (row, w) in rows {
                        self.inputs[ord].update(pack_row(row), *w);
                        pushed = true;
                    }
                }
            }
        }
        // No input change for this rule's body ⇒ no new bindings: skip the epoch
        // entirely (still advance the logical clock so a later real delta is
        // ordered after). This short-circuits the many no-op rebuild re-runs.
        let next_epoch = self.epoch + 1;
        if !pushed {
            self.epoch = next_epoch;
            return Ok(Vec::new());
        }

        self.captured.borrow_mut().clear();

        // Advance every input to the next epoch and flush, then step the worker
        // until the probe frontier passes the epoch we just closed — i.e. all
        // output for this epoch has been produced.
        for inp in &mut self.inputs {
            inp.advance_to(next_epoch);
            inp.flush();
        }
        let probe = self.probe.clone();
        self.worker.step_while(|| probe.less_than(&next_epoch));
        self.epoch = next_epoch;

        // Drain the capture buffer; consolidate duplicate rows that may have been
        // emitted across multiple internal steps within the epoch.
        let mut acc: HashMap<Vec<u32>, isize> = HashMap::new();
        for (row, w) in self.captured.borrow_mut().drain(..) {
            let key: Vec<u32> = (0..self.n_vars).map(|i| row[i]).collect();
            *acc.entry(key).or_insert(0) += w;
        }
        let out: Vec<(Vec<u32>, isize)> = acc.into_iter().filter(|(_, w)| *w != 0).collect();
        // SPIKE evidence (gated `FLOWLOG_DD_NATIVE_TRACE`): per-epoch INPUT delta
        // size vs OUTPUT binding-delta size. Incrementality shows up as: output
        // tracks the input delta, NOT the (growing) integral — epoch K does only
        // delta·integral work, never a full recompute. The `pos`/`neg` split shows
        // retraction (negative-weight outputs) flowing through DD's signed weights.
        if trace_enabled() {
            let in_n: usize = deltas.values().map(|v| v.len()).sum();
            let pos = out.iter().filter(|(_, w)| *w > 0).count();
            let neg = out.iter().filter(|(_, w)| *w < 0).count();
            #[allow(clippy::disallowed_macros)]
            {
                eprintln!(
                    "[dd_native] epoch {} : input_delta_rows={} -> output_binding_delta pos={} neg={}",
                    next_epoch, in_n, pos, neg
                );
            }
        }
        Ok(out)
    }
}

/// Pack a slice of column values into a fixed-width row (0-padded).
fn pack_row(vals: &[u32]) -> Row {
    let mut a = empty_row();
    for (i, v) in vals.iter().enumerate() {
        a[i] = *v;
    }
    a
}

/// Build a join key from selected columns (packed into the low slots).
fn pack_key(r: &Row, cols: &[usize]) -> Key {
    let mut a = empty_row();
    for (i, &c) in cols.iter().enumerate() {
        a[i] = r[c];
    }
    a
}

/// Distinct variables appearing in an atom (column order).
fn atom_vars(slots: &[Slot]) -> Vec<u32> {
    let mut out = Vec::new();
    for s in slots {
        if let Slot::Var(v) = s {
            if !out.contains(v) {
                out.push(*v);
            }
        }
    }
    out
}

/// Mark the canonical columns of an atom's variables as bound.
fn mark_bound(slots: &[Slot], var_col: &HashMap<u32, usize>, bound: &mut [bool]) {
    for s in slots {
        if let Slot::Var(v) = s {
            if let Some(&c) = var_col.get(v) {
                bound[c] = true;
            }
        }
    }
}

/// Match the first atom's relation row against its slots, producing the initial
/// canonical binding row (or empty vec if a const / repeated-var constraint
/// fails). Returns a `Vec` for `flat_map`.
fn bind_atom(r: &Row, slots: &[Slot], var_col: &HashMap<u32, usize>) -> Vec<Row> {
    let mut out = empty_row();
    let mut local: HashMap<u32, u32> = HashMap::new();
    for (i, s) in slots.iter().enumerate() {
        let val = r[i];
        match s {
            Slot::Const(c) => {
                if *c != val {
                    return Vec::new();
                }
            }
            Slot::Var(v) => {
                if let Some(&prev) = local.get(v) {
                    if prev != val {
                        return Vec::new();
                    }
                } else {
                    local.insert(*v, val);
                    out[var_col[v]] = val;
                }
            }
        }
    }
    vec![out]
}

/// Merge atom row `r` into binding `b`: already-bound columns must agree;
/// previously-unbound atom vars are written. Empty vec on constraint failure.
fn merge_atom_into(
    b: &Row,
    r: &Row,
    slots: &[Slot],
    var_col: &HashMap<u32, usize>,
    bound: &[bool],
) -> Vec<Row> {
    let mut out = *b;
    let mut local: HashMap<u32, u32> = HashMap::new();
    for (i, s) in slots.iter().enumerate() {
        let val = r[i];
        match s {
            Slot::Const(c) => {
                if *c != val {
                    return Vec::new();
                }
            }
            Slot::Var(v) => {
                if let Some(&prev) = local.get(v) {
                    if prev != val {
                        return Vec::new();
                    }
                    continue;
                }
                local.insert(*v, val);
                let c = var_col[v];
                if bound[c] {
                    if out[c] != val {
                        return Vec::new();
                    }
                } else {
                    out[c] = val;
                }
            }
        }
    }
    vec![out]
}

#[cfg(test)]
mod tests {
    use super::*;
    use egglog_numeric_id::NumericId;

    /// A transitive-closure hop plan `R(x,y), R(y,z)` over one relation, built
    /// directly (no full RuleIr).
    fn tc_plan(func: FunctionId) -> JoinPlan {
        let mut var_col = HashMap::new();
        var_col.insert(0u32, 0usize);
        var_col.insert(1u32, 1usize);
        var_col.insert(2u32, 2usize);
        JoinPlan {
            var_order: vec![0, 1, 2],
            var_col,
            atoms: vec![
                PlanAtom {
                    func,
                    slots: vec![Slot::Var(0), Slot::Var(1)],
                },
                PlanAtom {
                    func,
                    slots: vec![Slot::Var(1), Slot::Var(2)],
                },
            ],
        }
    }

    fn delta(
        func: FunctionId,
        rows: &[(&[u32], isize)],
    ) -> HashMap<FunctionId, Vec<(Vec<u32>, isize)>> {
        let mut m = HashMap::new();
        m.insert(func, rows.iter().map(|(r, w)| (r.to_vec(), *w)).collect());
        m
    }

    /// CRUX #1 + #2: build-once + drive-epochs feeding ONLY deltas, and the join
    /// stays incremental across epochs (epoch 2 fed only the new edge emits ONLY
    /// the new hop, not a re-derivation).
    #[test]
    fn dd_native_join_is_incremental() {
        let f = FunctionId::new(0);
        let plan = tc_plan(f);
        let mut pj = PersistentDdJoin::build(&plan).expect("build");

        // Epoch 1: seed edges (1,2),(2,3). Only hop is (1,2,3).
        let out1 = pj
            .step(&delta(f, &[(&[1, 2], 1), (&[2, 3], 1)]))
            .expect("step1");
        assert_eq!(out1, vec![(vec![1, 2, 3], 1)], "first hop");

        // Epoch 2: add ONLY new edge (3,4). Incremental join must emit ONLY the
        // new binding (2,3,4) — NOT re-derive (1,2,3).
        let out2 = pj.step(&delta(f, &[(&[3, 4], 1)])).expect("step2");
        assert_eq!(out2, vec![(vec![2, 3, 4], 1)], "only the new hop");

        // CRUX #3: retract edge (2,3). Two bindings used it; both retract
        // (negative weight) — bit-exact retraction via DD signed weights.
        let mut out3 = pj.step(&delta(f, &[(&[2, 3], -1)])).expect("step3");
        out3.sort();
        assert_eq!(
            out3,
            vec![(vec![1, 2, 3], -1), (vec![2, 3, 4], -1)],
            "retraction propagates"
        );
    }
}
