//! In-process, build-once, epoch-driven incremental body join on RAW
//! `differential-dataflow` + `timely` — the FlowLog analog of the Feldera
//! backend's `dbsp_join::PersistentJoin`.
//!
//! This is the ONLY join path for the FlowLog `Interpret` backend (driven by
//! [`crate::interpret::run_iteration`]); there is no host nested-loop fallback.
//! It panics (via the caller) on shapes it does not support — see `plan_join`.
//!
//! ## Two implementations: per-rule vs FUSED per-ruleset
//!
//! [`FusedDdJoin`] is the PERF-CRITICAL path the interpreter drives: ONE shared
//! timely `Worker` hosts ONE dataflow for a whole RULESET, every distinct body
//! relation is a single SHARED input `Collection`, and each rule is a join
//! sub-stream reading those shared collections (the DD analog of feldera's
//! `FusedJoin`). [`PersistentDdJoin`] is the original per-RULE worker; it is
//! retained for the bridge-level incrementality unit test and documents the base
//! architecture below. Both feed only signed DELTAS into never-cleared
//! InputSessions, so the arrangements persist across epochs = incremental.
//!
//! ## The base architecture (mirrors feldera/DBSP)
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
// `--wcoj` triangle worst-case-optimal join: dogsdogsdogs prefix-extension /
// AltNeu delta-query operators (vendored + patched `differential-dogs3`).
use differential_dataflow::VecCollection;
use differential_dogs3::altneu::AltNeu;
use differential_dogs3::{CollectionIndex, ProposeExtensionMethod};
use egglog_backend_trait::FunctionId;
use hashbrown::{HashMap, HashSet};
use timely::communication::allocator::thread::Thread;
use timely::communication::allocator::Allocator;
use timely::dataflow::operators::probe::Handle as ProbeHandle;
use timely::dataflow::operators::{Inspect, Probe};
use timely::dataflow::Scope;
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

impl Default for Row {
    #[inline]
    fn default() -> Self {
        empty_row()
    }
}

/// SPIKE evidence flag: `FLOWLOG_DD_NATIVE_TRACE=1` prints per-epoch input/output
/// delta sizes to stderr (proof of incrementality + retraction). Off by default.
fn trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOWLOG_DD_NATIVE_TRACE").is_some())
}

/// PROTOTYPE (gated `FLOWLOG_DELTA_REBUILD`, needs the canon flag so the UF funcs
/// are tagged transient): drive the rebuild join from δUF only. Two sub-steps:
/// A feeds δview and KEEPS only non-rebuild rules (congruence), DISCARDING the
/// rebuild's `δview ⋈ uf` output; B feeds δUF and keeps the rebuild's
/// `view ⋈ δuf` output. So the δview-driven derivative of the rebuild is never
/// emitted — sound iff new view rows are canonical (the eclass fix), which makes
/// that term empty. Unlike the old transient path it does NOT zero the UF
/// arrangement; it just doesn't let δview drive rebuild output.
pub(crate) fn delta_rebuild_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOWLOG_DELTA_REBUILD").is_some())
}

// ---------------------------------------------------------------------------
// Step-0 profiling counters (gated FLOWLOG_DD_PROF). Confirm/refute the
// per-rule-worker duplication hypothesis BEFORE refactoring: how many timely
// `Worker`s get spun up, how many `InputSession`s total, and where wall-time
// goes (worker.step vs the host-side prim re-run). Read+printed by `dd_prof_dump`.
// ---------------------------------------------------------------------------
use std::sync::atomic::{AtomicU64, Ordering};
/// Number of timely `Worker`s created (one per `PersistentDdJoin::build`).
pub(crate) static PROF_WORKERS: AtomicU64 = AtomicU64::new(0);
/// Total `InputSession`s created across all workers (sum of atom occurrences).
pub(crate) static PROF_INPUT_SESSIONS: AtomicU64 = AtomicU64::new(0);
/// Total time spent in `worker.step_while` (the DD epoch fixpoint loop).
pub(crate) static PROF_STEP_NS: AtomicU64 = AtomicU64::new(0);
/// Total time spent feeding deltas into InputSessions + advancing/flushing.
pub(crate) static PROF_FEED_NS: AtomicU64 = AtomicU64::new(0);
/// Number of `step` calls that actually clocked the worker (pushed a delta).
pub(crate) static PROF_STEP_CALLS: AtomicU64 = AtomicU64::new(0);
/// Total time spent re-running body primitives host-side over the bindings.
pub(crate) static PROF_PRIM_NS: AtomicU64 = AtomicU64::new(0);
/// Total time spent computing the per-rule signed body-relation delta.
pub(crate) static PROF_DELTA_NS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn prof_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    // The low-level feed/step/call counters update whenever EITHER the global
    // profile or the per-ruleset profile is requested — the per-ruleset path
    // reads their before/after deltas around each `step` to attribute
    // worker_step / feed time to the ruleset being run.
    *ON.get_or_init(|| {
        std::env::var_os("FLOWLOG_DD_PROF").is_some()
            || std::env::var_os("FLOWLOG_DD_RULESET_PROF").is_some()
    })
}

/// Per-ruleset profiling (gated `FLOWLOG_DD_RULESET_PROF`): attribute DD wall
/// time to the NAME of the ruleset being run, split into the same buckets as
/// the global profile, plus a call count and the summed input-delta row count.
pub(crate) fn ruleset_prof_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOWLOG_DD_RULESET_PROF").is_some())
}

#[inline]
fn add_ns(c: &AtomicU64, d: std::time::Duration) {
    c.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
}

/// One ruleset's accumulated DD profile (nanoseconds + counts).
#[derive(Default, Clone)]
pub(crate) struct RulesetProf {
    pub calls: u64,
    pub total_ns: u64,
    pub worker_step_ns: u64,
    pub feed_ns: u64,
    pub host_prim_ns: u64,
    pub delta_compute_ns: u64,
    pub delta_rows: u64,
}

/// Accumulator keyed by ruleset NAME. Only touched when
/// `FLOWLOG_DD_RULESET_PROF` is set (zero overhead otherwise).
pub(crate) fn ruleset_prof_table() -> &'static std::sync::Mutex<HashMap<String, RulesetProf>> {
    use std::sync::OnceLock;
    static TABLE: OnceLock<std::sync::Mutex<HashMap<String, RulesetProf>>> = OnceLock::new();
    TABLE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Record one ruleset's DD work for a single `run_rules` call. No-op unless
/// `FLOWLOG_DD_RULESET_PROF` is set.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ruleset_prof_record(
    ruleset: &str,
    total_ns: u64,
    worker_step_ns: u64,
    feed_ns: u64,
    host_prim_ns: u64,
    delta_compute_ns: u64,
    delta_rows: u64,
) {
    if !ruleset_prof_enabled() {
        return;
    }
    let mut table = ruleset_prof_table().lock().expect("ruleset prof lock");
    let e = table.entry(ruleset.to_string()).or_default();
    e.calls += 1;
    e.total_ns += total_ns;
    e.worker_step_ns += worker_step_ns;
    e.feed_ns += feed_ns;
    e.host_prim_ns += host_prim_ns;
    e.delta_compute_ns += delta_compute_ns;
    e.delta_rows += delta_rows;
}

/// Print the Step-0 profile to stderr if `FLOWLOG_DD_PROF` is set.
pub fn dd_prof_dump() {
    if !prof_enabled() {
        return;
    }
    let workers = PROF_WORKERS.load(Ordering::Relaxed);
    let sessions = PROF_INPUT_SESSIONS.load(Ordering::Relaxed);
    let step_ns = PROF_STEP_NS.load(Ordering::Relaxed);
    let feed_ns = PROF_FEED_NS.load(Ordering::Relaxed);
    let calls = PROF_STEP_CALLS.load(Ordering::Relaxed);
    let prim_ns = PROF_PRIM_NS.load(Ordering::Relaxed);
    let delta_ns = PROF_DELTA_NS.load(Ordering::Relaxed);
    #[allow(clippy::disallowed_macros)]
    {
        eprintln!(
            "[FLOWLOG_DD_PROF] workers={workers} input_sessions={sessions} \
             nonempty_step_calls={calls} worker_step={:.3}s feed={:.3}s \
             host_prim={:.3}s delta_compute={:.3}s",
            step_ns as f64 / 1e9,
            feed_ns as f64 / 1e9,
            prim_ns as f64 / 1e9,
            delta_ns as f64 / 1e9,
        );
    }
}

/// Print the per-ruleset DD profile to stderr if `FLOWLOG_DD_RULESET_PROF` is
/// set: one row per ruleset, sorted by total DD time descending, with each
/// ruleset's share of the grand total.
pub fn dd_ruleset_prof_dump() {
    if !ruleset_prof_enabled() {
        return;
    }
    let table = ruleset_prof_table().lock().expect("ruleset prof lock");
    if table.is_empty() {
        return;
    }
    let mut rows: Vec<(String, RulesetProf)> =
        table.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    rows.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));
    let grand_total: u64 = rows.iter().map(|(_, p)| p.total_ns).sum();
    let s = |ns: u64| ns as f64 / 1e9;
    #[allow(clippy::disallowed_macros)]
    {
        eprintln!(
            "[FLOWLOG_DD_RULESET_PROF] grand_total_dd={:.3}s",
            s(grand_total)
        );
        eprintln!(
            "{:<28} {:>6} {:>9} {:>6} {:>11} {:>9} {:>9} {:>11}",
            "ruleset", "calls", "total", "%dd", "worker_step", "feed", "host_prim", "delta_rows"
        );
        for (name, p) in &rows {
            let pct = if grand_total > 0 {
                100.0 * p.total_ns as f64 / grand_total as f64
            } else {
                0.0
            };
            eprintln!(
                "{:<28} {:>6} {:>8.3}s {:>5.1}% {:>10.3}s {:>8.3}s {:>8.3}s {:>11}",
                name,
                p.calls,
                s(p.total_ns),
                pct,
                s(p.worker_step_ns),
                s(p.feed_ns),
                s(p.host_prim_ns),
                p.delta_rows,
            );
        }
        // Cross-check: share of fused worker_step in rules whose body reads @uf.
        let uf_step = UF_BODY_STEP_NS.load(Ordering::Relaxed);
        let all_step = ALL_STEP_NS.load(Ordering::Relaxed);
        let uf_pct = if all_step > 0 {
            100.0 * uf_step as f64 / all_step as f64
        } else {
            0.0
        };
        eprintln!(
            "[FLOWLOG_DD_RULESET_PROF] uf_body_worker_step={:.3}s of {:.3}s total worker_step \
             = {:.1}% (rules whose body reads a UF_* table)",
            s(uf_step),
            s(all_step),
            uf_pct,
        );
    }
}

/// Cross-check accumulator for the per-ruleset profiler: nanoseconds of
/// (apportioned) fused worker_step attributable to rules whose BODY reads a
/// `@uf` table (`UF_*`), and the grand total worker_step nanos. Only touched
/// when `FLOWLOG_DD_RULESET_PROF` is set.
static UF_BODY_STEP_NS: AtomicU64 = AtomicU64::new(0);
static ALL_STEP_NS: AtomicU64 = AtomicU64::new(0);

/// Record one `run_rules` call's worker_step split into UF-body-reading vs all.
pub(crate) fn ruleset_uf_body_record(uf_body_step_ns: u64, all_step_ns: u64) {
    if !ruleset_prof_enabled() {
        return;
    }
    UF_BODY_STEP_NS.fetch_add(uf_body_step_ns, Ordering::Relaxed);
    ALL_STEP_NS.fetch_add(all_step_ns, Ordering::Relaxed);
}

/// A planned DD join: canonical body-variable order + the table atoms.
pub struct JoinPlan {
    /// `var_order[i]` is the variable id at binding-row column `i`.
    var_order: Vec<u32>,
    /// var id -> binding-row column index.
    var_col: HashMap<u32, usize>,
    /// Body table atoms in emission order.
    atoms: Vec<PlanAtom>,
    /// `Some` ⇒ row projection (`FLOWLOG_PROJECT`) found a column-reusing layout
    /// that fits `W` though the static var count does not; the binary chain uses
    /// the per-step columns instead of `var_col`. `None` ⇒ the historical static
    /// layout (every distinct var gets a permanent column).
    projection: Option<ProjectionPlan>,
}

struct PlanAtom {
    func: FunctionId,
    slots: Vec<Slot>,
}

impl JoinPlan {
    /// The variable id at each CAPTURED binding-row column, in column order. With
    /// projection this is the reduced surviving-var set (the build packs them into
    /// columns `0..head_vars.len()`); without, it is the full static var order.
    /// Either way the head scatter reads captured-row column `i` for var
    /// `var_order()[i]`, so it stays correct.
    pub fn var_order(&self) -> Vec<u32> {
        match &self.projection {
            Some(p) => p.head_vars.clone(),
            None => self.var_order.clone(),
        }
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
    // ROW PROJECTION (gated `FLOWLOG_PROJECT`, default-off): when the static
    // layout exceeds `W`, attempt a per-step register allocation that reuses
    // binding-row columns for variables whose liveness intervals do not overlap
    // (a body subterm var dies once the atom that consumes it has joined). If the
    // reused-column frontier still fits `W`, the binary chain can run the rule on
    // the existing fixed-width `Row` — this is what lets the giant Herbie seed
    // rules (962 vars, frontier ~22) lower. Off ⇒ the historical bail below.
    if project_enabled() {
        if let Some(proj) = build_projection(&atoms, &rule.head, &rule.body) {
            return Ok(JoinPlan {
                var_order,
                var_col,
                atoms,
                projection: Some(proj),
            });
        }
        // Projection could not fit `W` (frontier too wide) ⇒ fall through to the
        // historical reject.
    }

    if var_order.len() > W {
        return Err(format!("too many body vars {} > W {}", var_order.len(), W));
    }

    Ok(JoinPlan {
        var_order,
        var_col,
        atoms,
        projection: None,
    })
}

/// Is row projection enabled? (env `FLOWLOG_PROJECT`, default-off.) Flip the
/// default by changing this one expression.
pub(crate) fn project_enabled() -> bool {
    std::env::var_os("FLOWLOG_PROJECT").is_some()
}

/// A per-step binding-row column layout produced by [`build_projection`]: column
/// reuse (linear-scan register allocation over body-atom liveness) that keeps the
/// frontier within [`W`] so a rule whose STATIC var count exceeds `W` can still
/// run on the fixed-width `Row`. See the call site in [`plan_join`].
#[derive(Clone, Debug)]
pub(crate) struct ProjectionPlan {
    /// `step_col[i]` maps each variable LIVE during step `i` (atom `i`'s join) to
    /// its binding-row column at that step. A var's column is stable from its
    /// birth step to its death step but may differ across non-overlapping vars.
    pub step_col: Vec<HashMap<u32, usize>>,
    /// The surviving (head/body-prim-relevant) variables and their FINAL columns,
    /// in a deterministic order. Drives the reduced head-scatter `var_order`.
    pub head_vars: Vec<u32>,
    /// Final column of each surviving var (parallel to `head_vars`).
    pub head_cols: Vec<usize>,
}

/// Build a column-reusing layout for the body atoms, or `None` if the reused
/// frontier still exceeds `W`. Liveness is first-use..last-use over the EMITTED
/// atom order, EXTENDED so any variable read by the head or a body prim stays
/// live to the end (it must survive into the captured row). Linear-scan slot
/// assignment: a column freed by a dead var is reused by a later var's birth.
fn build_projection(
    atoms: &[PlanAtom],
    head: &[crate::compile::HeadOp],
    body: &[BodyOp],
) -> Option<ProjectionPlan> {
    use hashbrown::HashSet;
    let n = atoms.len();

    // Variables the head / body prims read: these must survive to the end.
    let mut survivor: HashSet<u32> = HashSet::new();
    collect_head_vars(head, &mut survivor);
    for op in body {
        if let BodyOp::Prim { args, ret, .. } = op {
            for s in args {
                if let Slot::Var(v) = s {
                    survivor.insert(*v);
                }
            }
            if let Slot::Var(v) = ret {
                survivor.insert(*v);
            }
        }
    }

    // first[v]/last[v] = first/last atom index where v appears. Survivors get
    // last = n-1 (live to the final step) so they are never freed mid-chain.
    let mut first: HashMap<u32, usize> = HashMap::new();
    let mut last: HashMap<u32, usize> = HashMap::new();
    for (i, a) in atoms.iter().enumerate() {
        for s in &a.slots {
            if let Slot::Var(v) = s {
                first.entry(*v).or_insert(i);
                last.insert(*v, i);
            }
        }
    }
    for v in &survivor {
        if let Some(l) = last.get_mut(v) {
            *l = n - 1;
        }
    }

    // Vars born / dying at each step (in deterministic var-id order so the layout
    // is reproducible).
    let mut births: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut deaths: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut vars: Vec<u32> = first.keys().copied().collect();
    vars.sort_unstable();
    for v in vars {
        births[first[&v]].push(v);
        deaths[last[&v]].push(v);
    }

    // Linear-scan register allocation. `col_of` is the current column of each
    // live var; `free` is a stack of reusable columns (lowest first, so the
    // layout is deterministic); `next` is the high-water column allocator.
    let mut col_of: HashMap<u32, usize> = HashMap::new();
    let mut free: Vec<usize> = Vec::new();
    let mut next: usize = 0;
    let mut step_col: Vec<HashMap<u32, usize>> = Vec::with_capacity(n);
    let mut peak = 0usize;
    for i in 0..n {
        // Births first (so this atom's fresh vars get columns before we snapshot).
        for &v in &births[i] {
            let c = if let Some(c) = free.pop() {
                c
            } else {
                let c = next;
                next += 1;
                c
            };
            col_of.insert(v, c);
        }
        peak = peak.max(col_of.len());
        if peak > W {
            return None;
        }
        // Snapshot the layout AT this step (after births, before deaths) — every
        // var the atom touches plus all still-live carried vars are present.
        step_col.push(col_of.clone());
        // Deaths free their columns for reuse by later births.
        for &v in &deaths[i] {
            if let Some(c) = col_of.remove(&v) {
                free.push(c);
            }
            // freeing low-first keeps assignment deterministic
            free.sort_unstable_by(|a, b| b.cmp(a));
        }
    }

    // Surviving vars + their FINAL columns (the columns they hold at the last
    // step, where they are guaranteed live). Deterministic order by var id.
    let final_layout = &step_col[n - 1];
    let mut head_vars: Vec<u32> = survivor
        .into_iter()
        .filter(|v| final_layout.contains_key(v))
        .collect();
    head_vars.sort_unstable();
    let head_cols: Vec<usize> = head_vars.iter().map(|v| final_layout[v]).collect();

    Some(ProjectionPlan {
        step_col,
        head_vars,
        head_cols,
    })
}

/// Collect every variable a head op references into `out`.
fn collect_head_vars(head: &[crate::compile::HeadOp], out: &mut hashbrown::HashSet<u32>) {
    use crate::compile::HeadOp;
    let add = |s: &Slot, out: &mut hashbrown::HashSet<u32>| {
        if let Slot::Var(v) = s {
            out.insert(*v);
        }
    };
    for op in head {
        match op {
            HeadOp::Set { slots, .. }
            | HeadOp::Remove { slots, .. }
            | HeadOp::Subsume { slots, .. } => {
                for s in slots {
                    add(s, out);
                }
            }
            HeadOp::Lookup { args, ret, .. } => {
                for s in args {
                    add(s, out);
                }
                out.insert(*ret);
            }
            HeadOp::Call { args, ret, .. } => {
                for s in args {
                    add(s, out);
                }
                out.insert(*ret);
            }
            HeadOp::Union { l, r } => {
                add(l, out);
                add(r, out);
            }
            HeadOp::Panic(_) => {}
        }
    }
}

// ===========================================================================
// `--wcoj` triangle worst-case-optimal join
// ===========================================================================
//
// Detects the reverse-distributivity triangle rule
//   (rewrite (Add (Mul a b) (Mul a c)) (Mul a (Add b c)))
// which lowers (term encoding) to three arity-4 atoms
//   A0  Mul(a, b, m1, x1)     [a, b,  m1, x1]
//   A1  Mul(a, c, m2, x2)     [a, c,  m2, x2]
//   A2  Add(m1, m2, o, x3)    [m1, m2, o,  x3]
// where A0,A1 are the SAME relation (a self-join sharing col-0 `a`), A2 is a
// different relation joining A0.col2=m1 and A1.col2=m2, and every named
// variable is distinct. The cyclic core is the triangle on (a, m1, m2): the
// binary join Mul_a ⋈ Mul_b on `a` materializes a `Σ_a deg(a)²` intermediate
// the WCOJ collapses to the output.

/// The recognized triangle, with the BINDING-ROW column index of each of the 9
/// distinct variables (a,b,m1,x1,c,m2,x2,o,x3) plus the two relation ids. All
/// downstream `extend_using` key selectors and the final `Row` assembly index
/// the binding row by these columns, so the WCOJ emits the SAME 9-column
/// binding row as the binary join (bit-exact head firing).
#[derive(Clone, Debug)]
pub(crate) struct TriangleShape {
    mul_func: FunctionId,
    add_func: FunctionId,
    // binding-row columns (positions in the n_vars-wide Row)
    col_a: usize,
    col_b: usize,
    col_m1: usize,
    col_x1: usize,
    col_c: usize,
    col_m2: usize,
    col_x2: usize,
    col_o: usize,
    col_x3: usize,
}

/// Recognize the triangle shape in `plan`, or `None` if it does not match.
/// Structural (no name/func-id hardcoding): three arity-4 atoms, atoms 0 and 1
/// the same relation sharing exactly their col-0 var, atom 2 a different
/// relation whose col-0 = atom0.col2 and col-1 = atom1.col2, and all 9 named
/// variables distinct. Stage 1 recognizes exactly this 3-atom triangle;
/// generalization is Stage 2.
pub(crate) fn detect_triangle(plan: &JoinPlan) -> Option<TriangleShape> {
    if plan.atoms.len() != 3 {
        return None;
    }
    // Each atom must be arity 4 with all-distinct Var slots (no consts, no
    // repeated vars within an atom).
    let atom_var = |i: usize| -> Option<[u32; 4]> {
        let s = &plan.atoms[i].slots;
        if s.len() != 4 {
            return None;
        }
        let mut out = [0u32; 4];
        for (j, slot) in s.iter().enumerate() {
            match slot {
                Slot::Var(v) => out[j] = *v,
                Slot::Const(_) => return None,
            }
        }
        // distinct within the atom
        for j in 0..4 {
            for k in (j + 1)..4 {
                if out[j] == out[k] {
                    return None;
                }
            }
        }
        Some(out)
    };
    let a0 = atom_var(0)?;
    let a1 = atom_var(1)?;
    let a2 = atom_var(2)?;

    let mul_func = plan.atoms[0].func;
    // A0, A1 same relation; A2 a different relation.
    if plan.atoms[1].func != mul_func {
        return None;
    }
    let add_func = plan.atoms[2].func;
    if add_func == mul_func {
        return None;
    }

    // A0 = [a, b, m1, x1]; A1 = [a, c, m2, x2] — share col-0 (a), nothing else.
    let (a, b, m1, x1) = (a0[0], a0[1], a0[2], a0[3]);
    let (a_, c, m2, x2) = (a1[0], a1[1], a1[2], a1[3]);
    if a != a_ {
        return None;
    }
    // A2 = [m1, m2, o, x3] — col0 = A0.col2 (m1), col1 = A1.col2 (m2).
    let (m1_, m2_, o, x3) = (a2[0], a2[1], a2[2], a2[3]);
    if m1_ != m1 || m2_ != m2 {
        return None;
    }

    // All 9 named variables distinct (rules out degenerate self-overlap the
    // generic shape would not produce; the binary join would treat such a rule
    // differently, so we leave it to the binary path).
    let vars = [a, b, m1, x1, c, m2, x2, o, x3];
    for i in 0..vars.len() {
        for j in (i + 1)..vars.len() {
            if vars[i] == vars[j] {
                return None;
            }
        }
    }

    let col = |v: u32| -> usize { plan.var_col[&v] };
    Some(TriangleShape {
        mul_func,
        add_func,
        col_a: col(a),
        col_b: col(b),
        col_m1: col(m1),
        col_x1: col(x1),
        col_c: col(c),
        col_m2: col(m2),
        col_x2: col(x2),
        col_o: col(o),
        col_x3: col(x3),
    })
}

// ===========================================================================
// Stage 2: GENERAL incremental WCOJ for an arbitrary >=3-atom CYCLIC body
// ===========================================================================
//
// Generalizes the hardcoded triangle (above) to any >=3-atom rule whose body
// join is CYCLIC — the class where a binary-join chain materializes an
// intermediate exceeding the output (the AGM blowup WCOJ removes). Acyclic and
// <=2-atom rules keep the binary `.join` chain (hybrid, Free-Join style): WCOJ
// buys nothing there and adds per-prefix index/intersection overhead.
//
// ## Detector (`detect_cyclic_cq`)
//
// Structural, no name/func-id dependence:
//   1. >= 3 table atoms, all slots distinct `Var`s within each atom, no consts
//      (a const/repeated-var atom is a SELECTION the binary path handles; we
//      leave it on the binary chain to stay conservative + bit-exact).
//   2. The body hypergraph is CYCLIC by GYO reduction (repeatedly drop an "ear"
//      vertex appearing in <=1 remaining edge, and any edge subset of another;
//      acyclic iff it reduces to empty). Cyclic ⇒ the binary chain blows up.
//   3. A full WCOJ plan is constructible (see `build_cq_plan`): for EVERY
//      driver atom there is a variable order extending the driver one variable
//      at a time, each step binding a variable that is the LAST unbound variable
//      of some atom (so that atom keys entirely on already-bound columns). If
//      any driver cannot be covered this way, we BAIL (return None → binary
//      chain). This is the "only fire where provably bit-exact" stance: the
//      construction mirrors the Stage-1 triangle exactly (full k-stream delta
//      decomposition, ALL reads inside the AltNeu scope), and we only emit it
//      when every variable is bound by an in-scope extend/propose.
//
// ## Plan (`CqPlan`)
//
// `var_col` (carried) + a `CqDriver` per atom. Each driver records its initial
// bound columns (the driver atom's vars, written straight from the delta row)
// then a sequence of `CqStep`s, each extending the prefix with ONE variable
// `bind_col` via a set of `CqExtender`s (`atom_idx` + the prefix columns the
// atom keys on + the relation column it proposes + whether the atom reads ALT
// (atom index < driver) or NEU (atom index > driver)). >=2 extenders for a step
// ⇒ multiway `extend` (the WCOJ intersection); exactly 1 ⇒ `propose_using`.

/// One extender in a WCOJ extension step: read relation `atom_idx`'s `propose_col`
/// keyed by the prefix's `key_cols`, from the ALT (old) or NEU (new) trace.
#[derive(Clone, Debug)]
pub(crate) struct CqExtender {
    /// Index into `CqPlan.atom_funcs` / the rule's atom list.
    atom_idx: usize,
    /// Binding-row columns (already bound in the prefix) this atom keys on.
    /// Parallel to `key_atom_cols`.
    key_cols: Vec<usize>,
    /// The atom's relation columns (0-based slot positions) holding the key
    /// vars, parallel to `key_cols` — used to build the atom's lookup index so
    /// its key matches the prefix key_selector's columns.
    key_atom_cols: Vec<usize>,
    /// The relation column (0-based in the atom's row) that proposes the var.
    propose_col: usize,
    /// true ⇒ read the ALT (old) trace; false ⇒ the NEU (new) trace.
    alt: bool,
}

/// One JOIN-variable extension step: bind binding-row column `bind_col` from
/// `exts`. The intersected variable is a JOIN variable (appears in >=2 atoms).
/// With >=2 extenders this is the multiway `extend` (the WCOJ intersection that
/// collapses the intermediate); with a single extender it is a `propose_using`.
#[derive(Clone, Debug)]
pub(crate) struct CqStep {
    /// The binding-row column this step writes.
    bind_col: usize,
    exts: Vec<CqExtender>,
}

/// A per-atom PAYLOAD recovery: once an atom's JOIN columns are all bound,
/// recover its payload variables (those appearing only in this atom) in ONE
/// `propose_using` keyed on the atom's join columns. `payload_cols[k]` =
/// `(binding_col, relation_col)` for the k-th payload var.
#[derive(Clone, Debug)]
pub(crate) struct CqRecover {
    atom_idx: usize,
    /// Binding columns (already bound) the atom keys on. Parallel to
    /// `key_atom_cols`.
    key_cols: Vec<usize>,
    /// The atom's relation columns holding the key (join) vars.
    key_atom_cols: Vec<usize>,
    /// `(binding_col, relation_col)` per payload var to write from the proposal.
    payload: Vec<(usize, usize)>,
    /// true ⇒ read the ALT (old) trace; false ⇒ the NEU (new) trace.
    alt: bool,
}

/// One delta query `dQ/dA_driver`: seed the prefix from the driver atom's delta
/// row, bind every JOIN variable via `steps` (the WCOJ core), then recover each
/// non-driver atom's payload via `recovers`. ALL inside the AltNeu scope.
#[derive(Clone, Debug)]
pub(crate) struct CqDriver {
    /// The driver atom's index.
    driver_idx: usize,
    /// `(binding_col, relation_col)` for each of the driver atom's variables —
    /// written straight from the delta row into the initial prefix.
    seed: Vec<(usize, usize)>,
    steps: Vec<CqStep>,
    recovers: Vec<CqRecover>,
}

/// A general WCOJ plan for a cyclic >=3-atom body.
#[derive(Clone, Debug)]
pub(crate) struct CqPlan {
    /// Relation id per atom (atom order = the rule's `plan.atoms` order).
    atom_funcs: Vec<FunctionId>,
    /// One delta query per atom.
    drivers: Vec<CqDriver>,
}

/// The variables of an atom (its distinct `Var` slots), or `None` if the atom
/// has any const or repeated var (we leave those selections on the binary path).
fn atom_distinct_vars(slots: &[Slot]) -> Option<Vec<u32>> {
    let mut out: Vec<u32> = Vec::with_capacity(slots.len());
    for s in slots {
        match s {
            Slot::Var(v) => {
                if out.contains(v) {
                    return None;
                }
                out.push(*v);
            }
            Slot::Const(_) => return None,
        }
    }
    Some(out)
}

/// GYO acyclicity test on the body hypergraph (atoms = hyperedges, vars =
/// vertices). Returns `true` iff the query is CYCLIC (GYO does not reduce to
/// empty). `atom_vars[i]` is atom i's vertex set.
fn is_cyclic_cq(atom_vars: &[Vec<u32>]) -> bool {
    // Work on owned mutable vertex sets; `removed[i]` drops an absorbed edge.
    let mut edges: Vec<HashSet<u32>> = atom_vars
        .iter()
        .map(|vs| vs.iter().copied().collect())
        .collect();
    let mut removed = vec![false; edges.len()];

    loop {
        let mut progress = false;

        // (a) Ear removal: drop a vertex that appears in <= 1 live edge.
        let mut vert_count: HashMap<u32, usize> = HashMap::new();
        for (i, e) in edges.iter().enumerate() {
            if removed[i] {
                continue;
            }
            for &v in e {
                *vert_count.entry(v).or_insert(0) += 1;
            }
        }
        let ears: Vec<u32> = vert_count
            .iter()
            .filter(|&(_, &c)| c <= 1)
            .map(|(&v, _)| v)
            .collect();
        if !ears.is_empty() {
            for e in edges.iter_mut() {
                for v in &ears {
                    e.remove(v);
                }
            }
            progress = true;
        }

        // (b) Absorb an edge that is a subset of another live edge.
        let live: Vec<usize> = (0..edges.len()).filter(|&i| !removed[i]).collect();
        'outer: for &i in &live {
            if edges[i].is_empty() {
                removed[i] = true;
                progress = true;
                continue;
            }
            for &j in &live {
                if i == j || removed[j] {
                    continue;
                }
                if edges[i].is_subset(&edges[j]) {
                    removed[i] = true;
                    progress = true;
                    continue 'outer;
                }
            }
        }

        if !progress {
            break;
        }
    }

    // Cyclic iff any non-empty edge remains (GYO got stuck).
    edges
        .iter()
        .enumerate()
        .any(|(i, e)| !removed[i] && !e.is_empty())
}

/// Build the general WCOJ plan for `plan`, or `None` if it is not constructible
/// bit-exactly. Splits variables into JOIN vars (in >=2 atoms — the cyclic core)
/// and PAYLOAD vars (in exactly 1 atom — eclass/extra columns). The WCOJ binds
/// each JOIN var one at a time (each being the last-unbound JOIN var of >=1
/// atom, so that atom keys on already-bound join columns), then recovers each
/// atom's payload via one `propose`. If any driver's join vars cannot be ordered
/// this way, bail (`None` ⇒ binary chain).
fn build_cq_plan(plan: &JoinPlan, atom_vars: &[Vec<u32>]) -> Option<CqPlan> {
    let n_atoms = plan.atoms.len();
    let var_col = &plan.var_col;
    let atom_funcs: Vec<FunctionId> = plan.atoms.iter().map(|a| a.func).collect();

    // relation-col lookup: atom i, variable v -> the 0-based slot position.
    let atom_var_col = |i: usize, v: u32| -> Option<usize> {
        plan.atoms[i]
            .slots
            .iter()
            .position(|s| matches!(s, Slot::Var(x) if *x == v))
    };

    // JOIN vars = vars appearing in >= 2 atoms. PAYLOAD vars = in exactly 1.
    let mut var_atom_count: HashMap<u32, usize> = HashMap::new();
    for vs in atom_vars {
        for &v in vs {
            *var_atom_count.entry(v).or_insert(0) += 1;
        }
    }
    let is_join = |v: u32| -> bool { var_atom_count.get(&v).copied().unwrap_or(0) >= 2 };
    let join_vars_of = |i: usize| -> Vec<u32> {
        atom_vars[i]
            .iter()
            .copied()
            .filter(|&v| is_join(v))
            .collect()
    };
    let all_join_vars: HashSet<u32> = atom_vars
        .iter()
        .flatten()
        .copied()
        .filter(|&v| is_join(v))
        .collect();

    let mut drivers: Vec<CqDriver> = Vec::with_capacity(n_atoms);
    for driver_idx in 0..n_atoms {
        // Seed: the driver atom's own variables (join + payload), straight from
        // the delta row — they are all immediately bound.
        let mut bound: HashSet<u32> = atom_vars[driver_idx].iter().copied().collect();
        let seed: Vec<(usize, usize)> = atom_vars[driver_idx]
            .iter()
            .map(|&v| (var_col[&v], atom_var_col(driver_idx, v).unwrap()))
            .collect();

        // Bind JOIN vars one at a time. The next join var must be the LAST
        // unbound JOIN var of >= 1 atom (payload vars never block — they are
        // recovered after, not used as keys). Prefer the var constrained by the
        // MOST atoms (most-constrained-variable = the deepest WCOJ
        // intersection); ties broken by var id for determinism.
        let mut steps: Vec<CqStep> = Vec::new();
        while bound.iter().filter(|&&v| is_join(v)).count() < all_join_vars.len() {
            let mut best: Option<(u32, Vec<usize>)> = None;
            for &v in &all_join_vars {
                if bound.contains(&v) {
                    continue;
                }
                let mut ready_atoms: Vec<usize> = Vec::new();
                for (i, vs) in atom_vars.iter().enumerate() {
                    if !vs.contains(&v) {
                        continue;
                    }
                    // v is bindable from atom i iff every OTHER JOIN var of atom
                    // i is already bound.
                    if vs
                        .iter()
                        .all(|&w| w == v || !is_join(w) || bound.contains(&w))
                    {
                        ready_atoms.push(i);
                    }
                }
                if ready_atoms.is_empty() {
                    continue;
                }
                let better = match &best {
                    None => true,
                    Some((bv, ba)) => {
                        (ready_atoms.len(), std::cmp::Reverse(v))
                            > (ba.len(), std::cmp::Reverse(*bv))
                    }
                };
                if better {
                    best = Some((v, ready_atoms));
                }
            }

            let (v, ready_atoms) = best?; // no ready join var ⇒ not constructible.
            let exts: Vec<CqExtender> = ready_atoms
                .iter()
                .map(|&i| {
                    // Key on the atom's OTHER join vars (all bound by now).
                    let key_vars: Vec<u32> =
                        join_vars_of(i).into_iter().filter(|&w| w != v).collect();
                    CqExtender {
                        atom_idx: i,
                        key_cols: key_vars.iter().map(|w| var_col[w]).collect(),
                        key_atom_cols: key_vars
                            .iter()
                            .map(|&w| atom_var_col(i, w).unwrap())
                            .collect(),
                        propose_col: atom_var_col(i, v).unwrap(),
                        alt: i < driver_idx,
                    }
                })
                .collect();
            steps.push(CqStep {
                bind_col: var_col[&v],
                exts,
            });
            bound.insert(v);
        }

        // Recover each NON-driver atom's payload vars (those appearing only in
        // it), keyed on the atom's join columns (all bound). The driver atom's
        // payload is already in the seed. An atom with NO payload var still must
        // be VALIDATED so its tuple constrains the result; we emit a recover
        // with an empty payload (a pure existence `propose`). That is sound (no
        // over-count): empty payload ⟺ every column is a join var ⟺ the key
        // covers the whole row, and rows are set-semantic, so the key matches at
        // most one row (weight 1). In the term encoding every atom carries an
        // eclass + extra column, so this empty-payload case does not arise.
        let mut recovers: Vec<CqRecover> = Vec::new();
        for (i, avs) in atom_vars.iter().enumerate() {
            if i == driver_idx {
                continue;
            }
            let payload_vars: Vec<u32> = avs.iter().copied().filter(|&v| !is_join(v)).collect();
            let key_vars = join_vars_of(i);
            recovers.push(CqRecover {
                atom_idx: i,
                key_cols: key_vars.iter().map(|w| var_col[w]).collect(),
                key_atom_cols: key_vars
                    .iter()
                    .map(|&w| atom_var_col(i, w).unwrap())
                    .collect(),
                payload: payload_vars
                    .iter()
                    .map(|&v| (var_col[&v], atom_var_col(i, v).unwrap()))
                    .collect(),
                alt: i < driver_idx,
            });
        }

        drivers.push(CqDriver {
            driver_idx,
            seed,
            steps,
            recovers,
        });
    }

    Some(CqPlan {
        atom_funcs,
        drivers,
    })
}

/// Detect a general WCOJ-worthy body: a >=3-atom CYCLIC query for which a
/// bit-exact WCOJ plan is constructible. Returns the plan, or `None` (stay on
/// the binary chain). See the module note above for the criteria.
pub(crate) fn detect_cyclic_cq(plan: &JoinPlan, allow_acyclic: bool) -> Option<CqPlan> {
    let trace = std::env::var_os("FLOWLOG_WCOJ_TRACE").is_some();
    if plan.atoms.len() < 3 {
        return None;
    }
    // All atoms must be pure distinct-Var atoms (no const/repeated-var
    // selections — those stay on the binary path).
    let mut atom_vars: Vec<Vec<u32>> = Vec::with_capacity(plan.atoms.len());
    for a in &plan.atoms {
        match atom_distinct_vars(&a.slots) {
            Some(vs) => atom_vars.push(vs),
            None => {
                #[allow(clippy::disallowed_macros)]
                if trace {
                    eprintln!(
                        "[WCOJ] detect_cyclic_cq: bail (atom with const/repeated var) atoms={}",
                        plan.atoms.len()
                    );
                }
                return None;
            }
        }
    }
    // Only fire where the binary chain provably blows up: a CYCLIC join.
    let cyclic = is_cyclic_cq(&atom_vars);
    #[allow(clippy::disallowed_macros)]
    if trace {
        eprintln!(
            "[WCOJ] detect_cyclic_cq: atoms={} cyclic={cyclic} atom_vars={atom_vars:?}",
            plan.atoms.len()
        );
    }
    // Broadening WCOJ to acyclic >=3-atom rules is off by default (callers
    // pass `allow_acyclic: false`). Tests that verify acyclic correctness
    // pass `true` explicitly.
    if !cyclic && !allow_acyclic {
        return None;
    }
    let plan_out = build_cq_plan(plan, &atom_vars);
    #[allow(clippy::disallowed_macros)]
    if trace && plan_out.is_none() {
        eprintln!("[WCOJ] detect_cyclic_cq: cyclic but NOT constructible -> binary chain");
    }
    plan_out
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
        if prof_enabled() {
            PROF_WORKERS.fetch_add(1, Ordering::Relaxed);
            PROF_INPUT_SESSIONS.fetch_add(plan.atoms.len() as u64, Ordering::Relaxed);
        }

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
        let prof = prof_enabled();
        let t_feed = std::time::Instant::now();
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
        if prof {
            add_ns(&PROF_FEED_NS, t_feed.elapsed());
            PROF_STEP_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        let t_step = std::time::Instant::now();
        let probe = self.probe.clone();
        self.worker.step_while(|| probe.less_than(&next_epoch));
        if prof {
            add_ns(&PROF_STEP_NS, t_step.elapsed());
        }
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

// ===========================================================================
// FusedDdJoin — ONE shared worker + ONE dataflow per RULESET
// ===========================================================================
//
// `PersistentDdJoin` builds one timely `Worker` PER RULE: a program with R
// atom-bearing rules spins up R independent workers, and a body relation read by
// K rules gets K separate `InputSession`s (each fed the same delta) + K separate
// arrangements, each stepped to fixpoint separately. Step-0 profiling
// (`FLOWLOG_DD_PROF`) on math-microbenchmark N=9 showed 79 workers, 173
// InputSessions, and 1.996s of 2.43s total spent inside `worker.step_while` — the
// per-rule-worker duplication is the dominant cost.
//
// `FusedDdJoin` collapses this to ONE worker hosting ONE `worker.dataflow(...)`
// scope for the whole ruleset (keyed by the sorted live rule-index list, exactly
// like feldera's `FusedJoin`). Within that scope:
//   - every DISTINCT body relation across all rules gets ONE `InputSession` →
//     ONE base `Collection`, `.distinct()`'d ONCE and SHARED by every atom
//     occurrence (in every rule) that reads it. Cloning a DD collection is a
//     handle copy, so the shared `.distinct()` arrangement is built once, not K
//     times — the dedup win.
//   - each rule is a left-deep join sub-stream reading those shared collections,
//     with its OWN `inspect_batch` capture into a per-rule `Rc<RefCell<Vec>>`.
// Per epoch: feed each relation's delta ONCE into its shared input, advance +
// step the SINGLE worker once, then drain each rule's capture buffer. The
// host-side prim re-run + `apply_head` are unchanged (the caller does them).
//
// The NEVER-CLEAR / fed-only-deltas invariant is preserved (the InputSessions
// persist across epochs = genuinely incremental), as is the external epoch-drive.

/// A fused, delta-fed body join for a WHOLE ruleset on a single shared timely
/// `Worker`. Built once via [`FusedDdJoin::build`]; driven across epochs via
/// [`FusedDdJoin::step`] with a SINGLE `worker.step_while` per call.
pub struct FusedDdJoin {
    worker: Worker,
    /// One shared input session per DISTINCT body relation across all rules.
    inputs: HashMap<FunctionId, InputSession<u32, Row, isize>>,
    /// Single probe on all rule outputs (they share the dataflow scope, so one
    /// probe gates the whole epoch's fixpoint).
    probe: ProbeHandle<u32>,
    /// The fused rules, in build (= sorted rule-index) order. Each carries its
    /// own capture buffer + var width.
    rules: Vec<FusedRule>,
    /// Current epoch (monotonic; advanced once per [`step`]).
    epoch: u32,
    /// Phase-2 transient-UF rebuild (gated `EGGLOG_CANON_AT_CREATION`): the set
    /// of input relations whose integral is kept at ZERO across epochs by a
    /// feed/step/retract pulse, so a rebuild rule's `view ⋈ @UF_Sf` join computes
    /// ONLY `view_all ⋈ δuf`. Empty unless the flag is on and a rule in this
    /// ruleset reads such a relation; then `step` runs the symmetric path.
    transient_funcs: HashSet<FunctionId>,
    /// `--fast-rebuild` (relational): when set, the δUF-driven rebuild substep
    /// split is engaged (drop the empty `δview⋈uf_old` term). Set from the
    /// backend config flag OR the `FLOWLOG_DELTA_REBUILD` env var at build time.
    delta_rebuild: bool,
}

/// One rule's lowering inside a [`FusedDdJoin`]: its rule index (for routing
/// bindings to its head), its per-epoch output capture buffer, and its width.
struct FusedRule {
    idx: usize,
    /// This rule's per-epoch output binding-delta capture (`inspect_batch`
    /// appends `(row, weight)`; drained by [`FusedDdJoin::step`]).
    captured: Rc<RefCell<Vec<(Row, isize)>>>,
    /// Number of canonical body variables (binding-row width in use).
    n_vars: usize,
    /// `func` -> atom-occurrence indices reading it within THIS rule (self-join
    /// fan-out is handled at build time via the shared collection, so this is
    /// only used to know which relations this rule reads).
    body_funcs: Vec<FunctionId>,
    /// Phase-2: true iff this rule's body reads a TRANSIENT relation (a rebuild
    /// rule reading `@UF_Sf`); captured from the +δuf pulse sub-step.
    reads_transient: bool,
}

impl FusedDdJoin {
    /// Build ONE worker + ONE dataflow for the whole ruleset. `plans` pairs each
    /// rule's index with its [`JoinPlan`], in the order they should fire.
    /// `transient_funcs` (Phase-2): input relations [`step`] pulses to keep at a
    /// zero integral; empty => the original symmetric `step` (no behavior change).
    pub fn build(
        plans: &[(usize, JoinPlan)],
        transient_funcs: &HashSet<FunctionId>,
        delta_rebuild: bool,
        wcoj: bool,
        allow_acyclic: bool,
    ) -> Result<FusedDdJoin> {
        let alloc = Allocator::Thread(Thread::default());
        let mut worker = Worker::new(
            WorkerConfig::default(),
            alloc,
            Some(std::time::Instant::now()),
        );
        if prof_enabled() {
            PROF_WORKERS.fetch_add(1, Ordering::Relaxed);
        }

        // Distinct body relations across all rules → one shared input each.
        let mut funcs: Vec<FunctionId> = Vec::new();
        for (_, plan) in plans {
            for a in &plan.atoms {
                if !funcs.contains(&a.func) {
                    funcs.push(a.func);
                }
            }
        }
        if prof_enabled() {
            PROF_INPUT_SESSIONS.fetch_add(funcs.len() as u64, Ordering::Relaxed);
        }

        // Owned per-rule plan snapshots so the `move` dataflow closure is 'static.
        struct RulePlan {
            idx: usize,
            atoms: Vec<Vec<Slot>>,
            atom_funcs: Vec<FunctionId>,
            var_col: HashMap<u32, usize>,
            n_vars: usize,
            body_funcs: Vec<FunctionId>,
            /// `--wcoj` and this rule is the recognized triangle ⇒ build the WCOJ
            /// delta query instead of the binary `.join` chain. `None` ⇒ binary.
            triangle: Option<TriangleShape>,
            /// `--wcoj` and this rule is a general detected cyclic >=3-atom body
            /// (NOT already matched by `triangle`) ⇒ the general WCOJ delta
            /// query. `None` ⇒ triangle or binary.
            cq: Option<CqPlan>,
            /// `FLOWLOG_PROJECT`: per-step column-reuse layout ⇒ the binary chain
            /// runs with projecting merges and packs the captured row to the
            /// surviving vars. `None` ⇒ the historical static-column chain.
            projection: Option<ProjectionPlan>,
        }
        let rule_plans: Vec<RulePlan> = plans
            .iter()
            .map(|(idx, plan)| {
                let atom_funcs: Vec<FunctionId> = plan.atoms.iter().map(|a| a.func).collect();
                let mut body_funcs: Vec<FunctionId> = Vec::new();
                for &f in &atom_funcs {
                    if !body_funcs.contains(&f) {
                        body_funcs.push(f);
                    }
                }
                // Detect WCOJ shapes ONLY when `--wcoj` is set; off ⇒ both
                // `None`, so the build is byte-identical to the binary path. The
                // Stage-1 triangle is kept as its own hand-built path (the
                // regression guarantee); other detected cyclic >=3-atom bodies
                // route to the general construction. A rule matched by the
                // triangle never also takes the general path.
                let triangle = if wcoj { detect_triangle(plan) } else { None };
                let cq = if wcoj && triangle.is_none() {
                    detect_cyclic_cq(plan, allow_acyclic)
                } else {
                    None
                };
                // With projection the CAPTURED row is packed to the surviving
                // vars (columns `0..head_vars.len()`), so the capture/scatter
                // width is `head_vars.len()`, not the full static var count.
                let n_vars = match &plan.projection {
                    Some(p) => p.head_vars.len(),
                    None => plan.var_order.len(),
                };
                RulePlan {
                    idx: *idx,
                    atoms: plan.atoms.iter().map(|a| a.slots.clone()).collect(),
                    atom_funcs,
                    var_col: plan.var_col.clone(),
                    n_vars,
                    body_funcs,
                    triangle,
                    cq,
                    projection: plan.projection.clone(),
                }
            })
            .collect();

        let probe: ProbeHandle<u32> = ProbeHandle::new();
        let probe_in = probe.clone();
        // Per-rule capture buffers, allocated outside the closure so we can keep a
        // clone here and route each rule's output to its head after `step`.
        let captures: Vec<Rc<RefCell<Vec<(Row, isize)>>>> = rule_plans
            .iter()
            .map(|_| Rc::new(RefCell::new(Vec::new())))
            .collect();
        let captures_in = captures.clone();
        let funcs_in = funcs.clone();
        // The per-rule metadata `FusedRule` needs (kept here; the closure consumes
        // `rule_plans` for the dataflow build).
        let rule_meta: Vec<(usize, usize, Vec<FunctionId>, bool)> = rule_plans
            .iter()
            .map(|rp| {
                let reads_transient = rp.body_funcs.iter().any(|f| transient_funcs.contains(f));
                (rp.idx, rp.n_vars, rp.body_funcs.clone(), reads_transient)
            })
            .collect();

        // PERF: the per-epoch input delta is already set-semantic — it is built
        // from a `HashSet` set-difference vs the fed view (`interpret::fused_bindings`),
        // so each row appears at most once with weight ±1. The input integral
        // therefore stays 0/1 per row WITHOUT `.distinct()`, making the input
        // distinct (a full integral + per-key consolidation every epoch, over the
        // LARGE relation integrals) pure overhead. Dropped by default; set
        // `FLOWLOG_DD_KEEP_INPUT_DISTINCT` to restore it. (Mirrors feldera's
        // `FELDERA_KEEP_INPUT_DISTINCT` finding.)
        let keep_input_distinct = std::env::var_os("FLOWLOG_DD_KEEP_INPUT_DISTINCT").is_some();
        let keep_output_distinct = std::env::var_os("FLOWLOG_DD_KEEP_OUTPUT_DISTINCT").is_some();
        let inputs = worker.dataflow::<u32, _, _>(move |scope| {
            // ONE shared input + base collection per distinct relation, shared by
            // every atom occurrence (in every rule) that reads it.
            let mut inputs: HashMap<FunctionId, InputSession<u32, Row, isize>> = HashMap::new();
            let mut rel_coll: HashMap<FunctionId, _> = HashMap::new();
            for &f in &funcs_in {
                let mut session: InputSession<u32, Row, isize> = InputSession::new();
                let base = session.to_collection(scope);
                let coll = if keep_input_distinct {
                    base.map(|r: Row| (r, ())).distinct().map(|(r, ())| r)
                } else {
                    base
                };
                inputs.insert(f, session);
                rel_coll.insert(f, coll);
            }

            for (rp, cap) in rule_plans.iter().zip(captures_in.iter()) {
                // This rule's per-atom collection vector, from the SHARED relation
                // collections (cloning a DD collection is just a handle copy).
                let n_atoms = rp.atoms.len();
                let atom_slots = &rp.atoms;
                let var_col = &rp.var_col;
                let n_vars = rp.n_vars;

                // `--wcoj`: the recognized triangle rule runs the worst-case-
                // optimal delta query (prefix-extension + AltNeu 3-stream
                // decomposition) over the SHARED Mul/Add input collections,
                // emitting the SAME n_vars-wide binding Row as the binary chain.
                let cur = if let Some(tri) = &rp.triangle {
                    // Diagnostic (gated `FLOWLOG_WCOJ_TRACE`): confirm which rules
                    // route to the WCOJ path. Mirrors the other `FLOWLOG_*` debug
                    // gates; zero cost when off.
                    #[allow(clippy::disallowed_macros)]
                    if std::env::var_os("FLOWLOG_WCOJ_TRACE").is_some() {
                        eprintln!(
                            "[WCOJ] triangle rule idx={} mul={:?} add={:?}",
                            rp.idx, tri.mul_func, tri.add_func
                        );
                    }
                    let mul = rel_coll[&tri.mul_func].clone();
                    let add = rel_coll[&tri.add_func].clone();
                    wcoj_triangle_collection(scope, mul, add, tri.clone())
                } else if let Some(cq) = &rp.cq {
                    // GENERAL WCOJ: a detected cyclic >=3-atom body. One shared
                    // input collection per atom (in atom order), fed to the
                    // k-stream delta decomposition. Emits the same n_vars-wide
                    // binding Row as the binary chain.
                    #[allow(clippy::disallowed_macros)]
                    if std::env::var_os("FLOWLOG_WCOJ_TRACE").is_some() {
                        eprintln!(
                            "[WCOJ] cyclic-cq rule idx={} atoms={} funcs={:?}",
                            rp.idx,
                            cq.atom_funcs.len(),
                            cq.atom_funcs
                        );
                    }
                    let colls: Vec<_> = cq.atom_funcs.iter().map(|f| rel_coll[f].clone()).collect();
                    wcoj_cq_collection(scope, &colls, cq.clone())
                } else if let Some(proj) = &rp.projection {
                    // PROJECTED binary chain (`FLOWLOG_PROJECT`): the binding row
                    // is laid out per-step with column REUSE (`proj.step_col`), so
                    // a rule whose static var count exceeds `W` still fits. The
                    // chain is otherwise identical to the static path below; the
                    // only differences are (a) every var's column is the per-step
                    // column instead of the static `var_col`, (b) each merge REMAPS
                    // carried-over vars whose column changed (reuse) and zeroes
                    // freed columns, and (c) a final `.map` packs the surviving
                    // vars into columns `0..head_vars.len()` for the capture.
                    let step_col = &proj.step_col;
                    let slots0 = atom_slots[0].clone();
                    let sc0 = step_col[0].clone();
                    let mut cur = rel_coll[&rp.atom_funcs[0]]
                        .clone()
                        .flat_map(move |r: Row| bind_atom(&r, &slots0, &sc0));

                    for i in 1..n_atoms {
                        let slots = atom_slots[i].clone();
                        let prev = &step_col[i - 1];
                        let curl = &step_col[i];
                        // Shared = atom vars already live (present in `prev`).
                        let shared: Vec<u32> = atom_vars(&slots)
                            .into_iter()
                            .filter(|v| prev.contains_key(v))
                            .collect();
                        // Left key reads each shared var from its PREVIOUS column.
                        let shared_cols_left: Vec<usize> = shared.iter().map(|v| prev[v]).collect();
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
                        let right = rel_coll[&rp.atom_funcs[i]]
                            .clone()
                            .map(move |r: Row| (pack_key(&r, &sac), r));

                        // Carried vars: live at BOTH prev and cur but NOT produced
                        // by this atom — copy them from prev[v] to cur[v]. Atom
                        // vars: written/validated at cur[v]. (prev_col, cur_col,
                        // is_shared, atom_col) per relevant var, precomputed.
                        let slotsc = slots.clone();
                        let prevc = prev.clone();
                        let curc = curl.clone();
                        cur = left
                            .join(right)
                            .flat_map(move |(_k, (b, r)): (Key, (Row, Row))| {
                                remap_merge_atom_into(&b, &r, &slotsc, &prevc, &curc)
                            });
                    }

                    // Final projection: pack the surviving vars (at their last-step
                    // columns `head_cols`) into the capture columns `0..k`, zeroing
                    // the rest, so the head scatter reads `bind[i]` for surviving
                    // var `i` exactly like the static path.
                    let head_cols = proj.head_cols.clone();
                    cur.map(move |b: Row| pack_key(&b, &head_cols))
                } else {
                    let mut bound = vec![false; n_vars];
                    let slots0 = atom_slots[0].clone();
                    let vc0 = var_col.clone();
                    let mut cur = rel_coll[&rp.atom_funcs[0]]
                        .clone()
                        .flat_map(move |r: Row| bind_atom(&r, &slots0, &vc0));
                    mark_bound(&atom_slots[0], var_col, &mut bound);

                    for i in 1..n_atoms {
                        let slots = atom_slots[i].clone();
                        let shared: Vec<u32> = atom_vars(&slots)
                            .into_iter()
                            .filter(|v| var_col.get(v).map(|&c| bound[c]).unwrap_or(false))
                            .collect();
                        let shared_cols_left: Vec<usize> =
                            shared.iter().map(|v| var_col[v]).collect();
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
                        let right = rel_coll[&rp.atom_funcs[i]]
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
                        mark_bound(&slots, var_col, &mut bound);
                    }
                    cur
                };

                // PERF: the output `.distinct()` is redundant here. `step`
                // accumulates each rule's binding deltas into a per-key weight map
                // and `interpret::fused_bindings` inspects only the SIGN of the net
                // weight (>0 ⇒ one env; net-zero already filtered). distinct would
                // clamp the binding multiplicity to {0,1}, but since only the sign
                // is observed and net-zero rows are dropped, the clamp is
                // unobservable. `.consolidate()` still collapses per-key
                // multiplicities so the captured batch is one signed row per key.
                // Dropped by default; set `FLOWLOG_DD_KEEP_OUTPUT_DISTINCT` to
                // restore. (Mirrors feldera's `FELDERA_KEEP_OUTPUT_DISTINCT`.)
                let consolidated = if keep_output_distinct {
                    cur.map(|b: Row| (b, ()))
                        .distinct()
                        .map(|(b, ())| b)
                        .consolidate()
                } else {
                    cur.consolidate()
                };
                let out = consolidated;

                let cap = Rc::clone(cap);
                out.inner
                    .inspect_batch(move |_t, batch| {
                        let mut buf = cap.borrow_mut();
                        for (row, _time, w) in batch.iter() {
                            buf.push((*row, *w));
                        }
                    })
                    .probe_with(&probe_in);
            }

            inputs
        });

        let rules: Vec<FusedRule> = rule_meta
            .into_iter()
            .zip(captures)
            .map(
                |((idx, n_vars, body_funcs, reads_transient), captured)| FusedRule {
                    idx,
                    captured,
                    n_vars,
                    body_funcs,
                    reads_transient,
                },
            )
            .collect();

        let transient_funcs: HashSet<FunctionId> = transient_funcs
            .iter()
            .copied()
            .filter(|f| inputs.contains_key(f))
            .collect();

        Ok(FusedDdJoin {
            worker,
            inputs,
            probe,
            rules,
            epoch: 0,
            transient_funcs,
            delta_rebuild,
        })
    }

    /// The rule indices this fused worker serves (build order).
    pub fn rule_indices(&self) -> Vec<usize> {
        self.rules.iter().map(|r| r.idx).collect()
    }

    /// The body relations the fused rule at build position `pos` reads.
    pub fn rule_body_funcs(&self, pos: usize) -> &[FunctionId] {
        &self.rules[pos].body_funcs
    }

    /// Feed one epoch of signed relation deltas into the SHARED inputs, advance
    /// the timestamp, run the SINGLE worker to this epoch's fixpoint, and return
    /// per-rule binding deltas. The outer `Vec` is in [`rule_indices`] order; each
    /// inner `Vec` is `(binding_row_as_var_order_vec, weight)`.
    ///
    /// CRUCIAL: the InputSessions are NEVER cleared — only the delta is pushed, so
    /// the DD arrangements persist and the join is genuinely incremental.
    pub fn step(
        &mut self,
        deltas: &HashMap<FunctionId, Vec<(Vec<u32>, isize)>>,
    ) -> Result<Vec<Vec<(Vec<u32>, isize)>>> {
        // The transient-UF rebuild (Phase 2) is a PROVEN-UNSOUND construction for
        // this rebuild structure — it is off unless explicitly opted in via
        // `FLOWLOG_ENABLE_PHASE2`, so the canonicalize-at-creation flag keeps its
        // bit-exact Phase-1 (symmetric) behavior. See the module note below and
        // `CANON_UF_NOTE.md`: the rebuild work lives in `δview ⋈ uf` (≈12.8k
        // surviving rewrites at math N=7), NOT `view_all ⋈ δuf` (≈37); dropping
        // the former collapses the fixpoint (1165/1330 → 86/74). Retained behind
        // the opt-in for the negative-result record + further investigation.
        if !self.transient_funcs.is_empty() && self.delta_rebuild {
            return self.step_delta_rebuild(deltas);
        }
        if self.transient_funcs.is_empty() || std::env::var_os("FLOWLOG_ENABLE_PHASE2").is_none() {
            return self.step_symmetric(deltas);
        }
        self.step_transient(deltas)
    }

    /// PROTOTYPE δUF-driven rebuild (see [`delta_rebuild_enabled`]). Same two
    /// persistent sub-steps as the measurement, but DROP the δview-driven rebuild
    /// output: A keeps only non-rebuild rules (congruence), B keeps rebuild rules
    /// (`view ⋈ δuf`). The UF arrangement is maintained normally (NOT zeroed).
    fn step_delta_rebuild(
        &mut self,
        deltas: &HashMap<FunctionId, Vec<(Vec<u32>, isize)>>,
    ) -> Result<Vec<Vec<(Vec<u32>, isize)>>> {
        let prof = prof_enabled();
        let mut accs: Vec<HashMap<Vec<u32>, isize>> =
            (0..self.rules.len()).map(|_| HashMap::new()).collect();

        // Sub-step A: feed δview (non-transient). Keep congruence; DROP the
        // rebuild's δview⋈uf output (the term the eclass fix makes empty).
        let ta = std::time::Instant::now();
        for (func, rows) in deltas {
            if self.transient_funcs.contains(func) {
                continue;
            }
            if let Some(inp) = self.inputs.get_mut(func) {
                for (row, w) in rows {
                    inp.update(pack_row(row), *w);
                }
            }
        }
        for rule in &self.rules {
            rule.captured.borrow_mut().clear();
        }
        let ea = self.epoch + 1;
        self.drive_to(ea, ta, prof);
        self.drain_into(&mut accs, |r| !r.reads_transient);

        // Sub-step B: feed δUF (transient). Keep the rebuild's view⋈δuf output.
        let tb = std::time::Instant::now();
        for (func, rows) in deltas {
            if !self.transient_funcs.contains(func) {
                continue;
            }
            if let Some(inp) = self.inputs.get_mut(func) {
                for (row, w) in rows {
                    inp.update(pack_row(row), *w);
                }
            }
        }
        for rule in &self.rules {
            rule.captured.borrow_mut().clear();
        }
        let eb = self.epoch + 2;
        self.drive_to(eb, tb, prof);
        self.drain_into(&mut accs, |r| r.reads_transient);
        self.epoch = eb;

        Ok(accs
            .into_iter()
            .map(|acc| acc.into_iter().filter(|(_, w)| *w != 0).collect())
            .collect())
    }

    /// Step the SINGLE worker to `epoch`, accounting feed/step time.
    fn drive_to(&mut self, epoch: u32, t_feed: std::time::Instant, prof: bool) {
        for inp in self.inputs.values_mut() {
            inp.advance_to(epoch);
            inp.flush();
        }
        if prof {
            add_ns(&PROF_FEED_NS, t_feed.elapsed());
            PROF_STEP_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        let t_step = std::time::Instant::now();
        let probe = self.probe.clone();
        self.worker.step_while(|| probe.less_than(&epoch));
        if prof {
            add_ns(&PROF_STEP_NS, t_step.elapsed());
        }
    }

    /// Drain each rule's capture buffer; keep a rule's output into `accs` only if
    /// `keep(rule)` (route a sub-step's output to the rules it is meant for).
    fn drain_into(&self, accs: &mut [HashMap<Vec<u32>, isize>], keep: impl Fn(&FusedRule) -> bool) {
        for (rule, acc) in self.rules.iter().zip(accs.iter_mut()) {
            let drained: Vec<(Row, isize)> = rule.captured.borrow_mut().drain(..).collect();
            if !keep(rule) {
                continue;
            }
            for (row, w) in drained {
                let key: Vec<u32> = (0..rule.n_vars).map(|i| row[i]).collect();
                *acc.entry(key).or_insert(0) += w;
            }
        }
    }

    /// The original symmetric incremental join: feed ALL deltas, one sub-step,
    /// capture every rule. Behavior-identical to the pre-Phase-2 `step`.
    fn step_symmetric(
        &mut self,
        deltas: &HashMap<FunctionId, Vec<(Vec<u32>, isize)>>,
    ) -> Result<Vec<Vec<(Vec<u32>, isize)>>> {
        let prof = prof_enabled();
        let t_feed = std::time::Instant::now();
        let mut pushed = false;
        for (func, rows) in deltas {
            if let Some(inp) = self.inputs.get_mut(func) {
                for (row, w) in rows {
                    inp.update(pack_row(row), *w);
                    pushed = true;
                }
            }
        }
        let next_epoch = self.epoch + 1;
        if !pushed {
            self.epoch = next_epoch;
            return Ok(vec![Vec::new(); self.rules.len()]);
        }

        for rule in &self.rules {
            rule.captured.borrow_mut().clear();
        }
        self.drive_to(next_epoch, t_feed, prof);
        self.epoch = next_epoch;

        if trace_enabled() {
            let total: usize = self.rules.iter().map(|r| r.captured.borrow().len()).sum();
            let reb: usize = self
                .rules
                .iter()
                .filter(|r| r.reads_transient)
                .map(|r| r.captured.borrow().len())
                .sum();
            let n_rules = self.rules.len();
            use egglog_numeric_id::NumericId;
            let funcs: Vec<u32> = deltas.keys().map(|f| f.rep()).collect();
            let delta_rows: usize = deltas.values().map(|r| r.len()).sum();
            #[allow(clippy::disallowed_macros)]
            {
                eprintln!(
                    "[dd_symmetric] n_rules={n_rules} delta_funcs={funcs:?} delta_rows={delta_rows} total_out={total} rebuild_rule_out={reb}"
                );
            }
        }

        let mut accs: Vec<HashMap<Vec<u32>, isize>> =
            (0..self.rules.len()).map(|_| HashMap::new()).collect();
        self.drain_into(&mut accs, |_| true);
        Ok(accs
            .into_iter()
            .map(|acc| acc.into_iter().filter(|(_, w)| *w != 0).collect())
            .collect())
    }

    /// Phase-2 transient-UF rebuild — **PROVEN UNSOUND for this rebuild
    /// structure; off unless `FLOWLOG_ENABLE_PHASE2` is set.** Three sub-steps so
    /// each rebuild rule sees `view_all ⋈ δuf` while the `δview ⋈ uf_old` term is
    /// never built: A feeds δview (transient integral = 0 ⇒ rebuild emits nothing;
    /// capture non-rebuild rules); B pulses +δuf (rebuild gets `view_all ⋈ δuf`;
    /// capture rebuild rules); C pulses −δuf (reset integral to 0; discard).
    ///
    /// ## Why this is unsound (the measured negative result)
    ///
    /// The premise (CANON_UF_NOTE.md Phase 2) was that canonical-at-creation makes
    /// `δview ⋈ uf_old` empty, so it can be dropped. Instrumenting the SYMMETRIC
    /// (correct) rebuilding worker on math N=7 refutes that: of its 28.6k total
    /// rebuild+congruence bindings, the rebuild rule alone produces **12,828
    /// surviving rewrites**, essentially all from `δview ⋈ uf` — because `@UF_Sf`
    /// is large and near-stable per rebuilding step (δuf = 2–35 rows) while δview
    /// is the large driver (hundreds–thousands of rows). Canonical-at-creation
    /// only makes a row canonical AT CREATION; a later union changes a child's
    /// leader and the EXISTING view row must be repaired — that repair IS the
    /// `δview ⋈ uf` term. `view_all ⋈ δuf` captures only ~37 of those 12.8k, so
    /// dropping `δview ⋈ uf` starves the rebuild → no new canonical view rows →
    /// congruence never fires on a growing view → the fixpoint collapses
    /// (math N=7 1165/1330 → 86/74; 52 rebuilding steps → 21). The asymmetry the
    /// design assumed is BACKWARDS: the work is on the δview side, not δuf.
    fn step_transient(
        &mut self,
        deltas: &HashMap<FunctionId, Vec<(Vec<u32>, isize)>>,
    ) -> Result<Vec<Vec<(Vec<u32>, isize)>>> {
        let prof = prof_enabled();
        let trace = trace_enabled();

        let mut view_pushed = false;
        let mut uf_rows: Vec<(FunctionId, Vec<u32>, isize)> = Vec::new();
        let t_feed_a = std::time::Instant::now();
        for (func, rows) in deltas {
            let Some(inp) = self.inputs.get_mut(func) else {
                continue;
            };
            if self.transient_funcs.contains(func) {
                for (row, w) in rows {
                    uf_rows.push((*func, row.clone(), *w));
                }
            } else {
                for (row, w) in rows {
                    inp.update(pack_row(row), *w);
                    view_pushed = true;
                }
            }
        }
        let view_delta_rows: usize = deltas
            .iter()
            .filter(|(f, _)| !self.transient_funcs.contains(*f))
            .map(|(_, r)| r.len())
            .sum();
        let uf_pushed = !uf_rows.is_empty();
        if !view_pushed && !uf_pushed {
            self.epoch += 1;
            return Ok(vec![Vec::new(); self.rules.len()]);
        }

        let mut accs: Vec<HashMap<Vec<u32>, isize>> =
            (0..self.rules.len()).map(|_| HashMap::new()).collect();

        // A (δview)
        for rule in &self.rules {
            rule.captured.borrow_mut().clear();
        }
        let ea = self.epoch + 1;
        self.drive_to(ea, t_feed_a, prof);
        let a_rebuild: usize = self
            .rules
            .iter()
            .filter(|r| r.reads_transient)
            .map(|r| r.captured.borrow().len())
            .sum();
        let a_congruence: usize = self
            .rules
            .iter()
            .filter(|r| !r.reads_transient)
            .map(|r| r.captured.borrow().len())
            .sum();
        self.drain_into(&mut accs, |r| !r.reads_transient);

        // B (+δuf)
        for rule in &self.rules {
            rule.captured.borrow_mut().clear();
        }
        let t_feed_b = std::time::Instant::now();
        for (func, row, w) in &uf_rows {
            if let Some(inp) = self.inputs.get_mut(func) {
                inp.update(pack_row(row), *w);
            }
        }
        let eb = self.epoch + 2;
        self.drive_to(eb, t_feed_b, prof);
        let b_rebuild: usize = self
            .rules
            .iter()
            .filter(|r| r.reads_transient)
            .map(|r| r.captured.borrow().len())
            .sum();
        self.drain_into(&mut accs, |r| r.reads_transient);

        // C (−δuf): reset integral, discard output.
        for rule in &self.rules {
            rule.captured.borrow_mut().clear();
        }
        let t_feed_c = std::time::Instant::now();
        for (func, row, w) in &uf_rows {
            if let Some(inp) = self.inputs.get_mut(func) {
                inp.update(pack_row(row), -*w);
            }
        }
        let ec = self.epoch + 3;
        self.drive_to(ec, t_feed_c, prof);
        for rule in &self.rules {
            rule.captured.borrow_mut().clear();
        }
        self.epoch = ec;

        if trace {
            #[allow(clippy::disallowed_macros)]
            {
                eprintln!(
                    "[dd_transient] n_rules={} view_delta_rows={view_delta_rows} uf_delta_rows={} \
                     congruence_out_A(kept)={a_congruence} \
                     rebuild_out_A(δview⋈uf,dropped)={a_rebuild} rebuild_out_B(view⋈δuf,kept)={b_rebuild}",
                    self.rules.len(),
                    uf_rows.len()
                );
            }
        }

        Ok(accs
            .into_iter()
            .map(|acc| acc.into_iter().filter(|(_, w)| *w != 0).collect())
            .collect())
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

/// Projecting merge for the `FLOWLOG_PROJECT` chain. Like [`merge_atom_into`] but
/// rebuilds the row from scratch under a NEW column layout: `prev` is the
/// left-row layout (step `i-1`), `cur` is the output layout (step `i`). Carried
/// vars (live in both layouts but not produced here) are copied `prev[v]→cur[v]`;
/// the atom's vars are validated (shared) or written (fresh) at `cur[v]`; every
/// other output column is left zeroed. This is what reuses freed columns and so
/// keeps the frontier within `W`. Empty vec on a const / repeated-var / shared-
/// var constraint failure (same semantics as the static merge).
fn remap_merge_atom_into(
    b: &Row,
    r: &Row,
    slots: &[Slot],
    prev: &HashMap<u32, usize>,
    cur: &HashMap<u32, usize>,
) -> Vec<Row> {
    let mut out = empty_row();
    // Carry over every still-live var (present in `cur`) that the left row
    // already holds (present in `prev`). Atom-fresh vars are absent from `prev`,
    // so they are NOT copied here — they are written from `r` below.
    for (&v, &pc) in prev {
        if let Some(&cc) = cur.get(&v) {
            out[cc] = b[pc];
        }
    }
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
                if let Some(&prior) = local.get(v) {
                    if prior != val {
                        return Vec::new();
                    }
                    continue;
                }
                local.insert(*v, val);
                // Every atom var is live at this step ⇒ present in `cur`.
                let cc = cur[v];
                if prev.contains_key(v) {
                    // Shared (already bound): the carried value must agree. (The
                    // join key already enforces this for the key columns; this
                    // also covers shared vars not in the key, if any.)
                    if out[cc] != val {
                        return Vec::new();
                    }
                } else {
                    // Fresh var born at this atom: write it.
                    out[cc] = val;
                }
            }
        }
    }
    vec![out]
}

// ---------------------------------------------------------------------------
// Stage 2: GENERAL `--wcoj` worst-case-optimal delta query (any cyclic body)
// ---------------------------------------------------------------------------

/// A WCOJ collection index over the `Row`-keyed encoding used by the general
/// construction: K = the prefix `Row` (with the relevant bound columns filled,
/// others 0), V = the single proposed variable value (`u32`).
type CqIndex<'scope> = CollectionIndex<Row, u32, AltNeu<u32>, isize>;

/// The key-selector closure type shared by every general-WCOJ extender: read
/// `cols` of the prefix `Row` into a fresh key `Row`. Writing it ONCE here (one
/// source location) makes every `extend_using(make_key_selector(...))` the SAME
/// concrete `CollectionExtender` type, so a step's extenders can live in one
/// `Vec` and be passed as a `&mut [&mut dyn PrefixExtender]` slice to `extend`.
fn cq_key_selector(cols: Vec<usize>) -> impl Fn(&Row) -> Row + Clone + 'static {
    move |p: &Row| {
        let mut k = empty_row();
        for (i, &c) in cols.iter().enumerate() {
            k[i] = p[c];
        }
        k
    }
}

/// Build the GENERAL worst-case-optimal join for a detected cyclic body
/// (`CqPlan`) as a collection of full-width binding `Row`s — the SAME shape the
/// binary `.join` chain emits, so the downstream consolidate / capture / env
/// scatter is bit-identical.
///
/// Implements the full k-stream delta decomposition (one delta query per body
/// atom) inside ONE `AltNeu<u32>` nested scope, exactly mirroring the Stage-1
/// triangle: in delta query `dQ/dA_d`, atoms with index < d read the ALT (old)
/// trace and atoms with index > d read the NEU (new) trace, so simultaneous
/// cross-atom updates are counted exactly once. ALL reads (the multiway
/// intersections AND the per-atom payload recoveries) are inside the AltNeu
/// scope. Each JOIN step binds ONE join variable: a >=2-extender step is the
/// WCOJ multiway `extend` (the intersection that collapses the intermediate); a
/// 1-extender step is a `propose_using`. After the join core is pinned, each
/// non-driver atom's payload vars are recovered (and the atom validated) by one
/// `propose` keyed on its join columns.
fn wcoj_cq_collection<'scope>(
    scope: Scope<'scope, u32>,
    colls: &[VecCollection<'scope, u32, Row, isize>],
    plan: CqPlan,
) -> VecCollection<'scope, u32, Row, isize> {
    use differential_dataflow::AsCollection;
    use timely::dataflow::operators::Concatenate;

    scope.scoped::<AltNeu<u32>, _, _>("WcojCq", move |inner| {
        // Enter every relation; build an ALT and a NEU copy of each.
        let alt: Vec<_> = colls.iter().map(|c| c.clone().enter(inner)).collect();
        let neu: Vec<_> = alt
            .iter()
            .map(|c| c.clone().delay(|t| AltNeu::neu(t.time)))
            .collect();

        // Per-driver delta query. We build the indexes each driver needs lazily
        // (cloning a CollectionIndex shares its traces, so repeats are cheap).
        let mut streams = Vec::with_capacity(plan.drivers.len());
        for driver in &plan.drivers {
            // Seed the prefix from the driver atom's delta row.
            let seed = driver.seed.clone();
            let prefix = alt[driver.driver_idx].clone().map(move |r: Row| {
                let mut p = empty_row();
                for &(bind_col, rel_col) in &seed {
                    p[bind_col] = r[rel_col];
                }
                p
            });

            let mut cur = prefix;
            for step in &driver.steps {
                // One index per extender (over the ALT or NEU copy of its atom),
                // keyed by the atom's key columns, proposing the step's var.
                let mut indexes: Vec<CqIndex> = step
                    .exts
                    .iter()
                    .map(|e| {
                        let src = if e.alt {
                            alt[e.atom_idx].clone()
                        } else {
                            neu[e.atom_idx].clone()
                        };
                        let key_atom_cols = e.key_atom_cols.clone();
                        let propose_col = e.propose_col;
                        // A JOIN-var proposal is SET-semantic: "does this atom hold
                        // a row with these key cols proposing this var?" — weight 1,
                        // regardless of how many full rows witness it. Without the
                        // `.distinct()`, an atom carrying PAYLOAD (or any) columns
                        // beyond (key, propose) yields the SAME (K,V) pair once per
                        // such row, and `propose`/`validate` MULTIPLY the prefix
                        // weight by that multiplicity (the `propose`/`validate`
                        // traces are non-distinct; only `count` is). That
                        // over-counted the acyclic star/path bodies (e.g. two F-rows
                        // sharing (v0,v1) doubled the binding weight). Dedup the
                        // (K,V) pairs so the proposal is exactly the SET of valid
                        // extensions. (Payload values are recovered separately, in
                        // the `recovers` step below, where multiplicity is correct.)
                        let kv = src
                            .map(move |r: Row| {
                                let mut k = empty_row();
                                for (i, &c) in key_atom_cols.iter().enumerate() {
                                    k[i] = r[c];
                                }
                                (k, r[propose_col])
                            })
                            .map(|kv| (kv, ()))
                            .distinct()
                            .map(|(kv, ())| kv);
                        CollectionIndex::index(kv)
                    })
                    .collect();

                // Build the extenders (all the same concrete type — see
                // `cq_key_selector`), then intersect via `extend` (>=2) or
                // recover via `propose_using` (==1).
                let mut extenders: Vec<_> = indexes
                    .iter_mut()
                    .zip(step.exts.iter())
                    .map(|(idx, e)| idx.extend_using(cq_key_selector(e.key_cols.clone())))
                    .collect();

                let bind_col = step.bind_col;
                cur = if extenders.len() == 1 {
                    cur.propose_using(&mut extenders[0])
                        .map(move |(mut p, val): (Row, u32)| {
                            p[bind_col] = val;
                            p
                        })
                } else {
                    let mut refs: Vec<
                        &mut dyn differential_dogs3::PrefixExtender<
                            AltNeu<u32>,
                            isize,
                            Prefix = Row,
                            Extension = u32,
                        >,
                    > = extenders
                        .iter_mut()
                        .map(|e| {
                            e as &mut dyn differential_dogs3::PrefixExtender<
                                AltNeu<u32>,
                                isize,
                                Prefix = Row,
                                Extension = u32,
                            >
                        })
                        .collect();
                    cur.extend(&mut refs[..])
                        .map(move |(mut p, val): (Row, u32)| {
                            p[bind_col] = val;
                            p
                        })
                };
            }

            // Payload recovery: every non-driver atom is now keyed entirely on
            // bound join columns. One `propose` per atom recovers its payload
            // vars (V = a Row packing the payload values) AND validates the
            // atom's existence (an atom with no payload still must be present).
            for rec in &driver.recovers {
                let src = if rec.alt {
                    alt[rec.atom_idx].clone()
                } else {
                    neu[rec.atom_idx].clone()
                };
                let key_atom_cols = rec.key_atom_cols.clone();
                let payload = rec.payload.clone();
                // Index (K = join-key Row, V = payload-values Row).
                let idx: CollectionIndex<Row, Row, AltNeu<u32>, isize> =
                    CollectionIndex::index(src.map(move |r: Row| {
                        let mut k = empty_row();
                        for (i, &c) in key_atom_cols.iter().enumerate() {
                            k[i] = r[c];
                        }
                        let mut v = empty_row();
                        for (slot, &(_bind, rel_col)) in payload.iter().enumerate() {
                            v[slot] = r[rel_col];
                        }
                        (k, v)
                    }));
                let mut ext = idx.extend_using(cq_key_selector(rec.key_cols.clone()));
                let payload2 = rec.payload.clone();
                cur = cur
                    .propose_using(&mut ext)
                    .map(move |(mut p, vals): (Row, Row)| {
                        for (slot, &(bind_col, _rel)) in payload2.iter().enumerate() {
                            p[bind_col] = vals[slot];
                        }
                        p
                    });
            }

            streams.push(cur.inner);
        }

        inner.concatenate(streams).as_collection().leave(scope)
    })
}

// ---------------------------------------------------------------------------
// `--wcoj` triangle worst-case-optimal delta query
// ---------------------------------------------------------------------------

/// Build the worst-case-optimal triangle join as a differential-dataflow
/// collection of full `n_vars`-wide binding `Row`s — the SAME shape the binary
/// `.join` chain emits, so the downstream consolidate / capture / env scatter
/// is bit-identical.
///
/// Implements the FULL 3-stream delta decomposition (one delta query per body
/// atom) inside a single `AltNeu<u32>` nested scope, following dogsdogsdogs'
/// `examples/delta_query_wcoj.rs`. The total atom order is A0(Mul_a) <
/// A1(Mul_b) < A2(Add): in each delta query, atoms BEFORE the driving atom read
/// the `alt` (old) trace and atoms AFTER read the `neu` (new) trace, so
/// simultaneous cross-atom updates (incl. the Mul self-join) are not
/// double-counted. The ONLY multiway-intersected variable is the core `a`
/// (driven by dA2: intersect the `a`-sets of the two Mul eclasses m1,m2 — the
/// `Σ_a deg(a)²` collapse); every other variable is functionally recovered by a
/// single in-scope `propose_using`. ALL reads (core intersection AND payload
/// recovery) are inside the AltNeu scope — unlike the spike shortcut whose
/// out-of-scope recovery drifted at high iteration counts.
fn wcoj_triangle_collection<'scope>(
    scope: Scope<'scope, u32>,
    mul: VecCollection<'scope, u32, Row, isize>,
    add: VecCollection<'scope, u32, Row, isize>,
    tri: TriangleShape,
) -> VecCollection<'scope, u32, Row, isize> {
    use differential_dataflow::AsCollection;
    use timely::dataflow::operators::Concatenate;

    // Raw-relation column layout (fixed by the term encoding's arity-4 schema):
    //   Mul row = [arg0=a/c, arg1=b/c-payload, eclass=m, rowextra=x]
    //   Add row = [m1, m2, eclass=o, rowextra=x]
    // (cols 0..3 are the RELATION columns; tri.col_* are the BINDING columns.)
    // Each stream assembles a full-width `empty_row()` and writes only the 9
    // triangle columns; columns >= n_vars stay 0 — identical to the binary
    // chain, which also leaves unused binding columns at 0. (n_vars itself is
    // not needed here; the downstream drain reads cols 0..rule.n_vars.)
    let (ca, cb, cm1, cx1) = (tri.col_a, tri.col_b, tri.col_m1, tri.col_x1);
    let (cc, cm2, cx2) = (tri.col_c, tri.col_m2, tri.col_x2);
    let (co, cx3) = (tri.col_o, tri.col_x3);

    scope.scoped::<AltNeu<u32>, _, _>("WcojTriangle", move |inner| {
        let mul = mul.enter(inner);
        let add = add.enter(inner);
        let mul_neu = mul.clone().delay(|t| AltNeu::neu(t.time));
        let add_neu = add.clone().delay(|t| AltNeu::neu(t.time));

        // --- indices over the entered collections ---------------------------
        // Mul indices: (K=eclass m, V=arg a), (K=arg a, V=eclass m),
        //              (K=(a,m), V=(b,x))  [payload recovery]
        let alt_mul_by_m = CollectionIndex::index(mul.clone().map(|r: Row| (r[2], r[0])));
        let alt_mul_by_a = CollectionIndex::index(mul.clone().map(|r: Row| (r[0], r[2])));
        let neu_mul_by_a = CollectionIndex::index(mul_neu.clone().map(|r: Row| (r[0], r[2])));
        let alt_mul_by_am =
            CollectionIndex::index(mul.clone().map(|r: Row| ((r[0], r[2]), (r[1], r[3]))));
        let neu_mul_by_am =
            CollectionIndex::index(mul_neu.clone().map(|r: Row| ((r[0], r[2]), (r[1], r[3]))));
        // Add indices: (K=m1, V=m2), (K=m2, V=m1), (K=(m1,m2), V=(o,x)).
        let neu_add_by_m1 = CollectionIndex::index(add_neu.clone().map(|r: Row| (r[0], r[1])));
        let neu_add_by_m2 = CollectionIndex::index(add_neu.clone().map(|r: Row| (r[1], r[0])));
        let neu_add_by_m1m2 =
            CollectionIndex::index(add_neu.clone().map(|r: Row| ((r[0], r[1]), (r[2], r[3]))));

        // ===== dQ/dA0 : driven by dMul as Mul_a(a,b,m1,x1) ==================
        // bound a,b,m1,x1 ; A1(Mul_b) NEU, A2(Add) NEU.
        let changes0 = {
            // initial prefix: place the Mul delta row's cols into a,b,m1,x1.
            let prefix = mul.clone().map(move |r: Row| {
                let mut p = empty_row();
                p[ca] = r[0];
                p[cb] = r[1];
                p[cm1] = r[2];
                p[cx1] = r[3];
                p
            });
            // 1. intersect m2: Add(m1,·)=m2 (NEU) ∩ Mul(a,·)=m2 (NEU).
            let with_m2 = prefix
                .extend(&mut [
                    &mut neu_add_by_m1.extend_using(move |p: &Row| p[cm1]),
                    &mut neu_mul_by_a.extend_using(move |p: &Row| p[ca]),
                ])
                .map(move |(mut p, m2): (Row, u32)| {
                    p[cm2] = m2;
                    p
                });
            // 2. recover (c,x2) for Mul_b given (a,m2) (NEU).
            let with_cx2 = with_m2
                .propose_using(&mut neu_mul_by_am.extend_using(move |p: &Row| (p[ca], p[cm2])))
                .map(move |(mut p, (c, x2)): (Row, (u32, u32))| {
                    p[cc] = c;
                    p[cx2] = x2;
                    p
                });
            // 3. recover (o,x3) for Add given (m1,m2) (NEU).
            with_cx2
                .propose_using(&mut neu_add_by_m1m2.extend_using(move |p: &Row| (p[cm1], p[cm2])))
                .map(move |(mut p, (o, x3)): (Row, (u32, u32))| {
                    p[co] = o;
                    p[cx3] = x3;
                    p
                })
        };

        // ===== dQ/dA1 : driven by dMul as Mul_b(a,c,m2,x2) ==================
        // bound a,c,m2,x2 ; A0(Mul_a) ALT, A2(Add) NEU.
        let changes1 = {
            let prefix = mul.clone().map(move |r: Row| {
                let mut p = empty_row();
                p[ca] = r[0];
                p[cc] = r[1];
                p[cm2] = r[2];
                p[cx2] = r[3];
                p
            });
            // 1. intersect m1: Add(·,m2)=m1 (NEU) ∩ Mul(a,·)=m1 (ALT).
            let with_m1 = prefix
                .extend(&mut [
                    &mut neu_add_by_m2.extend_using(move |p: &Row| p[cm2]),
                    &mut alt_mul_by_a.extend_using(move |p: &Row| p[ca]),
                ])
                .map(move |(mut p, m1): (Row, u32)| {
                    p[cm1] = m1;
                    p
                });
            // 2. recover (b,x1) for Mul_a given (a,m1) (ALT).
            let with_bx1 = with_m1
                .propose_using(&mut alt_mul_by_am.extend_using(move |p: &Row| (p[ca], p[cm1])))
                .map(move |(mut p, (b, x1)): (Row, (u32, u32))| {
                    p[cb] = b;
                    p[cx1] = x1;
                    p
                });
            // 3. recover (o,x3) for Add given (m1,m2) (NEU).
            with_bx1
                .propose_using(&mut neu_add_by_m1m2.extend_using(move |p: &Row| (p[cm1], p[cm2])))
                .map(move |(mut p, (o, x3)): (Row, (u32, u32))| {
                    p[co] = o;
                    p[cx3] = x3;
                    p
                })
        };

        // ===== dQ/dA2 : driven by dAdd(m1,m2,o,x3) =========================
        // bound m1,m2,o,x3 ; A0(Mul_a) ALT, A1(Mul_b) ALT.
        let changes2 = {
            let prefix = add.clone().map(move |r: Row| {
                let mut p = empty_row();
                p[cm1] = r[0];
                p[cm2] = r[1];
                p[co] = r[2];
                p[cx3] = r[3];
                p
            });
            // 1. intersect a: Mul(·,·)=m1 → a (ALT) ∩ Mul(·,·)=m2 → a (ALT).
            //    THE worst-case-optimal collapse.
            let with_a = prefix
                .extend(&mut [
                    &mut alt_mul_by_m.extend_using(move |p: &Row| p[cm1]),
                    &mut alt_mul_by_m.extend_using(move |p: &Row| p[cm2]),
                ])
                .map(move |(mut p, a): (Row, u32)| {
                    p[ca] = a;
                    p
                });
            // 2. recover (b,x1) for Mul_a given (a,m1) (ALT).
            let with_bx1 = with_a
                .propose_using(&mut alt_mul_by_am.extend_using(move |p: &Row| (p[ca], p[cm1])))
                .map(move |(mut p, (b, x1)): (Row, (u32, u32))| {
                    p[cb] = b;
                    p[cx1] = x1;
                    p
                });
            // 3. recover (c,x2) for Mul_b given (a,m2) (ALT).
            with_bx1
                .propose_using(&mut alt_mul_by_am.extend_using(move |p: &Row| (p[ca], p[cm2])))
                .map(move |(mut p, (c, x2)): (Row, (u32, u32))| {
                    p[cc] = c;
                    p[cx2] = x2;
                    p
                })
        };

        // Concat the three delta streams and leave the AltNeu scope.
        let streams = vec![changes0.inner, changes1.inner, changes2.inner];
        inner.concatenate(streams).as_collection().leave(scope)
    })
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
            projection: None,
        }
    }

    /// Build a `JoinPlan` directly from `(func, vars)` atoms (all slots are
    /// distinct `Var`s — the shape the WCOJ detector accepts).
    fn plan_of(atoms: &[(FunctionId, &[u32])]) -> JoinPlan {
        let mut var_order: Vec<u32> = Vec::new();
        let mut var_col: HashMap<u32, usize> = HashMap::new();
        let mut planned: Vec<PlanAtom> = Vec::new();
        for (func, vars) in atoms {
            for &v in *vars {
                if !var_col.contains_key(&v) {
                    var_col.insert(v, var_order.len());
                    var_order.push(v);
                }
            }
            planned.push(PlanAtom {
                func: *func,
                slots: vars.iter().map(|&v| Slot::Var(v)).collect(),
            });
        }
        JoinPlan {
            var_order,
            var_col,
            atoms: planned,
            projection: None,
        }
    }

    /// Feed one all-at-once delta of `rows` per func into a fresh `FusedDdJoin`
    /// (single rule idx 0, plan rebuilt from `atoms`), returning the rule's
    /// sorted output bindings.
    fn run_once(
        atoms: &[(FunctionId, &[u32])],
        wcoj: bool,
        rows: &HashMap<FunctionId, Vec<(Vec<u32>, isize)>>,
    ) -> Vec<(Vec<u32>, isize)> {
        let plan = plan_of(atoms);
        let mut j =
            FusedDdJoin::build(&[(0usize, plan)], &HashSet::new(), false, wcoj, true).unwrap();
        let mut out = j.step(rows).unwrap().remove(0);
        out.sort();
        out
    }

    /// ACYCLIC >=3-atom bodies must be bit-exact under the general WCOJ
    /// construction (broadened past the cyclic-only gate). We compare the WCOJ
    /// output against the binary chain (ground truth) row-for-row, on several
    /// acyclic shapes that previously over-derived (cross-product). The WCOJ
    /// detector is forced on for the acyclic shapes via `allow_acyclic: true`
    /// (production uses `false`; the test helper `run_once` passes `true`).
    #[test]
    fn wcoj_acyclic_bit_exact_vs_binary() {
        let f = FunctionId::new(0);
        let g = FunctionId::new(1);
        let h = FunctionId::new(2);

        // --- Shape 1: 3-atom STAR  F(v0,v1,v2), G(v0,v3), H(v1,v4) ----------
        {
            let atoms: &[(FunctionId, &[u32])] = &[(f, &[0, 1, 2]), (g, &[0, 3]), (h, &[1, 4])];
            assert!(
                detect_cyclic_cq(&plan_of(atoms), true).is_some(),
                "star fires WCOJ"
            );
            let mut rows = HashMap::new();
            rows.insert(
                f,
                vec![
                    (vec![1, 2, 100], 1),
                    (vec![1, 2, 101], 1),
                    (vec![1, 3, 102], 1),
                ],
            );
            rows.insert(
                g,
                vec![(vec![1, 10], 1), (vec![1, 11], 1), (vec![7, 70], 1)],
            );
            rows.insert(
                h,
                vec![(vec![2, 20], 1), (vec![3, 21], 1), (vec![9, 99], 1)],
            );
            let want = run_once(atoms, false, &rows);
            let got = run_once(atoms, true, &rows);
            assert_eq!(got, want, "STAR-3 acyclic WCOJ must match binary");
        }

        // --- Shape 1 with SELF-JOIN: F(v0,v1,v2), G(v0,v3), G(v1,v4) --------
        {
            let atoms: &[(FunctionId, &[u32])] = &[(f, &[0, 1, 2]), (g, &[0, 3]), (g, &[1, 4])];
            assert!(
                detect_cyclic_cq(&plan_of(atoms), true).is_some(),
                "self-join star fires WCOJ"
            );
            let mut rows = HashMap::new();
            rows.insert(
                f,
                vec![
                    (vec![1, 2, 100], 1),
                    (vec![1, 3, 101], 1),
                    (vec![4, 2, 102], 1),
                ],
            );
            rows.insert(
                g,
                vec![
                    (vec![1, 10], 1),
                    (vec![1, 11], 1),
                    (vec![2, 20], 1),
                    (vec![3, 21], 1),
                    (vec![4, 40], 1),
                    (vec![9, 99], 1),
                ],
            );
            let want = run_once(atoms, false, &rows);
            let got = run_once(atoms, true, &rows);
            assert_eq!(got, want, "STAR-3 self-join acyclic WCOJ must match binary");
        }

        // --- Shape 2: 4-atom STAR (3-way self-join) -------------------------
        // F(v0,v1,v2,v3), G(v0,v4), G(v1,v5), G(v2,v6)
        {
            let atoms: &[(FunctionId, &[u32])] =
                &[(f, &[0, 1, 2, 3]), (g, &[0, 4]), (g, &[1, 5]), (g, &[2, 6])];
            assert!(
                detect_cyclic_cq(&plan_of(atoms), true).is_some(),
                "4-star fires WCOJ"
            );
            let mut rows = HashMap::new();
            rows.insert(f, vec![(vec![1, 2, 3, 100], 1), (vec![1, 2, 8, 101], 1)]);
            rows.insert(
                g,
                vec![
                    (vec![1, 10], 1),
                    (vec![1, 11], 1),
                    (vec![2, 20], 1),
                    (vec![3, 30], 1),
                    (vec![8, 80], 1),
                    (vec![9, 99], 1),
                ],
            );
            let want = run_once(atoms, false, &rows);
            let got = run_once(atoms, true, &rows);
            assert_eq!(got, want, "STAR-4 acyclic WCOJ must match binary");
        }

        // --- Shape 3: 3-atom PATH  R(a,b), S(b,c), T(c,d) -------------------
        {
            let r = FunctionId::new(3);
            let s = FunctionId::new(4);
            let t = FunctionId::new(5);
            let atoms: &[(FunctionId, &[u32])] = &[(r, &[0, 1]), (s, &[1, 2]), (t, &[2, 3])];
            assert!(
                detect_cyclic_cq(&plan_of(atoms), true).is_some(),
                "path fires WCOJ"
            );
            let mut rows = HashMap::new();
            rows.insert(r, vec![(vec![1, 2], 1), (vec![1, 5], 1), (vec![9, 2], 1)]);
            rows.insert(s, vec![(vec![2, 3], 1), (vec![5, 6], 1), (vec![2, 7], 1)]);
            rows.insert(t, vec![(vec![3, 4], 1), (vec![6, 8], 1), (vec![7, 100], 1)]);
            let want = run_once(atoms, false, &rows);
            let got = run_once(atoms, true, &rows);
            assert_eq!(got, want, "PATH-3 acyclic WCOJ must match binary");
        }
    }

    /// The general WCOJ detector: fires on cyclic >=3-atom bodies, falls back to
    /// the binary chain (returns `None`) on 2-atom / acyclic / const shapes.
    #[test]
    fn detect_cyclic_cq_boundary() {
        let r = FunctionId::new(0);
        let s = FunctionId::new(1);

        // 2 atoms: never WCOJ (acyclic, no blowup).
        assert!(detect_cyclic_cq(&plan_of(&[(r, &[0, 1]), (s, &[1, 2])]), false).is_none());

        // 3-atom ACYCLIC (a path a-b-c-d): stays on the binary chain.
        let path = plan_of(&[(r, &[0, 1]), (r, &[1, 2]), (r, &[2, 3])]);
        assert!(!is_cyclic_cq(&[vec![0, 1], vec![1, 2], vec![2, 3]]));
        assert!(detect_cyclic_cq(&path, false).is_none());

        // 3-atom triangle (cyclic): fires the general path, with one delta query
        // per atom and every join var bound.
        let tri = plan_of(&[(r, &[0, 1]), (r, &[1, 2]), (r, &[2, 0])]);
        assert!(is_cyclic_cq(&[vec![0, 1], vec![1, 2], vec![2, 0]]));
        let cq = detect_cyclic_cq(&tri, false).expect("triangle is a cyclic CQ");
        assert_eq!(cq.drivers.len(), 3, "one delta query per atom");

        // 4-clique-with-payload (term-encoding shape: each edge atom carries a
        // fresh eclass + extra payload column). Cyclic core a,b,c,d; fires.
        let clique = plan_of(&[
            (r, &[0, 1, 10, 20]),
            (r, &[0, 2, 11, 21]),
            (r, &[0, 3, 12, 22]),
            (r, &[1, 2, 13, 23]),
            (r, &[1, 3, 14, 24]),
            (r, &[2, 3, 15, 25]),
        ]);
        let cqc = detect_cyclic_cq(&clique, false).expect("4-clique is a cyclic CQ");
        assert_eq!(cqc.drivers.len(), 6);
        // Each driver recovers every NON-driver atom's payload (n_atoms-1).
        for d in &cqc.drivers {
            assert_eq!(d.recovers.len(), 5);
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
