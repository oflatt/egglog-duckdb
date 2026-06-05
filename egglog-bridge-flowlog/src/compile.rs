//! Rule IR and row representation for the FlowLog backend.
//!
//! ## The load-bearing design choice (per the milestone brief)
//!
//! egglog's `(run N)` applies a ruleset **N times with bounded extension per
//! round** — a transitive-closure rule extends **N hops, NOT to full closure**.
//! FlowLog's incremental engine (`commit()`) would, on a *recursive* `.dl`,
//! saturate to a fixed point inside one `commit()`; that is *wrong* for
//! egglog's bounded iteration (it is the same trap the Feldera Phase-0 spike's
//! recursive DD scope fell into).
//!
//! So the M1 backend compiles a **non-recursive** flowlog program
//! (`transitive_step.dl`): `hop(x,z) :- path(x,y), edge(y,z).` One `commit()`
//! performs exactly **one round** of the join over the staged delta. The host
//! (`lib.rs::run_rules`) drives `(run N)` by feeding the previous round's new
//! `path` rows and calling `commit()` once per round — N calls = N hops.
//!
//! ## Row representation
//!
//! Every relation's rows are stored in the Rust-side mirror as a variable-width
//! boxed slice of `u32` (egglog [`Value`] reps), exactly `arity` columns wide.
//! `lookup_id` / `for_each` / `table_size` read the mirror.

use egglog_backend_trait::{ExternalFunctionId, FunctionId, QueryEntry, Value};

/// Upper bound on relation arity (sanity check; the mirror row is
/// variable-width, so this is generous).
pub const MAX_ARITY: usize = 64;

/// Uniform mirror row type: a variable-width boxed slice of `u32`
/// (egglog `Value` reps), exactly `arity` columns wide.
pub type Row = Box<[u32]>;

/// How a function resolves a functional-dependency conflict (two rows sharing
/// the same key columns with different output columns). Recognized from the
/// trait `MergeFn` (see `lib.rs::add_table`). Mirrors the Feldera backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeMode {
    /// Plain relation: the whole row is the key, no output column to resolve.
    Relation,
    /// `:merge old` / `AssertEq`: keep the existing value on conflict.
    Old,
    /// `:merge new`: keep the new value on conflict.
    New,
    /// `:merge (ordering-min old new)` / `UnionId`: keep the numerically
    /// smallest value (the union-find leader). Load-bearing for M2 rebuild.
    Min,
}

/// Pack a slice of `Value`s into a [`Row`] (exactly `vals.len()` columns).
pub fn pack_row(vals: &[Value]) -> Row {
    use egglog_numeric_id::NumericId;
    assert!(
        vals.len() <= MAX_ARITY,
        "row arity {} exceeds {MAX_ARITY}",
        vals.len()
    );
    vals.iter().map(|v| v.rep()).collect()
}

/// Read column `i` (0-based) out of a [`Row`].
#[inline]
pub fn row_col(r: &Row, i: usize) -> u32 {
    r[i]
}

/// Unpack the first `arity` columns of a [`Row`] into a `Vec<Value>`.
pub fn unpack_row(r: &Row, arity: usize) -> Vec<Value> {
    use egglog_numeric_id::NumericId;
    (0..arity).map(|i| Value::new(row_col(r, i))).collect()
}

/// One column reference in a rule body atom or head action: either a bound
/// variable (identified by its [`egglog_backend_trait::VariableId`] rep) or a
/// constant value.
#[derive(Clone, Debug)]
pub enum Slot {
    Var(u32),
    Const(u32),
}

/// Resolve a [`Slot`] to a concrete `u32` value: a constant resolves to itself,
/// a variable resolves through `get` (the current binding env), or `None` if the
/// variable is unbound.
pub fn slot_lookup(s: &Slot, get: &dyn Fn(u32) -> Option<u32>) -> Option<u32> {
    match s {
        Slot::Var(v) => get(*v),
        Slot::Const(c) => Some(*c),
    }
}

impl Slot {
    pub fn from_entry(e: &QueryEntry) -> Self {
        use egglog_numeric_id::NumericId;
        match e {
            QueryEntry::Var(v) => Slot::Var(v.id.rep()),
            QueryEntry::Const { val, .. } => Slot::Const(val.rep()),
        }
    }
}

/// A body atom: a function table reference with one [`Slot`] per column.
#[derive(Clone, Debug)]
pub struct BodyAtom {
    pub func: FunctionId,
    pub slots: Vec<Slot>,
}

impl BodyAtom {
    pub fn from_entries(func: FunctionId, entries: &[QueryEntry]) -> Self {
        BodyAtom {
            func,
            slots: entries.iter().map(Slot::from_entry).collect(),
        }
    }
}

/// One operation in a rule **body**, in emission order.
#[derive(Clone, Debug)]
pub enum BodyOp {
    /// Match a table/relation atom.
    Atom(BodyAtom),
    /// Evaluate a primitive `func(args..)` (M2+; recorded but not executed in
    /// the M1 flowlog-driven path).
    #[allow(dead_code)]
    Prim {
        id: ExternalFunctionId,
        args: Vec<Slot>,
        ret: Slot,
    },
}

/// One operation in a rule **head**, in emission order.
#[derive(Clone, Debug)]
pub enum HeadOp {
    /// `(set func(key..) = val)` / relation `insert` — slots are the full row.
    Set { func: FunctionId, slots: Vec<Slot> },
    /// `(delete func(key..))` — retraction (M2+).
    #[allow(dead_code)]
    Remove { func: FunctionId, slots: Vec<Slot> },
    /// `(subsume func(key..))` (M2+).
    #[allow(dead_code)]
    Subsume { func: FunctionId, slots: Vec<Slot> },
    /// RHS function lookup binding `ret` (M2+).
    #[allow(dead_code)]
    Lookup {
        func: FunctionId,
        args: Vec<Slot>,
        ret: u32,
    },
    /// RHS primitive call binding `ret` (M2+).
    #[allow(dead_code)]
    Call {
        id: ExternalFunctionId,
        args: Vec<Slot>,
        ret: u32,
    },
    /// `(union l r)` (M2+).
    #[allow(dead_code)]
    Union { l: Slot, r: Slot },
    /// `(panic msg)`.
    #[allow(dead_code)]
    Panic(String),
}

/// The compiled IR for one egglog rule.
///
/// `body` is an ordered list of table-atom matches (and, in later milestones,
/// primitive evaluations); `head` is an ordered list of writes. In M1 the
/// FlowLog-driven `run_rules` path recognizes the canonical
/// transitive-closure-step shape (two table body atoms joined on a shared
/// variable, one `set` head) and executes it through the bundled flowlog
/// incremental engine's `commit()`. The full ordered IR is retained so M2 can
/// add the host fallback / richer rule shapes the way Feldera M3 did.
#[derive(Clone, Debug, Default)]
pub struct RuleIr {
    pub name: String,
    pub body: Vec<BodyOp>,
    pub head: Vec<HeadOp>,
}
