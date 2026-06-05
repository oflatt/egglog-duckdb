//! Rule IR and DBSP circuit assembly for the Feldera backend.
//!
//! ## The load-bearing design choice (per the milestone brief)
//!
//! egglog's `(run N)` applies a ruleset **N times with rebuild between
//! iterations** — so a transitive-closure rule extends **N hops, NOT to full
//! closure**. The Phase-0 spike used DBSP's RECURSIVE scope, which saturates to
//! a fixed point inside one transaction; that is *wrong* for egglog's bounded
//! iteration.
//!
//! So this backend compiles user rules into a **non-recursive** circuit: one
//! `transaction()` performs exactly **one round** of rule application over the
//! current relation contents. The frontend's existing iteration loop drives
//! `(run N)` by calling `run_rules` N times — and because the circuit does one
//! hop per call, `(run 1)` and `(run 3)` produce different, bounded results.
//!
//! ## Milestone 2: ruleset-scoped execution + retraction-rebuild
//!
//! Each `run_rules(&[RuleId])` call runs a **subset** of rules. A DBSP circuit
//! is a static monolithic graph, so we build a *per-subset* circuit keyed by the
//! sorted rule-id list (cached). The host pushes the current mirror as the
//! circuit's input, runs one transaction, and reads back two diff streams per
//! relation: an **insert** stream and a **delete** stream. The host then folds
//! `(old ∪ inserts) \ deletes` into the mirror and resolves FD conflicts via the
//! relation's merge mode. Deletes are the term encoder's `(delete ...)` actions
//! (rebuild's retraction half); inserts are `(set ...)`.
//!
//! ## Row representation
//!
//! Every relation uses a single uniform row type [`Row`] = `Tup8<u32,…>` (eight
//! `u32` slots). egglog values are `u32` ([`Value`] reps); columns beyond a
//! relation's arity are padded with 0.

use egglog_backend_trait::{ExternalFunctionId, FunctionId, QueryEntry, Value};
use egglog_numeric_id::NumericId;

/// Max number of columns a relation may have.
/// Upper bound on relation arity (sanity check; the host row representation is
/// variable-width so this is generous — the term encoder's widest view tables
/// stay well under it).
pub const MAX_ARITY: usize = 64;

/// Uniform row type for every relation: a variable-width boxed slice of `u32`
/// (egglog `Value` reps), exactly `arity` columns wide. Milestones 1–2 used a
/// fixed `Tup8` because rows flowed through DBSP zsets; Milestone 3 evaluates
/// rules host-side (primitives must invoke through the embedded `Database`), so
/// the row is a plain hashable slice with no arity cap of 8.
pub type Row = Box<[u32]>;

/// How a function resolves a functional-dependency conflict (two rows sharing
/// the same key columns with different output columns). Recognized from the
/// trait `MergeFn` / the term encoder's `:merge` clause (see `lib.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeMode {
    /// Plain relation: the whole row is the key, no output column to resolve.
    Relation,
    /// `:merge old` / `AssertEq`: keep the existing value on conflict.
    Old,
    /// `:merge new`: keep the new value on conflict.
    New,
    /// `:merge (ordering-min old new)`: keep the numerically smallest value.
    /// This is the term encoder's `@uff` (uf-index) merge — load-bearing for
    /// rebuild: the function must hold the *minimum* representative per child.
    Min,
}

/// Pack a slice of `Value`s into a [`Row`] (exactly `vals.len()` columns).
pub fn pack_row(vals: &[Value]) -> Row {
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

impl Slot {
    pub fn from_entry(e: &QueryEntry) -> Self {
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

// ---------------------------------------------------------------------------
// Milestone 3: ordered, fully-general rule IR (host interpreter path)
// ---------------------------------------------------------------------------

/// One operation in a rule **body**, in emission order. The host interpreter
/// (`lib.rs::interpret_rule`) processes these left to right, threading a
/// variable→value binding environment.
#[derive(Clone, Debug)]
pub enum BodyOp {
    /// Match a table/relation atom: bind its variables (or constrain already-
    /// bound vars / constants) against every row of `func`.
    Atom(BodyAtom),
    /// Evaluate a primitive `func(args..)`. The last entry (`ret`) is the
    /// primitive's return slot: if it is an as-yet-unbound variable it is bound
    /// to the result; otherwise the result must equal the slot's value (a
    /// guard). `None` result (primitive failure, e.g. `!=` of equal args) prunes
    /// the match.
    Prim {
        id: ExternalFunctionId,
        args: Vec<Slot>,
        ret: Slot,
    },
}

/// One operation in a rule **head**, in emission order.
#[derive(Clone, Debug)]
pub enum HeadOp {
    /// `(set func(key..) = val)` / relation `insert` — the slots are the full
    /// row (inputs then output).
    Set { func: FunctionId, slots: Vec<Slot> },
    /// `(delete func(key..))` — slots address the row to retract by key.
    Remove { func: FunctionId, slots: Vec<Slot> },
    /// `(subsume func(key..))`.
    Subsume { func: FunctionId, slots: Vec<Slot> },
    /// RHS function lookup `(let v (func args..))`: look up the output column of
    /// `func` for the given input slots, binding `ret` to it (creating the row
    /// with a fresh id if absent — eq-sort constructor semantics).
    Lookup {
        func: FunctionId,
        args: Vec<Slot>,
        ret: u32,
    },
    /// RHS primitive call `(let v (prim args..))`: invoke the primitive and bind
    /// `ret` to the result.
    Call {
        id: ExternalFunctionId,
        args: Vec<Slot>,
        ret: u32,
    },
    /// `(union l r)` — merge two eq-sort ids.
    Union { l: Slot, r: Slot },
    /// `(panic msg)`.
    Panic(String),
}

/// The compiled IR for one egglog rule (Milestone 3 ordered form).
///
/// `body` is an ordered list of table-atom matches and primitive evaluations;
/// `head` is an ordered list of writes / lookups / unions. The host interpreter
/// runs the body as a nested-loop join over the current relation mirror,
/// evaluating primitives via the embedded `Database`, then applies the head
/// actions for each surviving binding.
#[derive(Clone, Debug, Default)]
pub struct RuleIr {
    pub name: String,
    pub body: Vec<BodyOp>,
    pub head: Vec<HeadOp>,
}

impl BodyAtom {
    pub fn from_entries(func: FunctionId, entries: &[QueryEntry]) -> Self {
        BodyAtom {
            func,
            slots: entries.iter().map(Slot::from_entry).collect(),
        }
    }
}

/// Resolve a [`Slot`] to a concrete `u32` value given a variable→value lookup.
/// Returns `None` if the slot is an unbound variable.
pub fn slot_lookup(s: &Slot, get: &dyn Fn(u32) -> Option<u32>) -> Option<u32> {
    match s {
        Slot::Var(v) => get(*v),
        Slot::Const(c) => Some(*c),
    }
}
