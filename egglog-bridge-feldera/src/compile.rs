//! Rule IR and DBSP circuit assembly for the Feldera backend (milestone 1).
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
//! ## Row representation
//!
//! Every relation uses a single uniform row type [`Row`] = `Tup8<u32,…>` (eight
//! `u32` slots, which is `DBData` in dbsp). egglog values are `u32`
//! ([`Value`] reps); columns beyond a relation's arity are padded with 0.
//! Using one monomorphic row type lets a *dynamically assembled* circuit wire
//! `OrdZSet<Row>` streams together without per-relation generics. (Arity is
//! capped at 8 for milestone 1; larger relations are a later-milestone concern
//! and are rejected at registration time.)
//!
//! ## Per-relation handles
//!
//! Each relation gets an input `ZSetHandle<Row>` (the host pushes the round's
//! starting facts as a delta) and an integrated output `OutputHandle` (the host
//! reads the accumulated relation back, exactly like the spike's
//! `integrate().output()` mirror). The Rust-side mirror in `lib.rs` folds the
//! consolidated output into a `HashSet<Row>` after each transaction.

use anyhow::{anyhow, Result};
use dbsp::{OrdZSet, OutputHandle, RootCircuit, Stream, ZSetHandle};
use egglog_backend_trait::{FunctionId, QueryEntry, Value};
use egglog_numeric_id::NumericId;
use hashbrown::HashMap;

/// Max number of columns a relation may have in milestone 1.
pub const MAX_ARITY: usize = 8;

/// Uniform row type for every relation. Eight `u32` slots; egglog `Value`s are
/// `u32`. Columns past a relation's arity are 0-padded.
pub type Row = dbsp::utils::Tup8<u32, u32, u32, u32, u32, u32, u32, u32>;

/// Pack a slice of `Value`s into a [`Row`] (0-padded).
pub fn pack_row(vals: &[Value]) -> Row {
    assert!(
        vals.len() <= MAX_ARITY,
        "row arity {} exceeds {MAX_ARITY}",
        vals.len()
    );
    let mut a = [0u32; MAX_ARITY];
    for (i, v) in vals.iter().enumerate() {
        a[i] = v.rep();
    }
    dbsp::utils::Tup8(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
}

/// Read column `i` (0-based) out of a [`Row`].
#[inline]
pub fn row_col(r: &Row, i: usize) -> u32 {
    let dbsp::utils::Tup8(a0, a1, a2, a3, a4, a5, a6, a7) = r;
    match i {
        0 => *a0,
        1 => *a1,
        2 => *a2,
        3 => *a3,
        4 => *a4,
        5 => *a5,
        6 => *a6,
        7 => *a7,
        _ => panic!("column index {i} out of range"),
    }
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
    fn from_entry(e: &QueryEntry) -> Self {
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

/// A head action: write a row into `func` built from these slots.
#[derive(Clone, Debug)]
pub struct HeadAction {
    pub func: FunctionId,
    pub slots: Vec<Slot>,
}

/// The compiled IR for one egglog rule.
///
/// Milestone 1 supports rules whose body is **1 or 2 table atoms** (no
/// primitives, no negation) and whose head is **one or more `set`/`insert`
/// actions**. This is enough for a single-join derivation, which is the
/// milestone's proof target.
#[derive(Clone, Debug, Default)]
pub struct RuleIr {
    pub name: String,
    pub body: Vec<BodyAtom>,
    pub head: Vec<HeadAction>,
}

/// Per-relation circuit handles produced by [`build_circuit`].
pub struct RelationHandles {
    pub input: ZSetHandle<Row>,
    pub output: OutputHandle<OrdZSet<Row>>,
}

/// Build a NON-recursive DBSP circuit for one egglog iteration over the given
/// relations and rules.
///
/// For each relation `r`:
///   `r_out = r_in  ∪  ⋃ over rules whose head targets r ( body-join → head )`
/// then `r_out.integrate().output()` so the host reads the accumulated set.
///
/// Crucially there is **no recursive scope**: `r_out` is a function of the
/// *current* inputs only, so one `transaction()` is one hop. The host feeds the
/// previous round's derived rows back as input deltas to take the next hop —
/// that Rust-side feedback loop is what realizes egglog's bounded `(run N)`.
///
/// `arities[&f]` gives the column count of relation `f`.
pub fn build_circuit(
    root: &mut RootCircuit,
    relations: &[FunctionId],
    arities: &HashMap<FunctionId, usize>,
    rules: &[RuleIr],
) -> Result<HashMap<FunctionId, RelationHandles>> {
    // 1. One input zset per relation.
    let mut inputs: HashMap<FunctionId, (Stream<RootCircuit, OrdZSet<Row>>, ZSetHandle<Row>)> =
        HashMap::new();
    for &f in relations {
        inputs.insert(f, root.add_input_zset::<Row>());
    }

    // 2. For each relation, accumulate the head-action contributions of every
    //    rule that targets it, unioned with the relation's own input stream.
    let mut head_streams: HashMap<FunctionId, Stream<RootCircuit, OrdZSet<Row>>> = HashMap::new();
    for &f in relations {
        head_streams.insert(f, inputs[&f].0.clone());
    }

    for rule in rules {
        let derived = compile_rule_body(rule, &inputs, arities)?;
        for (head_func, stream) in derived {
            let acc = head_streams
                .get(&head_func)
                .ok_or_else(|| anyhow!("rule `{}` writes unknown relation", rule.name))?
                .plus(&stream);
            head_streams.insert(head_func, acc);
        }
    }

    // 3. Integrate + output each relation so the host reads the accumulated set.
    let mut out = HashMap::new();
    for &f in relations {
        let (_, input) = inputs.remove(&f).unwrap();
        let stream = head_streams.remove(&f).unwrap();
        out.insert(
            f,
            RelationHandles {
                input,
                output: stream.integrate().output(),
            },
        );
    }
    Ok(out)
}

/// Compile a single rule's body+head into a list of `(head_func, Row stream)`
/// contributions. Returns one entry per head action.
fn compile_rule_body(
    rule: &RuleIr,
    inputs: &HashMap<FunctionId, (Stream<RootCircuit, OrdZSet<Row>>, ZSetHandle<Row>)>,
    arities: &HashMap<FunctionId, usize>,
) -> Result<Vec<(FunctionId, Stream<RootCircuit, OrdZSet<Row>>)>> {
    // A "binding environment" maps a variable id -> the join key/value column it
    // is currently materialized in. We compile the body left-to-right. After
    // processing the body, each head action projects bound variables/constants
    // into a Row.
    //
    // Milestone 1 supports 1 or 2 body atoms.
    match rule.body.len() {
        1 => {
            let atom = &rule.body[0];
            let stream = atom_stream(atom, inputs)?;
            // Bindings: var id -> column index within this atom's row.
            let env = atom_bindings(atom);
            // Each head action: map over the single body stream.
            let mut out = Vec::new();
            for head in &rule.head {
                let head_arity = *arities
                    .get(&head.func)
                    .ok_or_else(|| anyhow!("rule `{}`: head relation arity unknown", rule.name))?;
                let slots = head.slots.clone();
                let env = env.clone();
                let mapped = stream.map(move |r: &Row| {
                    project_head(&slots, head_arity, &|vid| {
                        let col = env.get(&vid).copied().expect("unbound head var");
                        row_col(r, col)
                    })
                });
                out.push((head.func, mapped));
            }
            Ok(out)
        }
        2 => {
            let a = &rule.body[0];
            let b = &rule.body[1];
            // Determine join variables: variables that appear in BOTH atoms.
            let env_a = atom_bindings(a);
            let env_b = atom_bindings(b);
            let mut join_vars: Vec<u32> = env_a
                .keys()
                .filter(|v| env_b.contains_key(*v))
                .copied()
                .collect();
            join_vars.sort_unstable();
            if join_vars.is_empty() {
                return Err(anyhow!(
                    "rule `{}`: 2-atom body with no shared variable (cartesian \
                     products are not supported in milestone 1)",
                    rule.name
                ));
            }
            // Key both atoms by the tuple of shared-variable columns (packed into
            // a single u64 for a 1- or 2-key join; milestone 1 supports up to two
            // join columns).
            let stream_a = atom_stream(a, inputs)?;
            let stream_b = atom_stream(b, inputs)?;
            let key_cols_a: Vec<usize> = join_vars.iter().map(|v| env_a[v]).collect();
            let key_cols_b: Vec<usize> = join_vars.iter().map(|v| env_b[v]).collect();

            let kca = key_cols_a.clone();
            let indexed_a = stream_a.map_index(move |r: &Row| (join_key(r, &kca), *r));
            let kcb = key_cols_b.clone();
            let indexed_b = stream_b.map_index(move |r: &Row| (join_key(r, &kcb), *r));

            let mut out = Vec::new();
            for head in &rule.head {
                let head_arity = *arities
                    .get(&head.func)
                    .ok_or_else(|| anyhow!("rule `{}`: head relation arity unknown", rule.name))?;
                let slots = head.slots.clone();
                let env_a = env_a.clone();
                let env_b = env_b.clone();
                // The join callback sees (key, row_a, row_b); resolve each head
                // var from whichever atom binds it (prefer a, fall back to b).
                let joined = indexed_a.join(&indexed_b, move |_key, ra: &Row, rb: &Row| {
                    project_head(&slots, head_arity, &|vid| {
                        if let Some(c) = env_a.get(&vid) {
                            row_col(ra, *c)
                        } else if let Some(c) = env_b.get(&vid) {
                            row_col(rb, *c)
                        } else {
                            panic!("unbound head var {vid}")
                        }
                    })
                });
                out.push((head.func, joined));
            }
            Ok(out)
        }
        n => Err(anyhow!(
            "rule `{}`: body has {n} atoms; milestone 1 supports 1 or 2",
            rule.name
        )),
    }
}

/// The input stream for a body atom's relation.
fn atom_stream(
    atom: &BodyAtom,
    inputs: &HashMap<FunctionId, (Stream<RootCircuit, OrdZSet<Row>>, ZSetHandle<Row>)>,
) -> Result<Stream<RootCircuit, OrdZSet<Row>>> {
    inputs
        .get(&atom.func)
        .map(|(s, _)| s.clone())
        .ok_or_else(|| anyhow!("body atom references unregistered relation"))
}

/// Map var id -> column index for the (last) occurrence of each variable in an
/// atom. Milestone-1 atoms are linear (no repeated variable within one atom),
/// which is the common Datalog body shape; a repeated variable would silently
/// take the last column here (an equality constraint within an atom is a later-
/// milestone feature).
fn atom_bindings(atom: &BodyAtom) -> HashMap<u32, usize> {
    let mut env = HashMap::new();
    for (i, s) in atom.slots.iter().enumerate() {
        if let Slot::Var(v) = s {
            env.insert(*v, i);
        }
    }
    env
}

/// Pack the shared-variable columns of a row into a `u64` join key (supports up
/// to two 32-bit key columns, which covers single- and double-equality joins).
#[inline]
fn join_key(r: &Row, cols: &[usize]) -> u64 {
    match cols.len() {
        1 => row_col(r, cols[0]) as u64,
        2 => ((row_col(r, cols[0]) as u64) << 32) | (row_col(r, cols[1]) as u64),
        _ => panic!("milestone 1 supports at most 2 join columns"),
    }
}

/// Project a head action's slots into a [`Row`], resolving each variable slot
/// through `resolve_var`.
fn project_head(slots: &[Slot], arity: usize, resolve_var: &dyn Fn(u32) -> u32) -> Row {
    debug_assert_eq!(slots.len(), arity, "head action arity mismatch");
    let mut a = [0u32; MAX_ARITY];
    for (i, s) in slots.iter().enumerate() {
        a[i] = match s {
            Slot::Var(v) => resolve_var(*v),
            Slot::Const(c) => *c,
        };
    }
    dbsp::utils::Tup8(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
}

// Re-export so callers can build `RuleIr` from trait `QueryEntry`s.
impl BodyAtom {
    pub fn from_entries(func: FunctionId, entries: &[QueryEntry]) -> Self {
        BodyAtom {
            func,
            slots: entries.iter().map(Slot::from_entry).collect(),
        }
    }
}

impl HeadAction {
    pub fn from_entries(func: FunctionId, entries: &[QueryEntry]) -> Self {
        HeadAction {
            func,
            slots: entries.iter().map(Slot::from_entry).collect(),
        }
    }
}
