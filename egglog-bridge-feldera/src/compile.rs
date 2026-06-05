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

use anyhow::{anyhow, Result};
use dbsp::{OrdZSet, OutputHandle, RootCircuit, Stream, ZSetHandle};
use egglog_backend_trait::{FunctionId, QueryEntry, Value};
use egglog_numeric_id::NumericId;
use hashbrown::HashMap;

/// Max number of columns a relation may have.
pub const MAX_ARITY: usize = 8;

/// Uniform row type for every relation. Eight `u32` slots; egglog `Value`s are
/// `u32`. Columns past a relation's arity are 0-padded.
pub type Row = dbsp::utils::Tup8<u32, u32, u32, u32, u32, u32, u32, u32>;

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

/// A body filter: a comparison between two slots that must hold for the rule to
/// fire. Milestone 2 supports `!=` (the only filter the term encoder's rebuild
/// rulesets use: `(!= b c)`).
#[derive(Clone, Debug)]
pub enum Filter {
    Ne(Slot, Slot),
}

/// A head action: write (`Set`) or retract (`Delete`) a row in `func` built from
/// these slots.
#[derive(Clone, Debug)]
pub struct HeadAction {
    pub func: FunctionId,
    pub slots: Vec<Slot>,
    pub delete: bool,
}

/// The compiled IR for one egglog rule.
///
/// Supports rules whose body is **1 or 2 table atoms** plus optional `!=`
/// filters, and whose head is one or more `set`/`delete` actions. This covers a
/// single-join derivation AND the term encoder's union-find maintenance
/// rulesets (path_compress / single_parent / uf_index).
#[derive(Clone, Debug, Default)]
pub struct RuleIr {
    pub name: String,
    pub body: Vec<BodyAtom>,
    pub filters: Vec<Filter>,
    pub head: Vec<HeadAction>,
}

/// Per-relation circuit handles produced by [`build_circuit`].
///
/// Two output streams: `inserts` (rows the rules `set`) and `deletes` (rows the
/// rules `delete`). Both are integrated so the host reads the *accumulated*
/// diff for this transaction's input. The host applies them against the mirror.
pub struct RelationHandles {
    pub input: ZSetHandle<Row>,
    pub inserts: OutputHandle<OrdZSet<Row>>,
    pub deletes: OutputHandle<OrdZSet<Row>>,
}

/// Build a NON-recursive DBSP circuit for one egglog iteration over the given
/// relations and the given rule subset.
///
/// For each relation `r` and each rule whose head targets `r`, the body
/// join/filter produces rows; `set` actions feed `r`'s insert stream and
/// `delete` actions feed `r`'s delete stream. The host folds both into the
/// mirror after the transaction. There is **no recursive scope**: one
/// `transaction()` is one hop.
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

    // 2. Accumulate per-relation insert and delete contributions. We start each
    //    relation's *insert* stream empty (the mirror already holds existing
    //    rows; the insert stream carries only NEW rows derived this round) and
    //    its *delete* stream empty.
    let empty_insert = |root: &mut RootCircuit| -> Stream<RootCircuit, OrdZSet<Row>> {
        // An always-empty input stream: a zset input we never push to.
        let (s, _h) = root.add_input_zset::<Row>();
        s
    };
    let mut insert_streams: HashMap<FunctionId, Stream<RootCircuit, OrdZSet<Row>>> = HashMap::new();
    let mut delete_streams: HashMap<FunctionId, Stream<RootCircuit, OrdZSet<Row>>> = HashMap::new();
    for &f in relations {
        insert_streams.insert(f, empty_insert(root));
        delete_streams.insert(f, empty_insert(root));
    }

    for rule in rules {
        let derived = compile_rule_body(rule, &inputs, arities)?;
        for (head_func, delete, stream) in derived {
            let map = if delete {
                &mut delete_streams
            } else {
                &mut insert_streams
            };
            let acc = map
                .get(&head_func)
                .ok_or_else(|| anyhow!("rule `{}` writes unknown relation", rule.name))?
                .plus(&stream);
            map.insert(head_func, acc);
        }
    }

    // 3. Integrate + output the insert and delete diff streams per relation.
    let mut out = HashMap::new();
    for &f in relations {
        let (_, input) = inputs.remove(&f).unwrap();
        let ins = insert_streams.remove(&f).unwrap();
        let del = delete_streams.remove(&f).unwrap();
        out.insert(
            f,
            RelationHandles {
                input,
                inserts: ins.integrate().output(),
                deletes: del.integrate().output(),
            },
        );
    }
    Ok(out)
}

/// Compile a single rule's body+filters+head into a list of
/// `(head_func, is_delete, Row stream)` contributions, one per head action.
fn compile_rule_body(
    rule: &RuleIr,
    inputs: &HashMap<FunctionId, (Stream<RootCircuit, OrdZSet<Row>>, ZSetHandle<Row>)>,
    arities: &HashMap<FunctionId, usize>,
) -> Result<Vec<(FunctionId, bool, Stream<RootCircuit, OrdZSet<Row>>)>> {
    match rule.body.len() {
        1 => {
            let atom = &rule.body[0];
            let stream = atom_stream(atom, inputs)?;
            let env = atom_bindings(atom);
            let filters = rule.filters.clone();
            let env_f = env.clone();
            // Apply filters to the single body stream.
            let stream = if filters.is_empty() {
                stream
            } else {
                stream.filter(move |r: &Row| {
                    eval_filters(&filters, &|vid| {
                        row_col(r, *env_f.get(&vid).expect("filter var unbound"))
                    })
                })
            };
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
                out.push((head.func, head.delete, mapped));
            }
            Ok(out)
        }
        2 => {
            let a = &rule.body[0];
            let b = &rule.body[1];
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
                     products are not supported)",
                    rule.name
                ));
            }
            let stream_a = atom_stream(a, inputs)?;
            let stream_b = atom_stream(b, inputs)?;
            let key_cols_a: Vec<usize> = join_vars.iter().map(|v| env_a[v]).collect();
            let key_cols_b: Vec<usize> = join_vars.iter().map(|v| env_b[v]).collect();
            // `stream_a`/`stream_b` are re-indexed per head action below (each
            // head action gets its own `map_index`); the join itself is shared
            // structurally but DBSP dedups identical operators.

            let mut out = Vec::new();
            for head in &rule.head {
                let head_arity = *arities
                    .get(&head.func)
                    .ok_or_else(|| anyhow!("rule `{}`: head relation arity unknown", rule.name))?;
                let slots = head.slots.clone();
                let filters = rule.filters.clone();
                let env_a = env_a.clone();
                let env_b = env_b.clone();
                let kca = key_cols_a.clone();
                let kcb = key_cols_b.clone();
                let ia = stream_a.map_index(move |r: &Row| (join_key(r, &kca), *r));
                let ib = stream_b.map_index(move |r: &Row| (join_key(r, &kcb), *r));
                // The join produces `Tup2(filter_ok, head_row)`; we drop pairs
                // whose filter failed, then project to the head row. Filters
                // reference *body* variables (resolved through ra/rb), so they
                // must be evaluated inside the join, not on the projected head.
                // `u8` (1=keep, 0=drop) is `DBData`; a raw `bool`/tuple is not
                // guaranteed to be, so we use `Tup2<u8, Row>` as the carrier.
                use dbsp::utils::Tup2;
                let joined = ia
                    .join(&ib, move |_key, ra: &Row, rb: &Row| {
                        let resolve = |vid: u32| -> u32 {
                            if let Some(c) = env_a.get(&vid) {
                                row_col(ra, *c)
                            } else if let Some(c) = env_b.get(&vid) {
                                row_col(rb, *c)
                            } else {
                                panic!("unbound var {vid}")
                            }
                        };
                        let ok = u8::from(eval_filters(&filters, &resolve));
                        let row = project_head(&slots, head_arity, &resolve);
                        Tup2(ok, row)
                    })
                    .filter(|Tup2(ok, _row): &Tup2<u8, Row>| *ok == 1)
                    .map(|Tup2(_ok, row): &Tup2<u8, Row>| *row);
                out.push((head.func, head.delete, joined));
            }
            Ok(out)
        }
        n => Err(anyhow!(
            "rule `{}`: body has {n} atoms; supported: 1 or 2",
            rule.name
        )),
    }
}

/// Evaluate the rule's filters against a variable resolver. All filters must
/// hold (conjunction).
fn eval_filters(filters: &[Filter], resolve: &dyn Fn(u32) -> u32) -> bool {
    filters.iter().all(|f| match f {
        Filter::Ne(l, r) => slot_val(l, resolve) != slot_val(r, resolve),
    })
}

#[inline]
fn slot_val(s: &Slot, resolve: &dyn Fn(u32) -> u32) -> u32 {
    match s {
        Slot::Var(v) => resolve(*v),
        Slot::Const(c) => *c,
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

/// Map var id -> column index for each variable in an atom.
fn atom_bindings(atom: &BodyAtom) -> HashMap<u32, usize> {
    let mut env = HashMap::new();
    for (i, s) in atom.slots.iter().enumerate() {
        if let Slot::Var(v) = s {
            env.insert(*v, i);
        }
    }
    env
}

/// Pack the shared-variable columns of a row into a `u64` join key.
#[inline]
fn join_key(r: &Row, cols: &[usize]) -> u64 {
    match cols.len() {
        1 => row_col(r, cols[0]) as u64,
        2 => ((row_col(r, cols[0]) as u64) << 32) | (row_col(r, cols[1]) as u64),
        _ => panic!("supported: at most 2 join columns"),
    }
}

/// Project a head action's slots into a [`Row`].
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

impl BodyAtom {
    pub fn from_entries(func: FunctionId, entries: &[QueryEntry]) -> Self {
        BodyAtom {
            func,
            slots: entries.iter().map(Slot::from_entry).collect(),
        }
    }
}

impl HeadAction {
    pub fn from_entries(func: FunctionId, entries: &[QueryEntry], delete: bool) -> Self {
        HeadAction {
            func,
            slots: entries.iter().map(Slot::from_entry).collect(),
            delete,
        }
    }
}
