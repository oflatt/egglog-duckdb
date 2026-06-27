use std::hash::Hasher;

use crate::Context;
use crate::{
    core::{CoreActionContext, CoreRule, GenericActionsExt, ResolvedCall},
    *,
};
use ast::{ResolvedAction, ResolvedExpr, ResolvedFact, ResolvedRule, ResolvedVar, Rule};
use core_relations::ExternalFunction;
use egglog_ast::generic_ast::GenericAction;
use egglog_bridge::ActionRegistry;
use enum_map::EnumMap;
use std::sync::{Arc, RwLock};

// `ExternalFunction` wrapper for `PurePrim`. Holds the primitive
// directly so the dispatch chain `external_funcs[id].invoke(...)` →
// `T::apply(...)` is just one vtable hop plus a direct call — no
// closure indirection that defeats inlining.
#[derive(Clone)]
struct PurePrimWrapper<T> {
    prim: T,
    /// The call-site [`Context`] this wrapper stamps onto the
    /// `PureState` before dispatching. `register_per_context` commits
    /// one wrapper per valid `Context` for the trait, so the
    /// typechecker's pick at each call site is encoded directly here.
    ctx: Context,
}

impl<T: PurePrim + Clone> ExternalFunction for PurePrimWrapper<T> {
    fn invoke(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        self.prim.apply(PureState::wrap(exec_state, self.ctx), args)
    }
}

// `ExternalFunction` wrapper for primitives that need the
// `ActionRegistry` (`ReadPrim`, `WritePrim`, `FullPrim`). One generic
// over the `Wrap` strategy that knows how to construct the right
// state type and dispatch to the primitive's `apply`.
#[derive(Clone)]
struct RegistryPrimWrapper<T, S> {
    prim: T,
    registry: Arc<RwLock<ActionRegistry>>,
    /// Stamped onto the state wrapper.
    ctx: Context,
    _wrap: std::marker::PhantomData<fn() -> S>,
}

trait RegistryWrap<T>: Clone + Send + Sync {
    fn invoke(
        prim: &T,
        exec_state: &mut ExecutionState,
        ctx: Context,
        args: &[Value],
        registry: &ActionRegistry,
    ) -> Option<Value>;
}

#[derive(Clone)]
struct WrapRead;
impl<T: ReadPrim> RegistryWrap<T> for WrapRead {
    #[inline]
    fn invoke(
        prim: &T,
        exec_state: &mut ExecutionState,
        ctx: Context,
        args: &[Value],
        registry: &ActionRegistry,
    ) -> Option<Value> {
        prim.apply(ReadState::wrap(exec_state, registry, ctx), args)
    }
}
#[derive(Clone)]
struct WrapWrite;
impl<T: WritePrim> RegistryWrap<T> for WrapWrite {
    #[inline]
    fn invoke(
        prim: &T,
        exec_state: &mut ExecutionState,
        ctx: Context,
        args: &[Value],
        registry: &ActionRegistry,
    ) -> Option<Value> {
        prim.apply(WriteState::wrap(exec_state, registry, ctx), args)
    }
}
#[derive(Clone)]
struct WrapFull;
impl<T: FullPrim> RegistryWrap<T> for WrapFull {
    #[inline]
    fn invoke(
        prim: &T,
        exec_state: &mut ExecutionState,
        ctx: Context,
        args: &[Value],
        registry: &ActionRegistry,
    ) -> Option<Value> {
        prim.apply(FullState::wrap(exec_state, registry, ctx), args)
    }
}

impl<T: Clone + Send + Sync + 'static, S: RegistryWrap<T> + 'static> ExternalFunction
    for RegistryPrimWrapper<T, S>
{
    fn invoke(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        let registry = self.registry.read().unwrap();
        S::invoke(&self.prim, exec_state, self.ctx, args, &registry)
    }
}

// Placeholder `ExternalFunction` wrapper used on the duckdb backend for
// registry-backed primitives (`ReadPrim`/`WritePrim`/`FullPrim`).
//
// DuckDB has no in-memory `ActionRegistry`: primitive *calls* are
// compiled to SQL by the duck rule-builder, which maps each
// `ExternalFunctionId` to its user-visible name via the
// `set_external_func_name` side-channel (set in `register_per_context`).
// The wrapper's `invoke` is therefore never reached on duckdb — we only
// need a registered id + name. This placeholder satisfies the
// `register_external_func` API without touching `action_registry()`
// (which is `unimplemented!()` on duckdb).
#[derive(Clone)]
struct DuckPlaceholderPrimWrapper {
    name: String,
}

impl ExternalFunction for DuckPlaceholderPrimWrapper {
    fn invoke(&self, _exec_state: &mut ExecutionState, _args: &[Value]) -> Option<Value> {
        unreachable!(
            "registry-backed primitive `{}` was invoked through its \
             ExternalFunction wrapper on the duckdb backend; duckdb compiles \
             primitive calls to SQL and should never reach this path",
            self.name
        )
    }
}

// Registry-free `ExternalFunction` wrapper for registry-backed primitives on
// backends without an in-memory `ActionRegistry` (`supports_action_registry()
// == false`: duckdb / flowlog / feldera). Unlike `RegistryPrimWrapper` it does
// not build a `ReadState`/`FullState` (which require the `ActionRegistry` those
// backends lack). On duckdb the wrapper is never invoked (primitive calls are
// compiled to SQL). On flowlog/feldera the interpreter DOES invoke it, so it
// must produce a value:
//   - `get-size!`: sum a frontend-maintained size snapshot
//     (`get_size_snapshot`), honoring the explicit `@<F>View` name filter the
//     term encoder emits (or all entries when called with no args). This is the
//     mode-invariant canonical egraph size (see `instrument_get_size`).
//   - any other registry-backed primitive: not yet supported on these backends
//     — panic with a clear message rather than silently returning a wrong value.
#[derive(Clone)]
struct RegistryFreePrimWrapper {
    name: String,
    get_size_snapshot: Arc<RwLock<HashMap<String, i64>>>,
}

impl ExternalFunction for RegistryFreePrimWrapper {
    fn invoke(&self, exec_state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        if self.name == "get-size!" {
            let snapshot = self.get_size_snapshot.read().unwrap();
            let total: i64 = if args.is_empty() {
                snapshot.values().copied().sum()
            } else {
                args.iter()
                    .map(|v| {
                        let name = exec_state.base_values().unwrap::<crate::sort::S>(*v).0;
                        snapshot.get(&name).copied().unwrap_or(0)
                    })
                    .sum()
            };
            return Some(exec_state.base_values().get::<i64>(total));
        }
        unreachable!(
            "registry-backed primitive `{}` has no registry-free implementation on \
             this backend (only `get-size!` is supported); it was invoked through \
             RegistryFreePrimWrapper",
            self.name
        )
    }
}

#[derive(Clone, Debug)]
pub struct FuncType {
    pub name: String,
    pub subtype: FunctionSubtype,
    pub input: Vec<ArcSort>,
    /// The first (primary) output sort. See [`FuncType::outputs`].
    pub output: ArcSort,
    /// Additional output sorts for tuple-output functions. Empty for ordinary functions.
    pub extra_outputs: Vec<ArcSort>,
}

impl FuncType {
    /// All output (value-column) sorts: the primary output followed by any extras.
    pub fn outputs(&self) -> impl Iterator<Item = &ArcSort> {
        std::iter::once(&self.output).chain(self.extra_outputs.iter())
    }

    /// The number of output (value) columns.
    pub fn num_outputs(&self) -> usize {
        1 + self.extra_outputs.len()
    }

    /// Whether this function has more than one output column.
    pub fn is_tuple_output(&self) -> bool {
        !self.extra_outputs.is_empty()
    }
}

impl PartialEq for FuncType {
    fn eq(&self, other: &Self) -> bool {
        if self.name == other.name
            && self.subtype == other.subtype
            && self.num_outputs() == other.num_outputs()
            && self
                .outputs()
                .zip(other.outputs())
                .all(|(a, b)| a.name() == b.name())
        {
            if self.input.len() != other.input.len() {
                return false;
            }
            for (a, b) in self.input.iter().zip(other.input.iter()) {
                if a.name() != b.name() {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
}

impl Eq for FuncType {}

impl Hash for FuncType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.subtype.hash(state);
        for out in self.outputs() {
            out.name().hash(state);
        }
        for inp in &self.input {
            inp.name().hash(state);
        }
    }
}
/// Validators take a termdag and arguments (as TermIds) and return
/// a newly computed TermId if the primitive application is valid,
/// or None if it is invalid.
pub type PrimitiveValidator = Arc<dyn Fn(&mut TermDag, &[TermId]) -> Option<TermId> + Send + Sync>;

#[derive(Clone)]
pub struct PrimitiveWithId {
    pub(crate) primitive: Arc<dyn Primitive>,
    pub(crate) validator: Option<PrimitiveValidator>,
    /// Runtime entrypoints for the contexts this primitive is valid in.
    /// The primitive definition is stored once, while each context keeps
    /// its own backend id so higher-order dispatch can still recover the
    /// application context at runtime.
    pub(crate) context_ids: EnumMap<Context, Option<ExternalFunctionId>>,
}

impl PrimitiveWithId {
    /// Takes the full signature of a primitive (both input and output types).
    /// Returns whether the primitive is compatible with this signature.
    pub fn accept(&self, tys: &[Arc<dyn Sort>], typeinfo: &TypeInfo) -> bool {
        let mut constraints = vec![];
        let lits: Vec<_> = (0..tys.len())
            .map(|i| AtomTerm::Literal(Span::Panic, Literal::Int(i as i64)))
            .collect();
        for (lit, ty) in lits.iter().zip(tys.iter()) {
            constraints.push(constraint::assign(lit.clone(), ty.clone()))
        }
        constraints.extend(
            self.primitive
                .get_type_constraints(&Span::Panic)
                .get(&lits, typeinfo),
        );
        let problem = Problem {
            constraints,
            range: HashSet::default(),
        };
        problem.solve(|sort| sort.name()).is_ok()
    }
}

impl Debug for PrimitiveWithId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Prim({})", self.primitive.name())
    }
}

/// Placeholder primitive for a UF-backed function's `:canon-prim`.
///
/// Registered at typecheck time so the name can be referenced and typechecked
/// (signature `(S S) -> S` so a call `(canon x)` resolves to one argument plus
/// an output of the eqsort `S`). Its backend id is rebound to the real
/// union-find canonicalizer external function at backend-build time
/// (`declare_function`), so its `apply` is never actually invoked.
#[derive(Clone)]
struct UfCanonPrimitive {
    name: String,
    sort: ArcSort,
}

impl Primitive for UfCanonPrimitive {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        constraint::AllEqualTypeConstraint::new(&self.name, span.clone())
            .with_all_arguments_sort(self.sort.clone())
            .with_exact_length(2)
            .into_box()
    }
}

impl PurePrim for UfCanonPrimitive {
    fn apply<'a, 'db>(&self, _state: PureState<'a, 'db>, _args: &[Value]) -> Option<Value> {
        panic!("uf canon primitive should be bound to an external function");
    }
}

/// Proof-mode find-or-refl primitive for canonicalize-at-creation (output column).
///
/// Signature `(S Proof) -> @UFPair_S`. Resolves `(@UF_Sf x)` — the flat UF index
/// frozen at the last completed rebuild — to its stored `(pair leader proof)`.
/// On a lookup MISS (a just-minted term whose UF index row does not exist yet),
/// it returns the *refl pair* `(pair x p)`, where `p` is the caller-supplied
/// proof of `x = x` (the term's `term_proof`, threaded in as an argument — a
/// local rule variable — rather than read from the table, since that set may be
/// staged within the same action block and not yet committed). No row is
/// inserted, so view tuple counts stay bit-exact with the full-rebuild
/// reference. The encoder feeds the pair's proof into the view proof via `Sym`.
#[derive(Clone)]
struct UfReflPrimitive {
    name: String,
    /// The eq-sort `S` (the first argument and the pair's first element).
    eq_sort: ArcSort,
    /// The proof sort (the second argument and the pair's second element).
    proof_sort: ArcSort,
    /// The `@UFPair_S` sort (the output), used to type the primitive.
    pair_sort: ArcSort,
    /// Whether congruence rebuilds the pair's first element (the eq-sort id).
    do_rebuild_first: bool,
    /// Whether congruence rebuilds the pair's second element (the proof).
    do_rebuild_second: bool,
    /// The flat UF index function name `@UF_Sf`.
    uf_function: String,
}

impl Primitive for UfReflPrimitive {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            &self.name,
            vec![
                self.eq_sort.clone(),
                self.proof_sort.clone(),
                self.pair_sort.clone(),
            ],
            span.clone(),
        )
        .into_box()
    }
}

impl ReadPrim for UfReflPrimitive {
    fn apply<'a, 'db>(&self, state: ReadState<'a, 'db>, args: &[Value]) -> Option<Value> {
        use crate::exec_state::{Core, Read};
        let x = args[0];
        let fallback_proof = args[1];
        // Hit: the UF index already has a (leader, proof) pair for x.
        if let Some(pair) = state.lookup(&self.uf_function, &[x]) {
            return Some(pair);
        }
        // Miss: x is its own leader; build the refl pair (pair x fallback_proof).
        let mut state = state;
        Some(state.register_container(crate::sort::PairContainer::new(
            self.do_rebuild_first,
            self.do_rebuild_second,
            x,
            fallback_proof,
        )))
    }
}

/// Proof-mode find-leader primitive for canonicalize-at-creation (child args).
///
/// Signature `(S) -> S`. Resolves `(@UF_Sf x)` to its stored leader
/// (`pair-first`), or — on a miss — to `x` itself (x is its own leader). No row
/// is inserted. This is the proof-mode analogue of the term-mode `@UF_Sf`
/// `DefaultVal::Identity` lookup-or-self: it returns the bare leader id for a
/// child argument, where only the canonical leader is needed (the view proof
/// never relates children to their leaders — see `add_term_and_view`).
#[derive(Clone)]
struct UfFindLeaderPrimitive {
    name: String,
    /// The eq-sort `S` (argument and output).
    eq_sort: ArcSort,
    /// The flat UF index function name `@UF_Sf`.
    uf_function: String,
}

impl Primitive for UfFindLeaderPrimitive {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        constraint::AllEqualTypeConstraint::new(&self.name, span.clone())
            .with_all_arguments_sort(self.eq_sort.clone())
            .with_exact_length(2)
            .into_box()
    }
}

impl ReadPrim for UfFindLeaderPrimitive {
    fn apply<'a, 'db>(&self, state: ReadState<'a, 'db>, args: &[Value]) -> Option<Value> {
        use crate::exec_state::{Core, Read};
        let x = args[0];
        // Hit: project the leader (pair-first) from the stored (leader, proof).
        if let Some(pair) = state.lookup(&self.uf_function, &[x]) {
            // The pair container's first element is the leader.
            let container = state.value_to_container::<crate::sort::PairContainer>(pair)?;
            return Some(container.first);
        }
        // Miss: x is its own leader.
        Some(x)
    }
}

/// Stores resolved typechecking information.
#[derive(Clone, Default)]
pub struct TypeInfo {
    mksorts: HashMap<String, MkSort>,
    // TODO(yz): I want to get rid of this as now we have user-defined primitives and constraint based type checking
    reserved_primitives: HashSet<&'static str>,
    pub(crate) sorts: HashMap<String, Arc<dyn Sort>>,
    primitives: HashMap<String, Vec<PrimitiveWithId>>,
    func_types: HashMap<String, FuncType>,
    pub(crate) global_sorts: HashMap<String, ArcSort>,
    /// Sorts that do not allow union (e.g., from `:no-union` sorts or relations).
    pub(crate) non_unionable_sorts: HashSet<String>,
}

// These methods need to be on the `EGraph` in order to
// register sorts and primitives with the backend.
impl EGraph {
    /// Add a user-defined sort to the e-graph.
    ///
    /// Also look at [`prelude::add_base_sort`] for a convenience method for adding user-defined sorts
    pub fn add_sort<S: Sort + 'static>(&mut self, sort: S, span: Span) -> Result<(), TypeError> {
        self.add_arcsort(Arc::new(sort), span)
    }

    /// Declare a sort. This corresponds to the `sort` keyword in egglog.
    /// It can either declares a new [`EqSort`] if `presort_and_args` is not provided,
    /// or an instantiation of a presort (e.g., containers like `Vec`).
    pub fn declare_sort(
        &mut self,
        name: impl Into<String>,
        presort_and_args: &Option<(String, Vec<Expr>)>,
        span: Span,
    ) -> Result<(), TypeError> {
        let name = name.into();
        if self.type_info.func_types.contains_key(&name) {
            return Err(TypeError::FunctionAlreadyBound(name, span));
        }

        let sort = match presort_and_args {
            None => Arc::new(EqSort { name }),
            Some((presort, args)) => {
                if let Some(mksort) = self.type_info.mksorts.get(presort) {
                    mksort(&mut self.type_info, name, args)?
                } else {
                    return Err(TypeError::PresortNotFound(presort.clone(), span));
                }
            }
        };

        self.add_arcsort(sort, span)
    }

    /// Add a user-defined sort to the e-graph.
    pub fn add_arcsort(&mut self, sort: ArcSort, span: Span) -> Result<(), TypeError> {
        sort.register_type(&mut *self.backend);

        let name = sort.name();
        match self.type_info.sorts.entry(name.to_owned()) {
            HEntry::Occupied(_) => Err(TypeError::SortAlreadyBound(name.to_owned(), span)),
            HEntry::Vacant(e) => {
                e.insert(sort.clone());
                sort.register_primitives(self);
                Ok(())
            }
        }
    }

    /// Register a [`PurePrim`]. Pass `None` for the validator if not
    /// using the proof checker.
    ///
    /// Pick the trait whose state wrapper matches the body's needs:
    /// [`PurePrim`] for pure ops, [`WritePrim`] for writes,
    /// [`ReadPrim`] for table reads, [`FullPrim`] for both. The Rust
    /// type checker enforces the body only uses methods the chosen
    /// state allows.
    pub fn add_pure_primitive<T>(&mut self, x: T, validator: Option<PrimitiveValidator>)
    where
        T: PurePrim + Clone,
    {
        if let Some(orig) = self.proof_state.original_typechecking.as_mut() {
            orig.add_pure_primitive(x.clone(), validator.clone());
        }
        self.register_per_context(x, validator, PureState::valid_contexts(), true, |x, ctx| {
            Box::new(PurePrimWrapper { prim: x, ctx })
        });
    }

    /// Register a [`WritePrim`]. Pass `None` for the validator if not
    /// using the proof checker.
    pub fn add_write_primitive<T>(&mut self, x: T, validator: Option<PrimitiveValidator>)
    where
        T: WritePrim + Clone,
    {
        if let Some(orig) = self.proof_state.original_typechecking.as_mut() {
            orig.add_write_primitive(x.clone(), validator.clone());
        }
        self.register_registry_primitive::<T, WrapWrite>(
            x,
            validator,
            WriteState::valid_contexts(),
        );
    }

    /// Register a [`ReadPrim`]. Pass `None` for the validator if not
    /// using the proof checker.
    pub fn add_read_primitive<T>(&mut self, x: T, validator: Option<PrimitiveValidator>)
    where
        T: ReadPrim + Clone,
    {
        if let Some(orig) = self.proof_state.original_typechecking.as_mut() {
            orig.add_read_primitive(x.clone(), validator.clone());
        }
        self.register_registry_primitive::<T, WrapRead>(x, validator, ReadState::valid_contexts());
    }

    /// Register a proof-mode `@UFPair_S` pair sort's canonicalize-at-creation
    /// primitives: `refl_prim` (`(S Proof) -> @UFPair_S`) and `find_leader_prim`
    /// (`(S) -> S`), both reading the flat UF index `uf_function` (`@UF_Sf`).
    /// Called from the `Sort` typecheck arm with the names from the sort's
    /// `:internal-uf-pair-prims` annotation (so it is round-trip safe).
    fn register_uf_refl_primitives(
        &mut self,
        pair_sort_name: &str,
        uf_function: &str,
        refl_prim: &str,
        find_leader_prim: &str,
    ) {
        // Idempotent: typecheck may revisit a sort declaration; only register the
        // primitives once.
        if self.type_info.is_primitive(refl_prim) {
            return;
        }
        let Some(pair_sort) = self.type_info.get_sort_by_name(pair_sort_name).cloned() else {
            return;
        };
        // The pair's inner sorts are `(S, Proof)`. Derive the per-element rebuild
        // flags exactly as the `pair` primitive does (`PairSort::make_container`):
        // congruence rebuilds eq-sort / eq-container-sort elements.
        let inner = pair_sort.inner_sorts();
        let [first, second] = inner.as_slice() else {
            return;
        };
        let eq_sort = first.clone();
        let proof_sort = second.clone();
        let do_rebuild_first = first.is_eq_sort() || first.is_eq_container_sort();
        let do_rebuild_second = second.is_eq_sort() || second.is_eq_container_sort();
        self.add_read_primitive(
            UfReflPrimitive {
                name: refl_prim.to_string(),
                eq_sort: eq_sort.clone(),
                proof_sort,
                pair_sort,
                do_rebuild_first,
                do_rebuild_second,
                uf_function: uf_function.to_string(),
            },
            None,
        );
        self.add_read_primitive(
            UfFindLeaderPrimitive {
                name: find_leader_prim.to_string(),
                eq_sort,
                uf_function: uf_function.to_string(),
            },
            None,
        );
    }

    /// Register a [`FullPrim`]. Pass `None` for the validator if not
    /// using the proof checker.
    pub fn add_full_primitive<T>(&mut self, x: T, validator: Option<PrimitiveValidator>)
    where
        T: FullPrim + Clone,
    {
        if let Some(orig) = self.proof_state.original_typechecking.as_mut() {
            orig.add_full_primitive(x.clone(), validator.clone());
        }
        self.register_registry_primitive::<T, WrapFull>(x, validator, FullState::valid_contexts());
    }

    fn register_registry_primitive<T, S>(
        &mut self,
        x: T,
        validator: Option<PrimitiveValidator>,
        valid_ctxs: &[Context],
    ) where
        T: Primitive + Clone,
        S: RegistryWrap<T> + 'static,
    {
        // Backends without an in-memory `ActionRegistry`
        // (`supports_action_registry() == false`: duckdb, flowlog, feldera)
        // cannot back a `RegistryPrimWrapper` (its construction clones
        // `action_registry()`, which is `unimplemented!()` there). Register a
        // registry-free wrapper instead — we still need the primitive's
        // id + name + type registered. On duckdb the primitive runs as
        // compiled SQL (the wrapper's `invoke` is never reached); on
        // flowlog/feldera the wrapper IS invoked by the interpreter, so
        // `RegistryFreePrimWrapper` dispatches it without an `ActionRegistry`
        // (e.g. `get-size!` reads the size snapshot; see `snapshot_get_size`).
        if !self.backend.supports_action_registry() {
            // DuckDB compiles primitive calls to SQL (the wrapper's `invoke` is
            // never reached — `get-size!` becomes the `__egglog_get_size()` UDF),
            // so it keeps the `unreachable!()` placeholder unchanged. FlowLog /
            // Feldera DO invoke the wrapper through their interpreters, so they
            // get the registry-free wrapper that reads the `get-size!` snapshot.
            let is_duckdb = self
                .backend
                .as_any()
                .downcast_ref::<egglog_bridge_duckdb::EGraph>()
                .is_some();
            let snapshot = self.proof_state.get_size_snapshot.clone();
            self.register_per_context(x, validator, valid_ctxs, false, move |x, _ctx| {
                if is_duckdb {
                    Box::new(DuckPlaceholderPrimWrapper {
                        name: x.name().to_owned(),
                    }) as Box<dyn ExternalFunction>
                } else {
                    Box::new(RegistryFreePrimWrapper {
                        name: x.name().to_owned(),
                        get_size_snapshot: snapshot.clone(),
                    }) as Box<dyn ExternalFunction>
                }
            });
            return;
        }

        let registry = self.backend.action_registry().clone();
        self.register_per_context(x, validator, valid_ctxs, false, move |x, ctx| {
            Box::new(RegistryPrimWrapper::<T, S> {
                prim: x,
                registry: registry.clone(),
                ctx,
                _wrap: std::marker::PhantomData,
            })
        });
    }

    /// Shared registration engine. Stores one primitive definition, plus
    /// one runtime id per valid [`Context`]. Each wrapper carries its
    /// specific context stamped onto the state wrapper at invoke time.
    fn register_per_context<T, F>(
        &mut self,
        x: T,
        validator: Option<PrimitiveValidator>,
        valid_ctxs: &[Context],
        pure: bool,
        mut build_wrapper: F,
    ) where
        T: Primitive + Clone,
        F: FnMut(T, Context) -> Box<dyn ExternalFunction>,
    {
        let primitive: Arc<dyn Primitive> = Arc::new(x.clone());
        let name = primitive.name().to_owned();
        let context_ids = EnumMap::from_fn(|ctx| {
            valid_ctxs.contains(&ctx).then(|| {
                let id = self
                    .backend
                    .register_external_func(build_wrapper(x.clone(), ctx));
                // DuckDB side-channel: register the primitive's
                // user-visible name so the duck rule-builder can
                // translate `ExternalFunctionId` references into
                // `Term::Prim(name, …)` calls in the duck IR.
                if let Some(duck) = self
                    .backend
                    .as_any_mut()
                    .downcast_mut::<egglog_bridge_duckdb::EGraph>()
                {
                    duck.set_external_func_name(id, name.clone());
                }
                // Feldera side-channel (Stage C of #23): record the names of
                // PURE primitives so `dbsp_join::plan_join` can lower an
                // arbitrary pure body prim to an ON-CIRCUIT call-prim node (the
                // closure re-evaluates the real prim through a shared engine
                // under a lock), making the `@congruence` / `@rebuild_cleanup` /
                // user value-prim rules DBSP-eligible instead of falling back to
                // the host nested-loop. Pure prims are idempotent (interning is
                // idempotent) so re-evaluating on-circuit is safe; impure/IO
                // prims (`pure == false`) are never recorded and stay ineligible.
                // The generic rep-comparison guards (`!=`, `bool-!=`, `or`,
                // `guard`, `ordering-min/max`) are pure and recognized BY NAME as
                // the in-join fast path (no lock); recording them here also lets
                // them act as the name signal `plan_join` consults.
                if pure
                    && let Some(feld) = self
                        .backend
                        .as_any_mut()
                        .downcast_mut::<egglog_bridge_feldera::EGraph>()
                {
                    feld.set_pure_prim_name(id, name.clone());
                }
                id
            })
        });
        self.type_info
            .primitives
            .entry(name)
            .or_default()
            .push(PrimitiveWithId {
                primitive,
                validator,
                context_ids,
            });
    }
}

impl EGraph {
    pub(crate) fn typecheck_program(
        &mut self,
        program: &Vec<NCommand>,
    ) -> Result<Vec<ResolvedNCommand>, TypeError> {
        let mut result = vec![];
        for command in program {
            result.push(self.typecheck_command(command)?);
        }
        Ok(result)
    }

    fn typecheck_command(&mut self, command: &NCommand) -> Result<ResolvedNCommand, TypeError> {
        // A2: the encoder-minted native-merge PROOF view is the one and only
        // tuple-output `term_constructor` view we permit (typechecking otherwise
        // rejects tuple outputs for views). Discriminate it precisely by its
        // `native_merge_views` membership (recorded by the encoder before it
        // declares the view), passed into `typecheck_function` so user-written
        // tuple constructors/views stay rejected.
        let native_merge_view = match command {
            NCommand::Function(fdecl) => self.proof_state.native_merge_views.contains(&fdecl.name),
            _ => false,
        };
        let symbol_gen = &mut self.parser.symbol_gen;

        let command: ResolvedNCommand = match command {
            NCommand::Function(fdecl) => {
                let resolved =
                    self.type_info
                        .typecheck_function(symbol_gen, fdecl, native_merge_view)?;
                // If this is a let binding, add it to global_sorts
                // This preserves bahavior for lets after desugaring
                if resolved.internal_let {
                    let output_sort = self.type_info.sorts.get(&fdecl.schema.output).unwrap();
                    self.type_info
                        .global_sorts
                        .insert(fdecl.name.clone(), output_sort.clone());
                }
                // Register the `:canon-prim` placeholder primitive (if any). Its
                // signature is `(S) -> S` for the function's eqsort S (the find-
                // or-self canonicalizer); the backend rebinds its external id to
                // the real union-find canonicalizer in `declare_function`. The
                // eqsort is the function's first input (which is `S` in both the
                // `(S) S` term-mode schema and the `(S S) Proof` proof-mode one).
                if let FunctionImpl::DisplacedUnionFind {
                    canon_prim: Some(canon_prim),
                    ..
                } = &resolved.impl_kind
                {
                    let eqsort_name = resolved
                        .schema
                        .input
                        .first()
                        .unwrap_or(&resolved.schema.output);
                    let Some(sort) = self.type_info.get_sort_by_name(eqsort_name) else {
                        return Err(TypeError::UndefinedSort(
                            eqsort_name.clone(),
                            resolved.span.clone(),
                        ));
                    };
                    let prim = UfCanonPrimitive {
                        name: canon_prim.clone(),
                        sort: sort.clone(),
                    };
                    self.add_pure_primitive(prim, None);
                }
                ResolvedNCommand::Function(resolved)
            }
            NCommand::NormRule { rule } => ResolvedNCommand::NormRule {
                rule: self
                    .type_info
                    .typecheck_rule(symbol_gen, rule, self.seminaive)?,
            },
            NCommand::Sort {
                span,
                name,
                presort_and_args,
                uf,
                proof_func,
                uf_pair_prims,
                unionable,
            } => {
                // Note this is bad since typechecking should be pure and idempotent
                // Otherwise typechecking the same program twice will fail
                self.declare_sort(name.clone(), presort_and_args, span.clone())?;
                // Mark as non-unionable if the sort declaration says so
                if !unionable {
                    self.type_info.non_unionable_sorts.insert(name.clone());
                }
                // Proof-mode canonicalize-at-creation: if this is a `@UFPair_S`
                // pair sort (carries `:internal-uf-pair-prims`), the `@UFPair_S`
                // ArcSort now exists, so register the sort's find-or-refl
                // primitives. The annotation is in the desugared text, so a
                // re-parse (desugar round-trip) re-registers them.
                if let Some((uf_function, refl_prim, find_leader_prim)) = uf_pair_prims {
                    self.register_uf_refl_primitives(
                        name,
                        uf_function,
                        refl_prim,
                        find_leader_prim,
                    );
                }
                ResolvedNCommand::Sort {
                    span: span.clone(),
                    name: name.clone(),
                    presort_and_args: presort_and_args.clone(),
                    uf: uf.clone(),
                    proof_func: proof_func.clone(),
                    uf_pair_prims: uf_pair_prims.clone(),
                    unionable: *unionable,
                }
            }
            NCommand::CoreAction(action @ Action::Let(span, var, _)) => {
                let action = self.type_info.typecheck_standalone_action(
                    symbol_gen,
                    action,
                    &Default::default(),
                    Context::Full,
                )?;
                self.ensure_global_name_prefix(span, var)?;
                let ResolvedAction::Let(_, resolved_var, _) = &action else {
                    unreachable!("typechecking an Action::Let should return ResolvedAction::Let")
                };
                self.type_info
                    .global_sorts
                    .insert(resolved_var.name.clone(), resolved_var.sort.clone());
                ResolvedNCommand::CoreAction(action)
            }
            NCommand::CoreAction(action) => {
                ResolvedNCommand::CoreAction(self.type_info.typecheck_standalone_action(
                    symbol_gen,
                    action,
                    &Default::default(),
                    Context::Full,
                )?)
            }
            NCommand::Extract(span, expr, variants) => {
                let res_expr = self.type_info.typecheck_standalone_expr(
                    symbol_gen,
                    expr,
                    &Default::default(),
                    Context::Full,
                )?;

                let res_variants = self.type_info.typecheck_standalone_expr(
                    symbol_gen,
                    variants,
                    &Default::default(),
                    Context::Full,
                )?;
                if res_variants.output_type().name() != I64Sort.name() {
                    return Err(TypeError::Mismatch {
                        expr: variants.clone(),
                        expected: I64Sort.to_arcsort(),
                        actual: res_variants.output_type(),
                    });
                }

                ResolvedNCommand::Extract(span.clone(), res_expr, res_variants)
            }
            NCommand::Check(span, facts) => ResolvedNCommand::Check(
                span.clone(),
                self.type_info.typecheck_facts(symbol_gen, facts)?,
            ),
            NCommand::Fail(span, cmd) => {
                ResolvedNCommand::Fail(span.clone(), Box::new(self.typecheck_command(cmd)?))
            }
            NCommand::RunSchedule(schedule) => ResolvedNCommand::RunSchedule(
                self.type_info.typecheck_schedule(symbol_gen, schedule)?,
            ),
            NCommand::Pop(span, n) => ResolvedNCommand::Pop(span.clone(), *n),
            NCommand::Push(n) => ResolvedNCommand::Push(*n),
            NCommand::AddRuleset(span, ruleset) => {
                ResolvedNCommand::AddRuleset(span.clone(), ruleset.clone())
            }
            NCommand::UnstableCombinedRuleset(span, name, sub_rulesets) => {
                ResolvedNCommand::UnstableCombinedRuleset(
                    span.clone(),
                    name.clone(),
                    sub_rulesets.clone(),
                )
            }
            NCommand::PrintOverallStatistics(span, file) => {
                ResolvedNCommand::PrintOverallStatistics(span.clone(), file.clone())
            }
            NCommand::PrintFunction(span, table, size, file, mode) => {
                ResolvedNCommand::PrintFunction(
                    span.clone(),
                    table.clone(),
                    *size,
                    file.clone(),
                    *mode,
                )
            }
            NCommand::PrintSize(span, n) => {
                // Should probably also resolve the function symbol here
                ResolvedNCommand::PrintSize(span.clone(), n.clone())
            }
            NCommand::ProveExists(span, constructor) => {
                let func_type = self
                    .type_info
                    .get_func_type(constructor)
                    .ok_or_else(|| TypeError::UnboundFunction(constructor.clone(), span.clone()))?;
                if func_type.subtype != FunctionSubtype::Constructor {
                    return Err(TypeError::ProveExistsRequiresConstructor(
                        constructor.clone(),
                        span.clone(),
                    ));
                }
                ResolvedNCommand::ProveExists(span.clone(), ResolvedCall::Func(func_type.clone()))
            }
            NCommand::Output { span, file, exprs } => {
                let exprs = exprs
                    .iter()
                    .map(|expr| {
                        self.type_info.typecheck_standalone_expr(
                            symbol_gen,
                            expr,
                            &Default::default(),
                            Context::Full,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                ResolvedNCommand::Output {
                    span: span.clone(),
                    file: file.clone(),
                    exprs,
                }
            }
            NCommand::Input { span, name, file } => ResolvedNCommand::Input {
                span: span.clone(),
                name: name.clone(),
                file: file.clone(),
            },
            NCommand::UserDefined(span, name, exprs) => {
                ResolvedNCommand::UserDefined(span.clone(), name.clone(), exprs.clone())
            }
        };
        if let ResolvedNCommand::NormRule { rule } = &command {
            self.warn_for_prefixed_non_globals_in_rule(rule)?;
        }
        Ok(command)
    }

    fn warn_for_prefixed_non_globals_in_var(
        &mut self,
        span: &Span,
        var: &ResolvedVar,
    ) -> Result<(), TypeError> {
        if var.is_global_ref {
            return Ok(());
        }
        if var.name.starts_with(crate::GLOBAL_NAME_PREFIX) {
            self.warn_prefixed_non_globals(span, &var.name)?;
        }
        Ok(())
    }

    fn warn_for_prefixed_non_globals_in_rule(
        &mut self,
        rule: &ResolvedRule,
    ) -> Result<(), TypeError> {
        let mut res: Result<(), TypeError> = Ok(());

        for fact in &rule.body {
            fact.visit_vars(&mut |span, var| {
                if res.is_ok() {
                    res = self.warn_for_prefixed_non_globals_in_var(span, var);
                }
            });
        }

        rule.head.visit_vars(&mut |span, var| {
            if res.is_ok() {
                res = self.warn_for_prefixed_non_globals_in_var(span, var);
            }
        });
        res
    }
}

impl TypeInfo {
    /// Adds a sort constructor to the typechecker's known set of types.
    pub fn add_presort<S: Presort>(&mut self, span: Span) -> Result<(), TypeError> {
        let name = S::presort_name();
        match self.mksorts.entry(name.to_owned()) {
            HEntry::Occupied(_) => Err(TypeError::SortAlreadyBound(name.to_owned(), span)),
            HEntry::Vacant(e) => {
                e.insert(S::make_sort);
                self.reserved_primitives.extend(S::reserved_primitives());
                Ok(())
            }
        }
    }

    /// Returns all sorts that satisfy the type and predicate.
    pub fn get_sorts_by<S: Sort>(&self, pred: impl Fn(&Arc<S>) -> bool) -> Vec<Arc<S>> {
        let mut results = Vec::new();
        for sort in self.sorts.values() {
            let sort = sort.clone().as_arc_any();
            if let Ok(sort) = Arc::downcast(sort)
                && pred(&sort)
            {
                results.push(sort);
            }
        }
        results
    }

    /// Returns all sorts based on the type.
    pub fn get_sorts<S: Sort>(&self) -> Vec<Arc<S>> {
        self.get_sorts_by(|_| true)
    }

    /// Returns a sort that satisfies the type and predicate.
    pub fn get_sort_by<S: Sort>(&self, pred: impl Fn(&Arc<S>) -> bool) -> Arc<S> {
        let results = self.get_sorts_by(pred);
        assert_eq!(
            results.len(),
            1,
            "Expected exactly one sort for type {}",
            std::any::type_name::<S>()
        );
        results.into_iter().next().unwrap()
    }

    /// Returns a sort based on the type.
    pub fn get_sort<S: Sort>(&self) -> Arc<S> {
        self.get_sort_by(|_| true)
    }

    /// Returns all sorts that satisfy the predicate.
    pub fn get_arcsorts_by(&self, f: impl Fn(&ArcSort) -> bool) -> Vec<ArcSort> {
        self.sorts.values().filter(|&x| f(x)).cloned().collect()
    }

    /// Returns a sort based on the predicate.
    pub fn get_arcsort_by(&self, f: impl Fn(&ArcSort) -> bool) -> ArcSort {
        let results = self.get_arcsorts_by(f);
        assert_eq!(
            results.len(),
            1,
            "Expected exactly one sort matching the given predicate"
        );
        results.into_iter().next().unwrap()
    }

    /// Returns the unique sort whose runtime values have Rust type `T`.
    pub fn get_arcsort_for_value_type<T: 'static>(&self) -> ArcSort {
        let results = self.get_arcsorts_by(|s| s.value_type() == Some(std::any::TypeId::of::<T>()));
        assert_eq!(
            results.len(),
            1,
            "Expected exactly one sort for type `{}`",
            std::any::type_name::<T>()
        );
        results.into_iter().next().unwrap()
    }

    /// Check if a sort allows union operations.
    /// A sort is unionable if it's an eq_sort and not marked as non-unionable
    /// (e.g., from `(sort Foo :no-union)` or relation desugaring).
    pub fn is_sort_unionable(&self, sort: &ArcSort) -> bool {
        sort.is_eq_sort() && !self.non_unionable_sorts.contains(sort.name())
    }

    fn function_to_functype(&self, func: &FunctionDecl) -> Result<FuncType, TypeError> {
        let resolve = |name: &String| -> Result<ArcSort, TypeError> {
            self.sorts
                .get(name)
                .cloned()
                .ok_or_else(|| TypeError::UndefinedSort(name.clone(), func.span.clone()))
        };
        let input = func
            .schema
            .input
            .iter()
            .map(&resolve)
            .collect::<Result<Vec<_>, _>>()?;
        let output = resolve(&func.schema.output)?;
        let extra_outputs = func
            .schema
            .extra_outputs
            .iter()
            .map(&resolve)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(FuncType {
            name: func.name.clone(),
            subtype: func.subtype,
            input,
            output,
            extra_outputs,
        })
    }

    fn typecheck_function(
        &mut self,
        symbol_gen: &mut SymbolGen,
        fdecl: &FunctionDecl,
        // A2: true only for the encoder-minted native-merge PROOF view, which is
        // permitted to be a tuple-output `term_constructor` view (otherwise
        // rejected). Discriminated by the caller via `native_merge_views`.
        is_native_merge_proof_view: bool,
    ) -> Result<ResolvedFunctionDecl, TypeError> {
        if self.sorts.contains_key(&fdecl.name) {
            return Err(TypeError::SortAlreadyBound(
                fdecl.name.clone(),
                fdecl.span.clone(),
            ));
        }
        if self.is_primitive(&fdecl.name) {
            return Err(TypeError::PrimitiveAlreadyBound(
                fdecl.name.clone(),
                fdecl.span.clone(),
            ));
        }
        // View tables (with term_constructor) must have at least one input (the e-class)
        if fdecl.term_constructor.is_some() && fdecl.schema.input.is_empty() {
            return Err(TypeError::TermConstructorNoInputs(
                fdecl.name.clone(),
                fdecl.span.clone(),
            ));
        }
        // For UF-backed functions, the `:canon-prim` name (if any) must be
        // fresh: it becomes a primitive placeholder, rebound at backend-build
        // time to the union-find canonicalizer external function.
        if let FunctionImpl::DisplacedUnionFind {
            canon_prim: Some(canon_prim),
            ..
        } = &fdecl.impl_kind
        {
            if self.sorts.contains_key(canon_prim) {
                return Err(TypeError::SortAlreadyBound(
                    canon_prim.clone(),
                    fdecl.span.clone(),
                ));
            }
            if self.func_types.contains_key(canon_prim) {
                return Err(TypeError::FunctionAlreadyBound(
                    canon_prim.clone(),
                    fdecl.span.clone(),
                ));
            }
            if self.is_primitive(canon_prim) {
                return Err(TypeError::PrimitiveAlreadyBound(
                    canon_prim.clone(),
                    fdecl.span.clone(),
                ));
            }
        }
        let ftype = self.function_to_functype(fdecl)?;
        if self.func_types.contains_key(&fdecl.name) {
            return Err(TypeError::FunctionAlreadyBound(
                fdecl.name.clone(),
                fdecl.span.clone(),
            ));
        }
        if let FunctionImpl::DisplacedUnionFind { onchange, .. } = &fdecl.impl_kind {
            if fdecl.merge.is_some() {
                return Err(TypeError::UfFunctionMerge(
                    fdecl.name.clone(),
                    fdecl.span.clone(),
                ));
            }
            // Schema must be `(S) S` for an EqSort S (term mode), or
            // `(S S) Proof` in proof mode: the two inputs are the eqsort being
            // unioned and the output carries the per-edge proof. The eqsort is
            // always `input[0]`.
            let eqsort_name = ftype.input.first().map(|s| s.name().to_owned());
            let term_schema = ftype.input.len() == 1
                && ftype.input[0].name() == ftype.output.name()
                && ftype.output.is_eq_sort();
            let proof_schema = ftype.input.len() == 2
                && ftype.input[0].name() == ftype.input[1].name()
                && ftype.input[0].is_eq_sort()
                && ftype.output.is_eq_sort();
            if fdecl.schema.input.len() != ftype.input.len() || !(term_schema || proof_schema) {
                return Err(TypeError::UfFunctionSchema(
                    fdecl.name.clone(),
                    fdecl.span.clone(),
                ));
            }
            if let Some(onchange) = onchange {
                let Some(rel_type) = self.func_types.get(onchange) else {
                    return Err(TypeError::UnboundFunction(
                        onchange.clone(),
                        fdecl.span.clone(),
                    ));
                };
                // `(relation R (S S S S S))` desugars to a constructor over a
                // fresh sort; require the first 5 inputs to be the function's
                // eqsort. In proof mode the relation has a trailing 6th `Proof`
                // column carrying the composed leader-change proof (any eq-sort),
                // so allow 5 or 6 inputs where the first 5 match.
                let out_name = eqsort_name.unwrap_or_else(|| ftype.output.name().to_owned());
                let valid_schema = (rel_type.input.len() == 5 || rel_type.input.len() == 6)
                    && rel_type.input[..5]
                        .iter()
                        .all(|sort| sort.name() == out_name);
                if !valid_schema {
                    return Err(TypeError::UfOnChangeSchema(
                        onchange.clone(),
                        out_name,
                        fdecl.span.clone(),
                    ));
                }
            }
        }
        self.func_types.insert(fdecl.name.clone(), ftype);
        let outputs: Vec<ArcSort> = fdecl
            .schema
            .outputs()
            .map(|name| self.sorts.get(name).unwrap().clone())
            .collect();
        let is_tuple = fdecl.schema.is_tuple_output();

        // Tuple outputs are only meaningful for custom functions (which carry a functional
        // dependency from keys to a tuple of values). Constructors and view tables mint/track a
        // single e-class id.
        //
        // A2 EXCEPTION: the encoder-minted native-merge PROOF view IS a
        // tuple-output `term_constructor` view — `(children) -> (eclass, Proof)`
        // with a `Columns([UnionIntoUfWithProof, EclassMinProof])` merge that does
        // congruence inline. It is permitted here (and ONLY here, keyed on
        // `is_native_merge_proof_view`); user-written tuple constructors/views are
        // still rejected.
        if is_tuple
            && !is_native_merge_proof_view
            && (fdecl.subtype == FunctionSubtype::Constructor || fdecl.term_constructor.is_some())
        {
            return Err(TypeError::TupleOutputNotAllowed(
                fdecl.name.clone(),
                fdecl.span.clone(),
            ));
        }
        if fdecl.subtype == FunctionSubtype::Constructor && !outputs[0].is_eq_sort() {
            return Err(TypeError::ConstructorOutputNotSort(
                fdecl.name.clone(),
                fdecl.span.clone(),
            ));
        }

        // For single-output functions the merge expression refers to `old`/`new`. For
        // tuple-output functions it refers to `old0`, `new0`, `old1`, `new1`, ... (one pair per
        // output column), and the whole merge is a `(values ...)` form.
        let mut bound_vars = IndexMap::default();
        let tuple_var_names: Vec<(String, String)> = (0..outputs.len())
            .map(|i| (format!("old{i}"), format!("new{i}")))
            .collect();
        if is_tuple {
            for (i, (old_name, new_name)) in tuple_var_names.iter().enumerate() {
                bound_vars.insert(old_name.as_str(), (fdecl.span.clone(), outputs[i].clone()));
                bound_vars.insert(new_name.as_str(), (fdecl.span.clone(), outputs[i].clone()));
            }
        } else {
            bound_vars.insert("old", (fdecl.span.clone(), outputs[0].clone()));
            bound_vars.insert("new", (fdecl.span.clone(), outputs[0].clone()));
        }

        let merge = match &fdecl.merge {
            // Merge expressions run as part of action-side table updates: writes are allowed, but
            // live DB reads would be untracked by seminaive rule execution.
            Some(merge) if is_tuple => {
                Some(self.typecheck_tuple_merge(symbol_gen, fdecl, merge, &outputs, &bound_vars)?)
            }
            Some(merge) => Some(self.typecheck_standalone_expr(
                symbol_gen,
                merge,
                &bound_vars,
                Context::Write,
            )?),
            None => None,
        };

        Ok(ResolvedFunctionDecl {
            name: fdecl.name.clone(),
            subtype: fdecl.subtype,
            impl_kind: fdecl.impl_kind.clone(),
            schema: fdecl.schema.clone(),
            resolved_schema: ResolvedCall::Func(self.func_types.get(&fdecl.name).unwrap().clone()),
            merge,
            cost: fdecl.cost,
            unextractable: fdecl.unextractable,
            internal_hidden: fdecl.internal_hidden,
            internal_let: fdecl.internal_let,
            span: fdecl.span.clone(),
            term_constructor: fdecl.term_constructor.clone(),
        })
    }

    /// Typecheck the `(values e0 e1 ...)` merge of a tuple-output function. Each `ei` is checked
    /// with `old0`/`new0`/... bound to the corresponding output columns, and must have the type of
    /// output column `i`. The result is a resolved `values` call carrying the output sorts.
    fn typecheck_tuple_merge(
        &self,
        symbol_gen: &mut SymbolGen,
        fdecl: &FunctionDecl,
        merge: &Expr,
        outputs: &[ArcSort],
        bound_vars: &IndexMap<&str, (Span, ArcSort)>,
    ) -> Result<ResolvedExpr, TypeError> {
        let args = match merge {
            GenericExpr::Call(_, head, args) if head.as_str() == "values" => args,
            _ => {
                return Err(TypeError::TupleMergeNotValues(
                    fdecl.name.clone(),
                    fdecl.span.clone(),
                ));
            }
        };
        if args.len() != outputs.len() {
            return Err(TypeError::TupleMergeArity {
                name: fdecl.name.clone(),
                expected: outputs.len(),
                actual: args.len(),
                span: fdecl.span.clone(),
            });
        }
        let mut resolved_args = Vec::with_capacity(args.len());
        for (arg, expected) in args.iter().zip(outputs) {
            let resolved =
                self.typecheck_standalone_expr(symbol_gen, arg, bound_vars, Context::Write)?;
            let actual = resolved.output_type();
            if actual.name() != expected.name() {
                return Err(TypeError::Mismatch {
                    expr: arg.clone(),
                    expected: expected.clone(),
                    actual,
                });
            }
            resolved_args.push(resolved);
        }
        Ok(GenericExpr::Call(
            merge.span(),
            ResolvedCall::Values(outputs.to_vec()),
            resolved_args,
        ))
    }

    fn typecheck_schedule(
        &self,
        symbol_gen: &mut SymbolGen,
        schedule: &Schedule,
    ) -> Result<ResolvedSchedule, TypeError> {
        let schedule = match schedule {
            Schedule::Repeat(span, times, schedule) => ResolvedSchedule::Repeat(
                span.clone(),
                *times,
                Box::new(self.typecheck_schedule(symbol_gen, schedule)?),
            ),
            Schedule::Sequence(span, schedules) => {
                let schedules = schedules
                    .iter()
                    .map(|schedule| self.typecheck_schedule(symbol_gen, schedule))
                    .collect::<Result<Vec<_>, _>>()?;
                ResolvedSchedule::Sequence(span.clone(), schedules)
            }
            Schedule::Saturate(span, schedule) => ResolvedSchedule::Saturate(
                span.clone(),
                Box::new(self.typecheck_schedule(symbol_gen, schedule)?),
            ),
            Schedule::Run(span, RunConfig { ruleset, until }) => {
                let until = until
                    .as_ref()
                    .map(|facts| self.typecheck_facts(symbol_gen, facts))
                    .transpose()?;
                ResolvedSchedule::Run(
                    span.clone(),
                    ResolvedRunConfig {
                        ruleset: ruleset.clone(),
                        until,
                    },
                )
            }
        };

        Result::Ok(schedule)
    }

    fn typecheck_rule(
        &self,
        symbol_gen: &mut SymbolGen,
        rule: &Rule,
        global_seminaive: bool,
    ) -> Result<ResolvedRule, TypeError> {
        let Rule {
            span,
            head,
            body,
            name,
            ruleset,
            unsafe_seminaive,
            naive,
            no_decomp,
        } = rule;
        let mut constraints = vec![];

        // This rule runs without seminaive if either the rule-local
        // `:naive` option or the global `EGraph::seminaive == false`
        // applies. Both must widen primitive-context selection to
        // Read/Full so primitives that read or write the database can
        // run; mirrors the backend's `self.seminaive && !rule.naive`
        // check at rule-build time.
        let seminaive = global_seminaive && !*naive;
        // `:unsafe-seminaive` keeps seminaive evaluation but widens the
        // typecheck (and backend) primitive contexts to Read/Full, so the
        // RHS may read the database — read-primitives (FullPrim) *and*
        // function-table lookups. See `GenericRule::unsafe_seminaive`.
        let context_seminaive = seminaive && !*unsafe_seminaive;
        let (query_ctx, action_ctx) = if context_seminaive {
            (Context::Pure, Context::Write)
        } else {
            (Context::Read, Context::Full)
        };

        let (query, mapped_query) = Facts(body.clone()).to_query(self, symbol_gen);
        constraints.extend(query.get_constraints(self, query_ctx)?);

        let mut binding = query.get_vars();
        // We lower to core actions with `union_to_set_optimization`
        // later in the pipeline. For typechecking we do not need it.
        let mut ctx = CoreActionContext::new(self, &mut binding, symbol_gen, false);
        let (actions, mapped_action) = head.to_core_actions(&mut ctx)?;

        let mut problem = Problem::default();
        problem.add_rule(
            &CoreRule {
                span: span.clone(),
                body: query,
                head: actions,
            },
            self,
            symbol_gen,
            query_ctx,
            action_ctx,
        )?;

        let assignment = problem
            .solve(|sort: &ArcSort| sort.name())
            .map_err(|e| e.to_type_error())?;

        let body: Vec<ResolvedFact> = assignment.annotate_facts(&mapped_query, self, query_ctx);
        let actions: ResolvedActions =
            assignment.annotate_actions(&mapped_action, self, action_ctx)?;

        // `unsafe-lookup` rules opt out of the "no function lookups in
        // actions" check — that's the whole point of the form. Every
        // other rule is still checked.
        if !unsafe_seminaive {
            self.check_lookup_actions(&actions)?;
        }

        Ok(ResolvedRule {
            span: span.clone(),
            body,
            head: actions,
            name: name.clone(),
            ruleset: ruleset.clone(),
            unsafe_seminaive: *unsafe_seminaive,
            naive: *naive,
            no_decomp: *no_decomp,
        })
    }

    fn check_lookup_expr(&self, expr: &ResolvedExpr) -> Result<(), TypeError> {
        if let Some(span) = self.expr_has_function_lookup(expr) {
            return Err(TypeError::LookupInRuleDisallowed(
                "function".to_string(),
                span,
            ));
        }
        Ok(())
    }

    fn check_lookup_actions(&self, actions: &ResolvedActions) -> Result<(), TypeError> {
        for action in actions.iter() {
            match action {
                GenericAction::Let(_, _, rhs) => self.check_lookup_expr(rhs)?,
                GenericAction::Set(_, _, args, rhs) => {
                    for arg in args.iter() {
                        self.check_lookup_expr(arg)?;
                    }
                    self.check_lookup_expr(rhs)?;
                }
                GenericAction::Union(_, lhs, rhs) => {
                    self.check_lookup_expr(lhs)?;
                    self.check_lookup_expr(rhs)?;
                }
                GenericAction::Change(_, _, _, args) => {
                    for arg in args.iter() {
                        self.check_lookup_expr(arg)?;
                    }
                }
                GenericAction::Panic(..) => {}
                GenericAction::Expr(_, expr) => self.check_lookup_expr(expr)?,
            }
        }
        Ok(())
    }

    pub fn typecheck_facts(
        &self,
        symbol_gen: &mut SymbolGen,
        facts: &[Fact],
    ) -> Result<Vec<ResolvedFact>, TypeError> {
        let (query, mapped_facts) = Facts(facts.to_vec()).to_query(self, symbol_gen);
        let mut problem = Problem::default();
        // Top-level query-shaped commands (e.g. `check`) are read-only:
        // primitives may inspect the database but not write to it.
        problem.add_query(&query, self, Context::Read)?;
        let assignment = problem
            .solve(|sort: &ArcSort| sort.name())
            .map_err(|e| e.to_type_error())?;
        let annotated_facts = assignment.annotate_facts(&mapped_facts, self, Context::Read);
        Ok(annotated_facts)
    }

    // Standalone expressions/actions use action lowering. Top-level commands
    // pass `Full`; function `:merge` reuses this path with `Write` because
    // merge expressions run during table updates.
    fn typecheck_standalone_actions(
        &self,
        symbol_gen: &mut SymbolGen,
        actions: &Actions,
        binding: &IndexMap<&str, (Span, ArcSort)>,
        context: Context,
    ) -> Result<ResolvedActions, TypeError> {
        let mut binding_set: IndexSet<String> =
            binding.keys().copied().map(str::to_string).collect();
        // We lower to core actions with `union_to_set_optimization`
        // later in the pipeline. For typechecking we do not need it.
        let mut ctx = CoreActionContext::new(self, &mut binding_set, symbol_gen, false);
        let (actions, mapped_action) = actions.to_core_actions(&mut ctx)?;
        let mut problem = Problem::default();

        problem.add_actions(&actions, self, symbol_gen, context)?;

        // add bindings from the context
        for (var, (span, sort)) in binding {
            problem.assign_local_var_type(var, span.clone(), sort.clone())?;
        }

        let assignment = problem
            .solve(|sort: &ArcSort| sort.name())
            .map_err(|e| e.to_type_error())?;

        let annotated_actions = assignment.annotate_actions(&mapped_action, self, context)?;
        Ok(annotated_actions)
    }

    fn typecheck_standalone_expr(
        &self,
        symbol_gen: &mut SymbolGen,
        expr: &Expr,
        binding: &IndexMap<&str, (Span, ArcSort)>,
        context: Context,
    ) -> Result<ResolvedExpr, TypeError> {
        let action = Action::Expr(expr.span(), expr.clone());
        let typechecked_action =
            self.typecheck_standalone_action(symbol_gen, &action, binding, context)?;
        match typechecked_action {
            ResolvedAction::Expr(_, expr) => Ok(expr),
            _ => unreachable!(),
        }
    }

    fn typecheck_standalone_action(
        &self,
        symbol_gen: &mut SymbolGen,
        action: &Action,
        binding: &IndexMap<&str, (Span, ArcSort)>,
        context: Context,
    ) -> Result<ResolvedAction, TypeError> {
        self.typecheck_standalone_actions(
            symbol_gen,
            &Actions::singleton(action.clone()),
            binding,
            context,
        )
        .map(|v| {
            assert_eq!(v.len(), 1);
            v.0.into_iter().next().unwrap()
        })
    }

    pub fn get_sort_by_name(&self, sym: &str) -> Option<&ArcSort> {
        self.sorts.get(sym)
    }

    pub fn get_prims(&self, sym: &str) -> Option<&[PrimitiveWithId]> {
        self.primitives.get(sym).map(Vec::as_slice)
    }

    pub fn is_primitive(&self, sym: &str) -> bool {
        self.primitives.contains_key(sym) || self.reserved_primitives.contains(sym)
    }

    /// Returns the current backend external id for the (single) primitive
    /// named `sym` valid in `ctx`, if there is exactly one such primitive
    /// registered. Used to refresh a `:canon-prim`'s id at rule-build time:
    /// the placeholder id baked into a resolved rule at typecheck time is
    /// rebound (via `replace_primitive_external_id`) once the UF function is
    /// declared, but already-resolved rules still carry the stale id. This
    /// lets `BackendRule::prim` pick up the rebound id.
    pub(crate) fn current_primitive_external_id(
        &self,
        sym: &str,
        ctx: Context,
    ) -> Option<ExternalFunctionId> {
        let prims = self.primitives.get(sym)?;
        if prims.len() != 1 {
            return None;
        }
        prims[0].context_ids[ctx]
    }

    /// Rebind the (single) primitive named `sym` so all of its per-context
    /// backend entrypoints point at `new_id`. Returns the previous backend ids
    /// (one per valid context) so the caller can free the now-unused
    /// placeholder entrypoints, or `None` if there is not exactly one primitive
    /// with that name. Used to rebind a `:canon-prim` placeholder to a UF
    /// canonicalizer external function.
    pub(crate) fn replace_primitive_external_id(
        &mut self,
        sym: &str,
        new_id: ExternalFunctionId,
    ) -> Option<Vec<ExternalFunctionId>> {
        let prims = self.primitives.get_mut(sym)?;
        if prims.len() != 1 {
            return None;
        }
        let mut old = Vec::new();
        for (_, slot) in prims[0].context_ids.iter_mut() {
            if let Some(id) = *slot {
                old.push(id);
                *slot = Some(new_id);
            }
        }
        Some(old)
    }

    pub fn primitive_has_validator(&self, id: ExternalFunctionId) -> bool {
        self.primitives
            .values()
            .flat_map(|v| v.iter())
            .any(|p| p.context_ids.iter().any(|(_, pid)| *pid == Some(id)) && p.validator.is_some())
    }

    pub fn get_func_type(&self, sym: &str) -> Option<&FuncType> {
        self.func_types.get(sym)
    }

    pub fn is_constructor(&self, sym: &str) -> bool {
        self.func_types
            .get(sym)
            .is_some_and(|f| f.subtype == FunctionSubtype::Constructor)
    }

    pub fn get_global_sort(&self, sym: &str) -> Option<&ArcSort> {
        self.global_sorts.get(sym)
    }

    pub fn is_global(&self, sym: &str) -> bool {
        self.global_sorts.contains_key(sym)
    }

    /// Check if an expression contains non-global function lookups (FunctionSubtype::Custom calls).
    /// Global function calls are allowed since they get desugared to constructors.
    /// Returns Some(span) if a lookup is found, None otherwise.
    pub fn expr_has_function_lookup(&self, expr: &ResolvedExpr) -> Option<Span> {
        use ast::GenericExpr;

        expr.find(&mut |e| {
            if let GenericExpr::Call(span, ResolvedCall::Func(func_type), _) = e
                && func_type.subtype == FunctionSubtype::Custom
                && !self.is_global(&func_type.name)
            {
                return Some(span.clone());
            }
            None
        })
    }
}

#[derive(Debug, Clone, Error)]
pub enum TypeError {
    #[error("{}\nArity mismatch, expected {expected} args: {expr}", .expr.span())]
    Arity { expr: Expr, expected: usize },
    #[error(
        "{}\n Expect expression {expr} to have type {}, but get type {}",
        .expr.span(), .expected.name(), .actual.name(),
    )]
    Mismatch {
        expr: Expr,
        expected: ArcSort,
        actual: ArcSort,
    },
    #[error("{1}\nUnbound symbol {0}")]
    Unbound(String, Span),
    #[error(
        "{1}\nVariable {0} is ungrounded. A variable is grounded when it appears as an argument to a constructor or function in the query, not just under primitives or equalities."
    )]
    Ungrounded(String, Span),
    #[error("{1}\nUndefined sort {0}")]
    UndefinedSort(String, Span),
    #[error("{1}\nUnbound function {0}")]
    UnboundFunction(String, Span),
    #[error("{1}\nprove-exists requires constructor function, but {0} is not a constructor")]
    ProveExistsRequiresConstructor(String, Span),
    #[error("{1}\nFunction already bound {0}")]
    FunctionAlreadyBound(String, Span),
    #[error("{1}\nSort {0} already declared.")]
    SortAlreadyBound(String, Span),
    #[error("{1}\nPrimitive {0} already declared.")]
    PrimitiveAlreadyBound(String, Span),
    #[error("Function type mismatch: expected {} => {}, actual {} => {}", .1.iter().map(|s| s.name().to_string()).collect::<Vec<_>>().join(", "), .0.name(), .3.iter().map(|s| s.name().to_string()).collect::<Vec<_>>().join(", "), .2.name())]
    FunctionTypeMismatch(ArcSort, Vec<ArcSort>, ArcSort, Vec<ArcSort>),
    #[error("{1}\nPresort {0} not found.")]
    PresortNotFound(String, Span),
    #[error("{}\nFailed to infer a type for: {}", .0.span(), .0)]
    InferenceFailure(Expr),
    #[error("{1}\nVariable {0} was already defined")]
    AlreadyDefined(String, Span),
    #[error("{1}\nThe output type of constructor function {0} must be sort")]
    ConstructorOutputNotSort(String, Span),
    #[error("{1}\nDisplaced-union-find function {0} cannot specify merge behaviour.")]
    UfFunctionMerge(String, Span),
    #[error("{1}\nDisplaced-union-find function {0} must have schema (S) S for an EqSort.")]
    UfFunctionSchema(String, Span),
    #[error("{2}\n:onchange relation {0} must have schema ({1} {1} {1} {1} {1}).")]
    UfOnChangeSchema(String, String, Span),
    #[error("{1}\nMerge expressions cannot call displaced-union-find function {0}.")]
    UfFunctionInMerge(String, Span),
    #[error("{1}\nValue lookup of non-constructor function {0} in rule is disallowed.")]
    LookupInRuleDisallowed(String, Span),
    #[error("{1}\nCannot set constructor {0}. Use `union` instead or declare {0} as a function.")]
    SetConstructorDisallowed(String, Span),
    #[error("All alternative definitions considered failed\n{}", .0.iter().map(|e| format!("  {e}\n")).collect::<Vec<_>>().join(""))]
    AllAlternativeFailed(Vec<TypeError>),
    #[error("{}\nCannot union values of sort {}", .1, .0.name())]
    NonEqsortUnion(ArcSort, Span),
    #[error("{}\nCannot union values of sort {} because it is marked as non-unionable (e.g. from a relation)", .1, .0.name())]
    NonUnionableSort(ArcSort, Span),
    #[error(
        "{1}\nView table {0} with :internal-term-constructor must have at least one input (the e-class)."
    )]
    TermConstructorNoInputs(String, Span),
    #[error(
        "{span}\nNon-global variable `{name}` must not start with `{}`.",
        crate::GLOBAL_NAME_PREFIX
    )]
    NonGlobalPrefixed { name: String, span: Span },
    #[error(
        "{span}\nGlobal `{name}` must start with `{}`.",
        crate::GLOBAL_NAME_PREFIX
    )]
    GlobalMissingPrefix { name: String, span: Span },
    #[error(
        "{1}\nFunction {0} has a tuple output, which is only allowed for plain functions (not constructors, relations, or view tables)."
    )]
    TupleOutputNotAllowed(String, Span),
    #[error(
        "{1}\nThe :merge of tuple-output function {0} must be a `(values ...)` form with one expression per output column."
    )]
    TupleMergeNotValues(String, Span),
    #[error(
        "{span}\nThe :merge of tuple-output function {name} has {actual} columns but the function has {expected} output columns."
    )]
    TupleMergeArity {
        name: String,
        expected: usize,
        actual: usize,
        span: Span,
    },
}

#[cfg(test)]
mod test {
    use crate::{EGraph, Error, typechecking::TypeError};

    #[test]
    fn test_arity_mismatch() {
        let mut egraph = EGraph::default();

        let prog = "
            (relation f (i64 i64))
            (rule ((f a b c)) ())
       ";
        let res = egraph.parse_and_run_program(None, prog);
        match res {
            Err(Error::TypeError(TypeError::Arity {
                expected: 2,
                expr: e,
            })) => {
                assert_eq!(e.span().string(), "(f a b c)");
            }
            _ => panic!("Expected arity mismatch, got: {res:?}"),
        }
    }
}
