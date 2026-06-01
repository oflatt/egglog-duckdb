//! User-facing execution state wrappers.
//!
//! Four wrappers around `core_relations::ExecutionState` expose different
//! subsets of the database API based on the context in which a primitive runs:
//!
//! | Wrapper       | DB reads | DB writes | Used for                                 |
//! |---------------|----------|-----------|------------------------------------------|
//! | `PureState`   | no       | no        | rule LHS (seminaive on)                  |
//! | `WriteState`  | no       | yes       | rule RHS (seminaive on)                  |
//! | `ReadState`   | yes      | no        | top-level query-shaped commands          |
//! | `FullState`   | yes      | yes       | top-level action-shaped commands, `eval` |
//!
//! Methods come from sealed capability traits implemented on each
//! wrapper:
//!
//! - [`Core`] — base values, counters, container interning, conversion
//!   sugar. Implemented for all four wrappers.
//! - [`Read`] — name-indexed table lookup (`state.lookup("name", &[…])`).
//!   Implemented for [`ReadState`] and [`FullState`].
//! - [`Write`] — name-indexed writes (`set`/`add_node`/`remove`/
//!   `subsume`/`union`/`panic`). Implemented for [`WriteState`] and
//!   [`FullState`].
//!
//! Privileged seams (`call_external_func`, `table_lookup`, raw
//! `&mut ExecutionState`) used by the `FunctionContainer` higher-order
//! dispatch live on the crate-private [`Internal`] trait. User code
//! cannot reach them.
//!
//! [`PurePrim`]: crate::PurePrim
//! [`WritePrim`]: crate::WritePrim
//! [`ReadPrim`]: crate::ReadPrim
//! [`FullPrim`]: crate::FullPrim

use std::ops::Deref;

use crate::api::{ApiError, BaseSortName, ColumnSort, Id, IntoColumn, IntoRow};
use crate::core_relations::{
    BaseValue, BaseValues, ContainerValue, ContainerValues, ExecutionState, ExternalFunctionId,
    Value,
};
use crate::Error;
use egglog_bridge::{ActionRegistry, TableAction, TableKind};

/// The four contexts a primitive may run in, named after the
/// capability profile they grant. Each variant maps 1:1 to one of the
/// state wrappers below: `Pure` ↔ [`PureState`], `Write` ↔ [`WriteState`],
/// `Read` ↔ [`ReadState`], `Full` ↔ [`FullState`]. The egglog
/// typechecker filters primitive definitions by whether they carry a
/// runtime id for the surrounding `Context` at each call site.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, enum_map::Enum)]
pub enum Context {
    /// No DB reads, no DB writes. The body (LHS) of a rule running
    /// under seminaive evaluation: a body read of live state means
    /// the rule won't re-fire when the read row's contents change
    /// in a later iteration; a body write makes no semantic sense.
    Pure,
    /// DB writes allowed, DB reads forbidden. The head (RHS) of a
    /// rule running under seminaive evaluation: same re-firing
    /// concern as `Pure` rules out reads, but staged writes are
    /// fine.
    Write,
    /// DB reads allowed, DB writes forbidden. Top-level query-shaped
    /// commands (`check`, condition evaluation) and the body of a
    /// `:naive` rule. Reads are safe because there is no seminaive
    /// epoch to violate.
    Read,
    /// DB reads and writes both allowed. Top-level action-shaped
    /// commands (`eval`, `let`, action-mode `run-schedule`) and the
    /// head of a `:naive` rule.
    Full,
}

impl Context {
    pub const ALL: [Context; 4] = [Context::Pure, Context::Write, Context::Read, Context::Full];
}

// =====================================================================
// Sealed traits.
//
// These traits are `pub(crate)`: external users cannot bring them
// into scope, so they cannot call the methods defined here even on
// values they have. They give the public capability traits (`Core`,
// `Write`) a way to reach the underlying `ExecutionState` from
// default methods, and carry the privileged seams used by the
// `FunctionContainer` higher-order dispatch.
// =====================================================================

/// Crate-private accessor + privileged-dispatch trait. Required
/// methods are the accessors that every wrapper supplies; default
/// methods are the privileged seams used by the `FunctionContainer`
/// higher-order dispatch.
pub(crate) trait Internal<'a, 'db: 'a>: 'a {
    fn es(&self) -> &ExecutionState<'db>;
    fn es_mut(&mut self) -> &mut ExecutionState<'db>;
    /// The call-site [`Context`] this primitive was invoked from.
    /// Stamped onto the wrapper at construction time by the
    /// `ExternalFunction` wrapper closure; read by
    /// [`Core::apply_function`] to route higher-order dispatch.
    fn ctx(&self) -> Context;

    fn call_external_func(&mut self, id: ExternalFunctionId, args: &[Value]) -> Option<Value> {
        self.es_mut().call_external_func(id, args)
    }
    fn raw_exec_state(&mut self) -> &mut ExecutionState<'db> {
        self.es_mut()
    }
}

/// Sealed accessor for the [`ActionRegistry`]. Implemented by every
/// wrapper that has a registry (`ReadState`, `WriteState`, `FullState`)
/// — the read- and write-side traits both look up `TableAction`s by
/// name through it.
pub(crate) trait RegistrySealed<'a, 'db: 'a>: Internal<'a, 'db> {
    fn registry(&self) -> &ActionRegistry;
}

// =====================================================================
// Public capability traits.
// =====================================================================

/// Core methods available on every state wrapper: base values,
/// counters, container interning, value/base/container conversion sugar.
/// Always seminaive-safe.
#[allow(private_bounds)]
pub trait Core<'a, 'db: 'a>: Internal<'a, 'db> {
    /// Base-value pool (interned primitives like `i64`, `String`, …).
    fn base_values(&self) -> &'a BaseValues {
        self.es().base_values()
    }
    /// Signal that rule execution should stop after this firing.
    fn trigger_early_stop(&self) {
        self.es().trigger_early_stop()
    }
    /// Has someone called `trigger_early_stop`?
    fn should_stop(&self) -> bool {
        self.es().should_stop()
    }
    /// Container values for this EGraph.
    fn container_values(&self) -> &'a ContainerValues {
        self.es().container_values()
    }

    /// Register a container value, returning its interned `Value`.
    fn register_container<C: ContainerValue>(&mut self, container: C) -> Value {
        // `container_values()` returns `&'a ContainerValues` — a reference
        // tied to the inner ExecutionState's lifetime, not to `&self` —
        // so it doesn't conflict with the subsequent `&mut` reborrow.
        // Avoiding the clone of `ExecutionState` here matters: this is
        // hot-path code (every container intern goes through it) and
        // the clone copies a non-trivial amount of state.
        let cv = self.container_values();
        let es = self.es_mut();
        cv.register_val(container, es)
    }

    /// Untyped `Value` → `T`. Skip-the-check escape; prefer
    /// [`Core::base`] (unchecked) or [`Core::extract`] (sort-checked).
    fn value_to_base<T: BaseValue>(&self, x: Value) -> T {
        self.es().base_values().unwrap::<T>(x)
    }

    /// Untyped `T` → `Value`. Prefer [`Core::id_of`] / [`Core::intern_typed`].
    fn base_to_value<T: BaseValue>(&self, x: T) -> Value {
        self.es().base_values().get::<T>(x)
    }

    /// Unchecked `Id` → `T`. Trusts the `Id`'s sort matches `T` (which
    /// holds for primitive bodies, where dispatch already filtered by
    /// signature). Use [`Core::extract`] for the sort-checked form.
    fn base<T: BaseValue>(&self, id: &Id) -> T {
        self.value_to_base::<T>(id.value())
    }

    /// `T` → `Id` with an explicit sort name. Use [`Core::intern_typed`]
    /// when the sort name is known at compile time via [`BaseSortName`].
    fn id_of<T: BaseValue>(&self, x: T, sort: &str) -> Id {
        Id::new(self.base_to_value::<T>(x), sort)
    }

    /// `T` → `Id`, sort name baked in at compile time via [`BaseSortName`].
    /// Standard base types only; for user-defined sorts pass the name
    /// explicitly via [`Core::id_of`].
    fn intern_typed<T: BaseSortName>(&self, x: T) -> Id {
        Id::with_sort(self.base_to_value::<T>(x), T::sort_name_arc())
    }

    /// Sort-checked `Id` → `T`. State-side counterpart to
    /// [`crate::EGraph::extract`]; standard base types only (the
    /// generic [`crate::EGraph::extract`] handles user-defined sorts
    /// via runtime `TypeId` lookup).
    fn extract<T: BaseSortName>(&self, id: &Id) -> Result<T, Error> {
        if id.sort() != T::SORT_NAME {
            return Err(ApiError::WrongOutputSort {
                table: format!("(extract::<{}>)", T::SORT_NAME),
                expected: T::SORT_NAME.to_string(),
                actual: id.sort().to_string(),
            }
            .into());
        }
        Ok(self.value_to_base::<T>(id.value()))
    }

    /// Look up the Rust container behind an egglog [`Value`], if any.
    fn value_to_container<T: ContainerValue>(
        &self,
        x: Value,
    ) -> Option<impl Deref<Target = T> + 'a> {
        self.es().container_values().get_val::<T>(x)
    }

    /// Intern a Rust container into the e-graph and return its
    /// [`Value`]. Sugar over `self.register_container(x)`.
    fn container_to_value<T: ContainerValue>(&mut self, x: T) -> Value {
        self.register_container(x)
    }

    /// Dispatch a wrapped `unstable-fn` value. This is the public entry
    /// point for higher-order primitive bodies: the call-site
    /// [`Context`] is stamped onto the state by the registration
    /// wrapper (see [`EGraph::add_pure_primitive`] and the matching
    /// `add_read_primitive` / `add_write_primitive` /
    /// `add_full_primitive`), so the caller can't supply a wrong
    /// context — there is no `ctx` parameter to lie about.
    ///
    /// [`EGraph::add_pure_primitive`]: crate::EGraph::add_pure_primitive
    fn apply_function(
        &mut self,
        fc: &crate::sort::FunctionContainer,
        args: &[Value],
    ) -> Option<Value> {
        let ctx = self.ctx();
        let mut pure = PureState::wrap(self.raw_exec_state(), ctx);
        fc.apply(&mut pure, args)
    }
}

/// Read-side methods — name-indexed table lookup. Implemented for
/// [`ReadState`] and [`FullState`] (not for [`PureState`] or
/// [`WriteState`] — a `Write` context can't depend on live DB state).
/// All methods return `Result`; misuse surfaces as [`crate::ApiError`].
#[allow(private_bounds)]
pub trait Read<'a, 'db: 'a>: Core<'a, 'db> + RegistrySealed<'a, 'db> {
    /// Read a function row's output. Returned [`Id`] carries the
    /// function's declared output sort.
    ///
    /// Function tables only. On a constructor, errors with
    /// `WrongSubtype` — use [`Read::eclass_of`] instead.
    fn lookup<K: IntoRow>(&self, name: &str, key: K) -> Result<Option<Id>, Error> {
        let action = lookup_action(self.registry(), name)?;
        check_subtype(name, &action, TableKind::Function, "function")?;
        let sorts = key.column_sorts();
        check_input_sorts(name, &action, &sorts)?;
        let key_values = key.into_values(self.base_values());
        let sort = action
            .output_sort_name()
            .cloned()
            .unwrap_or_else(crate::api::empty_sort);
        Ok(action
            .lookup(self.es(), &key_values)
            .map(|v| Id::with_sort(v, sort)))
    }

    /// Read a constructor row's eclass without minting on miss.
    /// Returned [`Id`] carries the constructor's output sort.
    ///
    /// Constructor / relation tables only. On a function, errors
    /// with `WrongSubtype` — use [`Read::lookup`] instead.
    fn eclass_of<K: IntoRow>(
        &self,
        name: &str,
        inputs: K,
    ) -> Result<Option<Id>, Error> {
        let action = lookup_action(self.registry(), name)?;
        check_subtype(
            name,
            &action,
            TableKind::Constructor,
            "constructor",
        )?;
        let sorts = inputs.column_sorts();
        check_input_sorts(name, &action, &sorts)?;
        let key_values = inputs.into_values(self.base_values());
        let sort = action
            .output_sort_name()
            .cloned()
            .unwrap_or_else(crate::api::empty_sort);
        Ok(action
            .lookup(self.es(), &key_values)
            .map(|v| Id::with_sort(v, sort)))
    }

    /// Whether a row with the given key exists. Any subtype; never mints.
    fn contains<K: IntoRow>(
        &self,
        name: &str,
        key: K,
    ) -> Result<bool, Error> {
        let action = lookup_action(self.registry(), name)?;
        let sorts = key.column_sorts();
        check_input_sorts(name, &action, &sorts)?;
        let key_values = key.into_values(self.base_values());
        Ok(action.lookup(self.es(), &key_values).is_some())
    }

    /// Lookup with input sort-checking disabled. Returned [`Id`]
    /// still carries the output sort tag. Use when the caller's
    /// inputs lack reliable sort tags (e.g., from `Vec<Id>` rows).
    fn lookup_raw(&self, name: &str, key: &[Id]) -> Result<Option<Id>, Error> {
        let action = lookup_action(self.registry(), name)?;
        let raw: Vec<Value> = key.iter().map(|id| id.value()).collect();
        let sort = action
            .output_sort_name()
            .cloned()
            .unwrap_or_else(crate::api::empty_sort);
        Ok(action
            .lookup(self.es(), &raw)
            .map(|v| Id::with_sort(v, sort)))
    }

    /// Return the current row count for the named table, or `None` if no table
    /// with that name is registered.
    fn table_size(&self, name: &str) -> Option<usize> {
        self.registry()
            .lookup_table(name)
            .map(|action| action.row_count(self.es()))
    }

    /// Snapshot the registered table names and their current row counts.
    fn table_sizes(&self) -> Vec<(&str, usize)> {
        self.registry().table_sizes(self.es())
    }
}

/// Write-side methods — name-indexed inserts/removes/subsumes plus
/// union and panic. Implemented for [`WriteState`] and [`FullState`].
/// All methods return `Result`; misuse surfaces as [`crate::ApiError`].
#[allow(private_bounds)]
pub trait Write<'a, 'db: 'a>: Core<'a, 'db> + RegistrySealed<'a, 'db> {
    /// `(set (f k) v)`. Function tables only — constructors error
    /// (use [`Write::add_node`]).
    fn set<K: IntoRow, V: IntoColumn>(
        &mut self,
        name: &str,
        key: K,
        value: V,
    ) -> Result<(), Error> {
        let action = lookup_action(self.registry(), name)?;
        check_subtype(name, &action, TableKind::Function, "function")?;
        let key_sorts = key.column_sorts();
        check_input_sorts(name, &action, &key_sorts)?;
        check_output_sort(name, &action, &value.column_sort())?;
        let bv = self.base_values();
        let mut row = key.into_values(bv);
        row.push(value.into_value(bv));
        action.insert(self.es_mut(), row.into_iter());
        Ok(())
    }

    /// `(Cons k1 k2 ...)` — mint or look up a constructor's eclass.
    /// Constructor / relation tables only — functions error (use
    /// [`Write::set`]). Returned [`Id`] carries the output sort.
    fn add_node<R: IntoRow>(
        &mut self,
        name: &str,
        inputs: R,
    ) -> Result<Id, Error> {
        let action = lookup_action(self.registry(), name)?;
        check_subtype(
            name,
            &action,
            TableKind::Constructor,
            "constructor",
        )?;
        let sorts = inputs.column_sorts();
        check_input_sorts(name, &action, &sorts)?;
        let bv = self.base_values();
        let key = inputs.into_values(bv);
        let value = action
            .lookup_or_insert(self.es_mut(), &key)
            .expect("constructor lookup_or_insert returned None");
        let sort = action
            .output_sort_name()
            .cloned()
            .unwrap_or_else(crate::api::empty_sort);
        Ok(Id::with_sort(value, sort))
    }

    /// Remove a row from the named table. Works for any subtype.
    fn remove<K: IntoRow>(
        &mut self,
        name: &str,
        key: K,
    ) -> Result<(), Error> {
        let action = lookup_action(self.registry(), name)?;
        let sorts = key.column_sorts();
        check_input_sorts(name, &action, &sorts)?;
        let key_values = key.into_values(self.base_values());
        action.remove(self.es_mut(), &key_values);
        Ok(())
    }

    /// Subsume a row in the named table.
    fn subsume<K: IntoRow>(
        &mut self,
        name: &str,
        key: K,
    ) -> Result<(), Error> {
        let action = lookup_action(self.registry(), name)?;
        let sorts = key.column_sorts();
        check_input_sorts(name, &action, &sorts)?;
        let key_values = key.into_values(self.base_values());
        action.subsume(self.es_mut(), key_values.into_iter());
        Ok(())
    }

    /// Union two values in the e-graph's union-find. Both must be in
    /// the same eq-sort.
    fn union(&mut self, x: Id, y: Id) -> Result<(), Error> {
        if x.sort() != y.sort() {
            return Err(ApiError::UnionSortMismatch {
                left: x.sort().to_string(),
                right: y.sort().to_string(),
            }
            .into());
        }
        let action = *self.registry().union_action();
        action.union(self.es_mut(), x.value(), y.value());
        Ok(())
    }

    /// Trigger a panic from a primitive. Always returns `None` so the
    /// caller can propagate with `?`.
    fn panic(&mut self) -> Option<()> {
        let panic_id = self.registry().default_panic_id();
        self.es_mut().call_external_func(panic_id, &[]);
        None
    }
}

fn lookup_action(
    registry: &ActionRegistry,
    name: &str,
) -> Result<TableAction, Error> {
    registry
        .lookup_table(name)
        .cloned()
        .ok_or_else(|| ApiError::MissingTable { name: name.to_string() }.into())
}

fn check_subtype(
    name: &str,
    action: &TableAction,
    expected: TableKind,
    expected_label: &'static str,
) -> Result<(), Error> {
    if action.kind() == expected {
        return Ok(());
    }
    let actual_label = match action.kind() {
        TableKind::Function => "function",
        TableKind::Constructor => "constructor",
    };
    Err(ApiError::WrongSubtype {
        name: name.to_string(),
        expected: expected_label,
        actual: actual_label,
    }
    .into())
}

fn check_input_sorts(
    table: &str,
    action: &TableAction,
    provided: &[ColumnSort],
) -> Result<(), Error> {
    let expected = action.input_sort_names();
    if expected.is_empty() {
        // Table registered without sort names — skip the check
        // entirely (the typed API can't validate; raw is fine).
        return Ok(());
    }
    if provided.len() != expected.len() {
        return Err(ApiError::WrongArity {
            table: table.to_string(),
            expected: expected.len(),
            got: provided.len(),
        }
        .into());
    }
    for (i, (got, want)) in provided.iter().zip(expected.iter()).enumerate() {
        if let Some(got) = got.name() {
            if got != want.as_ref() {
                return Err(ApiError::WrongColumnSort {
                    table: table.to_string(),
                    column: i,
                    expected: want.to_string(),
                    actual: got.to_string(),
                }
                .into());
            }
        }
        // Unchecked columns (e.g. bare `Value`) skip.
    }
    Ok(())
}

fn check_output_sort(
    table: &str,
    action: &TableAction,
    provided: &ColumnSort,
) -> Result<(), Error> {
    let Some(expected) = action.output_sort_name() else {
        // No sort name registered — skip.
        return Ok(());
    };
    if let Some(got) = provided.name() {
        if got != expected.as_ref() {
            return Err(ApiError::WrongOutputSort {
                table: table.to_string(),
                expected: expected.to_string(),
                actual: got.to_string(),
            }
            .into());
        }
    }
    Ok(())
}

// =====================================================================
// The four wrapper types — plain structs, methods come from traits.
// =====================================================================

/// Wrapper for [`Context::Pure`]. Implements [`Core`] only.
///
/// ```compile_fail
/// // Pure context cannot write: `Write` is not implemented.
/// use egglog::Write;
/// fn _no_writes<'a, 'db>(state: &mut egglog::PureState<'a, 'db>) {
///     state.set("foo", (1_i64,), 2_i64);
/// }
/// ```
///
/// ```compile_fail
/// // Pure context cannot reach raw `ExecutionState`.
/// fn _no_raw<'a, 'db>(state: &mut egglog::PureState<'a, 'db>) {
///     state.raw_exec_state();
/// }
/// ```
pub struct PureState<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
    /// The call-site [`Context`] the wrapping primitive was invoked
    /// from. Stamped by the wrapper closure at invocation time and
    /// read by [`PureState::apply_function`]; user code cannot
    /// observe or modify it directly.
    pub(crate) ctx: Context,
}

/// Wrapper for [`Context::Read`]. Implements [`Core`] + [`Read`].
pub struct ReadState<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
    pub(crate) registry: &'a ActionRegistry,
    pub(crate) ctx: Context,
}

/// Wrapper for [`Context::Write`]. Implements [`Core`] + [`Write`].
///
/// ```compile_fail
/// // Write context cannot reach raw `ExecutionState`.
/// fn _no_raw<'a, 'db>(state: &mut egglog::WriteState<'a, 'db>) {
///     state.raw_exec_state();
/// }
/// ```
pub struct WriteState<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
    pub(crate) registry: &'a ActionRegistry,
    pub(crate) ctx: Context,
}

/// Wrapper for [`Context::Full`]. Implements [`Core`] + [`Read`] + [`Write`].
///
/// ```compile_fail
/// // Even `FullState` cannot reach the raw `ExecutionState`.
/// fn _no_raw<'a, 'db>(state: &mut egglog::FullState<'a, 'db>) {
///     state.raw_exec_state();
/// }
/// ```
pub struct FullState<'a, 'db> {
    pub(crate) inner: &'a mut ExecutionState<'db>,
    pub(crate) registry: &'a ActionRegistry,
    pub(crate) ctx: Context,
}

impl<'a, 'db: 'a> PureState<'a, 'db> {
    pub(crate) fn wrap(es: &'a mut ExecutionState<'db>, ctx: Context) -> Self {
        Self { inner: es, ctx }
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &Context::ALL
    }
}

impl<'a, 'db: 'a> ReadState<'a, 'db> {
    pub(crate) fn wrap(
        es: &'a mut ExecutionState<'db>,
        registry: &'a ActionRegistry,
        ctx: Context,
    ) -> Self {
        Self {
            inner: es,
            registry,
            ctx,
        }
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::Read, Context::Full]
    }
}

impl<'a, 'db: 'a> WriteState<'a, 'db> {
    pub(crate) fn wrap(
        es: &'a mut ExecutionState<'db>,
        registry: &'a ActionRegistry,
        ctx: Context,
    ) -> Self {
        Self {
            inner: es,
            registry,
            ctx,
        }
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::Write, Context::Full]
    }
}

impl<'a, 'db: 'a> FullState<'a, 'db> {
    pub(crate) fn wrap(
        es: &'a mut ExecutionState<'db>,
        registry: &'a ActionRegistry,
        ctx: Context,
    ) -> Self {
        Self {
            inner: es,
            registry,
            ctx,
        }
    }
    pub const fn valid_contexts() -> &'static [Context] {
        &[Context::Full]
    }
}

// =====================================================================
// Trait impls. The wrappers implement the sealed accessor traits;
// the public capability traits' default methods do all the rest.
// =====================================================================

impl<'a, 'db: 'a> Internal<'a, 'db> for PureState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
    fn ctx(&self) -> Context {
        self.ctx
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for PureState<'a, 'db> {}

impl<'a, 'db: 'a> Internal<'a, 'db> for ReadState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
    fn ctx(&self) -> Context {
        self.ctx
    }
}
impl<'a, 'db: 'a> RegistrySealed<'a, 'db> for ReadState<'a, 'db> {
    fn registry(&self) -> &ActionRegistry {
        self.registry
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for ReadState<'a, 'db> {}
impl<'a, 'db: 'a> Read<'a, 'db> for ReadState<'a, 'db> {}

impl<'a, 'db: 'a> Internal<'a, 'db> for WriteState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
    fn ctx(&self) -> Context {
        self.ctx
    }
}
impl<'a, 'db: 'a> RegistrySealed<'a, 'db> for WriteState<'a, 'db> {
    fn registry(&self) -> &ActionRegistry {
        self.registry
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for WriteState<'a, 'db> {}
impl<'a, 'db: 'a> Write<'a, 'db> for WriteState<'a, 'db> {}

impl<'a, 'db: 'a> Internal<'a, 'db> for FullState<'a, 'db> {
    fn es(&self) -> &ExecutionState<'db> {
        self.inner
    }
    fn es_mut(&mut self) -> &mut ExecutionState<'db> {
        self.inner
    }
    fn ctx(&self) -> Context {
        self.ctx
    }
}
impl<'a, 'db: 'a> RegistrySealed<'a, 'db> for FullState<'a, 'db> {
    fn registry(&self) -> &ActionRegistry {
        self.registry
    }
}
impl<'a, 'db: 'a> Core<'a, 'db> for FullState<'a, 'db> {}
impl<'a, 'db: 'a> Read<'a, 'db> for FullState<'a, 'db> {}
impl<'a, 'db: 'a> Write<'a, 'db> for FullState<'a, 'db> {}
