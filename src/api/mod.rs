//! Typed row encoding for the [`crate::EGraph`] surface API.
//!
//! [`IntoRow`] / [`IntoColumn`] convert Rust values into egglog row data on
//! the way *in* (insert, lookup keys, query patterns).  [`FromRow`] /
//! [`FromColumn`] convert row data back into Rust values on the way *out*
//! (lookup return values, query iteration).
//!
//! Every input column carries a runtime [`ColumnSort`] tag. The trait
//! methods that consume rows ([`crate::Read::lookup`],
//! [`crate::Write::set`], etc.) validate each column's tag against the
//! table's declared schema and return [`crate::ApiError::WrongColumnSort`]
//! on mismatch. Base values (`i64`, `String`, `f64`, …) tag with the
//! corresponding egglog sort name automatically; eclass ids must come
//! through the [`Id`] wrapper, which pairs a [`Value`] with a runtime
//! sort name. [`Value`] itself participates as an unchecked escape
//! hatch ([`ColumnSort::Unchecked`]).

use std::sync::Arc;

use crate::core_relations::{BaseValue, BaseValues, Value};
use crate::sort;
use thiserror::Error;

// ---------------------------------------------------------------------
// BaseSortName — egglog sort name as a compile-time const for the
// standard Rust base types. Used internally by the `add_primitive!`
// macro and by [`IntoColumn`] impls to attach the correct sort tag
// to base [`Id`]s and [`ColumnSort::Named`] without a runtime lookup.
// ---------------------------------------------------------------------

/// Maps a Rust [`BaseValue`] type to its egglog sort name as a
/// compile-time const. Implemented for the standard base types
/// (`i64`, `bool`, `()`, `f64`-via-[`sort::F`], `String`-via-[`sort::S`],
/// [`sort::Z`], [`sort::Q`]). User-defined base sorts don't need to
/// implement this trait — [`crate::EGraph::intern`] /
/// [`crate::EGraph::extract`] look up the sort name by `TypeId` at
/// runtime from the registered sorts.
pub trait BaseSortName: BaseValue {
    const SORT_NAME: &'static str;
}

impl BaseSortName for i64 {
    const SORT_NAME: &'static str = "i64";
}
impl BaseSortName for bool {
    const SORT_NAME: &'static str = "bool";
}
impl BaseSortName for () {
    const SORT_NAME: &'static str = "Unit";
}
impl BaseSortName for sort::F {
    const SORT_NAME: &'static str = "f64";
}
impl BaseSortName for sort::S {
    const SORT_NAME: &'static str = "String";
}
impl BaseSortName for sort::Z {
    const SORT_NAME: &'static str = "BigInt";
}
impl BaseSortName for sort::Q {
    const SORT_NAME: &'static str = "BigRat";
}

// ---------------------------------------------------------------------
// ApiError — runtime check failures from the typed `Read` / `Write`
// trait methods and from [`crate::EGraph::with_full_state`] callers.
// ---------------------------------------------------------------------

/// Runtime errors from the typed Rust API surface.
///
/// These signal a misuse of the typed methods that the API can detect
/// dynamically — wrong table subtype, wrong arity, mismatched column
/// sorts, etc. They are *not* egglog typecheck errors and *not*
/// backend / e-graph failures; for those, see [`enum@crate::Error`].
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("no table named `{name}` is registered")]
    MissingTable { name: String },

    #[error(
        "table `{name}` is a {actual}; this method is only valid for {expected} tables"
    )]
    WrongSubtype {
        name: String,
        expected: &'static str,
        actual: &'static str,
    },

    #[error("table `{table}`: expected {expected} input columns, got {got}")]
    WrongArity {
        table: String,
        expected: usize,
        got: usize,
    },

    #[error(
        "table `{table}`, column {column}: expected sort `{expected}`, got value of sort `{actual}`"
    )]
    WrongColumnSort {
        table: String,
        column: usize,
        expected: String,
        actual: String,
    },

    #[error(
        "table `{table}` output: expected sort `{expected}`, got value of sort `{actual}`"
    )]
    WrongOutputSort {
        table: String,
        expected: String,
        actual: String,
    },

    #[error(
        "union: left value has sort `{left}`, right value has sort `{right}`"
    )]
    UnionSortMismatch { left: String, right: String },

    #[error("no egglog sort is registered for Rust type `{rust_type}`")]
    UnknownBaseSort { rust_type: &'static str },
}

mod sealed {
    pub trait Sealed {}
}

// ---------------------------------------------------------------------
// Id — runtime-tagged eclass / column value
// ---------------------------------------------------------------------

/// A [`Value`] tagged with the egglog sort it belongs to.
///
/// `Id` is what the typed API returns for constructor eclasses
/// ([`crate::Write::add_node`], [`crate::Read::eclass_of`]) and what it
/// expects on the way back in for eq-sort columns and unions. The sort
/// tag lets [`crate::Write::set`] / [`crate::Read::lookup`] / etc.
/// validate at runtime that an eclass id of sort `Math` isn't being
/// passed where a `List` is expected.
///
/// The sort name follows the egglog convention — `"Math"`, `"List"`,
/// `"i64"`, `"String"`, etc.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Id {
    value: Value,
    sort: Arc<str>,
}

impl Id {
    /// Wrap a raw [`Value`] with the given sort name. The caller
    /// asserts the value really is of that sort — no runtime check
    /// happens here; the tag is consulted later when the `Id` flows
    /// through a typed API call.
    pub fn new(value: Value, sort: impl Into<Arc<str>>) -> Self {
        Self { value, sort: sort.into() }
    }

    pub fn value(&self) -> Value {
        self.value
    }

    pub fn sort(&self) -> &str {
        &self.sort
    }
}

// ---------------------------------------------------------------------
// Input side: IntoRow + IntoColumn
// ---------------------------------------------------------------------

/// Runtime sort tag for a column on the input side. Used by the typed
/// API to validate inputs against a table's declared schema.
#[derive(Clone, Debug)]
pub enum ColumnSort {
    /// The column has a known sort name. Compared exactly against the
    /// table's expected sort for that column position.
    Named(Arc<str>),
    /// No sort information attached. Skip the runtime check — the
    /// caller is responsible for getting the column right. Used by
    /// the bare [`Value`] [`IntoColumn`] impl.
    Unchecked,
}

/// Convert a Rust value into a row of egglog [`Value`]s, with a sort
/// tag per column.
///
/// Implemented for:
/// - A bare [`IntoColumn`] value (e.g. `1_i64` or an [`Id`]) — produces
///   a single-column row.
/// - Tuples up to arity 8 of [`IntoColumn`] values.
/// - [`RawValues`] as an escape hatch for already-converted multi-column
///   rows (sort checks are skipped for every column).
pub trait IntoRow {
    fn into_values(self, bv: &BaseValues) -> Vec<Value>;
    /// Sort tags for each column, in order. Length must equal the
    /// `into_values` result length.
    fn column_sorts(&self) -> Vec<ColumnSort>;
}

/// A single column of an egglog row, on the input side.
///
/// This is a sealed trait — additional impls live in the egglog crate.
pub trait IntoColumn: sealed::Sealed {
    fn into_value(self, bv: &BaseValues) -> Value;
    /// Runtime sort tag for this column. Consulted by the typed API
    /// to validate against the table's declared schema.
    fn column_sort(&self) -> ColumnSort;
}

// ---------------------------------------------------------------------
// Output side: FromRow + FromColumn
// ---------------------------------------------------------------------

/// Convert a row of egglog [`Value`]s back into a Rust value.
///
/// Implemented for:
/// - `()` — discards the row, useful for "did this match" queries.
/// - A bare [`FromColumn`] type — extracts a single column.
/// - Tuples up to arity 8 of [`FromColumn`] types.
/// - `Vec<Value>` as an escape hatch for rows with non-base columns.
pub trait FromRow: Sized {
    fn from_values(values: &[Value], bv: &BaseValues) -> Self;
}

/// A single column of an egglog row, on the output side.
///
/// This is a sealed trait — additional impls live in the egglog crate.
pub trait FromColumn: sealed::Sealed {
    fn from_value(value: Value, bv: &BaseValues) -> Self;
}

// ---------------------------------------------------------------------
// Escape hatch: RawValues
// ---------------------------------------------------------------------

/// Escape hatch wrapper — pass already-converted [`Value`] columns when
/// the caller wants to bypass per-column sort checking entirely.
#[derive(Clone, Debug)]
pub struct RawValues(pub Vec<Value>);

impl IntoRow for RawValues {
    fn into_values(self, _bv: &BaseValues) -> Vec<Value> {
        self.0
    }
    fn column_sorts(&self) -> Vec<ColumnSort> {
        vec![ColumnSort::Unchecked; self.0.len()]
    }
}

impl FromRow for Vec<Value> {
    fn from_values(values: &[Value], _bv: &BaseValues) -> Self {
        values.to_vec()
    }
}

impl FromRow for () {
    fn from_values(_values: &[Value], _bv: &BaseValues) -> Self {}
}

// ---------------------------------------------------------------------
// Base column impls — symmetric for IntoColumn / FromColumn
// ---------------------------------------------------------------------

macro_rules! impl_column_for_base {
    ( $( ($ty:ty, $sort_name:literal) ),+ $(,)? ) => {
        $(
            impl sealed::Sealed for $ty {}
            impl IntoColumn for $ty {
                fn into_value(self, bv: &BaseValues) -> Value {
                    bv.get::<$ty>(self)
                }
                fn column_sort(&self) -> ColumnSort {
                    ColumnSort::Named(Arc::from($sort_name))
                }
            }
            impl FromColumn for $ty {
                fn from_value(value: Value, bv: &BaseValues) -> Self {
                    bv.unwrap::<$ty>(value)
                }
            }
        )+
    };
}

impl_column_for_base!(
    (i64, "i64"),
    (bool, "bool"),
    ((), "Unit"),
    (sort::F, "f64"),
    (sort::S, "String"),
    (sort::Z, "BigInt"),
    (sort::Q, "BigRat"),
);

// `String` is sugar — egglog's String sort uses `sort::S` (`Boxed<String>`).
impl sealed::Sealed for String {}
impl IntoColumn for String {
    fn into_value(self, bv: &BaseValues) -> Value {
        bv.get::<sort::S>(self.into())
    }
    fn column_sort(&self) -> ColumnSort {
        ColumnSort::Named(Arc::from("String"))
    }
}
impl FromColumn for String {
    fn from_value(value: Value, bv: &BaseValues) -> Self {
        bv.unwrap::<sort::S>(value).0
    }
}

// `&str` is one-directional input sugar.
impl sealed::Sealed for &str {}
impl IntoColumn for &str {
    fn into_value(self, bv: &BaseValues) -> Value {
        bv.get::<sort::S>(self.to_string().into())
    }
    fn column_sort(&self) -> ColumnSort {
        ColumnSort::Named(Arc::from("String"))
    }
}

// `f64` is sugar for `sort::F`.
impl sealed::Sealed for f64 {}
impl IntoColumn for f64 {
    fn into_value(self, bv: &BaseValues) -> Value {
        use ordered_float::OrderedFloat;
        bv.get::<sort::F>(OrderedFloat(self).into())
    }
    fn column_sort(&self) -> ColumnSort {
        ColumnSort::Named(Arc::from("f64"))
    }
}
impl FromColumn for f64 {
    fn from_value(value: Value, bv: &BaseValues) -> Self {
        bv.unwrap::<sort::F>(value).0.0
    }
}

// `Id` carries its sort tag from construction.
impl sealed::Sealed for Id {}
impl IntoColumn for Id {
    fn into_value(self, _bv: &BaseValues) -> Value {
        self.value
    }
    fn column_sort(&self) -> ColumnSort {
        ColumnSort::Named(self.sort.clone())
    }
}

// Bare `Value` is the unchecked escape hatch.
impl sealed::Sealed for Value {}
impl IntoColumn for Value {
    fn into_value(self, _bv: &BaseValues) -> Value {
        self
    }
    fn column_sort(&self) -> ColumnSort {
        ColumnSort::Unchecked
    }
}
impl FromColumn for Value {
    fn from_value(value: Value, _bv: &BaseValues) -> Self {
        value
    }
}

// ---------------------------------------------------------------------
// Single-column blanket impls
// ---------------------------------------------------------------------

impl<A: IntoColumn> IntoRow for A {
    fn into_values(self, bv: &BaseValues) -> Vec<Value> {
        vec![self.into_value(bv)]
    }
    fn column_sorts(&self) -> Vec<ColumnSort> {
        vec![self.column_sort()]
    }
}

// Note: no blanket `impl<A: FromColumn> FromRow for A` — would conflict
// with the `Vec<Value>` impl. Single-column outputs use the (A,) tuple form.

// ---------------------------------------------------------------------
// Tuple impls — symmetric for IntoRow / FromRow
// ---------------------------------------------------------------------

macro_rules! impl_row_for_tuple {
    ( $( ($($name:ident),+) ),+ $(,)? ) => {
        $(
            #[allow(non_snake_case)]
            impl<$($name: IntoColumn),+> IntoRow for ($($name,)+) {
                fn into_values(self, bv: &BaseValues) -> Vec<Value> {
                    let ($($name,)+) = self;
                    vec![ $( $name.into_value(bv) ),+ ]
                }
                fn column_sorts(&self) -> Vec<ColumnSort> {
                    let ($($name,)+) = self;
                    vec![ $( $name.column_sort() ),+ ]
                }
            }

            #[allow(non_snake_case)]
            impl<$($name: FromColumn),+> FromRow for ($($name,)+) {
                fn from_values(values: &[Value], bv: &BaseValues) -> Self {
                    let arity = [$(stringify!($name)),+].len();
                    assert!(
                        values.len() == arity,
                        "FromRow: expected {} values for tuple of arity {}, got {} (use Vec<Value> or () to discard extras)",
                        arity, arity, values.len(),
                    );
                    let mut iter = values.iter().copied();
                    ( $( $name::from_value(iter.next().unwrap(), bv), )+ )
                }
            }
        )+
    };
}

impl_row_for_tuple! {
    (T1),
    (T1, T2),
    (T1, T2, T3),
    (T1, T2, T3, T4),
    (T1, T2, T3, T4, T5),
    (T1, T2, T3, T4, T5, T6),
    (T1, T2, T3, T4, T5, T6, T7),
    (T1, T2, T3, T4, T5, T6, T7, T8),
}
