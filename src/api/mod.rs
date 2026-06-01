//! Typed row encoding for the [`crate::EGraph`] surface API.
//!
//! [`IntoRow`] / [`IntoColumn`] convert Rust values into egglog row
//! data on the way *in* (insert, lookup keys, query patterns). Reads
//! always come back as `Vec<Id>` (or `Option<Id>` for single-result
//! reads), with each [`Id`] tagged with the column's declared sort —
//! users don't supply a Rust output type at the read site. Convert
//! an [`Id`] to a Rust base value via [`crate::EGraph::extract::<T>`]
//! at the system boundary.
//!
//! Every input column carries a runtime [`ColumnSort`] tag. The trait
//! methods that consume rows ([`crate::Read::lookup`],
//! [`crate::Write::set`], etc.) validate each column's tag against the
//! table's declared schema and return [`crate::ApiError::WrongColumnSort`]
//! on mismatch. Base values (`i64`, `String`, `f64`, …) tag with the
//! corresponding egglog sort name automatically; eclass ids and base
//! values held by users come through the [`Id`] wrapper, which pairs
//! a value with a runtime sort name.

use std::sync::Arc;

use crate::core_relations::{BaseValue, BaseValues, Value};
use crate::sort;
use smallvec::{smallvec, SmallVec};
use thiserror::Error;

/// Row buffer used by [`IntoRow::into_values`]. Inline up to 4 columns
/// so common keys (1–3 args + an output) don't allocate.
pub type RowValues = SmallVec<[Value; 4]>;
/// Sort tags returned by [`IntoRow::column_sorts`]. Same inline budget
/// as [`RowValues`].
pub type RowSorts = SmallVec<[ColumnSort; 4]>;

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
    /// Per-type cached `Arc<str>` for [`Self::SORT_NAME`]. Returning a
    /// clone of a static avoids allocating a fresh `Arc<str>` on every
    /// [`crate::Core::intern_typed`] call — see issue PR #901 perf
    /// discussion.
    fn sort_name_arc() -> Arc<str>;
}

macro_rules! impl_base_sort_name {
    ($($ty:ty => $name:literal),+ $(,)?) => {
        $(
            impl BaseSortName for $ty {
                const SORT_NAME: &'static str = $name;
                fn sort_name_arc() -> Arc<str> {
                    static CACHE: std::sync::OnceLock<Arc<str>> = std::sync::OnceLock::new();
                    CACHE.get_or_init(|| Arc::from($name)).clone()
                }
            }
        )+
    };
}

impl_base_sort_name! {
    i64 => "i64",
    bool => "bool",
    () => "Unit",
    sort::F => "f64",
    sort::S => "String",
    sort::Z => "BigInt",
    sort::Q => "BigRat",
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

/// Shared empty `Arc<str>` cloned by transient untagged ids inside the
/// crate (primitive dispatch wrappers). Never observable from outside —
/// every user-facing `Id` carries a real sort name.
pub(crate) fn empty_sort() -> Arc<str> {
    static EMPTY: std::sync::OnceLock<Arc<str>> = std::sync::OnceLock::new();
    EMPTY.get_or_init(|| Arc::from("")).clone()
}

impl Id {
    /// Wrap a raw [`Value`] with the given sort name. The caller
    /// asserts the value really is of that sort — no runtime check
    /// happens here; the tag is consulted later when the `Id` flows
    /// through a typed API call.
    pub fn new(value: Value, sort: impl Into<Arc<str>>) -> Self {
        Self {
            value,
            sort: sort.into(),
        }
    }

    /// Transient untagged id for internal dispatch use only. The
    /// primitive wrapper produces these when handing `&[Value]` to a
    /// primitive's `apply`; the wrapper immediately unwraps the
    /// returned id via [`Id::value`] before anything sort-checked sees
    /// it, so users never observe an untagged id.
    #[inline]
    pub(crate) fn untagged(value: Value) -> Self {
        Self {
            value,
            sort: empty_sort(),
        }
    }

    /// Internal constructor for read paths — the sort `Arc<str>` is
    /// reused from the table's stored schema (one atomic increment, no
    /// allocation).
    #[inline]
    pub(crate) fn with_sort(value: Value, sort: Arc<str>) -> Self {
        Self { value, sort }
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
    /// Compile-time sort name. Same semantics as [`ColumnSort::Named`]
    /// but avoids an `Arc` allocation on the hot path — base type
    /// [`IntoColumn`] impls use this.
    Static(&'static str),
    /// No sort information attached. Skip the runtime check — the
    /// caller is responsible for getting the column right. Used by
    /// the bare [`Value`] [`IntoColumn`] impl.
    Unchecked,
}

impl ColumnSort {
    /// Returns the sort name if this is a checked variant.
    #[inline]
    pub(crate) fn name(&self) -> Option<&str> {
        match self {
            ColumnSort::Named(s) => Some(s.as_ref()),
            ColumnSort::Static(s) => Some(s),
            ColumnSort::Unchecked => None,
        }
    }
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
    fn into_values(self, bv: &BaseValues) -> RowValues;
    /// Sort tags for each column, in order. Length must equal the
    /// `into_values` result length.
    fn column_sorts(&self) -> RowSorts;
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
// Escape hatch: RawValues
// ---------------------------------------------------------------------

/// Escape hatch wrapper — pass already-converted [`Value`] columns when
/// the caller wants to bypass per-column sort checking entirely.
#[derive(Clone, Debug)]
pub struct RawValues(pub Vec<Value>);

impl IntoRow for RawValues {
    fn into_values(self, _bv: &BaseValues) -> RowValues {
        self.0.into_iter().collect()
    }
    fn column_sorts(&self) -> RowSorts {
        smallvec![ColumnSort::Unchecked; self.0.len()]
    }
}

// ---------------------------------------------------------------------
// Base column impls
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
                    ColumnSort::Static($sort_name)
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
        ColumnSort::Static("String")
    }
}
// `&str` is one-directional input sugar.
impl sealed::Sealed for &str {}
impl IntoColumn for &str {
    fn into_value(self, bv: &BaseValues) -> Value {
        bv.get::<sort::S>(self.to_string().into())
    }
    fn column_sort(&self) -> ColumnSort {
        ColumnSort::Static("String")
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
        ColumnSort::Static("f64")
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
// ---------------------------------------------------------------------
// Single-column blanket impls
// ---------------------------------------------------------------------

impl<A: IntoColumn> IntoRow for A {
    fn into_values(self, bv: &BaseValues) -> RowValues {
        smallvec![self.into_value(bv)]
    }
    fn column_sorts(&self) -> RowSorts {
        smallvec![self.column_sort()]
    }
}

// ---------------------------------------------------------------------
// Tuple impls — IntoRow only (FromRow was dropped; reads always
// return `Vec<Id>` so users never write a tuple type for a read).
// ---------------------------------------------------------------------

macro_rules! impl_row_for_tuple {
    ( $( ($($name:ident),+) ),+ $(,)? ) => {
        $(
            #[allow(non_snake_case)]
            impl<$($name: IntoColumn),+> IntoRow for ($($name,)+) {
                fn into_values(self, bv: &BaseValues) -> RowValues {
                    let ($($name,)+) = self;
                    smallvec![ $( $name.into_value(bv) ),+ ]
                }
                fn column_sorts(&self) -> RowSorts {
                    let ($($name,)+) = self;
                    smallvec![ $( $name.column_sort() ),+ ]
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
