//! DuckDB-backed executor for a small subset of egglog's resolved IR.
//!
//! Phase 1.1 scope: relations + functions with outputs and merge
//! modes (`:merge old`, `:merge new`). No term encoding, no UF, no
//! primitives. See `../duckdb-backend-plan.md` for the full plan.
//!
//! Design notes:
//! - One DuckDB table per registered relation/function.
//! - Every table carries a `ts BIGINT NOT NULL` column for seminaive.
//! - `next_ts` and `last_run_at_<rule>` live in Rust state; they are
//!   bind parameters in generated SQL, never database rows.
//! - Each rule with N function-table atoms compiles to N seminaive
//!   variants (one per focused atom), emitted as separate prepared
//!   `INSERT INTO target SELECT ...` statements with appropriate
//!   `ON CONFLICT` clauses depending on merge mode.

use anyhow::{Result, anyhow};
use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use duckdb::{Connection, ToSql};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub mod uf;
use uf::UfTable;

/// DuckDB scalar UDF that wraps `UfTable::find_ro`. Registered as
/// `duck_uf_<sort>_find(x BIGINT) -> BIGINT` for each native-UF
/// function. Uses the read-only walk (no path compression) so the
/// UDF can run from inside a SELECT without taking a write lock
/// on the table; we re-compress periodically off the hot path.
struct UfFindScalar;

impl VScalar for UfFindScalar {
    type State = Arc<Mutex<UfTable>>;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let n = input.len();
            let input_vec = input.flat_vector(0);
            let inputs = input_vec.as_slice_with_len::<i64>(n);
            let mut output_vec = output.flat_vector();
            let outputs = output_vec.as_mut_slice::<i64>();
            let uf = state.lock().unwrap();
            for i in 0..n {
                outputs[i] = uf.find_ro(inputs[i]);
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeHandle::from(LogicalTypeId::Bigint)],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// State shared by all bigint/bigrat-related UDFs: a snapshot of the
/// base-value pool whose interior intern tables are Arc-shared with
/// the EGraph's pool, plus the resolved `BaseValueId`s for `Z` and
/// (when registered) `Q`. Cloning is cheap and preserves intern
/// sharing — every clone agrees on Value handles, so a `Q` interned
/// by the UDF at row-time appears in the EGraph's pool by the time a
/// rule action consumes it.
///
/// `bigrat_ty` is `None` until `BigRatSort` has been registered. Since
/// `BigIntSort` registers first (and Herbie programs need both to do
/// anything), the bigrat UDFs delay registration to
/// `set_external_func_name`-time when the `Q` id is known to exist.
#[derive(Clone)]
struct BigPoolState {
    pool: base_values::DuckdbBaseValuePool,
    bigint_ty: egglog_backend_trait::BaseValueId,
    bigrat_ty: Option<egglog_backend_trait::BaseValueId>,
}

/// UDF for the `from-string` primitive (egglog's `S -> Z` constructor
/// for `BigInt`). Takes a VARCHAR row, parses it as a `BigInt`, and
/// interns the result in the shared base-value pool. Returns the
/// `Value`'s `u32` rep widened to `BIGINT`.
///
/// On parse failure: the bridge backend's `from-string` is fallible
/// and returns `None`; we emit `NULL` so a downstream rule action
/// referencing the result drops out via standard SQL NULL propagation.
struct FromStringScalar;

impl VScalar for FromStringScalar {
    type State = BigPoolState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            use egglog_backend_trait::{BaseValuePool, Value};
            use egglog_numeric_id::NumericId;
            use num::BigInt;
            use std::any::TypeId;
            use std::str::FromStr;

            // Strings on duck are stored as interned `Boxed<String>`
            // handles in BIGINT columns (matches bridge encoding). So
            // this UDF's input is a BIGINT handle, not a raw VARCHAR.
            // Unwrap the handle to get the string, then parse as BigInt.
            let string_ty = state
                .pool
                .get_ty_by_type_id(TypeId::of::<egglog_core_relations::Boxed<String>>());

            let n = input.len();
            let in_vec = input.flat_vector(0);
            let in_slice = in_vec.as_slice_with_len::<i64>(n);

            let mut results: Vec<Option<i64>> = Vec::with_capacity(n);
            for i in 0..n {
                let handle = Value::new(in_slice[i] as u32);
                // A garbage (out-of-range) handle => `None` => SQL NULL for this
                // row (a discarded speculative `ON CONFLICT` row); real handles
                // unwrap normally.
                if !state.pool.handle_in_range(string_ty, handle) {
                    results.push(None);
                    continue;
                }
                let boxed = state.pool.unwrap_dyn(string_ty, handle);
                let Some(s_boxed) = boxed.downcast_ref::<egglog_core_relations::Boxed<String>>()
                else {
                    results.push(None);
                    continue;
                };
                // Mirror the bridge's `-?>` semantics: parse failure
                // means the rule firing should not produce a value.
                // Encode as NULL so downstream actions/filters drop
                // rows that resolve to NULL.
                results.push(BigInt::from_str(s_boxed.0.as_str()).ok().map(|bi| {
                    let z = egglog_core_relations::Boxed::new(bi);
                    let val = state.pool.intern_dyn(state.bigint_ty, Box::new(z));
                    val.rep() as i64
                }));
            }

            let mut out_vec = output.flat_vector();
            {
                let out_slice = out_vec.as_mut_slice::<i64>();
                for (i, r) in results.iter().enumerate() {
                    out_slice[i] = r.unwrap_or(0);
                }
            }
            for (i, r) in results.iter().enumerate() {
                if r.is_none() {
                    out_vec.set_null(i);
                }
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeHandle::from(LogicalTypeId::Bigint)],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// UDF for the `bigint` primitive (egglog's `i64 -> Z` constructor for
/// `BigInt`). Takes a BIGINT row holding a raw `i64`, widens it into a
/// `num::BigInt`, and interns the result as `Z = Boxed<BigInt>` in the
/// shared base-value pool. Returns the `Value`'s `u32` rep widened to
/// `BIGINT`. Total (infallible), so no NULL handling like `from-string`.
struct BigintConstructorScalar;

impl VScalar for BigintConstructorScalar {
    type State = BigPoolState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            use egglog_backend_trait::BaseValuePool;
            use egglog_numeric_id::NumericId;
            use num::BigInt;

            let n = input.len();
            let in_vec = input.flat_vector(0);
            let in_slice = in_vec.as_slice_with_len::<i64>(n);

            let mut out_vec = output.flat_vector();
            let out_slice = out_vec.as_mut_slice::<i64>();

            for i in 0..n {
                let z = egglog_core_relations::Boxed::new(BigInt::from(in_slice[i]));
                let val = state.pool.intern_dyn(state.bigint_ty, Box::new(z));
                out_slice[i] = val.rep() as i64;
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeHandle::from(LogicalTypeId::Bigint)],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// UDF for the `bigrat` primitive (egglog's `Z × Z -> Q` constructor
/// for `BigRat`). Inputs are two `BIGINT` columns holding `Z` handles
/// produced by `__egglog_from_string` (or any other source); the UDF
/// unwraps each to a `num::BigInt`, builds a `num::BigRational`, and
/// interns it as `Q = Boxed<BigRational>` in the shared pool.
///
/// State requires `bigrat_ty` to be `Some(...)` — the EGraph's
/// `set_external_func_name` path checks this before registering the
/// UDF (BigRatSort always registers before any `bigrat` call site).
struct BigratScalar;

impl VScalar for BigratScalar {
    type State = BigPoolState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            use egglog_backend_trait::{BaseValuePool, Value};
            use egglog_numeric_id::NumericId;
            use num::{BigInt, BigRational};

            let bigrat_ty = state
                .bigrat_ty
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    "BigratScalar: bigrat type id missing (BigRatSort not registered before \
                 __egglog_bigrat was called)"
                        .into()
                })?;

            let n = input.len();
            let num_col = input.flat_vector(0);
            let den_col = input.flat_vector(1);
            let num_slice = num_col.as_slice_with_len::<i64>(n);
            let den_slice = den_col.as_slice_with_len::<i64>(n);

            let mut out_vec = output.flat_vector();
            let out_slice = out_vec.as_mut_slice::<i64>();

            for i in 0..n {
                let num_val = Value::new(num_slice[i] as u32);
                let den_val = Value::new(den_slice[i] as u32);
                let num_boxed = state.pool.unwrap_dyn(state.bigint_ty, num_val);
                let den_boxed = state.pool.unwrap_dyn(state.bigint_ty, den_val);
                let num_bigint: &egglog_core_relations::Boxed<BigInt> = num_boxed
                    .downcast_ref()
                    .ok_or_else(|| -> Box<dyn std::error::Error> {
                        "BigratScalar: numerator was not a Boxed<BigInt>".into()
                    })?;
                let den_bigint: &egglog_core_relations::Boxed<BigInt> = den_boxed
                    .downcast_ref()
                    .ok_or_else(|| -> Box<dyn std::error::Error> {
                        "BigratScalar: denominator was not a Boxed<BigInt>".into()
                    })?;
                let q = egglog_core_relations::Boxed::new(BigRational::new(
                    num_bigint.0.clone(),
                    den_bigint.0.clone(),
                ));
                let val = state.pool.intern_dyn(bigrat_ty, Box::new(q));
                out_slice[i] = val.rep() as i64;
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
            ],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// BigRat operations that need a UDF wrapper because their egglog
/// name (`+`, `<`, `round`, ...) is overloaded across sorts and DuckDB
/// SQL has no native concept of arbitrary-precision rationals. The
/// frontend renames each call site to a sort-specific duck name (e.g.
/// `bigrat-add`); each variant of this enum corresponds to one such
/// name and selects the operation the UDF performs.
///
/// Unary ops (`Neg`/`Abs`/…) take one `Q` handle and return one. Binary
/// ops (`Add`/…/`Pow`) take two. Comparison ops (`Lt`/`Gt`/…) take two
/// `Q` handles and return BOOLEAN. `Numer`/`Denom` return a `Z` handle.
/// `ToF64` returns a DOUBLE.
#[derive(Copy, Clone, Debug)]
enum BigratOp {
    // Q × Q → Q (fallible for +, -, *, /, pow via checked_*)
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Pow,
    // Q → Q
    Neg,
    Abs,
    Floor,
    Ceil,
    Round,
    Sqrt,
    Log,
    Cbrt,
    // Q × Q → BOOL
    Lt,
    Gt,
    Le,
    Ge,
    // Q → Z
    Numer,
    Denom,
    // Q → f64
    ToF64,
}

impl BigratOp {
    fn from_duck_name(name: &str) -> Option<Self> {
        Some(match name {
            "bigrat-add" => BigratOp::Add,
            "bigrat-sub" => BigratOp::Sub,
            "bigrat-mul" => BigratOp::Mul,
            "bigrat-div" => BigratOp::Div,
            "bigrat-min" => BigratOp::Min,
            "bigrat-max" => BigratOp::Max,
            "bigrat-pow" => BigratOp::Pow,
            "bigrat-neg" => BigratOp::Neg,
            "bigrat-abs" => BigratOp::Abs,
            "bigrat-floor" => BigratOp::Floor,
            "bigrat-ceil" => BigratOp::Ceil,
            "bigrat-round" => BigratOp::Round,
            "bigrat-sqrt" => BigratOp::Sqrt,
            "bigrat-log" => BigratOp::Log,
            "bigrat-cbrt" => BigratOp::Cbrt,
            "bigrat-lt" => BigratOp::Lt,
            "bigrat-gt" => BigratOp::Gt,
            "bigrat-le" => BigratOp::Le,
            "bigrat-ge" => BigratOp::Ge,
            "bigrat-numer" => BigratOp::Numer,
            "bigrat-denom" => BigratOp::Denom,
            "bigrat-to-f64" => BigratOp::ToF64,
            _ => return None,
        })
    }

    fn is_unary(&self) -> bool {
        use BigratOp::*;
        matches!(
            self,
            Neg | Abs | Floor | Ceil | Round | Sqrt | Log | Cbrt | Numer | Denom | ToF64
        )
    }

    fn is_comparison(&self) -> bool {
        use BigratOp::*;
        matches!(self, Lt | Gt | Le | Ge)
    }

    fn returns_z(&self) -> bool {
        matches!(self, BigratOp::Numer | BigratOp::Denom)
    }

    fn returns_f64(&self) -> bool {
        matches!(self, BigratOp::ToF64)
    }
}

/// State for the family of bigrat UDFs. Holds a clone of the pool
/// (intern tables Arc-shared with the EGraph), the `Z` and `Q`
/// `BaseValueId`s, and the specific op this UDF instance performs.
#[derive(Clone)]
struct BigratExecState {
    pool: base_values::DuckdbBaseValuePool,
    bigint_ty: egglog_backend_trait::BaseValueId,
    bigrat_ty: egglog_backend_trait::BaseValueId,
    op: BigratOp,
}

/// Run a unary or binary BigRat → BigRat operation. Returns
/// `Some(Q)` on success, `None` for fallible ops on bad inputs
/// (division by zero, `pow` with fractional exponent, etc.) so the
/// UDF can emit SQL NULL and downstream rules drop out via NULL
/// propagation. Mirrors the bridge-side closures in
/// `egglog::sort::bigrat::register_primitives`.
fn run_bigrat_q(op: BigratOp, args: &[num::BigRational]) -> Option<num::BigRational> {
    use num::traits::{One, Signed, Zero};
    use num::{BigInt, BigRational};
    let one = || BigRational::one();
    let zero = || BigRational::zero();
    let _ = one;
    let _ = zero;
    match op {
        BigratOp::Add => Some(&args[0] + &args[1]),
        BigratOp::Sub => Some(&args[0] - &args[1]),
        BigratOp::Mul => Some(&args[0] * &args[1]),
        BigratOp::Div => {
            if args[1].is_zero() {
                None
            } else {
                Some(&args[0] / &args[1])
            }
        }
        BigratOp::Min => Some(args[0].clone().min(args[1].clone())),
        BigratOp::Max => Some(args[0].clone().max(args[1].clone())),
        BigratOp::Pow => {
            let a = &args[0];
            let b = &args[1];
            if !b.is_integer() {
                return None;
            }
            if a.is_zero() {
                if b.is_zero() {
                    return Some(BigRational::one());
                }
                if b.numer() > &BigInt::from(0) {
                    return Some(BigRational::zero());
                }
                return None;
            }
            let is_neg = b.numer() < &BigInt::from(0);
            let adj_base = if is_neg { a.recip() } else { a.clone() };
            let adj_exp = if is_neg {
                BigRational::new(-b.numer().clone(), b.denom().clone())
            } else {
                b.clone()
            };
            let exp_i64 = adj_exp.numer().to_string().parse::<i64>().ok()?;
            if exp_i64 < 0 {
                return None;
            }
            let exp = exp_i64 as usize;
            num::traits::checked_pow(adj_base, exp)
        }
        BigratOp::Neg => Some(-args[0].clone()),
        BigratOp::Abs => Some(BigRational::new(
            args[0].numer().abs().clone(),
            args[0].denom().clone(),
        )),
        BigratOp::Floor => Some(args[0].floor()),
        BigratOp::Ceil => Some(args[0].ceil()),
        BigratOp::Round => Some(args[0].round()),
        BigratOp::Sqrt => {
            // Closed-form sqrt only if both numer and denom are
            // perfect squares — matches the bridge's behavior in
            // `bigrat.rs`.
            if args[0].numer() < &BigInt::from(0) {
                return None;
            }
            let n_sqrt = args[0].numer().sqrt();
            let d_sqrt = args[0].denom().sqrt();
            if &(n_sqrt.clone() * n_sqrt.clone()) == args[0].numer()
                && &(d_sqrt.clone() * d_sqrt.clone()) == args[0].denom()
            {
                Some(BigRational::new(n_sqrt, d_sqrt))
            } else {
                None
            }
        }
        BigratOp::Log => {
            // Bridge only handles `log(1) = 0`; everything else
            // panics. Match that conservatively (just return None).
            if args[0].numer() == &BigInt::from(1) && args[0].denom() == &BigInt::from(1) {
                Some(BigRational::zero())
            } else {
                None
            }
        }
        BigratOp::Cbrt => {
            // Bridge only handles `cbrt(1) = 1`.
            if args[0].numer() == &BigInt::from(1) && args[0].denom() == &BigInt::from(1) {
                Some(BigRational::one())
            } else {
                None
            }
        }
        BigratOp::Lt
        | BigratOp::Gt
        | BigratOp::Le
        | BigratOp::Ge
        | BigratOp::Numer
        | BigratOp::Denom
        | BigratOp::ToF64 => unreachable!("not a Q-returning op: {op:?}"),
    }
}

/// Unwrap a `Q`-handle (BIGINT) row into a `BigRational`.
fn unwrap_bigrat(
    pool: &base_values::DuckdbBaseValuePool,
    bigrat_ty: egglog_backend_trait::BaseValueId,
    raw: i64,
) -> std::result::Result<num::BigRational, Box<dyn std::error::Error>> {
    use egglog_backend_trait::{BaseValuePool, Value};
    use egglog_numeric_id::NumericId;
    let val = Value::new(raw as u32);
    let boxed = pool.unwrap_dyn(bigrat_ty, val);
    let q: &egglog_core_relations::Boxed<num::BigRational> =
        boxed
            .downcast_ref()
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                "expected Boxed<BigRational> from pool".into()
            })?;
    Ok(q.0.clone())
}

/// UDF for unary `Q → Q` bigrat operations. Variants of [`BigratOp`]
/// with `is_unary() && !returns_z() && !returns_f64() && !is_comparison()`.
struct BigratUnaryQScalar;

impl VScalar for BigratUnaryQScalar {
    type State = BigratExecState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            use egglog_backend_trait::BaseValuePool;
            use egglog_numeric_id::NumericId;
            let n = input.len();
            let in_vec = input.flat_vector(0);
            let in_slice = in_vec.as_slice_with_len::<i64>(n);

            let mut results: Vec<Option<i64>> = Vec::with_capacity(n);
            for i in 0..n {
                let q = unwrap_bigrat(&state.pool, state.bigrat_ty, in_slice[i])?;
                let out = run_bigrat_q(state.op, &[q]);
                results.push(out.map(|r| {
                    let boxed = egglog_core_relations::Boxed::new(r);
                    state
                        .pool
                        .intern_dyn(state.bigrat_ty, Box::new(boxed))
                        .rep() as i64
                }));
            }
            let mut out_vec = output.flat_vector();
            {
                let out_slice = out_vec.as_mut_slice::<i64>();
                for (i, r) in results.iter().enumerate() {
                    out_slice[i] = r.unwrap_or(0);
                }
            }
            for (i, r) in results.iter().enumerate() {
                if r.is_none() {
                    out_vec.set_null(i);
                }
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeHandle::from(LogicalTypeId::Bigint)],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// UDF for binary `Q × Q → Q` bigrat operations.
struct BigratBinaryQScalar;

impl VScalar for BigratBinaryQScalar {
    type State = BigratExecState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            use egglog_backend_trait::BaseValuePool;
            use egglog_numeric_id::NumericId;
            let n = input.len();
            let a_vec = input.flat_vector(0);
            let b_vec = input.flat_vector(1);
            let a_slice = a_vec.as_slice_with_len::<i64>(n);
            let b_slice = b_vec.as_slice_with_len::<i64>(n);

            let mut results: Vec<Option<i64>> = Vec::with_capacity(n);
            for i in 0..n {
                let a = unwrap_bigrat(&state.pool, state.bigrat_ty, a_slice[i])?;
                let b = unwrap_bigrat(&state.pool, state.bigrat_ty, b_slice[i])?;
                let out = run_bigrat_q(state.op, &[a, b]);
                results.push(out.map(|r| {
                    let boxed = egglog_core_relations::Boxed::new(r);
                    state
                        .pool
                        .intern_dyn(state.bigrat_ty, Box::new(boxed))
                        .rep() as i64
                }));
            }
            let mut out_vec = output.flat_vector();
            {
                let out_slice = out_vec.as_mut_slice::<i64>();
                for (i, r) in results.iter().enumerate() {
                    out_slice[i] = r.unwrap_or(0);
                }
            }
            for (i, r) in results.iter().enumerate() {
                if r.is_none() {
                    out_vec.set_null(i);
                }
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
            ],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// UDF for `Q × Q → BOOL` bigrat comparison operations. Returns
/// BOOLEAN; the rule_builder side wraps it in a `Filter` atom, which
/// becomes a SQL `WHERE` clause.
struct BigratCmpScalar;

impl VScalar for BigratCmpScalar {
    type State = BigratExecState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let n = input.len();
            let a_vec = input.flat_vector(0);
            let b_vec = input.flat_vector(1);
            let a_slice = a_vec.as_slice_with_len::<i64>(n);
            let b_slice = b_vec.as_slice_with_len::<i64>(n);

            let mut out_vec = output.flat_vector();
            let out_slice = out_vec.as_mut_slice::<bool>();
            for i in 0..n {
                let a = unwrap_bigrat(&state.pool, state.bigrat_ty, a_slice[i])?;
                let b = unwrap_bigrat(&state.pool, state.bigrat_ty, b_slice[i])?;
                out_slice[i] = match state.op {
                    BigratOp::Lt => a < b,
                    BigratOp::Gt => a > b,
                    BigratOp::Le => a <= b,
                    BigratOp::Ge => a >= b,
                    _ => unreachable!("not a comparison op: {:?}", state.op),
                };
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
            ],
            LogicalTypeHandle::from(LogicalTypeId::Boolean),
        )]
    }
}

/// UDF for `Q → Z` (`numer`/`denom`).
struct BigratNumDenomScalar;

impl VScalar for BigratNumDenomScalar {
    type State = BigratExecState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            use egglog_backend_trait::BaseValuePool;
            use egglog_numeric_id::NumericId;
            let n = input.len();
            let in_vec = input.flat_vector(0);
            let in_slice = in_vec.as_slice_with_len::<i64>(n);

            let mut out_vec = output.flat_vector();
            let out_slice = out_vec.as_mut_slice::<i64>();
            for i in 0..n {
                let q = unwrap_bigrat(&state.pool, state.bigrat_ty, in_slice[i])?;
                let bi = match state.op {
                    BigratOp::Numer => q.numer().clone(),
                    BigratOp::Denom => q.denom().clone(),
                    _ => unreachable!(),
                };
                let z = egglog_core_relations::Boxed::new(bi);
                out_slice[i] = state.pool.intern_dyn(state.bigint_ty, Box::new(z)).rep() as i64;
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeHandle::from(LogicalTypeId::Bigint)],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// BigInt binary `Z × Z → Z` operations that need a UDF wrapper for
/// the same reason as [`BigratOp`]: their egglog name (`+`, `*`, …) is
/// overloaded across sorts, and DuckDB SQL has no arbitrary-precision
/// integers (a raw SQL `+` on the BIGINT *handle* columns would do
/// pool-index arithmetic, not BigInt math). The frontend renames each
/// BigInt call site to a sort-specific duck name (`bigint-mul`, …);
/// each variant here selects the operation the UDF performs.
#[derive(Copy, Clone, Debug)]
enum BigintOp {
    Add,
    Sub,
    Mul,
    // Div / Rem are fallible (None on divide-by-zero), matching the
    // bridge's `-?>` semantics in `egglog::sort::bigint`.
    Div,
    Rem,
    And,
    Or,
    Xor,
    Min,
    Max,
}

impl BigintOp {
    fn from_duck_name(name: &str) -> Option<Self> {
        Some(match name {
            "bigint-add" => BigintOp::Add,
            "bigint-sub" => BigintOp::Sub,
            "bigint-mul" => BigintOp::Mul,
            "bigint-div" => BigintOp::Div,
            "bigint-rem" => BigintOp::Rem,
            "bigint-and" => BigintOp::And,
            "bigint-or" => BigintOp::Or,
            "bigint-xor" => BigintOp::Xor,
            "bigint-min" => BigintOp::Min,
            "bigint-max" => BigintOp::Max,
            _ => return None,
        })
    }
}

/// State for the family of bigint arithmetic UDFs. Holds a clone of
/// the pool (intern tables Arc-shared with the EGraph), the `Z`
/// `BaseValueId`, and the specific op this UDF instance performs.
#[derive(Clone)]
struct BigintExecState {
    pool: base_values::DuckdbBaseValuePool,
    bigint_ty: egglog_backend_trait::BaseValueId,
    op: BigintOp,
}

/// Run a binary BigInt → BigInt operation. Returns `Some(Z)` on
/// success, `None` for fallible ops on bad inputs (`/` or `%` by zero)
/// so the UDF emits SQL NULL and downstream rules drop via NULL
/// propagation. Mirrors the bridge-side closures in
/// `egglog::sort::bigint::register_primitives`.
fn run_bigint(op: BigintOp, a: &num::BigInt, b: &num::BigInt) -> Option<num::BigInt> {
    use num::Zero;
    Some(match op {
        BigintOp::Add => a + b,
        BigintOp::Sub => a - b,
        BigintOp::Mul => a * b,
        BigintOp::Div => {
            if b.is_zero() {
                return None;
            }
            a / b
        }
        BigintOp::Rem => {
            if b.is_zero() {
                return None;
            }
            a % b
        }
        BigintOp::And => a & b,
        BigintOp::Or => a | b,
        BigintOp::Xor => a ^ b,
        BigintOp::Min => a.min(b).clone(),
        BigintOp::Max => a.max(b).clone(),
    })
}

/// Unwrap a `Z`-handle (BIGINT) row into a `BigInt`. Returns `None` for an
/// OUT-OF-RANGE handle — a garbage value DuckDB's speculative `ON CONFLICT`
/// evaluation can hand us for discarded (non-conflicting) rows — so the caller
/// emits SQL NULL for that row instead of crashing on an out-of-bounds pool
/// lookup. A legitimately-stored handle is always in range, so real values are
/// unaffected.
fn unwrap_bigint(
    pool: &base_values::DuckdbBaseValuePool,
    bigint_ty: egglog_backend_trait::BaseValueId,
    raw: i64,
) -> Option<num::BigInt> {
    use egglog_backend_trait::{BaseValuePool, Value};
    use egglog_numeric_id::NumericId;
    let val = Value::new(raw as u32);
    if !pool.handle_in_range(bigint_ty, val) {
        return None;
    }
    let boxed = pool.unwrap_dyn(bigint_ty, val);
    let z: &egglog_core_relations::Boxed<num::BigInt> = boxed.downcast_ref()?;
    Some(z.0.clone())
}

/// UDF for binary `Z × Z → Z` bigint operations. Variants of
/// [`BigintOp`]; fallible ops emit SQL NULL on bad inputs.
struct BigintBinaryScalar;

impl VScalar for BigintBinaryScalar {
    type State = BigintExecState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            use egglog_backend_trait::BaseValuePool;
            use egglog_numeric_id::NumericId;
            let n = input.len();
            let a_vec = input.flat_vector(0);
            let b_vec = input.flat_vector(1);
            let a_slice = a_vec.as_slice_with_len::<i64>(n);
            let b_slice = b_vec.as_slice_with_len::<i64>(n);

            let mut results: Vec<Option<i64>> = Vec::with_capacity(n);
            for i in 0..n {
                // A garbage (out-of-range) handle => `None` => SQL NULL for this
                // row (a discarded speculative `ON CONFLICT` row); real handles
                // unwrap normally.
                let ab = unwrap_bigint(&state.pool, state.bigint_ty, a_slice[i])
                    .zip(unwrap_bigint(&state.pool, state.bigint_ty, b_slice[i]));
                results.push(ab.and_then(|(a, b)| {
                    run_bigint(state.op, &a, &b).map(|r| {
                        let boxed = egglog_core_relations::Boxed::new(r);
                        state
                            .pool
                            .intern_dyn(state.bigint_ty, Box::new(boxed))
                            .rep() as i64
                    })
                }));
            }
            let mut out_vec = output.flat_vector();
            {
                let out_slice = out_vec.as_mut_slice::<i64>();
                for (i, r) in results.iter().enumerate() {
                    out_slice[i] = r.unwrap_or(0);
                }
            }
            for (i, r) in results.iter().enumerate() {
                if r.is_none() {
                    out_vec.set_null(i);
                }
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
            ],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// State for the string-handling UDFs: the shared base-value pool
/// (so we can intern result strings) and the `BaseValueId` for
/// `Boxed<String>` (so we can unwrap input handles). Same Arc-shared
/// model as `BigPoolState` / `BigratExecState`.
#[derive(Clone)]
struct StringPoolState {
    pool: base_values::DuckdbBaseValuePool,
    string_ty: egglog_backend_trait::BaseValueId,
    /// Optional `BaseValueId` for `Boxed<BigInt>` — required only for
    /// the `bigint-to-string` UDF.
    bigint_ty: Option<egglog_backend_trait::BaseValueId>,
}

/// Unwrap a string-handle (BIGINT) row value into a Rust `String`.
fn unwrap_string(
    pool: &base_values::DuckdbBaseValuePool,
    string_ty: egglog_backend_trait::BaseValueId,
    raw: i64,
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    use egglog_backend_trait::{BaseValuePool, Value};
    use egglog_numeric_id::NumericId;
    let val = Value::new(raw as u32);
    let boxed = pool.unwrap_dyn(string_ty, val);
    let s: &egglog_core_relations::Boxed<String> =
        boxed
            .downcast_ref()
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                "expected Boxed<String> from pool".into()
            })?;
    Ok(s.0.clone())
}

/// Intern a Rust `String` into the pool, returning the i64 handle.
fn intern_string(
    pool: &base_values::DuckdbBaseValuePool,
    string_ty: egglog_backend_trait::BaseValueId,
    s: String,
) -> i64 {
    use egglog_backend_trait::BaseValuePool;
    use egglog_numeric_id::NumericId;
    let boxed = egglog_core_relations::Boxed::new(s);
    pool.intern_dyn(string_ty, Box::new(boxed)).rep() as i64
}

/// UDF for `string-concat` — variadic `String × ... → String`. Each
/// input is a BIGINT handle to an interned `Boxed<String>`; unwrap,
/// concat, intern the result, return the new handle.
struct StringConcatScalar;

impl VScalar for StringConcatScalar {
    type State = StringPoolState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let n = input.len();
            let arity = input.num_columns();
            // Materialize per-column handle values up-front so we can
            // drop the borrowed `FlatVector` wrappers before the
            // `unwrap_string` pool calls (which need `&state.pool`
            // unborrowed).
            let mut cols: Vec<Vec<i64>> = Vec::with_capacity(arity);
            for c in 0..arity {
                let v = input.flat_vector(c);
                cols.push(v.as_slice_with_len::<i64>(n).to_vec());
            }
            let mut results: Vec<i64> = Vec::with_capacity(n);
            for i in 0..n {
                let mut buf = String::new();
                for c in 0..arity {
                    let s = unwrap_string(&state.pool, state.string_ty, cols[c][i])?;
                    buf.push_str(&s);
                }
                results.push(intern_string(&state.pool, state.string_ty, buf));
            }
            let mut out_vec = output.flat_vector();
            let out_slice = out_vec.as_mut_slice::<i64>();
            out_slice[..n].copy_from_slice(&results);
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::variadic(
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// UDF for `replace` — `String × String × String → String`. All inputs
/// are BIGINT handles; unwrap, run `String::replace`, intern result.
struct ReplaceScalar;

impl VScalar for ReplaceScalar {
    type State = StringPoolState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let n = input.len();
            let hay_col = input.flat_vector(0);
            let needle_col = input.flat_vector(1);
            let repl_col = input.flat_vector(2);
            let hay = hay_col.as_slice_with_len::<i64>(n);
            let needle = needle_col.as_slice_with_len::<i64>(n);
            let repl = repl_col.as_slice_with_len::<i64>(n);
            let mut results: Vec<i64> = Vec::with_capacity(n);
            for i in 0..n {
                let h = unwrap_string(&state.pool, state.string_ty, hay[i])?;
                let n_s = unwrap_string(&state.pool, state.string_ty, needle[i])?;
                let r = unwrap_string(&state.pool, state.string_ty, repl[i])?;
                results.push(intern_string(
                    &state.pool,
                    state.string_ty,
                    h.replace(&n_s, &r),
                ));
            }
            let mut out_vec = output.flat_vector();
            let out_slice = out_vec.as_mut_slice::<i64>();
            out_slice[..n].copy_from_slice(&results);
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
            ],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// UDF for `count-matches` — `String × String → i64`. Inputs are
/// BIGINT handles; output is the raw count (i64 with `MAY_UNBOX =
/// true`, so the value IS the handle for small counts — no interning
/// needed).
struct CountMatchesScalar;

impl VScalar for CountMatchesScalar {
    type State = StringPoolState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let n = input.len();
            let hay_col = input.flat_vector(0);
            let needle_col = input.flat_vector(1);
            let hay = hay_col.as_slice_with_len::<i64>(n);
            let needle = needle_col.as_slice_with_len::<i64>(n);
            let mut out_vec = output.flat_vector();
            let out_slice = out_vec.as_mut_slice::<i64>();
            for i in 0..n {
                let h = unwrap_string(&state.pool, state.string_ty, hay[i])?;
                let n_s = unwrap_string(&state.pool, state.string_ty, needle[i])?;
                out_slice[i] = if n_s.is_empty() {
                    0
                } else {
                    h.matches(&n_s).count() as i64
                };
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
                LogicalTypeHandle::from(LogicalTypeId::Bigint),
            ],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// UDF for `to-string` (i64 overload) — `i64 → String`. Input is the
/// i64 value (or its handle if it doesn't fit in `i64::MAY_UNBOX`'s
/// 31-bit fast path). We `pool_unwrap::<i64>` to get the raw value
/// either way, format with `Display`, intern as Boxed<String>.
struct I64ToStringScalar;

impl VScalar for I64ToStringScalar {
    type State = StringPoolState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            use egglog_backend_trait::Value;
            use egglog_numeric_id::NumericId;
            let n = input.len();
            let in_vec = input.flat_vector(0);
            let in_slice = in_vec.as_slice_with_len::<i64>(n);
            let mut out_vec = output.flat_vector();
            let out_slice = out_vec.as_mut_slice::<i64>();
            for i in 0..n {
                let val = Value::new(in_slice[i] as u32);
                let v: i64 = egglog_backend_trait::pool_unwrap::<i64>(&state.pool, val);
                out_slice[i] = intern_string(&state.pool, state.string_ty, v.to_string());
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeHandle::from(LogicalTypeId::Bigint)],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// UDF for `to-string` (f64 overload) — `f64 → String`. Input is the
/// raw `DOUBLE` value (f64 columns are still SQL-native).
struct F64ToStringScalar;

impl VScalar for F64ToStringScalar {
    type State = StringPoolState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let n = input.len();
            let in_vec = input.flat_vector(0);
            let in_slice = in_vec.as_slice_with_len::<f64>(n);
            let mut out_vec = output.flat_vector();
            let out_slice = out_vec.as_mut_slice::<i64>();
            for i in 0..n {
                // Match the bridge's `to-string` for f64: `format!("{:?}",
                // a.0.0)` (see src/sort/f64.rs:43) — uses Debug formatting,
                // which gives back a decimal that parses round-trip.
                out_slice[i] =
                    intern_string(&state.pool, state.string_ty, format!("{:?}", in_slice[i]));
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeHandle::from(LogicalTypeId::Double)],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// UDF for `to-string` (BigInt overload) — `Z → String`. Input is a
/// `Boxed<BigInt>` handle.
struct BigIntToStringScalar;

impl VScalar for BigIntToStringScalar {
    type State = StringPoolState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            use egglog_backend_trait::{BaseValuePool, Value};
            use egglog_numeric_id::NumericId;
            let bigint_ty = state
                .bigint_ty
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    "BigIntToStringScalar: bigint_ty missing (BigIntSort not registered before \
                 bigint-to-string was called)"
                        .into()
                })?;
            let n = input.len();
            let in_vec = input.flat_vector(0);
            let in_slice = in_vec.as_slice_with_len::<i64>(n);
            // Collect results first: a garbage (out-of-range) handle => `None` =>
            // SQL NULL (a discarded speculative `ON CONFLICT` row), never a crash.
            let mut results: Vec<Option<i64>> = Vec::with_capacity(n);
            for &raw in in_slice.iter().take(n) {
                let val = Value::new(raw as u32);
                let r = if state.pool.handle_in_range(bigint_ty, val) {
                    let boxed = state.pool.unwrap_dyn(bigint_ty, val);
                    boxed
                        .downcast_ref::<egglog_core_relations::Boxed<num::BigInt>>()
                        .map(|z| intern_string(&state.pool, state.string_ty, z.0.to_string()))
                } else {
                    None
                };
                results.push(r);
            }
            let mut out_vec = output.flat_vector();
            {
                let out_slice = out_vec.as_mut_slice::<i64>();
                for (i, r) in results.iter().enumerate() {
                    out_slice[i] = r.unwrap_or(0);
                }
            }
            for (i, r) in results.iter().enumerate() {
                if r.is_none() {
                    out_vec.set_null(i);
                }
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeHandle::from(LogicalTypeId::Bigint)],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// State for the `get-size!` UDF: a shared snapshot of function-table
/// row counts. The duckdb EGraph refreshes this immediately before
/// each `query`/`:until`-style check via `refresh_table_sizes`, then
/// the UDF reads from it during the check's single SELECT. Variadic
/// arg filtering isn't supported yet — `Herbie`'s use is
/// `(get-size!)` with no args, which sums all entries.
#[derive(Clone, Default)]
struct GetSizeState {
    table_sizes: Arc<Mutex<HashMap<String, i64>>>,
}

/// UDF wrapper for `egglog-experimental`'s `get-size!` primitive. The
/// primitive itself walks `ExecutionState::table_ids()` to sum row
/// counts; the duckdb backend has no `ExecutionState` accessible from
/// inside a UDF, so we mirror table sizes into `GetSizeState` and
/// expose them through this zero-arity function. The egglog frontend
/// renames every duckdb-bound `get-size!` call site to this UDF via
/// the `rename_prim` hook so the rule_builder emits
/// `__egglog_get_size()` in the body SQL.
struct GetSizeScalar;

impl VScalar for GetSizeScalar {
    type State = GetSizeState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let n = input.len();
            let total: i64 = state
                .table_sizes
                .lock()
                .map_err(|e| -> Box<dyn std::error::Error> {
                    format!("GetSizeScalar: mutex poisoned: {e}").into()
                })?
                .values()
                .copied()
                .sum();
            let mut out_vec = output.flat_vector();
            let out_slice = out_vec.as_mut_slice::<i64>();
            for i in 0..n {
                out_slice[i] = total;
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }
}

/// UDF for `Q → DOUBLE` (`to-f64`).
struct BigratToF64Scalar;

impl VScalar for BigratToF64Scalar {
    type State = BigratExecState;

    unsafe fn invoke(
        state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            use num::ToPrimitive;
            let n = input.len();
            let in_vec = input.flat_vector(0);
            let in_slice = in_vec.as_slice_with_len::<i64>(n);

            let mut results: Vec<Option<f64>> = Vec::with_capacity(n);
            for i in 0..n {
                let q = unwrap_bigrat(&state.pool, state.bigrat_ty, in_slice[i])?;
                results.push(q.to_f64());
            }
            let mut out_vec = output.flat_vector();
            {
                let out_slice = out_vec.as_mut_slice::<f64>();
                for (i, r) in results.iter().enumerate() {
                    out_slice[i] = r.unwrap_or(0.0);
                }
            }
            for (i, r) in results.iter().enumerate() {
                if r.is_none() {
                    out_vec.set_null(i);
                }
            }
            Ok(())
        }
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeHandle::from(LogicalTypeId::Bigint)],
            LogicalTypeHandle::from(LogicalTypeId::Double),
        )]
    }
}

/// True iff `ruleset` is one of the term-encoding maintenance
/// rulesets that exists purely to keep the SQL union-find tables
/// canonical. When `--duck-native-uf` is on the in-memory
/// union-find subsumes their job, so we skip them at run time.
///
/// Names are generated by `symbol_gen.fresh(prefix)`, where prefix
/// values come from `proof_encoding_helpers.rs`:
///   - `single_parent` — pairwise singleparent rule.
///   - `parent` — path compression (`path_compress_ruleset_name` is
///      generated from the `"parent"` prefix).
///   - `uf_function_index` — mirrors raw pname into the function-
///      form UF table; superseded by the native UF + UDF.
///   - `uf_change_drain` — PR #782's `@uf_change_drain_rule*` drain of the
///      `@UFChange_S` onchange relation (the `--native-uf` encoding). DuckDB
///      never populates that relation (no leader-change callback), so the
///      drain has nothing to delete; skip it (the host-pass owns the rebuild).
fn is_uf_maintenance_ruleset(ruleset: &str) -> bool {
    matches!(
        ruleset
            .trim_end_matches(|c: char| c.is_ascii_digit())
            .trim_start_matches('@'),
        "uf_function_index" | "parent" | "single_parent" | "uf_change_drain"
    )
}

/// Term encoding emits a `congruence_rule<N>` per EqSort constructor
/// that self-joins the constructor's view to find same-input/different-
/// id pairs and unions them. The backend can do the same work in one
/// SQL statement at the end of each iteration — see
/// `emit_inline_congruence` — so under `--duck-native-uf` we skip the
/// encoding-emitted rule entirely.
fn is_congruence_rule_name(rule_name: &str) -> bool {
    rule_name
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .trim_start_matches('@')
        == "congruence_rule"
}

/// Term-encoded rebuild lives in a ruleset named `rebuilding<N>`
/// (the `<N>` is a `symbol_gen.fresh` suffix). The runner uses this
/// to know when an iteration just finished rebuilding — that's when
/// inline-congruence can profitably scan for new dupes created by
/// the rebuild rule's canonicalizing INSERTs.
fn is_rebuilding_ruleset(ruleset: &str) -> bool {
    ruleset
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .trim_start_matches('@')
        == "rebuilding"
}

/// Strip a rule name down to its family prefix: drop the leading `@`, a
/// trailing `#<idx>` seminaive-instance suffix, then trailing digits. E.g.
/// `@rebuild_rule12#5` -> `rebuild_rule`. The duck backend's trait surface
/// never sees the egglog RULESET name (`new_rule` takes only a `desc`), so
/// PR #782 maintenance/rebuild rules must be recognized by NAME instead.
fn rule_family(rule_name: &str) -> &str {
    rule_name
        .trim_start_matches('@')
        .split('#')
        .next()
        .unwrap_or(rule_name)
        .trim_end_matches(|c: char| c.is_ascii_digit())
}

/// True for PR #782's `@rebuild_rule<N>` canonicalization rules (the
/// `@rebuilding` ruleset). Recognized by NAME (the ruleset isn't available on
/// the duck trait surface). Under `--native-uf` these are rewritten into the
/// SQL host-pass form (`EGraph::rewrite_native_uf_rule`).
fn is_rebuild_rule_name(rule_name: &str) -> bool {
    rule_family(rule_name) == "rebuild_rule"
}

/// True for PR #782's `@uf_change_drain_rule<N>` drain rules (the
/// `@uf_change_drain` ruleset). Recognized by NAME. Under `--native-uf` these
/// are DROPPED at compile time: DuckDB never populates the `@UFChange_S`
/// relation they drain (no leader-change callback), so they have nothing to do.
fn is_uf_change_drain_rule_name(rule_name: &str) -> bool {
    rule_family(rule_name) == "uf_change_drain_rule"
}

/// Sanitize a function name for use as a DuckDB UDF identifier.
/// Same scheme as the temp-table naming: `@` → empty, `_` doubled,
/// non-alnum → `_`.
fn sanitize_for_udf(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    out
}

mod backend_impl;
pub mod base_values;
mod compile;
mod external_func;
mod rule_builder;
mod term_build_merge;

/// Quote a SQL identifier with double quotes, escaping any embedded
/// double quote. Necessary because egglog identifiers can contain
/// `@`, `$`, etc., which DuckDB rejects in unquoted form.
///
/// DuckDB compares quoted identifiers case-insensitively (so
/// `"AConst"` and `"aConst"` collide as table names). We encode the
/// case bits with a two-step escape so distinct case-shapes always
/// produce distinct lowered identifiers:
///   1. `_` → `__` (escape literal underscore so it can't be
///      confused with the uppercase marker).
///   2. ASCII uppercase `X` → `_<lowercase x>`.
pub fn q(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for c in name.chars() {
        if c == '_' {
            out.push('_');
            out.push('_');
        } else if c.is_ascii_uppercase() {
            out.push('_');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    format!("\"{}\"", out.replace('"', "\"\""))
}

/// Format a list of comma-separated SQL fragments as a prefix that
/// is followed by something else, with a trailing comma when the
/// list is non-empty and an empty string otherwise. Used to safely
/// concatenate column lists with the always-present trailing `ts`
/// column without producing `(, ts)` when there are no other cols.
/// Bind a rule template by substituting `?1` -> `last`,
/// `?2` -> `cur`, and execute it. Values are i64s, so direct
/// inlining is safe.
///
/// We tried `prepare_cached` (with a generous statement-cache capacity
/// so templates don't evict each other) and binding `?1`/`?2` as real
/// params instead of substituting them: it does NOT help. Profiling
/// the cached path shows preparing is only ~2% of run time (e.g. 0.05s
/// of a 2.6s math-microbenchmark N=9 run); the other ~89% is the SQL
/// `raw_execute` itself (the joins/scans over growing tables), which
/// caching the plan cannot speed up. So plain `execute` with string
/// substitution is kept — it is simplest and no slower.
pub(crate) fn exec_bound(conn: &Connection, sql: &str, last: i64, cur: i64) -> Result<usize> {
    let bound = sql
        .replace("?1", &last.to_string())
        .replace("?2", &cur.to_string());
    if std::env::var("DUCK_TRACE_BOUND").is_ok() {
        eprintln!("[duck/bound last={last} cur={cur}] {bound}");
    }
    Ok(conn.execute(&bound, [])?)
}

/// Execute all variants of a single rule for one seminaive
/// iteration. Pulled out of `run_iteration_in*` so both entry points
/// share the same shape and so per-rule timing accounting lives in
/// one place. Borrows `self.rules` (read-only) and the timing fields
/// disjointly — the caller passes those as `&mut` references.
#[allow(clippy::too_many_arguments)]
/// Per-iteration choice for how to run gated (native-UF recovery)
/// variants under the adaptive rebuild path. `FullScan` runs the
/// original full-scan rendering (default / non-adaptive); `Delta`
/// runs the changed-set semijoin rendering against the materialized
/// `__UF_CHANGED__` temp table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateMode {
    FullScan,
    Delta,
}

fn run_rule_variants(
    rule: &CompiledRule,
    last: i64,
    cur: i64,
    conn: &Connection,
    time_mat_ns: &mut u64,
    time_mat_act_ns: &mut u64,
    time_act_ns: &mut u64,
    rules_affected: &mut u64,
    rule_perf_ns: &mut HashMap<String, (u64, u64)>,
    table_watermarks: &mut HashMap<String, i64>,
    rules_skipped: &mut u64,
    gate_mode: GateMode,
) -> Result<usize> {
    // Watermark gate: if no body table has had inserts since this
    // rule last ran, every variant's seminaive predicate is empty.
    // Skip the whole rule. Falls through for rules with no body
    // (shouldn't happen, but cheap to handle).
    if !rule.body_tables.is_empty() {
        let mut any_fresh = false;
        for t in &rule.body_tables {
            let wm = table_watermarks.get(t).copied().unwrap_or(0);
            if wm >= last {
                any_fresh = true;
                break;
            }
        }
        if !any_fresh {
            *rules_skipped = rules_skipped.wrapping_add(1);
            return Ok(0);
        }
    }

    let mut total: usize = 0;
    let mut rule_mat_ns: u64 = 0;
    let mut rule_act_ns: u64 = 0;
    let trace = std::env::var("DUCK_TRACE_RULE_AFFECTED").is_ok();
    let trace_sql = std::env::var("DUCK_TRACE_SQL").is_ok();
    for variant in &rule.variants {
        // Gated variant: skip unless its trigger table has been
        // updated since this rule last ran. Used by the native-UF
        // recovery variants, which only need to fire when a new
        // union has landed.
        if let Some(gt) = &variant.gate_table {
            let wm = table_watermarks.get(gt).copied().unwrap_or(0);
            if wm < last {
                continue;
            }
        }
        // Adaptive rebuild: when the runner asked for `Delta` and this
        // gated variant has a delta rendering, run the changed-set
        // semijoin scan instead of the full scan. The delta's
        // materialize/temp_table override the parent's; its actions
        // (which SELECT from the same temp table) are reused from the
        // parent when the delta carries none. Falls back to full scan
        // for non-gated variants and variants without a delta.
        let use_delta = gate_mode == GateMode::Delta && variant.delta.is_some();
        let eff_materialize: Option<&String> = if use_delta {
            variant.delta.as_ref().unwrap().materialize.as_ref()
        } else {
            variant.materialize.as_ref()
        };
        let eff_temp: Option<&String> = if use_delta {
            variant.delta.as_ref().unwrap().temp_table.as_ref()
        } else {
            variant.temp_table.as_ref()
        };
        let eff_actions: &[CompiledAction] = if use_delta {
            let d = variant.delta.as_ref().unwrap();
            if d.actions.is_empty() {
                &variant.actions
            } else {
                &d.actions
            }
        } else {
            &variant.actions
        };
        let mut skip_actions = false;
        if let Some(mat_sql_template) = eff_materialize {
            if trace_sql {
                eprintln!("[duck/mat] {mat_sql_template}");
            }
            let t0 = std::time::Instant::now();
            exec_bound(conn, mat_sql_template, last, cur)?;
            let dt = t0.elapsed().as_nanos() as u64;
            *time_mat_ns = time_mat_ns.wrapping_add(dt);
            rule_mat_ns = rule_mat_ns.wrapping_add(dt);
            // Skip the action set if the materialize produced
            // zero rows. CREATE TABLE AS SELECT is DDL so its
            // `execute` return is 0 regardless of SELECT row
            // count; we use a follow-up COUNT(*) on the temp
            // table to actually find out. The count is O(1)
            // against DuckDB's row-count metadata.
            if let Some(temp) = eff_temp {
                let count_sql = format!("SELECT COUNT(*) FROM {}", q(temp));
                let n: i64 = conn.query_row(&count_sql, [], |r| r.get(0))?;
                if n == 0 {
                    skip_actions = true;
                }
            }
        }
        if skip_actions {
            continue;
        }
        for act in eff_actions {
            if trace_sql {
                eprintln!(
                    "[duck/{}] {}",
                    if eff_materialize.is_some() {
                        "mat-act"
                    } else {
                        "act"
                    },
                    act.sql,
                );
            }
            let t0 = std::time::Instant::now();
            let n = exec_bound(conn, &act.sql, last, cur)?;
            let dt = t0.elapsed().as_nanos() as u64;
            if eff_materialize.is_some() {
                *time_mat_act_ns = time_mat_act_ns.wrapping_add(dt);
            } else {
                *time_act_ns = time_act_ns.wrapping_add(dt);
            }
            rule_act_ns = rule_act_ns.wrapping_add(dt);
            *rules_affected = rules_affected.wrapping_add(n as u64);
            total += n;
            if n > 0 {
                if let Some(target) = &act.target {
                    let e = table_watermarks.entry(target.clone()).or_insert(0);
                    if cur > *e {
                        *e = cur;
                    }
                }
                if trace {
                    eprintln!("[duck/rule_n] {} +{n}", rule.name);
                }
            }
        }
    }
    if rule_mat_ns != 0 || rule_act_ns != 0 {
        let e = rule_perf_ns.entry(rule.name.clone()).or_insert((0, 0));
        e.0 = e.0.wrapping_add(rule_mat_ns);
        e.1 = e.1.wrapping_add(rule_act_ns);
    }
    Ok(total)
}

pub(crate) fn prefix_with_comma(parts: &[String]) -> String {
    if parts.is_empty() {
        String::new()
    } else {
        format!("{}, ", parts.join(", "))
    }
}

/// The (very small) set of column types we currently understand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnTy {
    I64,
    Bool,
    F64,
    Str,
    /// A pair of two i64s, stored as a DuckDB `STRUCT(first BIGINT,
    /// second BIGINT)`. Used by the term encoding's proof mode to
    /// bundle `(leader_id, proof_id)` as the output of the
    /// `uf_function_<sort>` table. The bridge exposes three SQL
    /// primitives that operate on this type: `pair`, `pair-first`,
    /// and `pair-second`.
    PairI64,
}

impl ColumnTy {
    fn sql(self) -> &'static str {
        match self {
            ColumnTy::I64 => "BIGINT",
            ColumnTy::Bool => "BOOLEAN",
            ColumnTy::F64 => "DOUBLE",
            ColumnTy::Str => "VARCHAR",
            ColumnTy::PairI64 => "STRUCT(first BIGINT, second BIGINT)",
        }
    }
}

/// Marker [`BaseValue`] registered with the trait `BaseValuePool` so
/// the duck pipeline can carry "this column is a Pair sort" through
/// the existing [`egglog_backend_trait::ColumnTy::Base`] surface
/// without extending the trait IR. Never actually constructed at
/// runtime — pair values live in DuckDB `STRUCT` columns and are
/// computed inline by the `pair` / `pair-first` / `pair-second` SQL
/// primitives.
#[derive(Clone, Hash, Eq, PartialEq, Debug, Default)]
pub struct DuckPairMarker;

impl egglog_core_relations::BaseValue for DuckPairMarker {}

/// Merge mode for functions with outputs. Mirrors egglog's
/// `:merge old` / `:merge new` keywords.
///
/// No longer `Copy`: the `Fold` variant carries an owned SQL string. All call
/// sites move or borrow it (the `== Some(MergeMode::Min)` comparison still works
/// via `PartialEq`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeMode {
    /// First-set wins. `ON CONFLICT DO NOTHING`.
    Old,
    /// Latest-set wins. `ON CONFLICT DO UPDATE` of the output and ts.
    New,
    /// `(ordering-min old new)` — output is `LEAST(existing, incoming)`.
    /// Used by the UF function-index table so it always holds the
    /// smallest known representative for each input, letting
    /// singleparent skip its pairwise self-join.
    Min,
    /// Native VALUE-FOLD custom `:merge` (`--native-merge`): a pure-primitive fold
    /// over `old`/`new` (e.g. `(from-string (to-string (* new old)))`) compiled to
    /// a SQL scalar expression over the existing row (`c{out}`) and the incoming
    /// row (`EXCLUDED.c{out}`). Resolved in-SQL via
    /// `ON CONFLICT DO UPDATE SET c{out} = <expr>, ts = EXCLUDED.ts`. The stored
    /// string is the fully-built `<expr>` (see [`mergefn_to_sql`]).
    Fold(String),
}

/// A literal value usable in seed inserts and `check`/`lookup`.
///
/// `Hash` / `Eq` / `PartialEq` are derived so `compile::dedupe_body_atoms`
/// can key on `(name, input args)` to collapse duplicate body atoms.
/// `f64` doesn't implement `Eq`/`Hash` natively; we wrap it through
/// `to_bits()` in the impls below so the trait bounds are satisfied
/// for the dedup pass (NaN-equality semantics don't matter here:
/// rule bodies never contain NaN literals).
#[derive(Debug, Clone)]
pub enum Literal {
    I64(i64),
    Bool(bool),
    F64(f64),
    Str(String),
}

impl PartialEq for Literal {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Literal::I64(a), Literal::I64(b)) => a == b,
            (Literal::Bool(a), Literal::Bool(b)) => a == b,
            (Literal::F64(a), Literal::F64(b)) => a.to_bits() == b.to_bits(),
            (Literal::Str(a), Literal::Str(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for Literal {}
impl std::hash::Hash for Literal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Literal::I64(v) => {
                0u8.hash(state);
                v.hash(state);
            }
            Literal::Bool(v) => {
                1u8.hash(state);
                v.hash(state);
            }
            Literal::F64(v) => {
                2u8.hash(state);
                v.to_bits().hash(state);
            }
            Literal::Str(v) => {
                3u8.hash(state);
                v.hash(state);
            }
        }
    }
}

impl ToSql for Literal {
    fn to_sql(&self) -> duckdb::Result<duckdb::types::ToSqlOutput<'_>> {
        match self {
            Literal::I64(i) => i.to_sql(),
            Literal::Bool(b) => b.to_sql(),
            Literal::F64(f) => f.to_sql(),
            Literal::Str(s) => s.as_str().to_sql(),
        }
    }
}

/// A term in a rule body or action.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    Var(String),
    Lit(Literal),
    /// A primitive expression: arithmetic (`+`, `-`, `*`, `/`),
    /// comparison (`<`, `<=`, `>`, `>=`, `=`, `!=`), or boolean
    /// (`and`, `or`, `not`). The op name is mapped to a SQL
    /// operator at codegen; see `compile.rs::prim_sql`.
    Prim(String, Vec<Term>),
    /// Read a function's output value as an expression. Compiles to
    /// `(SELECT c<out> FROM <name> WHERE c0 = arg0 AND … LIMIT 1)`.
    /// Used for term-encoding's globals: `(let v (foo 1 2))` becomes
    /// a synthetic `(function v () Sort :no-merge)` plus `(set (v)
    /// ...)`, and later references `(v)` are reads of this function.
    /// Cannot be used on relations (no output column).
    ///
    /// `identity_on_miss`: when true (the term encoder's flat UF-index
    /// `@UF_Sf`, declared `DefaultVal::Identity`), a missing key must resolve
    /// to the (single) key itself rather than NULL — the read compiles to
    /// `COALESCE((SELECT c1 …), <arg0>)`.
    FuncCall {
        name: String,
        args: Vec<Term>,
        identity_on_miss: bool,
    },
}

impl Term {
    pub fn var(name: impl Into<String>) -> Self {
        Term::Var(name.into())
    }
    pub fn i64(v: i64) -> Self {
        Term::Lit(Literal::I64(v))
    }
    pub fn prim(op: impl Into<String>, args: Vec<Term>) -> Self {
        Term::Prim(op.into(), args)
    }
}

/// A body atom of a rule.
#[derive(Debug, Clone)]
pub enum Atom {
    /// A function-table atom. `args.len()` must match the function's
    /// full arity (inputs for relations; inputs + 1 for functions
    /// with outputs).
    Func { name: String, args: Vec<Term> },
    /// A pure-primitive constraint: the term must evaluate to true.
    /// Examples: `(< x 5)`, `(!= a b)`. Compiles to a SQL WHERE
    /// constraint.
    Filter(Term),
    /// Bind `var` to the value of `expr` for the rest of the body
    /// and any subsequent actions. Used for `(= var (primitive ...))`
    /// patterns in egglog rule bodies. Compiles to nothing on its
    /// own — it just extends the body's variable→SQL binding.
    Bind { var: String, expr: Term },
}

/// A rule action.
#[derive(Debug, Clone)]
pub enum Action {
    /// Insert a row. The trailing arg is the output value for
    /// functions; for relations all args are key columns.
    Insert { name: String, args: Vec<Term> },
    /// Delete a row matched by its key columns. For functions, the
    /// args are the input columns only (output is ignored). For
    /// relations, args are all the columns.
    Delete { name: String, key_args: Vec<Term> },
    /// Allocate a fresh EqSort ID via the constructor table `name`,
    /// inserting `(args..., fresh_id)`, and bind `var` to the
    /// allocated ID for use in subsequent actions of the same rule.
    /// Compiles into a `nextval('seq')` column in the rule's
    /// materialized match table; subsequent actions reference `var`
    /// as a regular term.
    LetCtor {
        var: String,
        name: String,
        args: Vec<Term>,
    },
    /// Bind `var` to a pure expression (no constructor allocation,
    /// no insert). Compiles into a non-side-effecting column of the
    /// rule's materialized match table. Subsequent actions reference
    /// `var` as a regular term.
    LetExpr { var: String, expr: Term },
    /// `(panic msg)` action: raise a runtime error if the rule body
    /// matches at all. Compiles to `SELECT error('msg') FROM <body>
    /// LIMIT 1` — DuckDB only evaluates `error()` on returned rows, so
    /// no match = no error.
    Panic { msg: String },
}

/// A whole rule.
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    /// Ruleset the rule lives in. Empty string for the default
    /// ruleset. `run_iteration_for(ruleset)` runs only rules whose
    /// ruleset matches.
    pub ruleset: String,
    pub body: Vec<Atom>,
    pub actions: Vec<Action>,
}

/// Internal: schema for a single registered function. Public so
/// frontend diagnostic dumps can read it; not part of the stable
/// API.
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    /// All schema columns (inputs followed by output/ID, if any).
    pub cols: Vec<ColumnTy>,
    /// Number of "input" columns from the user perspective. For
    /// relations this is `cols.len()`; for functions and EqSort
    /// constructors it's `cols.len() - 1`.
    pub inputs_len: usize,
    /// `Some` if this is a function with an output column; `None`
    /// for relations and EqSort constructors.
    pub merge: Option<MergeMode>,
    /// True iff this is an EqSort constructor: PK covers all cols
    /// (so multiple distinct IDs per input set are allowed) and
    /// `allocate_and_insert` is the intended insertion path.
    pub eq_sort_ctor: bool,
    /// True iff this function is backed by a native union-find
    /// (`--duck-native-uf` flag on and merge is `(ordering-min …)`).
    /// The rule compiler demotes body atoms against this function
    /// to scalar UDF calls instead of joining the SQL table.
    pub native_uf_udf: Option<String>,
    /// For an EqSort constructor under `--duck-native-uf`, the name
    /// of the raw UF (`pname`) table that union assertions land in.
    /// When set, the runner emits an *inline congruence* SQL at the
    /// end of each iteration: pairs of rows with matching inputs but
    /// different IDs are read out of this view and pushed into pname
    /// as `(max, min)` union assertions. Replaces the
    /// `@congruence_rule*` rules emitted by term encoding.
    pub eq_sort_pname: Option<String>,
    /// True iff this function uses identity-on-miss lookup semantics
    /// (`DefaultVal::Identity`): a function-table lookup of an absent key
    /// resolves to the key itself, with no row inserted. Used by the
    /// canonicalize-at-creation encoding for the flat UF-index `@UF_Sf`.
    /// Only valid for a single-key function (2 columns) whose key and output
    /// share a type. A `Term::FuncCall` against such a function compiles to
    /// `COALESCE((SELECT c1 …), <key>)` in `compile::term_sql`.
    pub identity_on_miss: bool,
    /// `Some` for a TERM-BUILD custom `:merge` view (`--native-merge`): the
    /// retained `MergeFn` tree (a top-level `IfEq`/`Seq` of `Construct` /
    /// `TableInsert` / canon `Function` nodes) lowered by the frontend's
    /// `translate_term_build_merge_to_seq`. Its presence marks `<self>` as a
    /// term-build view: the view is registered ALL-COLUMNS keyed (conflicting
    /// `(set (@FView key) eclass)` writes coexist), and
    /// [`EGraph::emit_term_build_merges`] resolves FD conflicts at the iteration
    /// boundary by compiling this tree to a sequence of bulk SQL statements
    /// (mint constructors set-based with hash-consing, write back the merged
    /// eclass). `None` for every other table. The tree carries `FunctionId`s
    /// resolved against `EGraph::backend_function_names`.
    pub merge_tree: Option<egglog_backend_trait::MergeFn>,
    /// `Some(uf_table)` for a NATIVE-CONGRUENCE constructor view (`--native-merge`
    /// with `supports_native_congruence_merge()` true): the `@<C>View` of an eq-sort
    /// constructor, declared in the BASELINE all-columns Unit-relation shape
    /// `(children..., eclass) -> Unit` (so its rows COEXIST — the relation's all-cols
    /// PK only de-dupes identical rows). Set by [`Backend::register_native_merge_view`]
    /// AFTER `declare` (the encoder dropped the view's `@congruence_rule*` and records
    /// the view -> UF association). `uf_table` is the per-sort UF-backed function
    /// (`@UF_Sf`) of the view's eclass column sort — the same table the dropped
    /// `@congruence_rule*` wrote union edges into (via `(set (@UF_Sf larger) smaller)`),
    /// and the table `sync_native_ufs` scans `(c0, c1, ts)` edges out of (its
    /// `native_uf_pname` maps it to ITSELF; there is no separate `@UF_S` parent on
    /// DuckDB).
    ///
    /// Its presence marks the view as native-congruence: at the iteration boundary
    /// [`EGraph::emit_native_congruence`] reads each FD conflict (same children, two
    /// eclasses) via a self-join and inserts the `(GREATEST, LEAST, ts)` union edge
    /// into `uf_table` — exactly what the dropped rule did. `None` for every other
    /// table.
    pub native_congruence_uf: Option<String>,
}

impl FunctionInfo {
    pub(crate) fn arity(&self) -> usize {
        self.cols.len()
    }
    pub(crate) fn has_output(&self) -> bool {
        self.merge.is_some()
    }
}

/// The executor.
pub struct EGraph {
    conn: Connection,
    pub functions: HashMap<String, FunctionInfo>,
    rules: Vec<CompiledRule>,
    next_ts: i64,
    /// Per source-rule "last run at" — the ts at which it last ran.
    last_run_at: HashMap<String, i64>,
    /// Cumulative count of rows affected by every rule action's SQL
    /// across all iterations. The frontend uses snapshots of this
    /// counter to detect saturation precisely (vs total tuple count,
    /// which can balance to zero when deletes match inserts).
    rules_affected: u64,
    /// Per-category nanosecond counters for SQL `execute` calls.
    /// Exposed by `DUCK_PERF_DUMP=1` so we can see where the per-
    /// iteration wall time goes. Not part of the public API; debug
    /// hook only.
    time_mat_ns: u64,
    time_mat_act_ns: u64,
    time_act_ns: u64,
    /// (mat_ns, act_ns) per rule name. Populated alongside the
    /// global accumulators. Used by `DUCK_PERF_DUMP` to surface
    /// which individual rules dominate.
    rule_perf_ns: HashMap<String, (u64, u64)>,
    /// Ruleset of each rule, mirrored so `DUCK_PERF_DUMP` can roll
    /// per-rule timings up to per-ruleset.
    rule_to_ruleset: HashMap<String, String>,
    /// Per-table "max `ts` of any insert" watermark. Bumped on
    /// every successful row-affecting INSERT (top-level seeds and
    /// rule actions alike). The seminaive predicate fires on rows
    /// with `ts >= last_run_at`, so a rule whose every body table
    /// has `watermark < last_run_at` cannot match anything — we
    /// skip it entirely.
    table_watermarks: HashMap<String, i64>,
    /// Count of rule firings short-circuited by the watermark gate.
    /// Surfaced by `DUCK_PERF_DUMP`.
    rules_skipped: u64,
    /// Whether `--duck-native-uf` is in effect for this `EGraph`.
    /// Set once by `enable_native_uf` before any tables/rules are
    /// registered; everything downstream branches on this.
    native_uf_enabled: bool,
    /// Whether the program was term-encoded in proof-tracking mode.
    /// Set once by `enable_proofs` from the frontend's
    /// `with_duckdb_backend` when `DuckBackendConfig::proofs` is on.
    ///
    /// In proof mode the rule-action hash-cons for eq-sort
    /// constructors looks up the *bare* term table (which keeps each
    /// application's own id) rather than the canonicalized
    /// `@<name>View` (whose id column the rebuild rewrites to the
    /// e-class leader). Reusing the leader is faster — it folds
    /// congruent terms onto one id — but it discards which
    /// constructor the rule actually produced, so proof extraction
    /// reconstructs a different (equal) term than the rule head and
    /// the proof checker rejects it. Term identity only matters for
    /// proofs, so non-proof runs keep the faster leader hash-cons.
    proofs_enabled: bool,
    /// Per-sort native union-find tables, keyed by the egglog
    /// function name they're standing in for (the `:merge
    /// (ordering-min old new)` function emitted by term encoding).
    /// The corresponding SQL table is still created — writes flow
    /// through it so other rulesets that haven't been ported can
    /// still read it via JOIN — but a DuckDB scalar UDF
    /// (`duck_uf_<sanitized_name>_find`) is also registered that
    /// lets queries look up the canonical root in O(α(n)) memory.
    ///
    /// `Arc<Mutex>` so the UDF (which DuckDB stores for the lifetime
    /// of the connection) can lock state for find/union. We pin
    /// DuckDB to a single thread (`SET threads = 1`) for determinism
    /// so contention is nil — the Mutex is for shared ownership.
    native_ufs: HashMap<String, Arc<Mutex<UfTable>>>,
    /// For each native-UF function name, the corresponding raw UF
    /// (pname) table name. Term encoding always emits the pair
    /// `@UF_<sort>` (relation-style, holds raw union assertions)
    /// and `@UF_<sort>f` (function-form, ordering-min merge); we
    /// derive pname by stripping a trailing `f` at native-UF
    /// registration time and store it explicitly so the sync
    /// scanner doesn't have to re-derive on every iteration.
    native_uf_pname: HashMap<String, String>,
    /// Per native-UF function, the maximum `ts` we've already
    /// pulled into the in-memory union-find. The sync scan reads
    /// only rows newer than this on each iteration.
    last_uf_sync_ts: HashMap<String, i64>,
    /// Per eq-sort view, the `ts` through which inline-congruence
    /// has already scanned. Inline-cong's "new" side restricts to
    /// rows with `ts >= last_inline_cong_at AND ts < cur`, matching
    /// the seminaive triangulation that `@congruence_rule` did in
    /// the encoding. Without this, pairs that come from top-level
    /// `insert_terms` (which never line up with any `cur`) get
    /// silently skipped.
    last_inline_cong_at: HashMap<String, i64>,
    /// Per TERM-BUILD view (`merge_tree.is_some()`), the `ts` through which
    /// `emit_term_build_merges` has already resolved FD conflicts. The conflict
    /// self-join restricts its "new" side to rows with `ts >= last_termbuild_at`
    /// so each conflict pair is processed once.
    last_termbuild_at: HashMap<String, i64>,
    /// Per NATIVE-CONGRUENCE view (`native_congruence_uf.is_some()`), the `ts`
    /// through which `emit_native_congruence` has already resolved FD conflicts.
    /// The conflict self-join restricts its window to rows with
    /// `ts >= last_native_cong_at` so each conflict pair is processed once.
    last_native_cong_at: HashMap<String, i64>,
    /// Diagnostic counter: total rows pulled into native UFs across
    /// all syncs. Surfaced via `DUCK_PERF_DUMP` (under the native-uf
    /// flag) so we can see the work the SQL → memory bridge is
    /// doing.
    native_uf_unions_synced: u64,
    /// FunctionId -> registered table name. Populated by
    /// `Backend::add_table`; indexed by `FunctionId::rep()`. Trait
    /// callers receive numeric ids and we look up the underlying
    /// duckdb table name through this vector.
    ///
    /// Only used by the (currently stubbed) `Backend` trait impl in
    /// `backend_impl.rs`. The existing parallel pipeline in
    /// `src/backend_duckdb.rs` registers tables by name through
    /// `add_function` / `add_relation_with_pname` /
    /// `add_eq_sort_constructor` directly and does not consult this
    /// vector.
    backend_function_names: Vec<String>,
    /// Report verbosity, stored on behalf of
    /// `Backend::set_report_level`. The trait impl is stub-shaped in
    /// Phase 2 Commit 9; the field is here so the setter has a place
    /// to write to without affecting the rest of the backend.
    backend_report_level: egglog_backend_trait::ReportLevel,
    /// Zero-sized stub `ContainerPool`. DuckDB does not support
    /// containers (see `docs/backend_trait_design.md`); the field
    /// exists so `Backend::container_pool` / `container_pool_mut` can
    /// return a `&dyn ContainerPool`.
    backend_container_pool: backend_impl::DuckdbContainerPool,
    /// RuleId -> rule name registered through the trait. Populated by
    /// `RuleBuilderOps::build` (see `rule_builder.rs`). Indexed by
    /// `RuleId::rep()`. The corresponding compiled rule lives in
    /// `RuleId.rep() -> index into self.rules`. The trait surface
    /// hands callers a `RuleId` from `Backend::new_rule(...).build()`
    /// and later asks the backend to run specific ids via
    /// `Backend::run_rules(&[ids])`. We translate each id back into a
    /// slot in `self.rules` (the compiled-rule vector) so the runtime
    /// can dispatch *exactly* those rules — no ruleset-name detour, no
    /// risk of co-firing freed siblings that happen to share a name.
    ///
    /// `None` entries mark slots whose rule has been freed (via
    /// `Backend::free_rule`) or whose builder produced an empty
    /// action list (no-op rule that was never compiled).
    backend_rule_indices: Vec<Option<usize>>,
    /// In-process base-value pool. Stores typed intern tables for
    /// every `BaseValue` type registered through the trait, including
    /// `i64`/`bool`/`f64`/`String`/`Unit` and user-defined impls.
    ///
    /// Concrete DuckDB SQL columns store the `Value`'s `u32` as
    /// `BIGINT`; the pool's `intern_dyn` / `unwrap_dyn` provide the
    /// mapping between typed primitives and `Value`. Wraps
    /// `egglog_core_relations::BaseValues` 1:1 — see
    /// `base_values.rs`.
    backend_base_value_pool: base_values::DuckdbBaseValuePool,
    /// Storage for user-registered primitives + deferred-panic
    /// sentinels. Indexed by `ExternalFunctionId::rep()`. See
    /// `external_func.rs` for slot semantics. Wiring the stored
    /// functions to live DuckDB UDFs is deferred to Commit 14.
    backend_external_funcs: external_func::DuckdbExternalFuncRegistry,
    /// Primitive names whose DuckDB scalar UDF wrapper we've already
    /// registered on `self.conn`. Re-registering the same name is an
    /// error in DuckDB, so we dedupe at the source.
    registered_builtin_udfs: std::collections::HashSet<String>,
    /// Snapshot of function-table row counts, shared with the
    /// `get-size!` UDF state. `refresh_table_sizes` repopulates this
    /// just before each existence-check query so the UDF returns a
    /// consistent answer for that query.
    table_sizes: Arc<Mutex<HashMap<String, i64>>>,
    /// `--fast-rebuild`: DROP the always-empty `δview ⋈ uf_old` term of the
    /// rebuild seminaive derivative (the new-view-rows probe; empty under
    /// canonicalize-at-creation), keeping only `view ⋈ δuf`. Set by
    /// [`enable_fast_rebuild`](Self::enable_fast_rebuild) (from
    /// `DuckBackendConfig::fast_rebuild`), or via the legacy
    /// `DUCK_ADAPTIVE_REBUILD` / `DUCK_DELTA_REBUILD` env vars (back-compat).
    ///
    /// This term-drop is ORTHOGONAL to whether the native UF is on:
    /// - WITHOUT `--native-uf`: the RELATIONAL δuf fast-rebuild (compile.rs's
    ///   `delta_rebuild_view_idx` decomposition — drop the VIEW-focus branch of
    ///   the relational `view ⋈ @UF_Sf` rebuild rule).
    /// - WITH `--native-uf` ([`native_uf_enabled`](Self::native_uf_enabled)):
    ///   drop the rebuild host-pass's δview-focus seminaive branch (see
    ///   `native_uf_drop_delta_view` in `add_rule`). The native-UF
    ///   `__UF_CHANGED__` adaptive delta scan is a SEPARATE mechanism, engaged by
    ///   `--native-uf` alone ([`adaptive_rebuild`](Self::adaptive_rebuild)); it
    ///   accumulates the set of ids whose canonical changed (drained from the
    ///   native UFs) and per rebuild iteration chooses between the
    ///   `__UF_CHANGED__` semijoin and the full-scan gated variant regardless of
    ///   `--fast-rebuild`.
    fast_rebuild: bool,
    /// Switch-over fraction theta: when the accumulated changed-set
    /// size exceeds `theta * |UF id-space|`, run the full scan and
    /// reset; otherwise run the delta semijoin and accumulate.
    /// Tunable via `DUCK_ADAPTIVE_THETA`.
    adaptive_theta: f64,
    /// Whether to log per-iteration adaptive mode choices + sizes.
    /// Gated behind `DUCK_ADAPTIVE_DEBUG`.
    adaptive_debug: bool,
    /// Accumulated changed-id set (canonical-changed ids since the
    /// last full-scan reset). Only maintained when `adaptive_rebuild`
    /// is on. Reset to empty after any full-scan gated variant runs.
    adaptive_changed_ids: std::collections::HashSet<i64>,
    /// `--native-uf --duckdb` only: maps the PR #782 `@canon_S`
    /// find-or-self primitive's [`ExternalFunctionId`] (returned by
    /// [`Backend::add_uf_function`]) to the find UDF name
    /// (`duck_uf_<sort>_find`) of its UF function. The frontend rebinds the
    /// primitive name to this id via `rename_prim`/`set_external_func_name`;
    /// when that happens we copy the mapping into `native_uf_canon_prim_udf`
    /// keyed by the egglog primitive NAME (which is what shows up as
    /// `Term::Prim(name, …)` in the compiled rule IR).
    native_uf_canon_id_udf: HashMap<egglog_backend_trait::ExternalFunctionId, String>,
    /// `--native-uf --duckdb` only: maps a PR #782 `@canon_S` find-or-self
    /// primitive's egglog NAME to its find UDF name. The compile pre-pass
    /// (`compile::rewrite_canon_prims`) rewrites every `Term::Prim(canon_name,
    /// [x])` into a UDF-call prim so `prim_sql` emits `duck_uf_<sort>_find(x)`
    /// (the SQL host-pass find), replacing the structural relational heuristic.
    native_uf_canon_prim_udf: HashMap<String, String>,
    /// `--native-uf --duckdb` only: maps a native-UF find UDF name
    /// (`duck_uf_<sort>_find`) back to the `@UF_Sf` union table it reads. The
    /// `@UF_Sf` table is where unions land (and whose watermark advances when a
    /// union does), so the rebuild host-pass compiles a gated full-scan variant
    /// keyed on it: when a union landed since the rule last ran, re-scan ALL
    /// view rows and re-canonicalize the stale ones (the seminaive variant only
    /// sees newly-inserted view rows; old rows go stale when a later union
    /// displaces one of their ids).
    native_uf_udf_to_table: HashMap<String, String>,
    /// `--native-uf --duckdb` only: names of PR #782's `@UFChange_S` onchange
    /// relations (one per eq-sort). DuckDB never runs the leader-change
    /// callback that would populate them, so the rebuild rule's join against
    /// them produces nothing; the compile pre-pass strips the onchange body
    /// atom (and the filters it bound) so the rebuild drives off a view scan +
    /// `@canon_S`-find guard (the SQL host-pass rebuild).
    native_uf_onchange: std::collections::HashSet<String>,
    /// Ids displaced by the most recent `sync_native_ufs` (the
    /// "recent delta"). The adaptive mode decision compares THIS
    /// against `theta * id_space`, not the full accumulated set:
    /// the accumulated set is what the delta scan must semijoin
    /// against for correctness (it must cover every id displaced
    /// since the last full re-canonicalization), but using it for
    /// the switch-over would make the decision saturate to
    /// always-full-scan as the set grows. Keying the decision off
    /// the recent delta lets delta mode engage whenever the current
    /// iteration's churn is small, while a full scan (which resets
    /// the accumulated set) is chosen on bursty iterations.
    adaptive_recent_displaced: usize,
}

struct CompiledRule {
    name: String,
    ruleset: String,
    variants: Vec<CompiledVariant>,
    /// Distinct function-table names appearing in the rule body. The
    /// runner skips the whole rule when none of these tables has had
    /// rows inserted since the rule's `last_run_at` — there's nothing
    /// for any variant to match. Pure functional / filter atoms
    /// don't appear here.
    body_tables: Vec<String>,
}

/// One seminaive variant of a rule, ready to execute.
pub(crate) struct CompiledVariant {
    /// Optional CREATE TEMP TABLE SQL that materializes the body
    /// matches (with any `nextval()`-allocated ids per LetCtor).
    /// `None` for variants whose actions are all simple inserts/
    /// deletes — those just run directly.
    pub(crate) materialize: Option<String>,
    /// Name of the temp table created by `materialize`. Used by the
    /// runner to short-circuit the action set when the temp table
    /// is empty after materialization.
    pub(crate) temp_table: Option<String>,
    /// Action SQLs to run in order. When `materialize` is Some, each
    /// action SELECTs from the temp table; otherwise from the body.
    pub(crate) actions: Vec<CompiledAction>,
    /// `Some(table)` for native-UF recovery variants. The runner
    /// skips this variant unless `table`'s watermark has advanced
    /// since the rule last ran — it exists only to catch updates
    /// that the demoted UF subqueries would otherwise miss (a UF
    /// row changed but no view row did this iter), so when the UF
    /// is quiet there's nothing to do.
    pub(crate) gate_table: Option<String>,
    /// For gated (native-UF recovery) variants only, under the
    /// adaptive rebuild path: an alternate rendering of this variant
    /// whose body scan is restricted to rows that reference a changed
    /// id (`(t0.c0 IN (SELECT id FROM __UF_CHANGED__) OR ...)` over
    /// the eq-sort id columns). The runner picks between this
    /// `delta` rendering and the full-scan rendering above per
    /// iteration based on the accumulated changed-set size. `None`
    /// for non-gated variants, when no id column was detected, or
    /// when the adaptive flag is off. The SQL references the literal
    /// placeholder table name [`UF_CHANGED_PLACEHOLDER`], which the
    /// runner materializes per iteration.
    pub(crate) delta: Option<DeltaVariant>,
}

/// Delta-restricted rendering of a gated variant (adaptive path).
/// Mirrors the full-scan `materialize`/`temp_table`/`actions` of its
/// parent [`CompiledVariant`] but with the changed-set semijoin
/// predicate ANDed into the body scan.
pub(crate) struct DeltaVariant {
    pub(crate) materialize: Option<String>,
    pub(crate) temp_table: Option<String>,
    pub(crate) actions: Vec<CompiledAction>,
}

/// Placeholder table name embedded in the delta variant SQL. The
/// runner rewrites it to the real changed-set temp table just before
/// executing the delta-restricted gated variant. Kept distinct from
/// any real table name so a stray substitution is obvious.
pub(crate) const UF_CHANGED_PLACEHOLDER: &str = "__UF_CHANGED__";

/// Synthetic `Term::Prim` op-name prefix the `--native-uf --duckdb` compile
/// pre-pass ([`EGraph::rewrite_native_uf_rule`]) uses to mark a PR #782
/// `@canon_S` find-or-self call rewritten to its find UDF. The rest of the
/// op-name is the UDF name (`duck_uf_<sort>_find`); `compile::prim_sql` emits
/// `<udf>(arg)`. A prefix that no real egglog primitive uses, so a stray match
/// is impossible.
pub(crate) const NATIVE_UF_FIND_PRIM_PREFIX: &str = "__duck_uf_find:";

/// Apply `f` to every variable name referenced anywhere in `t` (recursing
/// through `Prim`/`FuncCall` args). Used by the native-UF rebuild host-pass
/// rewrite to find filters that dangle after the onchange atom is stripped.
fn term_for_each_var(t: &Term, f: &mut impl FnMut(&str)) {
    match t {
        Term::Var(v) => f(v),
        Term::Lit(_) => {}
        Term::Prim(_, args) | Term::FuncCall { args, .. } => {
            for a in args {
                term_for_each_var(a, f);
            }
        }
    }
}

/// Apply `f` to every native-UF find UDF name embedded in a rewritten
/// `Term::Prim("__duck_uf_find:<udf>", …)` op (recursing through args). Used to
/// compute the rebuild host-pass gate tables. See [`NATIVE_UF_FIND_PRIM_PREFIX`].
fn term_for_each_find_udf(t: &Term, f: &mut impl FnMut(&str)) {
    match t {
        Term::Var(_) | Term::Lit(_) => {}
        Term::Prim(op, args) => {
            if let Some(udf) = op.strip_prefix(NATIVE_UF_FIND_PRIM_PREFIX) {
                f(udf);
            }
            for a in args {
                term_for_each_find_udf(a, f);
            }
        }
        Term::FuncCall { args, .. } => {
            for a in args {
                term_for_each_find_udf(a, f);
            }
        }
    }
}

/// One rule action paired with the table it writes to (so the
/// watermark tracker can bump that table's high-water-mark after a
/// successful row-affecting execute). `target` is `None` for no-op
/// LetCtor/LetExpr placeholders and for Panic.
pub(crate) struct CompiledAction {
    pub(crate) sql: String,
    pub(crate) target: Option<String>,
}

impl EGraph {
    pub fn new() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        // Tuning: DuckDB defaults to row-order preservation, which
        // uses extra memory to track insertion order through joins.
        // We don't care about row order — egglog is set-semantics.
        // Disabling drops a sizeable chunk of per-iteration overhead
        // on workloads that do many INSERT…SELECT.
        conn.execute("SET preserve_insertion_order = false", [])?;
        // Herbie egglog dumps inline deeply-nested `(bigrat
        // (from-string …) (from-string …))` constants into rule
        // bodies; the resulting WHERE clauses + SELECT lists can
        // easily blow past DuckDB's default expression-depth cap of
        // 1000. Bump it generously so taylor*.egg dumps with 100+
        // levels of nesting compile.
        conn.execute("SET max_expression_depth = 100000", [])?;
        // Determinism: parallel execution can produce rows from a
        // SELECT in any order. With hash-cons (or any rule action
        // that side-effects via nextval()), per-row evaluation order
        // determines which sequence values get burned for which
        // matches, which affects unification chains and thus tuple
        // counts at bounded iteration. Force single-threaded so our
        // output is reproducible run-to-run.
        conn.execute("SET threads = 1", [])?;
        // We tried `PRAGMA enable_object_cache = true` here and a
        // microbenchmark showed per-statement overhead at ~35-43us
        // regardless. Per-stmt dispatch isn't the bottleneck on
        // math-microbenchmark; the actual SQL work is. Leaving the
        // pragma off keeps memory usage simpler.
        // Sequence for fresh EqSort IDs. Term-encoded constructors
        // call `nextval` once per allocation; collisions across
        // rows with the same inputs are intentional — congruence
        // rules will unify them later.
        conn.execute("CREATE SEQUENCE __egglog_eqsort_seq START 1", [])?;
        Ok(Self {
            conn,
            functions: HashMap::new(),
            rules: Vec::new(),
            next_ts: 0,
            last_run_at: HashMap::new(),
            rules_affected: 0,
            time_mat_ns: 0,
            time_mat_act_ns: 0,
            time_act_ns: 0,
            rule_perf_ns: HashMap::new(),
            rule_to_ruleset: HashMap::new(),
            table_watermarks: HashMap::new(),
            rules_skipped: 0,
            native_uf_enabled: false,
            proofs_enabled: false,
            native_ufs: HashMap::new(),
            native_uf_pname: HashMap::new(),
            last_uf_sync_ts: HashMap::new(),
            last_inline_cong_at: HashMap::new(),
            last_termbuild_at: HashMap::new(),
            last_native_cong_at: HashMap::new(),
            native_uf_unions_synced: 0,
            backend_function_names: Vec::new(),
            backend_report_level: egglog_backend_trait::ReportLevel::default(),
            backend_container_pool: backend_impl::DuckdbContainerPool,
            backend_rule_indices: Vec::new(),
            backend_base_value_pool: base_values::DuckdbBaseValuePool::default(),
            backend_external_funcs: external_func::DuckdbExternalFuncRegistry::default(),
            registered_builtin_udfs: std::collections::HashSet::new(),
            table_sizes: Arc::new(Mutex::new(HashMap::new())),
            // Back-compat: either env var still turns the flag on. The flag is
            // the real interface (set via `enable_fast_rebuild`); `native_uf_enabled`
            // (set later) decides whether it routes to the adaptive native path
            // or the relational δuf path. `DUCK_ADAPTIVE_REBUILD` was the
            // native-path env, `DUCK_DELTA_REBUILD` the relational-path env.
            fast_rebuild: std::env::var("DUCK_ADAPTIVE_REBUILD").is_ok()
                || std::env::var("DUCK_DELTA_REBUILD").is_ok(),
            adaptive_theta: std::env::var("DUCK_ADAPTIVE_THETA")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|t| t.is_finite() && *t >= 0.0)
                .unwrap_or(0.1),
            adaptive_debug: std::env::var("DUCK_ADAPTIVE_DEBUG").is_ok(),
            adaptive_changed_ids: std::collections::HashSet::new(),
            adaptive_recent_displaced: 0,
            native_uf_canon_id_udf: HashMap::new(),
            native_uf_canon_prim_udf: HashMap::new(),
            native_uf_udf_to_table: HashMap::new(),
            native_uf_onchange: std::collections::HashSet::new(),
        })
    }

    /// Turn on the native union-find path. Must be called before
    /// any tables or rules are registered — `declare` checks this
    /// flag when it sees a `:merge (ordering-min old new)` function
    /// to decide whether to also spin up a `UfTable` + UDF.
    pub fn enable_native_uf(&mut self) {
        self.native_uf_enabled = true;
    }

    /// Turn on the `--fast-rebuild` delta-scoped rebuild. With `--native-uf`
    /// this engages the adaptive native rebuild (see
    /// [`adaptive_rebuild`](Self::adaptive_rebuild)); without it, the relational
    /// δuf fast-rebuild. Order-independent w.r.t. [`enable_native_uf`](Self::enable_native_uf):
    /// the routing is decided at run time from both flags.
    pub fn enable_fast_rebuild(&mut self) {
        self.fast_rebuild = true;
    }

    /// True iff the ADAPTIVE native rebuild is engaged. This is the DEFAULT
    /// under the native UF (`--native-uf`): native-UF provides the displaced-id
    /// deltas, so the gated rebuild variant scopes to `view ⋈ δuf`. The adaptive
    /// path accumulates the displaced-id set and chooses per-iteration between the
    /// `__UF_CHANGED__` delta semijoin and a full scan (the full-scan fallback
    /// preserves correctness when the changed set isn't small).
    ///
    /// `--fast-rebuild` is ORTHOGONAL to this gate. Under native-UF it DROPS the
    /// rebuild rule's δview-focus seminaive branch (the new-view-rows probe,
    /// empty under canon-at-creation; see `native_uf_drop_delta_view` in
    /// `add_rule` / `compile::compile_rule`), so `--native-uf` (full rebuild,
    /// keeps the δview probe) and `--native-uf --fast-rebuild` (drops it) are
    /// distinct. Without the native UF, `--fast-rebuild` instead drives the
    /// relational δuf path (compile-time, see `delta_rebuild_view_idx`).
    fn adaptive_rebuild(&self) -> bool {
        self.native_uf_enabled
    }

    /// True iff `name` is a registered native union-find function (`@UF_Sf`,
    /// registered via [`Backend::add_uf_function`] on the `--native-uf --duckdb`
    /// path). Used by the rule builder to keep both columns of a union write
    /// `(set (@UF_Sf lhs) rhs)` instead of stripping the trailing relation
    /// "output" entry.
    pub(crate) fn is_native_uf_function(&self, name: &str) -> bool {
        self.native_ufs.contains_key(name)
    }

    /// `--native-uf --duckdb` compile pre-pass. Rewrites a PR #782 rule into
    /// the SQL host-pass form before [`compile::compile_rule`]:
    ///
    /// 1. **Canon-prim -> find UDF.** Every `Term::Prim(@canon_S, [x])` (the
    ///    find-or-self primitive bound to a UF function) becomes
    ///    `Term::Prim("__duck_uf_find:<udf>", [x])`, which `compile::prim_sql`
    ///    emits as `duck_uf_<sort>_find(x)` — the in-core UF answer. This is the
    ///    EXPLICIT replacement for the relational structural demote heuristic.
    ///
    /// 2. **Rebuild host-pass.** A `@rebuild_rule` (in the `@rebuilding`
    ///    ruleset) joins the always-empty `@UFChange_S` onchange relation
    ///    against the view and filters `(= cj disp_)`. DuckDB never populates
    ///    `@UFChange_S` (no leader-change callback), so the join yields nothing.
    ///    We STRIP the onchange body atom and every Filter/Bind that references
    ///    a variable the onchange atom (uniquely) bound, leaving a view scan +
    ///    `(guard (or (bool-!= ci (@canon_S ci)) ...))` — i.e. re-canonicalize
    ///    every stale view row via the find UDF, exactly the relational
    ///    rebuild's host-pass.
    fn rewrite_native_uf_rule(&self, mut rule: Rule) -> Rule {
        // (2) Strip onchange atoms (rebuild host-pass) FIRST, so the canon
        // rewrite in (1) doesn't touch the dropped filters. Recognized by NAME
        // (`@rebuild_rule*`) — the duck trait surface drops the egglog ruleset.
        if is_rebuild_rule_name(&rule.name) {
            // Vars bound by an onchange (`@UFChange_S`) body atom.
            let mut onchange_vars: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            // Vars bound by any *other* Func atom (the view).
            let mut other_func_vars: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            for atom in &rule.body {
                if let Atom::Func { name, args } = atom {
                    let is_onchange = self.native_uf_onchange.contains(name);
                    for t in args {
                        if let Term::Var(v) = t {
                            if is_onchange {
                                onchange_vars.insert(v.clone());
                            } else {
                                other_func_vars.insert(v.clone());
                            }
                        }
                    }
                }
            }
            // A var is "onchange-only" if no surviving Func atom binds it.
            let onchange_only: std::collections::HashSet<String> = onchange_vars
                .into_iter()
                .filter(|v| !other_func_vars.contains(v))
                .collect();
            let refs_onchange_only = |t: &Term| -> bool {
                let mut found = false;
                term_for_each_var(t, &mut |v| {
                    if onchange_only.contains(v) {
                        found = true;
                    }
                });
                found
            };
            rule.body.retain(|atom| match atom {
                // Drop the onchange relation atom itself.
                Atom::Func { name, .. } => !self.native_uf_onchange.contains(name),
                // Drop filters/binds that reference an onchange-only var
                // (e.g. `(= cj disp_)`), which are now dangling.
                Atom::Filter(t) => !refs_onchange_only(t),
                Atom::Bind { var, expr } => {
                    !onchange_only.contains(var) && !refs_onchange_only(expr)
                }
            });
        }

        // (1) Canon-prim -> find UDF, across all body atoms and actions.
        if !self.native_uf_canon_prim_udf.is_empty() {
            for atom in &mut rule.body {
                match atom {
                    Atom::Func { args, .. } => {
                        for t in args.iter_mut() {
                            self.rewrite_canon_in_term(t);
                        }
                    }
                    Atom::Filter(t) => self.rewrite_canon_in_term(t),
                    Atom::Bind { expr, .. } => self.rewrite_canon_in_term(expr),
                }
            }
            for action in &mut rule.actions {
                match action {
                    Action::Insert { args, .. } | Action::LetCtor { args, .. } => {
                        for t in args.iter_mut() {
                            self.rewrite_canon_in_term(t);
                        }
                    }
                    Action::Delete { key_args, .. } => {
                        for t in key_args.iter_mut() {
                            self.rewrite_canon_in_term(t);
                        }
                    }
                    Action::LetExpr { expr, .. } => self.rewrite_canon_in_term(expr),
                    Action::Panic { .. } => {}
                }
            }
        }
        rule
    }

    /// Rewrite `Term::Prim(@canon_S, [x])` -> `Term::Prim("__duck_uf_find:<udf>",
    /// [x])` in place, recursing into nested prim/func-call args. See
    /// [`Self::rewrite_native_uf_rule`].
    fn rewrite_canon_in_term(&self, t: &mut Term) {
        match t {
            Term::Prim(op, args) => {
                for a in args.iter_mut() {
                    self.rewrite_canon_in_term(a);
                }
                if let Some(udf) = self.native_uf_canon_prim_udf.get(op) {
                    *op = format!("{NATIVE_UF_FIND_PRIM_PREFIX}{udf}");
                }
            }
            Term::FuncCall { args, .. } => {
                for a in args.iter_mut() {
                    self.rewrite_canon_in_term(a);
                }
            }
            Term::Var(_) | Term::Lit(_) => {}
        }
    }

    /// Distinct `@UF_Sf` union tables whose find UDF the (already canon-prim-
    /// rewritten) rule references. Used to gate the rebuild host-pass full-scan
    /// variant. See [`Self::rewrite_native_uf_rule`] / `compile::compile_rule`.
    fn native_uf_rule_gate_tables(&self, rule: &Rule) -> Vec<String> {
        let mut udfs: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut collect = |t: &Term| {
            term_for_each_find_udf(t, &mut |udf| {
                udfs.insert(udf.to_string());
            });
        };
        for atom in &rule.body {
            match atom {
                Atom::Func { args, .. } => args.iter().for_each(&mut collect),
                Atom::Filter(t) => collect(t),
                Atom::Bind { expr, .. } => collect(expr),
            }
        }
        for action in &rule.actions {
            match action {
                Action::Insert { args, .. } | Action::LetCtor { args, .. } => {
                    args.iter().for_each(&mut collect)
                }
                Action::Delete { key_args, .. } => key_args.iter().for_each(&mut collect),
                Action::LetExpr { expr, .. } => collect(expr),
                Action::Panic { .. } => {}
            }
        }
        let mut tables: Vec<String> = Vec::new();
        for udf in udfs {
            if let Some(t) = self.native_uf_udf_to_table.get(&udf)
                && !tables.contains(t)
            {
                tables.push(t.clone());
            }
        }
        tables
    }

    /// Register PR #782's UF-backed `:impl displaced-union-find` function
    /// `name` (`@UF_Sf`) on the `--native-uf --duckdb` path, returning its find
    /// UDF name (`duck_uf_<sort>_find`).
    ///
    /// Unlike the relational path (which spins the native UF up lazily in
    /// `declare` when it sees a `:merge ordering-min` function paired with a
    /// `@UF_S` parent table), the #782 path is driven by an *explicit*
    /// [`Backend::add_uf_function`] call. There is no separate parent table:
    /// `name` itself is both the union-write target and the sync-scan source.
    /// We register it as an append-only 2-column relation `(S S)` so each
    /// `(set (@UF_Sf lhs) rhs)` lands as a `(lhs, rhs, ts)` row (PK over both
    /// columns, `ON CONFLICT DO NOTHING` — distinct union pairs survive; a
    /// function's `lhs`-only PK would drop a second union of the same lhs).
    /// `sync_native_ufs` scans new rows into the in-core [`UfTable`] and the
    /// find UDF answers `@canon_S`/extraction finds.
    pub(crate) fn register_native_uf_function(&mut self, name: &str) -> Result<String> {
        if self.functions.contains_key(name) {
            return Err(anyhow!("native UF {name}: already registered"));
        }
        // Append-only 2-column relation (both columns eq-sort ids -> BIGINT).
        self.declare(
            name,
            FunctionInfo {
                cols: vec![ColumnTy::I64, ColumnTy::I64],
                inputs_len: 2,
                merge: None,
                eq_sort_ctor: false,
                native_uf_udf: None,
                eq_sort_pname: None,
                identity_on_miss: false,
                merge_tree: None,
                native_congruence_uf: None,
            },
        )?;
        let uf = Arc::new(Mutex::new(UfTable::new()));
        let udf_name = format!("duck_uf_{}_find", sanitize_for_udf(name));
        self.conn
            .register_scalar_function_with_state::<UfFindScalar>(&udf_name, &uf)
            .map_err(|e| anyhow!("failed to register UF UDF {udf_name}: {e}"))?;
        self.native_ufs.insert(name.to_string(), uf);
        self.native_uf_udf_to_table
            .insert(udf_name.clone(), name.to_string());
        // The scan source is the function itself (no separate `@UF_S` parent).
        self.native_uf_pname
            .insert(name.to_string(), name.to_string());
        // Tag the function so the watermark-substitution in `add_rule` maps it
        // to its own pname (a no-op) rather than leaving the gate stuck at 0.
        if let Some(info) = self.functions.get_mut(name) {
            info.native_uf_udf = Some(udf_name.clone());
        }
        Ok(udf_name)
    }

    /// Mark this `EGraph` as running a proof-tracking term encoding.
    /// Must be called before any rules are registered. Switches the
    /// eq-sort constructor hash-cons (in rule actions) to look up the
    /// bare term table instead of the canonicalized view, so each
    /// constructor application keeps its own id and proof extraction
    /// reconstructs the term the rule actually produced. See
    /// [`EGraph::proofs_enabled`].
    pub fn enable_proofs(&mut self) {
        self.proofs_enabled = true;
    }

    /// Return (registering if necessary) the [`BaseValueId`] used as a
    /// marker for Pair-sort columns. The trait pipeline carries the
    /// "this column stores a `(i64, i64)` pair" signal through the
    /// regular [`ColumnTy::Base`] surface; `trait_col_ty_to_duck`
    /// recognizes this id and maps it to
    /// [`DuckColumnTy::PairI64`] (a DuckDB `STRUCT(first BIGINT,
    /// second BIGINT)`). The frontend's `ContainerSortImpl::column_ty`
    /// calls this when the wrapped sort is a `PairSort` and the
    /// backend is duck. Other backends keep their existing
    /// container-based representation.
    pub fn pair_column_ty(&mut self) -> egglog_backend_trait::ColumnTy {
        use egglog_backend_trait::{BaseValuePool, pool_register_type};
        let pool: &mut dyn BaseValuePool = &mut self.backend_base_value_pool;
        let id = pool_register_type::<DuckPairMarker>(pool);
        egglog_backend_trait::ColumnTy::Base(id)
    }

    /// Associate a primitive name with a previously registered
    /// [`egglog_backend_trait::ExternalFunctionId`]. The frontend's
    /// typechecker calls this after `register_external_func` when the
    /// backend is duckdb, so the duckdb rule-builder can later
    /// translate `ExternalFunctionId` references into
    /// `Term::Prim(name, ...)` calls in the duck IR.
    ///
    /// For primitives that need to run real Rust code at SQL eval
    /// time (currently `from-string` and `bigrat` — the
    /// constants-only path Herbie's dumps exercise), this also
    /// registers a DuckDB scalar UDF named `__egglog_<sanitized>`
    /// that calls into the shared base-value pool. compile.rs's
    /// `prim_sql` routes those primitive names to the UDF.
    ///
    /// No-op for unknown ids.
    pub fn set_external_func_name(
        &mut self,
        id: egglog_backend_trait::ExternalFunctionId,
        name: String,
    ) {
        self.backend_external_funcs.set_name(id, name.clone());
        // `--native-uf --duckdb`: the frontend rebinds the PR #782 `@canon_S`
        // find-or-self primitive name to the canon-prim id returned by
        // `add_uf_function`. Record `canon-name -> find UDF` so the compile
        // pre-pass can rewrite `Term::Prim(canon-name, [x])` into a UDF call.
        if let Some(udf) = self.native_uf_canon_id_udf.get(&id).cloned() {
            self.native_uf_canon_prim_udf.insert(name.clone(), udf);
        }
        self.register_builtin_prim_udf(&name);
    }

    /// Register a DuckDB scalar UDF for a known builtin primitive
    /// name, if we haven't already.
    ///
    /// Two families of names:
    /// - `from-string` / `bigrat` — the BigInt / BigRat constructors
    ///   (Herbie writes every numeric literal as
    ///   `(bigrat (from-string "...") (from-string "..."))`).
    /// - `bigrat-<op>` — sort-specific aliases the frontend emits via
    ///   `rename_prim` for BigRat-overloaded ops (`+`, `<`, `round`, …).
    ///   Each registers an `__egglog_<name>` UDF parameterized by
    ///   [`BigratOp`].
    ///
    /// The required `BaseValueId`s for `Z` and (for `bigrat`/`bigrat-…`)
    /// `Q` must already be registered in the pool —
    /// `BaseSortImpl::register_type` does this when the egglog frontend
    /// calls `add_base_sort`. The frontend calls this method indirectly
    /// via `set_external_func_name` only *after* sort registration, so
    /// the ids resolve.
    fn register_builtin_prim_udf(&mut self, name: &str) {
        use egglog_backend_trait::BaseValuePool;
        use num::{BigInt, BigRational};
        use std::any::TypeId;
        if self.registered_builtin_udfs.contains(name) {
            return;
        }
        let pool_dyn: &dyn BaseValuePool = &self.backend_base_value_pool;
        let bigint_type_id = TypeId::of::<egglog_core_relations::Boxed<BigInt>>();
        let bigint_ty_opt = if pool_dyn.has_ty(bigint_type_id) {
            Some(pool_dyn.get_ty_by_type_id(bigint_type_id))
        } else {
            None
        };
        // String UDFs need `Boxed<String>` registered — `StringSort` is
        // declared by `with_backend` before any frontend code calls in
        // here, so this lookup always succeeds for our test paths.
        let string_type_id = TypeId::of::<egglog_core_relations::Boxed<String>>();
        let string_ty_opt = if pool_dyn.has_ty(string_type_id) {
            Some(pool_dyn.get_ty_by_type_id(string_type_id))
        } else {
            None
        };
        // Handle string UDFs first so they don't depend on BigInt
        // being registered (the BigInt early-return below was added
        // for bigrat-family UDFs).
        let result_str: Option<duckdb::Result<()>> = match name {
            "string-concat" => string_ty_opt.map(|string_ty| {
                let state = StringPoolState {
                    pool: self.backend_base_value_pool.clone(),
                    string_ty,
                    bigint_ty: bigint_ty_opt,
                };
                self.conn
                    .register_scalar_function_with_state::<StringConcatScalar>(
                        "__egglog_string_concat",
                        &state,
                    )
            }),
            "replace" => string_ty_opt.map(|string_ty| {
                let state = StringPoolState {
                    pool: self.backend_base_value_pool.clone(),
                    string_ty,
                    bigint_ty: bigint_ty_opt,
                };
                self.conn
                    .register_scalar_function_with_state::<ReplaceScalar>(
                        "__egglog_replace",
                        &state,
                    )
            }),
            "count-matches" => string_ty_opt.map(|string_ty| {
                let state = StringPoolState {
                    pool: self.backend_base_value_pool.clone(),
                    string_ty,
                    bigint_ty: bigint_ty_opt,
                };
                self.conn
                    .register_scalar_function_with_state::<CountMatchesScalar>(
                        "__egglog_count_matches",
                        &state,
                    )
            }),
            "i64-to-string" => string_ty_opt.map(|string_ty| {
                let state = StringPoolState {
                    pool: self.backend_base_value_pool.clone(),
                    string_ty,
                    bigint_ty: bigint_ty_opt,
                };
                self.conn
                    .register_scalar_function_with_state::<I64ToStringScalar>(
                        "__egglog_i64_to_string",
                        &state,
                    )
            }),
            "f64-to-string" => string_ty_opt.map(|string_ty| {
                let state = StringPoolState {
                    pool: self.backend_base_value_pool.clone(),
                    string_ty,
                    bigint_ty: bigint_ty_opt,
                };
                self.conn
                    .register_scalar_function_with_state::<F64ToStringScalar>(
                        "__egglog_f64_to_string",
                        &state,
                    )
            }),
            "bigint-to-string" => {
                string_ty_opt
                    .zip(bigint_ty_opt)
                    .map(|(string_ty, _bigint_ty)| {
                        let state = StringPoolState {
                            pool: self.backend_base_value_pool.clone(),
                            string_ty,
                            bigint_ty: bigint_ty_opt,
                        };
                        self.conn
                            .register_scalar_function_with_state::<BigIntToStringScalar>(
                                "__egglog_bigint_to_string",
                                &state,
                            )
                    })
            }
            _ => None,
        };
        if let Some(r) = result_str {
            match r {
                Ok(()) => {
                    self.registered_builtin_udfs.insert(name.to_string());
                }
                Err(e) => log::warn!("failed to register builtin UDF for {name}: {e}"),
            }
            return;
        }

        // Bigrat-family UDFs require BigInt; bail if not registered.
        let Some(bigint_ty) = bigint_ty_opt else {
            return;
        };
        let bigrat_type_id = TypeId::of::<egglog_core_relations::Boxed<BigRational>>();
        let bigrat_ty = if pool_dyn.has_ty(bigrat_type_id) {
            Some(pool_dyn.get_ty_by_type_id(bigrat_type_id))
        } else {
            None
        };
        let result: duckdb::Result<()> = match name {
            "bigint" => {
                let state = BigPoolState {
                    pool: self.backend_base_value_pool.clone(),
                    bigint_ty,
                    bigrat_ty,
                };
                self.conn
                    .register_scalar_function_with_state::<BigintConstructorScalar>(
                        "__egglog_bigint",
                        &state,
                    )
            }
            "from-string" => {
                let state = BigPoolState {
                    pool: self.backend_base_value_pool.clone(),
                    bigint_ty,
                    bigrat_ty,
                };
                self.conn
                    .register_scalar_function_with_state::<FromStringScalar>(
                        "__egglog_from_string",
                        &state,
                    )
            }
            "bigrat" => {
                if bigrat_ty.is_none() {
                    return;
                }
                let state = BigPoolState {
                    pool: self.backend_base_value_pool.clone(),
                    bigint_ty,
                    bigrat_ty,
                };
                self.conn
                    .register_scalar_function_with_state::<BigratScalar>("__egglog_bigrat", &state)
            }
            "get-size!" => {
                let state = GetSizeState {
                    table_sizes: Arc::clone(&self.table_sizes),
                };
                self.conn
                    .register_scalar_function_with_state::<GetSizeScalar>(
                        "__egglog_get_size",
                        &state,
                    )
            }
            other if BigintOp::from_duck_name(other).is_some() => {
                let op = BigintOp::from_duck_name(other).unwrap();
                let state = BigintExecState {
                    pool: self.backend_base_value_pool.clone(),
                    bigint_ty,
                    op,
                };
                let udf_name = format!("__egglog_{}", other.replace('-', "_"));
                self.conn
                    .register_scalar_function_with_state::<BigintBinaryScalar>(&udf_name, &state)
            }
            other if BigratOp::from_duck_name(other).is_some() => {
                let Some(bigrat_ty) = bigrat_ty else { return };
                let op = BigratOp::from_duck_name(other).unwrap();
                let state = BigratExecState {
                    pool: self.backend_base_value_pool.clone(),
                    bigint_ty,
                    bigrat_ty,
                    op,
                };
                let udf_name = format!("__egglog_{}", other.replace('-', "_"));
                if op.is_comparison() {
                    self.conn
                        .register_scalar_function_with_state::<BigratCmpScalar>(&udf_name, &state)
                } else if op.returns_z() {
                    self.conn
                        .register_scalar_function_with_state::<BigratNumDenomScalar>(
                            &udf_name, &state,
                        )
                } else if op.returns_f64() {
                    self.conn
                        .register_scalar_function_with_state::<BigratToF64Scalar>(&udf_name, &state)
                } else if op.is_unary() {
                    self.conn
                        .register_scalar_function_with_state::<BigratUnaryQScalar>(
                            &udf_name, &state,
                        )
                } else {
                    self.conn
                        .register_scalar_function_with_state::<BigratBinaryQScalar>(
                            &udf_name, &state,
                        )
                }
            }
            _ => return,
        };
        match result {
            Ok(()) => {
                self.registered_builtin_udfs.insert(name.to_string());
            }
            Err(e) => {
                log::warn!("failed to register builtin UDF for {name}: {e}");
            }
        }
    }

    /// Look up the primitive name associated with `id`, if any.
    /// Returns `None` for unregistered or unnamed ids.
    pub fn external_func_name(&self, id: egglog_backend_trait::ExternalFunctionId) -> Option<&str> {
        self.backend_external_funcs.name(id)
    }

    /// Borrow the underlying [`base_values::DuckdbBaseValuePool`].
    /// Public so the egglog frontend's extraction path can reach the
    /// inner `BaseValues` through `inner()` without depending on the
    /// crate-private field name.
    pub fn base_value_pool_typed(&self) -> &base_values::DuckdbBaseValuePool {
        &self.backend_base_value_pool
    }

    /// Constant-fold a builtin primitive call: if every arg is a
    /// `Literal` and `name` is a primitive we know how to evaluate
    /// off-line (`from-string`, `bigrat`, `bigrat-add`, …), return a
    /// `Term::Lit(...)` with the pre-interned result. Otherwise
    /// return `None` and the caller emits the regular `Term::Prim`
    /// call.
    ///
    /// This is the load-bearing performance fix for Herbie on duckdb:
    /// Herbie's rules contain hundreds of inlined
    /// `(bigrat (from-string "X") (from-string "Y"))` constants. Without
    /// folding, each one expands to a nested UDF chain in SQL — the
    /// resulting rule WHERE/SELECT lists blow past DuckDB's expression
    /// depth limit and trigger massive per-row UDF overhead. With
    /// folding the same call collapses to a single i64 handle, and
    /// repeated references share the same handle thanks to intern
    /// stability.
    pub fn fold_builtin_prim(&self, name: &str, args: &[Term]) -> Option<Term> {
        use egglog_backend_trait::{BaseValuePool, Value};
        use egglog_numeric_id::NumericId;
        use num::{BigInt, BigRational};
        use std::any::TypeId;
        use std::str::FromStr;

        let pool_dyn: &dyn BaseValuePool = &self.backend_base_value_pool;
        let bigint_ty_id = TypeId::of::<egglog_core_relations::Boxed<BigInt>>();
        let bigrat_ty_id = TypeId::of::<egglog_core_relations::Boxed<BigRational>>();
        if !pool_dyn.has_ty(bigint_ty_id) {
            return None;
        }
        let bigint_ty = pool_dyn.get_ty_by_type_id(bigint_ty_id);
        let bigrat_ty = if pool_dyn.has_ty(bigrat_ty_id) {
            Some(pool_dyn.get_ty_by_type_id(bigrat_ty_id))
        } else {
            None
        };

        // Helper: extract a string from an interned-handle arg.
        // Strings live in the base-value pool now (matches bridge
        // encoding); the arg's `Literal::I64` is the handle.
        let string_ty = if pool_dyn.has_ty(TypeId::of::<egglog_core_relations::Boxed<String>>()) {
            Some(pool_dyn.get_ty_by_type_id(TypeId::of::<egglog_core_relations::Boxed<String>>()))
        } else {
            None
        };
        let as_str = |t: &Term| -> Option<String> {
            if let Term::Lit(Literal::I64(handle)) = t {
                let ty = string_ty?;
                let val = Value::new(*handle as u32);
                let boxed = pool_dyn.unwrap_dyn(ty, val);
                let s: &egglog_core_relations::Boxed<String> = boxed.downcast_ref()?;
                Some(s.0.clone())
            } else {
                None
            }
        };
        // Helper: extract an i64 literal arg.
        fn as_i64(t: &Term) -> Option<i64> {
            if let Term::Lit(Literal::I64(i)) = t {
                Some(*i)
            } else {
                None
            }
        }
        // Helper: unwrap a Q-handle (BIGINT) into a BigRational.
        let unwrap_q = |raw: i64| -> Option<BigRational> {
            let bigrat_ty = bigrat_ty?;
            let val = Value::new(raw as u32);
            let boxed = pool_dyn.unwrap_dyn(bigrat_ty, val);
            let q: &egglog_core_relations::Boxed<BigRational> = boxed.downcast_ref()?;
            Some(q.0.clone())
        };
        // Helper: intern a BigInt as Z and return its Term::Lit.
        let intern_z = |bi: BigInt| -> Term {
            let z = egglog_core_relations::Boxed::new(bi);
            let v = pool_dyn.intern_dyn(bigint_ty, Box::new(z));
            Term::Lit(Literal::I64(v.rep() as i64))
        };
        // Helper: intern a BigRational as Q and return its Term::Lit.
        let intern_q = |q: BigRational| -> Option<Term> {
            let bigrat_ty = bigrat_ty?;
            let qb = egglog_core_relations::Boxed::new(q);
            let v = pool_dyn.intern_dyn(bigrat_ty, Box::new(qb));
            Some(Term::Lit(Literal::I64(v.rep() as i64)))
        };

        match name {
            "from-string" => {
                let s = as_str(args.first()?)?;
                let bi = BigInt::from_str(&s).ok()?;
                Some(intern_z(bi))
            }
            "bigrat" => {
                let n_raw = as_i64(args.first()?)?;
                let d_raw = as_i64(args.get(1)?)?;
                let n_val = Value::new(n_raw as u32);
                let d_val = Value::new(d_raw as u32);
                let n_boxed = pool_dyn.unwrap_dyn(bigint_ty, n_val);
                let d_boxed = pool_dyn.unwrap_dyn(bigint_ty, d_val);
                let n: &egglog_core_relations::Boxed<BigInt> = n_boxed.downcast_ref()?;
                let d: &egglog_core_relations::Boxed<BigInt> = d_boxed.downcast_ref()?;
                let q = BigRational::new(n.0.clone(), d.0.clone());
                intern_q(q)
            }
            other => {
                let op = BigratOp::from_duck_name(other)?;
                // Comparison ops fold to a Bool literal — emit it
                // and the rule_builder's downstream Filter on this
                // term sees `Lit(Bool(true/false))`.
                if op.is_comparison() {
                    let a = unwrap_q(as_i64(args.first()?)?)?;
                    let b = unwrap_q(as_i64(args.get(1)?)?)?;
                    let r = match op {
                        BigratOp::Lt => a < b,
                        BigratOp::Gt => a > b,
                        BigratOp::Le => a <= b,
                        BigratOp::Ge => a >= b,
                        _ => unreachable!(),
                    };
                    return Some(Term::Lit(Literal::Bool(r)));
                }
                // numer/denom fold to Z handles.
                if op.returns_z() {
                    let q = unwrap_q(as_i64(args.first()?)?)?;
                    let bi = match op {
                        BigratOp::Numer => q.numer().clone(),
                        BigratOp::Denom => q.denom().clone(),
                        _ => unreachable!(),
                    };
                    return Some(intern_z(bi));
                }
                // to-f64 folds to F64 literal.
                if op.returns_f64() {
                    use num::ToPrimitive;
                    let q = unwrap_q(as_i64(args.first()?)?)?;
                    return Some(Term::Lit(Literal::F64(q.to_f64()?)));
                }
                // Unary Q→Q: round, sqrt, neg, abs, ceil, floor, log, cbrt
                if op.is_unary() {
                    let q = unwrap_q(as_i64(args.first()?)?)?;
                    let result = run_bigrat_q(op, &[q])?;
                    return intern_q(result);
                }
                // Binary Q×Q→Q: add, sub, mul, div, min, max, pow
                let a = unwrap_q(as_i64(args.first()?)?)?;
                let b = unwrap_q(as_i64(args.get(1)?)?)?;
                let result = run_bigrat_q(op, &[a, b])?;
                intern_q(result)
            }
        }
    }

    /// Refresh the row-count snapshot consulted by the `get-size!`
    /// UDF. Queries each function table once via `SELECT COUNT(*)`
    /// and writes the result into the shared `table_sizes` map.
    /// Internal-symbol-prefixed tables (term-encoding bookkeeping)
    /// are excluded so the result matches the bridge `get-size!`
    /// primitive, which applies the same filter.
    ///
    /// Callers should invoke this immediately before each
    /// existence-check query that may reference `get-size!`. The
    /// snapshot stays valid for the duration of that query.
    pub fn refresh_table_sizes(&self) -> Result<()> {
        const INTERNAL_PREFIX: &str = "@";
        // Filter to non-internal tables. Order is stable so the
        // returned columns line up with `names` for read-out.
        let names: Vec<&str> = self
            .functions
            .keys()
            .filter(|n| !n.starts_with(INTERNAL_PREFIX))
            .map(|s| s.as_str())
            .collect();
        if names.is_empty() {
            let mut guard = self
                .table_sizes
                .lock()
                .map_err(|e| anyhow!("table_sizes mutex poisoned: {e}"))?;
            guard.clear();
            return Ok(());
        }
        // Batch into a single SELECT with N scalar `(SELECT COUNT(*)
        // FROM t)` subqueries. Avoids N round-trips per :until check
        // (Herbie's `(repeat 50 …)` schedules call `get-size!` once
        // per iteration; the per-query overhead added up to seconds).
        let cols: Vec<String> = names
            .iter()
            .map(|n| format!("(SELECT COUNT(*) FROM {})", q(n)))
            .collect();
        let sql = format!("SELECT {}", cols.join(", "));
        let mut new_sizes: HashMap<String, i64> = HashMap::new();
        self.conn
            .query_row(&sql, [], |row| {
                for (i, n) in names.iter().enumerate() {
                    let count: i64 = row.get(i)?;
                    new_sizes.insert((*n).to_string(), count);
                }
                Ok(())
            })
            .map_err(|e| anyhow!("refresh_table_sizes: batched COUNT failed: {e}"))?;
        let mut guard = self
            .table_sizes
            .lock()
            .map_err(|e| anyhow!("table_sizes mutex poisoned: {e}"))?;
        *guard = new_sizes;
        Ok(())
    }

    /// Currently registered native union-find tables, for read-only
    /// inspection (testing, diagnostics). Mutating these directly
    /// would skip the queue + drain protocol; callers that need to
    /// add unions should go through the SQL path.
    pub fn native_uf_table_count(&self) -> usize {
        self.native_ufs.len()
    }

    /// Total rows of union assertions pulled into native UFs since
    /// this `EGraph` was created. Surfaced by `DUCK_PERF_DUMP`.
    pub fn native_uf_unions_synced(&self) -> u64 {
        self.native_uf_unions_synced
    }

    /// Pull every union assertion newer than the last sync from
    /// each native UF's corresponding pname (raw UF) table, apply
    /// them via the queue + drain protocol, and advance the
    /// per-UF watermark.
    ///
    /// The find UDF reads from each `UfTable` via `find_ro`; this
    /// method is what keeps that read view consistent with whatever
    /// rule actions have written into the SQL pname tables. The
    /// runner calls it at the top of each iteration so any UDF
    /// invocations within that iteration's rules see fresh state.
    fn sync_native_ufs(&mut self) -> Result<usize> {
        if self.native_ufs.is_empty() {
            return Ok(0);
        }
        let mut total_displaced: usize = 0;
        // Snapshot names to avoid double-borrowing self.
        let names: Vec<String> = self.native_ufs.keys().cloned().collect();
        for uf_name in names {
            let pname = match self.native_uf_pname.get(&uf_name) {
                Some(p) => p.clone(),
                None => continue,
            };
            // Watermark check: if pname's max-ts watermark hasn't
            // advanced past what we synced, there's nothing new.
            let pname_wm = self.table_watermarks.get(&pname).copied().unwrap_or(0);
            let last_synced = self.last_uf_sync_ts.get(&uf_name).copied().unwrap_or(-1);
            if pname_wm <= last_synced {
                continue;
            }
            let sql = format!(
                "SELECT c0, c1, ts FROM {} WHERE ts > {}",
                q(&pname),
                last_synced,
            );
            let uf_arc = self.native_ufs[&uf_name].clone();
            let mut stmt = self.conn.prepare(&sql)?;
            let mut rows = stmt.query([])?;
            let mut max_ts_seen = last_synced;
            let mut count: u64 = 0;
            let accumulate = self.adaptive_rebuild();
            {
                let mut uf = uf_arc.lock().unwrap();
                while let Some(row) = rows.next()? {
                    let a: i64 = row.get(0)?;
                    let b: i64 = row.get(1)?;
                    let ts: i64 = row.get(2)?;
                    uf.enqueue_union(a, b);
                    if ts > max_ts_seen {
                        max_ts_seen = ts;
                    }
                    count += 1;
                }
                if count > 0 {
                    uf.drain_pending();
                    let drained = uf.drain_displaced();
                    total_displaced += drained.len();
                    // Adaptive path: keep the displaced ids around so
                    // the next rebuild iteration can restrict its scan
                    // to rows referencing one of them. Cleared only
                    // after a full scan re-canonicalizes everything.
                    if accumulate {
                        self.adaptive_changed_ids.extend(drained);
                    }
                }
            }
            self.native_uf_unions_synced = self.native_uf_unions_synced.wrapping_add(count);
            self.last_uf_sync_ts.insert(uf_name, max_ts_seen);
        }
        if self.adaptive_rebuild() {
            self.adaptive_recent_displaced = total_displaced;
        }
        Ok(total_displaced)
    }

    /// Adaptive denominator: the total number of rows in the eq-sort
    /// view tables the rebuild scans. This is the "relevant table
    /// size" — the cost of a full-scan rebuild iteration scales with
    /// it, and a delta scan pays off only when the accumulated
    /// changed set is a small fraction of it. Using the view row
    /// count (rather than the UF non-root count) is essential for the
    /// sparse case: a workload can have a huge view but only a few
    /// unioned ids, where a UF-based denominator would be tiny and
    /// wrongly force full scans. Counts are read from DuckDB's O(1)
    /// row-count metadata, summed over the eq-sort views.
    fn adaptive_id_space(&self) -> usize {
        // The relevant tables are exactly the body tables of rules
        // that carry a gated rebuild variant — i.e. the views the
        // gated full/delta scans read. Summing their row counts gives
        // the true scan footprint regardless of the view-naming
        // convention the term encoding chose.
        let mut tables: Vec<&str> = Vec::new();
        for rule in &self.rules {
            if rule.variants.iter().any(|v| v.gate_table.is_some()) {
                for t in &rule.body_tables {
                    if !tables.contains(&t.as_str()) {
                        tables.push(t.as_str());
                    }
                }
            }
        }
        let mut total: i64 = 0;
        for t in tables {
            if let Ok(n) = self
                .conn
                .query_row(&format!("SELECT COUNT(*) FROM {}", q(t)), [], |r| {
                    r.get::<_, i64>(0)
                })
            {
                total += n;
            }
        }
        total.max(0) as usize
    }

    /// True iff any of these rules carries a gated (native-UF
    /// recovery) variant. The adaptive decision/reset only matters
    /// for such iterations — non-gated iterations (user rules, UF
    /// maintenance) never run a full scan, so resetting the
    /// accumulated changed set on them would discard the delta a
    /// later rebuild iteration still needs.
    fn rules_have_gated(&self, indices: &[usize]) -> bool {
        indices.iter().any(|&i| {
            self.rules
                .get(i)
                .is_some_and(|r| r.variants.iter().any(|v| v.gate_table.is_some()))
        })
    }

    /// Decide this iteration's gate mode and, when `Delta`, (re)build
    /// and populate the `__UF_CHANGED__` temp table from the
    /// accumulated changed-id set. Returns `FullScan` (the default,
    /// non-adaptive behavior) unless the adaptive flag is on AND the
    /// accumulated changed set is small relative to the id space.
    ///
    /// `FullScan` mode signals (via the returned mode) that the
    /// caller must reset `adaptive_changed_ids` *after* the rebuild
    /// rules run — a full scan re-canonicalizes every row, so the
    /// accumulated delta is fully consumed and can be safely dropped,
    /// which bounds the set below `theta * id_space` going forward.
    fn adaptive_gate_mode(&mut self) -> Result<GateMode> {
        if !self.adaptive_rebuild() {
            return Ok(GateMode::FullScan);
        }
        let changed = self.adaptive_changed_ids.len();
        let recent = self.adaptive_recent_displaced;
        let id_space = self.adaptive_id_space();
        // Decide on the RECENT churn (ids displaced by the latest
        // sync), not the full accumulated set: the accumulated set is
        // what the delta scan must cover for correctness, so it only
        // shrinks at a full-scan reset, but it would otherwise grow
        // monotonically and pin the decision to always-full-scan.
        // A burst of recent unions (> theta * id_space) means many
        // rows just went stale — cheaper to full-scan and reset; a
        // quiet iteration engages delta.
        // Engage delta only when BOTH the recent churn and the
        // total accumulated changed set are below `theta * id_space`.
        // Bounding on the accumulated set is what guarantees
        // no-regression: the delta scan semijoins against that set,
        // and since we cannot soundly reset it (see
        // `adaptive_reset_changed`), we instead refuse delta once it
        // would exceed the full scan's cost — which on heavy eqsat
        // happens almost immediately, so heavy workloads fall back to
        // the full scan (identical to the baseline). Light / sparse
        // workloads keep a small accumulated set and stay in delta.
        let threshold = (self.adaptive_theta * id_space as f64).ceil() as usize;
        let mode = if id_space == 0 || recent > threshold || changed > threshold {
            GateMode::FullScan
        } else {
            GateMode::Delta
        };
        if self.adaptive_debug {
            eprintln!(
                "[duck/adaptive] mode={:?} recent={} accumulated={} id_space={} theta={} threshold={}",
                mode, recent, changed, id_space, self.adaptive_theta, threshold,
            );
        }
        if mode == GateMode::Delta {
            self.materialize_uf_changed()?;
        }
        Ok(mode)
    }

    /// (Re)create the `__UF_CHANGED__` temp table holding the current
    /// accumulated changed-id set, one `id BIGINT` row each. Delta
    /// gated variants semijoin against it. Rebuilt fresh each delta
    /// iteration so it reflects the latest accumulation.
    fn materialize_uf_changed(&mut self) -> Result<()> {
        self.conn.execute(
            &format!(
                "CREATE OR REPLACE TEMP TABLE {} (id BIGINT)",
                UF_CHANGED_PLACEHOLDER
            ),
            [],
        )?;
        if self.adaptive_changed_ids.is_empty() {
            return Ok(());
        }
        // Insert as a single multi-row VALUES statement. ids are i64
        // values drawn from the eq-sort sequence — no SQL-injection
        // surface. The set only grows while delta mode is engaged
        // (small recent + small accumulated relative to the views),
        // so it stays bounded on the workloads delta actually runs.
        let values: String = self
            .adaptive_changed_ids
            .iter()
            .map(|id| format!("({id})"))
            .collect::<Vec<_>>()
            .join(", ");
        self.conn.execute(
            &format!("INSERT INTO {UF_CHANGED_PLACEHOLDER} (id) VALUES {values}"),
            [],
        )?;
        Ok(())
    }

    /// Clear the accumulated changed-id set. Called after a full-scan
    /// rebuild iteration, which re-canonicalizes every row and thus
    /// consumes the entire pending delta.
    fn adaptive_reset_changed(&mut self) {
        // Resetting the accumulated set to empty after a full scan is
        // UNSOUND with the current seminaive rebuild/congruence
        // pipeline: the full-scan gated variant restricts to `ts <
        // ?2` (it skips current-iteration rows), so it does not
        // re-canonicalize everything, and a later delta scan with the
        // emptied set then misses rows that reference an id displaced
        // before the reset — observed as un-merged congruent
        // duplicates on math-microbenchmark. So we DON'T reset by
        // default; the accumulated set is kept complete and the mode
        // decision keys off the recent (per-sync) churn instead. The
        // env override is retained for experimentation only.
        if self.adaptive_rebuild() && std::env::var("DUCK_RESET_AFTER_FULLSCAN").is_ok() {
            self.adaptive_changed_ids.clear();
        }
    }

    /// Bump `table`'s watermark to `ts` if `ts` is newer than the
    /// current value. Cheap: called from every insert path.
    fn bump_watermark(&mut self, table: &str, ts: i64) {
        let e = self.table_watermarks.entry(table.to_string()).or_insert(0);
        if ts > *e {
            *e = ts;
        }
    }

    /// Total rule firings short-circuited by the watermark gate
    /// since this `EGraph` was created.
    pub fn rules_skipped(&self) -> u64 {
        self.rules_skipped
    }

    /// Per-category nanosecond accumulators for SQL `execute` calls.
    /// Returns `(materialize, materialized_action, simple_action)`.
    /// Read after `run_program` to see where time goes.
    pub fn perf_timings_ns(&self) -> (u64, u64, u64) {
        (self.time_mat_ns, self.time_mat_act_ns, self.time_act_ns)
    }

    /// Run `EXPLAIN ANALYZE` against the materialize SELECT of each
    /// variant of the top-`top` rules by total time. Uses `?1 = 0`
    /// and `?2 = next_ts + 1` so the scan covers all rows currently
    /// in each body table — that's the "everything fits" plan
    /// DuckDB picks once the DB is fully populated, which is what
    /// we care about for steady-state behavior. Returns a
    /// pre-formatted multi-line report.
    pub fn explain_top_rules(&self, top: usize) -> anyhow::Result<String> {
        let mut by_total: Vec<(String, u64)> = self
            .rule_perf_ns
            .iter()
            .map(|(n, (m, a))| (n.clone(), m + a))
            .collect();
        by_total.sort_by(|x, y| y.1.cmp(&x.1));
        let last: i64 = 0;
        let cur: i64 = self.next_ts + 1;
        let mut out = String::new();
        for (name, total) in by_total.into_iter().take(top) {
            let Some(rule) = self.rules.iter().find(|r| r.name == name) else {
                continue;
            };
            out.push_str(&format!(
                "=== {name} (ruleset={}, total {:.3}s) ===\n",
                rule.ruleset,
                total as f64 / 1e9,
            ));
            for (vi, variant) in rule.variants.iter().enumerate() {
                let Some(mat) = &variant.materialize else {
                    out.push_str(&format!("  variant {vi}: no materialize\n"));
                    continue;
                };
                // The materialize SQL is
                // `CREATE OR REPLACE TEMP TABLE "..." AS SELECT ...`.
                // Strip the wrapper so EXPLAIN ANALYZE doesn't
                // (re)create a temp table as a side effect.
                let Some(idx) = mat.find(" AS SELECT ") else {
                    out.push_str(&format!("  variant {vi}: unexpected SQL shape\n"));
                    continue;
                };
                let inner = &mat[idx + " AS ".len()..];
                let bound = inner
                    .replace("?1", &last.to_string())
                    .replace("?2", &cur.to_string());
                let explain_sql = format!("EXPLAIN ANALYZE {bound}");
                out.push_str(&format!("\n  --- variant {vi} ---\n"));
                match self.conn.prepare(&explain_sql) {
                    Ok(mut stmt) => {
                        let mut rows = stmt.query([])?;
                        while let Some(row) = rows.next()? {
                            // EXPLAIN ANALYZE returns
                            // (explain_key, explain_value).
                            let val: Result<String, _> = row.get::<_, String>(1);
                            if let Ok(v) = val {
                                for line in v.lines() {
                                    out.push_str("    ");
                                    out.push_str(line);
                                    out.push('\n');
                                }
                            }
                        }
                    }
                    Err(e) => {
                        out.push_str(&format!("    error: {e}\n    sql: {explain_sql}\n"));
                    }
                }
            }
            out.push('\n');
        }
        Ok(out)
    }

    /// `(rule_name, ruleset, mat_ns, act_ns)` rows sorted by total
    /// time descending. Empty if the program hasn't run yet.
    pub fn perf_per_rule(&self) -> Vec<(String, String, u64, u64)> {
        let mut rows: Vec<(String, String, u64, u64)> = self
            .rule_perf_ns
            .iter()
            .map(|(rn, &(m, a))| {
                let rs = self.rule_to_ruleset.get(rn).cloned().unwrap_or_default();
                (rn.clone(), rs, m, a)
            })
            .collect();
        rows.sort_by(|x, y| (y.2 + y.3).cmp(&(x.2 + x.3)));
        rows
    }

    /// Classify a single (rule_name, ruleset) into a coarse phase
    /// bucket. In the default `--duckdb` (non-native-uf) path the
    /// term-encoding maintenance rules carry an empty ruleset, so we
    /// key off the rule NAME (which is reliably populated:
    /// `@rebuild_rule*`, `@congruence_rule*`, `*uf_update`,
    /// `*parent*`, `*cleanup*`); we still consult the ruleset name as
    /// a fallback for the native-uf path.
    fn classify_rule(rule_name: &str, ruleset: &str) -> &'static str {
        let bare_name = rule_name.trim_start_matches('@');
        // Strip a trailing `#<idx>` instance suffix, then trailing
        // digits, to get the family prefix.
        let fam = bare_name
            .split('#')
            .next()
            .unwrap_or(bare_name)
            .trim_end_matches(|c: char| c.is_ascii_digit());
        if fam.starts_with("rebuild_rule") || is_rebuilding_ruleset(ruleset) {
            "rebuild"
        } else if fam.starts_with("congruence_rule")
            || is_congruence_rule_name(rule_name)
            || ruleset
                .trim_end_matches(|c: char| c.is_ascii_digit())
                .trim_start_matches('@')
                == "congruence"
        {
            "congruence"
        } else if fam.ends_with("uf_update")
            || fam.contains("parent")
            || fam.contains("uf_function_index")
            || is_uf_maintenance_ruleset(ruleset)
        {
            "maintenance"
        } else if fam.contains("cleanup")
            || ruleset
                .trim_end_matches(|c: char| c.is_ascii_digit())
                .trim_start_matches('@')
                == "cleanup"
        {
            "cleanup"
        } else {
            "user"
        }
    }

    /// `(label, kind, mat_ns, act_ns, n_rules)` rows rolled up from
    /// the per-rule timings and grouped by phase `kind`, sorted by
    /// total time descending. `kind` is one of `"rebuild"`,
    /// `"congruence"`, `"maintenance"` (UF canonicalization: parent /
    /// single_parent / uf_function_index / uf_update), `"cleanup"`,
    /// or `"user"` (the actual rewrite rules). `label` mirrors `kind`.
    /// Surfaced by `DUCK_PERF_DUMP` so we can attribute runtime to
    /// rebuild/UF-maintenance vs user rewrites. Classification keys on
    /// rule name because term-encoded maintenance rules carry an empty
    /// ruleset in the default path.
    pub fn perf_per_ruleset(&self) -> Vec<(String, &'static str, u64, u64, u64)> {
        let mut by_kind: HashMap<&'static str, (u64, u64, u64)> = HashMap::new();
        for (rn, &(m, a)) in &self.rule_perf_ns {
            let rs = self
                .rule_to_ruleset
                .get(rn)
                .map(String::as_str)
                .unwrap_or("");
            let kind = Self::classify_rule(rn, rs);
            let e = by_kind.entry(kind).or_insert((0, 0, 0));
            e.0 = e.0.wrapping_add(m);
            e.1 = e.1.wrapping_add(a);
            e.2 += 1;
        }
        let mut rows: Vec<(String, &'static str, u64, u64, u64)> = by_kind
            .into_iter()
            .map(|(kind, (m, a, n))| (kind.to_string(), kind, m, a, n))
            .collect();
        rows.sort_by(|x, y| (y.2 + y.3).cmp(&(x.2 + x.3)));
        rows
    }

    /// Cumulative count of rows affected by rule action SQLs across
    /// all iterations so far. Frontend snapshots this around its
    /// schedule loops to detect saturation precisely.
    pub fn rules_affected_total(&self) -> u64 {
        self.rules_affected
    }

    /// Diagnostic-only access to the underlying DuckDB connection,
    /// used by the frontend's `dump_tables`. Not for general use.
    pub fn conn_for_dump(&self) -> &Connection {
        &self.conn
    }

    fn debug_sql(&self, label: &str, sql: &str) {
        if std::env::var("DUCK_TRACE_SQL").is_ok() {
            eprintln!("[duck/{label}] {sql}");
        }
    }

    /// Allocate a fresh EqSort ID, insert a row into a constructor
    /// table with that ID, and return the ID. Used by term encoding
    /// for `(let v (C a b))` patterns at top level.  May produce
    /// rows with duplicate input keys but distinct IDs; the
    /// term-encoding-generated congruence rule will unify those
    /// IDs in UF and the rebuild rule will clean up.
    pub fn allocate_and_insert(&mut self, name: &str, inputs: &[Literal]) -> Result<i64> {
        let info = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("no such function {name}"))?;
        if inputs.len() != info.inputs_len {
            return Err(anyhow!(
                "wrong input arity for `{name}`: got {}, expected {}",
                inputs.len(),
                info.inputs_len
            ));
        }
        if !info.eq_sort_ctor {
            return Err(anyhow!(
                "`{name}` is not registered as an EqSort constructor"
            ));
        }
        // Allocate a fresh ID. The raw constructor table is never
        // queried (term encoding routes all reads through
        // `@<name>View`) so we skip the INSERT and just take the next
        // sequence value. The caller's subsequent `(set @<name>View
        // args fresh_id) ()` writes the canonical row.
        let _ = inputs; // arity validated above; values flow through the view.
        let id: i64 = self
            .conn
            .query_row("SELECT nextval('__egglog_eqsort_seq')", [], |r| r.get(0))?;
        Ok(id)
    }

    /// Register a relation (no output column). Inserts use
    /// `ON CONFLICT DO NOTHING` — duplicates are silently dropped.
    pub fn add_relation(&mut self, name: &str, inputs: &[ColumnTy]) -> Result<()> {
        self.add_relation_with_pname(name, inputs, None)
    }

    /// Like `add_relation` but tags the relation with a pname for
    /// inline-congruence. Used by the DuckDB frontend for the
    /// `@_X_view` relations emitted by term encoding: the view is a
    /// `(inputs…, id)` relation that stores canonical e-nodes, and
    /// same-input/different-id rows must be unified through the
    /// associated pname. When `pname` is `Some(_)` and the runner is
    /// in `--duck-native-uf` mode, the equivalent of
    /// `@congruence_rule<N>` runs as an end-of-iter SQL.
    pub fn add_relation_with_pname(
        &mut self,
        name: &str,
        inputs: &[ColumnTy],
        pname: Option<&str>,
    ) -> Result<()> {
        let pname_resolved = pname.and_then(|p| {
            if self.functions.contains_key(p) {
                Some(p.to_string())
            } else {
                None
            }
        });
        self.declare(
            name,
            FunctionInfo {
                cols: inputs.to_vec(),
                inputs_len: inputs.len(),
                merge: None,
                eq_sort_ctor: false,
                native_uf_udf: None,
                eq_sort_pname: pname_resolved,
                identity_on_miss: false,
                merge_tree: None,
                native_congruence_uf: None,
            },
        )
    }

    /// Register a function with an output column and merge mode.
    /// PRIMARY KEY covers only the input columns; the output column
    /// is updated or kept on conflict according to `merge`.
    pub fn add_function(
        &mut self,
        name: &str,
        inputs: &[ColumnTy],
        output: ColumnTy,
        merge: MergeMode,
    ) -> Result<()> {
        let mut cols = inputs.to_vec();
        cols.push(output);
        self.declare(
            name,
            FunctionInfo {
                cols,
                inputs_len: inputs.len(),
                merge: Some(merge),
                eq_sort_ctor: false,
                native_uf_udf: None,
                eq_sort_pname: None,
                identity_on_miss: false,
                merge_tree: None,
                native_congruence_uf: None,
            },
        )
    }

    /// Register a TERM-BUILD custom `:merge` view function (`--native-merge`):
    /// an FD view `(children) -> eclass` whose merge body builds e-nodes. Unlike
    /// [`Self::add_function`], its PRIMARY KEY covers ALL columns (children +
    /// eclass), so two conflicting `(set (@FView key) eclass)` writes COEXIST
    /// rather than collapsing — [`Self::emit_term_build_merges`] reads the pair
    /// via a self-join at the iteration boundary, runs the retained `tree` as
    /// set-based SQL, and writes the merged eclass back. Reads still use the
    /// function-output form `(= e (@FView key))` (the all-cols PK does not change
    /// the read SQL; conflicts are resolved before the next iteration reads).
    /// `tree` is the lowered `MergeFn` (top-level `IfEq`/`Seq`).
    pub fn add_term_build_view_function(
        &mut self,
        name: &str,
        inputs: &[ColumnTy],
        output: ColumnTy,
        tree: egglog_backend_trait::MergeFn,
    ) -> Result<()> {
        let mut cols = inputs.to_vec();
        cols.push(output);
        self.declare(
            name,
            FunctionInfo {
                cols,
                inputs_len: inputs.len(),
                // `Old` => `ON CONFLICT DO NOTHING`; combined with the all-cols PK
                // (set below by `merge_tree.is_some()` in `declare`/`conflict_clause`)
                // this keeps every DISTINCT `(key, eclass)` row while de-duping
                // identical re-inserts.
                merge: Some(MergeMode::Old),
                eq_sort_ctor: false,
                native_uf_udf: None,
                eq_sort_pname: None,
                identity_on_miss: false,
                merge_tree: Some(tree),
                native_congruence_uf: None,
            },
        )
    }

    /// Register an EqSort constructor — a table whose last column is
    /// an EqSort ID allocated by `allocate_and_insert` from a global
    /// sequence. The PRIMARY KEY covers ALL columns (including the
    /// ID), so calling the constructor with the same inputs but a
    /// fresh ID never conflicts. Multiple rows per input key are the
    /// expected, intentional state — congruence rules emitted by
    /// term encoding unify the resulting IDs in UF later.
    pub fn add_eq_sort_constructor(
        &mut self,
        name: &str,
        inputs: &[ColumnTy],
        pname: Option<&str>,
    ) -> Result<()> {
        let mut cols = inputs.to_vec();
        cols.push(ColumnTy::I64); // the EqSort ID column
        // The pname must already be declared so the runner can emit
        // the inline-congruence INSERT against a real table. If the
        // caller passes a name we haven't seen, we can't honor the
        // optimization safely — drop it back to None and let the
        // term encoding's `@congruence_rule*` handle congruence.
        let pname_resolved = pname.and_then(|p| {
            if self.functions.contains_key(p) {
                Some(p.to_string())
            } else {
                None
            }
        });
        self.declare(
            name,
            FunctionInfo {
                cols,
                inputs_len: inputs.len(),
                merge: None,
                eq_sort_ctor: true,
                native_uf_udf: None,
                eq_sort_pname: pname_resolved,
                identity_on_miss: false,
                merge_tree: None,
                native_congruence_uf: None,
            },
        )?;
        Ok(())
    }

    fn declare(&mut self, name: &str, info: FunctionInfo) -> Result<()> {
        if self.functions.contains_key(name) {
            return Err(anyhow!("function {name} already registered"));
        }
        let col_decls: Vec<String> = info
            .cols
            .iter()
            .enumerate()
            .map(|(i, ty)| format!("c{i} {} NOT NULL", ty.sql()))
            .collect();
        // PK width:
        // - relations and eq-sort constructors: cover ALL columns
        //   (eq-sort ctors expect duplicate input keys with distinct
        //   IDs; relations have no output to exclude).
        // - TERM-BUILD views (`merge_tree.is_some()`): cover ALL columns too, so
        //   two conflicting `(set (@FView key) eclass)` writes COEXIST for the
        //   self-join in `emit_term_build_merges` (resolved at the iter boundary).
        // - other functions with merge mode: cover input columns only, so
        //   ON CONFLICT can update the output.
        // (NATIVE-CONGRUENCE views need no special case here: they are registered as
        // BASELINE all-columns Unit relations — `merge: None` — which already cover
        // all columns; `register_native_merge_view` tags them with
        // `native_congruence_uf` AFTER `declare`.)
        let pk_width = match (&info.merge, info.eq_sort_ctor, info.merge_tree.is_some()) {
            (Some(_), false, false) => info.inputs_len,
            _ => info.cols.len(),
        };
        let pk: Vec<String> = (0..pk_width).map(|i| format!("c{i}")).collect();
        let pk_clause = if pk.is_empty() {
            String::new()
        } else {
            format!(", PRIMARY KEY ({})", pk.join(", "))
        };
        // For 0-column tables (nullary relations / nullary
        // constructor helpers from term encoding), DuckDB rejects
        // an empty leading list — skip the leading comma.
        let col_list = if col_decls.is_empty() {
            "ts BIGINT NOT NULL".to_string()
        } else {
            format!("{}, ts BIGINT NOT NULL", col_decls.join(", "))
        };
        let sql = format!("CREATE TABLE {} ({col_list}{pk_clause})", q(name));
        self.debug_sql("create", &sql);
        self.conn.execute(&sql, [])?;
        // We tried two flavors of auxiliary indexes here and both
        // were a net loss on math-microbenchmark:
        //  - Secondary B-tree indexes on each non-leading input
        //    column (intended to accelerate rebuild variants that
        //    join `view.c_i = uf.c0` for i > 0). DuckDB's planner
        //    correctly chose hash joins over index seeks (the NEW
        //    side is small, the build cost is amortized), so the
        //    indexes went unused while costing per-insert
        //    maintenance. mm-microbenchmark slowed by ~26%.
        //  - Index on the `ts` column (intended to speed up the
        //    seminaive `ts >= ?1 AND ts < ?2` range filter). Lost
        //    to DuckDB's built-in zone maps for monotonically-
        //    inserted ts; insert maintenance dominated. Slowed by
        //    ~14%.
        // Insert-heavy analytical workloads on DuckDB don't want
        // OLTP-style auxiliary indexes — the columnar storage and
        // zone maps already cover the access patterns we use.
        // Native-UF registration. We attach a UfTable + scalar UDF
        // when the function has `:merge (ordering-min old new)` —
        // term encoding emits exactly one such function per eq-sort
        // (the function-form UF, `@_u_f___<sort>f`). The SQL table
        // is still created above; writes go through it so the
        // existing rulesets see them. The UDF is for fast reads.
        let mut info = info;
        if self.native_uf_enabled && info.merge == Some(MergeMode::Min) {
            // pname (raw UF) is what term encoding writes raw union
            // assertions into. It's declared *before* the function-
            // form table and follows the `@UF_<sort>` / `@UF_<sort>f`
            // naming convention: pname is the function name with a
            // trailing `f` stripped. Resolve and check up front so
            // we don't half-register if the convention is broken.
            let pname = name.strip_suffix('f').ok_or_else(|| {
                anyhow!("native UF {name}: name doesn't end in `f`; can't derive pname")
            })?;
            if !self.functions.contains_key(pname) {
                return Err(anyhow!(
                    "native UF {name}: expected pname `{pname}` to be declared first"
                ));
            }
            let uf = Arc::new(Mutex::new(UfTable::new()));
            let udf_name = format!("duck_uf_{}_find", sanitize_for_udf(name));
            self.conn
                .register_scalar_function_with_state::<UfFindScalar>(&udf_name, &uf)
                .map_err(|e| anyhow!("failed to register UF UDF {udf_name}: {e}"))?;
            self.native_ufs.insert(name.to_string(), uf);
            self.native_uf_pname
                .insert(name.to_string(), pname.to_string());
            info.native_uf_udf = Some(udf_name);
        }
        self.functions.insert(name.to_string(), info);
        Ok(())
    }

    /// Compile and store a rule. Compilation produces one SQL
    /// statement per (variant × action).
    pub fn add_rule(&mut self, rule: Rule) -> Result<()> {
        if std::env::var("DUCK_DUMP_RULES").is_ok() {
            eprintln!(
                "[duck/add_rule] name={} ruleset={}",
                rule.name, rule.ruleset
            );
            for atom in &rule.body {
                eprintln!("  body: {atom:?}");
            }
            for action in &rule.actions {
                eprintln!("  action: {action:?}");
            }
        }
        // Under `--duck-native-uf` the UF-maintenance rulesets are
        // skipped at every iter (the native UF maintains the same
        // invariants in-memory). Compile-time, some of their rules
        // use primitives the bridge doesn't yet support (notably
        // `pair` / `pair-first` / `pair-second` that proof-mode's
        // UF function index emits). Skipping the compile for those
        // rules outright avoids a hard error on programs that would
        // never execute the rules anyway.
        if self.native_uf_enabled && is_uf_maintenance_ruleset(&rule.ruleset) {
            return Ok(());
        }
        // `--native-uf --duckdb`: DROP the `@uf_change_drain_rule*` rules — they
        // delete from the always-empty `@UFChange_S` onchange relation (DuckDB
        // runs no leader-change callback), so they have nothing to do. The duck
        // trait surface never sees the egglog ruleset, so recognize by NAME.
        if self.native_uf_enabled && is_uf_change_drain_rule_name(&rule.name) {
            return Ok(());
        }
        // `--native-uf --duckdb` (PR #782 encoding): rewrite the rule into its
        // SQL host-pass form — canon-prim calls become find-UDF calls, and the
        // rebuild rule's always-empty `@UFChange_S` join is stripped to a view
        // scan. No-op for the relational `--duck-native-uf` path (no canon
        // prims / onchange relations registered).
        let rule = if self.native_uf_enabled {
            self.rewrite_native_uf_rule(rule)
        } else {
            rule
        };
        // Gate tables for the native-UF rebuild host-pass: the distinct `@UF_Sf`
        // union tables whose find UDF the (rewritten) rule references. The
        // rebuild rule needs a gated full-scan variant per such table so stale
        // OLD view rows are re-canonicalized when a union lands (the seminaive
        // variant only catches newly-inserted rows). Only for rebuild rules;
        // empty otherwise (so non-rebuild rules and the relational path are
        // unaffected).
        let native_uf_gate_tables: Vec<String> =
            if self.native_uf_enabled && is_rebuild_rule_name(&rule.name) {
                self.native_uf_rule_gate_tables(&rule)
            } else {
                Vec::new()
            };
        // Relational δuf fast-rebuild engages only WITHOUT the native UF:
        // `--fast-rebuild --native-uf` routes to the adaptive native path
        // (`adaptive_rebuild()`), never this relational decomposition. (The
        // legacy `DUCK_DELTA_REBUILD` env folds into `self.fast_rebuild`; gating
        // on `!native_uf_enabled` here is also what prevents the broken
        // `DUCK_DELTA_REBUILD` + native-uf combination.)
        let delta_rebuild = self.fast_rebuild && !self.native_uf_enabled;
        // `--native-uf --fast-rebuild`: drop the rebuild host-pass's δview-focus
        // seminaive branch (process this iteration's NEW view rows through the
        // find UDF — empty under canon-at-creation but it costs to probe). The
        // gated `__UF_CHANGED__` `view ⋈ δuf` semijoin then does all the rebuild
        // work, mirroring the relational `+fastrb` δview-drop. WITHOUT
        // `--fast-rebuild` (plain `--native-uf` = +nuf), the δview branch is
        // KEPT, so +nuf (full) and +nuf+fastrb (dropped) are distinct, matching
        // plain vs +fastrb on the relational side. `compile_rule` further gates
        // this on `native_uf_gate_tables` being non-empty (rebuild rules only),
        // so user rules are unaffected.
        let native_uf_drop_delta_view = self.native_uf_enabled && self.fast_rebuild;
        let mut compiled = compile::compile_rule(
            &rule,
            &self.functions,
            self.proofs_enabled,
            &native_uf_gate_tables,
            delta_rebuild,
            native_uf_drop_delta_view,
        )?;
        // Body tables = distinct Atom::Func names. The watermark gate
        // reads this set to decide whether the rule has anything new
        // to look at on a given iteration.
        //
        // Native-UF substitution: when a body atom references a
        // function with a registered native UF (e.g.
        // `@_u_f___mathf`), the corresponding SQL table is no
        // longer written to — rule actions go through the union
        // UDF or land in the raw pname table. So its watermark
        // never advances, and gating on it would permanently
        // block the rule after its first run. We substitute pname
        // (the raw UF, where union assertions land) so the gate
        // tracks the table that actually receives writes. Caused
        // a silent miss before this fix: rebuild rules would skip
        // because the @UF_*f watermark stayed at 0 even as new
        // unions accumulated in the native UF.
        let mut bt: Vec<String> = Vec::new();
        for atom in &rule.body {
            if let Atom::Func { name, .. } = atom {
                let effective = if let Some(info) = self.functions.get(name) {
                    if info.native_uf_udf.is_some() {
                        // Map @UF_<sort>f -> @UF_<sort> (pname).
                        self.native_uf_pname
                            .get(name)
                            .cloned()
                            .unwrap_or_else(|| name.clone())
                    } else {
                        name.clone()
                    }
                } else {
                    name.clone()
                };
                if !bt.iter().any(|n| n == &effective) {
                    bt.push(effective);
                }
            }
        }
        // `--native-uf --duckdb` rebuild host-pass: the rewritten rebuild rule
        // has no `@UF_Sf` body atom (the `@UFChange_S` join was stripped), so
        // its body_tables would be just the view — and the rule-level watermark
        // gate would SKIP the whole rule (incl. the gated full-scan variant)
        // when only a union landed (which doesn't touch the view). Add the
        // union tables to body_tables so a union advances the gate and the
        // rebuild re-fires to re-canonicalize stale OLD view rows.
        for gate in &native_uf_gate_tables {
            if !bt.iter().any(|n| n == gate) {
                bt.push(gate.clone());
            }
        }
        compiled.body_tables = bt;
        self.last_run_at.insert(rule.name.clone(), 0);
        self.rule_to_ruleset
            .insert(rule.name.clone(), rule.ruleset.clone());
        self.rules.push(compiled);
        Ok(())
    }

    /// Seed an initial fact at `ts = 0`. With seminaive predicate
    /// `focused.ts >= last_run_at` and `last_run_at` starting at 0,
    /// the first iteration will see all seeded rows.
    pub fn insert(&mut self, name: &str, args: &[Literal]) -> Result<()> {
        let info = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("no such function {name}"))?;
        if args.len() != info.arity() {
            return Err(anyhow!(
                "wrong arity for {name}: got {}, expected {}",
                args.len(),
                info.arity()
            ));
        }
        // Inline literal values directly; `?N`-style binding through
        // `&[&dyn ToSql]` slices has been flaky in our context. All
        // values are i64/bool/f64/string literals with safe SQL
        // representations.
        let cols: Vec<String> = (0..args.len()).map(|i| format!("c{i}")).collect();
        let conflict = conflict_clause(info);
        let cols_prefix = prefix_with_comma(&cols);
        let arg_sqls: Vec<String> = args.iter().map(crate::compile::lit_sql_pub).collect();
        let arg_prefix = prefix_with_comma(&arg_sqls);
        let cur_ts = self.next_ts;
        let sql_unfiltered = format!(
            "INSERT INTO {} ({cols_prefix}ts) VALUES ({arg_prefix}{cur_ts}) {conflict}",
            q(name),
        );
        let sql = sql_unfiltered.trim_end();
        self.debug_sql("insert", sql);
        let n = self.conn.execute(sql, [])?;
        if n > 0 {
            self.bump_watermark(name, cur_ts);
        }
        Ok(())
    }

    /// Insert at `ts = 0` with arbitrary `Term` values (literals,
    /// primitive expressions, or subquery reads of other functions).
    /// Used for term-encoded top-level sets where the value column
    /// references another global table.
    pub fn insert_terms(&mut self, name: &str, args: &[Term]) -> Result<()> {
        let info = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("no such function {name}"))?;
        if args.len() != info.arity() {
            return Err(anyhow!(
                "wrong arity for {name}: got {}, expected {}",
                args.len(),
                info.arity()
            ));
        }
        let cols: Vec<String> = (0..args.len()).map(|i| format!("c{i}")).collect();
        let arg_sqls: Vec<String> = args
            .iter()
            .map(|t| compile::term_sql_no_binding(t, "<top-level>"))
            .collect::<Result<_>>()?;
        let conflict = conflict_clause(info);
        let cols_prefix = prefix_with_comma(&cols);
        let arg_prefix = prefix_with_comma(&arg_sqls);
        let cur_ts = self.next_ts;
        let sql = format!(
            "INSERT INTO {} ({cols_prefix}ts) SELECT {arg_prefix}{cur_ts} {conflict}",
            q(name),
        );
        self.debug_sql("insert_terms", &sql);
        let n = self.conn.execute(&sql, [])?;
        if n > 0 {
            self.bump_watermark(name, cur_ts);
        }
        Ok(())
    }

    /// Run exactly the rules at `indices` once. The indices reference
    /// positions in `self.rules`. Each rule advances its own
    /// `last_run_at`, and the global `next_ts` bumps once per call so
    /// the seminaive predicates compute against a consistent
    /// snapshot. Used by the trait surface's `run_rules(&[ids])` —
    /// the frontend gives us specific `RuleId`s and we translate them
    /// to indices before calling here.
    pub fn run_iteration_for_indices(&mut self, indices: &[usize]) -> Result<usize> {
        let synced_displaced = self.sync_native_ufs()?;
        self.rules_affected = self.rules_affected.wrapping_add(synced_displaced as u64);
        // Only consult the adaptive decision (and risk resetting the
        // accumulated changed set) on iterations that actually run a
        // gated rebuild variant. Other iterations never full-scan, so
        // touching the changed set here would drop a delta the next
        // rebuild iteration still needs.
        let has_gated = self.rules_have_gated(indices);
        let gate_mode = if has_gated {
            self.adaptive_gate_mode()?
        } else {
            GateMode::FullScan
        };
        self.next_ts += 1;
        let cur = self.next_ts;
        let mut total: usize = synced_displaced;
        let last_run_ats: HashMap<String, i64> = self.last_run_at.clone();
        let skip_uf_maintenance = self.native_uf_enabled;
        for &idx in indices {
            let Some(rule) = self.rules.get(idx) else {
                continue;
            };
            if skip_uf_maintenance && is_uf_maintenance_ruleset(&rule.ruleset) {
                continue;
            }
            let last = *last_run_ats.get(&rule.name).unwrap_or(&0);
            total += run_rule_variants(
                rule,
                last,
                cur,
                &self.conn,
                &mut self.time_mat_ns,
                &mut self.time_mat_act_ns,
                &mut self.time_act_ns,
                &mut self.rules_affected,
                &mut self.rule_perf_ns,
                &mut self.table_watermarks,
                &mut self.rules_skipped,
                gate_mode,
            )?;
            self.last_run_at.insert(rule.name.clone(), cur);
        }
        // TERM-BUILD custom `:merge` (`--native-merge`): resolve FD conflicts on
        // each touched term-build view by running the lowered tree as set-based
        // SQL (the frontend dropped the `@merge_rule` for these functions). This
        // is the frontend rule-driver path (the trait `run_rules` calls
        // `run_iteration_for_indices`), so the term-build pass must run here —
        // it is watermark-gated, so iterations that touched no term-build view
        // are ~free.
        if self.native_uf_enabled {
            total += self.emit_term_build_merges(cur)?;
            // Native CONSTRUCTOR-CONGRUENCE (`--native-merge` with
            // `supports_native_congruence_merge()` true): resolve FD conflicts on
            // each touched native-congruence view (`native_congruence_uf.is_some()`)
            // by emitting `(larger, smaller)` union edges into the view's relational
            // UF (`@UF_S`) and deleting the loser rows. This is the host pass that
            // replaces the dropped `@congruence_rule*`. Run AFTER term-build so the
            // constructor (`@<C>View`) rows a term-build merge minted this iteration
            // also get their FD conflicts congruence-collapsed in the same pass. It
            // is watermark-gated (iterations touching no native-congruence view are
            // ~free); the resulting `@UF_S` edges are drained by `sync_native_ufs`
            // and the views re-canonicalized by `@rebuild_rule*` next iteration.
            total += self.emit_native_congruence(cur)?;
        }
        // A full scan re-canonicalized every row, so the accumulated
        // delta is fully consumed — reset it. Only on gated
        // iterations (see above).
        if has_gated && gate_mode == GateMode::FullScan {
            self.adaptive_reset_changed();
        }
        Ok(total)
    }

    /// Run rules whose ruleset is in `allowed` once. Empty set means
    /// "run all rules". Returns total rows inserted.
    pub fn run_iteration_in_set(&mut self, allowed: &[&str]) -> Result<usize> {
        let allow_all = allowed.is_empty();
        // Pull every new union assertion from the SQL pname tables
        // into the native UFs (no-op when `--duck-native-uf` is off).
        // Must happen before any rule in this iteration so UDF reads
        // see fresh state. The return value is the count of IDs
        // displaced by this sync; we feed it into `rules_affected`
        // so the outer saturate (which uses that counter as its
        // "did anything change" signal in
        // `backend_duckdb::Saturate`) keeps iterating while there
        // are still unions to propagate — required when we skip
        // rulesets like @single_parent whose pname churn was the
        // only source of the signal before.
        let synced_displaced = self.sync_native_ufs()?;
        self.rules_affected = self.rules_affected.wrapping_add(synced_displaced as u64);
        // Adaptive decision applies only when a gated rebuild variant
        // will actually run this iteration (see `run_iteration_for_indices`).
        let skip_uf_maintenance = self.native_uf_enabled;
        let runs = |rule: &CompiledRule| -> bool {
            (allow_all || allowed.iter().any(|rs| rule.ruleset == *rs))
                && !(skip_uf_maintenance && is_uf_maintenance_ruleset(&rule.ruleset))
        };
        let has_gated = self
            .rules
            .iter()
            .any(|r| runs(r) && r.variants.iter().any(|v| v.gate_table.is_some()));
        let gate_mode = if has_gated {
            self.adaptive_gate_mode()?
        } else {
            GateMode::FullScan
        };
        // Build the iteration with the existing single-ruleset path
        // by using a closure on the rule list. Mirrors run_iteration_in
        // but checks set membership.
        self.next_ts += 1;
        let cur = self.next_ts;
        let mut total: usize = synced_displaced;
        let last_run_ats: HashMap<String, i64> = self.last_run_at.clone();
        for rule in &self.rules {
            if !runs(rule) {
                continue;
            }
            let last = *last_run_ats.get(&rule.name).unwrap_or(&0);
            total += run_rule_variants(
                rule,
                last,
                cur,
                &self.conn,
                &mut self.time_mat_ns,
                &mut self.time_mat_act_ns,
                &mut self.time_act_ns,
                &mut self.rules_affected,
                &mut self.rule_perf_ns,
                &mut self.table_watermarks,
                &mut self.rules_skipped,
                gate_mode,
            )?;
            self.last_run_at.insert(rule.name.clone(), cur);
        }
        if has_gated && gate_mode == GateMode::FullScan {
            self.adaptive_reset_changed();
        }
        Ok(total)
    }

    /// Run rules in `ruleset` once (or all rules when `ruleset` is
    /// `None`). Returns total rows inserted across rules and
    /// variants.
    pub fn run_iteration_in(&mut self, ruleset: Option<&str>) -> Result<usize> {
        let synced_displaced = self.sync_native_ufs()?;
        self.rules_affected = self.rules_affected.wrapping_add(synced_displaced as u64);
        let skip_uf_maintenance = self.native_uf_enabled;
        // Predicate mirroring the in-loop skip conditions, so the
        // adaptive decision/reset only engages on iterations that
        // actually run a gated rebuild variant.
        let runs = |rule: &CompiledRule| -> bool {
            if let Some(rs) = ruleset
                && rule.ruleset != rs
            {
                return false;
            }
            if skip_uf_maintenance && is_uf_maintenance_ruleset(&rule.ruleset) {
                return false;
            }
            if skip_uf_maintenance && is_congruence_rule_name(&rule.name) {
                return false;
            }
            true
        };
        let has_gated = self
            .rules
            .iter()
            .any(|r| runs(r) && r.variants.iter().any(|v| v.gate_table.is_some()));
        let gate_mode = if has_gated {
            self.adaptive_gate_mode()?
        } else {
            GateMode::FullScan
        };
        self.next_ts += 1;
        let cur = self.next_ts;
        let mut total: usize = synced_displaced;
        let last_run_ats: HashMap<String, i64> = self.last_run_at.clone();
        for rule in &self.rules {
            if !runs(rule) {
                continue;
            }
            let last = *last_run_ats.get(&rule.name).unwrap_or(&0);
            total += run_rule_variants(
                rule,
                last,
                cur,
                &self.conn,
                &mut self.time_mat_ns,
                &mut self.time_mat_act_ns,
                &mut self.time_act_ns,
                &mut self.rules_affected,
                &mut self.rule_perf_ns,
                &mut self.table_watermarks,
                &mut self.rules_skipped,
                gate_mode,
            )?;
            self.last_run_at.insert(rule.name.clone(), cur);
        }
        // After rules fire, run the inline-congruence pass over
        // every EqSort constructor table that received new rows
        // this iteration. The encoding's `@congruence_rule*` rules
        // are skipped above; this is where their work lands instead.
        //
        // Only fire on rulesets that can produce new view rows: the
        // user's iter (`ruleset = None`) and `@rebuilding` (where
        // the rebuild rule canonicalizes view rows). Other rulesets
        // (cleanup, single_parent, parent, uf_function_index) only
        // delete or push to pname, never insert into views, so the
        // emit would scan empty `v1` sets repeatedly.
        let should_emit_cong = self.native_uf_enabled
            && (ruleset.is_none() || ruleset.is_some_and(is_rebuilding_ruleset));
        // TERM-BUILD custom `:merge` (`--native-merge`): resolve FD conflicts on
        // each touched term-build view by running the lowered tree as set-based
        // SQL. Run BEFORE `emit_inline_congruence` so the constructor (`@<C>View`)
        // rows the term-build mints get congruence-collapsed in the SAME
        // iteration's inline-congruence pass (and across iterations via
        // `sync_native_ufs`). Same ruleset gate as inline-congruence (only
        // rulesets that can produce new view rows).
        if should_emit_cong {
            total += self.emit_term_build_merges(cur)?;
        }
        if should_emit_cong {
            total += self.emit_inline_congruence(cur)?;
        }
        if has_gated && gate_mode == GateMode::FullScan {
            self.adaptive_reset_changed();
        }
        let _ = synced_displaced;
        Ok(total)
    }

    /// Native TERM-BUILD custom `:merge` resolution (`--native-merge`), INLINE as
    /// set-based SQL. For each term-build view (`merge_tree.is_some()`) whose
    /// watermark advanced since the last pass:
    ///
    ///   1. CONFLICT BATCH: self-join the all-cols-keyed view on its key columns
    ///      to read every `(keys.., old_eclass, new_eclass)` FD conflict over the
    ///      seminaive window (`old`/`new` are two different eclasses written for
    ///      the same key). A deterministic orientation (`old = GREATEST eclass,
    ///      new = LEAST`) makes the batch order reproducible.
    ///   2. RUN the compiled tree (see [`term_build_merge`]): a sequence of bulk
    ///      mint/view INSERTs that build the merge body's constructors set-based
    ///      with hash-consing, then a write-back of the merged eclass.
    ///   3. WRITE-BACK: delete the conflicting `(key, old)` / `(key, new)` rows
    ///      and insert `(key, merged)` — restoring the view's FD before the next
    ///      iteration reads it.
    ///
    /// Returns the number of conflict rows resolved (a saturation signal). The
    /// minted `@<C>View` rows get congruence-collapsed by the following
    /// `emit_inline_congruence` / cross-iteration `sync_native_ufs`.
    fn emit_term_build_merges(&mut self, cur: i64) -> Result<usize> {
        // Snapshot eligible (view, n_keys) pairs without holding a borrow.
        let entries: Vec<(String, usize)> = self
            .functions
            .iter()
            .filter_map(|(name, info)| {
                info.merge_tree.as_ref()?;
                // Skip if no writes since our last pass (watermark gate).
                let wm = self.table_watermarks.get(name).copied().unwrap_or(0);
                let last_at = self.last_termbuild_at.get(name).copied().unwrap_or(0);
                if wm < last_at {
                    return None;
                }
                // inputs_len = number of key (children) columns; the eclass is the
                // single output at index inputs_len.
                Some((name.clone(), info.inputs_len))
            })
            .collect();

        let mut total: usize = 0;
        for (view, n_keys) in entries {
            let last_at = self.last_termbuild_at.get(&view).copied().unwrap_or(0);
            let out_col = n_keys; // eclass column index.

            // Compile the retained tree to (statements, merged_expr, guard).
            let tree = self.functions[&view]
                .merge_tree
                .clone()
                .expect("term-build view");
            let (compiled, guard) = term_build_merge::compile_term_build(self, &tree, n_keys)?;

            // (1) Build the conflict batch as a TEMP table. Self-join the view on
            // its key columns. The rule-encoded `@merge_rule` fires with
            // `(= (ordering-max old new) new)` — i.e. `old = min(old,new)`,
            // `new = max(old,new)`. Match that orientation so the minted terms
            // (e.g. `C1(old,new)`) are byte-identical: `old = LEAST eclass,
            // new = GREATEST` (`v1.c{out} < v2.c{out}`). Each unordered pair
            // appears once. The seminaive window: at least one side is new
            // (`ts >= last_at`); no upper bound (term-build writes land at `cur`,
            // mirroring inline-congruence's `wm`-gated side).
            let key_join: Vec<String> = (0..n_keys).map(|i| format!("v1.c{i} = v2.c{i}")).collect();
            let key_proj: Vec<String> = (0..n_keys).map(|i| format!("v1.c{i} AS c{i}")).collect();
            let batch_sql = format!(
                "CREATE OR REPLACE TEMP TABLE batch AS \
                 SELECT {keyproj}, \
                        v1.c{out_col} AS old, v2.c{out_col} AS new \
                 FROM {v} v1 JOIN {v} v2 \
                   ON {keyjoin} AND v1.c{out_col} < v2.c{out_col} \
                 WHERE v1.ts >= ?1 OR v2.ts >= ?1",
                keyproj = key_proj.join(", "),
                v = q(&view),
                keyjoin = key_join.join(" AND "),
            );
            // `exec_bound` string-substitutes `?1`->last_at, `?2`->cur (the
            // project-wide convention; duckdb-rs numbered params aren't used).
            exec_bound(&self.conn, &batch_sql, last_at, cur)?;
            let n_conflicts: i64 = self
                .conn
                .query_row("SELECT COUNT(*) FROM batch", [], |r| r.get(0))?;
            if n_conflicts == 0 {
                self.last_termbuild_at.insert(view.clone(), cur);
                continue;
            }

            // Apply the IfEq guard (filter out canon-equal pairs: those mint
            // nothing and keep `old`). Rewrite `batch` to the kept rows.
            if let Some(guard_pred) = &guard {
                let filtered = format!(
                    "CREATE OR REPLACE TEMP TABLE batch AS \
                     SELECT * FROM batch b WHERE {guard_pred}"
                );
                exec_bound(&self.conn, &filtered, last_at, cur)?;
            }

            // (2) Run the compiled mint/view statements (each over `batch b`).
            for stmt in &compiled.stmts {
                let n = exec_bound(&self.conn, stmt, last_at, cur)?;
                self.rules_affected = self.rules_affected.wrapping_add(n as u64);
            }
            // Bump the watermark of every table the merge minted into, so the
            // seminaive skip-gate of downstream rules (the `@<C>View` rebuild /
            // congruence rules and the user rewrites that read these views)
            // re-fires on the freshly-minted rows. Without this, those rules'
            // `run_rule_variants` gate sees a stale watermark and skips them, so
            // the minted constructors never get congruence-collapsed or rewritten.
            for t in &compiled.written_tables {
                self.bump_watermark(t, cur);
            }

            // (3) Write-back: delete the conflicting view rows and insert
            // `(keys.., merged)`. The merged eclass = `compiled.merged_expr`.
            let key_cols: Vec<String> = (0..n_keys).map(|i| format!("c{i}")).collect();
            let key_sel: Vec<String> = (0..n_keys).map(|i| format!("b.c{i}")).collect();
            // Delete both conflicting outputs for each batch key.
            let del_old = format!(
                "DELETE FROM {v} WHERE ({kc}, c{out_col}) IN (SELECT {ks}, b.old FROM batch b)",
                v = q(&view),
                kc = key_cols.join(", "),
                ks = key_sel.join(", "),
            );
            let del_new = format!(
                "DELETE FROM {v} WHERE ({kc}, c{out_col}) IN (SELECT {ks}, b.new FROM batch b)",
                v = q(&view),
                kc = key_cols.join(", "),
                ks = key_sel.join(", "),
            );
            let nd1 = exec_bound(&self.conn, &del_old, last_at, cur)?;
            let nd2 = exec_bound(&self.conn, &del_new, last_at, cur)?;
            // Insert the merged row.
            let ins = format!(
                "INSERT INTO {v} ({kc}, c{out_col}, ts) \
                 SELECT {ks}, {merged}, ?2 FROM batch b \
                 ON CONFLICT DO NOTHING",
                v = q(&view),
                kc = key_cols.join(", "),
                ks = key_sel.join(", "),
                merged = compiled.merged_expr,
            );
            let ni = exec_bound(&self.conn, &ins, last_at, cur)?;
            self.rules_affected = self.rules_affected.wrapping_add((nd1 + nd2 + ni) as u64);
            self.bump_watermark(&view, cur);
            total += n_conflicts as usize;
            self.last_termbuild_at.insert(view.clone(), cur);
        }
        Ok(total)
    }

    /// Native CONSTRUCTOR-CONGRUENCE resolution (`--native-merge` with
    /// `supports_native_congruence_merge()` true), INLINE as set-based SQL. This is
    /// the host pass that REPLACES the dropped `@congruence_rule*` self-join rule.
    ///
    /// The encoder declares each constructor's `@<C>View` in the BASELINE all-columns
    /// Unit-relation shape `(children..., eclass) -> Unit` (NOT the FD `(children) ->
    /// eclass` function shape the collapse-at-insert backends use — see
    /// `native_merge_views_coexist` / `uses_fd_native_merge_view`), so two
    /// `(set (@<C>View children eclass))` writes with the same children but different
    /// eclasses COEXIST as distinct rows (the relation's all-cols PK only de-dupes
    /// identical rows). The `@congruence_rule*` is dropped and the view -> UF
    /// association is recorded on `FunctionInfo::native_congruence_uf` by
    /// `register_native_merge_view`.
    ///
    /// For each native-congruence view (`native_congruence_uf.is_some()`) whose
    /// watermark advanced since the last pass:
    ///   1. SELF-JOIN the view on its children columns to read every `(children, e1,
    ///      e2)` FD conflict (same children, two eclasses) over the seminaive window
    ///      (`v1` is the delta `ts ∈ [last, cur)`, `v2` is any `ts < cur`) — the same
    ///      delta semantics as the dropped `@congruence_rule*` / `emit_inline_congruence`.
    ///   2. INSERT the union edge `(GREATEST(e1,e2), LEAST(e1,e2), cur)` into the
    ///      view's per-sort UF-backed function (`native_congruence_uf`, i.e.
    ///      `@UF_Sf`) — exactly the `(larger, smaller)` row the dropped
    ///      `@congruence_rule*` wrote (via `(set (@UF_Sf larger) smaller)`) — with
    ///      `ON CONFLICT DO NOTHING` to dedupe. `sync_native_ufs` then drains these
    ///      edges into the in-core UF on the next iteration, and the existing
    ///      `@rebuild_rule*` re-canonicalizes the views (its full-`(children, eclass)`
    ///      delete retracts the stale non-leader row — no eager delete needed here,
    ///      mirroring the rule encoding's lifecycle exactly so bounded `(run N)` is
    ///      bit-exact).
    ///
    /// Returns the number of conflict rows resolved (a saturation signal, like
    /// `emit_term_build_merges`). Reuses `emit_inline_congruence`'s self-join +
    /// `GREATEST`/`LEAST` edge SQL and `emit_term_build_merges`'s TEMP-table batch +
    /// watermark gate/bump structure.
    fn emit_native_congruence(&mut self, cur: i64) -> Result<usize> {
        // Snapshot eligible (view, uf_table, n_keys) triples without a borrow held.
        let entries: Vec<(String, String, usize)> = self
            .functions
            .iter()
            .filter_map(|(name, info)| {
                let uf_table = info.native_congruence_uf.clone()?;
                // The native-congruence view is the BASELINE all-columns Unit
                // relation `@<C>View (children..., eclass)`: every column is a "key"
                // column (`inputs_len == cols.len()`), the LAST being the eclass. So
                // `n_keys` (the children count) is `inputs_len - 1` and the eclass
                // sits at index `inputs_len - 1`. Need at least one child column to
                // form an FD conflict.
                if info.inputs_len < 2 {
                    return None;
                }
                let n_keys = info.inputs_len - 1;
                // Skip if no writes since our last pass (watermark gate).
                let wm = self.table_watermarks.get(name).copied().unwrap_or(0);
                let last_at = self.last_native_cong_at.get(name).copied().unwrap_or(0);
                if wm < last_at {
                    return None;
                }
                Some((name.clone(), uf_table, n_keys))
            })
            .collect();

        let mut total: usize = 0;
        for (view, uf_table, n_keys) in entries {
            let last_at = self.last_native_cong_at.get(&view).copied().unwrap_or(0);
            let out_col = n_keys; // eclass column index.

            // (1) Build the conflict batch as a TEMP table. Self-join the all-cols-
            // keyed view on its key (children) columns. Orientation `v1.c{out} <
            // v2.c{out}` keeps the winner = min eclass (matching union-by-min) and
            // names the loser = max eclass (`v2`); each unordered pair appears once.
            // Seminaive window mirroring the dropped `@congruence_rule*` (and
            // `emit_inline_congruence`): `v1` is the NEW-since-last-pass delta
            // (`ts >= ?1 AND ts < ?2`) and `v2` covers everything earlier-or-equal
            // (`ts < ?2`). The current iteration's freshly-inserted rows are the
            // delta side `v1`; `v2 < cur` covers ALL prior rows. The pair is oriented
            // by value (NOT by which side is `v1`): `loser = GREATEST(eclasses)`,
            // `winner = LEAST` — so a conflict is caught whether the smaller- or the
            // larger-eclass row is the new delta. `v1.c{out} <> v2.c{out}` filters
            // equal-eclass self-joins. A pair where BOTH sides are new appears twice
            // (v1/v2 swapped) but yields the same `(GREATEST, LEAST)` edge, deduped
            // by the UF table's `ON CONFLICT DO NOTHING`. This matches the rule
            // encoding's delta semantics exactly, so the same conflicts are found in
            // the same iterations and the bounded-`(run N)` fixpoint is identical.
            let key_join: Vec<String> = (0..n_keys).map(|i| format!("v1.c{i} = v2.c{i}")).collect();
            let key_proj: Vec<String> = (0..n_keys).map(|i| format!("v1.c{i} AS c{i}")).collect();
            let batch_sql = format!(
                "CREATE OR REPLACE TEMP TABLE native_cong_batch AS \
                 SELECT {keyproj}, \
                        GREATEST(v1.c{out_col}, v2.c{out_col}) AS loser, \
                        LEAST(v1.c{out_col}, v2.c{out_col}) AS winner \
                 FROM {v} v1 JOIN {v} v2 \
                   ON {keyjoin} AND v1.c{out_col} <> v2.c{out_col} \
                 WHERE v1.ts >= ?1 AND v1.ts < ?2 AND v2.ts < ?2",
                keyproj = key_proj.join(", "),
                v = q(&view),
                keyjoin = key_join.join(" AND "),
            );
            exec_bound(&self.conn, &batch_sql, last_at, cur)?;
            let n_conflicts: i64 =
                self.conn
                    .query_row("SELECT COUNT(*) FROM native_cong_batch", [], |r| r.get(0))?;
            if n_conflicts == 0 {
                self.last_native_cong_at.insert(view.clone(), cur);
                continue;
            }

            // (2) Insert the union edges into the per-sort UF-backed function
            // (`@UF_Sf`): `(GREATEST=loser, LEAST=winner, cur)` — the `(larger,
            // smaller)` row the dropped `@congruence_rule*` wrote (via
            // `(set (@UF_Sf larger) smaller)`). `ON CONFLICT DO NOTHING` dedupes
            // against its all-cols `(c0, c1)` PK; `sync_native_ufs` drains
            // `(c0, c1, ts)` into the in-core UF.
            let ins_edge = format!(
                "INSERT INTO {u} (c0, c1, ts) \
                 SELECT b.loser, b.winner, ?2 FROM native_cong_batch b \
                 ON CONFLICT DO NOTHING",
                u = q(&uf_table),
            );
            let ne = exec_bound(&self.conn, &ins_edge, last_at, cur)?;
            if ne > 0 {
                self.rules_affected = self.rules_affected.wrapping_add(ne as u64);
                self.bump_watermark(&uf_table, cur);
            }

            // NOTE: we do NOT delete the loser view row here. This pass is a faithful
            // replica of the dropped `@congruence_rule*`, which ONLY emits the union
            // edge — it leaves both conflicting `(children, e1)`/`(children, e2)`
            // rows in place and lets the `@rebuild_rule*` retract the stale
            // (non-leader) row once the leader change propagates (the same lifecycle
            // the native-UF rule encoding uses, where this view is all-cols keyed and
            // the rebuild's now-DELETE-before-INSERT order — see
            // `native_merge_views_coexist` — re-canonicalizes without self-wiping).
            // Eagerly deleting the loser here desynchronizes from that lifecycle and
            // under-merges within a bounded `(run N)`.
            total += n_conflicts as usize;
            self.last_native_cong_at.insert(view.clone(), cur);
        }
        Ok(total)
    }

    /// For each EqSort constructor with a registered pname whose
    /// table watermark advanced this iter, INSERT into the pname any
    /// same-input/different-id pair as a `(max, min, cur)` union
    /// assertion. Returns the total rows inserted across all tables.
    ///
    /// Semantics match the term encoding's `@congruence_rule<N>`
    /// scanning new view rows against the current view, with
    /// `ON CONFLICT DO NOTHING` to dedupe pairs that two
    /// `v1`/`v2` orderings would otherwise emit twice.
    fn emit_inline_congruence(&mut self, cur: i64) -> Result<usize> {
        let mut total: usize = 0;
        // Snapshot the eligible (view, pname, inputs_len) triples so
        // we don't hold a borrow on `self.functions` during execute.
        // A function is eligible iff it has a pname registered and
        // can hold same-input/different-id rows (i.e. it has an ID
        // column). For both `eq_sort_ctor` and view relations the ID
        // is the last column.
        let entries: Vec<(String, String, usize)> = self
            .functions
            .iter()
            .filter_map(|(name, info)| {
                let pname = info.eq_sort_pname.clone()?;
                if info.cols.len() < 2 {
                    return None;
                }
                // For eq_sort_ctor (term tables) `inputs_len` is the
                // input arity; the ID is at index `inputs_len`. For
                // view relations there is no separate output column —
                // all `cols` are part of the row; the ID sits at the
                // last position by convention (term encoding emits
                // views shaped as `(inputs…, id)`).
                let inputs_len = if info.eq_sort_ctor {
                    info.inputs_len
                } else {
                    info.cols.len() - 1
                };
                if inputs_len == 0 {
                    return None;
                }
                // Skip if no inserts since the last inline-cong scan.
                // `wm` is the max-ever `ts` of any insert into this
                // view; `last_at` is the upper end of the previous
                // scan window. If `wm < last_at` the scan range is
                // guaranteed empty. (Note: NOT `wm < cur` — the
                // view's last write might be from a few iters ago and
                // we'd still want to pick up pairs that involve it
                // until they've been scanned.)
                let wm = self.table_watermarks.get(name).copied().unwrap_or(0);
                let last_at = self.last_inline_cong_at.get(name).copied().unwrap_or(0);
                if wm < last_at {
                    return None;
                }
                Some((name.clone(), pname, inputs_len))
            })
            .collect();
        for (view, pname, inputs_len) in entries {
            let id_col = inputs_len; // ID is at index `inputs_len`.
            let join_preds: Vec<String> = (0..inputs_len)
                .map(|i| format!("v1.c{i} = v2.c{i}"))
                .collect();
            // Seminaive triangulation: `v1` is the "new since last
            // scan" side (`ts >= last_inline_cong_at`); `v2` covers
            // everything earlier-or-equal (`ts < cur`, since the
            // current iter's rows are accounted for via `v1`). The
            // pair (v1=old, v2=new) is the same pair as (v1=new,
            // v2=old) with the two halves swapped, so we don't need
            // to emit a second branch — `ON CONFLICT DO NOTHING` on
            // pname dedupes if v1 and v2 are both new.
            let last_at = self.last_inline_cong_at.get(&view).copied().unwrap_or(0);
            let sql = format!(
                "INSERT INTO {} (c0, c1, ts) \
                 SELECT GREATEST(v1.c{id_col}, v2.c{id_col}), \
                        LEAST(v1.c{id_col}, v2.c{id_col}), \
                        ?2 \
                 FROM {} v1, {} v2 \
                 WHERE v1.ts >= ?1 AND v1.ts < ?2 AND v2.ts < ?2 \
                   AND v1.c{id_col} <> v2.c{id_col} \
                   AND {} \
                 ON CONFLICT DO NOTHING",
                q(&pname),
                q(&view),
                q(&view),
                join_preds.join(" AND "),
            );
            let n = exec_bound(&self.conn, &sql, last_at, cur)?;
            if n > 0 {
                self.rules_affected = self.rules_affected.wrapping_add(n as u64);
                self.bump_watermark(&pname, cur);
                total += n;
            }
            self.last_inline_cong_at.insert(view, cur);
        }
        Ok(total)
    }

    /// Run all rules (any ruleset) once. Convenience over
    /// `run_iteration_in(None)`.
    pub fn run_iteration(&mut self) -> Result<usize> {
        self.run_iteration_in(None)
    }

    /// Run rules in `ruleset` until no iteration adds any rows.
    pub fn run_to_saturation_in(&mut self, ruleset: Option<&str>) -> Result<(usize, i64)> {
        let mut iters = 0;
        loop {
            iters += 1;
            if self.run_iteration_in(ruleset)? == 0 {
                return Ok((iters, self.next_ts));
            }
        }
    }

    /// Run all rules to saturation. Convenience over
    /// `run_to_saturation_in(None)`.
    pub fn run_to_saturation(&mut self) -> Result<(usize, i64)> {
        self.run_to_saturation_in(None)
    }

    /// Whether a row matching the given args exists. For functions
    /// with outputs, you may pass either input-only args (asks "is
    /// this key present?") or full args including output (asks "is
    /// this exact row present?").
    pub fn check_exists(&self, name: &str, args: &[Literal]) -> Result<bool> {
        let info = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("no such function {name}"))?;
        if args.len() != info.arity() && args.len() != info.inputs_len {
            return Err(anyhow!(
                "check_exists arity for {name}: got {}, want {} or {}",
                args.len(),
                info.inputs_len,
                info.arity(),
            ));
        }
        let where_parts: Vec<String> = (0..args.len())
            .map(|i| format!("c{i} = ?{}", i + 1))
            .collect();
        let where_clause = if where_parts.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", where_parts.join(" AND "))
        };
        let sql = format!("SELECT COUNT(*) FROM {}{where_clause}", q(name));
        let params: Vec<&dyn ToSql> = args.iter().map(|a| a as &dyn ToSql).collect();
        let n: i64 = self.conn.query_row(&sql, params.as_slice(), |r| r.get(0))?;
        Ok(n > 0)
    }

    /// Look up the output value of a function for given inputs.
    /// Returns `None` for missing rows; errors if called on a relation.
    pub fn lookup_i64(&self, name: &str, inputs: &[Literal]) -> Result<Option<i64>> {
        let info = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("no such function {name}"))?;
        if !info.has_output() {
            return Err(anyhow!("{name} is a relation; lookup_i64 not applicable"));
        }
        if inputs.len() != info.inputs_len {
            return Err(anyhow!(
                "lookup_i64 arity for {name}: got {}, expected {}",
                inputs.len(),
                info.inputs_len,
            ));
        }
        if info.cols[info.inputs_len] != ColumnTy::I64 {
            return Err(anyhow!("{name}'s output is not i64"));
        }
        let where_parts: Vec<String> = (0..inputs.len())
            .map(|i| format!("c{i} = ?{}", i + 1))
            .collect();
        let out_col = info.inputs_len;
        let where_clause = if where_parts.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", where_parts.join(" AND "))
        };
        let sql = format!("SELECT c{out_col} FROM {}{where_clause}", q(name),);
        let params: Vec<&dyn ToSql> = inputs.iter().map(|a| a as &dyn ToSql).collect();
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params.as_slice())?;
        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    /// The number of columns of a registered table, or `None` if
    /// the name isn't registered.
    pub fn function_arity(&self, name: &str) -> Option<usize> {
        self.functions.get(name).map(|f| f.cols.len())
    }

    /// Whether at least one row matches the given body atoms,
    /// interpreted as a conjunctive query. This is the same query
    /// machinery rules use, minus the seminaive focus predicate —
    /// the check passes iff the body would have any match.
    pub fn body_exists(&self, atoms: &[Atom]) -> Result<bool> {
        let sql = compile::compile_body_select(atoms, &self.functions)?;
        let n: i64 = self.conn.query_row(&sql, [], |r| r.get(0))?;
        Ok(n > 0)
    }

    /// Whether any row in `name` matches the given pre-built WHERE
    /// fragments. Each fragment is a `cN = literal` constraint;
    /// the caller must have already validated arity. Used for
    /// existential checks on relations.
    pub fn relation_exists_raw(&self, name: &str, where_parts: &[String]) -> Result<bool> {
        let where_clause = if where_parts.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", where_parts.join(" AND "))
        };
        let sql = format!("SELECT COUNT(*) FROM {}{}", q(name), where_clause);
        let n: i64 = self.conn.query_row(&sql, [], |r| r.get(0))?;
        Ok(n > 0)
    }

    pub fn count(&self, name: &str) -> Result<i64> {
        Ok(self
            .conn
            .query_row(&format!("SELECT COUNT(*) FROM {}", q(name)), [], |r| {
                r.get(0)
            })?)
    }
}

/// Build the `ON CONFLICT ...` clause for an INSERT into the given
/// table. Used by both seeding (`EGraph::insert`) and rule-action
/// codegen (`compile.rs`).
pub(crate) fn conflict_clause(info: &FunctionInfo) -> String {
    // `declare` emits an empty PK clause iff the PK width is 0.
    // Without a PK, DuckDB rejects `ON CONFLICT` entirely. For
    // these tables (nullary functions with output) we just emit
    // plain INSERTs; second-write semantics are then "duplicate
    // rows accumulate", which matches `:no-merge` if the user
    // never re-sets, and is wrong otherwise. Term encoding's use
    // of nullary `:no-merge` functions is for `let`-binding
    // globals, and those are set exactly once at declaration time,
    // so this is safe in practice.
    // TERM-BUILD views are all-cols keyed (see `declare`): their writes coexist
    // and are resolved by `emit_term_build_merges`, so the conflict clause is a
    // plain `DO NOTHING` (de-dupe identical re-inserts), NOT a `DO UPDATE` on the
    // output column. (NATIVE-CONGRUENCE views are BASELINE all-columns Unit
    // relations with `merge: None`, which already yield `DO NOTHING` via the
    // `None` arm below — no special case needed.)
    if info.merge_tree.is_some() {
        return "ON CONFLICT DO NOTHING".to_string();
    }
    let pk_width = match (&info.merge, info.eq_sort_ctor) {
        (Some(_), false) => info.inputs_len,
        _ => info.cols.len(),
    };
    if pk_width == 0 {
        return String::new();
    }
    if info.eq_sort_ctor {
        // EqSort constructor inserts come with a freshly-allocated
        // ID column, so collisions on the all-cols PK shouldn't
        // happen in practice. If they do (caller passed a literal
        // ID), drop quietly.
        return "ON CONFLICT DO NOTHING".to_string();
    }
    match &info.merge {
        None => "ON CONFLICT DO NOTHING".to_string(),
        Some(MergeMode::Old) => "ON CONFLICT DO NOTHING".to_string(),
        Some(MergeMode::New) => {
            let out_col = info.inputs_len;
            format!("ON CONFLICT DO UPDATE SET c{out_col} = EXCLUDED.c{out_col}, ts = EXCLUDED.ts")
        }
        Some(MergeMode::Min) => {
            let out_col = info.inputs_len;
            // Unprefixed `c{out_col}` on the RHS refers to the
            // existing target row; `EXCLUDED.c{out_col}` is the new
            // value being inserted.
            format!(
                "ON CONFLICT DO UPDATE SET c{out_col} = LEAST(c{out_col}, EXCLUDED.c{out_col}), ts = EXCLUDED.ts"
            )
        }
        Some(MergeMode::Fold(sql)) => {
            // Native VALUE-FOLD custom `:merge`: the fold expression `sql` was
            // built by `mergefn_to_sql` over `c{out}` (the existing row) and
            // `EXCLUDED.c{out}` (the incoming row), so it resolves the FD conflict
            // in-SQL directly.
            let out_col = info.inputs_len;
            format!("ON CONFLICT DO UPDATE SET c{out_col} = {sql}, ts = EXCLUDED.ts")
        }
    }
}
