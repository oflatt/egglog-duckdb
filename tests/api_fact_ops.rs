//! Tests for the typed fact-ops API on the [`Read`] / [`Write`]
//! traits, accessed outside a rule via [`EGraph::with_full_state`].
//!
//! `with_full_state` flushes pending writes only when the closure
//! returns, so a read in the same closure won't see a preceding
//! write — tests below split write and read into separate closures
//! to reflect that.

use egglog::prelude::*;
use egglog::{Error, RawValues};

fn make_eg_with_function() -> EGraph {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(function f (i64) i64 :no-merge)")
        .unwrap();
    eg
}

#[test]
fn test_set_then_lookup_function() -> Result<(), Error> {
    let mut eg = make_eg_with_function();
    eg.with_full_state(|mut fs| fs.set("f", (1_i64,), 42_i64))?;
    let got: Option<i64> = eg.with_full_state(|fs| fs.lookup::<_, i64>("f", 1_i64))?;
    assert_eq!(got, Some(42));
    Ok(())
}

#[test]
fn test_lookup_missing_returns_none() -> Result<(), Error> {
    let mut eg = make_eg_with_function();
    let got: Option<i64> = eg.with_full_state(|fs| fs.lookup::<_, i64>("f", 999_i64))?;
    assert_eq!(got, None);
    Ok(())
}

#[test]
fn test_contains_function() -> Result<(), Error> {
    let mut eg = make_eg_with_function();
    eg.with_full_state(|mut fs| fs.set("f", (1_i64,), 42_i64))?;
    let (has1, has999) = eg.with_full_state(|fs| -> Result<_, Error> {
        Ok((fs.contains("f", 1_i64)?, fs.contains("f", 999_i64)?))
    })?;
    assert!(has1);
    assert!(!has999);
    Ok(())
}

#[test]
fn test_remove_function() -> Result<(), Error> {
    let mut eg = make_eg_with_function();
    eg.with_full_state(|mut fs| fs.set("f", (1_i64,), 42_i64))?;
    assert!(eg.with_full_state(|fs| fs.contains("f", 1_i64))?);

    eg.with_full_state(|mut fs| fs.remove("f", 1_i64))?;
    assert!(!eg.with_full_state(|fs| fs.contains("f", 1_i64))?);

    // Removing again is a no-op.
    eg.with_full_state(|mut fs| fs.remove("f", 1_i64))?;
    assert!(!eg.with_full_state(|fs| fs.contains("f", 1_i64))?);
    Ok(())
}

#[test]
fn test_relation_add_node_and_contains() -> Result<(), Error> {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(relation R (i64 i64))")?;
    eg.with_full_state(|mut fs| -> Result<_, Error> {
        fs.add_node("R", (1_i64, 2_i64))?;
        Ok(())
    })?;
    let (a, b, c) = eg.with_full_state(|fs| -> Result<_, Error> {
        Ok((
            fs.contains("R", (1_i64, 2_i64))?,
            fs.contains("R", (1_i64, 3_i64))?,
            fs.contains("R", (2_i64, 1_i64))?,
        ))
    })?;
    assert!(a);
    assert!(!b);
    assert!(!c);
    Ok(())
}

#[test]
fn test_relation_remove() -> Result<(), Error> {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(relation R (i64 i64))")?;
    eg.with_full_state(|mut fs| -> Result<_, Error> {
        fs.add_node("R", (1_i64, 2_i64))?;
        fs.add_node("R", (3_i64, 4_i64))?;
        Ok(())
    })?;
    let (a, b) = eg.with_full_state(|fs| -> Result<_, Error> {
        Ok((
            fs.contains("R", (1_i64, 2_i64))?,
            fs.contains("R", (3_i64, 4_i64))?,
        ))
    })?;
    assert!(a);
    assert!(b);

    eg.with_full_state(|mut fs| fs.remove("R", (1_i64, 2_i64)))?;
    let (a, b) = eg.with_full_state(|fs| -> Result<_, Error> {
        Ok((
            fs.contains("R", (1_i64, 2_i64))?,
            fs.contains("R", (3_i64, 4_i64))?,
        ))
    })?;
    assert!(!a);
    assert!(b);
    Ok(())
}

#[test]
fn test_constructor_add_node_returns_id() -> Result<(), Error> {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")?;

    // Zero-arg constructor uses RawValues(vec![]) — `()` would be a Unit column.
    // Calling add_node again with the same inputs returns the same id.
    let (cons, cons2, nil) = eg.with_full_state(|mut fs| -> Result<_, Error> {
        let nil = fs.add_node("Nil", RawValues(vec![]))?;
        let cons = fs.add_node("Cons", (1_i64, nil.clone()))?;
        let cons2 = fs.add_node("Cons", (1_i64, nil.clone()))?;
        Ok((cons, cons2, nil))
    })?;
    assert_eq!(cons.value(), cons2.value());
    assert_eq!(cons.sort(), "List");

    let (nil_present, cons_present) = eg.with_full_state(|fs| -> Result<_, Error> {
        Ok((
            fs.contains("Nil", RawValues(vec![]))?,
            fs.contains("Cons", (1_i64, nil))?,
        ))
    })?;
    assert!(nil_present);
    assert!(cons_present);
    Ok(())
}

#[test]
fn test_eclass_of_constructor() -> Result<(), Error> {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")?;
    let (cons, nil) = eg.with_full_state(|mut fs| -> Result<_, Error> {
        let nil = fs.add_node("Nil", RawValues(vec![]))?;
        let cons = fs.add_node("Cons", (1_i64, nil.clone()))?;
        Ok((cons, nil))
    })?;
    let (existing, absent) = eg.with_full_state(|fs| -> Result<_, Error> {
        Ok((
            fs.eclass_of("Cons", (1_i64, nil.clone()))?,
            fs.eclass_of("Cons", (99_i64, nil))?,
        ))
    })?;
    assert_eq!(existing.map(|id| id.value()), Some(cons.value()));
    assert!(absent.is_none());
    Ok(())
}

#[test]
fn test_lookup_on_constructor_errors() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")
        .unwrap();
    let result = eg.with_full_state(|fs| fs.lookup::<_, i64>("Cons", (1_i64, 0_i64)));
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Cons") && err.contains("constructor"), "got: {err}");
}

#[test]
fn test_eclass_of_on_function_errors() {
    let mut eg = make_eg_with_function();
    let result = eg.with_full_state(|fs| fs.eclass_of("f", 1_i64));
    let err = result.unwrap_err().to_string();
    assert!(err.contains("`f`") && err.contains("function"), "got: {err}");
}

#[test]
fn test_set_constructor_errors() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")
        .unwrap();
    let result = eg.with_full_state(|mut fs| fs.set("Nil", RawValues(vec![]), 0_i64));
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Nil") && err.contains("constructor"), "got: {err}");
}

#[test]
fn test_add_node_function_errors() {
    let mut eg = make_eg_with_function();
    let result = eg.with_full_state(|mut fs| fs.add_node("f", 1_i64));
    let err = result.unwrap_err().to_string();
    assert!(err.contains("`f`") && err.contains("function"), "got: {err}");
}

#[test]
fn test_wrong_column_sort_errors() {
    // Sending a `String` where the table expects an `i64`.
    let mut eg = make_eg_with_function();
    let result = eg.with_full_state(|mut fs| fs.set("f", ("hello".to_string(),), 42_i64));
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("expected sort `i64`") && err.contains("got value of sort `String`"),
        "got: {err}"
    );
}

#[test]
fn test_wrong_output_sort_errors() {
    // Sending a String value where the table's output is i64.
    let mut eg = make_eg_with_function();
    let result = eg.with_full_state(|mut fs| fs.set("f", (1_i64,), "wrong".to_string()));
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("output") && err.contains("expected sort `i64`"),
        "got: {err}"
    );
}

#[test]
fn test_wrong_arity_errors() {
    // Sending 2 args where the table expects 1.
    let mut eg = make_eg_with_function();
    let result = eg.with_full_state(|mut fs| fs.set("f", (1_i64, 2_i64), 42_i64));
    let err = result.unwrap_err().to_string();
    assert!(err.contains("expected 1 input"), "got: {err}");
}

#[test]
fn test_union_sort_mismatch_errors() -> Result<(), Error> {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(
        None,
        "(datatype Math (Num i64))
         (datatype List (Nil))",
    )?;
    let (math, list) = eg.with_full_state(|mut fs| -> Result<_, Error> {
        let math = fs.add_node("Num", 7_i64)?;
        let list = fs.add_node("Nil", RawValues(vec![]))?;
        Ok((math, list))
    })?;
    // Union of Math + List should error.
    let result = eg.with_full_state(|mut fs| fs.union(math, list));
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Math") && err.contains("List") && err.contains("union"),
        "got: {err}"
    );
    Ok(())
}

#[test]
fn test_set_replaces_function_value() -> Result<(), Error> {
    let mut eg = make_eg_with_function();
    eg.with_full_state(|mut fs| fs.set("f", (5_i64,), 50_i64))?;
    let got: Option<i64> = eg.with_full_state(|fs| fs.lookup::<_, i64>("f", 5_i64))?;
    assert_eq!(got, Some(50));
    Ok(())
}

#[test]
fn test_set_unknown_table_errors() {
    let mut eg = EGraph::default();
    let result = eg.with_full_state(|mut fs| fs.set("nope", (1_i64,), 2_i64));
    assert!(result.is_err());
}

#[test]
fn test_lookup_unknown_table_errors() {
    let mut eg = EGraph::default();
    let result = eg.with_full_state(|fs| fs.lookup::<_, i64>("nope", 1_i64));
    assert!(result.is_err());
}

#[test]
fn test_higher_arity_function() -> Result<(), Error> {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(function g (i64 i64 i64) i64 :no-merge)")?;
    eg.with_full_state(|mut fs| fs.set("g", (1_i64, 2_i64, 3_i64), 7_i64))?;
    let (v, has) = eg.with_full_state(|fs| -> Result<_, Error> {
        let v: Option<i64> = fs.lookup::<_, i64>("g", (1_i64, 2_i64, 3_i64))?;
        let has = fs.contains("g", (1_i64, 2_i64, 3_i64))?;
        Ok((v, has))
    })?;
    assert_eq!(v, Some(7));
    assert!(has);
    Ok(())
}

#[test]
fn test_string_inputs() -> Result<(), Error> {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(function name-length (String) i64 :no-merge)")?;
    eg.with_full_state(|mut fs| fs.set("name-length", ("hello".to_string(),), 5_i64))?;
    let got: Option<i64> =
        eg.with_full_state(|fs| fs.lookup::<_, i64>("name-length", "hello".to_string()))?;
    assert_eq!(got, Some(5));
    Ok(())
}

// ---------------------------------------------------------------------
// Dynamic type-error tests for the EGraph::intern / extract path.
// ---------------------------------------------------------------------

#[test]
fn test_intern_roundtrip() -> Result<(), Error> {
    let eg = EGraph::default();
    let id = eg.intern::<i64>(42)?;
    assert_eq!(id.sort(), "i64");
    assert_eq!(eg.extract::<i64>(id)?, 42);
    Ok(())
}

#[test]
fn test_extract_wrong_sort_errors() -> Result<(), Error> {
    let eg = EGraph::default();
    // Intern as String, try to extract as i64 — sort mismatch should error.
    let s_id = eg.intern::<egglog::sort::S>(egglog::sort::S::new("hello".to_string()))?;
    let result = eg.extract::<i64>(s_id);
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("expected sort `i64`") && err.contains("got value of sort `String`"),
        "got: {err}"
    );
    Ok(())
}

#[test]
fn test_intern_unknown_base_sort_errors() {
    // Make a fresh BaseValue Rust type with no egglog sort registered.
    #[derive(Clone, Hash, PartialEq, Eq, Debug)]
    struct MyCustomBase(i64);
    impl egglog::BaseValue for MyCustomBase {}

    let eg = EGraph::default();
    let result = eg.intern::<MyCustomBase>(MyCustomBase(7));
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("MyCustomBase") && err.contains("no egglog sort is registered"),
        "got: {err}"
    );
}

#[test]
fn test_extract_unknown_base_sort_errors() -> Result<(), Error> {
    #[derive(Clone, Hash, PartialEq, Eq, Debug)]
    struct AnotherCustom(bool);
    impl egglog::BaseValue for AnotherCustom {}

    let eg = EGraph::default();
    // We have a valid Id from an i64 intern, but we'll try to extract
    // as a type that has no registered sort.
    let id = eg.intern::<i64>(0)?;
    let result = eg.extract::<AnotherCustom>(id);
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("AnotherCustom") && err.contains("no egglog sort is registered"),
        "got: {err}"
    );
    Ok(())
}
