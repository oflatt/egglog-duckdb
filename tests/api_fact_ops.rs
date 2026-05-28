//! Tests for the typed fact-ops API on the [`Read`] / [`Write`]
//! traits, accessed outside a rule via [`EGraph::with_full_state`].
//!
//! `with_full_state` flushes pending writes only when the closure
//! returns, so a read in the same closure won't see a preceding
//! write — tests below split write and read into separate closures
//! to reflect that.

use egglog::prelude::*;
use egglog::RawValues;

fn make_eg_with_function() -> EGraph {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(function f (i64) i64 :no-merge)")
        .unwrap();
    eg
}

#[test]
fn test_set_then_lookup_function() {
    let mut eg = make_eg_with_function();
    eg.with_full_state(|mut fs| fs.set("f", (1_i64,), 42_i64));
    let got: Option<i64> = eg.with_full_state(|fs| fs.lookup::<_, i64>("f", 1_i64));
    assert_eq!(got, Some(42));
}

#[test]
fn test_lookup_missing_returns_none() {
    let mut eg = make_eg_with_function();
    let got: Option<i64> = eg.with_full_state(|fs| fs.lookup::<_, i64>("f", 999_i64));
    assert_eq!(got, None);
}

#[test]
fn test_contains_function() {
    let mut eg = make_eg_with_function();
    eg.with_full_state(|mut fs| fs.set("f", (1_i64,), 42_i64));
    let (has1, has999) =
        eg.with_full_state(|fs| (fs.contains("f", 1_i64), fs.contains("f", 999_i64)));
    assert!(has1);
    assert!(!has999);
}

#[test]
fn test_remove_function() {
    let mut eg = make_eg_with_function();
    eg.with_full_state(|mut fs| fs.set("f", (1_i64,), 42_i64));
    assert!(eg.with_full_state(|fs| fs.contains("f", 1_i64)));

    eg.with_full_state(|mut fs| fs.remove("f", 1_i64));
    assert!(!eg.with_full_state(|fs| fs.contains("f", 1_i64)));

    // Removing again is a no-op.
    eg.with_full_state(|mut fs| fs.remove("f", 1_i64));
    assert!(!eg.with_full_state(|fs| fs.contains("f", 1_i64)));
}

#[test]
fn test_relation_add_node_and_contains() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(relation R (i64 i64))")
        .unwrap();
    eg.with_full_state(|mut fs| {
        fs.add_node("R", (1_i64, 2_i64));
    });
    let (a, b, c) = eg.with_full_state(|fs| {
        (
            fs.contains("R", (1_i64, 2_i64)),
            fs.contains("R", (1_i64, 3_i64)),
            fs.contains("R", (2_i64, 1_i64)),
        )
    });
    assert!(a);
    assert!(!b);
    assert!(!c);
}

#[test]
fn test_relation_remove() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(relation R (i64 i64))")
        .unwrap();
    eg.with_full_state(|mut fs| {
        fs.add_node("R", (1_i64, 2_i64));
        fs.add_node("R", (3_i64, 4_i64));
    });
    let (a, b) = eg.with_full_state(|fs| {
        (
            fs.contains("R", (1_i64, 2_i64)),
            fs.contains("R", (3_i64, 4_i64)),
        )
    });
    assert!(a);
    assert!(b);

    eg.with_full_state(|mut fs| fs.remove("R", (1_i64, 2_i64)));
    let (a, b) = eg.with_full_state(|fs| {
        (
            fs.contains("R", (1_i64, 2_i64)),
            fs.contains("R", (3_i64, 4_i64)),
        )
    });
    assert!(!a);
    assert!(b);
}

#[test]
fn test_constructor_add_node_returns_eclass() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")
        .unwrap();

    // Zero-arg constructor uses RawValues(vec![]) — `()` would be a Unit column.
    // Calling add_node again with the same inputs returns the same eclass.
    let (cons, cons2, nil) = eg.with_full_state(|mut fs| {
        let nil = fs.add_node("Nil", RawValues(vec![])).unwrap();
        let cons = fs.add_node("Cons", (1_i64, nil)).unwrap();
        let cons2 = fs.add_node("Cons", (1_i64, nil)).unwrap();
        (cons, cons2, nil)
    });
    assert_eq!(cons, cons2);

    let (nil_present, cons_present) = eg.with_full_state(|fs| {
        (
            fs.contains("Nil", RawValues(vec![])),
            fs.contains("Cons", (1_i64, nil)),
        )
    });
    assert!(nil_present);
    assert!(cons_present);
}

#[test]
fn test_eclass_of_constructor() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")
        .unwrap();
    let (cons, nil) = eg.with_full_state(|mut fs| {
        let nil = fs.add_node("Nil", RawValues(vec![])).unwrap();
        let cons = fs.add_node("Cons", (1_i64, nil)).unwrap();
        (cons, nil)
    });
    let (existing, absent) = eg.with_full_state(|fs| {
        (
            fs.eclass_of("Cons", (1_i64, nil)),
            fs.eclass_of("Cons", (99_i64, nil)),
        )
    });
    assert_eq!(existing, Some(cons));
    assert!(absent.is_none());
}

#[test]
#[should_panic(expected = "Read::lookup called on constructor")]
fn test_lookup_on_constructor_panics() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")
        .unwrap();
    eg.with_full_state(|fs| fs.lookup::<_, i64>("Cons", (1_i64, 0_i64)));
}

#[test]
#[should_panic(expected = "Read::eclass_of called on function")]
fn test_eclass_of_on_function_panics() {
    let mut eg = make_eg_with_function();
    eg.with_full_state(|fs| fs.eclass_of("f", 1_i64));
}

#[test]
#[should_panic(expected = "Write::set called on constructor")]
fn test_set_constructor_panics() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(datatype List (Cons i64 List) (Nil))")
        .unwrap();
    eg.with_full_state(|mut fs| fs.set("Nil", RawValues(vec![]), 0_i64));
}

#[test]
#[should_panic(expected = "Write::add_node called on function")]
fn test_add_node_function_panics() {
    let mut eg = make_eg_with_function();
    eg.with_full_state(|mut fs| {
        fs.add_node("f", 1_i64);
    });
}

#[test]
fn test_set_replaces_function_value() {
    let mut eg = make_eg_with_function();
    eg.with_full_state(|mut fs| fs.set("f", (5_i64,), 50_i64));
    let got: Option<i64> = eg.with_full_state(|fs| fs.lookup::<_, i64>("f", 5_i64));
    assert_eq!(got, Some(50));
}

#[test]
#[should_panic(expected = "missing table action")]
fn test_set_unknown_table_panics() {
    let mut eg = EGraph::default();
    eg.with_full_state(|mut fs| fs.set("nope", (1_i64,), 2_i64));
}

#[test]
#[should_panic(expected = "missing table action")]
fn test_lookup_unknown_table_panics() {
    let mut eg = EGraph::default();
    eg.with_full_state(|fs| fs.lookup::<_, i64>("nope", 1_i64));
}

#[test]
fn test_higher_arity_function() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(function g (i64 i64 i64) i64 :no-merge)")
        .unwrap();
    eg.with_full_state(|mut fs| fs.set("g", (1_i64, 2_i64, 3_i64), 7_i64));
    let (v, has) = eg.with_full_state(|fs| {
        let v: Option<i64> = fs.lookup::<_, i64>("g", (1_i64, 2_i64, 3_i64));
        let has = fs.contains("g", (1_i64, 2_i64, 3_i64));
        (v, has)
    });
    assert_eq!(v, Some(7));
    assert!(has);
}

#[test]
fn test_string_inputs() {
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, "(function name-length (String) i64 :no-merge)")
        .unwrap();
    eg.with_full_state(|mut fs| fs.set("name-length", ("hello".to_string(),), 5_i64));
    let got: Option<i64> =
        eg.with_full_state(|fs| fs.lookup::<_, i64>("name-length", "hello".to_string()));
    assert_eq!(got, Some(5));
}
