//! Tests for the typed `EGraph::table_rows` and `EGraph::query` API.
//!
//! Reads no longer take user-supplied output type parameters — both
//! return `Vec<Vec<Id>>` with each `Id` tagged with the column's
//! declared sort. To convert an `Id` to a Rust base value, use
//! `EGraph::extract::<T>(id)` (sort-checked) or `EGraph::base::<T>(&id)`
//! (unchecked).

use egglog::prelude::*;
use egglog::Error;

/// Iterate a function table — rows come back as `[input_id, output_id]`.
#[test]
fn query_table_i64_to_i64() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(function f (i64) i64 :no-merge)
(set (f 1) 42)
(set (f 2) 43)
(set (f 7) 100)
",
    )?;

    let rows = egraph.table_rows("f")?;
    let mut pairs: Vec<(i64, i64)> = rows
        .iter()
        .map(|row| {
            assert_eq!(row.len(), 2);
            assert_eq!(row[0].sort(), "i64");
            assert_eq!(row[1].sort(), "i64");
            (
                egraph.extract::<i64>(row[0].clone()).unwrap(),
                egraph.extract::<i64>(row[1].clone()).unwrap(),
            )
        })
        .collect();
    pairs.sort();
    assert_eq!(pairs, vec![(1, 42), (2, 43), (7, 100)]);
    Ok(())
}

/// Pattern query over a relation — each match row has one `Id` per
/// declared variable, tagged with that variable's sort.
#[test]
fn query_pattern_relation_one_var() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(relation R (i64))
(R 1)
(R 2)
(R 7)
",
    )?;

    let results = egraph.query(vars![x: i64], facts![(R x)])?;
    let mut xs: Vec<i64> = results
        .iter()
        .map(|row| {
            assert_eq!(row[0].sort(), "i64");
            egraph.extract::<i64>(row[0].clone()).unwrap()
        })
        .collect();
    xs.sort();
    assert_eq!(xs, vec![1, 2, 7]);
    Ok(())
}

/// Zero-var query: returns one empty row per match.
#[test]
fn query_pattern_zero_vars_match() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(relation R (i64 i64))
(R 1 2)
",
    )?;

    let hits = egraph.query(vars![], facts![(R 1 2)])?;
    assert_eq!(hits.len(), 1);
    assert!(hits[0].is_empty());

    let misses = egraph.query(vars![], facts![(R 5 5)])?;
    assert!(misses.is_empty());

    Ok(())
}

/// Iterating a constructor table: rows are `[inputs..., eclass]`,
/// each `Id` tagged with its sort.
#[test]
fn query_constructor_table_eclass_values() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(sort Math)
(constructor Add (i64 i64) Math)
(let $a (Add 1 2))
(let $b (Add 3 4))
",
    )?;

    let rows = egraph.table_rows("Add")?;
    assert_eq!(rows.len(), 2);
    for row in &rows {
        assert_eq!(row.len(), 3);
        assert_eq!(row[0].sort(), "i64");
        assert_eq!(row[1].sort(), "i64");
        assert_eq!(row[2].sort(), "Math");
    }

    let mut input_pairs: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| {
            (
                egraph.extract::<i64>(r[0].clone()).unwrap(),
                egraph.extract::<i64>(r[1].clone()).unwrap(),
            )
        })
        .collect();
    input_pairs.sort();
    assert_eq!(input_pairs, vec![(1, 2), (3, 4)]);
    Ok(())
}

/// Relation rows expose the synthetic eclass column as a trailing
/// `Id` whose sort is the synthetic relation sort.
#[test]
fn query_relation_exposes_synthetic_output() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(relation R (i64))
(R 1)
(R 2)
",
    )?;

    let raw = egraph.table_rows("R")?;
    for row in &raw {
        assert_eq!(row.len(), 2, "relation row exposes (input, eclass)");
        assert_eq!(row[0].sort(), "i64");
    }

    // To get just the inputs as Ids tagged with i64, use the pattern-query form.
    let inputs = egraph.query(vars![x: i64], facts![(R x)])?;
    let mut xs: Vec<i64> = inputs
        .iter()
        .map(|row| egraph.extract::<i64>(row[0].clone()).unwrap())
        .collect();
    xs.sort();
    assert_eq!(xs, vec![1, 2]);
    Ok(())
}

/// Pattern query binding multiple base-sort variables.
#[test]
fn query_pattern_two_vars() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(
        None,
        "
(function f (i64) i64 :no-merge)
(set (f 1) 10)
(set (f 2) 20)
(set (f 3) 30)
",
    )?;

    let results = egraph.query(vars![x: i64, y: i64], facts![(= (f x) y)])?;
    let mut pairs: Vec<(i64, i64)> = results
        .iter()
        .map(|row| {
            (
                egraph.extract::<i64>(row[0].clone()).unwrap(),
                egraph.extract::<i64>(row[1].clone()).unwrap(),
            )
        })
        .collect();
    pairs.sort();
    assert_eq!(pairs, vec![(1, 10), (2, 20), (3, 30)]);
    Ok(())
}

/// Empty function: zero rows.
#[test]
fn query_empty_table() -> Result<(), Error> {
    let mut egraph = EGraph::default();
    egraph.parse_and_run_program(None, "(function h (i64) i64 :no-merge)")?;
    let rows = egraph.table_rows("h")?;
    assert!(rows.is_empty());
    Ok(())
}

/// Missing function: error.
#[test]
fn query_missing_table_errors() {
    let egraph = EGraph::default();
    let err = egraph.table_rows("nonexistent").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("nonexistent") || msg.contains("Unbound"),
        "expected unbound-function-style error, got: {msg}"
    );
}
