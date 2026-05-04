//! Cross-cutting integration tests for the API redesign.
//!
//! Each test below pulls together features from multiple branches:
//! `oflatt-api-typed-egraph` (Read/Write trait methods + EGraph::with_full_state
//! + raw Value), `oflatt-api-rule-ergo` (rust_rule! macro, drop &mut on
//! container interning), `oflatt-api-cleanup` (declare-table builder), and
//! `oflatt-api-block-macro-draft` (egglog! proc macro). The point is to verify
//! these pieces compose — that you can write a realistic program touching all
//! four surfaces.

use egglog::prelude::*;
use egglog::{Error, RawValues, rust_rule};
use egglog_macros::egglog;

/// End-to-end: declare a self-contained block via `egglog!` (the
/// macro validates at compile time), then drive the e-graph from
/// Rust via the typed `set` / `lookup` / `query` methods, and step
/// the rule via the standard ruleset machinery.
///
/// Note: the egglog! block must be self-contained because the
/// proc-macro typechecks against a *fresh* EGraph at expansion time
/// — it can't see Rust-side declarations like `eg.declare(...)`.
#[test]
fn integration_full_pipeline() -> Result<(), Error> {
    let mut eg = EGraph::default();

    egglog!(
        eg,
        "(function fib (i64) i64 :no-merge)
         (ruleset fib_rs)
         (rule ((= f0 (fib x)) (= f1 (fib (+ x 1))))
               ((set (fib (+ x 2)) (+ f0 f1)))
               :ruleset fib_rs)"
    )?;

    // Seed via the typed Write surface.
    eg.with_full_state(|mut fs| {
        fs.set("fib", (0_i64,), 0_i64);
        fs.set("fib", (1_i64,), 1_i64);
    });

    // Step the named ruleset.
    for _ in 0..10 {
        run_ruleset(&mut eg, "fib_rs")?;
    }

    // Typed query (EGraph method — compiles a one-shot query plan).
    let rows: Vec<(i64, i64)> = eg.query::<(i64, i64)>("fib")?;
    let fib5 = rows.iter().find(|(k, _)| *k == 5).map(|(_, v)| *v);
    assert_eq!(fib5, Some(5));

    // Typed lookup via with_full_state.
    let (v, present) = eg.with_full_state(|fs| {
        (
            fs.lookup::<_, i64>("fib", 5_i64),
            fs.contains("fib", 5_i64),
        )
    });
    assert_eq!(v, Some(5));
    assert!(present);

    Ok(())
}

/// Declare-table builder (cleanup) + typed runtime API (typed-egraph)
/// without going through `egglog!`. This is the path users on the
/// "I want full Rust control, no DSL strings" track will take.
#[test]
fn integration_declare_builder_with_typed_api() -> Result<(), Error> {
    let mut eg = EGraph::default();

    eg.declare("f").input("i64").output("i64").function(None)?;
    eg.declare("R").input("i64").input("i64").relation()?;

    eg.with_full_state(|mut fs| {
        fs.set("f", (1_i64,), 42_i64);
        fs.add_node("R", (1_i64, 2_i64));
    });

    let (v, present) = eg.with_full_state(|fs| {
        (
            fs.lookup::<_, i64>("f", 1_i64),
            fs.contains("R", (1_i64, 2_i64)),
        )
    });
    assert_eq!(v, Some(42));
    assert!(present);

    Ok(())
}

/// Eclass-id `Value` flows through `set` (as a key column) and back
/// out of `eclass_of` — exercises the row trait surface end-to-end.
#[test]
fn integration_eclass_round_trip() -> Result<(), Error> {
    let mut eg = EGraph::default();

    egglog!(
        eg,
        "(datatype List (Cons i64 List) (Nil))
         (function list_length (List) i64 :no-merge)"
    )?;

    // Build a Cons chain + write via the eclass-id key column, all
    // batched in one closure (one flush at the end).
    let (two_one_nil, one_nil) = eg.with_full_state(|mut fs| {
        let nil = fs.add_node("Nil", RawValues(vec![])).unwrap();
        let one_nil = fs.add_node("Cons", (1_i64, nil)).unwrap();
        let two_one_nil = fs.add_node("Cons", (2_i64, one_nil)).unwrap();
        // Use the eclass id Value as a row key — Value is IntoColumn.
        fs.set("list_length", (two_one_nil,), 2_i64);
        (two_one_nil, one_nil)
    });

    // Read back in a second closure (so the writes have flushed).
    let (length, head_eclass) = eg.with_full_state(|fs| {
        (
            fs.lookup::<_, i64>("list_length", two_one_nil),
            fs.eclass_of("Cons", (2_i64, one_nil)),
        )
    });
    assert_eq!(length, Some(2));
    assert_eq!(head_eclass, Some(two_one_nil));

    Ok(())
}

/// `rust_rule!` (rule-ergo) calling `ctx.set` (typed-egraph) inside
/// the action body — verifies the macro's bindings struct + the typed
/// write API compose.
#[test]
fn integration_rust_rule_with_set() -> Result<(), Error> {
    let mut eg = EGraph::default();
    egglog!(
        eg,
        "(function fib (i64) i64 :no-merge)
         (set (fib 0) 0)
         (set (fib 1) 1)"
    )?;

    let ruleset = "fib_ruleset";
    add_ruleset(&mut eg, ruleset)?;

    rust_rule!(
        &mut eg,
        "step",
        ruleset,
        vars![x: i64, f0: i64, f1: i64],
        facts![ (= f0 (fib x)) (= f1 (fib (+ x 1))) ],
        |ctx, b| {
            // Typed bindings AND typed insert: no value_to_base /
            // base_to_value / iterator construction at the call site.
            ctx.set("fib", (b.x + 2,), b.f0 + b.f1);
            Some(())
        }
    )?;

    for _ in 0..10 {
        run_ruleset(&mut eg, ruleset)?;
    }

    let v = eg.with_full_state(|fs| fs.lookup::<_, i64>("fib", 8_i64));
    assert_eq!(v, Some(21));
    Ok(())
}

/// Subtype-mismatch panics: each direction-specific method panics
/// (programmer bug) when called on the wrong subtype. One panic per
/// `#[should_panic]` test.
#[test]
#[should_panic(expected = "Write::set called on constructor")]
fn integration_set_on_constructor_panics() {
    let mut eg = EGraph::default();
    egglog!(
        eg,
        "(function f (i64) i64 :no-merge)
         (datatype List (Nil))"
    )
    .unwrap();
    eg.with_full_state(|mut fs| fs.set("Nil", RawValues(vec![]), 0_i64));
}

#[test]
#[should_panic(expected = "Write::add_node called on function")]
fn integration_add_node_on_function_panics() {
    let mut eg = EGraph::default();
    egglog!(eg, "(function f (i64) i64 :no-merge)").unwrap();
    eg.with_full_state(|mut fs| {
        fs.add_node("f", 1_i64);
    });
}

#[test]
#[should_panic(expected = "Read::lookup called on constructor")]
fn integration_lookup_on_constructor_panics() {
    let mut eg = EGraph::default();
    egglog!(eg, "(datatype List (Nil))").unwrap();
    eg.with_full_state(|fs| fs.lookup::<_, i64>("Nil", RawValues(vec![])));
}

#[test]
#[should_panic(expected = "Read::eclass_of called on function")]
fn integration_eclass_of_on_function_panics() {
    let mut eg = EGraph::default();
    egglog!(eg, "(function f (i64) i64 :no-merge)").unwrap();
    eg.with_full_state(|fs| fs.eclass_of("f", 1_i64));
}
