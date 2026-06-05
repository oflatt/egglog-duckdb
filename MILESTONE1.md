# Milestone 1 — FlowLog backend: bounded `(run N)` stepping behind the trait

**Verdict: ACHIEVED.** Bounded `(run N)` works end-to-end behind the
`egglog_backend_trait::Backend` interface on a **live, in-process flowlog-rs
`DatalogIncrementalEngine`** (Differential Dataflow), and it matches the
reference backend (`egglog_bridge::EGraph`) round-for-round on the same program.
The load-bearing requirement — **one egglog iteration per `run_rules` call**
(one flowlog `commit()` = one hop), NOT saturate-in-one-call — is realized with
a **non-recursive** flowlog program driven N times, plus a Rust-side
materialized mirror fed by `commit()`'s per-epoch deltas.

Crate: `egglog-bridge-flowlog/` (added to the workspace `members`). File layout
mirrors the Feldera backend: `lib.rs`, `compile.rs`, `rule_builder.rs`,
`base_values.rs`, `external_func.rs`, plus `engine.rs` (the flowlog-specific
piece that owns the generated incremental engine) and the build-time
`transitive_step.dl` + `build.rs`.

flowlog-rs is the local clone at `/tmp/flowlog-main` (workspace v0.4.0;
flowlog-build 0.3.0 / flowlog-runtime 0.2.2), depended on via path deps. Built
on rustc 1.91.0; pulls timely 0.29 + differential-dataflow 0.23.

## The `(run 1)` vs `(run 3)` proof (actual test output)

Program (single-join transitive-closure step over the chain `1->2->3->4`):

```
edge(x,y), path(x,y)   seeded with {(1,2),(2,3),(3,4)}
path(x,z) :- path(x,y), edge(y,z).
```

Test `run1_vs_run3_bounded_and_matches_reference`
(`egglog-bridge-flowlog/tests/run_n_proof.rs`) drives BOTH backends through the
`Backend` trait and prints (verbatim):

```
run(1) reference = {(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
run(1) flowlog   = {(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
run(3) reference = {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
run(3) flowlog   = {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
test run1_vs_run3_bounded_and_matches_reference ... ok
```

- `(run 1)` adds **one hop** (`(1,3)`, `(2,4)`) but does **NOT** contain the
  3-hop pair `(1,4)` — bounded, not saturated.
- `(run 3)` reaches the full closure including `(1,4)`.
- FlowLog **equals the reference backend** at both N — the faithfulness proof.
  `run(1) != run(3)` is the bounded-iteration proof.

Reproduce:
```
cargo test --release --manifest-path egglog-bridge-flowlog/Cargo.toml \
  --test run_n_proof -- --nocapture
```

## How the per-iteration model maps onto flowlog-rs

- The bundled `transitive_step.dl` is **non-recursive**:
  `hop(x,z) :- path(x,y), edge(y,z).` with `edge` and `path` as runtime-staged
  command inputs (`IO="command"`) and `hop` as the `.output`. A *recursive*
  `.dl` would let flowlog saturate to the full closure inside ONE `commit()` —
  exactly the trap the Feldera Phase-0 spike's recursive DD scope fell into, and
  wrong for egglog's bounded iteration.
- `build.rs` compiles it in `ExecutionMode::DatalogInc`, producing a
  `DatalogIncrementalEngine` (`insert_edge`/`insert_path`/`remove_*` staging +
  `commit()`), `include!`d in `engine.rs`.
- `run_rules` = one bounded hop. On the **first** call it stages all current
  `edge` + `path` rows and `commit()`s once: the join yields the 1-hop
  extension as `hop` deltas. On **subsequent** calls it stages only the
  previous round's NEW `path` rows (a host-side feedback buffer), so each
  `commit()` is exactly one further hop — bounded extension, NOT saturation.
  This is the proven Feldera host-feedback model, but driving real flowlog DD.
- The host folds each epoch's `IncrementalResults.hop: Vec<((i32,i32), i32)>`
  deltas into a Rust-side materialized mirror (PLAN §4 design A); `for_each` /
  `lookup_id` / `table_size` read the mirror. `run_rules` reports `changed`
  honestly (mirror grew?) so the frontend's outer loop terminates.
- M1 is monotone (no retraction yet); the engine wrapper ignores non-positive
  `commit()` diffs. Retraction-rebuild (`remove_*` = `-1`) is an M2 piece.

The rule shape is recognized structurally in `lib.rs::recognize_step`: two table
body atoms sharing a join variable, one `set` head binding the outer two
variables. This is what lets a rule the frontend builds at runtime map onto the
build-time-fixed `.dl`. Rules outside this shape error clearly (they are M2+).

## Trait methods: implemented vs stubbed

Implemented (real bodies): `add_table`, `table_size`, `approx_table_size`,
`for_each`, `for_each_while`, `lookup_id`, `add_values`, `add_term`,
`insert_rows`, `lookup_constructor_rows`, `get_canon_repr` (identity — no UF
yet), `fresh_id`, `clear_table`, `base_values`, `base_value_pool(_mut)`,
`base_value_constant_dyn`, `new_rule`, `free_rule`, `run_rules` (drives flowlog
`commit()`), `flush_updates`, `register_external_func` / `free_external_func` /
`new_panic` (registered into the embedded `Database`), capability flags,
`set_report_level`, `dump_debug_info`, `as_any` / `as_any_mut`.

`RuleBuilderOps` implemented: `new_var`, `new_var_named`, `query_table`, `set`.
The remaining ops (`query_prim`, `call_external_func`, `lookup`, `subsume`,
`remove`, `union`, `panic`, `rename_prim`) accumulate into the IR but are not
exercised by the M1 flowlog-driven path (recognized rule shape only).

Stubbed / errors (mirroring DuckDB/Feldera gating; deferred to later
milestones): `with_execution_state_dyn`, `action_registry_any`, `clone_boxed`
(push/pop — snapshot-and-replay), container pool (all empty/error),
`supports_inline_table_lookups` / `supports_subsumption` /
`supports_complex_merge` / `supports_containers` = `false`. Rules outside the
recognized transitive-closure-step shape error at `run_rules`.

## The FlowLog crux — runtime rule installation (investigated; M2 plan)

**The problem.** flowlog compiles `.dl` -> Rust at **BUILD** time (`build.rs`
calls `flowlog_build::Builder::compile`, which reads `$OUT_DIR` and writes
`$OUT_DIR/<stem>.rs`; the host crate then compiles that source normally).
egglog defines rules at **RUNTIME** from the program it is fed. This is the
FlowLog analog of Feldera's "static-circuit-rebuild" risk, but **harder**:
Feldera builds its dataflow circuit in-process with no compile step, whereas
flowlog's natural unit is "generate Rust source, then `rustc` it." For the M1
proof a build-time-fixed `.dl` is acceptable (per the brief) and is what this
crate ships.

**What the source investigation found (concrete, not speculative):**

- `flowlog_build::Builder::compile()` only requires `$OUT_DIR` to be set and
  emits Rust **source** — it does **not** invoke `rustc` itself. The expensive
  part (timely + differential-dataflow + the generated engine) is compiled by
  the *host crate's* normal build, not by flowlog-build.
- The internal `compile_one` / `build::assemble` take an arbitrary
  `out_dir: &Path` (`crates/flowlog-build/src/lib.rs:198-224`). Only the public
  `compile()` wrapper hard-codes `cargo_out_dir()` (reads the `OUT_DIR` env
  var). So **codegen to an arbitrary directory at runtime is feasible today**
  by setting `OUT_DIR` before calling `compile()` (the emitted
  `cargo:rerun-if-changed=` stdout lines are harmless outside a build), or with
  a small upstream patch exposing `compile_to_dir(out_dir)`.
- The generated module is self-contained: it `use`s the flowlog-runtime
  re-exports of timely/DD, so a generated `.rs` plus a tiny crate that
  `include!`s it and depends on `flowlog-runtime` is a complete compilation
  unit.

**Feasibility verdict: feasible but heavy.** The path is:
`emit .dl` → `flowlog codegen -> .rs` (cheap; milliseconds — pure source gen) →
`rustc`/`cargo build` a `cdylib` against `flowlog-runtime` (**this dominates**;
a cold build of the generated engine + its DD/timely deps was ~45s here, though
a *warm* incremental rebuild of just the regenerated engine crate — DD/timely
already compiled — is far cheaper, on the order of a few seconds) → `dlopen` the
cdylib and call its engine constructor through a stable extern interface.

**Cost shape and why it is the M2 crux:** egglog installs *many* rules (the term
and proof encoders emit dozens), and the schedule interleaves rule installation
with `check`/`extract`/new facts. Recompiling a dylib per rule-set change is
seconds-class latency — unviable if done naively per rule, acceptable if done
**once per "frozen schema"** (compile all installed rules into a single `.dl`
the first time `run_rules` is called, reuse the dylib until the rule set
changes). This is the same "freeze schema / no more rules" signal the Feldera M1
notes asked for, made sharper here because the rebuild crosses a `rustc`
boundary.

**Recommended M2 plan for runtime rule installation:**

1. **Compile-all-rules-once (the standing-dataflow model).** Accumulate every
   installed rule's IR; on first `run_rules` (or an explicit "schema frozen"
   hook), translate the whole rule set to one `.dl`, codegen + build a cdylib
   **once**, `dlopen` it, and drive it with `commit()` for the rest of the
   program. Rebuild the dylib only when the rule set actually changes (rare in
   practice: the frontend installs rules up front, then runs the schedule).
2. **Stable C ABI shim around the generated engine.** The generated
   `DatalogIncrementalEngine` has non-`#[repr(C)]` Rust types (tuple aliases,
   `IncrementalResults`), so dlopen needs a thin extern `"C"` wrapper:
   `engine_new`, `engine_insert(rel_id, *const i32, len)`,
   `engine_commit() -> serialized deltas`, `engine_free`. Generate this shim
   alongside the engine (small templating change), or vendor a fixed shim that
   the codegen targets.
3. **Cache keyed by rule-set hash.** Skip the rustc step when the `.dl` text is
   unchanged (hash → cached cdylib path), so re-running the same program is
   instant after the first build.
4. **Fallback / hybrid.** Keep a host-side interpreter (the path the Feldera
   backend ultimately took for its richer milestones) for rule shapes or
   schedule fragments where the dylib round-trip isn't worth it, and reserve the
   flowlog dylib for the hot saturation loop where DD's native semi-naïve +
   incremental retraction pays off.

The unblocking insight is that **the flowlog frontend codegen is cheap and
dir-agnostic; only the downstream `rustc`/DD compile is heavy**, and that cost
amortizes if we compile the whole rule set once and rebuild only on schema
change. So runtime rule installation is a real engineering cost (a `rustc`
boundary Feldera did not have) but is *not* a dead end — it is the central M2
deliverable.

## Not attempted (correctly out of scope for M1)

Eq-sorts + hash-cons + union-find/rebuild (M2 — the real "egraph on flowlog"
crux; needs retraction via `remove_*`, which flowlog's incremental mode supports
first-class), primitives + base-value functors (M3), `:merge new`/complex
merges, proofs, containers, push/pop (`clone_boxed`), and — above all — runtime
rule installation (the crux documented above). None are contradicted by this
milestone; they are the next gates.
