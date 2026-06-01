# Changes

## [Unreleased] - ReleaseDate

- Fix seminaive matching after nested containers rebuild in place by propagating dirty container ids through parent containers.
- Render nullary AST calls without a trailing space, e.g. (foo) instead of (foo ).
- Add a BigRat to-i64 primitive for integral rationals.
- Add f64 exp, log, and sqrt primitives.
- Add `RunReport::can_stop` so scheduler progress can be reported separately from database updates.
- Desugar `relation`s to `constructor`s to simplify the language and implementation. Relations no longer return unit `()` values.
- Refactored API to use [`TermId`] more consistently instead of `Term` where possible, simplifying egglog code.
- **Typed primitive surface for seminaive safety (#772).** Custom primitives now pick one of `PurePrim` / `ReadPrim` / `WritePrim` / `FullPrim` based on what the body needs, and register via the matching `add_*_primitive`. Rust enforces capability bounds via the state wrapper passed to the body; the egglog typechecker enforces context bounds. See the `egglog::exec_state` module docs and the `*Prim` trait docs for the full picture. Migration: `rust_rule` callbacks now take `&mut WriteState` (replacing `RustRuleContext`); a new `rust_rule_full` gives action callbacks read access. Higher-order primitives over `unstable-fn` values dispatch via `state.apply_function(&fc, args)`.
- Expose `Read::table_size(name)` and `Read::table_sizes()` so read-capable primitives can inspect row counts without raw execution-state access, while avoiding an all-table scan when only one table is needed.
- **`:naive` rule option.** Individual rules can opt out of seminaive evaluation with `:naive` (e.g. `(rule (...) (...) :ruleset r :naive)`). The query and actions then typecheck under the permissive `Read` / `Full` contexts so primitives that read or write the database — including HOFs that wrap custom-function lookups — can run inside the rule. The rule is matched against the entire database every iteration. Use this when correctness depends on reading e-graph state from inside a query and the seminaive trade-offs (untracked dependencies, missed re-firings) are unacceptable.
- **Typed read/write API on the `Read` / `Write` traits (#745, #751).** Name-indexed methods that mirror the egglog DSL one-to-one. Same surface inside and outside a rule:
  - Inside a rule: `add_rust_rule` / `add_rust_rule_full` callbacks already receive a `WriteState` / `FullState` with these methods.
  - Outside a rule: call `EGraph::with_full_state(|fs| ...)` to drive the same methods. Pending writes flush once, **after** the closure returns — a read-after-write in the same closure is not visible. Split write and read into separate `with_full_state` calls; batching many writes in one closure is the fast path (one flush + rebuild).
  - `Write::set(table, key, value) -> Result<(), Error>` — `(set (f k) v)`.
  - `Write::add_node(table, inputs) -> Result<Id, Error>` — mint or look up a constructor / relation eclass; the returned `Id` is a `Value` paired with its sort name.
  - `Write::remove(table, key) -> Result<(), Error>` — remove a row from any subtype.
  - `Write::union(x: Id, y: Id) -> Result<(), Error>` — union two eclasses; errors if their sorts differ.
  - `Read::lookup::<_, V: BaseValue>(table, key) -> Result<Option<V>, Error>` — read a function's output value.
  - `Read::eclass_of(table, inputs) -> Result<Option<Id>, Error>` — read a constructor's eclass without minting.
  - `Read::contains(table, key) -> Result<bool, Error>` — row presence, any subtype.
  - `Read::lookup_raw(name, &[Value]) -> Result<Option<Value>, Error>` — untyped escape hatch when the caller already has a `&[Value]`. Skips sort checks.
- **Runtime type checks.** Every typed trait method validates inputs and reports failures as `Error::ApiError` instead of panicking:
  - `WrongSubtype` — `set` on a constructor, `add_node` on a function, `lookup` on a constructor, `eclass_of` on a function.
  - `WrongArity` — wrong number of input columns for the table.
  - `WrongColumnSort` — a column value's sort doesn't match the declared input sort (e.g., passing a `String` where the table wants `i64`, or an `Id` of sort `Math` where `List` is expected).
  - `WrongOutputSort` — `set`'s value column has the wrong sort.
  - `UnionSortMismatch` — `union(x, y)` where `x.sort() != y.sort()`.
  - `MissingTable` — table name not registered.
- **`Id` typed eclass / base value.** A `Value` paired with a runtime egglog sort name. Returned by `add_node` / `eclass_of` / `intern`. Implements `IntoColumn` so it flows back through `set` / `add_node` / `union` / `subsume` with its sort tag for validation. The plumbing: `egglog_bridge::FunctionConfig` carries per-column `sort_names`; `TableAction` exposes them via `input_sort_names()` / `output_sort_name()`. Bare `Value` is still an `IntoColumn` (unchecked) for the escape-hatch case where the caller knows the sort.
- **`EGraph::intern::<T>(x) -> Result<Id, Error>`** and **`EGraph::extract::<T>(id: Id) -> Result<T, Error>`** look up the egglog sort name for `T` by `TypeId` at runtime from the registered sorts (`TypeInfo::rust_type_to_sort_name`, populated on `add_arcsort` from `Sort::value_type()`). User-added base sorts work automatically — no extra trait impls required. `intern` errors with `ApiError::UnknownBaseSort` if `T` has no registered egglog sort; `extract` additionally errors with `ApiError::WrongOutputSort` if the `Id`'s sort tag doesn't match `T`. Raw `EGraph::base_to_value` / `EGraph::value_to_base` stay public as untyped low-level conversion. `EGraph::lookup_function` demoted to `pub(crate)`; `EGraph::get_canonical_value` removed (was unused externally).
- `EGraph::table_rows::<R: FromRow>(table) -> Result<Vec<R>, Error>` iterates all rows of a named table — row shape depends on subtype: functions expose `(input..., output)`, constructors and relations expose `(input..., eclass)`. `EGraph::query::<R: FromRow>(vars, facts) -> Result<Vec<R>, Error>` runs a pattern query. Both stay on `EGraph` (they compile a fresh query plan, can't run inside a rule callback). `EGraph::intern::<T>(x)` / `EGraph::extract::<T>(v)` cover base-value conversion outside a rule.
- Row trait surface in `crate::api`: `IntoRow`, `IntoColumn` (with `column_sort()` for runtime tag), `FromRow`, `FromColumn`, plus `RawValues` escape hatch.
- **Primitive trait `apply` signatures (`PurePrim` / `WritePrim` / `ReadPrim` / `FullPrim`) use `&[Id] -> Option<Id>`** — `Id` is now the currency throughout the typed API, including primitive authoring. The bridge dispatch wrappers convert `&[Value]` → `&[Id]` (currently with empty / unchecked sort tags; per-signature input sort tagging is a future refinement) and unwrap the returned `Id` for the bridge.
- **`Core::base<T>(&id)` / `Core::id_of<T>(x, sort)` / `Core::intern_typed<T: BaseSortName>(x)`** helpers on the state-wrapper trait for ergonomic `Id` handling inside primitive bodies. `base` extracts a Rust base value from an `Id`; `id_of` and `intern_typed` construct an `Id` from a Rust base value (the latter with the sort name baked in via the [`BaseSortName`] trait for the standard set).
- **`EGraph::base<T>(&id) -> T`** — unchecked counterpart to `extract` on the `EGraph` itself. The previously public `EGraph::value_to_base` / `EGraph::base_to_value` were demoted to `pub(crate)` — external callers should reach for `intern` / `extract` (sort-checked) or `base` (unchecked) instead.
- **Reads no longer take user-supplied output type parameters.** Users declare table sorts once via egglog and from then on operate on tagged `Id`s for table results:
  - `Read::lookup(name, key) -> Result<Option<Id>, Error>` (no `V` param) — returned `Id` carries the function's output sort.
  - `EGraph::table_rows(table) -> Result<Vec<Vec<Id>>, Error>` (no `R` param) — each `Id` tagged with the column's declared sort.
  - `EGraph::query(vars, facts) -> Result<Vec<Vec<Id>>, Error>` (no `R` param) — each row's `Id`s tagged with each declared `vars![…]` sort.
  - `Read::lookup_raw(name, &[Id]) -> Option<Id>` — the returned `Id` also carries the output sort tag now (previously empty).
  - `Core::extract<T: BaseSortName>(&id) -> Result<T, Error>` — state-side counterpart to `EGraph::extract`, for use inside primitive bodies / `with_full_state` closures. Constrained to the standard base types; user-defined sorts go through `EGraph::extract` (which uses runtime `TypeId` lookup).
- **`FromRow` / `FromColumn` traits dropped.** No longer needed at the public API — reads always return `Vec<Id>` / `Option<Id>` and users convert via `extract::<T>` only when exiting to a Rust base value. The trait surface in `crate::api` is now just `IntoRow` / `IntoColumn` (input side) + the `Id` / `ColumnSort` / `RawValues` types.
- **`BaseSortName`** trait — egglog sort name as a compile-time `const` for the standard Rust base types (`i64`, `bool`, `()`, `f64`, `String`, `BigInt`, `BigRat`). Used internally by `add_primitive!` and by `Core::intern_typed`. User-defined base sorts don't implement this — `EGraph::intern` / `EGraph::extract` look up sort names by `TypeId` at runtime.
- **`Value` is no longer re-exported from `egglog`** — `pub use core_relations::Value` was demoted to `pub(crate)`. `Id` is now the sole user-visible identifier type. Two API surfaces shifted with it:
  - `FromRow for Vec<Value>` → `FromRow for Vec<Id>`. Untagged-row escape now yields `Id`s with empty sort tags.
  - `Read::lookup_raw(name, &[Value]) -> Option<Value>` → `Read::lookup_raw(name, &[Id]) -> Option<Id>`. Takes / returns `Id` with empty sort tags.
  
  Callers needing raw `Value` access (e.g., to pass into low-level `Core::value_to_base`) can still reach it via `Id::value()` — but the type name itself is no longer publicly nameable from `egglog`. Internal users (sort impls, primitives in the `egglog` crate) continue to use `Value` via `crate::Value`. Anyone who really wants the raw type can import `core_relations::Value` directly, but it is no longer the documented path.

## [2.0.0] - 2026-02-11

Bigger changes

- Index catalog optimized for small set of indices (#719)
- Warn when globals lack the $ prefix; require globals to use the `$` prefix; missing prefixes now log a warning by default and can be upgraded to errors with `--strict-mode` or `EGraph::set_strict_mode`. (#722)
- Rename global vars in tests (#792, #800)
- Make interactive mode a delimiter (#729)
- Enable type-aware macros for fresh! sugar (#741)
- Proof preparation and term encoding (#742, #743, #765, #789)
- Export let bindings in the serialized format so they are visualized; Renames `ignore_viz` to `let_binding` (#701)
- Add snapshot tests (#778)

Bug fixes

- Fix Incorrect Unstable Function Behavior (#739)
- Run all tests in the workspace in CI (#776)

Performance improvements

- Low-level optimization for rebuilding (#754)
- Improve merge performance by being precise (#766)
- Avoid excessive cross-crate monomorphization (#773)
- Remove duplicate variables using functional dependency (#777)
- Memcpy for parallel writes and fix compilation failures (#779)

Misc. improvements

- Pin cargo codspeed version to fix CI (#734)
- Expose type constraints related APIs (#747)
- Remove lazy_static (#714)
- Simplify extract option handling (#759)
- Add longer extraction benchmark (#760)
- Specify that extractor does not support DAG costs (#763)
- Helpers for getting table sizes in primitives (#752)
- Refactor query planning (#780)
- Disable tracing tests (#787)
- Add initial early stopping support and use it for panic functions (#788)
- Update links in README for egglog resources (#798)


## [1.0.0] - 2025-10-18

This is the first release of egglog that is based on our new database-first, highly parallel backend.

**Abandoned features**

- `extract` is now a command instead of an action, which means calling `extract` within a rule is not allowed.
  Instead, the user is encouraged to use `print-function`.

Features

- Cost trait (#605)
- A new set of Rust APIs in `egglog::prelude` (#586)
- User-defined commands (#597)
- Scheduler interface for custom scheduling (#587)

Misc. Improvements

- Improves usability of `print-function` (#640)
- Desugar `rewrite`s to use `set`s when possible (#626)
- Grounded-ness check for ungrounded variables (#635)
- Don't panic when extracting nonexistent term (#629) 
- Documentation improvements (#634)
- Add parallelism flag and remove nondeterminism flag (#640, #642)
- Emit prompt and debug info when running from REPL (#672)
- Add support for the :unextractable flag for datatype variants (#712)
- Move egglog ast into its own crates (#670)

## [0.5.0] - 2025-6-9

This is the last major release before we switch to a database-first, highly parallel new backend.

Improvements

- Make `EGraph` thread-safe (#517)
- Support for egglog-python (#522)
- Throws type errors when unioning non-EqSort values (#561)
- Improvements to tests (#529)
- Improvements to error messages (#555)
- Makes union-find struct externally accessible (for container implementation) (#560)
- Disallow shadowing and interpret underscores as wildcards (#565)
- Faster `(push)` implementation

Bug fixes

- Fix value generations when `subsume`-ing a tuple in a relation (#569)
- Fixes to the new parser (#559)
- Rebuild after running commands instead of before (#573)

Benchmarks, serialization, and web demo

- Improvements to serialization (#520)
- Added eggcc benchmarks (#527)
- Fixes web demo escaping (#564, #566)
- Moves webdemo into a separate repository (#591)
- Fixes to Codspeed (#572)

## [0.4.0] - 2025-1-20

Semantic change (BREAKING)

- Split `function` into `constructor` and `functions` with merge functions. (#461)
- Remove `:default` keyword. (#461)
- Disallow lookup functions in the right hand side. (#461)
- Remove `:on_merge`, `:cost`, and `:unextractable` from functions, require `:no-merge` (#485)

Language features

- Add multi-sets (#446, #454, #471)
- Recursive datatypes with `datatype*` (#432)
- Add `BigInt` and `BigRat` and move `Rational` to `egglog-experimental` (#457, #475, #499)

Command-line interface and web demo

- Display build info when in binary mode (#427)
- Expose egglog CLI (#507, #510)
- Add a new interactive visualizer (#426)
- Disable build script for library builds (#467)

Rust interface improvements

- Make the type constraint system user-extensible (#509)
- New extensible parser (#435, #450, #484, #489, #497, #498, #506)
- Remove `Value::tag` when in release mode (#448)

Extraction

- Remove unused 'serde-1' attribute (#465)
- Extract egraph-serialize features  (#466)
- Expose extraction module publicly (#503)
- Use `set-of` instead of `set-insert` for extraction result of sets. (#514)

Bug fixes

- Fix the behavior of i64 primitives on overflow (#502)
- Fix memory blowup issue in `TermDag::to_string`
- Fix the issue that rule names are ignored (#500)

Cleanups and improvements

- Allow disabling messages for performance (#492)
- Determinize egglog (#438, #439)
- Refactor sort extraction API (#495)
- Add automated benchmarking to continuous integration (#443)
- Improvements to performance of testing (#458)
- Other small cleanups and improvements (#428, #429, #433, #434, #436, #437, #440, #442, #444, #445, #449, #453, #456, #469, #474, #477, #490, #491, #494, #501, #504, #508, #511)

## [0.3.0] - 2024-10-02

Cleanups

- Remove `declare` and `calc` keywords (#418, #419)
- Fix determinism bug from new combined ruleset code (#406)
- Fix performance bug in typechecking containers (#395)
- Minor improvements to the web demo (#413, #414, #415)
- Add power operators to i64 and f64 (#412)

Error reporting

- Report the source locations for errors (#389, #398, #405)

Serialization

- Include subsumption information in serialization (#424)
- Move splitting primitive nodes into the serialize library (#407)
- Support omitted nodes (#394)
- Support Class ID <-> Value conversion (#396)

REPL

- Evaluate multiple lines at once (#402)
- Show build information in the REPL (#427)

Higher-order functions (UNSTABLE)

- Infer types of function values based on names (#400)

Import relation from files

- Accept f64 function arguments #384

## [0.2.0] - 2024-05-24

Usability

- Improve statistics for runs (#284)
- Improve user-defined primitive support (#280, #288)
- Improve serialization (#293)
- Add more container primitives (#306)

Web demo

- Add slidemode in the web demo (#302)
- Fix box shadowing problem (#372)

Refactor

- Big refactoring to the intermediate representation (#320)
- Make global variables a syntactic sugar (#338)
- Drop experimental implementation for proofs and terms (#320, #342)

New features

- Support Subsumptions (#301)
- Add basic support for first-class, higher-order functions (UNSTABLE) (#348)
- Support combined rulesets (UNSTABLE) (#362)

Others

- Numerous bug fixes

## [0.1.0] - 2023-10-31

This is egglog's first release! Egglog is ready for use, but is still fairly experimental. Expect some significant changes in the future.

- Egglog is better than [egg](https://github.com/egraphs-good/egg) in many ways, including performance and new features.
- Egglog now includes cargo documentation for the language interface.

As of yet, the rust interface is not documented or well supported. We recommend using the language interface. Egglog also lacks proofs, a feature that egg has.


[Unreleased]: https://github.com/egraphs-good/egglog/compare/v2.0.0...HEAD
[0.1.0]: https://github.com/egraphs-good/egglog/tree/v0.1.0
[0.2.0]: https://github.com/egraphs-good/egglog/tree/v0.2.0
[0.3.0]: https://github.com/egraphs-good/egglog/tree/v0.3.0
[0.4.0]: https://github.com/egraphs-good/egglog/tree/v0.4.0
[0.5.0]: https://github.com/egraphs-good/egglog/tree/v0.5.0
[1.0.0]: https://github.com/egraphs-good/egglog/tree/v1.0.0
[2.0.0]: https://github.com/egraphs-good/egglog/tree/v2.0.0


See release-instructions.md for more information on how to do a release.
