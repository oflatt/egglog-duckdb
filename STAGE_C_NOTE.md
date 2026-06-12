# FlowLog Stage C — DD is the only join path

Branch: `spike-flowlog-dd` (worktree `.perf-wt/flowlog-dd`). HEAD before this work:
`d328d594` (Stage B: 62/62 bit-exact on the DD path, gated behind
`EGGLOG_FLOWLOG_DD_NATIVE=1`).

## Goal achieved

The in-process Differential-Dataflow dataflow (`dd_native::PersistentDdJoin`) is
now the DEFAULT and ONLY join/bindings path for the FlowLog `Interpret` backend.
The host nested-loop interpreter and all its scaffolding are deleted. There is no
fallback: a rule the DD plan cannot lower `panic!`s with a specific reason.

## What was deleted

### `egglog-bridge-flowlog/src/interpret.rs` — 1446 → 440 lines (~1006 removed)

- **Host nested-loop / seminaive join**: `seminaive_bindings`, `step_atom`,
  `match_atom`, `empty_rows`, and the `step_body` table-atom branch (the
  delta-first nested-loop scan, greedy prim application, per-variant dedup).
- **Persistent join index**: the `IndexStore` struct + `FuncIndices`
  (`apply_insert`/`apply_remove`/`build_index`), `record_insert`/`record_remove`/
  `forget`/`sync`/`index_for`, and the pointer-stamped buffered-`pending` flush.
- **Incremental seminaive delta machinery**: `SeenState`, per-function `version`/
  `present_since`/`delta_log`/`last_remove_version`, `delta_since`,
  `window_insert_only`, `version_of`, `prune`, the per-rule version cursors, and
  the `read[f] \ snapshot` removal-fallback scan.
- **`delta_timer` module** (`FLOWLOG_DELTA_TIMER` phase timer) and its `Drop`
  hook.
- The `eg.dd_native_enabled` env gate (`run_iteration` no longer branches on it).

### `egglog-bridge-flowlog/src/lib.rs` (34 insertions, 104 deletions)

- Removed fields: `dd_enabled`, `dd_drivers`, `host_rule_runs`, `index_store`,
  `dd_native_enabled`. Removed `enable_dd_join()` and `flowlog_join_stats()`
  (replaced by `flowlog_dd_rule_runs()`).
- `seen` changed from `HashMap<usize, HashMap<FunctionId, SeenState>>` to
  `HashMap<usize, ()>` (a per-rule fire-once marker; see "what DD reuses" below).
- Removed the `EGGLOG_FLOWLOG_DD` / `EGGLOG_FLOWLOG_DD_NATIVE` env reads from the
  constructor; removed `index_store` maintenance hooks from `resolve_merge` and
  `clear_table`; removed the `pub mod dd_join` declaration, the
  `subprocess::DriverHandle` import, and the `Drop for EGraph` delta_timer dump.
- `free_rule` now also drops the rule's `dd_native` / `dd_native_fed` entries.

### Files removed

- `egglog-bridge-flowlog/src/dd_join.rs` (401 lines) — the obsolete shell-out DD
  path (`EGGLOG_FLOWLOG_DD=1`, clear-and-restage-every-call over a compiled
  subprocess). Cleanly obsolete now that the in-process raw-DD path exists; it
  was only reachable via the deleted `dd_enabled` gate.
- `egglog-bridge-flowlog/tests/dd_join_proof.rs` (157 lines) — `#[ignore]`d proof
  of the shell-out DD path; it depended on `enable_dd_join` + `flowlog_join_stats`
  and exercised `dd_join.rs`, so it was removed with that path.

Total: ~1006 lines from `interpret.rs` + 401 (`dd_join.rs`) + 157
(`dd_join_proof.rs`) + a net ~70 from `lib.rs` ≈ **~1560 lines removed**.

## What was KEPT because the DD path reuses it

- **`run_iteration`'s orchestration**: the start-of-iteration `Rc`-shared read
  snapshot, `next_id` change-detection, batched-removes-then-sets-then-FD-merge
  write application, `apply_head`/`build_row`/`resolve`/`lookup_or_create`. The DD
  driver (`run_iteration_dd_native` in Stage B) was merged INTO `run_iteration`
  (it is now the only body), so there is a single iteration loop.
- **`dd_native_bindings`** and **`dd_native.rs` / `PersistentDdJoin`** — the DD
  join itself.
- **The primitive re-run helper**: the old `step_body`'s prim branch survives as
  `step_prim` (table-atom branch removed). `dd_native_bindings` calls it to
  re-run `!=` guards / value prims host-side over the DD bindings, and for the
  atom-less rule body.
- **`seen` (now `HashMap<usize, ()>`)**: the atom-less-rule fire-once marker. This
  is load-bearing for the DD path: `(rule () …)` has no body table atom, so the
  DD dataflow has no input relation to drive it — `dd_native_bindings` evaluates
  it host-side and uses a `seen` entry to fire it exactly once. This is the one
  piece that *looked* like host-only seminaive scaffolding but is required by DD
  (per the Stage B note), so it was kept exactly. `free_rule` still clears it.

## Ineligible-rule panic behavior

After deletion there is no host fallback. In `dd_native_bindings`, when
`dd_native::plan_join(rule)` returns `Err(reason)` the driver panics:

```
FlowLog DD join cannot lower rule {name:?}: {reason}
(no host fallback; the DD dataflow is the only join path)
```

`plan_join` returns `Err` when: a body atom's arity exceeds the fixed binding-row
width cap `W=48`, the number of distinct body variables exceeds `W`, or the rule
has no body table atom (the atom-less case is handled separately before
`plan_join` is reached, so that arm is not hit in practice). For the current
corpus the residual is empty — every program that reaches the join lowers to DD —
so the panic never triggers on the test suite. Exotic/unsupported shapes panicking
is acceptable (signed off by the manager/user).

## dd_join.rs: REMOVED

Removed. It was the shell-out recompute-by-clearing DD path, reachable only via the
deleted `dd_enabled` gate, fully superseded by the in-process `dd_native`. Removal
was clean (only `dd_join_proof.rs`, also removed, referenced it). The M1/M2
in-process and shell-out *transitive-closure* paths (`engine.rs`, `subprocess.rs`,
`codegen.rs`, `run_one_hop*`, used by `run_n_proof.rs` / `run_n_shellout_proof.rs`)
are a separate feature and were left untouched.

## Load-bearing surprises

- The `seen` map looked like deletable host seminaive state but is required by the
  DD path for atom-less `(rule () …)` fire-once semantics (see above). Kept,
  re-typed to `HashMap<usize, ()>`.

## Final gate results

1. Build: `CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 cargo build --release --bin egglog` — OK.
2. Flowlog files suite WITHOUT the DD flag (DD is default):
   `EGGLOG_TEST_FLOWLOG=1 ./target/release/deps/files-<hash> flowlog --test-threads=1`
   → **62 passed; 0 failed**.
3. math-microbenchmark bit-exact vs reference: N=7 `(Add 1165) (Mul 1330)`,
   N=9 `(Add 12067) (Mul 11825)`. (Both match the reference backend exactly.)
4. `cargo clippy -p egglog-bridge-flowlog --tests --release -- -D warnings` — clean.
5. CHANGELOG.md updated with a Stage-C bullet.

## Ready commit message

```
FlowLog Stage C: DD dataflow is the only join path (delete host interpreter)

Make the in-process raw-differential-dataflow body join
(dd_native::PersistentDdJoin) the default and ONLY join/bindings path for the
FlowLog Interpret backend, and delete the now-dead host nested-loop machinery.

Deleted (~1560 lines): the host seminaive nested-loop join (seminaive_bindings,
step_atom, match_atom), the persistent delta-maintained IndexStore + its hooks,
the incremental seminaive delta machinery (SeenState, version/present_since/
delta_log, delta_since, removal-fallback scan), the FLOWLOG_DELTA_TIMER phase
timer, the EGGLOG_FLOWLOG_DD_NATIVE env gate, and the obsolete shell-out DD path
dd_join.rs + dd_join_proof.rs.

There is no host fallback: a rule the DD plan cannot lower (binding row > the
fixed width cap W=48, or an over-wide atom arity) now panics with a specific
reason, mirroring the Feldera backend's Stage-C posture.

Kept because the DD path reuses them: run_iteration's orchestration (read
snapshot, apply_head, write/FD-merge application), the prim re-run helper (now
step_prim), lookup_or_create, and the atom-less-rule fire-once marker (seen, now
HashMap<usize, ()> — the one rule shape evaluated host-side, since DD has no
input relation to drive (rule () …)).

Full flowlog files suite 62/62 with NO flag; math-microbenchmark bit-exact at
N=7 (1165/1330) and N=9 (12067/11825); clippy clean. FlowLog-only changes.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```
