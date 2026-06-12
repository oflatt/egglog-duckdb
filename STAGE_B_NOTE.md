# STAGE B: FlowLog in-process DD path — full coverage + wall inventory

Branch `spike-flowlog-dd`, worktree `.perf-wt/flowlog-dd`. No git ops performed;
all changes left in the working tree. Read `SPIKE_NOTE.md` first (Stage-A verdict
and staged plan); this note is what Stage B added on top.

## TL;DR

The Stage-A spike already lowered the FlowLog body join onto raw
differential-dataflow bit-exactly. Stage B's job was full coverage + a definitive
wall inventory. The result is sharper than expected:

- **62 / 62 flowlog-eligible test programs pass bit-exact on the DD-native path**
  (`EGGLOG_FLOWLOG_DD_NATIVE=1`), up from 59/62 at Stage A.
- The ONLY change needed to get from 59 to 62 — and indeed the only DD-path
  (join-engine) wall in the ENTIRE 89-program corpus — was the **fixed binding
  row width**. Raising it from `W=16` → `W=48` cleared every join-engine panic.
- **math-microbenchmark is bit-exact at N=7 (Add 1165 / Mul 1330), N=9
  (Add 12067 / Mul 11825), and the committed N=11 (Add 641743 / Mul 345075, all
  13 functions matching)** — the full `@uf` rebuild/retraction fixpoint is proven
  at scale, not just on tiny inputs.
- Every program the test harness EXCLUDES (the other 27) is blocked by the
  flowlog **frontend** (term-encoder / proof-encoding / push-pop `clone_boxed` /
  global handling) **before the join engine ever runs**. None is a DD limitation.

So: the host interpreter's join path is now fully redundant on every program that
reaches it. See `WALL_INVENTORY.md` for the per-program catalog.

## What was implemented (Stage B)

### 1. Wide fixed-arity DD row (work item #1) — the one real win

`egglog-bridge-flowlog/src/dd_native.rs`:

- Bumped the fixed binding-row width `W` from 16 → **48** (mirrors feldera's
  `JOIN_WIDTH=32`, plus headroom for the corpus's widest reachable rule:
  `luminal-llama`'s `@rebuild_rule34` uses **35 distinct body vars**, a wide-arity
  congruence-closure rebuild).
- This forced a **newtype `Row([u32; W])`** instead of the bare `[u32; W]`:
  DD `.join`/`.distinct` require `timely::ExchangeData = Serialize + Deserialize`,
  and serde only derives those for arrays up to length 32. The newtype carries a
  hand-written serde impl (`serialize_tuple(W)` / a `SeqAccess` visitor) that
  lifts the cap. `Ord`/`Hash`/`Clone`/`Copy` still auto-derive for any size, and
  `Index`/`IndexMut` keep call sites (`row[i]`) unchanged. Added `serde` as a
  direct dep of the crate.
- Raising `W` further is a one-line change (it is purely a fixed array size,
  `W*4` bytes/row); a rule exceeding `W` is reported as a row-width-cap wall.

### Coverage status of the other work items

- **Rule-shape coverage (#2):** already complete from the spike. The DD plan
  (`plan_join`) handles >=1 table atom, multi-atom left-deep joins, self-joins
  (`occ_of_func` fan-out), constants, repeated vars, and shared-var joins. The
  no-shared-var case degenerates to a unit-key join (cartesian), which is correct.
  Atom-less rules `(rule () …)` fire-once via the caller's `dd_native_bindings`
  fast path (reuses `seen` as a fired marker). All head actions
  (`set`/`remove`/`subsume`-noop/`lookup_or_create`/`call`/`union`-as-uf-write)
  are applied by the shared `apply_head`, identical to the host path → bit-exact
  writes + FD-merge.
- **Primitives (#3):** the spike's split runs the relational join on DD and
  re-runs body prims **host-side** over the produced bindings via
  `eval_prim_internal` — the SAME engine the host path uses, so it is already
  bit-exact for ALL prims (`+`, `int-div`, `string-concat`, `!=`, `or`, `guard`,
  `ordering-min/max`, f64 ops, …). No prim is a wall. The feldera-style
  on-circuit `PrimEngine`/`apply_steps` path (rep-inline + locked re-eval in a
  `.flat_map`) is a perf refinement, NOT a coverage requirement — left for later;
  see "What's left" below.
- **Full rebuild at scale (#4):** verified bit-exact at N=7/9/11 (above).
- **Single shared worker (#5, Stage-3 refinement):** NOT done. Each rule still
  owns its own timely `Worker` + dataflow (`eg.dd_native: HashMap<rule_idx,
  PersistentDdJoin>`). For programs with hundreds of rebuild rules this is many
  workers; consolidating onto one shared worker (one `worker.dataflow` per rule)
  is the obvious next step but is a sizeable change and does not affect coverage
  or bit-exactness, so it was deferred. NOTED, not blocking.

## What's left (before deleting the host interpreter)

1. **On-circuit prims (perf, not coverage).** Today the prim tail runs host-side
   after the DD join drains. Correct + bit-exact, but it materializes bindings
   out of the dataflow. Porting feldera's `RepKind`/`Cond`/`PrimStep`/`apply_steps`
   would keep prims on-circuit. Direct port from `egglog-bridge-feldera/src/dbsp_join.rs`.
2. **Shared timely worker** (work item #5) — consolidate per-rule workers.
3. **The deletion itself** is a separate, human-approved step (mirrors feldera
   Stage C / #33): make DD the only join path and `panic!` on the residual. The
   residual is now EMPTY for every program that reaches the join engine — see the
   recommendation in `WALL_INVENTORY.md`.

## How to run

```
cd .perf-wt/flowlog-dd
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 cargo build --release --bin egglog

# unit: build-once / incremental / retraction
cargo test -p egglog-bridge-flowlog --release --lib dd_native

# FULL flowlog suite through the DD-native path (62 trials, all bit-exact):
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 cargo test --release --test files --no-run
EGGLOG_TEST_FLOWLOG=1 EGGLOG_FLOWLOG_DD_NATIVE=1 \
  ./target/release/deps/files-<hash> flowlog --test-threads=1

# same suite with the DD flag OFF (default host interpreter, also 62/62) — proves
# the new path is fully flag-gated and the default path is byte-for-byte unchanged:
EGGLOG_TEST_FLOWLOG=1 ./target/release/deps/files-<hash> flowlog --test-threads=1

# math at N=7 / N=9 / N=11 (bit-exact vs reference) — edit (run 11) in the .egg
# or use the .tmp/math_n{7,9}.egg variants:
EGGLOG_FLOWLOG_DD_NATIVE=1 ./target/release/egglog --flowlog tests/math-microbenchmark.egg
./target/release/egglog tests/math-microbenchmark.egg   # reference

# clippy clean on new code:
cargo clippy -p egglog-bridge-flowlog --tests --release -- -D warnings
```

## Files touched

- `egglog-bridge-flowlog/src/dd_native.rs` — `W=48`, newtype `Row` + serde,
  `pack_key` via `empty_row()`.
- `egglog-bridge-flowlog/Cargo.toml` — added `serde` dep (for the wide-row serde).
- `Cargo.lock` — serde feature wiring for the crate.
- (no change to `interpret.rs` / `lib.rs` — the orchestration was already complete
  from Stage A.)
