# Herbie-across-backends — Handoff

Self-contained handoff for a fresh agent on a new machine. Everything you need is
committed; local memory, the scratchpad, and uncommitted experiment flags did **not**
travel — their conclusions are captured below.

## What the paper is about

**Thesis: add proofs to an e-graph engine as a front-end compiler pass, instead of
instrumenting the whole engine.** Production e-graph engines bolt proof-production into
every part of a bespoke engine (rewriting, congruence, union-find, rebuilding) — invasive
and hard to port. Instead we **encode the e-graph as a compiler pass** onto a *minimal
relational backend* that only needs a handful of generic operations: **run queries,
insert into tables, delete rows, merge (union-find)**. Equality saturation *and proofs*
fall out of that encoding — the backend stays a generic database, unaware of e-graphs or
proofs. (Proofs in particular become ordinary recorded rows / derivations in the encoding
rather than engine instrumentation.)

**Mechanism.** egglog's term/proof encoding lowers an e-graph program into relational
rules + tables over this minimal interface. The *same* encoding can target very different
databases — which is the portability claim.

**Portability — 4 backends, one encoding:**
- `bridge` — core-relations, egglog's native relational engine (also the bit-exact reference).
- `duckdb` — SQL / columnar OLAP.
- `feldera` — DBSP (incremental view maintenance).
- `flowlog` — differential-dataflow.

**Correctness bar.** The encoded (term) representation must be **bit-exact to egglog's
native (normal) encoding** — different representation, same semantics. Every backend must
reproduce the reference (`bridge` normal) tuple counts *exactly*. (Verified via
`measure_sizes`; see below.)

**The three results we're after:**
1. **Portability works** — all 4 backends run real workloads (Herbie, math-microbenchmark) bit-exactly. *Status: ✓ for bridge/feldera/flowlog; duckdb is a documented plan-bound gap on Herbie.*
2. **Different backends win on different workloads** — the compelling result: because the encoding is portable, you can pick a backend suited to the workload (incremental dataflow vs SQL vs the native join engine), and they have genuinely different performance profiles. *Status: this is exactly what the cross-backend perf grid (next step) quantifies.*
3. **Encoded performance ≈ native egglog** — close the gap between the *encoded* path and egglog's *default non-encoded* `bridge` performance. The encoding shouldn't cost much over native egglog. *Status: ongoing — the feldera/flowlog perf optimization (next step) is precisely this.*

So the running tasks map onto the paper: "term == normal bit-exact" is the **correctness
bar**; the 4-backend status is **portability**; and the upcoming feldera/flowlog perf work
serves both **result #2** (different backends win) and **result #3** (encoded ≈ native).

- Repo: `oflatt/egglog-duckdb` (remote `duckfork`). Branch `egglog-encoding-main` → pushed to `main`. The 4-backend **code** work is through `bf07b455`; `HANDOFF.md` + the eval sample are doc commits on top.
- A fork of `egraphs-good/egglog` carrying the 4-backend work.

## Build & run
- `cargo build --release -p egglog-experimental` → `target/release/egglog-experimental`. First duckdb build compiles duckdb-sys (C++, ~3–4 min).
- Backend = CLI flag on the binary: `--duckdb`, `--feldera`, `--flowlog`; no flag = native **bridge** (core-relations). `--term-encoding` = term encoding on bridge; `--proofs` = proof mode.
- `make test` (release), `make nits` / `make fixnits` before stopping.

## Status — Herbie across the 4 backends (all blockers RESOLVED)
| Backend | Engine | Herbie |
|---|---|---|
| **bridge** | core-relations (generic/WCOJ join) | runs all — the bit-exact REFERENCE |
| **flowlog** | differential-dataflow | runs full sample, **bit-exact** |
| **feldera** | DBSP | runs full sample, **bit-exact** |
| **duckdb** | SQL | runs the constructs but is PLAN-BOUND → times out; **documented capability gap** |

## Key results (for the paper)
1. **Term encoding is bit-exact to normal.** The apparent #18 divergence (~half the rewrites on some files) was *not* eqsat — it was the `get-size!` **size proxy** over-counting under term encoding (it summed the monotonic hash-cons term tables instead of the canonical `@<F>View` rows), tripping `:until (<= N (get-size!))` budgets early. Fixed in `37b2e197` (term-mode `get-size!` counts `@View`). With the budget removed, term == normal to the tuple (rewrite1280: 17780 == 17780).
2. **duckdb is plan-bound, not execution-bound:** profiled **99% planning / 0.9% execution**. DuckDB re-plans every per-rule SQL statement — no plan cache (issue duckdb#17237, closed "not planned") — and isn't built for many small queries. Levers measured:
   - `SET disabled_optimizers=...` → **1.38×, always bit-exact** (plan-independent). The one clean lever.
   - A **naive / no-seminaive** prototype gave 1.78× on the one-shot Herbie `lower`, **but it OVER-DERIVES on a real iterating workload** (math-microbenchmark run=3: bridge Add/Mul 69/77 vs naive 117/92 — mismatch). The prototype dropped the `ts < cur` *upper* bound, so a rule consumes rows its own round produced. **It is an unsound 1-shot hack, not a seminaive replacement.** A correct naive (keep `ts<cur`, drop only the per-focus lower bound + the `UNION ALL`) would be bit-exact but recover less.
   - Even with the optimizer fully off it is **98.6% planning** — the per-statement parse+bind is the irreducible floor. **No constant-factor lever closes the >8500× gap; the capability-gap verdict stands.**
3. **Row projection** unblocks wide rules on the fixed-width dataflow backends: register-allocation over the binding row (first-use..last-use liveness + linear-scan slot reuse), so the live-var *frontier* (≪ row width) fits — the giant 962-var Herbie seed rules and 34-var rules lower bit-exactly. flowlog `W=48`; feldera `JOIN_WIDTH=32` (+ post-join relayout for its call-prim `bigrat`/`from-string` literals). Default-on; kill-switches `FLOWLOG_NO_PROJECT` / `FELDERA_NO_PROJECT`.
4. **ReadPrims work generically on dataflow backends.** New `Backend::supports_action_registry()` capability (false on duck/flowlog/feldera, mirrors `supports_subsumption`) gates a registry-free prim wrapper + a frontend `name→table_size` snapshot in `check_facts`; `get-size!` is the first. Future value-reading ReadPrims work with zero extra code. The key bug: the term encoder dropped the `:until` rule's `:naive` flag → `get-size!` (Read-only) re-resolved in `Pure` context → "Unbound".

## Commits this effort (all on `duckfork/main`, ending `bf07b455`)
- `dbc968cc`,`2d673da6`,`2aa6556e` — Blocker-1: globals everywhere incl. proofs
- `2cb3f6b9`,`0ccc614b` — Herbie eval support (strip multi-extract + back-off scheduler, suite axis)
- `9139aa4a` — structured `(print-size)` parse (fixed vacuous parity — surfaced #18)
- `6f350a01`,`b454a08c` — flowlog row projection default-on
- `722a8902` — feldera row projection default-on
- `37b2e197` — `get-size!` `@View` term-count fix (resolves #18)
- `bf07b455` — `get-size!`/ReadPrim on flowlog+feldera (generic, frontend-only)

## NEXT STEPS
1. **Run the Herbie eval** (was task #10) across bridge / feldera / flowlog now that all three run — the cross-backend **parity + performance** grid.
2. **Optimize feldera/flowlog performance** on whatever the eval flags as slow vs bridge (the gold-standard generic-join engine). This is the dataflow-backend perf contribution. **duckdb perf is done** — documented gap, do not invest.
3. Lower priority: Herbie proof validation (needs a `bigrat` validator); `math.egg` 26=463 fiat-global proof bug; flowlog `math-microbenchmark` "Undefined sort Rational".

## How to run the eval
- Harness: `eval/bench_backends.py` (see `eval/README.md`, `eval/BLOCKERS.md`). Sample: `paper-benchmarks/herbie/sample100/` (committed with this handoff, 106 files).
- **Bit-exact verification:** use `measure_sizes(BIN, file, flags, env, timeout)` from `bench_backends` — it strips eval-only commands, appends `(print-size)`, and returns full per-function tuple counts. Compare each backend vs `bridge`. **Never grep for parity** — full tuple counts catch bugs headline checks miss.
- Herbie files use `(run R :until (<= N (get-size!)))` budgets + `multi-extract` + a back-off scheduler; `measure_sizes` strips the eval-incompatible bits.
- **Memory:** run the eval with `--jobs <= 2` (math-microbenchmark ~2 GB/cell); the duckdb test suite can OOM — `--test-threads=1` or filter.

## Working method / gotchas
- New optimizations go **behind a flag** (never break baselines) and become an orthogonal **eval axis**.
- One shared `target/release/egglog-experimental`: concurrent edit+build across parallel work corrupts it — isolate work in **git worktrees** (separate target dirs). Background builds survive turn boundaries; foreground agent-turn builds can get reaped.
- `bridge` is the gold standard (generic/WCOJ join). feldera/flowlog use binary-join chains (body order, no reordering) — compare per-rule to find the blowup. WCOJ-on-DD exists (`dogsdogsdogs`; `--wcoj` on flowlog).

## What did NOT travel (intentional)
The duckdb perf experiments (`DUCK_DISABLE_OPT`, `DUCK_NAIVE` flags + `DUCK_PROFILE_SPLIT` profiling) and the rejected #17 rule-batching prototype were **uncommitted** — their code did not transfer (we don't want it committed), but their conclusions are in §Key results #2. This tree is clean at `bf07b455`. An open upstream PR exists: `TimelyDataflow/differential-dataflow#765` (lookup_map fix).
