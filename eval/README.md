# Backend performance eval

Benchmarks egglog across the cross product of **backend** × **encoding** and
renders the results interactively with [eval-live](https://github.com/oflatt/eval-live).

|              | normal       | term-encoding       | proofs               |
| ------------ | ------------ | ------------------- | -------------------- |
| **bridge**   | *(no flags)* | `--term-encoding`   | `--proofs`           |
| **duckdb**   | —            | `--duckdb`          | `--duckdb --proofs`  |
| **feldera**  | —            | `--feldera`         | `--feldera --proofs` |
| **flowlog**  | —            | `--flowlog`         | `--flowlog --proofs` |

bridge is the only backend with a real **normal** (non-term-encoded) mode; it
is kept as the baseline. **duckdb**, **feldera**, and **flowlog** are
term-encoding-only — their backend flag already implies the term encoding, so
each runs just two cells: term-encoding (the bare backend flag) and proofs
(proof-instrumented term encoding, `--proofs`). The degenerate "normal" cell
for those backends would be the same engine as term-encoding, so it is skipped
rather than double-run. `--feldera`, `--flowlog`, and `--duckdb` are mutually
exclusive. This yields 9 cells: bridge × 3 encodings + (duckdb, feldera,
flowlog) × 2 encodings.

The benchmark binary is **`egglog-experimental`** (a CLI superset of plain
egglog), so the Herbie dumps' scheduler / `multi-extract` / `get-size!` forms
parse while mainline benchmarks run unchanged.

## The paper benchmark corpus

- **math-microbenchmark/math_full.egg** — runs (slow under duckdb/proofs).
- **herbie dumps** (`herbie/dump-egglog/dumps.tar.zst`, 1260 files) — run via
  egglog-experimental. Point `--path` at the tarball; it's auto-extracted to a
  sibling `.extracted/` dir (gitignored). Use `--limit N` to sample.
- **pointer-analysis/main.egg** — *excluded*: it `(input …)`s cclyzer++ CSV
  facts that aren't redistributed in this repo, so it can't run standalone.

## Run

```bash
pip install git+https://github.com/oflatt/eval-live.git   # once, for --serve

# FULL SWEEP: every .egg under tests/ across all 4 backends × 2–3 encodings.
# Run this on a QUIET machine (no other load, ideally the concurrent Rust
# build finished) so the wall-clock timings are meaningful.
python3 eval/bench_backends.py tests/ --timeout 300 --runs 3 --warmup 1

# default corpus is tests/, so this is equivalent and also opens the viewer:
python3 eval/bench_backends.py --serve

# the Herbie dumps (auto-extracted from the tarball), sampling 20:
python3 eval/bench_backends.py paper-benchmarks/herbie/dump-egglog/dumps.tar.zst --limit 20

# point it at any file or directory:
python3 eval/bench_backends.py tests/web-demo --runs 3 --warmup 1

# re-open the viewer on existing results without re-running:
python3 eval/bench_backends.py --justserve
```

## Timing mode: paper vs. fast-coverage (IMPORTANT)

By default cells run **sequentially**, one subprocess at a time. Two flags
change the scheduling *across* cells (the per-cell warmup + timed runs are
identical in every mode — only which cells run at the same time changes):

- **`--paper`** — **PAPER MODE.** Forces strictly sequential execution
  (parallelism OFF) so each cell runs uncontended and its wall-clock time is
  accurate. **All numbers reported in the paper must be produced with
  `--paper`.** It overrides `--parallel`/`--jobs` (sequential wins) and bumps
  `--warmup` to at least 1.
- **`--parallel` / `--jobs N`** — **FAST COVERAGE only.** Runs benchmark cells
  **concurrently** across a `ProcessPoolExecutor`. `--jobs N` implies
  `--parallel`; bare `--parallel` defaults to `min(8, cpu_count-1)` workers.
  This is for quickly seeing *which cells run vs. error* over a large corpus.
  Concurrent egglog/duckdb/feldera/flowlog processes contend for CPU, so the
  recorded per-cell wall-times are **inflated and NOT paper-quality** — never
  cite them as timings.

```bash
# fast coverage sweep over a big corpus (timings inflated, don't cite):
python3 eval/bench_backends.py tests/ --parallel            # min(8, cpu-1) jobs
python3 eval/bench_backends.py tests/ --jobs 4              # explicit job count

# accurate, paper-quality timing (run on a QUIET machine):
python3 eval/bench_backends.py tests/ --paper --runs 5 --timeout 600
```

`results.json` records a top-level **`"timing_mode"`** field
(`"paper-sequential"`, `"parallel-Njobs"`, or `"sequential"`) so the graphs/CDF
never silently mix paper-quality and contended timings.

**Memory / OOM caveat:** each worker spawns a full egglog/duckdb/feldera/flowlog
subprocess, so `--parallel` runs N benchmark engines at once and can OOM on
memory-heavy benchmarks (duckdb especially). The `min(8, cpu-1)` default cap
helps; lower `--jobs` if you see workers killed.

The default corpus is now **`tests/`** (the paper-benchmarks corpus mostly
doesn't run standalone — see `BLOCKERS.md`). Some benchmarks (e.g. `rectangle`)
legitimately exceed `--timeout`; that is data, not a harness failure — the cell
is recorded as a timeout error and simply never reaches "completed" in the
completion-CDF graph.

Useful flags: `--runs N`, `--warmup N`, `--timeout SECONDS`, `--debug` (use the
debug build), `--output PATH`, `--port N`, `--paper` (accurate sequential
timing), `--parallel` / `--jobs N` (fast contended coverage; see above).

A cell that errors or times out is recorded in the `errors` table instead of
`timings`. Results stream to `eval/results.json` after each benchmark, so you
can `--justserve` while a long run is still going.

## Files

- `bench_backends.py` — the driver (cross product, subprocess timing, JSON db, viewer).
- `graphs.py` — eval-live graphs/tables (runs in-browser via Pyodide). Includes
  a **Completion CDF / performance profile**: x = wall-clock T (log), y = number
  of benchmarks each treatment completed within T, one step curve per
  `(backend, encoding)`. Up-and-to-the-left is better; timed-out/errored cells
  never reach "completed" so they don't contribute, and curves may plateau at
  different heights.
- `results.json` — output.
