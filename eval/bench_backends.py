#!/usr/bin/env python3
"""Benchmark egglog across the cross product of {backend} x {encoding}.

Backends:  bridge (default in-memory), duckdb (`--duckdb`),
           feldera (`--feldera`), flowlog (`--flowlog`).
Encodings: normal, term-encoding (`--term-encoding`), proofs (`--proofs`).

For every (benchmark file, backend, encoding) cell we run the egglog CLI as a
subprocess `--runs` times (after `--warmup` discarded runs) and record the
wall-clock timings. Cells that error (non-zero exit, timeout) are recorded in
an `errors` table instead. Results are written to a JSON database that
`eval-live` renders interactively (`--serve`).

bridge is the only backend with a meaningful "normal" (non-term-encoded) mode;
it is kept as a baseline. duckdb, feldera, and flowlog are term-encoding-only:
their backend flag already implies the term encoding, so for them we run two
cells, term-encoding and proofs (proofs = proof-instrumented term encoding).
The degenerate "normal" cell for those backends is the same engine as
term-encoding and is therefore skipped (not double-run).

Usage:
    python3 eval/bench_backends.py tests/           # benchmark every .egg under tests/
    python3 eval/bench_backends.py --serve          # also open the eval-live viewer
    python3 eval/bench_backends.py path/to/dir      # benchmark every .egg under a dir
    python3 eval/bench_backends.py tests/ --runs 5 --warmup 1 --timeout 600
    python3 eval/bench_backends.py tests/ --paper    # accurate sequential timing (paper)
    python3 eval/bench_backends.py tests/ --parallel # fast contended coverage sweep
    python3 eval/bench_backends.py --justserve      # skip benchmarking, view existing results

Scheduling: by default cells run sequentially. `--paper` forces strictly
sequential, uncontended execution for accurate paper-quality timings (and
overrides --parallel/--jobs). `--parallel`/`--jobs N` run cells concurrently
across a process pool for FAST COVERAGE only -- contended wall-times are
inflated and must not be cited. results.json records the `timing_mode`.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]

# (name, backend-selecting CLI flags, term-encoding-only?) for each backend.
# bridge has a real "normal" (non-term) mode; the others are term-only: their
# backend flag already implies the term encoding.
BACKENDS = [
    ("bridge", [], False),
    ("duckdb", ["--duckdb"], True),
    ("feldera", ["--feldera"], True),
    ("flowlog", ["--flowlog"], True),
]
# (name, extra CLI flags) per encoding axis.
ENCODINGS = [
    ("normal", []),
    ("term-encoding", ["--term-encoding"]),
    ("proofs", ["--proofs"]),
]


def conditions():
    """The cross product backends x encodings, as
    (condition, backend, encoding, flags) tuples, with degenerate cells
    skipped:

    * bridge      -> normal, term-encoding, proofs  (full baseline)
    * term-only   -> term-encoding, proofs

    For a term-only backend the "normal" cell is identical to its
    "term-encoding" cell (the backend flag already implies term encoding), so
    we skip "normal" rather than double-run it. Its "term-encoding" cell needs
    no `--term-encoding` flag (that would be redundant / can panic the
    backend); "proofs" adds `--proofs` for proof-instrumented term encoding.
    """
    for backend, bflags, term_only in BACKENDS:
        for encoding, eflags in ENCODINGS:
            if term_only and encoding == "normal":
                # Degenerate: same engine as this backend's term-encoding cell.
                continue
            if term_only and encoding == "term-encoding":
                # Backend flag already implies term encoding; don't re-pass it.
                flags = list(bflags)
            else:
                flags = bflags + eflags
            yield (f"{backend}-{encoding}", backend, encoding, flags)


class BenchDB:
    """Minimal results database, serialized to the JSON shape eval-live reads:
    {"timings": [...], "errors": [...]}."""

    def __init__(self, timing_mode="unknown"):
        self.timings = []
        self.errors = []
        # Provenance: "paper-sequential" (accurate, uncontended) vs
        # "parallel-Njobs" (fast coverage, contended -> inflated wall-times).
        self.timing_mode = timing_mode

    def add_timing(self, benchmark, backend, mode, condition, timing_list):
        self.timings.append({
            "benchmark": benchmark,
            "backend": backend,
            "mode": mode,
            "condition": condition,
            "timing_list": timing_list,
        })

    def add_error(self, benchmark, backend, mode, condition, error):
        self.errors.append({
            "benchmark": benchmark,
            "backend": backend,
            "mode": mode,
            "condition": condition,
            "error": error,
        })

    def to_dict(self):
        return {"timing_mode": self.timing_mode,
                "timings": self.timings, "errors": self.errors}

    def save_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def build_egglog(release: bool) -> Path:
    """Build the egglog-experimental CLI and return its path. We use the
    experimental binary (a strict CLI superset of plain egglog) so the
    Herbie dumps' scheduler / multi-extract forms parse, while mainline
    benchmarks still run unchanged."""
    profile = ["--release"] if release else []
    print(f"Building egglog-experimental ({'release' if release else 'debug'})...", flush=True)
    subprocess.run(
        ["cargo", "build", *profile, "-p", "egglog-experimental", "--bin", "egglog-experimental"],
        cwd=WORKSPACE, check=True,
    )
    target = "release" if release else "debug"
    binary = WORKSPACE / "target" / target / "egglog-experimental"
    if not binary.exists():
        sys.exit(f"egglog-experimental binary not found at {binary}")
    return binary


def find_benchmarks(path: Path) -> list[Path]:
    # A `.tar.zst` (e.g. the Herbie dumps) is extracted once to a sibling
    # `<name>.extracted/` dir and benchmarked from there.
    if path.is_file() and path.name.endswith(".tar.zst"):
        dest = path.with_name(path.name.replace(".tar.zst", "") + ".extracted")
        if not dest.exists():
            print(f"Extracting {path.name} -> {dest} ...", flush=True)
            dest.mkdir(parents=True)
            subprocess.run(["tar", "-xf", str(path), "-C", str(dest)], check=True)
        return sorted(dest.rglob("*.egg"))
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.egg"))


def parse_duck_phases(stderr: str):
    """Parse the `--- by class ---` block emitted by the duckdb backend's
    DUCK_PERF_DUMP into {phase: {"search": s, "apply": s}}. Returns {} if
    the block is absent (non-duckdb backend, or env flag unset)."""
    phases = {}
    in_block = False
    for line in stderr.splitlines():
        if "by class" in line:
            in_block = True
            continue
        if not in_block:
            continue
        # Format: "    1.364s  search   1.151s  apply   0.212s   42.0%wall  rebuild"
        toks = line.replace("s", " ").split()
        # Expect: <total> search <search> apply <apply> <pct>%wall <kind>
        if "search" in line and "apply" in line and line.strip():
            parts = line.split()
            try:
                kind = parts[-1]
                # values are at fixed positions: [0]=total, [2]=search, [4]=apply
                search = float(parts[2].rstrip("s"))
                apply_ = float(parts[4].rstrip("s"))
                phases[kind] = {"search": search, "apply": apply_}
            except (IndexError, ValueError):
                in_block = False
        else:
            in_block = False
    return phases


def run_once(binary: Path, flags: list[str], bench: Path, timeout: float,
             capture_phases: bool = False):
    """Run one invocation. Returns (elapsed_seconds, None) on success or
    (None, error_message) on failure/timeout. When `capture_phases` is set
    (duckdb backend), enables DUCK_PERF_DUMP and returns
    (elapsed, None, phases) instead."""
    cmd = [str(binary), *flags, str(bench)]
    env = None
    if capture_phases:
        import os
        env = dict(os.environ, DUCK_PERF_DUMP="1")
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env=env)
    except subprocess.TimeoutExpired:
        return (None, f"timeout after {timeout}s", {}) if capture_phases \
            else (None, f"timeout after {timeout}s")
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        msg = tail[-1] if tail else f"exit code {proc.returncode}"
        return (None, f"exit {proc.returncode}: {msg}", {}) if capture_phases \
            else (None, f"exit {proc.returncode}: {msg}")
    if capture_phases:
        return elapsed, None, parse_duck_phases(proc.stderr or "")
    return elapsed, None


def bench_cell(binary, bench, rel, condition, backend, mode, flags, warmup, runs, timeout):
    """Run one (benchmark, backend, encoding) cell: `warmup` discarded runs
    followed by `runs` timed runs. Returns a result dict that the main process
    folds into the DB. Pure (no shared state) so it is safe to run in a worker
    process; a failing cell returns an error result rather than raising, so one
    cell's failure never kills the pool."""
    # Capture per-phase timing (search vs apply, by ruleset class) for the
    # duckdb backend, which emits a DUCK_PERF_DUMP breakdown on stderr.
    capture = backend == "duckdb"

    # Warm-up runs (discarded): pay one-time costs (page cache, etc.).
    for _ in range(warmup):
        run_once(binary, flags, bench, timeout)

    timings = []
    phase_runs = []
    for _ in range(runs):
        out = run_once(binary, flags, bench, timeout, capture_phases=capture)
        if capture:
            elapsed, err, phases = out
        else:
            elapsed, err = out
            phases = None
        if err is not None:
            return {"benchmark": rel, "backend": backend, "mode": mode,
                    "condition": condition, "error": err}
        timings.append(round(elapsed, 6))
        if phases:
            phase_runs.append(phases)

    result = {"benchmark": rel, "backend": backend, "mode": mode,
              "condition": condition, "timing_list": timings}
    if phase_runs:
        # Median across runs, per phase, for search and apply seconds.
        kinds = set().union(*[set(p) for p in phase_runs])
        agg = {}
        for k in kinds:
            for field in ("search", "apply"):
                vals = sorted(p[k][field] for p in phase_runs if k in p)
                if vals:
                    agg.setdefault(k, {})[field] = vals[len(vals) // 2]
        result["duck_phases"] = agg
    return result


def _bench_cell_worker(task):
    """Picklable entry point for ProcessPoolExecutor workers. `task` is the
    tuple of positional args for `bench_cell`."""
    return bench_cell(*task)


def apply_cell_result(result, db):
    """Fold a cell result (from `bench_cell`) into the DB and log it."""
    rel = result["benchmark"]
    backend = result["backend"]
    mode = result["mode"]
    condition = result["condition"]
    if "error" in result:
        db.add_error(rel, backend, mode, condition, result["error"])
        print(f"    {rel}: {condition:24} ERROR: {result['error']}", flush=True)
        return
    timings = result["timing_list"]
    db.add_timing(rel, backend, mode, condition, timings)
    mean = sum(timings) / len(timings)
    print(f"    {rel}: {condition:24} mean {mean:8.3f}s  (runs: {timings})", flush=True)
    phases = result.get("duck_phases")
    if phases:
        # Per-phase median search/apply seconds (duckdb DUCK_PERF_DUMP).
        order = sorted(phases, key=lambda k: -(phases[k].get("search", 0)
                                               + phases[k].get("apply", 0)))
        parts = [f"{k}={phases[k].get('search', 0) + phases[k].get('apply', 0):.3f}s"
                 f"(s{phases[k].get('search', 0):.3f}/a{phases[k].get('apply', 0):.3f})"
                 for k in order]
        print(f"      phases: {'  '.join(parts)}", flush=True)


def serve_results(results_path: Path, port: int):
    """Embed results + graphs into a self-contained HTML page and serve it."""
    import http.server
    import webbrowser
    import eval_live

    css = eval_live.css()
    js = eval_live.js()
    results_json = results_path.read_text()
    eval_live_py = eval_live.pyodide_lib()
    graph_script_path = Path(__file__).resolve().parent / "graphs.py"
    graph_script = graph_script_path.read_text() if graph_script_path.exists() else ""

    pyodide_tag = ""
    init_graphs_args = ""
    if graph_script:
        pyodide_tag = '<script src="https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js"></script>'
        init_graphs_args = f", {json.dumps(graph_script)}, {json.dumps(eval_live_py)}"

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>egglog backend eval</title>
<style>body {{ font-family: system-ui, sans-serif; margin: 0; padding: 2rem 3rem;
background: #f5f6f8; color: #1a1a1a; }} {css}</style>
{pyodide_tag}</head><body>
<div id="tables"></div>
<script>{js}
initEvalLive("tables", {results_json}, "egglog backends"{init_graphs_args});</script>
</body></html>"""

    page_bytes = page.encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(page_bytes)))
            self.end_headers()
            self.wfile.write(page_bytes)

        def log_message(self, *_):
            pass

    server = http.server.HTTPServer(("", port), Handler)
    url = f"http://localhost:{port}"
    print(f"\nServing eval-live at {url}  (Ctrl-C to stop)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default=None,
                        help="benchmark file or directory (default: tests/)")
    parser.add_argument("--runs", type=int, default=3, help="timed runs per cell (default 3)")
    parser.add_argument("--warmup", type=int, default=1, help="discarded warm-up runs (default 1)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-run timeout in seconds (default 300)")
    parser.add_argument("--limit", type=int, default=None,
                        help="benchmark at most N files (sample large corpora like the Herbie dumps)")
    parser.add_argument("--output", default=str(WORKSPACE / "eval" / "results.json"),
                        help="results JSON path")
    parser.add_argument("--debug", action="store_true",
                        help="use the debug build instead of release")
    parser.add_argument("--parallel", action="store_true",
                        help="run benchmark cells CONCURRENTLY via a process pool. "
                             "FAST COVERAGE only (which cells run vs error) -- "
                             "concurrent processes contend for CPU so per-cell "
                             "wall-times are INFLATED and NOT paper-quality. "
                             "Ignored when --paper is given.")
    parser.add_argument("--jobs", type=int, default=None,
                        help="number of concurrent worker processes (implies "
                             "--parallel). Default when --parallel is given: "
                             "min(8, cpu_count-1). Keep modest -- concurrent "
                             "egglog/duckdb/feldera/flowlog processes can OOM.")
    parser.add_argument("--paper", action="store_true",
                        help="PAPER MODE: force strictly SEQUENTIAL execution "
                             "(parallelism OFF) for accurate, uncontended "
                             "timings. Overrides --parallel/--jobs and ensures "
                             "warmup>=1. Paper numbers must use this mode.")
    parser.add_argument("--serve", action="store_true", help="open the eval-live viewer after running")
    parser.add_argument("--justserve", action="store_true", help="skip benchmarking; just serve results")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.justserve:
        serve_results(Path(args.output), args.port)
        return

    # Resolve scheduling mode. --paper forces strictly sequential (accurate,
    # uncontended timing) and overrides --parallel/--jobs. Otherwise --jobs
    # implies --parallel; --parallel with no --jobs uses a sensible cap.
    if args.paper:
        jobs = 1
        timing_mode = "paper-sequential"
        # Paper mode wants at least one warm-up to absorb one-time costs.
        if args.warmup < 1:
            args.warmup = 1
    elif args.parallel or args.jobs is not None:
        if args.jobs is not None:
            jobs = max(1, args.jobs)
        else:
            jobs = max(1, min(8, (os.cpu_count() or 1) - 1))
        timing_mode = f"parallel-{jobs}jobs" if jobs > 1 else "sequential"
    else:
        jobs = 1
        timing_mode = "sequential"

    binary = build_egglog(release=not args.debug)

    bench_path = Path(args.path) if args.path else (WORKSPACE / "tests")
    if not bench_path.is_absolute():
        bench_path = (WORKSPACE / bench_path)
    benchmarks = find_benchmarks(bench_path)
    if not benchmarks:
        sys.exit(f"no .egg benchmarks found under {bench_path}")
    if args.limit is not None:
        benchmarks = benchmarks[:args.limit]

    conds = list(conditions())
    sched = "PAPER (sequential)" if args.paper else (
        f"parallel ({jobs} jobs)" if jobs > 1 else "sequential")
    print(f"\n{len(benchmarks)} benchmark(s) x {len(conds)} condition(s), "
          f"{args.runs} run(s) each (warmup {args.warmup}, timeout {args.timeout}s)\n"
          f"timing mode: {timing_mode}  [{sched}]\n")

    db = BenchDB(timing_mode=timing_mode)

    def rel_of(bench):
        return (str(bench.relative_to(WORKSPACE))
                if str(bench).startswith(str(WORKSPACE)) else str(bench))

    # Build the full task list: one task per (benchmark, condition) cell. Each
    # task is fully self-contained so it can run in a worker process.
    tasks = []
    for bench in benchmarks:
        rel = rel_of(bench)
        for condition, backend, mode, flags in conds:
            tasks.append((binary, bench, rel, condition, backend, mode, flags,
                          args.warmup, args.runs, args.timeout))

    if jobs > 1:
        # FAST COVERAGE: schedule all cells across a process pool. Per-cell
        # logic (warmup + timed runs) is unchanged; only cross-cell scheduling
        # differs. Results are collected as they complete and folded into the
        # DB in the main process, so a failing cell can't kill the pool.
        done = 0
        total = len(tasks)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_bench_cell_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                done += 1
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001 - keep the pool alive
                    t = futures[fut]
                    result = {"benchmark": t[2], "backend": t[4], "mode": t[5],
                              "condition": t[3], "error": f"worker crashed: {exc}"}
                print(f"[{done}/{total}]", flush=True)
                apply_cell_result(result, db)
                db.save_json(args.output)  # incremental
    else:
        # SEQUENTIAL: one process at a time (paper-quality / default).
        last_rel = None
        for i, t in enumerate(tasks, 1):
            rel = t[2]
            if rel != last_rel:
                print(f"[{i}/{len(tasks)}] {rel}", flush=True)
                last_rel = rel
            result = bench_cell(*t)
            apply_cell_result(result, db)
            db.save_json(args.output)  # incremental: write after each cell

    print(f"\nResults written to {args.output}  (timing_mode: {timing_mode})")
    if args.serve:
        serve_results(Path(args.output), args.port)


if __name__ == "__main__":
    main()
