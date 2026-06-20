#!/usr/bin/env python3
"""Benchmark egglog across the backend-encoding treatment matrix.

This drives the `egglog-experimental` CLI: it registers the experimental sorts
and commands then dispatches to `egglog::cli`, so it carries the full backend /
encoding surface -- the `--native-uf` flag, the backend selectors (`--duckdb`,
`--feldera`, `--flowlog`), and the fast-rebuild env vars exercised by the
matrix. Its `src/main.rs` also pre-builds the duckdb-backed egraph in native-UF
mode when `--native-uf` (or `--duck-native-uf`) is set, so `--native-uf
--duckdb` actually engages native-UF rather than emitting UF-backed functions
against a relational backend.

Treatment axes
--------------
* backend  : bridge (default in-memory), duckdb (`--duckdb`),
             feldera (`--feldera`), flowlog (`--flowlog`).
* encoding : normal, term-encoding (`--term-encoding`), proofs (`--proofs`).
* native-UF: `--native-uf` -- the host-pass union-find encoding (vs relational
             UF). Only meaningful on term/proof cells (the relational-UF
             encoding it replaces only exists under term encoding).
* fast-rebuild: `--fast-rebuild` -- a custom/specialized rebuild vs the
             pure-engine rebuild (dataflow backends: custom host-pass vs pure
             DD engine; bridge/duckdb: on-engine specialized vs full). The flag
             drives it for ALL backends INCLUDING the bridge. It is ORTHOGONAL
             to native-UF: the two axes cross into four distinct, bit-exact
             rebuild cells per term-like cell -- plain, +fastrb, +nuf,
             +nuf+fastrb.
* wcoj     : `--wcoj` -- worst-case-optimal join, flowlog ONLY (rejected on the
             other backends). Crosses with flowlog's rebuild cells so each gets
             a `+wcoj` variant; correctness-preserving (same family reference).

The reference / oracle cell is **bridge-normal** (in-memory bridge, no term
encoding). Every other cell's correctness is judged against it by tuple-count
parity (see below).

Metrics captured per cell
-------------------------
* wall-clock: the timed subprocess elapsed (median/mean of `--runs`).
* peak RSS  : maximum resident set size in BYTES, via macOS `/usr/bin/time -l`.
* per-phase : a per-ruleset rebuild/canonicalize/congruence bucket profile.
              bridge uses `--save-report` (RunReport JSON); duckdb uses
              DUCK_PERF_DUMP; feldera uses FELDERA_PROFILE; flowlog uses
              FLOWLOG_DD_RULESET_PROF.
* parity    : tuple-count parity vs the bridge-normal oracle (deterministic
              `(print-size)` per-function counts; NOT extraction, which has
              benign tie-breaking covered by the Rust tests).

For every cell we run the egglog CLI as a subprocess `--runs` times (after
`--warmup` discarded runs). Cells that error (non-zero exit, timeout) are
recorded in an `errors` table instead. Files that error on the oracle itself
are skipped entirely (not all `tests/` files are benchmarkable). Results are
written to a JSON database that `eval-live` renders interactively (`--serve`).

Usage:
    python3 eval/bench_backends.py tests/ --paper            # full matrix (sequential)
    python3 eval/bench_backends.py tests/ --paper --limit 5  # small pilot
    python3 eval/bench_backends.py --serve                   # open the eval-live viewer
    python3 eval/bench_backends.py --justserve               # view existing results

Mode: runs PARALLEL across half the CPU cores by default (`--jobs N` to
override; half, not all, for memory headroom -- heavy cells are multi-GB) --
the normal dev mode, since parity is exact regardless of contention and
parallel wall-clock is a fine approximation while iterating. `--paper` forces
strictly SEQUENTIAL (jobs=1) for the FINAL eval's uncontended, paper-quality
timings/RSS.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
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

# The reference / oracle condition: a file is benchmarkable iff this cell runs.
ORACLE_CONDITION = "bridge-normal"

# Parity reference per ENCODING FAMILY. The term encoding's `(print-size)`
# REPRODUCES the normal-mode tuple counts exactly (the view-table filtering in
# `EGraph::print_size` reports the term_constructor name with the view size), so
# every non-proof treatment -- term-encoding, native-uf, fast-rebuild, on any
# backend -- is diffed against the single bridge-normal reference. This is
# exactly tests/files.rs's shared-snapshot model. Proof mode gets its own
# reference (its output can legitimately differ from normal).
REFERENCE_BY_FAMILY = {
    "normal": "bridge-normal",
    "term-encoding": "bridge-normal",
    "proofs": "bridge-proofs",
}


def _encoding_family(condition):
    """Encoding family of a condition label, for picking its parity reference.
    The `+nuf` / `+fastrb` axis suffixes don't change the family."""
    if "proofs" in condition:
        return "proofs"
    if "term-encoding" in condition:
        return "term-encoding"
    return "normal"

# Per-backend env var that ALSO turns on the specialized (fast) rebuild. The
# primary driver is now the `--fast-rebuild` CLI flag, which is carried in the
# cell's `flags` for EVERY backend INCLUDING the bridge (the bridge gained a
# relational fast-rebuild in the committed work); these env knobs are kept as
# legacy/redundant fallbacks for the backends that still honor them. bridge has
# no env knob (the flag alone drives it), so its +fastrb cell is flag-only.
FAST_REBUILD_ENV = {
    "bridge": None,  # flag-only: `--fast-rebuild` drives the bridge fast-rebuild
    "duckdb": "DUCK_DELTA_REBUILD",
    "feldera": "FELDERA_DELTA_REBUILD",
    "flowlog": "FLOWLOG_DELTA_REBUILD",
}

# Per-backend env var that turns on the per-ruleset / per-phase profile dump.
PROFILE_ENV = {
    "duckdb": "DUCK_PERF_DUMP",
    "feldera": "FELDERA_PROFILE",
    "flowlog": "FLOWLOG_DD_RULESET_PROF",
}


def conditions():
    """Enumerate the VALID cells of the treatment matrix as
    (condition, backend, encoding, flags, native_uf, fast_rebuild) tuples.

    Base cells (backend x encoding), degenerate ones skipped:

    * bridge      -> normal, term-encoding, proofs  (normal = the oracle)
    * term-only   -> term-encoding, proofs

    For a term-only backend the "normal" cell is identical to its
    "term-encoding" cell (the backend flag already implies term encoding), so
    we skip "normal" rather than double-run it. Its "term-encoding" cell needs
    no `--term-encoding` flag (that would be redundant / can panic the
    backend); "proofs" adds `--proofs` for proof-instrumented term encoding.

    On top of each term-like base cell we cross TWO orthogonal rebuild axes,
    yielding FOUR distinct, bit-exact rebuild cells per term-like cell:

    * native-UF (`--native-uf`): the host-pass union-find encoding (vs the
      relational-UF encoding). Only on term/proof cells (the relational-UF
      encoding it replaces only exists under term encoding). Never on the
      bridge-normal oracle.
    * fast-rebuild (`--fast-rebuild`): a custom/specialized rebuild vs the
      pure-engine rebuild. On the dataflow backends (feldera/flowlog) it
      selects a custom host-pass rebuild instead of the pure DD-engine rebuild;
      on bridge/duckdb it selects an on-engine specialized rebuild instead of
      the full one. The `--fast-rebuild` CLI flag drives it for ALL backends,
      INCLUDING the bridge (the bridge gained relational fast-rebuild in the
      committed work). All four cells are real and bit-exact:
        - plain        : relational UF, pure-engine rebuild              (no flag)
        - +fastrb      : relational UF, specialized rebuild              (--fast-rebuild)
        - +nuf         : native UF,     pure-engine rebuild              (--native-uf)
        - +nuf+fastrb  : native UF,     specialized rebuild              (--native-uf --fast-rebuild)

    PROOFS is a FULL orthogonal axis. `--proofs` is crossed with every treatment
    combo, on every backend that supports it, and the eval measures `--proofs`
    PERFORMANCE (the proof-instrumented term encoding). Proof CORRECTNESS is
    validated separately by tests/files.rs (proof_testing), NOT re-extracted
    per-cell here -- so there is no false-pass concern from the tuple-count parity.
    native-UF + proofs validates on bridge, feldera, and flowlog (verified via
    --proof-testing); only duckdb can't build proof-mode native-UF functions, so
    its proofs cells get no `+nuf` variant (see `nuf_ok`).

    WCOJ (`--wcoj`): worst-case-optimal join, flowlog-only (rejected elsewhere).
    Crossed into BOTH flowlog term-encoding AND proofs cells: proofs+wcoj is
    validated sound (the WCOJ join path preserves proof terms -- verified via
    --proof-testing on a cyclic-CQ-with-check file). So every flowlog term/proof
    rebuild cell gets a `+wcoj` variant. WCOJ is correctness-preserving, so it
    diffs against the same family reference (REFERENCE_BY_FAMILY is unchanged).

    The condition label carries the extra axes: "+nuf" for native-UF,
    "+fastrb" for fast-rebuild, "+wcoj" for the WCOJ join. The bare base label
    is relational-UF with the pure-engine rebuild.
    """
    for backend, bflags, term_only in BACKENDS:
        for encoding, eflags in ENCODINGS:
            if term_only and encoding == "normal":
                # Degenerate: same engine as this backend's term-encoding cell.
                continue
            if term_only and encoding == "term-encoding":
                # Backend flag already implies term encoding; don't re-pass it.
                base_flags = list(bflags)
            else:
                base_flags = bflags + eflags
            base_label = f"{backend}-{encoding}"

            # A term/proof cell exercises the union-find encoding, so the
            # native-UF and fast-rebuild axes apply to it. The bridge-normal
            # oracle (and any other non-term cell) has no UF axes.
            term_like = encoding in ("term-encoding", "proofs")

            if not term_like:
                # Non-term cell (the bridge-normal oracle): a single plain cell.
                yield (base_label, backend, encoding, list(base_flags), False, False)
                continue

            # native-UF + proofs validates on bridge, feldera, AND flowlog (proofs
            # are tracked at the encoding level, independent of the host-pass
            # rebuild -- verified via --proof-testing). Only duckdb can't build
            # proof-mode native-UF functions, so exclude its +nuf-on-proofs cells.
            nuf_ok = (encoding == "term-encoding") or (backend != "duckdb")

            # The four rebuild cells (suffix, native_uf, fast_rebuild). All four
            # are real, distinct, and bit-exact. +fastrb / +nuf+fastrb add the
            # `--fast-rebuild` CLI flag for EVERY backend (bridge included).
            rebuild_cells = [
                ("", False, False),               # plain: relational UF, pure-engine rebuild
                ("+fastrb", False, True),         # relational UF, specialized rebuild
            ]
            if nuf_ok:
                rebuild_cells += [
                    ("+nuf", True, False),         # native UF, pure-engine rebuild
                    ("+nuf+fastrb", True, True),   # native UF, specialized rebuild
                ]

            # WCOJ cross-product: flowlog ONLY (other backends reject --wcoj), now
            # crossed into BOTH term-encoding and proofs cells. proofs+wcoj is
            # validated sound (the WCOJ join path preserves proof terms -- verified
            # via --proof-testing on a cyclic-CQ-with-check file); the eval measures
            # --proofs perf and tests/files.rs validates proof correctness.
            if backend == "flowlog":
                wcoj_variants = [("", []), ("+wcoj", ["--wcoj"])]
            else:
                wcoj_variants = [("", [])]

            for suffix, native_uf, fast_rebuild in rebuild_cells:
                flags = list(base_flags)
                if native_uf:
                    flags.append("--native-uf")
                if fast_rebuild:
                    flags.append("--fast-rebuild")
                for wcoj_suffix, wcoj_flags in wcoj_variants:
                    yield (f"{base_label}{suffix}{wcoj_suffix}", backend, encoding,
                           flags + wcoj_flags, native_uf, fast_rebuild)


class BenchDB:
    """Minimal results database, serialized to the JSON shape eval-live reads:
    {"timings": [...], "errors": [...]}."""

    def __init__(self, timing_mode="unknown"):
        self.timings = []
        self.errors = []
        # Files excluded up front (not run at all) because they don't run under
        # bridge-normal or aren't supported by the term encoding. Kept separate
        # from `errors` so the errors table shows only real backend failures on
        # supported files, not noise from whole files we chose not to benchmark.
        self.skipped = []
        # Cell-level WARNINGS: a (condition, file) that failed for an EXPECTED,
        # non-bug reason -- a timeout, or a backend/encoding feature the file uses
        # that isn't supported. Separate from `errors` (real, unexpected bugs);
        # surfaced as a warnings table.
        self.warnings = []
        # Provenance: "paper-sequential" (accurate, uncontended) vs
        # "parallel-Njobs" (fast coverage, contended -> inflated wall-times).
        self.timing_mode = timing_mode

    def add_timing(self, benchmark, backend, mode, condition, timing_list,
                   rss=None, phases=None, parity=None, parity_diff=None,
                   sizes=None, command=None):
        row = {
            "benchmark": benchmark,
            "suite": suite_of(benchmark),
            "backend": backend,
            "mode": mode,
            "condition": condition,
            "command": command,
            "timing_list": timing_list,
        }
        # Peak resident set size in BYTES (max across timed runs), via
        # `/usr/bin/time -l`.
        if rss is not None:
            row["rss"] = rss
        # Per-phase rebuild/canonicalize/congruence bucket seconds (and the raw
        # per-source profile under "raw").
        if phases is not None:
            row["phases"] = phases
        # Tuple-count parity vs the bridge-normal oracle.
        if parity is not None:
            row["parity"] = parity
        if parity_diff:
            row["parity_diff"] = parity_diff
        # The captured per-function (Name, size) counts (sorted list of pairs).
        if sizes is not None:
            row["sizes"] = sizes
        self.timings.append(row)

    def add_error(self, benchmark, backend, mode, condition, error, command=None):
        self.errors.append({
            "benchmark": benchmark,
            "suite": suite_of(benchmark),
            "backend": backend,
            "mode": mode,
            "condition": condition,
            "command": command,
            "error": error,
        })

    def add_skip(self, benchmark, reason):
        self.skipped.append({"benchmark": benchmark, "suite": suite_of(benchmark),
                             "reason": reason})

    def add_warning(self, benchmark, backend, mode, condition, reason):
        # A cell that failed for an EXPECTED, non-bug reason: a timeout, or a
        # feature the backend/encoding does not support (push/pop, proofs-
        # incompatible commands, etc.). Kept OUT of `errors` so that table shows
        # only unexpected failures (real bugs); surfaced as a warnings table.
        self.warnings.append({
            "benchmark": benchmark, "suite": suite_of(benchmark),
            "backend": backend, "mode": mode,
            "condition": condition, "reason": reason,
        })

    def to_dict(self):
        return {"timing_mode": self.timing_mode,
                "timings": self.timings, "errors": self.errors,
                "skipped": self.skipped, "warnings": self.warnings}

    def save_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def build_egglog(build: bool, release: bool) -> Path:
    """Build and return the `egglog-experimental` CLI binary. The matrix drives
    egglog-experimental (NOT mainline egglog): it registers the experimental
    surface then dispatches to `egglog::cli`, so it carries `--native-uf` + the
    backend flags + the fast-rebuild env vars, and its `src/main.rs` pre-builds
    the duckdb egraph in native-UF mode for `--native-uf --duckdb`.

    There is no prebuilt experimental binary -- it must be compiled fresh at the
    current commit (which includes the main.rs `--native-uf` fix). We run
    exactly ONE serial `cargo build -p egglog-experimental` and use the freshly
    built `target/{release,debug}/egglog-experimental`. (`build` is accepted for
    backwards compatibility but the build always happens.)"""
    del build  # build is unconditional: no prebuilt experimental binary exists.
    profile = ["--release"] if release else []
    print(f"Building egglog-experimental ({'release' if release else 'debug'})...", flush=True)
    subprocess.run(
        ["cargo", "build", *profile,
         "-p", "egglog-experimental", "--bin", "egglog-experimental"],
        cwd=WORKSPACE, check=True,
    )
    target = "release" if release else "debug"
    binary = WORKSPACE / "target" / target / "egglog-experimental"
    if not binary.exists():
        sys.exit(f"egglog-experimental binary not found at {binary}")
    return binary


def find_benchmarks(path: Path) -> list[Path]:
    # A `.tar.zst` (e.g. the Herbie dumps) is extracted once to a sibling
    # `<name>.extracted/` dir and benchmarked from there. We benchmark EVERY
    # file (no whole-file skips): heavy files like math-microbenchmark work for
    # the fast treatments (normal 0.5s, term 8s, duckdb 39s); only the dataflow
    # backends blow up, and those individual cells just hit the per-cell timeout.
    if path.is_file() and path.name.endswith(".tar.zst"):
        dest = path.with_name(path.name.replace(".tar.zst", "") + ".extracted")
        if not dest.exists():
            print(f"Extracting {path.name} -> {dest} ...", flush=True)
            dest.mkdir(parents=True)
            subprocess.run(["tar", "-xf", str(path), "-C", str(dest)], check=True)
        return sorted(dest.rglob("*.egg"))
    if path.is_file():
        return [path]
    # Exclude fail-typecheck negative tests: they are EXPECTED to fail (mirrors
    # tests/files.rs `should_fail()` = path contains "fail-typecheck"), so they
    # are not benchmarks -- running them only adds expected-failure noise. (Do
    # NOT filter by the bare "fail" substring: container-fail.egg and
    # repro-small-rebuild-fail-term-encoding.egg are positive tests.)
    # Also drop leaked eval temp files: `measure_sizes` / extract-strip write
    # `tmp<8 chars>.egg` into the bench dir and unlink them on clean exit, but a
    # KILLED run can leak them into the corpus.
    return sorted(p for p in path.rglob("*.egg")
                  if "fail-typecheck" not in str(p)
                  and not re.fullmatch(r"tmp[a-z0-9_]{8}\.egg", p.name))


def suite_of(rel: str) -> str:
    """Group a benchmark by corpus 'suite' for the results tables/filters.
    `tests/` and `egglog-experimental/tests/` are the 'egglog' suite; each paper
    corpus keeps its own suite (herbie / math-microbenchmark / pointer-analysis)."""
    if "paper-benchmarks/herbie" in rel:
        return "herbie"
    if "paper-benchmarks/math-microbenchmark" in rel:
        return "math-microbenchmark"
    if "paper-benchmarks/pointer-analysis" in rel:
        return "pointer-analysis"
    if "paper-benchmarks" in rel:
        return "paper"
    return "egglog"


# --- Per-phase profile: bucket per-ruleset/per-class times into a uniform
# {rebuild, canonicalize, congruence, other} schema across backends. ---

def _phase_bucket(name: str) -> str:
    """Map a ruleset/class name to one of the uniform phase buckets."""
    n = name.lower()
    if "rebuild" in n:
        return "rebuild"
    if "congruence" in n:
        return "congruence"
    # UF maintenance / canonicalization rulesets.
    if any(k in n for k in ("canon", "uf", "single_parent", "singleparent",
                            "path_compress", "maintenance", "demote")):
        return "canonicalize"
    return "other"


def parse_duck_phases(stderr: str):
    """Parse the `--- by class ---` block from the duckdb DUCK_PERF_DUMP into
    {class: {"search": s, "apply": s}}. Returns {} if absent."""
    phases = {}
    in_block = False
    for line in stderr.splitlines():
        if "by class" in line:
            in_block = True
            continue
        if not in_block:
            continue
        # Format: "    1.364s  search   1.151s  apply   0.212s   42.0%wall  rebuild"
        if "search" in line and "apply" in line and line.strip():
            parts = line.split()
            try:
                kind = parts[-1]
                search = float(parts[2].rstrip("s"))
                apply_ = float(parts[4].rstrip("s"))
                phases[kind] = {"search": search, "apply": apply_}
            except (IndexError, ValueError):
                in_block = False
        else:
            in_block = False
    return phases


def parse_bridge_report(report_path: Path):
    """Load a bridge `--save-report` RunReport JSON and reduce it to per-ruleset
    {search_and_apply, merge, rebuild} seconds. Returns {} on any failure."""
    try:
        rep = json.loads(report_path.read_text())
    except (OSError, ValueError):
        return {}

    def dur(d):
        return d.get("secs", 0) + d.get("nanos", 0) / 1e9 if d else 0.0

    sa = rep.get("search_and_apply_time_per_ruleset", {})
    mg = rep.get("merge_time_per_ruleset", {})
    rb = rep.get("rebuild_time_per_ruleset", {})
    out = {}
    for rs in set(sa) | set(mg) | set(rb):
        out[rs] = {"search_and_apply": dur(sa.get(rs)),
                   "merge": dur(mg.get(rs)), "rebuild": dur(rb.get(rs))}
    return out


def parse_flowlog_prof(stderr: str):
    """Parse the FLOWLOG_DD_RULESET_PROF per-ruleset table into
    {ruleset: total_seconds}. Returns {} if absent."""
    phases = {}
    in_table = False
    for line in stderr.splitlines():
        if line.startswith("ruleset") and "total" in line and "%dd" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("[FLOWLOG_DD_RULESET_PROF]") or not line.strip():
            in_table = False
            continue
        # "canonicalize                     25    0.023s  32.9% ..."
        parts = line.split()
        if len(parts) >= 3:
            try:
                total = float(parts[2].rstrip("s"))
                phases[parts[0]] = total
            except ValueError:
                in_table = False
    return phases


def parse_feldera_prof(stderr: str):
    """Parse the feldera FELDERA_PROFILE `[PROF]` lines into coarse phase
    seconds. Feldera has no per-ruleset rebuild/canonicalize/congruence split,
    so we report what it does expose (circuit_step / read_clone / fed_diff) and
    leave the bucketed split as a TODO. Returns {} if absent."""
    phases = {}
    for line in stderr.splitlines():
        if not line.startswith("[PROF]"):
            continue
        for key in ("circuit_step", "read_clone", "fed_diff", "step(transaction)",
                    "feed", "read(consolidate)"):
            m = re.search(re.escape(key) + r"=([0-9.]+)s", line)
            if m:
                phases[key.replace("(transaction)", "").replace("(consolidate)", "")] = \
                    float(m.group(1))
    # TODO(feldera per-phase): expose a per-ruleset rebuild/canonicalize/
    # congruence split from the DBSP circuit; FELDERA_PROFILE is circuit-global.
    return phases


def bucket_phases(backend: str, raw: dict) -> dict:
    """Fold a backend's raw per-source profile into the uniform
    {rebuild, canonicalize, congruence, other} seconds schema, plus keep the
    raw map under "raw" for drill-down. Returns {} if there is nothing."""
    if not raw:
        return {}
    buckets = {"rebuild": 0.0, "canonicalize": 0.0, "congruence": 0.0, "other": 0.0}
    if backend == "duckdb":
        # {class: {"search", "apply"}} -> bucket by class name (search+apply).
        for cls, st in raw.items():
            buckets[_phase_bucket(cls)] += st.get("search", 0) + st.get("apply", 0)
    elif backend == "bridge":
        # {ruleset: {search_and_apply, merge, rebuild}} -> rebuild explicit;
        # search_and_apply+merge bucketed by ruleset name.
        for rs, st in raw.items():
            buckets["rebuild"] += st.get("rebuild", 0)
            other = st.get("search_and_apply", 0) + st.get("merge", 0)
            b = _phase_bucket(rs)
            buckets[b if b != "rebuild" else "other"] += other
    elif backend == "flowlog":
        # {ruleset: total_seconds} -> bucket by ruleset name.
        for rs, total in raw.items():
            buckets[_phase_bucket(rs)] += total
    elif backend == "feldera":
        # No per-ruleset split; circuit_step is the closest to "apply" work.
        buckets["other"] = sum(raw.values())
    buckets = {k: round(v, 6) for k, v in buckets.items() if v}
    if not buckets:
        return {}
    buckets["raw"] = raw
    return buckets


# --- Correctness: per-function tuple-count parity vs the bridge-normal oracle.

# `(print-size)` (no-arg) emits ONE s-expression `((name size) (name size) ...)`,
# appended last. We parse the output into structured s-exprs and take the LAST
# top-level block that is a list of `(name integer)` pairs -- the tests/files.rs
# approach (structured output, NOT a text/regex scrape). This handles any egglog
# symbol name (`Sound-/`, `bad-merge?`, `bop->string`, `List<PtrPointees`, ...);
# a regex name charset previously dropped the whole block for those -> vacuous
# empty==empty parity. Stray warning sexps (e.g. `(let @v28 (R 0))`, whose
# top-level elements are not all `(name int)` pairs) can never qualify, so they
# cannot clobber the real block or pass vacuously.


def _sexpr_blocks(text: str):
    """Parse `text` into its top-level s-expressions (nested lists of str atoms),
    tolerating interleaved non-sexpr noise. Respects "..." strings and ; comments."""
    toks = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
        elif c == ";":
            j = text.find("\n", i)
            i = n if j == -1 else j + 1
        elif c in "()":
            toks.append(c)
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            toks.append(text[i:min(j + 1, n)])
            i = min(j + 1, n)
        else:
            j = i
            while j < n and text[j] not in ' \t\r\n();"':
                j += 1
            toks.append(text[i:j])
            i = j
    exprs, stack, cur = [], [], None
    for t in toks:
        if t == "(":
            new = []
            if cur is not None:
                cur.append(new)
                stack.append(cur)
            cur = new
        elif t == ")":
            if cur is None:
                continue  # unbalanced ')' -- ignore
            if stack:
                cur = stack.pop()
            else:
                exprs.append(cur)
                cur = None
        elif cur is not None:
            cur.append(t)
        # top-level atoms (cur is None) are ignored
    return exprs


def parse_sizes(text: str):
    """Extract the appended `(print-size)` block's `(name size)` counts via a
    STRUCTURED s-expr parse (no regex name-charset assumptions; mirrors
    tests/files.rs's structured output). Returns the LAST top-level block that is
    a list of `(name integer)` pairs, as a sorted [name, count] list, or []."""
    def _is_int(s):
        return isinstance(s, str) and s.lstrip("-").isdigit()

    best = None
    for e in _sexpr_blocks(text):
        if isinstance(e, list) and e and all(
            isinstance(p, list) and len(p) == 2 and isinstance(p[0], str) and _is_int(p[1])
            for p in e
        ):
            best = e  # print-size is appended last -> keep the last qualifying block
    if best is None:
        return []
    sizes = {p[0]: int(p[1]) for p in best}
    return sorted([name, n] for name, n in sizes.items())


def run_once(binary: Path, flags: list[str], bench: Path, timeout: float,
             env_extra: dict | None = None, capture_phases_for: str | None = None):
    """Run one egglog invocation wrapped in `/usr/bin/time -l` for peak RSS.

    Returns a dict {elapsed, rss, stdout, stderr, phases} on success, or
    {error: msg} on failure/timeout. `env_extra` adds env vars (fast-rebuild /
    profile flags). `capture_phases_for` is the backend whose per-phase profile
    to parse from stderr (and, for bridge, the `--save-report` file)."""
    env = dict(os.environ, **(env_extra or {}))

    report_file = None
    cmd_flags = list(flags)
    if capture_phases_for == "bridge":
        report_file = tempfile.NamedTemporaryFile(
            suffix=".report.json", delete=False)
        report_file.close()
        cmd_flags += ["--save-report", report_file.name]

    # Wrap with macOS `/usr/bin/time -l` for peak RSS (bytes). It prints the
    # resource summary to ITS stderr, which we separate from the program's.
    cmd = ["/usr/bin/time", "-l", str(binary), *cmd_flags, str(bench)]
    start = time.perf_counter()
    # `start_new_session=True` puts `/usr/bin/time` AND its `egglog` child in a
    # fresh process group so a timeout can SIGKILL the WHOLE tree. Without it,
    # the timeout kills only `/usr/bin/time` and the `egglog` grandchild orphans
    # + keeps running — piling up across a full run's many timeouts.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.communicate(timeout=10)  # reap the killed group
        except subprocess.TimeoutExpired:
            pass
        if report_file:
            Path(report_file.name).unlink(missing_ok=True)
        return {"error": f"timeout after {timeout}s"}
    elapsed = time.perf_counter() - start

    # `/usr/bin/time -l` appends its resource block to stderr. Split it off so
    # the program's own stderr (and parity output) stays clean.
    prog_stderr, rss = _split_time_block(stderr or "")

    if proc.returncode != 0:
        if report_file:
            Path(report_file.name).unlink(missing_ok=True)
        msg = _failure_msg((prog_stderr or "") + "\n" + (stdout or ""),
                           proc.returncode)
        return {"error": f"exit {proc.returncode}: {msg}"}

    phases_raw = {}
    if capture_phases_for == "duckdb":
        phases_raw = parse_duck_phases(prog_stderr)
    elif capture_phases_for == "flowlog":
        phases_raw = parse_flowlog_prof(prog_stderr)
    elif capture_phases_for == "feldera":
        phases_raw = parse_feldera_prof(prog_stderr)
    elif capture_phases_for == "bridge" and report_file:
        phases_raw = parse_bridge_report(Path(report_file.name))
        Path(report_file.name).unlink(missing_ok=True)

    return {"elapsed": elapsed, "rss": rss,
            "stdout": stdout or "", "stderr": prog_stderr,
            "phases_raw": phases_raw}


_TIME_RSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M)
# The `/usr/bin/time -l` resource block starts with a "<real> <user> <sys>" line.
_TIME_HEADER_RE = re.compile(r"^\s*[\d.]+ real\s+[\d.]+ user\s+[\d.]+ sys", re.M)


def _split_time_block(stderr: str):
    """Separate the trailing `/usr/bin/time -l` resource block from the
    program's own stderr. Returns (program_stderr, rss_bytes_or_None)."""
    rss = None
    m = _TIME_RSS_RE.search(stderr)
    if m:
        rss = int(m.group(1))
    # Drop everything from the resource header line onward.
    h = _TIME_HEADER_RE.search(stderr)
    prog = stderr[:h.start()] if h else stderr
    return prog, rss


def measure_sizes(binary, bench, flags, env_extra, timeout):
    """Run the cell ONCE with `(print-size)` appended to capture per-function
    tuple counts (for parity). Returns (sizes_list, None) or (None, error)."""
    # Strip the same way bench_cell does, so the support GATE (which calls
    # measure_sizes directly on the raw file) tests the SAME program the timed
    # runs do. Otherwise Herbie files gate out on un-stripped multi-extract /
    # scheduler under --term-encoding even though the stripped program runs.
    src = strip_back_off_scheduler(strip_extract_commands(bench.read_text()))
    tmp = tempfile.NamedTemporaryFile(
        suffix=".egg", delete=False, mode="w", dir=str(bench.parent))
    try:
        tmp.write(src)
        tmp.write("\n(print-size)\n")
        tmp.close()
        out = run_once(binary, flags, Path(tmp.name), timeout, env_extra=env_extra)
        if "error" in out:
            return None, out["error"]
        # print-size output can land on stdout or stderr depending on backend;
        # parse BOTH.
        sizes = parse_sizes(out["stdout"] + "\n" + out["stderr"])
        return sizes, None
    finally:
        # tolerate a cleanup race: parallel cells + the binary's own --proofs
        # runs can leave/remove tmp*.egg in the dir, so the temp may be gone
        Path(tmp.name).unlink(missing_ok=True)


_HEAD_ATOM_RE = re.compile(r"[^\s()\";]+")


def _skip_string(src, i):
    """src[i] == '\"'; return the index just past the closing quote."""
    j, n = i + 1, len(src)
    while j < n:
        if src[j] == "\\":
            j += 2
            continue
        if src[j] == '"':
            return j + 1
        j += 1
    return n


def _form_end(src, i):
    """src[i] == '('; return the index just past the matching ')'."""
    depth, j, n = 0, i, len(src)
    while j < n:
        c = src[j]
        if c == '"':
            j = _skip_string(src, j)
            continue
        if c == ";":
            k = src.find("\n", j)
            j = n if k == -1 else k
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return n


def _head_atom(form):
    """First token after the opening '(' of a form (the command head)."""
    m = _HEAD_ATOM_RE.match(form[1:].lstrip())
    return m.group(0) if m else ""


def strip_extract_commands(src):
    """Remove top-level `(extract ...)` and `(multi-extract ...)` commands. Both
    are OUTPUT-ONLY -- they print extracted term(s) and do NOT modify the e-graph,
    so removing them leaves tuple counts (parity) unchanged. Several
    backends/encodings can't run extraction (feldera/flowlog, duckdb
    extract-of-expr, proofs), so the eval strips it to measure eqsat perf instead
    of erroring on output. Herbie dumps end in `multi-extract` (1215/1260 files)
    and use no plain `extract`, so stripping multi-extract is what unblocks them.
    Respects `;` comments and `"..."` string literals; only TOP-LEVEL forms go."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == ";":
            j = src.find("\n", i)
            j = n if j == -1 else j + 1
            out.append(src[i:j])
            i = j
        elif c == '"':
            j = _skip_string(src, i)
            out.append(src[i:j])
            i = j
        elif c == "(":
            j = _form_end(src, i)
            if _head_atom(src[i:j]) in ("extract", "multi-extract"):
                if j < n and src[j] == "\n":   # swallow the trailing newline
                    j += 1
            else:
                out.append(src[i:j])
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _peek_head(src, i, n):
    """src[i] == '('; return (head-atom token, index just past it)."""
    k = i + 1
    while k < n and src[k] in " \t\n":
        k += 1
    e = k
    while e < n and src[e] not in " \t\n()\";":
        e += 1
    return src[k:e], e


def strip_back_off_scheduler(src):
    """Rewrite Herbie's back-off-scheduler run-schedules into scheduler-free ones:
      `(let-scheduler NAME (back-off))`  -> removed
      `(run-with NAME RULESET ...)`      -> `(run RULESET ...)`
    The `repeat`/`saturate`/`:until` structure is preserved, so the iteration
    count is unchanged -- only the back-off rule-selection heuristic is dropped
    (the scheduler-free run-schedule then lowers to core, which every backend
    runs uniformly). Herbie only ever uses `(back-off)` schedulers (620/1260
    files), so stripping every `let-scheduler` + `run-with` is safe here; this is
    the "drop the back-off scheduler, same iters" treatment for the paper.
    Comment/string-aware; only those two heads are touched."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == ";":
            j = src.find("\n", i)
            j = n if j == -1 else j + 1
            out.append(src[i:j])
            i = j
        elif c == '"':
            j = _skip_string(src, i)
            out.append(src[i:j])
            i = j
        elif c == "(":
            head, he = _peek_head(src, i, n)
            if head == "let-scheduler":
                i = _form_end(src, i)          # drop the scheduler definition
            elif head == "run-with":
                out.append("(run")             # `(run-with NAME ...` -> `(run ...`
                p = he
                while p < n and src[p] in " \t\n":
                    p += 1
                while p < n and src[p] not in " \t\n()\";":
                    p += 1                      # skip the scheduler-name token
                i = p
            else:
                out.append("(")
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def bench_cell(binary, bench, rel, condition, backend, mode, flags,
               native_uf, fast_rebuild, warmup, runs, timeout):
    """Run one matrix cell: `warmup` discarded runs, `runs` timed runs (each
    wrapped for peak RSS + per-phase profile), and one extra `(print-size)` run
    for tuple-count parity. Returns a result dict folded into the DB. A failing
    cell returns an error result rather than raising."""
    # Assemble the cell's env. `rebuild_env` is the treatment-relevant env
    # (fast-rebuild knob) that MUST be present for both the timed runs and the
    # parity run. `prof_env` adds the per-phase profile dump, which is enabled
    # ONLY for the timed runs -- the profile block pollutes stderr that the
    # `(print-size)` parity run scrapes (e.g. duckdb's per-rule dump can emit a
    # stray `(Const 2)`), so the parity run must run profile-free.
    rebuild_env = {}
    if fast_rebuild:
        knob = FAST_REBUILD_ENV.get(backend)
        if knob:
            rebuild_env[knob] = "1"
    env_extra = dict(rebuild_env)
    prof_env = PROFILE_ENV.get(backend)
    if prof_env:
        env_extra[prof_env] = "1"

    # Reproducible invocation for this cell: the treatment env knob (e.g.
    # DUCK_DELTA_REBUILD=1 for +fastrb) + flags + file, exactly as the timed
    # runs invoke it (no profile env, no print-size). Stored per row so the
    # table is self-documenting -- e.g. it shows that +fastrb does NOT pass
    # --native-uf (fast-rebuild is relational-UF + an env knob; native-UF is a
    # separate, mutually exclusive treatment that adds --native-uf).
    command = " ".join(filter(None, [
        " ".join(f"{k}={v}" for k, v in sorted(rebuild_env.items())),
        Path(binary).name, *flags, rel]))

    # Strip top-level `(extract ...)` commands: they are OUTPUT-ONLY (don't
    # change the e-graph, so tuple-count parity is unchanged) and several
    # backends/encodings can't run extraction (feldera/flowlog, duckdb
    # extract-of-expr, proofs). Run a stripped copy so the cell measures eqsat
    # perf instead of erroring on output. Applies to EVERY cell (incl. the
    # bridge-normal reference), so parity stays apples-to-apples.
    src = bench.read_text()
    stripped = strip_back_off_scheduler(strip_extract_commands(src))
    tmp_bench = None
    if stripped != src:
        st = tempfile.NamedTemporaryFile(suffix=".egg", delete=False, mode="w",
                                         dir=str(bench.parent))
        st.write(stripped)
        st.close()
        tmp_bench = Path(st.name)
        bench = tmp_bench

    try:
        # Warm-up runs (discarded): pay one-time costs (page cache, etc.). Warm
        # the treatment's actual rebuild path; skip the profile (warmup untimed).
        for _ in range(warmup):
            run_once(binary, flags, bench, timeout, env_extra=rebuild_env)

        timings = []
        rss_vals = []
        phase_runs = []
        for _ in range(runs):
            out = run_once(binary, flags, bench, timeout, env_extra=env_extra,
                           capture_phases_for=backend)
            if "error" in out:
                return {"benchmark": rel, "backend": backend, "mode": mode,
                        "condition": condition, "command": command,
                        "error": out["error"]}
            timings.append(round(out["elapsed"], 6))
            if out.get("rss"):
                rss_vals.append(out["rss"])
            if out.get("phases_raw"):
                phase_runs.append(out["phases_raw"])

        result = {"benchmark": rel, "backend": backend, "mode": mode,
                  "condition": condition, "command": command,
                  "timing_list": timings}
        if rss_vals:
            # Peak RSS = max observed across timed runs (bytes).
            result["rss"] = max(rss_vals)

        if phase_runs:
            # Bucket the median run's raw profile into the uniform schema.
            mid = phase_runs[len(phase_runs) // 2]
            buckets = bucket_phases(backend, mid)
            if buckets:
                result["phases"] = buckets

        # Tuple-count parity: one extra run with `(print-size)` appended. Use the
        # profile-free env (the fast-rebuild knob still applies; the profile dump
        # does NOT, so it can't pollute the scraped size output).
        sizes, size_err = measure_sizes(binary, bench, flags, rebuild_env, timeout)
        if size_err is None and sizes is not None:
            result["sizes"] = sizes
        return result
    finally:
        if tmp_bench is not None:
            tmp_bench.unlink(missing_ok=True)


def _failure_msg(text, returncode):
    """Best failure line from a cell's combined stderr/stdout. Prefers the Rust
    panic MESSAGE (not the trailing 'note: run with RUST_BACKTRACE' line) and
    egglog `[ERROR]` lines, so the stored error is classifiable -- the last line
    alone is often useless."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    if not lines:
        return f"exit code {returncode}"
    for i, l in enumerate(lines):
        if "panicked at" in l:
            m = re.search(r"panicked at [^:]*:\d+:\d+:\s*(\S.*)", l)
            if m:
                return m.group(1)
            return lines[i + 1] if i + 1 < len(lines) else l
    for i, l in enumerate(lines):
        if "[ERROR]" in l:
            # Keep the "Offending command:" detail if it follows, so the stored
            # message names WHICH command (and stays classifiable).
            for l2 in lines[i + 1:i + 4]:
                if "Offending command" in l2:
                    return f"{l} {l2.strip()}"
            return l
    return lines[-1]


def classify_unsupported(error_text):
    """Return a short reason if `error_text` is a KNOWN backend/encoding
    capability gap (-> recorded under `warnings`, not `errors`), else None (a
    real, unexpected failure that stays in `errors`). `extract` is stripped
    before running and missing primitives are bugs to FIX, so neither is
    reclassified here."""
    t = error_text or ""
    if "timeout" in t.lower():
        return "timeout (too slow at the cell timeout)"
    if "Unrecognized user-defined command" in t:
        return "command unsupported on this backend (e.g. run-schedule)"
    if "clone_boxed" in t:
        return "push/pop unsupported (clone_boxed deferred)"
    # proofs-encoding rejections. The proof-term encoder gates unsupported
    # commands (run-schedule, :no-merge/:unextractable functions, multi-extract).
    # With the [ERROR]-preferring capture the message reads "...not supported by
    # the current proof term encoding impl"; the "Offending command:" detail may
    # or may not be present -- match either form.
    if ("proof term encoding" in t or "Offending command" in t
            or ":no-merge" in t or ":unextractable" in t):
        return "proofs encoding: command unsupported"
    if "does not yet support proof-mode native-UF" in t:
        return "proofs + native-UF unsupported (duckdb)"
    return None


def apply_cell_result(result, db, oracle_sizes):
    """Fold a cell result into the DB and log it. `oracle_sizes` is
    {(benchmark, encoding-family): sorted [name, count] list}; a cell's parity
    is its size-set vs the bridge reference of ITS OWN encoding family."""
    rel = result["benchmark"]
    backend = result["backend"]
    mode = result["mode"]
    condition = result["condition"]
    if "error" in result:
        reason = classify_unsupported(result["error"])
        if reason is not None:
            db.add_warning(rel, backend, mode, condition, reason)
            print(f"    {rel}: {condition:28} warning ({reason})", flush=True)
        else:
            db.add_error(rel, backend, mode, condition, result["error"],
                         command=result.get("command"))
            print(f"    {rel}: {condition:28} ERROR: {result['error']}", flush=True)
        return

    timings = result["timing_list"]
    rss = result.get("rss")
    phases = result.get("phases")
    sizes = result.get("sizes")

    # Correctness: compare against the bridge reference of THIS cell's encoding
    # family. Non-proof treatments (term/native-uf/fast-rebuild) reproduce
    # normal-mode counts, so they target bridge-normal (files.rs shared
    # snapshot); proofs target bridge-proofs.
    parity = None
    parity_diff = None
    ref_label = REFERENCE_BY_FAMILY.get(_encoding_family(condition))
    oracle = oracle_sizes.get((rel, ref_label))
    # Require NON-EMPTY sizes on both sides: an unparseable `(print-size)` block
    # returns [], and `[] == []` would be a vacuous false-pass (parity=True with
    # nothing compared). Leave parity=None when there are no real counts to
    # compare -- mirrors files.rs, which refuses to assert on an empty snapshot.
    if sizes:
        if condition == ref_label:
            parity = True  # this cell IS its family's reference
        elif oracle:
            parity = (sizes == oracle)
            if not parity:
                parity_diff = _size_diff(oracle, sizes)

    db.add_timing(rel, backend, mode, condition, timings,
                  rss=rss, phases=phases, parity=parity,
                  parity_diff=parity_diff, sizes=sizes,
                  command=result.get("command"))

    mean = sum(timings) / len(timings)
    rss_mb = f"{rss / 1e6:7.1f}MB" if rss else "      ?MB"
    par = "" if parity is None else (" parity=OK" if parity else " parity=MISMATCH")
    print(f"    {rel}: {condition:28} mean {mean:8.3f}s  rss {rss_mb}{par}",
          flush=True)
    if phases:
        parts = [f"{k}={phases[k]:.3f}s" for k in
                 ("rebuild", "canonicalize", "congruence", "other")
                 if k in phases]
        if parts:
            print(f"      phases: {'  '.join(parts)}", flush=True)
    if parity_diff:
        print(f"      parity diff: {parity_diff}", flush=True)


def _size_diff(oracle, cell):
    """Human-readable diff of two sorted [name, count] size-sets."""
    od = dict(oracle)
    cd = dict(cell)
    diffs = []
    for name in sorted(set(od) | set(cd)):
        o = od.get(name)
        c = cd.get(name)
        if o != c:
            diffs.append(f"{name}: oracle={o} cell={c}")
    return "; ".join(diffs)


def render_results(results_path: Path, outdir: Path, fmt: str = "pdf"):
    """Render the graphs.py registry to `outdir` as `fmt` (PDF) files + tables
    as CSV, using LOCAL matplotlib (no browser/Pyodide). The on-disk
    counterpart to serve_results -- what we use to generate paper figures."""
    import importlib.util
    import eval_live

    graph_script_path = Path(__file__).resolve().parent / "graphs.py"
    if not graph_script_path.exists():
        sys.exit(f"no graphs.py at {graph_script_path}")
    # Executing graphs.py builds a Registry and assigns `eval_live.registry`.
    spec = importlib.util.spec_from_file_location("eval_graphs", graph_script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    reg = getattr(eval_live, "registry", None)
    if reg is None:
        sys.exit("graphs.py did not set eval_live.registry")

    data = json.loads(results_path.read_text())
    written = reg.render_to_dir(data, str(outdir), fmt=fmt)
    print(f"Rendered {len(written)} files to {outdir}/")
    for p in written:
        print(f"  {p}")


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
    parser.add_argument("--jobs", type=int, default=0,
                        help="concurrent cells (default 0 = half the CPU cores, for "
                             "memory headroom; the normal dev mode). Parity is exact "
                             "regardless of contention; "
                             "parallel wall-clock is approximate. `--paper` forces "
                             "sequential (jobs=1) for the final uncontended timings.")
    parser.add_argument("--warmup", type=int, default=1, help="discarded warm-up runs (default 1)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="per-run timeout in seconds (default 60)")
    parser.add_argument("--limit", type=int, default=None,
                        help="benchmark at most N files (small pilot / sampling)")
    parser.add_argument("--output", default=str(WORKSPACE / "eval" / "results.json"),
                        help="results JSON path")
    parser.add_argument("--debug", action="store_true",
                        help="use the debug build instead of release")
    parser.add_argument("--build", action="store_true",
                        help="accepted for backwards compatibility; the "
                             "egglog-experimental binary is always built fresh "
                             "(ONE serial `cargo build -p egglog-experimental`) "
                             "since there is no prebuilt experimental binary.")
    parser.add_argument("--paper", action="store_true",
                        help="PAPER MODE: force strictly SEQUENTIAL (jobs=1) for "
                             "uncontended, paper-quality timings/RSS. Use for the "
                             "final eval; otherwise runs parallel (all cores).")
    parser.add_argument("--serve", action="store_true", help="open the eval-live viewer after running")
    parser.add_argument("--justserve", action="store_true", help="skip benchmarking; just serve results")
    parser.add_argument("--render", metavar="DIR", default=None,
                        help="skip benchmarking; render graphs (PDF) + tables "
                             "(CSV) from the results JSON into DIR and exit")
    parser.add_argument("--render-format", default="pdf",
                        help="format for --render graphs (default pdf)")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.render:
        render_results(Path(args.output), Path(args.render), fmt=args.render_format)
        return

    if args.justserve:
        serve_results(Path(args.output), args.port)
        return

    # PARALLEL by default (all CPU cores): the normal dev mode. Parity is exact
    # regardless of contention, and contended wall-clock is a fine approximation
    # while iterating. Only the FINAL paper eval needs clean numbers: `--paper`
    # forces SEQUENTIAL (jobs=1) for uncontended paper-quality wall-clock/RSS.
    if args.paper:
        args.jobs = 1
    elif args.jobs <= 0:
        # Half the cores by default: heavy cells (mmb feldera/duckdb are multi-GB
        # each) blow up memory at full-core concurrency. `--jobs N` to override.
        args.jobs = max(1, (os.cpu_count() or 2) // 2)
    timing_mode = ("paper-sequential" if args.jobs <= 1
                   else f"parallel-{args.jobs}jobs (approx timings)")
    if args.warmup < 1:
        args.warmup = 1

    binary = build_egglog(build=args.build, release=not args.debug)

    bench_path = Path(args.path) if args.path else (WORKSPACE / "tests")
    if not bench_path.is_absolute():
        bench_path = (WORKSPACE / bench_path)
    benchmarks = find_benchmarks(bench_path)
    if not benchmarks:
        sys.exit(f"no .egg benchmarks found under {bench_path}")
    if args.limit is not None:
        benchmarks = benchmarks[:args.limit]

    conds = list(conditions())
    print(f"\n{len(benchmarks)} benchmark(s) x {len(conds)} condition(s), "
          f"{args.runs} run(s) each (warmup {args.warmup}, timeout {args.timeout}s)\n"
          f"timing mode: {timing_mode}  [PAPER (sequential)]\n"
          f"parity reference: bridge-normal (non-proof), "
          f"bridge-proofs (proofs)\n")

    db = BenchDB(timing_mode=timing_mode)

    def rel_of(bench):
        return (str(bench.relative_to(WORKSPACE))
                if str(bench).startswith(str(WORKSPACE)) else str(bench))

    # Per-file references + gating. For each file we run the bridge reference
    # cells (bridge-normal for non-proof treatments, bridge-proofs for proof
    # ones) and store their print-size counts; every other cell is diffed
    # against the reference of ITS OWN family (REFERENCE_BY_FAMILY) -- term /
    # native-uf / fast-rebuild all reproduce normal-mode counts, so they target
    # bridge-normal (tests/files.rs's shared-snapshot model).
    #
    # GATING: a file is benchmarked only if it runs under bridge-normal AND is
    # supported by the term encoding (bridge-term-encoding runs). The whole
    # matrix is built on the term encoding, so a file the term encoder rejects
    # (container/higher-order sorts, non-eq-sort globals, unsupported commands)
    # would only produce a column of errors on every backend -- pure noise. We
    # exclude those up front (recorded under `skipped`, NOT `errors`).
    ref_labels = set(REFERENCE_BY_FAMILY.values())
    TERM_GATE = "bridge-term-encoding"
    probe_conds = [c for c in conds if c[0] in ref_labels or c[0] == TERM_GATE]

    runnable = []        # benchmarks that pass the normal + term-encoding gate
    oracle_sizes = {}    # {(rel, ref_condition_label): sorted [name, count] list}
    print("Establishing references + term-encoding support per file ...",
          flush=True)
    for bench in benchmarks:
        rel = rel_of(bench)
        ran = set()
        for cond_label, backend, mode, flags, _nuf, _fr in probe_conds:
            sizes, err = measure_sizes(binary, bench, flags, {}, args.timeout)
            # Empty sizes = an unparseable (print-size) block; treat like an error
            # so we never establish a vacuous (empty) oracle for this file.
            if err is not None or not sizes:
                continue
            ran.add(cond_label)
            if cond_label in ref_labels:
                oracle_sizes[(rel, cond_label)] = sizes
        if ORACLE_CONDITION not in ran:
            db.add_skip(rel, "does not run under bridge-normal")
            print(f"    SKIP {rel}: bridge-normal errored", flush=True)
        elif TERM_GATE not in ran:
            db.add_skip(rel, "not supported by term encoding")
            print(f"    SKIP {rel}: not supported by term encoding", flush=True)
        else:
            runnable.append(bench)
    db.save_json(args.output)
    print(f"  {len(runnable)}/{len(benchmarks)} files benchmarkable "
          f"({len(db.skipped)} skipped: not normal+term-supported)\n", flush=True)

    # Build the full task list over runnable files: one task per cell.
    tasks = []
    for bench in runnable:
        rel = rel_of(bench)
        for condition, backend, mode, flags, native_uf, fast_rebuild in conds:
            tasks.append((binary, bench, rel, condition, backend, mode, flags,
                          native_uf, fast_rebuild, args.warmup, args.runs,
                          args.timeout))

    if args.jobs > 1:
        # PARALLEL (correctness/coverage only): run cells concurrently. bench_cell
        # shells out, so threads suffice (GIL released during subprocess.run).
        # apply_cell_result + save run in the MAIN thread (as_completed), so the
        # shared db needs no lock; only the timings/RSS are contended.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"PARALLEL: {args.jobs} concurrent cells "
              f"(parity exact; wall-clock approx -- use --paper for final timings)\n",
              flush=True)
        done = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(bench_cell, *t) for t in tasks]
            for fut in as_completed(futures):
                apply_cell_result(fut.result(), db, oracle_sizes)
                done += 1
                if done % 20 == 0:
                    db.save_json(args.output)
        db.save_json(args.output)
    else:
        # SEQUENTIAL: one process at a time (paper-quality).
        last_rel = None
        for i, t in enumerate(tasks, 1):
            rel = t[2]
            if rel != last_rel:
                print(f"[{i}/{len(tasks)}] {rel}", flush=True)
                last_rel = rel
            result = bench_cell(*t)
            apply_cell_result(result, db, oracle_sizes)
            db.save_json(args.output)  # incremental: write after each cell

    print(f"\nResults written to {args.output}  (timing_mode: {timing_mode})")
    if args.serve:
        serve_results(Path(args.output), args.port)


if __name__ == "__main__":
    main()
