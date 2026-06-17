"""Graph and table definitions for the egglog backend eval.

Runs in the browser via Pyodide (eval-live). Each graph fn takes the results
dict ({"timings": [...], "errors": [...]}) and returns a matplotlib Figure;
each table fn returns a list[dict] (rendered as a filterable table).
"""
import eval_live


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _by_bench_cond(data):
    """{benchmark: {condition: mean_time}}."""
    from collections import defaultdict
    out = defaultdict(dict)
    for row in data.get("timings", []):
        out[row["benchmark"]][row["condition"]] = _mean(row.get("timing_list", []))
    return out


def _all_conditions(data):
    conds = []
    for row in data.get("timings", []):
        if row["condition"] not in conds:
            conds.append(row["condition"])
    return sorted(conds)


# Per-benchmark grouped charts become unreadable AND blow past matplotlib's
# 65536px canvas limit past a few dozen benchmarks (the full tests/ corpus is
# ~170). Focus each wide chart on the N largest entries; the full data lives in
# the tables and results.json.
_MAX_BARS = 40
_MAX_FIG_WIDTH_IN = 300  # 300in * 150dpi = 45000px, safely under the 65536 cap


def _top_keys(value_by_cond, limit=_MAX_BARS):
    """The `limit` keys with the largest max-across-conditions value, ordered by
    that value (largest first). Returns all (sorted) when there are <= limit."""
    return sorted(value_by_cond.keys(),
                  key=lambda k: max(value_by_cond[k].values(), default=0),
                  reverse=True)[:limit]


def _fig_width(n_bars, per_bar):
    return min(max(8, n_bars * per_bar), _MAX_FIG_WIDTH_IN)


def mean_time_grouped(data):
    """Grouped bar chart: per benchmark, one bar per condition (mean time)."""
    import matplotlib.pyplot as plt

    by_bench = _by_bench_cond(data)
    conds = _all_conditions(data)
    benches = _top_keys(by_bench)
    n = len(conds)

    fig, ax = plt.subplots(figsize=(_fig_width(len(benches), 1.6), 5))
    width = 0.8 / max(n, 1)
    for j, cond in enumerate(conds):
        xs = [i + j * width for i in range(len(benches))]
        ys = [by_bench[b].get(cond, 0) for b in benches]
        ax.bar(xs, ys, width=width, label=cond)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(benches))])
    ax.set_xticklabels([b.split("/")[-1] for b in benches], rotation=30, ha="right")
    ax.set_ylabel("mean wall-clock (s)")
    extra = f" (top {len(benches)} of {len(by_bench)})" if len(by_bench) > len(benches) else ""
    ax.set_title("Mean time per benchmark by condition" + extra)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def geomean_by_condition(data):
    """Bar chart: geometric-mean time per condition across benchmarks that
    completed under EVERY condition (apples-to-apples)."""
    import matplotlib.pyplot as plt
    from math import exp, log

    by_bench = _by_bench_cond(data)
    conds = _all_conditions(data)
    complete = [b for b, cm in by_bench.items() if set(cm.keys()) == set(conds)]

    geos = []
    for cond in conds:
        vals = [by_bench[b][cond] for b in complete if by_bench[b][cond] > 0]
        geos.append(exp(_mean([log(v) for v in vals])) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#5b8def" if c.startswith("bridge") else "#e08a3c" for c in conds]
    ax.bar(conds, geos, color=colors)
    ax.set_ylabel("geomean time (s)")
    ax.set_title(f"Geomean time by condition ({len(complete)} fully-complete benchmarks)")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    plt.tight_layout()
    return fig


def completion_cdf(data):
    """Performance profile / CDF: x = wall-clock time T (log scale), y = number
    of benchmarks each treatment COMPLETED within time T. One right-continuous
    step curve per condition (backend, encoding).

    A treatment that completes more benchmarks faster sits up-and-to-the-left.
    Errored / timed-out cells never produce a timing, so they never reach
    "completed" and simply don't contribute to that treatment's curve. Curves
    can therefore plateau at different heights (different completed-counts),
    which is the point of a performance profile.
    """
    import matplotlib.pyplot as plt

    conds = _all_conditions(data)
    # Per condition, the per-benchmark completion time (mean of its timings).
    times_by_cond = {c: [] for c in conds}
    for row in data.get("timings", []):
        tl = row.get("timing_list", [])
        if tl:
            times_by_cond[row["condition"]].append(_mean(tl))

    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in conds:
        ts = sorted(times_by_cond[cond])
        if not ts:
            continue
        # Step curve: at each completion time T the count of benchmarks done by
        # T jumps by one. Right-continuous: draw [t_k, t_{k+1}) at height k+1.
        xs = [ts[0]]
        ys = [0]
        for k, t in enumerate(ts):
            xs.extend([t, t])
            ys.extend([k, k + 1])
        # Extend the final plateau to the right so the curve's height is visible.
        xs.append(ts[-1] * 1.05)
        ys.append(len(ts))
        style = "--" if cond.endswith("-proofs") else "-"
        ax.step(xs, ys, where="post", label=f"{cond} ({len(ts)})", linestyle=style)

    ax.set_xscale("log")
    ax.set_xlabel("wall-clock time T (s, log scale)")
    ax.set_ylabel("benchmarks completed within T")
    ax.set_title("Completion CDF / performance profile by condition")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, title="condition (n completed)")
    plt.tight_layout()
    return fig


def mean_time_table(data):
    """Pivot: one row per benchmark, a column per condition (mean seconds)."""
    by_bench = _by_bench_cond(data)
    conds = _all_conditions(data)
    rows = []
    for bench in sorted(by_bench.keys()):
        row = {"benchmark": bench}
        for cond in conds:
            v = by_bench[bench].get(cond)
            row[cond] = round(v, 4) if v is not None else ""
        rows.append(row)
    return rows


# --- Peak memory (RSS) ------------------------------------------------------

def _by_bench_cond_field(data, field, transform=lambda x: x):
    """{benchmark: {condition: transform(row[field])}} for rows that carry
    `field`."""
    from collections import defaultdict
    out = defaultdict(dict)
    for row in data.get("timings", []):
        if field in row and row[field] is not None:
            out[row["benchmark"]][row["condition"]] = transform(row[field])
    return out


def peak_memory_grouped(data):
    """Grouped bar chart: per benchmark, one bar per condition (peak RSS, MB)."""
    import matplotlib.pyplot as plt

    by_bench = _by_bench_cond_field(data, "rss", lambda b: b / 1e6)
    conds = _all_conditions(data)
    benches = _top_keys(by_bench)
    n = len(conds)

    fig, ax = plt.subplots(figsize=(_fig_width(len(benches), 1.6), 5))
    width = 0.8 / max(n, 1)
    for j, cond in enumerate(conds):
        xs = [i + j * width for i in range(len(benches))]
        ys = [by_bench[b].get(cond, 0) for b in benches]
        ax.bar(xs, ys, width=width, label=cond)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(benches))])
    ax.set_xticklabels([b.split("/")[-1] for b in benches], rotation=30, ha="right")
    ax.set_ylabel("peak RSS (MB)")
    extra = f" (top {len(benches)} of {len(by_bench)})" if len(by_bench) > len(benches) else ""
    ax.set_title("Peak memory per benchmark by condition" + extra)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def peak_memory_table(data):
    """Pivot: one row per benchmark, a column per condition (peak RSS, MB)."""
    by_bench = _by_bench_cond_field(data, "rss", lambda b: b / 1e6)
    conds = _all_conditions(data)
    rows = []
    for bench in sorted(by_bench.keys()):
        row = {"benchmark": bench}
        for cond in conds:
            v = by_bench[bench].get(cond)
            row[f"{cond} (MB)"] = round(v, 1) if v is not None else ""
        rows.append(row)
    return rows


# --- Correctness / parity ---------------------------------------------------

def parity_table(data):
    """Pivot: one row per benchmark, a column per condition. Each cell is the
    tuple-count parity vs the bridge-normal oracle: OK / MISMATCH / (blank if
    the cell errored or has no parity result)."""
    from collections import defaultdict
    by_bench = defaultdict(dict)
    for row in data.get("timings", []):
        p = row.get("parity")
        if p is None:
            continue
        by_bench[row["benchmark"]][row["condition"]] = "OK" if p else "MISMATCH"
    conds = _all_conditions(data)
    rows = []
    for bench in sorted(by_bench.keys()):
        row = {"benchmark": bench}
        for cond in conds:
            row[cond] = by_bench[bench].get(cond, "")
        rows.append(row)
    return rows


def parity_summary(data):
    """One row per condition: how many benchmarks matched / mismatched the
    bridge-normal oracle (tuple-count parity)."""
    from collections import defaultdict
    ok = defaultdict(int)
    bad = defaultdict(int)
    bad_files = defaultdict(list)
    for row in data.get("timings", []):
        p = row.get("parity")
        if p is None:
            continue
        if p:
            ok[row["condition"]] += 1
        else:
            bad[row["condition"]] += 1
            bad_files[row["condition"]].append(row["benchmark"].split("/")[-1])
    rows = []
    for cond in _all_conditions(data):
        if cond not in ok and cond not in bad:
            continue
        rows.append({
            "condition": cond,
            "parity OK": ok.get(cond, 0),
            "MISMATCH": bad.get(cond, 0),
            "mismatched files": ", ".join(sorted(bad_files.get(cond, []))[:8]),
        })
    return rows


# --- Per-phase profile (rebuild / canonicalize / congruence) ----------------

_PHASE_BUCKETS = ("rebuild", "canonicalize", "congruence", "other")


def phase_breakdown_table(data):
    """One row per (benchmark, condition) that has a per-phase profile, with a
    column per bucket (seconds) plus the bucket totals."""
    rows = []
    for row in data.get("timings", []):
        phases = row.get("phases")
        if not phases:
            continue
        r = {"benchmark": row["benchmark"], "condition": row["condition"]}
        for b in _PHASE_BUCKETS:
            v = phases.get(b)
            r[b] = round(v, 4) if v else ""
        rows.append(r)
    return rows


def phase_stacked(data):
    """Stacked bar: per (benchmark, condition) with a profile, stack the
    rebuild/canonicalize/congruence/other bucket seconds. One bar per cell."""
    import matplotlib.pyplot as plt

    cells = []
    for row in data.get("timings", []):
        phases = row.get("phases")
        if not phases:
            continue
        label = f"{row['benchmark'].split('/')[-1]}\n{row['condition']}"
        cells.append((label, phases))
    if not cells:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no per-phase profiles captured", ha="center")
        ax.axis("off")
        return fig

    n_cells = len(cells)
    cells.sort(key=lambda c: sum(c[1].get(b, 0) or 0 for b in _PHASE_BUCKETS),
               reverse=True)
    cells = cells[:_MAX_BARS]
    fig, ax = plt.subplots(figsize=(_fig_width(len(cells), 0.6), 5))
    xs = range(len(cells))
    bottoms = [0.0] * len(cells)
    colors = {"rebuild": "#d9534f", "canonicalize": "#5bc0de",
              "congruence": "#5cb85c", "other": "#bbbbbb"}
    for b in _PHASE_BUCKETS:
        ys = [c[1].get(b, 0) or 0 for c in cells]
        ax.bar(xs, ys, bottom=bottoms, label=b, color=colors[b])
        bottoms = [a + y for a, y in zip(bottoms, ys)]
    ax.set_xticks(list(xs))
    ax.set_xticklabels([c[0] for c in cells], rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("phase time (s)")
    extra = f" (top {len(cells)} of {n_cells})" if n_cells > len(cells) else ""
    ax.set_title("Per-phase bucket breakdown (rebuild / canonicalize / congruence)" + extra)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def mean_time_filter(filtered_rows, data):
    keep = {r["benchmark"] for r in filtered_rows}
    return {
        **data,
        "timings": [r for r in data.get("timings", []) if r["benchmark"] in keep],
        "errors": [r for r in data.get("errors", []) if r["benchmark"] in keep],
    }


def speedup_vs_bridge_normal(data):
    """Pivot of slowdown factors relative to bridge-normal (x times slower).
    Only benchmarks that have a bridge-normal baseline are shown."""
    by_bench = _by_bench_cond(data)
    conds = _all_conditions(data)
    base = "bridge-normal"
    rows = []
    for bench in sorted(by_bench.keys()):
        cm = by_bench[bench]
        if base not in cm or cm[base] <= 0:
            continue
        row = {"benchmark": bench, f"{base} (s)": round(cm[base], 4)}
        for cond in conds:
            if cond == base:
                continue
            v = cm.get(cond)
            row[f"{cond} x"] = round(v / cm[base], 2) if v else ""
        rows.append(row)
    return rows


def skipped_table(data):
    """Files excluded up front (not benchmarked) and why -- e.g. unsupported by
    the term encoding. Kept out of the errors table so that table shows only
    real backend failures on supported files."""
    return [{"benchmark": s.get("benchmark"), "reason": s.get("reason")}
            for s in data.get("skipped", [])]


reg = eval_live.Registry()
reg.graph("Mean time per benchmark", mean_time_grouped)
reg.graph("Peak memory per benchmark", peak_memory_grouped)
reg.graph("Geomean by condition", geomean_by_condition)
reg.graph("Completion CDF (performance profile)", completion_cdf)
reg.graph("Per-phase breakdown (stacked)", phase_stacked)
# Only the mean-time computed table is registered, to keep the viewer
# uncluttered. The other table fns (peak_memory_table, parity_table,
# parity_summary, phase_breakdown_table, speedup_vs_bridge_normal,
# skipped_table) stay DEFINED but UNregistered -- their data still lives in the
# raw timings/errors/skipped tables and results.json. Re-add a reg.table(...)
# line to bring any back. (Note: this also governs what --render writes.)
reg.table("Mean time (s) per condition", mean_time_table, filter_source=mean_time_filter)
eval_live.registry = reg
