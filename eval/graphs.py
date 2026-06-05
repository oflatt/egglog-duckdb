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


def mean_time_grouped(data):
    """Grouped bar chart: per benchmark, one bar per condition (mean time)."""
    import matplotlib.pyplot as plt

    by_bench = _by_bench_cond(data)
    conds = _all_conditions(data)
    benches = sorted(by_bench.keys())
    n = len(conds)

    fig, ax = plt.subplots(figsize=(max(8, len(benches) * 1.6), 5))
    width = 0.8 / max(n, 1)
    for j, cond in enumerate(conds):
        xs = [i + j * width for i in range(len(benches))]
        ys = [by_bench[b].get(cond, 0) for b in benches]
        ax.bar(xs, ys, width=width, label=cond)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(benches))])
    ax.set_xticklabels([b.split("/")[-1] for b in benches], rotation=30, ha="right")
    ax.set_ylabel("mean wall-clock (s)")
    ax.set_title("Mean time per benchmark by condition")
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


reg = eval_live.Registry()
reg.graph("Mean time per benchmark", mean_time_grouped)
reg.graph("Geomean by condition", geomean_by_condition)
reg.graph("Completion CDF (performance profile)", completion_cdf)
reg.table("Mean time (s) per condition", mean_time_table, filter_source=mean_time_filter)
reg.table("Slowdown vs bridge-normal", speedup_vs_bridge_normal)
eval_live.registry = reg
