"""
thesis_plots.py  —  Publication-quality figures from thesis_stats.py exports.

Outputs (SESSION/analysis/thesis/figures/):
  fig_learning_curves_{bench}.pdf   — success rate vs episode, thesis methods × each benchmark
  fig_auc_boxplot.pdf               — recovery AUC distributions (new_task switches)
  fig_ttt_delta_boxplot.pdf         — TTT delta vs scratch (new_task switches)
  fig_retention_heatmap.pdf         — benchmark × method × seed retention grid
  fig_retention_bars.pdf            — bar chart: retention rate per benchmark per method
  fig_prev_task_success.pdf         — previous-task success at end of sequence
  fig_abc_final_tasks.pdf           — ABC final task breakdown (current + previous)

Usage:
    python3 thesis_plots.py SESSION_DIR
"""
from __future__ import annotations
import csv, math, sys, os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_METHODS_ORDER = [
    "ddqn_vanilla",
    "oracle_reset",
    "oracle_segmented",
    "oracle_segmented_distill_l001",
    "der_plus_plus",
]
METHODS_ORDER = list(DEFAULT_METHODS_ORDER)
METHOD_LABELS = {
    "ddqn_vanilla":                  "DDQN Vanilla",
    "oracle_reset":                  "Oracle Reset",
    "oracle_segmented":              "Oracle Seg.",
    "oracle_segmented_distill_l001": r"Oracle Seg. + Distill ($\lambda$=0.001)",
    "der_plus_plus":                 "DER++",
}
METHOD_SHORT_LABELS = {
    "ddqn_vanilla": "V",
    "oracle_reset": "R",
    "oracle_segmented": "S",
    "oracle_segmented_distill_l001": "S+D",
    "der_plus_plus": "DER++",
}
METHOD_COLORS = {
    "ddqn_vanilla":                  "#888888",
    "oracle_reset":                  "#e07b39",
    "oracle_segmented":              "#4878d0",
    "oracle_segmented_distill_l001": "#6acc65",
    "der_plus_plus":                 "#d65f5f",
}
METHOD_LS = {
    "ddqn_vanilla":                  ":",
    "oracle_reset":                  "--",
    "oracle_segmented":              "-",
    "oracle_segmented_distill_l001": "-.",
    "der_plus_plus":                 (0, (3, 1, 1, 1)),
}
BENCH_ORDER = [
    "gw9_goal_balanced_ab_v1",
    "gw9_goal_balanced_ac_v1",
    "gw9_goal_balanced_aba_v1",
    "gw9_goal_balanced_abc_v1",
]
BENCH_LABELS = {
    "gw9_goal_balanced_ab_v1":  "AB",
    "gw9_goal_balanced_ac_v1":  "AC",
    "gw9_goal_balanced_aba_v1": "ABA",
    "gw9_goal_balanced_abc_v1": "ABC",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ─────────────────────────────────────────────────────────────────────────────
def _f(s):
    try:
        return float(s)
    except Exception:
        return float("nan")

def load_csv(p: Path) -> list[dict]:
    with open(p, newline="") as f:
        return list(csv.DictReader(f))

def mean_sem(vals):
    vals = [v for v in vals if not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(vals) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in vals) / (n - 1)
    return mu, math.sqrt(var / n)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def method_short_label(method: str) -> str:
    return METHOD_SHORT_LABELS.get(method, method)


def resolve_methods_order(root: Path, rows: list[dict]) -> list[str]:
    present_methods = []
    for row in rows:
        method = str(row.get("method", "")).strip()
        if method and method not in present_methods:
            present_methods.append(method)

    ordered: list[str] = []
    session_config = root / "session_config.json"
    if session_config.exists():
        try:
            import json
            config = json.loads(session_config.read_text())
            for method in str(config.get("methods_csv", "")).split(","):
                method = method.strip()
                if method and method in present_methods and method not in ordered:
                    ordered.append(method)
        except Exception:
            pass

    for method in DEFAULT_METHODS_ORDER:
        if method in present_methods and method not in ordered:
            ordered.append(method)

    for method in present_methods:
        if method not in ordered:
            ordered.append(method)
    return ordered


def infer_seed_count(rows: list[dict]) -> int:
    seeds = {str(row.get("seed", "")).strip() for row in rows if str(row.get("seed", "")).strip()}
    return len(seeds)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Learning curves  (one PDF per benchmark)
# ─────────────────────────────────────────────────────────────────────────────
def fig_learning_curves(ev_rows: list[dict], out_dir: Path):
    """One figure per benchmark, showing mean ± SEM success rate vs episode."""
    seed_count = infer_seed_count(ev_rows)

    # Group by (benchmark, method, episode, eval_scope="current_task")
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # data[bench][method][episode] = [success_rate values across seeds]
    tasks_at_ep = defaultdict(lambda: defaultdict(dict))  # bench → episode → task_id (first seed)
    eps_per_task = {}  # bench → episodes_per_task
    num_tasks_map = {}  # bench → num_tasks

    for r in ev_rows:
        if r["eval_scope"] != "current_task":
            continue
        b, m = r["benchmark"], r["method"]
        ep = int(r["episode"])
        sr = _f(r["success_rate"])
        data[b][m][ep].append(sr)
        if b not in eps_per_task:
            eps_per_task[b] = int(r["episodes_per_task"])
            num_tasks_map[b] = int(r["num_tasks"])

    for b in BENCH_ORDER:
        if b not in data:
            continue
        bl = BENCH_LABELS[b]
        ept = eps_per_task.get(b, 400)
        nt = num_tasks_map.get(b, 2)

        fig, ax = plt.subplots(figsize=(7, 3.5))

        for m in METHODS_ORDER:
            if m not in data[b]:
                continue
            eps_sorted = sorted(data[b][m].keys())
            mus, sems = [], []
            for ep in eps_sorted:
                mu, se = mean_sem(data[b][m][ep])
                mus.append(mu)
                sems.append(se)
            xs = np.array(eps_sorted)
            ys = np.array(mus)
            ses = np.array(sems)
            label = method_label(m)
            color = METHOD_COLORS[m]
            ls = METHOD_LS[m]
            ax.plot(xs, ys, label=label, color=color, ls=ls, lw=1.6)
            ax.fill_between(xs, ys - ses, ys + ses, alpha=0.15, color=color)

        # Task switch markers
        for t in range(1, nt):
            ax.axvline(t * ept, color="black", lw=0.8, ls="--", alpha=0.5)
            ax.text(t * ept + ept * 0.02, 0.97, f"Task {t+1}", va="top",
                    fontsize=8, color="black", alpha=0.7, transform=ax.get_xaxis_transform())

        ax.set_xlim(0, nt * ept)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Learning Curves — Benchmark {bl}")
        ax.axhline(0.8, color="gray", lw=0.8, ls=":", alpha=0.6, label="Threshold (0.8)")
        ax.legend(loc="lower right", framealpha=0.85)
        ax.grid(True, alpha=0.3)

        out_path = out_dir / f"fig_learning_curves_{bl.lower()}.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  saved: {out_path.name}")

    # Also one combined 2×2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False, sharey=True)
    axes = axes.flatten()
    handles, labels_leg = [], []
    for idx, b in enumerate(BENCH_ORDER):
        if b not in data:
            continue
        ax = axes[idx]
        bl = BENCH_LABELS[b]
        ept = eps_per_task.get(b, 400)
        nt = num_tasks_map.get(b, 2)
        for m in METHODS_ORDER:
            if m not in data[b]:
                continue
            eps_sorted = sorted(data[b][m].keys())
            mus = [mean_sem(data[b][m][ep])[0] for ep in eps_sorted]
            ses = [mean_sem(data[b][m][ep])[1] for ep in eps_sorted]
            xs = np.array(eps_sorted)
            ys = np.array(mus)
            se_arr = np.array(ses)
            color = METHOD_COLORS[m]
            ls = METHOD_LS[m]
            h, = ax.plot(xs, ys, color=color, ls=ls, lw=1.4)
            ax.fill_between(xs, ys - se_arr, ys + se_arr, alpha=0.12, color=color)
            if idx == 0:
                handles.append(h)
                labels_leg.append(method_label(m))
        for t in range(1, nt):
            ax.axvline(t * ept, color="black", lw=0.7, ls="--", alpha=0.4)
        ax.axhline(0.8, color="gray", lw=0.7, ls=":", alpha=0.5)
        ax.set_xlim(0, nt * ept)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Benchmark {bl}", fontsize=11)
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel("Success Rate", fontsize=9) if idx % 2 == 0 else None
        ax.grid(True, alpha=0.25)

        legend_cols = min(max(1, len(labels_leg)), 3)
        fig.legend(handles, labels_leg, loc="lower center", ncol=legend_cols, bbox_to_anchor=(0.5, -0.03),
               framealpha=0.9, fontsize=9)
    fig.suptitle(
        f"Learning Curves — All Benchmarks (Mean ± SEM, {seed_count} seeds)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out_path = out_dir / "fig_learning_curves_all.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: AUC boxplots (new_task switches)
# ─────────────────────────────────────────────────────────────────────────────
def fig_auc_boxplot(sw_rows: list[dict], out_dir: Path):
    data = defaultdict(list)  # method → [auc values]
    data_bench = defaultdict(lambda: defaultdict(list))  # (bench, method) → [auc]

    for r in sw_rows:
        if r["switch_type"] != "new_task":
            continue
        v = _f(r["recovery_auc"])
        if not math.isnan(v):
            data[r["method"]].append(v)
            data_bench[r["benchmark"]][r["method"]].append(v)

    fig, axes = plt.subplots(1, 5, figsize=(14, 4), sharey=True)
    panel_titles = ["All"] + [BENCH_LABELS[b] for b in BENCH_ORDER]
    panel_data = [data] + [data_bench[b] for b in BENCH_ORDER]

    for ax, title, d in zip(axes, panel_titles, panel_data):
        boxes = [d.get(m, []) for m in METHODS_ORDER]
        bp = ax.boxplot(boxes, patch_artist=True, notch=False,
                        medianprops=dict(color="black", lw=2),
                        whiskerprops=dict(lw=1.2),
                        capprops=dict(lw=1.2),
                        flierprops=dict(marker="o", ms=3, alpha=0.5))
        for patch, m in zip(bp["boxes"], METHODS_ORDER):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(1, len(METHODS_ORDER) + 1))
        ax.set_xticklabels([method_short_label(m) for m in METHODS_ORDER], fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Recovery AUC (new task)")

    legend_patches = [mpatches.Patch(color=METHOD_COLORS[m], alpha=0.7, label=method_label(m))
                      for m in METHODS_ORDER]
    fig.legend(handles=legend_patches, loc="lower center", ncol=min(len(METHODS_ORDER), 5),
               bbox_to_anchor=(0.5, -0.08), fontsize=8, framealpha=0.9)
    fig.suptitle("Recovery AUC — New Task Switches",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    out_path = out_dir / "fig_auc_boxplot.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: TTT delta boxplots
# ─────────────────────────────────────────────────────────────────────────────
def fig_ttt_delta_boxplot(sw_rows: list[dict], out_dir: Path):
    data = defaultdict(list)
    data_bench = defaultdict(lambda: defaultdict(list))

    for r in sw_rows:
        if r["switch_type"] != "new_task":
            continue
        v = _f(r["ttt_delta"])
        if not math.isnan(v):
            data[r["method"]].append(v)
            data_bench[r["benchmark"]][r["method"]].append(v)

    fig, axes = plt.subplots(1, 5, figsize=(14, 4), sharey=False)
    panel_titles = ["All"] + [BENCH_LABELS[b] for b in BENCH_ORDER]
    panel_data = [data] + [data_bench[b] for b in BENCH_ORDER]

    for ax, title, d in zip(axes, panel_titles, panel_data):
        boxes = [d.get(m, []) for m in METHODS_ORDER]
        # Clip extreme outliers for visual clarity
        all_vals = [v for box in boxes for v in box if not math.isnan(v)]
        if all_vals:
            q1, q3 = np.percentile(all_vals, 25), np.percentile(all_vals, 75)
            iqr = q3 - q1
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        else:
            lo, hi = -5000, 5000

        bp = ax.boxplot(boxes, patch_artist=True, notch=False,
                        medianprops=dict(color="black", lw=2),
                        whiskerprops=dict(lw=1.2),
                        capprops=dict(lw=1.2),
                        flierprops=dict(marker="o", ms=3, alpha=0.5))
        for patch, m in zip(bp["boxes"], METHODS_ORDER):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.7)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(1, len(METHODS_ORDER) + 1))
        ax.set_xticklabels([method_short_label(m) for m in METHODS_ORDER], fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel(r"$\Delta$TTT vs Scratch (steps)")
        ax.set_ylim(lo, hi)

    legend_patches = [mpatches.Patch(color=METHOD_COLORS[m], alpha=0.7, label=method_label(m))
                      for m in METHODS_ORDER]
    fig.legend(handles=legend_patches, loc="lower center", ncol=min(len(METHODS_ORDER), 5),
               bbox_to_anchor=(0.5, -0.08), fontsize=8, framealpha=0.9)
    fig.suptitle(r"$\Delta$TTT vs Scratch — New Task Switches (negative = forward transfer)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    out_path = out_dir / "fig_ttt_delta_boxplot.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Retention heatmap (benchmark × method, previous-task success)
# ─────────────────────────────────────────────────────────────────────────────
def fig_retention_heatmap(ret_rows: list[dict], out_dir: Path):
    """Heatmap: rows = benchmark, cols = method, cell = mean prev-task success."""
    seed_count = infer_seed_count(ret_rows)
    grid = np.full((len(BENCH_ORDER), len(METHODS_ORDER)), float("nan"))
    for r in ret_rows:
        if r["task_role"] != "previous_task":
            continue
        b = r["benchmark"]; m = r["method"]
        if b not in BENCH_ORDER or m not in METHODS_ORDER:
            continue
        bi = BENCH_ORDER.index(b); mi = METHODS_ORDER.index(m)
        if math.isnan(grid[bi, mi]):
            grid[bi, mi] = _f(r["success_rate"])
        else:
            # accumulate for averaging later
            pass

    # Re-aggregate properly
    agg = defaultdict(list)
    for r in ret_rows:
        if r["task_role"] != "previous_task":
            continue
        v = _f(r["success_rate"])
        if not math.isnan(v):
            agg[(r["benchmark"], r["method"])].append(v)

    for (b, m), vals in agg.items():
        if b in BENCH_ORDER and m in METHODS_ORDER:
            bi = BENCH_ORDER.index(b)
            mi = METHODS_ORDER.index(m)
            grid[bi, mi] = sum(vals) / len(vals)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(METHODS_ORDER)))
    ax.set_xticklabels([method_short_label(m) for m in METHODS_ORDER], fontsize=9)
    ax.set_yticks(range(len(BENCH_ORDER)))
    ax.set_yticklabels([BENCH_LABELS[b] for b in BENCH_ORDER], fontsize=9)

    # Annotate cells
    for bi in range(len(BENCH_ORDER)):
        for mi in range(len(METHODS_ORDER)):
            v = grid[bi, mi]
            if not math.isnan(v):
                color = "black" if 0.3 < v < 0.7 else "white"
                ax.text(mi, bi, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean previous-task success rate")
    ax.set_title(
        f"Previous-Task Success at End of Sequence\n(Retention — mean over {seed_count} seeds)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / "fig_retention_heatmap.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Retention bar chart (all_unique threshold)
# ─────────────────────────────────────────────────────────────────────────────
def fig_retention_bars(tables_dir: Path, out_dir: Path):
    """Bar chart: fraction of runs where all unique tasks >= 0.8, per benchmark × method."""
    rows = load_csv(tables_dir / "tab_retention_by_benchmark.csv")

    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    for ax, b in zip(axes, BENCH_ORDER):
        bl = BENCH_LABELS[b]
        bench_rows = [r for r in rows if r["benchmark"] == b]
        methods_in = [r["method"] for r in bench_rows]
        rates = [float(r["retention_rate"]) for r in bench_rows]
        ns = [int(r["n_runs"]) for r in bench_rows]
        ks = [int(r["n_retained"]) for r in bench_rows]
        xs = range(len(METHODS_ORDER))
        for i, m in enumerate(METHODS_ORDER):
            if m in methods_in:
                idx = methods_in.index(m)
                k, n = ks[idx], ns[idx]
                ax.bar(i, rates[idx], color=METHOD_COLORS[m], alpha=0.8, width=0.6,
                       edgecolor="black", lw=0.6)
                ax.text(i, rates[idx] + 0.02, f"{k}/{n}", ha="center", fontsize=8)
            else:
                ax.bar(i, 0, color="lightgray", alpha=0.5, width=0.6)

        ax.axhline(0.8, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.set_title(f"Benchmark {bl}", fontsize=10)
        ax.set_xticks(range(len(METHODS_ORDER)))
        ax.set_xticklabels([method_short_label(m) for m in METHODS_ORDER], fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.grid(True, axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Retention Rate (all tasks ≥ 0.8)")

    legend_patches = [mpatches.Patch(color=METHOD_COLORS[m], alpha=0.8, label=method_label(m))
                      for m in METHODS_ORDER]
    fig.legend(handles=legend_patches, loc="lower center", ncol=min(len(METHODS_ORDER), 3),
               bbox_to_anchor=(0.5, -0.12), fontsize=9, framealpha=0.9)
    fig.suptitle("Retention Rate per Benchmark\n(fraction of runs: all unique tasks ≥ 0.8 at end)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = out_dir / "fig_retention_bars.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: ABC final task breakdown
# ─────────────────────────────────────────────────────────────────────────────
def fig_abc_final_tasks(tables_dir: Path, out_dir: Path):
    """Grouped bar chart: final success per task (A, B, C) per method in ABC benchmark."""
    rows = load_csv(tables_dir / "tab_final_tasks.csv")
    abc_rows = [r for r in rows if r["benchmark"] == "gw9_goal_balanced_abc_v1"]
    seed_count = max((int(float(r["n"])) for r in abc_rows if r.get("n")), default=0)

    # task IDs in ABC — collect unique task_ids preserving order
    seen_tasks = []
    for r in abc_rows:
        if r["task_id"] not in seen_tasks:
            seen_tasks.append(r["task_id"])

    # We want: for each method, the current_task (C) success and previous_task (A, B) success
    fig, ax = plt.subplots(figsize=(8, 4))
    n_methods = len(METHODS_ORDER)
    # For simplicity: group by method, within each method plot task A, B, C bars
    tasks_short = [t.split("_")[-1].upper() for t in seen_tasks]

    x = np.arange(len(METHODS_ORDER))
    width = 0.22
    task_colors = ["#4878d0", "#e07b39", "#6acc65"]  # A, B, C

    for ti, (tid, tshort) in enumerate(zip(seen_tasks, tasks_short)):
        task_means = []
        task_ses = []
        for m in METHODS_ORDER:
            vals_cur = [r for r in abc_rows if r["method"] == m and r["task_id"] == tid
                        and r["task_role"] == "current_task"]
            vals_prev = [r for r in abc_rows if r["method"] == m and r["task_id"] == tid
                         and r["task_role"] == "previous_task"]
            # Use whichever role has data (current_task for last task, previous_task for earlier)
            vals = vals_cur or vals_prev
            if vals:
                # vals has one row per (method, task_id, role) with success_mean
                mu = float(vals[0]["success_mean"])
                se = float(vals[0]["success_se"]) if vals[0]["success_se"] else 0.0
            else:
                mu, se = 0.0, 0.0
            task_means.append(mu)
            task_ses.append(se if not math.isnan(se) else 0.0)

        offset = (ti - len(seen_tasks) / 2 + 0.5) * width
        bars = ax.bar(x + offset, task_means, width, label=f"Task {tshort}",
                      color=task_colors[ti % 3], alpha=0.8, edgecolor="black", lw=0.5)
        ax.errorbar(x + offset, task_means, yerr=task_ses, fmt="none",
                    color="black", capsize=3, lw=1.2)

    ax.axhline(0.8, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([method_short_label(m) for m in METHODS_ORDER], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Final Success Rate")
    ax.set_title(
        f"ABC Benchmark — Final Task Success per Method (Mean ± SE, {seed_count} seeds)",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "fig_abc_final_tasks.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Revisit AUC (ABA only) — strip/swarm-like dot plot
# ─────────────────────────────────────────────────────────────────────────────
def fig_revisit_auc(sw_rows: list[dict], out_dir: Path):
    seed_count = infer_seed_count(sw_rows)
    aba_bench = "gw9_goal_balanced_aba_v1"
    data = defaultdict(list)
    for r in sw_rows:
        if r["benchmark"] != aba_bench:
            continue
        if r["switch_type"] != "revisit_task":
            continue
        v = _f(r["recovery_auc"])
        if not math.isnan(v):
            data[r["method"]].append(v)

    if not any(data.values()):
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for i, m in enumerate(METHODS_ORDER):
        vals = data[m]
        if not vals:
            continue
        # jitter
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(i + jitter, vals, color=METHOD_COLORS[m], alpha=0.6, s=30, zorder=3)
        mu, se = mean_sem(vals)
        ax.plot([i - 0.25, i + 0.25], [mu, mu], color="black", lw=2.5, zorder=4)
        ax.plot([i, i], [mu - se, mu + se], color="black", lw=1.5, zorder=4)

    ax.axhline(0.8, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.set_xticks(range(len(METHODS_ORDER)))
    ax.set_xticklabels([method_short_label(m) for m in METHODS_ORDER], fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("Revisit Recovery AUC")
    ax.set_title(
        f"Revisit AUC — ABA Benchmark ({seed_count} seeds)\nHorizontal bar = mean, error = ±SEM",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "fig_revisit_auc.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(session_dir: str):
    global METHODS_ORDER
    root = Path(session_dir)
    thesis = root / "analysis" / "thesis"
    export = thesis / "export"
    tables = thesis / "tables"
    fig_dir = thesis / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ev = load_csv(export / "eval_curves_clean.csv")
    sw = load_csv(export / "switch_metrics_clean.csv")
    ret = load_csv(export / "retention_matrix.csv")
    METHODS_ORDER = resolve_methods_order(root, ev or sw or ret)

    print("Generating figures...")
    fig_learning_curves(ev, fig_dir)
    fig_auc_boxplot(sw, fig_dir)
    fig_ttt_delta_boxplot(sw, fig_dir)
    fig_retention_heatmap(ret, fig_dir)
    fig_retention_bars(tables, fig_dir)
    fig_abc_final_tasks(tables, fig_dir)
    fig_revisit_auc(sw, fig_dir)
    print(f"\nAll figures saved to: {fig_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 thesis_plots.py SESSION_DIR")
        sys.exit(1)
    main(sys.argv[1])
