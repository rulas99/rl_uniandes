"""
thesis_env_plots.py  —  Environment visualizations for thesis document.

Generates:
  fig_env_overview.pdf      — 9×9 grid with wall, agent start, goals A/B/C
  fig_env_benchmarks.pdf    — 4 panels: AB, AC, ABA, ABC (task sequences)
  fig_env_grid_only.pdf     — clean grid (no labels) for custom annotation

Usage:
    python3 thesis_env_plots.py [OUTPUT_DIR]
    # default output: current directory
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Environment spec  (gw9_goal_balanced)
# ─────────────────────────────────────────────────────────────────────────────
SIZE = 9
OBSTACLES = [(0, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (8, 4)]
# Gaps in the wall: x=1 and x=7

AGENT_START = (4, 8)   # center top

GOALS = {
    "A": (1, 0),
    "B": (7, 0),
    "C": (4, 0),
}

BENCHMARKS = {
    "AB":  ["A", "B"],
    "AC":  ["A", "C"],
    "ABA": ["A", "B", "A"],
    "ABC": ["A", "B", "C"],
}

# Colors
WALL_COLOR    = "#444444"
FLOOR_COLOR   = "#f5f5f0"
GRID_COLOR    = "#cccccc"
AGENT_COLOR   = "#2c7bb6"
GOAL_COLORS   = {"A": "#d73027", "B": "#1a9641", "C": "#f28000"}
GOAL_MARKERS  = {"A": "*", "B": "*", "C": "*"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ─────────────────────────────────────────────────────────────────────────────
# Core drawing function
# ─────────────────────────────────────────────────────────────────────────────
def draw_grid(ax, active_goals: list[str], show_agent: bool = True,
              show_task_label: bool = False, task_sequence: list[str] | None = None,
              highlight_current: str | None = None, title: str = ""):
    """
    Draw one 9×9 grid panel.

    active_goals: which goal labels to draw (e.g. ["A", "B", "C"])
    highlight_current: if set, draws that goal larger/brighter (current task)
    """
    # Background
    ax.set_facecolor(FLOOR_COLOR)
    ax.set_xlim(-0.5, SIZE - 0.5)
    ax.set_ylim(-0.5, SIZE - 0.5)
    ax.set_aspect("equal")

    # Grid lines
    for i in range(SIZE + 1):
        ax.axhline(i - 0.5, color=GRID_COLOR, lw=0.6, zorder=1)
        ax.axvline(i - 0.5, color=GRID_COLOR, lw=0.6, zorder=1)

    # Obstacle cells
    for (ox, oy) in OBSTACLES:
        rect = mpatches.FancyBboxPatch(
            (ox - 0.48, oy - 0.48), 0.96, 0.96,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=WALL_COLOR, zorder=2
        )
        ax.add_patch(rect)

    # Wall gap indicators (subtle arrows at gaps x=1 and x=7)
    for gx in [1, 7]:
        ax.annotate("", xy=(gx, 3.55), xytext=(gx, 4.45),
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
                    zorder=3)

    # Goals
    for label in active_goals:
        gx, gy = GOALS[label]
        color = GOAL_COLORS[label]
        is_current = (label == highlight_current)
        size = 520 if is_current else 380
        alpha = 1.0 if is_current else 0.85
        # Filled circle
        ax.scatter(gx, gy, s=size, c=color, marker="*", zorder=5, alpha=alpha,
                   edgecolors="white", linewidths=0.8)
        # Label
        offset = 0.42
        ax.text(gx, gy + offset, label, ha="center", va="bottom",
                fontsize=9 if is_current else 8,
                fontweight="bold" if is_current else "normal",
                color=color, zorder=6)

    # Agent start
    if show_agent:
        ax.scatter(*AGENT_START, s=260, c=AGENT_COLOR, marker="o", zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.text(AGENT_START[0], AGENT_START[1] - 0.42, "S",
                ha="center", va="top", fontsize=8, color=AGENT_COLOR,
                fontweight="bold", zorder=6)

    # Axis ticks
    ax.set_xticks(range(SIZE))
    ax.set_yticks(range(SIZE))
    ax.tick_params(labelsize=7, length=2)
    ax.set_xlabel("x", fontsize=8, labelpad=2)
    ax.set_ylabel("y", fontsize=8, labelpad=2)

    if title:
        ax.set_title(title, fontsize=10, pad=4)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Overview — single grid showing all goals + agent start
# ─────────────────────────────────────────────────────────────────────────────
def fig_overview(out_dir: Path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    draw_grid(ax, active_goals=["A", "B", "C"], show_agent=True,
              title="9×9 GridWorld Environment")

    # Legend
    legend_items = [
        mpatches.Patch(color=AGENT_COLOR, label="Agent start (S)"),
        mpatches.Patch(color=GOAL_COLORS["A"], label="Goal A  (1, 0)"),
        mpatches.Patch(color=GOAL_COLORS["B"], label="Goal B  (7, 0)"),
        mpatches.Patch(color=GOAL_COLORS["C"], label="Goal C  (4, 0)"),
        mpatches.Patch(color=WALL_COLOR, label="Wall (obstacle)"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=7.5,
              framealpha=0.92, edgecolor="#aaaaaa",
              bbox_to_anchor=(1.0, 1.0))

    # Annotate wall gap
    ax.text(1, 4.62, "gap", ha="center", fontsize=6.5, color="#666666")
    ax.text(7, 4.62, "gap", ha="center", fontsize=6.5, color="#666666")

    fig.tight_layout()
    out_path = out_dir / "fig_env_overview.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Benchmark panels (2×2)
# ─────────────────────────────────────────────────────────────────────────────
def fig_benchmarks(out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.5))
    axes = axes.flatten()

    bench_names = ["AB", "AC", "ABA", "ABC"]
    bench_subtitles = [
        "AB: New task transfer (2 tasks)",
        "AC: Positive forward transfer (2 tasks)",
        "ABA: Revisit / plasticity (3 phases)",
        "ABC: 3-task retention chain",
    ]

    for ax, bname, subtitle in zip(axes, bench_names, bench_subtitles):
        seq = BENCHMARKS[bname]
        unique_goals = list(dict.fromkeys(seq))  # preserve order, deduplicate

        draw_grid(ax, active_goals=unique_goals, show_agent=True,
                  highlight_current=seq[0], title=subtitle)

        # Draw task sequence as a timeline below the grid
        # We'll use text annotation inside the plot at y=-.45 (outside grid)
        seq_str = "  →  ".join(
            [f"Task {i+1}: {g}" for i, g in enumerate(seq)]
        )
        ax.text(4, -0.85, seq_str, ha="center", va="top", fontsize=8,
                color="#333333", style="italic",
                transform=ax.transData)

        # Color-coded phase boxes below
        box_y = -1.3
        box_w = SIZE / len(seq) - 0.15
        for i, g in enumerate(seq):
            bx = i * (SIZE / len(seq)) + box_w / 2 + 0.1 - 0.5
            fc = GOAL_COLORS[g]
            rect = mpatches.FancyBboxPatch(
                (bx - box_w / 2, box_y - 0.28), box_w, 0.56,
                boxstyle="round,pad=0.05", linewidth=1,
                facecolor=fc, edgecolor="white", alpha=0.85, zorder=5,
                transform=ax.transData, clip_on=False
            )
            ax.add_patch(rect)
            ax.text(bx, box_y, f"Phase {i+1}\nGoal {g}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", zorder=6,
                    transform=ax.transData, clip_on=False)

        ax.set_ylim(-1.8, SIZE - 0.5)  # make room for timeline

    # Shared legend at bottom
    legend_items = [
        mpatches.Patch(color=AGENT_COLOR, label="Agent start (S)"),
        mpatches.Patch(color=GOAL_COLORS["A"], label="Goal A (1,0)"),
        mpatches.Patch(color=GOAL_COLORS["B"], label="Goal B (7,0)"),
        mpatches.Patch(color=GOAL_COLORS["C"], label="Goal C (4,0)"),
        mpatches.Patch(color=WALL_COLOR, label="Wall / obstacle"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.01), fontsize=9, framealpha=0.9)

    fig.suptitle("Benchmark Task Sequences — 9×9 GridWorld\n"
                 "(* = goal location, S = fixed agent start, wall with gaps at x=1, x=7)",
                 fontsize=11, y=1.01)
    fig.tight_layout(h_pad=3.5, w_pad=2.0)
    out_path = out_dir / "fig_env_benchmarks.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Clean grid — no labels, for custom annotation in LaTeX/Inkscape
# ─────────────────────────────────────────────────────────────────────────────
def fig_clean(out_dir: Path):
    fig, ax = plt.subplots(figsize=(4.0, 4.0))

    ax.set_facecolor(FLOOR_COLOR)
    ax.set_xlim(-0.5, SIZE - 0.5)
    ax.set_ylim(-0.5, SIZE - 0.5)
    ax.set_aspect("equal")

    for i in range(SIZE + 1):
        ax.axhline(i - 0.5, color=GRID_COLOR, lw=0.7, zorder=1)
        ax.axvline(i - 0.5, color=GRID_COLOR, lw=0.7, zorder=1)

    for (ox, oy) in OBSTACLES:
        rect = mpatches.FancyBboxPatch(
            (ox - 0.48, oy - 0.48), 0.96, 0.96,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=WALL_COLOR, zorder=2
        )
        ax.add_patch(rect)

    # Goals — all three
    for label, (gx, gy) in GOALS.items():
        ax.scatter(gx, gy, s=400, c=GOAL_COLORS[label], marker="*", zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.text(gx, gy + 0.4, label, ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=GOAL_COLORS[label], zorder=6)

    # Agent
    ax.scatter(*AGENT_START, s=250, c=AGENT_COLOR, marker="o", zorder=5,
               edgecolors="white", linewidths=1.0)
    ax.text(AGENT_START[0], AGENT_START[1] - 0.4, "S", ha="center", va="top",
            fontsize=8, color=AGENT_COLOR, fontweight="bold", zorder=6)

    ax.set_xticks(range(SIZE))
    ax.set_yticks(range(SIZE))
    ax.tick_params(labelsize=7, length=2)

    fig.tight_layout()
    out_path = out_dir / "fig_env_grid_only.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Compact horizontal strip — all 4 benchmarks side by side (for paper)
# ─────────────────────────────────────────────────────────────────────────────
def fig_benchmarks_strip(out_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    bench_names = ["AB", "AC", "ABA", "ABC"]

    for ax, bname in zip(axes, bench_names):
        seq = BENCHMARKS[bname]
        unique_goals = list(dict.fromkeys(seq))

        draw_grid(ax, active_goals=unique_goals, show_agent=True,
                  highlight_current=None, title=f"Benchmark {bname}")

        # Phase labels at bottom
        phase_str = " → ".join([f"$\\mathbf{{{g}}}$" for g in seq])
        ax.text(4, -0.75, phase_str, ha="center", va="top", fontsize=9,
                color="#222222", transform=ax.transData)

        ax.set_ylim(-1.2, SIZE - 0.5)

    legend_items = [
        mpatches.Patch(color=AGENT_COLOR, label="Agent start (S)"),
        mpatches.Patch(color=GOAL_COLORS["A"], label="Goal A (1,0) — left"),
        mpatches.Patch(color=GOAL_COLORS["B"], label="Goal B (7,0) — right"),
        mpatches.Patch(color=GOAL_COLORS["C"], label="Goal C (4,0) — centre"),
        mpatches.Patch(color=WALL_COLOR, label="Wall (gaps at x=1, x=7)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.06), fontsize=9, framealpha=0.9)
    fig.suptitle("9×9 GridWorld — Benchmark Task Sequences", fontsize=12, y=1.02)
    fig.tight_layout(w_pad=1.5)
    out_path = out_dir / "fig_env_benchmarks_strip.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(out_dir: str = "."):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("Generating environment figures...")
    fig_overview(out)
    fig_benchmarks(out)
    fig_benchmarks_strip(out)
    fig_clean(out)
    print(f"\nAll environment figures saved to: {out}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    main(out_dir)
