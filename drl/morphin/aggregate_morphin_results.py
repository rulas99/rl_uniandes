from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate MORPHIN experiment outputs")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance ** 0.5


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(path.parent for path in root_dir.rglob("summary.json"))
    runs_index: list[dict[str, object]] = []
    episode_long: list[dict[str, object]] = []
    eval_long: list[dict[str, object]] = []
    switch_long: list[dict[str, object]] = []
    detector_long: list[dict[str, object]] = []
    detection_long: list[dict[str, object]] = []

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)

    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        config = json.loads((run_dir / "config.json").read_text())
        run_row = {
            "run_dir": str(run_dir),
            "method": summary["method"],
            "benchmark": summary["benchmark"],
            "seed": summary["seed"],
            "mode": summary["mode"],
            "total_episodes": summary["total_episodes"],
            "total_steps": summary["total_steps"],
            "avg_episode_success": summary["avg_episode_success"],
            "overall_time_to_threshold": summary.get("overall_time_to_threshold"),
            "final_seen_success_mean": summary["final_seen_success_mean"],
            "current_task_final_success": summary["current_task_final_success"],
            "num_detected_drifts": summary.get("num_detected_drifts"),
            "num_false_alarms": summary.get("num_false_alarms"),
            "mean_detection_delay_episodes": summary.get("mean_detection_delay_episodes"),
            "switch_recovery_auc_success_mean": summary["switch_recovery_auc_success_mean"],
            "switch_time_to_threshold_mean": summary["switch_time_to_threshold_mean"],
            "adaptation_gain_vs_scratch_mean": summary["adaptation_gain_vs_scratch_mean"],
            "shock_drop_mean": summary["shock_drop_mean"],
        }
        runs_index.append(run_row)
        grouped[str(summary["method"])].append(summary)

        for row in read_csv(run_dir / "episode_metrics.csv"):
            episode_long.append({"run_dir": str(run_dir), **summary, **row})
        for row in read_csv(run_dir / "eval_metrics.csv"):
            eval_long.append({"run_dir": str(run_dir), **summary, **row})
        for row in read_csv(run_dir / "switch_metrics.csv"):
            switch_long.append({"run_dir": str(run_dir), **summary, **row})
        for row in read_csv(run_dir / "detector_events.csv"):
            detector_long.append({"run_dir": str(run_dir), **summary, **row})
        for row in read_csv(run_dir / "detection_metrics.csv"):
            detection_long.append({"run_dir": str(run_dir), **summary, **row})

    group_rows: list[dict[str, object]] = []
    for method, summaries in sorted(grouped.items()):
        final_values = [float(item["final_seen_success_mean"]) for item in summaries]
        threshold_values = [
            float(item["overall_time_to_threshold"])
            for item in summaries
            if item.get("overall_time_to_threshold") is not None
        ]
        adapt_values = [float(item["adaptation_gain_vs_scratch_mean"]) for item in summaries if not math.isnan(float(item["adaptation_gain_vs_scratch_mean"]))]
        recovery_values = [float(item["switch_recovery_auc_success_mean"]) for item in summaries if not math.isnan(float(item["switch_recovery_auc_success_mean"]))]
        final_mean, final_std = mean_std(final_values)
        threshold_mean, threshold_std = mean_std(threshold_values)
        recovery_mean, recovery_std = mean_std(recovery_values)
        adapt_mean, adapt_std = mean_std(adapt_values)
        group_rows.append(
            {
                "method": method,
                "num_runs": len(summaries),
                "final_seen_success_mean": final_mean,
                "final_seen_success_std": final_std,
                "overall_time_to_threshold_mean": threshold_mean,
                "overall_time_to_threshold_std": threshold_std,
                "switch_recovery_auc_success_mean": recovery_mean,
                "switch_recovery_auc_success_std": recovery_std,
                "adaptation_gain_vs_scratch_mean": adapt_mean,
                "adaptation_gain_vs_scratch_std": adapt_std,
            }
        )

    write_csv(output_dir / "runs_index.csv", runs_index)
    write_csv(output_dir / "episode_metrics_long.csv", episode_long)
    write_csv(output_dir / "eval_metrics_long.csv", eval_long)
    write_csv(output_dir / "switch_metrics_long.csv", switch_long)
    write_csv(output_dir / "detector_events_long.csv", detector_long)
    write_csv(output_dir / "detection_metrics_long.csv", detection_long)
    write_csv(output_dir / "group_summary.csv", group_rows)

    report_lines = [
        "# MORPHIN Experiment Report",
        "",
        f"Runs discovered: {len(run_dirs)}",
        "",
        "## Group Summary",
        "",
    ]
    for row in group_rows:
        report_lines.append(
            f"- `{row['method']}`: final_seen_success={row['final_seen_success_mean']:.4f} "
            f"+/- {row['final_seen_success_std']:.4f}, "
            f"recovery_auc={row['switch_recovery_auc_success_mean']:.4f} "
            f"+/- {row['switch_recovery_auc_success_std']:.4f}"
        )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n")

    print(
        json.dumps(
            {
                "num_runs": len(run_dirs),
                "outputs": {
                    "runs_index_csv": str(output_dir / "runs_index.csv"),
                    "episode_metrics_long_csv": str(output_dir / "episode_metrics_long.csv"),
                    "eval_metrics_long_csv": str(output_dir / "eval_metrics_long.csv"),
                    "switch_metrics_long_csv": str(output_dir / "switch_metrics_long.csv"),
                    "detector_events_long_csv": str(output_dir / "detector_events_long.csv"),
                    "detection_metrics_long_csv": str(output_dir / "detection_metrics_long.csv"),
                    "group_summary_csv": str(output_dir / "group_summary.csv"),
                    "report_md": str(output_dir / "report.md"),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
