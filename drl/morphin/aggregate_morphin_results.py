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


def safe_float(value: object) -> float | None:
    if value in {"", None}:
        return None
    value_f = float(value)
    if math.isnan(value_f):
        return None
    return value_f


def ordered_task_ids(task_sequence: object) -> list[str]:
    if not isinstance(task_sequence, list):
        return []
    ordered: list[str] = []
    for item in task_sequence:
        if isinstance(item, str):
            task_id = item
        elif isinstance(item, dict) and "task_id" in item:
            task_id = str(item["task_id"])
        else:
            continue
        if task_id not in ordered:
            ordered.append(task_id)
    return ordered


def mean_valid(rows: list[dict[str, str]], key: str, switch_type: str | None = None) -> float:
    values: list[float] = []
    for row in rows:
        if switch_type is not None and str(row.get("switch_type")) != switch_type:
            continue
        value = row.get(key)
        if value in {"", None}:
            continue
        value_f = float(value)
        if math.isnan(value_f):
            continue
        values.append(value_f)
    return float(sum(values) / len(values)) if values else math.nan


def summarize_summaries(
    summaries: list[dict[str, object]],
    *,
    method: str | None = None,
    benchmark: str | None = None,
) -> dict[str, object]:
    final_values = [float(item["final_seen_success_mean"]) for item in summaries]
    current_values = [float(item["current_task_final_success"]) for item in summaries]
    threshold_values = [
        float(item["overall_time_to_threshold_eval_steps"])
        for item in summaries
        if item.get("overall_time_to_threshold_eval_steps") is not None
    ]
    adapt_values = [
        float(item["adaptation_gain_vs_scratch_steps_mean"])
        for item in summaries
        if item.get("adaptation_gain_vs_scratch_steps_mean") is not None
        and not math.isnan(float(item["adaptation_gain_vs_scratch_steps_mean"]))
    ]
    recovery_values = [
        float(item["switch_recovery_auc_success_eval_mean"])
        for item in summaries
        if item.get("switch_recovery_auc_success_eval_mean") is not None
        and not math.isnan(float(item["switch_recovery_auc_success_eval_mean"]))
    ]
    new_task_recovery_values = [
        float(item["new_task_recovery_auc_success_eval_mean"])
        for item in summaries
        if item.get("new_task_recovery_auc_success_eval_mean") is not None
        and not math.isnan(float(item["new_task_recovery_auc_success_eval_mean"]))
    ]
    new_task_ttt_values = [
        float(item["new_task_time_to_threshold_eval_steps_mean"])
        for item in summaries
        if item.get("new_task_time_to_threshold_eval_steps_mean") is not None
        and not math.isnan(float(item["new_task_time_to_threshold_eval_steps_mean"]))
    ]
    ttt_delta_values = [
        float(item["switch_time_to_threshold_eval_steps_delta_vs_scratch_mean"])
        for item in summaries
        if item.get("switch_time_to_threshold_eval_steps_delta_vs_scratch_mean") is not None
        and not math.isnan(float(item["switch_time_to_threshold_eval_steps_delta_vs_scratch_mean"]))
    ]
    log_gain_values = [
        float(item["switch_log_adaptation_gain_vs_scratch_steps_mean"])
        for item in summaries
        if item.get("switch_log_adaptation_gain_vs_scratch_steps_mean") is not None
        and not math.isnan(float(item["switch_log_adaptation_gain_vs_scratch_steps_mean"]))
    ]
    new_task_ttt_delta_values = [
        float(item["new_task_time_to_threshold_eval_steps_delta_vs_scratch_mean"])
        for item in summaries
        if item.get("new_task_time_to_threshold_eval_steps_delta_vs_scratch_mean") is not None
        and not math.isnan(float(item["new_task_time_to_threshold_eval_steps_delta_vs_scratch_mean"]))
    ]
    new_task_log_gain_values = [
        float(item["new_task_log_adaptation_gain_vs_scratch_steps_mean"])
        for item in summaries
        if item.get("new_task_log_adaptation_gain_vs_scratch_steps_mean") is not None
        and not math.isnan(float(item["new_task_log_adaptation_gain_vs_scratch_steps_mean"]))
    ]
    new_task_gain_values = [
        float(item["new_task_adaptation_gain_vs_scratch_steps_mean"])
        for item in summaries
        if item.get("new_task_adaptation_gain_vs_scratch_steps_mean") is not None
        and not math.isnan(float(item["new_task_adaptation_gain_vs_scratch_steps_mean"]))
    ]
    revisit_recovery_values = [
        float(item["revisit_task_recovery_auc_success_eval_mean"])
        for item in summaries
        if item.get("revisit_task_recovery_auc_success_eval_mean") is not None
        and not math.isnan(float(item["revisit_task_recovery_auc_success_eval_mean"]))
    ]
    revisit_ttt_delta_values = [
        float(item["revisit_task_time_to_threshold_eval_steps_delta_vs_scratch_mean"])
        for item in summaries
        if item.get("revisit_task_time_to_threshold_eval_steps_delta_vs_scratch_mean") is not None
        and not math.isnan(float(item["revisit_task_time_to_threshold_eval_steps_delta_vs_scratch_mean"]))
    ]
    revisit_log_gain_values = [
        float(item["revisit_task_log_adaptation_gain_vs_scratch_steps_mean"])
        for item in summaries
        if item.get("revisit_task_log_adaptation_gain_vs_scratch_steps_mean") is not None
        and not math.isnan(float(item["revisit_task_log_adaptation_gain_vs_scratch_steps_mean"]))
    ]
    final_unique_tasks_ge_threshold_values = [
        float(item["final_unique_tasks_success_ge_threshold"])
        for item in summaries
        if item.get("final_unique_tasks_success_ge_threshold") is not None
        and not math.isnan(float(item["final_unique_tasks_success_ge_threshold"]))
    ]
    final_unique_task_fraction_values = [
        float(item["final_unique_task_success_fraction_ge_threshold"])
        for item in summaries
        if item.get("final_unique_task_success_fraction_ge_threshold") is not None
        and not math.isnan(float(item["final_unique_task_success_fraction_ge_threshold"]))
    ]
    final_all_unique_values = [
        float(item["final_all_unique_tasks_success_ge_threshold"])
        for item in summaries
        if item.get("final_all_unique_tasks_success_ge_threshold") is not None
        and not math.isnan(float(item["final_all_unique_tasks_success_ge_threshold"]))
    ]
    final_mean, final_std = mean_std(final_values)
    current_mean, current_std = mean_std(current_values)
    threshold_mean, threshold_std = mean_std(threshold_values)
    recovery_mean, recovery_std = mean_std(recovery_values)
    ttt_delta_mean, ttt_delta_std = mean_std(ttt_delta_values)
    log_gain_mean, log_gain_std = mean_std(log_gain_values)
    new_task_recovery_mean, new_task_recovery_std = mean_std(new_task_recovery_values)
    new_task_ttt_mean, new_task_ttt_std = mean_std(new_task_ttt_values)
    new_task_ttt_delta_mean, new_task_ttt_delta_std = mean_std(new_task_ttt_delta_values)
    new_task_log_gain_mean, new_task_log_gain_std = mean_std(new_task_log_gain_values)
    new_task_gain_mean, new_task_gain_std = mean_std(new_task_gain_values)
    revisit_recovery_mean, revisit_recovery_std = mean_std(revisit_recovery_values)
    revisit_ttt_delta_mean, revisit_ttt_delta_std = mean_std(revisit_ttt_delta_values)
    revisit_log_gain_mean, revisit_log_gain_std = mean_std(revisit_log_gain_values)
    final_unique_tasks_ge_threshold_mean, final_unique_tasks_ge_threshold_std = mean_std(
        final_unique_tasks_ge_threshold_values
    )
    final_unique_task_fraction_mean, final_unique_task_fraction_std = mean_std(
        final_unique_task_fraction_values
    )
    final_all_unique_rate, final_all_unique_std = mean_std(final_all_unique_values)
    adapt_mean, adapt_std = mean_std(adapt_values)
    row: dict[str, object] = {
        "num_runs": len(summaries),
        "headline_recovery_auc_eval": (
            new_task_recovery_mean if not math.isnan(new_task_recovery_mean) else recovery_mean
        ),
        "final_seen_success_mean": final_mean,
        "final_seen_success_std": final_std,
        "current_task_final_success_mean": current_mean,
        "current_task_final_success_std": current_std,
        "overall_time_to_threshold_mean": threshold_mean,
        "overall_time_to_threshold_std": threshold_std,
        "switch_recovery_auc_success_eval_mean": recovery_mean,
        "switch_recovery_auc_success_eval_std": recovery_std,
        "switch_time_to_threshold_eval_steps_delta_vs_scratch_mean": ttt_delta_mean,
        "switch_time_to_threshold_eval_steps_delta_vs_scratch_std": ttt_delta_std,
        "switch_log_adaptation_gain_vs_scratch_steps_mean": log_gain_mean,
        "switch_log_adaptation_gain_vs_scratch_steps_std": log_gain_std,
        "adaptation_gain_vs_scratch_steps_mean": adapt_mean,
        "adaptation_gain_vs_scratch_steps_std": adapt_std,
        "new_task_recovery_auc_success_eval_mean": new_task_recovery_mean,
        "new_task_recovery_auc_success_eval_std": new_task_recovery_std,
        "new_task_time_to_threshold_eval_steps_mean": new_task_ttt_mean,
        "new_task_time_to_threshold_eval_steps_std": new_task_ttt_std,
        "new_task_time_to_threshold_eval_steps_delta_vs_scratch_mean": new_task_ttt_delta_mean,
        "new_task_time_to_threshold_eval_steps_delta_vs_scratch_std": new_task_ttt_delta_std,
        "new_task_log_adaptation_gain_vs_scratch_steps_mean": new_task_log_gain_mean,
        "new_task_log_adaptation_gain_vs_scratch_steps_std": new_task_log_gain_std,
        "new_task_adaptation_gain_vs_scratch_steps_mean": new_task_gain_mean,
        "new_task_adaptation_gain_vs_scratch_steps_std": new_task_gain_std,
        "revisit_task_recovery_auc_success_eval_mean": revisit_recovery_mean,
        "revisit_task_recovery_auc_success_eval_std": revisit_recovery_std,
        "revisit_task_time_to_threshold_eval_steps_delta_vs_scratch_mean": revisit_ttt_delta_mean,
        "revisit_task_time_to_threshold_eval_steps_delta_vs_scratch_std": revisit_ttt_delta_std,
        "revisit_task_log_adaptation_gain_vs_scratch_steps_mean": revisit_log_gain_mean,
        "revisit_task_log_adaptation_gain_vs_scratch_steps_std": revisit_log_gain_std,
        "final_unique_tasks_success_ge_threshold_mean": final_unique_tasks_ge_threshold_mean,
        "final_unique_tasks_success_ge_threshold_std": final_unique_tasks_ge_threshold_std,
        "final_unique_task_success_fraction_ge_threshold_mean": final_unique_task_fraction_mean,
        "final_unique_task_success_fraction_ge_threshold_std": final_unique_task_fraction_std,
        "final_all_unique_tasks_success_ge_threshold_rate": final_all_unique_rate,
        "final_all_unique_tasks_success_ge_threshold_std": final_all_unique_std,
        "final_all_unique_tasks_success_ge_threshold_count": int(sum(1 for value in final_all_unique_values if value >= 1.0)),
    }
    if method is not None:
        row["method"] = method
    if benchmark is not None:
        row["benchmark"] = benchmark
    return row


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(path.parent for path in root_dir.rglob("summary.json"))
    runs_index: list[dict[str, object]] = []
    episode_long: list[dict[str, object]] = []
    eval_long: list[dict[str, object]] = []
    final_task_long: list[dict[str, object]] = []
    update_long: list[dict[str, object]] = []
    switch_long: list[dict[str, object]] = []
    train_switch_long: list[dict[str, object]] = []
    detector_long: list[dict[str, object]] = []
    detection_long: list[dict[str, object]] = []
    switch_type_summary: list[dict[str, object]] = []
    benchmark_summary: list[dict[str, object]] = []
    benchmark_method_summary: list[dict[str, object]] = []
    benchmark_switch_type_summary: list[dict[str, object]] = []
    final_task_summary: list[dict[str, object]] = []

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    grouped_by_benchmark: dict[str, list[dict[str, object]]] = defaultdict(list)
    grouped_by_benchmark_method: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        config = json.loads((run_dir / "config.json").read_text())
        switch_rows_for_run = read_csv(run_dir / "switch_metrics.csv")
        derived_summary_metrics = {
            "switch_recovery_auc_success_eval_mean": mean_valid(
                switch_rows_for_run, "recovery_auc_success_eval"
            ),
            "switch_time_to_threshold_eval_steps_mean": mean_valid(
                switch_rows_for_run, "time_to_threshold_eval_steps"
            ),
            "switch_time_to_threshold_eval_steps_delta_vs_scratch_mean": mean_valid(
                switch_rows_for_run, "time_to_threshold_eval_steps_delta_vs_scratch"
            ),
            "switch_log_adaptation_gain_vs_scratch_steps_mean": mean_valid(
                switch_rows_for_run, "log_adaptation_gain_vs_scratch_steps"
            ),
            "new_task_recovery_auc_success_eval_mean": mean_valid(
                switch_rows_for_run, "recovery_auc_success_eval", switch_type="new_task"
            ),
            "new_task_time_to_threshold_eval_steps_mean": mean_valid(
                switch_rows_for_run, "time_to_threshold_eval_steps", switch_type="new_task"
            ),
            "new_task_time_to_threshold_eval_steps_delta_vs_scratch_mean": mean_valid(
                switch_rows_for_run, "time_to_threshold_eval_steps_delta_vs_scratch", switch_type="new_task"
            ),
            "new_task_log_adaptation_gain_vs_scratch_steps_mean": mean_valid(
                switch_rows_for_run, "log_adaptation_gain_vs_scratch_steps", switch_type="new_task"
            ),
            "new_task_adaptation_gain_vs_scratch_steps_mean": mean_valid(
                switch_rows_for_run, "adaptation_gain_vs_scratch_steps", switch_type="new_task"
            ),
            "revisit_task_recovery_auc_success_eval_mean": mean_valid(
                switch_rows_for_run, "recovery_auc_success_eval", switch_type="revisit_task"
            ),
            "revisit_task_time_to_threshold_eval_steps_delta_vs_scratch_mean": mean_valid(
                switch_rows_for_run, "time_to_threshold_eval_steps_delta_vs_scratch", switch_type="revisit_task"
            ),
            "revisit_task_log_adaptation_gain_vs_scratch_steps_mean": mean_valid(
                switch_rows_for_run, "log_adaptation_gain_vs_scratch_steps", switch_type="revisit_task"
            ),
        }
        summary = {
            **summary,
            **{k: v for k, v in derived_summary_metrics.items() if k not in summary or summary.get(k) is None},
        }
        task_sequence = summary.get("task_sequence") or config.get("task_sequence") or []
        unique_task_ids = ordered_task_ids(task_sequence)
        success_threshold = safe_float(config.get("success_threshold")) or 0.8
        final_unique_tasks_success_ge_threshold = 0
        final_unique_task_success_fraction_ge_threshold = math.nan
        final_all_unique_tasks_success_ge_threshold = 0
        final_eval_success_by_task: dict[str, float] = {}
        eval_rows_for_run = read_csv(run_dir / "eval_metrics.csv")
        final_eval_rows_for_run = [
            row for row in eval_rows_for_run if str(row.get("eval_scope")) == "seen_tasks_end"
        ]
        if final_eval_rows_for_run:
            last_episode = max(int(row["episode"]) for row in final_eval_rows_for_run)
            final_eval_rows_for_run = [
                row for row in final_eval_rows_for_run if int(row["episode"]) == last_episode
            ]
            for row in final_eval_rows_for_run:
                success_rate = safe_float(row.get("success_rate"))
                if success_rate is not None:
                    final_eval_success_by_task[str(row["task_id"])] = success_rate
            final_unique_tasks_success_ge_threshold = sum(
                1
                for task_id in unique_task_ids
                if final_eval_success_by_task.get(task_id, -math.inf) >= success_threshold
            )
            if unique_task_ids:
                final_unique_task_success_fraction_ge_threshold = (
                    final_unique_tasks_success_ge_threshold / len(unique_task_ids)
                )
                final_all_unique_tasks_success_ge_threshold = int(
                    final_unique_tasks_success_ge_threshold == len(unique_task_ids)
                )
        summary.update(
            {
                "num_unique_tasks": len(unique_task_ids),
                "success_threshold": success_threshold,
                "final_unique_tasks_success_ge_threshold": final_unique_tasks_success_ge_threshold,
                "final_unique_task_success_fraction_ge_threshold": final_unique_task_success_fraction_ge_threshold,
                "final_all_unique_tasks_success_ge_threshold": final_all_unique_tasks_success_ge_threshold,
            }
        )
        run_row = {
            "run_dir": str(run_dir),
            "method": summary["method"],
            "benchmark": summary["benchmark"],
            "seed": summary["seed"],
            "mode": summary["mode"],
            "total_episodes": summary["total_episodes"],
            "total_steps": summary["total_steps"],
            "avg_episode_success": summary["avg_episode_success"],
            "overall_time_to_threshold_eval_steps": summary.get("overall_time_to_threshold_eval_steps"),
            "overall_time_to_threshold_eval_episodes": summary.get("overall_time_to_threshold_eval_episodes"),
            "final_seen_success_mean": summary["final_seen_success_mean"],
            "current_task_final_success": summary["current_task_final_success"],
            "num_detected_drifts": summary.get("num_detected_drifts"),
            "num_false_alarms": summary.get("num_false_alarms"),
            "mean_detection_delay_episodes": summary.get("mean_detection_delay_episodes"),
            "switch_recovery_auc_success_eval_mean": summary.get("switch_recovery_auc_success_eval_mean"),
            "switch_time_to_threshold_eval_steps_mean": summary.get("switch_time_to_threshold_eval_steps_mean"),
            "switch_time_to_threshold_eval_steps_delta_vs_scratch_mean": summary.get("switch_time_to_threshold_eval_steps_delta_vs_scratch_mean"),
            "switch_log_adaptation_gain_vs_scratch_steps_mean": summary.get("switch_log_adaptation_gain_vs_scratch_steps_mean"),
            "adaptation_gain_vs_scratch_steps_mean": summary.get("adaptation_gain_vs_scratch_steps_mean"),
            "initial_gap_vs_scratch_mean": summary.get("initial_gap_vs_scratch_mean"),
            "new_task_recovery_auc_success_eval_mean": summary.get("new_task_recovery_auc_success_eval_mean"),
            "new_task_time_to_threshold_eval_steps_mean": summary.get("new_task_time_to_threshold_eval_steps_mean"),
            "new_task_time_to_threshold_eval_steps_delta_vs_scratch_mean": summary.get("new_task_time_to_threshold_eval_steps_delta_vs_scratch_mean"),
            "new_task_log_adaptation_gain_vs_scratch_steps_mean": summary.get("new_task_log_adaptation_gain_vs_scratch_steps_mean"),
            "new_task_adaptation_gain_vs_scratch_steps_mean": summary.get("new_task_adaptation_gain_vs_scratch_steps_mean"),
            "revisit_task_recovery_auc_success_eval_mean": summary.get("revisit_task_recovery_auc_success_eval_mean"),
            "revisit_task_time_to_threshold_eval_steps_delta_vs_scratch_mean": summary.get("revisit_task_time_to_threshold_eval_steps_delta_vs_scratch_mean"),
            "revisit_task_log_adaptation_gain_vs_scratch_steps_mean": summary.get("revisit_task_log_adaptation_gain_vs_scratch_steps_mean"),
            "num_unique_tasks": summary.get("num_unique_tasks"),
            "success_threshold": summary.get("success_threshold"),
            "final_unique_tasks_success_ge_threshold": summary.get("final_unique_tasks_success_ge_threshold"),
            "final_unique_task_success_fraction_ge_threshold": summary.get("final_unique_task_success_fraction_ge_threshold"),
            "final_all_unique_tasks_success_ge_threshold": summary.get("final_all_unique_tasks_success_ge_threshold"),
            "threshold_min_consecutive_evals": summary.get("threshold_min_consecutive_evals"),
        }
        runs_index.append(run_row)
        grouped[str(summary["method"])].append(summary)
        grouped_by_benchmark[str(summary["benchmark"])].append(summary)
        grouped_by_benchmark_method[(str(summary["benchmark"]), str(summary["method"]))].append(summary)

        for row in read_csv(run_dir / "episode_metrics.csv"):
            episode_long.append({"run_dir": str(run_dir), **summary, **row})
        for row in eval_rows_for_run:
            eval_long.append({"run_dir": str(run_dir), **summary, **row})
        if final_eval_rows_for_run:
            current_task_id = ""
            if task_sequence:
                last_task = task_sequence[-1]
                current_task_id = str(last_task if isinstance(last_task, str) else last_task["task_id"])
            for row in final_eval_rows_for_run:
                final_task_long.append(
                    {
                        "run_dir": str(run_dir),
                        **summary,
                        **row,
                        "task_role": (
                            "current_task"
                            if str(row["task_id"]) == current_task_id
                            else "previous_task"
                        ),
                    }
                )
        for row in read_csv(run_dir / "update_metrics.csv"):
            update_long.append({"run_dir": str(run_dir), **summary, **row})
        for row in switch_rows_for_run:
            switch_long.append({"run_dir": str(run_dir), **summary, **row})
        for row in read_csv(run_dir / "switch_metrics_train.csv"):
            train_switch_long.append({"run_dir": str(run_dir), **summary, **row})
        for row in read_csv(run_dir / "detector_events.csv"):
            detector_long.append({"run_dir": str(run_dir), **summary, **row})
        for row in read_csv(run_dir / "detection_metrics.csv"):
            detection_long.append({"run_dir": str(run_dir), **summary, **row})

    group_rows: list[dict[str, object]] = []
    for method, summaries in sorted(grouped.items()):
        group_rows.append(summarize_summaries(summaries, method=method))

    for benchmark, summaries in sorted(grouped_by_benchmark.items()):
        benchmark_summary.append(summarize_summaries(summaries, benchmark=benchmark))

    for (benchmark, method), summaries in sorted(grouped_by_benchmark_method.items()):
        benchmark_method_summary.append(
            summarize_summaries(summaries, benchmark=benchmark, method=method)
        )

    group_rows = sorted(
        group_rows,
        key=lambda row: (
            float("-inf") if math.isnan(float(row["headline_recovery_auc_eval"])) else float(row["headline_recovery_auc_eval"]),
            float("-inf") if math.isnan(float(row["current_task_final_success_mean"])) else float(row["current_task_final_success_mean"]),
        ),
        reverse=True,
    )

    write_csv(output_dir / "runs_index.csv", runs_index)
    write_csv(output_dir / "episode_metrics_long.csv", episode_long)
    write_csv(output_dir / "eval_metrics_long.csv", eval_long)
    write_csv(output_dir / "final_task_metrics_long.csv", final_task_long)
    write_csv(output_dir / "update_metrics_long.csv", update_long)
    write_csv(output_dir / "switch_metrics_long.csv", switch_long)
    write_csv(output_dir / "switch_metrics_train_long.csv", train_switch_long)
    write_csv(output_dir / "detector_events_long.csv", detector_long)
    write_csv(output_dir / "detection_metrics_long.csv", detection_long)
    write_csv(output_dir / "group_summary.csv", group_rows)
    write_csv(output_dir / "benchmark_summary.csv", benchmark_summary)
    write_csv(output_dir / "benchmark_method_summary.csv", benchmark_method_summary)

    final_task_grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in final_task_long:
        final_task_grouped[
            (
                str(row["benchmark"]),
                str(row["method"]),
                str(row["task_id"]),
                str(row["task_role"]),
            )
        ].append(row)
    for (benchmark, method, task_id, task_role), rows in sorted(final_task_grouped.items()):
        success_values = [float(row["success_rate"]) for row in rows]
        zero_final_runs = sum(1 for value in success_values if value <= 0.0)
        perfect_final_runs = sum(1 for value in success_values if value >= 1.0)
        final_mean, final_std = mean_std(success_values)
        final_task_summary.append(
            {
                "benchmark": benchmark,
                "method": method,
                "task_id": task_id,
                "task_role": task_role,
                "num_runs": len(rows),
                "final_success_mean": final_mean,
                "final_success_std": final_std,
                "num_zero_final_runs": zero_final_runs,
                "num_perfect_final_runs": perfect_final_runs,
            }
        )
    write_csv(output_dir / "final_task_summary.csv", final_task_summary)

    switch_grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    benchmark_switch_grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in switch_long:
        switch_grouped[(str(row["method"]), str(row.get("switch_type", "unknown")))].append(row)
        benchmark_switch_grouped[
            (
                str(row["benchmark"]),
                str(row["method"]),
                str(row.get("switch_type", "unknown")),
            )
        ].append(row)
    for (method, switch_type), rows in sorted(switch_grouped.items()):
        auc_values = [float(row["recovery_auc_success_eval"]) for row in rows]
        ttt_values = [
            float(row["time_to_threshold_eval_steps"])
            for row in rows
            if row.get("time_to_threshold_eval_steps") not in {"", None}
        ]
        gain_values = [
            float(row["adaptation_gain_vs_scratch_steps"])
            for row in rows
            if row.get("adaptation_gain_vs_scratch_steps") not in {"", None}
            and not math.isnan(float(row["adaptation_gain_vs_scratch_steps"]))
        ]
        delta_values = [
            float(row["time_to_threshold_eval_steps_delta_vs_scratch"])
            for row in rows
            if row.get("time_to_threshold_eval_steps_delta_vs_scratch") not in {"", None}
            and not math.isnan(float(row["time_to_threshold_eval_steps_delta_vs_scratch"]))
        ]
        log_gain_values = [
            float(row["log_adaptation_gain_vs_scratch_steps"])
            for row in rows
            if row.get("log_adaptation_gain_vs_scratch_steps") not in {"", None}
            and not math.isnan(float(row["log_adaptation_gain_vs_scratch_steps"]))
        ]
        auc_mean, auc_std = mean_std(auc_values)
        ttt_mean, ttt_std = mean_std(ttt_values)
        gain_mean, gain_std = mean_std(gain_values)
        delta_mean, delta_std = mean_std(delta_values)
        log_gain_mean, log_gain_std = mean_std(log_gain_values)
        switch_type_summary.append(
            {
                "method": method,
                "switch_type": switch_type,
                "num_switches": len(rows),
                "recovery_auc_success_eval_mean": auc_mean,
                "recovery_auc_success_eval_std": auc_std,
                "time_to_threshold_eval_steps_mean": ttt_mean,
                "time_to_threshold_eval_steps_std": ttt_std,
                "time_to_threshold_eval_steps_delta_vs_scratch_mean": delta_mean,
                "time_to_threshold_eval_steps_delta_vs_scratch_std": delta_std,
                "log_adaptation_gain_vs_scratch_steps_mean": log_gain_mean,
                "log_adaptation_gain_vs_scratch_steps_std": log_gain_std,
                "adaptation_gain_vs_scratch_steps_mean": gain_mean,
                "adaptation_gain_vs_scratch_steps_std": gain_std,
            }
        )
    write_csv(output_dir / "switch_type_summary.csv", switch_type_summary)
    for (benchmark, method, switch_type), rows in sorted(benchmark_switch_grouped.items()):
        auc_values = [float(row["recovery_auc_success_eval"]) for row in rows]
        ttt_values = [
            float(row["time_to_threshold_eval_steps"])
            for row in rows
            if row.get("time_to_threshold_eval_steps") not in {"", None}
        ]
        gain_values = [
            float(row["adaptation_gain_vs_scratch_steps"])
            for row in rows
            if row.get("adaptation_gain_vs_scratch_steps") not in {"", None}
            and not math.isnan(float(row["adaptation_gain_vs_scratch_steps"]))
        ]
        delta_values = [
            float(row["time_to_threshold_eval_steps_delta_vs_scratch"])
            for row in rows
            if row.get("time_to_threshold_eval_steps_delta_vs_scratch") not in {"", None}
            and not math.isnan(float(row["time_to_threshold_eval_steps_delta_vs_scratch"]))
        ]
        log_gain_values = [
            float(row["log_adaptation_gain_vs_scratch_steps"])
            for row in rows
            if row.get("log_adaptation_gain_vs_scratch_steps") not in {"", None}
            and not math.isnan(float(row["log_adaptation_gain_vs_scratch_steps"]))
        ]
        auc_mean, auc_std = mean_std(auc_values)
        ttt_mean, ttt_std = mean_std(ttt_values)
        gain_mean, gain_std = mean_std(gain_values)
        delta_mean, delta_std = mean_std(delta_values)
        log_gain_mean, log_gain_std = mean_std(log_gain_values)
        benchmark_switch_type_summary.append(
            {
                "benchmark": benchmark,
                "method": method,
                "switch_type": switch_type,
                "num_switches": len(rows),
                "recovery_auc_success_eval_mean": auc_mean,
                "recovery_auc_success_eval_std": auc_std,
                "time_to_threshold_eval_steps_mean": ttt_mean,
                "time_to_threshold_eval_steps_std": ttt_std,
                "time_to_threshold_eval_steps_delta_vs_scratch_mean": delta_mean,
                "time_to_threshold_eval_steps_delta_vs_scratch_std": delta_std,
                "log_adaptation_gain_vs_scratch_steps_mean": log_gain_mean,
                "log_adaptation_gain_vs_scratch_steps_std": log_gain_std,
                "adaptation_gain_vs_scratch_steps_mean": gain_mean,
                "adaptation_gain_vs_scratch_steps_std": gain_std,
            }
        )
    write_csv(output_dir / "benchmark_switch_type_summary.csv", benchmark_switch_type_summary)

    report_lines = [
        "# MORPHIN Experiment Report",
        "",
        f"Runs discovered: {len(run_dirs)}",
        "",
        "Headline ranking uses `recovery_auc_eval` first and `current_task_final_success` second.",
        "Continual comparisons emphasize per-switch metrics; `overall_time_to_threshold` is diagnostic only.",
        "",
        "## Group Summary",
        "",
    ]
    for row in group_rows:
        report_lines.append(
            f"- `{row['method']}`: final_seen_success={row['final_seen_success_mean']:.4f} "
            f"+/- {row['final_seen_success_std']:.4f}, "
            f"current_task_final_success={row['current_task_final_success_mean']:.4f} "
            f"+/- {row['current_task_final_success_std']:.4f}, "
            f"all_unique_final_rate={row['final_all_unique_tasks_success_ge_threshold_rate']:.4f} "
            f"({int(row['final_all_unique_tasks_success_ge_threshold_count'])}/{int(row['num_runs'])}), "
            f"final_unique_tasks_ge_thr={row['final_unique_tasks_success_ge_threshold_mean']:.4f} "
            f"+/- {row['final_unique_tasks_success_ge_threshold_std']:.4f}, "
            f"headline_recovery_auc_eval={row['headline_recovery_auc_eval']:.4f}, "
            f"new_task_auc={row['new_task_recovery_auc_success_eval_mean']:.4f} "
            f"+/- {row['new_task_recovery_auc_success_eval_std']:.4f}, "
            f"new_task_ttt_steps={row['new_task_time_to_threshold_eval_steps_mean']:.2f} "
            f"+/- {row['new_task_time_to_threshold_eval_steps_std']:.2f}, "
            f"ttt_delta_vs_scratch={row['new_task_time_to_threshold_eval_steps_delta_vs_scratch_mean']:.2f} "
            f"+/- {row['new_task_time_to_threshold_eval_steps_delta_vs_scratch_std']:.2f}, "
            f"log_gain_vs_scratch={row['new_task_log_adaptation_gain_vs_scratch_steps_mean']:.4f} "
            f"+/- {row['new_task_log_adaptation_gain_vs_scratch_steps_std']:.4f}"
        )
    if benchmark_method_summary:
        report_lines.extend(
            [
                "",
                "## Benchmark x Method Summary",
                "",
            ]
        )
        for row in benchmark_method_summary:
            report_lines.append(
                f"- `{row['benchmark']}` / `{row['method']}`: "
                f"headline_recovery_auc_eval={row['headline_recovery_auc_eval']:.4f}, "
                f"all_unique_final_rate={row['final_all_unique_tasks_success_ge_threshold_rate']:.4f} "
                f"({int(row['final_all_unique_tasks_success_ge_threshold_count'])}/{int(row['num_runs'])}), "
                f"final_unique_tasks_ge_thr={row['final_unique_tasks_success_ge_threshold_mean']:.4f}, "
                f"new_task_auc={row['new_task_recovery_auc_success_eval_mean']:.4f} "
                f"+/- {row['new_task_recovery_auc_success_eval_std']:.4f}, "
                f"new_task_ttt_steps={row['new_task_time_to_threshold_eval_steps_mean']:.2f}, "
                f"ttt_delta_vs_scratch={row['new_task_time_to_threshold_eval_steps_delta_vs_scratch_mean']:.2f}, "
                f"log_gain_vs_scratch={row['new_task_log_adaptation_gain_vs_scratch_steps_mean']:.4f}, "
                f"current_task_final_success={row['current_task_final_success_mean']:.4f}, "
                f"final_seen_success={row['final_seen_success_mean']:.4f}"
            )
    if group_rows:
        report_lines.extend(
            [
                "",
                "## Final Retention Summary",
                "",
            ]
        )
        for row in group_rows:
            report_lines.append(
                f"- `{row['method']}`: "
                f"all_unique_final_rate={row['final_all_unique_tasks_success_ge_threshold_rate']:.4f} "
                f"({int(row['final_all_unique_tasks_success_ge_threshold_count'])}/{int(row['num_runs'])}), "
                f"mean_unique_tasks_ge_thr={row['final_unique_tasks_success_ge_threshold_mean']:.4f} "
                f"+/- {row['final_unique_tasks_success_ge_threshold_std']:.4f}, "
                f"mean_unique_task_fraction_ge_thr={row['final_unique_task_success_fraction_ge_threshold_mean']:.4f} "
                f"+/- {row['final_unique_task_success_fraction_ge_threshold_std']:.4f}"
            )
    if final_task_summary:
        report_lines.extend(
            [
                "",
                "## Final Task Summary",
                "",
            ]
        )
        final_task_summary_grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in final_task_summary:
            final_task_summary_grouped[(str(row["benchmark"]), str(row["method"]))].append(row)
        for (benchmark, method), rows in sorted(final_task_summary_grouped.items()):
            task_fragments = []
            for row in sorted(rows, key=lambda item: (str(item["task_role"]) != "current_task", str(item["task_id"]))):
                task_fragments.append(
                    f"{row['task_id']}[{row['task_role']}]={float(row['final_success_mean']):.4f}"
                )
            report_lines.append(
                f"- `{benchmark}` / `{method}`: " + ", ".join(task_fragments)
            )
    if switch_type_summary:
        report_lines.extend(
            [
                "",
                "## Switch-Type Summary",
                "",
            ]
        )
        for row in switch_type_summary:
            report_lines.append(
                f"- `{row['method']}` / `{row['switch_type']}`: "
                f"recovery_auc_eval={row['recovery_auc_success_eval_mean']:.4f} "
                f"+/- {row['recovery_auc_success_eval_std']:.4f}, "
                f"ttt_steps={row['time_to_threshold_eval_steps_mean']:.2f} "
                f"+/- {row['time_to_threshold_eval_steps_std']:.2f}, "
                f"ttt_delta_vs_scratch={row['time_to_threshold_eval_steps_delta_vs_scratch_mean']:.2f} "
                f"+/- {row['time_to_threshold_eval_steps_delta_vs_scratch_std']:.2f}, "
                f"log_gain_vs_scratch={row['log_adaptation_gain_vs_scratch_steps_mean']:.4f} "
                f"+/- {row['log_adaptation_gain_vs_scratch_steps_std']:.4f}"
            )
    if benchmark_switch_type_summary:
        report_lines.extend(
            [
                "",
                "## Benchmark x Switch-Type Summary",
                "",
            ]
        )
        for row in benchmark_switch_type_summary:
            report_lines.append(
                f"- `{row['benchmark']}` / `{row['method']}` / `{row['switch_type']}`: "
                f"recovery_auc_eval={row['recovery_auc_success_eval_mean']:.4f} "
                f"+/- {row['recovery_auc_success_eval_std']:.4f}, "
                f"ttt_steps={row['time_to_threshold_eval_steps_mean']:.2f} "
                f"+/- {row['time_to_threshold_eval_steps_std']:.2f}, "
                f"ttt_delta_vs_scratch={row['time_to_threshold_eval_steps_delta_vs_scratch_mean']:.2f} "
                f"+/- {row['time_to_threshold_eval_steps_delta_vs_scratch_std']:.2f}, "
                f"log_gain_vs_scratch={row['log_adaptation_gain_vs_scratch_steps_mean']:.4f} "
                f"+/- {row['log_adaptation_gain_vs_scratch_steps_std']:.4f}"
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
                    "final_task_metrics_long_csv": str(output_dir / "final_task_metrics_long.csv"),
                    "update_metrics_long_csv": str(output_dir / "update_metrics_long.csv"),
                    "switch_metrics_long_csv": str(output_dir / "switch_metrics_long.csv"),
                    "switch_metrics_train_long_csv": str(output_dir / "switch_metrics_train_long.csv"),
                    "detector_events_long_csv": str(output_dir / "detector_events_long.csv"),
                    "detection_metrics_long_csv": str(output_dir / "detection_metrics_long.csv"),
                    "benchmark_summary_csv": str(output_dir / "benchmark_summary.csv"),
                    "benchmark_method_summary_csv": str(output_dir / "benchmark_method_summary.csv"),
                    "switch_type_summary_csv": str(output_dir / "switch_type_summary.csv"),
                    "benchmark_switch_type_summary_csv": str(output_dir / "benchmark_switch_type_summary.csv"),
                    "final_task_summary_csv": str(output_dir / "final_task_summary.csv"),
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
