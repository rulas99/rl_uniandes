from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from rl.rl_uniandes.drl.morphin.aggregate_morphin_results import mean_std, summarize_summaries


DEFAULT_CSV_NAMES = [
    "runs_index.csv",
    "eval_metrics_long.csv",
    "final_task_metrics_long.csv",
    "switch_metrics_long.csv",
    "switch_metrics_train_long.csv",
    "detector_events_long.csv",
    "detection_metrics_long.csv",
]
HEAVY_CSV_NAMES = [
    "episode_metrics_long.csv",
    "update_metrics_long.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine MORPHIN shard analyses")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--session-root", type=str, required=True)
    parser.add_argument("--shard-session", action="append", required=True)
    parser.add_argument("--include-heavy", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def union_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = union_fieldnames(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def combine_csv_from_dirs(source_dirs: list[Path], csv_name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source_dir in source_dirs:
        rows.extend(read_csv(source_dir / csv_name))
    return rows


def summarize_final_tasks(final_task_long: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in final_task_long:
        grouped[
            (
                str(row["benchmark"]),
                str(row["method"]),
                str(row["task_id"]),
                str(row["task_role"]),
            )
        ].append(row)

    summary_rows: list[dict[str, object]] = []
    for (benchmark, method, task_id, task_role), rows in sorted(grouped.items()):
        success_values = [float(row["success_rate"]) for row in rows]
        zero_final_runs = sum(1 for value in success_values if value <= 0.0)
        perfect_final_runs = sum(1 for value in success_values if value >= 1.0)
        final_mean, final_std = mean_std(success_values)
        summary_rows.append(
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
    return summary_rows


def summarize_switch_types(
    switch_long: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    switch_type_summary: list[dict[str, object]] = []
    benchmark_switch_type_summary: list[dict[str, object]] = []

    switch_grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    benchmark_switch_grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)

    for row in switch_long:
        switch_grouped[(str(row["method"]), str(row.get("switch_type", "unknown")))].append(row)
        benchmark_switch_grouped[
            (
                str(row["benchmark"]),
                str(row["method"]),
                str(row.get("switch_type", "unknown")),
            )
        ].append(row)

    def build_row(rows: list[dict[str, str]]) -> tuple[float, float, float, float, float, float, float, float, float, float]:
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
        return (
            auc_mean,
            auc_std,
            ttt_mean,
            ttt_std,
            gain_mean,
            gain_std,
            delta_mean,
            delta_std,
            log_gain_mean,
            log_gain_std,
        )

    for (method, switch_type), rows in sorted(switch_grouped.items()):
        (
            auc_mean,
            auc_std,
            ttt_mean,
            ttt_std,
            gain_mean,
            gain_std,
            delta_mean,
            delta_std,
            log_gain_mean,
            log_gain_std,
        ) = build_row(rows)
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

    for (benchmark, method, switch_type), rows in sorted(benchmark_switch_grouped.items()):
        (
            auc_mean,
            auc_std,
            ttt_mean,
            ttt_std,
            gain_mean,
            gain_std,
            delta_mean,
            delta_std,
            log_gain_mean,
            log_gain_std,
        ) = build_row(rows)
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

    return switch_type_summary, benchmark_switch_type_summary


def build_report(
    output_dir: Path,
    num_runs: int,
    group_rows: list[dict[str, object]],
    benchmark_method_summary: list[dict[str, object]],
    final_task_summary: list[dict[str, object]],
    switch_type_summary: list[dict[str, object]],
    benchmark_switch_type_summary: list[dict[str, object]],
) -> None:
    report_lines = [
        "# MORPHIN Experiment Report",
        "",
        f"Runs discovered: {num_runs}",
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
        grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in final_task_summary:
            grouped[(str(row["benchmark"]), str(row["method"]))].append(row)
        for (benchmark, method), rows in sorted(grouped.items()):
            task_fragments = []
            for row in sorted(rows, key=lambda item: (str(item["task_role"]) != "current_task", str(item["task_id"]))):
                task_fragments.append(
                    f"{row['task_id']}[{row['task_role']}]={float(row['final_success_mean']):.4f}"
                )
            report_lines.append(f"- `{benchmark}` / `{method}`: " + ", ".join(task_fragments))
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


def combine_scope(source_dirs: list[Path], output_dir: Path, *, include_heavy: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    merged: dict[str, list[dict[str, str]]] = {}
    csv_names = list(DEFAULT_CSV_NAMES)
    if include_heavy:
        csv_names.extend(HEAVY_CSV_NAMES)

    for csv_name in csv_names:
        rows = combine_csv_from_dirs(source_dirs, csv_name)
        merged[csv_name] = rows
        write_csv(output_dir / csv_name, rows)

    runs_index = merged["runs_index.csv"]
    switch_long = merged["switch_metrics_long.csv"]
    final_task_long = merged["final_task_metrics_long.csv"]

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    grouped_by_benchmark: dict[str, list[dict[str, object]]] = defaultdict(list)
    grouped_by_benchmark_method: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in runs_index:
        method = str(row["method"])
        benchmark = str(row["benchmark"])
        grouped[method].append(row)
        grouped_by_benchmark[benchmark].append(row)
        grouped_by_benchmark_method[(benchmark, method)].append(row)

    group_rows = [summarize_summaries(rows, method=method) for method, rows in sorted(grouped.items())]
    benchmark_summary = [
        summarize_summaries(rows, benchmark=benchmark)
        for benchmark, rows in sorted(grouped_by_benchmark.items())
    ]
    benchmark_method_summary = [
        summarize_summaries(rows, benchmark=benchmark, method=method)
        for (benchmark, method), rows in sorted(grouped_by_benchmark_method.items())
    ]
    group_rows = sorted(
        group_rows,
        key=lambda row: (
            float("-inf")
            if math.isnan(float(row["headline_recovery_auc_eval"]))
            else float(row["headline_recovery_auc_eval"]),
            float("-inf")
            if math.isnan(float(row["current_task_final_success_mean"]))
            else float(row["current_task_final_success_mean"]),
        ),
        reverse=True,
    )

    final_task_summary = summarize_final_tasks(final_task_long)
    switch_type_summary, benchmark_switch_type_summary = summarize_switch_types(switch_long)

    write_csv(output_dir / "group_summary.csv", group_rows)
    write_csv(output_dir / "benchmark_summary.csv", benchmark_summary)
    write_csv(output_dir / "benchmark_method_summary.csv", benchmark_method_summary)
    write_csv(output_dir / "final_task_summary.csv", final_task_summary)
    write_csv(output_dir / "switch_type_summary.csv", switch_type_summary)
    write_csv(output_dir / "benchmark_switch_type_summary.csv", benchmark_switch_type_summary)

    build_report(
        output_dir=output_dir,
        num_runs=len(runs_index),
        group_rows=group_rows,
        benchmark_method_summary=benchmark_method_summary,
        final_task_summary=final_task_summary,
        switch_type_summary=switch_type_summary,
        benchmark_switch_type_summary=benchmark_switch_type_summary,
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    session_root = Path(args.session_root)
    shard_sessions = [Path(p) for p in args.shard_session]
    shard_analysis_dirs = [session / "analysis" for session in shard_sessions]

    combine_scope(shard_analysis_dirs, output_dir, include_heavy=args.include_heavy)

    benchmark_names: set[str] = set()
    for analysis_dir in shard_analysis_dirs:
        by_benchmark = analysis_dir / "by_benchmark"
        if not by_benchmark.exists():
            continue
        for path in by_benchmark.iterdir():
            if path.is_dir():
                benchmark_names.add(path.name)

    by_benchmark_out = output_dir / "by_benchmark"
    for benchmark in sorted(benchmark_names):
        source_dirs = [
            session / "analysis" / "by_benchmark" / benchmark
            for session in shard_sessions
            if (session / "analysis" / "by_benchmark" / benchmark).is_dir()
        ]
        if source_dirs:
            combine_scope(source_dirs, by_benchmark_out / benchmark, include_heavy=args.include_heavy)

    (session_root / "analysis_combiner.ok").write_text("ok\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
