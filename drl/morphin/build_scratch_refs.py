from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scratch reference file from scratch_task runs")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--min-final-success", type=float, default=0.8)
    parser.add_argument("--min-valid-runs", type=int, default=3)
    parser.add_argument("--min-valid-fraction", type=float, default=0.6)
    return parser.parse_args()


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def pstdev_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir)
    output_json = Path(args.output_json)
    grouped: dict[str, dict[str, list[float]]] = {}

    for summary_path in sorted(root_dir.rglob("summary.json")):
        run_dir = summary_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        config = json.loads(config_path.read_text())
        if summary.get("mode") != "scratch_task":
            continue
        task_sequence = summary.get("task_sequence") or config.get("task_sequence") or []
        if not task_sequence:
            continue
        task_id = str(task_sequence[0] if isinstance(task_sequence[0], str) else task_sequence[0]["task_id"])
        grouped.setdefault(
            task_id,
            {
                "final_seen_success_mean": [],
                "time_to_threshold_eval_steps": [],
                "time_to_threshold_eval_episodes": [],
                "valid_final_seen_success_mean": [],
                "valid_time_to_threshold_eval_steps": [],
                "valid_time_to_threshold_eval_episodes": [],
            },
        )
        final_success = (
            float(summary["final_seen_success_mean"])
            if summary.get("final_seen_success_mean") is not None
            else None
        )
        is_valid = final_success is not None and final_success >= float(args.min_final_success)
        if summary.get("overall_time_to_threshold_eval_steps") is not None:
            value = float(summary["overall_time_to_threshold_eval_steps"])
            grouped[task_id]["time_to_threshold_eval_steps"].append(value)
            if is_valid:
                grouped[task_id]["valid_time_to_threshold_eval_steps"].append(value)
        if summary.get("overall_time_to_threshold_eval_episodes") is not None:
            value = float(summary["overall_time_to_threshold_eval_episodes"])
            grouped[task_id]["time_to_threshold_eval_episodes"].append(value)
            if is_valid:
                grouped[task_id]["valid_time_to_threshold_eval_episodes"].append(value)
        if final_success is not None:
            grouped[task_id]["final_seen_success_mean"].append(final_success)
            if is_valid:
                grouped[task_id]["valid_final_seen_success_mean"].append(final_success)

    refs: dict[str, dict[str, float | None]] = {}
    for task_id, values in grouped.items():
        valid_steps = values.get("valid_time_to_threshold_eval_steps", [])
        valid_episodes = values.get("valid_time_to_threshold_eval_episodes", [])
        valid_success = values.get("valid_final_seen_success_mean", [])
        steps_values = valid_steps if valid_steps else values.get("time_to_threshold_eval_steps", [])
        episode_values = valid_episodes if valid_episodes else values.get("time_to_threshold_eval_episodes", [])
        success_values = valid_success if valid_success else values["final_seen_success_mean"]
        num_runs_total = len(values["final_seen_success_mean"])
        num_runs_valid = len(valid_success)
        valid_run_fraction = (num_runs_valid / num_runs_total) if num_runs_total else None
        is_reference_stable = bool(
            num_runs_valid >= int(args.min_valid_runs)
            and (valid_run_fraction is not None and valid_run_fraction >= float(args.min_valid_fraction))
        )
        if num_runs_valid < int(args.min_valid_runs):
            stability_flag = "insufficient_valid_runs"
        elif valid_run_fraction is None or valid_run_fraction < float(args.min_valid_fraction):
            stability_flag = "low_valid_fraction"
        else:
            stability_flag = "ok"
        refs[task_id] = {
            "time_to_threshold_eval_steps": median_or_none(steps_values),
            "time_to_threshold_eval_episodes": median_or_none(episode_values),
            "final_seen_success_mean": mean_or_none(success_values),
            "selected_time_to_threshold_eval_steps_mean": mean_or_none(steps_values),
            "selected_time_to_threshold_eval_steps_std": pstdev_or_none(steps_values),
            "selected_time_to_threshold_eval_episodes_mean": mean_or_none(episode_values),
            "selected_time_to_threshold_eval_episodes_std": pstdev_or_none(episode_values),
            "selected_final_seen_success_std": pstdev_or_none(success_values),
            "all_time_to_threshold_eval_steps_mean": mean_or_none(values.get("time_to_threshold_eval_steps", [])),
            "all_time_to_threshold_eval_steps_std": pstdev_or_none(values.get("time_to_threshold_eval_steps", [])),
            "all_time_to_threshold_eval_episodes_mean": mean_or_none(values.get("time_to_threshold_eval_episodes", [])),
            "all_time_to_threshold_eval_episodes_std": pstdev_or_none(values.get("time_to_threshold_eval_episodes", [])),
            "all_final_seen_success_mean": mean_or_none(values["final_seen_success_mean"]),
            "all_final_seen_success_std": pstdev_or_none(values["final_seen_success_mean"]),
            "num_runs_total": num_runs_total,
            "num_runs_valid": num_runs_valid,
            "valid_run_fraction": valid_run_fraction,
            "success_threshold_used": float(args.min_final_success),
            "ref_selection": "valid_only" if valid_success else "all_runs_fallback",
            "is_reference_stable": is_reference_stable,
            "stability_flag": stability_flag,
            "min_valid_runs_required": int(args.min_valid_runs),
            "min_valid_fraction_required": float(args.min_valid_fraction),
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(refs, indent=2))
    print(json.dumps({"output_json": str(output_json), "num_tasks": len(refs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
