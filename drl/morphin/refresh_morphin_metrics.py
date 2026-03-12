from __future__ import annotations

import argparse
import csv
import json
import math
import importlib.util
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    module_path = Path(__file__).resolve().parent / "analysis_metrics.py"
    spec = importlib.util.spec_from_file_location("analysis_metrics_local", module_path)
    analysis_metrics = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(analysis_metrics)
    compute_eval_switch_metrics = analysis_metrics.compute_eval_switch_metrics
    compute_train_switch_metrics = analysis_metrics.compute_train_switch_metrics
else:
    from .analysis_metrics import compute_eval_switch_metrics
    from .analysis_metrics import compute_train_switch_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh derived MORPHIN switch metrics from existing run logs")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--scratch-refs-map-json", type=str, default="")
    parser.add_argument("--refresh-summary", type=int, default=1)
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


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def infer_scratch_refs_map(root_dir: Path, explicit_path: str) -> dict[str, str]:
    if explicit_path:
        return json.loads(Path(explicit_path).read_text())
    for candidate_root in [root_dir, *root_dir.parents]:
        candidate = candidate_root / "scratch_refs_by_benchmark.json"
        if candidate.exists():
            return json.loads(candidate.read_text())
    return {}


def mean_valid(rows: list[dict[str, object]], key: str, switch_type: str | None = None) -> float:
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
    return float(np.mean(values)) if values else math.nan


def refresh_summary(
    summary: dict[str, object],
    eval_switch_rows: list[dict[str, object]],
    train_switch_rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        **summary,
        "num_switches": int(len(eval_switch_rows)),
        "num_new_task_switches": int(sum(1 for row in eval_switch_rows if str(row.get("switch_type")) == "new_task")),
        "num_revisit_task_switches": int(sum(1 for row in eval_switch_rows if str(row.get("switch_type")) == "revisit_task")),
        "switch_recovery_auc_success_eval_mean": mean_valid(eval_switch_rows, "recovery_auc_success_eval"),
        "switch_time_to_threshold_eval_episodes_mean": mean_valid(eval_switch_rows, "time_to_threshold_eval_episodes"),
        "switch_time_to_threshold_eval_steps_mean": mean_valid(eval_switch_rows, "time_to_threshold_eval_steps"),
        "adaptation_gain_vs_scratch_steps_mean": mean_valid(eval_switch_rows, "adaptation_gain_vs_scratch_steps"),
        "switch_time_to_threshold_eval_steps_delta_vs_scratch_mean": mean_valid(
            eval_switch_rows, "time_to_threshold_eval_steps_delta_vs_scratch"
        ),
        "switch_log_adaptation_gain_vs_scratch_steps_mean": mean_valid(
            eval_switch_rows, "log_adaptation_gain_vs_scratch_steps"
        ),
        "adaptation_gain_vs_scratch_episodes_mean": mean_valid(
            eval_switch_rows, "adaptation_gain_vs_scratch_episodes"
        ),
        "initial_gap_vs_scratch_mean": mean_valid(eval_switch_rows, "initial_gap_vs_scratch"),
        "new_task_recovery_auc_success_eval_mean": mean_valid(
            eval_switch_rows, "recovery_auc_success_eval", switch_type="new_task"
        ),
        "new_task_time_to_threshold_eval_episodes_mean": mean_valid(
            eval_switch_rows, "time_to_threshold_eval_episodes", switch_type="new_task"
        ),
        "new_task_time_to_threshold_eval_steps_mean": mean_valid(
            eval_switch_rows, "time_to_threshold_eval_steps", switch_type="new_task"
        ),
        "new_task_adaptation_gain_vs_scratch_steps_mean": mean_valid(
            eval_switch_rows, "adaptation_gain_vs_scratch_steps", switch_type="new_task"
        ),
        "new_task_time_to_threshold_eval_steps_delta_vs_scratch_mean": mean_valid(
            eval_switch_rows, "time_to_threshold_eval_steps_delta_vs_scratch", switch_type="new_task"
        ),
        "new_task_log_adaptation_gain_vs_scratch_steps_mean": mean_valid(
            eval_switch_rows, "log_adaptation_gain_vs_scratch_steps", switch_type="new_task"
        ),
        "new_task_initial_gap_vs_scratch_mean": mean_valid(
            eval_switch_rows, "initial_gap_vs_scratch", switch_type="new_task"
        ),
        "revisit_task_recovery_auc_success_eval_mean": mean_valid(
            eval_switch_rows, "recovery_auc_success_eval", switch_type="revisit_task"
        ),
        "revisit_task_time_to_threshold_eval_episodes_mean": mean_valid(
            eval_switch_rows, "time_to_threshold_eval_episodes", switch_type="revisit_task"
        ),
        "revisit_task_time_to_threshold_eval_steps_mean": mean_valid(
            eval_switch_rows, "time_to_threshold_eval_steps", switch_type="revisit_task"
        ),
        "revisit_task_adaptation_gain_vs_scratch_steps_mean": mean_valid(
            eval_switch_rows, "adaptation_gain_vs_scratch_steps", switch_type="revisit_task"
        ),
        "revisit_task_time_to_threshold_eval_steps_delta_vs_scratch_mean": mean_valid(
            eval_switch_rows, "time_to_threshold_eval_steps_delta_vs_scratch", switch_type="revisit_task"
        ),
        "revisit_task_log_adaptation_gain_vs_scratch_steps_mean": mean_valid(
            eval_switch_rows, "log_adaptation_gain_vs_scratch_steps", switch_type="revisit_task"
        ),
        "train_switch_recovery_auc_success_mean": mean_valid(train_switch_rows, "recovery_auc_success"),
    }


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir)
    scratch_refs_map = infer_scratch_refs_map(root_dir=root_dir, explicit_path=args.scratch_refs_map_json)
    updated_runs = 0

    for run_dir in sorted(path.parent for path in root_dir.rglob("summary.json")):
        summary_path = run_dir / "summary.json"
        config_path = run_dir / "config.json"
        episode_path = run_dir / "episode_metrics.csv"
        eval_path = run_dir / "eval_metrics.csv"
        if not config_path.exists() or not episode_path.exists() or not eval_path.exists():
            continue
        summary = load_json(summary_path)
        if summary.get("mode") != "continual":
            continue
        config = load_json(config_path)
        episode_rows = read_csv(episode_path)
        eval_rows = read_csv(eval_path)
        switch_episodes = [
            episode_idx
            for episode_idx, row in enumerate(episode_rows)
            if int(str(row.get("actual_boundary", 0))) == 1
        ]
        benchmark = str(summary.get("benchmark") or config.get("benchmark") or "")
        scratch_refs = None
        scratch_ref_path = scratch_refs_map.get(benchmark)
        if scratch_ref_path:
            scratch_refs = json.loads(Path(scratch_ref_path).read_text())
        train_switch_rows = compute_train_switch_metrics(
            episode_rows=episode_rows,
            switch_episodes=switch_episodes,
            recovery_window=int(config.get("recovery_window", 25)),
            threshold=float(config.get("success_threshold", 0.8)),
            ema_alpha=0.1,
            scratch_refs=scratch_refs,
        )
        eval_switch_rows = compute_eval_switch_metrics(
            eval_rows=eval_rows,
            switch_episodes=switch_episodes,
            threshold=float(config.get("success_threshold", 0.8)),
            min_consecutive=int(config.get("threshold_min_consecutive_evals", 1)),
            scratch_refs=scratch_refs,
            episode_rows=episode_rows,
        )
        write_csv(run_dir / "switch_metrics_train.csv", train_switch_rows)
        write_csv(run_dir / "switch_metrics.csv", eval_switch_rows)
        if int(args.refresh_summary):
            refreshed = refresh_summary(summary=summary, eval_switch_rows=eval_switch_rows, train_switch_rows=train_switch_rows)
            summary_path.write_text(json.dumps(refreshed, indent=2))
        updated_runs += 1

    print(json.dumps({"updated_runs": updated_runs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
