from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate scratch reference quality before continual runs")
    parser.add_argument("--refs-json", type=str, required=True)
    parser.add_argument("--task-ids-csv", type=str, default="")
    parser.add_argument("--min-final-success", type=float, default=0.8)
    parser.add_argument("--min-passing-fraction", type=float, default=1.0)
    parser.add_argument("--require-stable", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refs = json.loads(Path(args.refs_json).read_text())
    task_ids = [task_id.strip() for task_id in args.task_ids_csv.split(",") if task_id.strip()]
    if not task_ids:
        task_ids = sorted(refs)

    per_task: dict[str, dict[str, object]] = {}
    num_passing = 0
    for task_id in task_ids:
        ref = refs.get(task_id) or {}
        final_success = ref.get("final_seen_success_mean")
        is_stable = bool(ref.get("is_reference_stable", False))
        passes = final_success is not None and float(final_success) >= float(args.min_final_success)
        if args.require_stable:
            passes = passes and is_stable
        per_task[task_id] = {
            "passes": passes,
            "final_seen_success_mean": final_success,
            "is_reference_stable": is_stable,
            "num_runs_total": ref.get("num_runs_total"),
            "num_runs_valid": ref.get("num_runs_valid"),
            "valid_run_fraction": ref.get("valid_run_fraction"),
            "stability_flag": ref.get("stability_flag"),
        }
        if passes:
            num_passing += 1

    num_tasks = len(task_ids)
    passing_fraction = (num_passing / num_tasks) if num_tasks else 0.0
    ok = passing_fraction >= float(args.min_passing_fraction)
    print(
        json.dumps(
            {
                "ok": ok,
                "num_tasks": num_tasks,
                "num_passing": num_passing,
                "passing_fraction": passing_fraction,
                "min_passing_fraction": float(args.min_passing_fraction),
                "min_final_success": float(args.min_final_success),
                "require_stable": bool(args.require_stable),
                "tasks": per_task,
            },
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
