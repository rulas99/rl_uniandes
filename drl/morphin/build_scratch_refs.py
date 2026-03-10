from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scratch reference file from scratch_task runs")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


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
        grouped.setdefault(task_id, {"time_to_threshold": [], "final_seen_success_mean": []})
        if summary.get("overall_time_to_threshold") is not None:
            grouped[task_id]["time_to_threshold"].append(float(summary["overall_time_to_threshold"]))
        if summary.get("final_seen_success_mean") is not None:
            grouped[task_id]["final_seen_success_mean"].append(float(summary["final_seen_success_mean"]))

    refs: dict[str, dict[str, float | None]] = {}
    for task_id, values in grouped.items():
        refs[task_id] = {
            "time_to_threshold": (
                statistics.median(values["time_to_threshold"]) if values["time_to_threshold"] else None
            ),
            "final_seen_success_mean": (
                statistics.mean(values["final_seen_success_mean"]) if values["final_seen_success_mean"] else None
            ),
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(refs, indent=2))
    print(json.dumps({"output_json": str(output_json), "num_tasks": len(refs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
