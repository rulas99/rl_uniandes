from __future__ import annotations

import argparse
import csv
import json
import importlib.util
import sys
from collections import deque
from pathlib import Path

if __package__ in {None, ""}:
    module_path = Path(__file__).resolve().parent / "envs" / "gridworld_switch.py"
    spec = importlib.util.spec_from_file_location("gridworld_switch_local", module_path)
    gridworld_switch = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = gridworld_switch
    spec.loader.exec_module(gridworld_switch)
    BENCHMARK_DEFAULT_OBS_MODE = gridworld_switch.BENCHMARK_DEFAULT_OBS_MODE
    BENCHMARK_LIBRARY = gridworld_switch.BENCHMARK_LIBRARY
    GridWorldTaskSpec = gridworld_switch.GridWorldTaskSpec
    TASK_LIBRARY = gridworld_switch.TASK_LIBRARY
else:
    from .envs.gridworld_switch import BENCHMARK_DEFAULT_OBS_MODE
    from .envs.gridworld_switch import BENCHMARK_LIBRARY
    from .envs.gridworld_switch import GridWorldTaskSpec
    from .envs.gridworld_switch import TASK_LIBRARY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe GridWorld benchmark structure and path balance")
    parser.add_argument("--benchmarks-csv", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="")
    return parser.parse_args()


def shortest_path_length(task: GridWorldTaskSpec) -> int | None:
    frontier: deque[tuple[tuple[int, int], int]] = deque([(task.agent_start, 0)])
    visited = {task.agent_start}
    blocked = set(task.obstacles) | set(task.holes)
    while frontier:
        (x_coord, y_coord), distance = frontier.popleft()
        if (x_coord, y_coord) == task.goal:
            return distance
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_pos = (x_coord + dx, y_coord + dy)
            if (
                0 <= next_pos[0] < task.size
                and 0 <= next_pos[1] < task.size
                and next_pos not in blocked
                and next_pos not in visited
            ):
                visited.add(next_pos)
                frontier.append((next_pos, distance + 1))
    return None


def benchmark_rows(benchmarks: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for benchmark in benchmarks:
        task_ids = BENCHMARK_LIBRARY[benchmark]
        unique_task_ids = list(dict.fromkeys(task_ids))
        path_lengths = {
            task_id: shortest_path_length(TASK_LIBRARY[task_id])
            for task_id in unique_task_ids
        }
        valid_lengths = [length for length in path_lengths.values() if length is not None]
        rows.append(
            {
                "benchmark": benchmark,
                "obs_mode": BENCHMARK_DEFAULT_OBS_MODE[benchmark],
                "task_sequence": ",".join(task_ids),
                "unique_tasks": ",".join(unique_task_ids),
                "num_tasks": len(task_ids),
                "num_unique_tasks": len(unique_task_ids),
                "num_switches": max(0, len(task_ids) - 1),
                "has_revisit": int(len(unique_task_ids) < len(task_ids)),
                "action_space_size": 4,
                "path_lengths": json.dumps(path_lengths, sort_keys=True),
                "min_shortest_path": min(valid_lengths) if valid_lengths else None,
                "max_shortest_path": max(valid_lengths) if valid_lengths else None,
                "path_span": (max(valid_lengths) - min(valid_lengths)) if valid_lengths else None,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.benchmarks_csv:
        benchmarks = [item.strip() for item in args.benchmarks_csv.split(",") if item.strip()]
    else:
        benchmarks = sorted(BENCHMARK_LIBRARY)
    rows = benchmark_rows(benchmarks)
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output_path, rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
