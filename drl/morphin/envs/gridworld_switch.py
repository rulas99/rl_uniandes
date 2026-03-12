from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BASE_OBSTACLES = (
    (3, 4),
    (0, 3),
    (1, 3),
    (4, 2),
    (1, 1),
    (3, 1),
    (1, 0),
)

SYMMETRIC_GOAL_OBSTACLES = (
    (2, 3),
    (2, 2),
    (2, 1),
)


@dataclass(frozen=True)
class GridWorldTaskSpec:
    task_id: str
    size: int = 5
    agent_start: tuple[int, int] = (2, 4)
    goal: tuple[int, int] = (0, 0)
    obstacles: tuple[tuple[int, int], ...] = BASE_OBSTACLES
    holes: tuple[tuple[int, int], ...] = ()
    reward_scale: float = 100.0
    step_penalty: float = 1.0
    invalid_move_penalty: float = 2.0
    hole_penalty: float = 10.0
    block_on_obstacle: bool = True
    max_episode_steps: int = 150

    def env_kwargs(self) -> dict[str, object]:
        return {
            "size": self.size,
            "reward_scale": self.reward_scale,
            "step_penalty": self.step_penalty,
            "obstacles": list(self.obstacles),
            "holes": list(self.holes),
            "invalid_move_penalty": self.invalid_move_penalty,
            "hole_penalty": self.hole_penalty,
            "block_on_obstacle": self.block_on_obstacle,
        }

    def reset_options(self) -> dict[str, object]:
        return {
            "agent_start": tuple(self.agent_start),
            "goal": tuple(self.goal),
            "obstacles": list(self.obstacles),
            "holes": list(self.holes),
        }

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["obstacles"] = [list(coords) for coords in self.obstacles]
        data["holes"] = [list(coords) for coords in self.holes]
        data["goal"] = list(self.goal)
        data["agent_start"] = list(self.agent_start)
        return data


TASK_LIBRARY: dict[str, GridWorldTaskSpec] = {
    "gw_goal_a": GridWorldTaskSpec(task_id="gw_goal_a", goal=(0, 0)),
    "gw_goal_b": GridWorldTaskSpec(task_id="gw_goal_b", goal=(4, 0)),
    "gw_goal_c": GridWorldTaskSpec(task_id="gw_goal_c", goal=(4, 4)),
    "gw_goal_bal_a": GridWorldTaskSpec(
        task_id="gw_goal_bal_a",
        goal=(0, 0),
        obstacles=SYMMETRIC_GOAL_OBSTACLES,
    ),
    "gw_goal_bal_b": GridWorldTaskSpec(
        task_id="gw_goal_bal_b",
        goal=(4, 0),
        obstacles=SYMMETRIC_GOAL_OBSTACLES,
    ),
    "gw_goal_bal_c": GridWorldTaskSpec(
        task_id="gw_goal_bal_c",
        goal=(0, 2),
        obstacles=SYMMETRIC_GOAL_OBSTACLES,
    ),
    "gw_dyn_a": GridWorldTaskSpec(task_id="gw_dyn_a", goal=(0, 0)),
    "gw_dyn_b": GridWorldTaskSpec(
        task_id="gw_dyn_b",
        goal=(0, 0),
        obstacles=((3, 4), (4, 4), (0, 3), (3, 2), (1, 1), (3, 1), (1, 0)),
        holes=((2, 2), (4, 1)),
        hole_penalty=20.0,
    ),
}


BENCHMARK_LIBRARY: dict[str, list[str]] = {
    "gw_goal_switch_aba_v1": ["gw_goal_a", "gw_goal_b", "gw_goal_a"],
    "gw_hidden_goal_aba_v1": ["gw_goal_a", "gw_goal_b", "gw_goal_a"],
    "gw_hidden_goal_balanced_ab_v1": ["gw_goal_bal_a", "gw_goal_bal_b"],
    "gw_hidden_goal_balanced_ac_v1": ["gw_goal_bal_a", "gw_goal_bal_c"],
    "gw_hidden_goal_balanced_aba_v1": ["gw_goal_bal_a", "gw_goal_bal_b", "gw_goal_bal_a"],
    "gw_goal_conditioned_aba_v1": ["gw_goal_a", "gw_goal_b", "gw_goal_a"],
    "gw_goal_conditioned_balanced_ab_v1": ["gw_goal_bal_a", "gw_goal_bal_b"],
    "gw_goal_conditioned_balanced_ac_v1": ["gw_goal_bal_a", "gw_goal_bal_c"],
    "gw_goal_conditioned_balanced_ca_v1": ["gw_goal_bal_c", "gw_goal_bal_a"],
    "gw_goal_conditioned_balanced_acb_v1": ["gw_goal_bal_a", "gw_goal_bal_c", "gw_goal_bal_b"],
    "gw_goal_conditioned_balanced_aba_v1": ["gw_goal_bal_a", "gw_goal_bal_b", "gw_goal_bal_a"],
    "gw_goal_switch_abca_v1": ["gw_goal_a", "gw_goal_b", "gw_goal_c", "gw_goal_a"],
    "gw_dynamics_switch_aba_v1": ["gw_dyn_a", "gw_dyn_b", "gw_dyn_a"],
}


BENCHMARK_DEFAULT_OBS_MODE: dict[str, str] = {
    "gw_goal_switch_aba_v1": "agent_target",
    "gw_hidden_goal_aba_v1": "agent_only",
    "gw_hidden_goal_balanced_ab_v1": "agent_only",
    "gw_hidden_goal_balanced_ac_v1": "agent_only",
    "gw_hidden_goal_balanced_aba_v1": "agent_only",
    "gw_goal_conditioned_aba_v1": "agent_target",
    "gw_goal_conditioned_balanced_ab_v1": "agent_target",
    "gw_goal_conditioned_balanced_ac_v1": "agent_target",
    "gw_goal_conditioned_balanced_ca_v1": "agent_target",
    "gw_goal_conditioned_balanced_acb_v1": "agent_target",
    "gw_goal_conditioned_balanced_aba_v1": "agent_target",
    "gw_goal_switch_abca_v1": "agent_target",
    "gw_dynamics_switch_aba_v1": "grid_channels",
}


def build_task_sequence(
    benchmark: str,
    task_ids_csv: str | None = None,
) -> list[GridWorldTaskSpec]:
    if task_ids_csv:
        task_ids = [task_id.strip() for task_id in task_ids_csv.split(",") if task_id.strip()]
    else:
        if benchmark not in BENCHMARK_LIBRARY:
            raise KeyError(f"Unknown benchmark: {benchmark}")
        task_ids = BENCHMARK_LIBRARY[benchmark]
    sequence = []
    for task_id in task_ids:
        if task_id not in TASK_LIBRARY:
            raise KeyError(f"Unknown task id: {task_id}")
        sequence.append(TASK_LIBRARY[task_id])
    return sequence


def unique_task_ids_for_benchmark(benchmark: str) -> list[str]:
    ordered: list[str] = []
    for task_id in BENCHMARK_LIBRARY[benchmark]:
        if task_id not in ordered:
            ordered.append(task_id)
    return ordered


def make_gridworld_env(
    task: GridWorldTaskSpec,
    seed: int | None = None,
    max_episode_steps: int | None = None,
) -> Any:
    import gymnasium as gym

    env = gym.make(
        "entropia/GridWorld-v0",
        max_episode_steps=(task.max_episode_steps if max_episode_steps is None else int(max_episode_steps)),
        **task.env_kwargs(),
    )
    if seed is not None:
        env.reset(seed=seed, options=task.reset_options())
    return env
