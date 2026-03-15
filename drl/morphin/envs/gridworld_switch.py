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

NINE_BY_NINE_BALANCED_OBSTACLES = (
    (0, 4),
    (2, 4),
    (3, 4),
    (4, 4),
    (5, 4),
    (6, 4),
    (8, 4),
)

NINE_BY_NINE_CALIBRATED_FREE_CELLS = (
    (4, 8),
    (4, 7),
    (4, 6),
    (4, 5),
    (4, 4),
    (4, 3),
    (4, 2),
    (4, 1),
    (4, 0),
    (3, 4),
    (2, 4),
    (1, 4),
    (1, 3),
    (1, 5),
    (5, 4),
    (6, 4),
    (7, 4),
    (7, 3),
    (7, 5),
    (3, 2),
    (5, 2),
)

NINE_BY_NINE_CALIBRATED_V2_FREE_CELLS = (
    (4, 8),
    (4, 7),
    (4, 6),
    (4, 5),
    (3, 5),
    (2, 5),
    (2, 4),
    (2, 3),
    (1, 3),
    (5, 5),
    (6, 5),
    (6, 4),
    (6, 3),
    (7, 3),
    (3, 4),
    (3, 3),
    (4, 3),
    (4, 2),
)

NINE_BY_NINE_CALIBRATED_V3_FREE_CELLS = (
    (4, 8),
    (4, 7),
    (4, 6),
    (4, 5),
    (3, 5),
    (3, 4),
    (3, 3),
    (2, 3),
    (1, 3),
    (5, 5),
    (5, 4),
    (5, 3),
    (6, 3),
    (7, 3),
    (4, 3),
    (4, 2),
)


def _obstacles_from_free(
    size: int,
    free_cells: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    free_set = {tuple(cell) for cell in free_cells}
    return tuple(
        (x, y)
        for y in range(size)
        for x in range(size)
        if (x, y) not in free_set
    )


NINE_BY_NINE_CALIBRATED_OBSTACLES = _obstacles_from_free(
    size=9,
    free_cells=NINE_BY_NINE_CALIBRATED_FREE_CELLS,
)

NINE_BY_NINE_CALIBRATED_V2_OBSTACLES = _obstacles_from_free(
    size=9,
    free_cells=NINE_BY_NINE_CALIBRATED_V2_FREE_CELLS,
)

NINE_BY_NINE_CALIBRATED_V3_OBSTACLES = _obstacles_from_free(
    size=9,
    free_cells=NINE_BY_NINE_CALIBRATED_V3_FREE_CELLS,
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
    "gw9_goal_bal_a": GridWorldTaskSpec(
        task_id="gw9_goal_bal_a",
        size=9,
        agent_start=(4, 8),
        goal=(1, 0),
        obstacles=NINE_BY_NINE_BALANCED_OBSTACLES,
        max_episode_steps=250,
    ),
    "gw9_goal_bal_b": GridWorldTaskSpec(
        task_id="gw9_goal_bal_b",
        size=9,
        agent_start=(4, 8),
        goal=(7, 0),
        obstacles=NINE_BY_NINE_BALANCED_OBSTACLES,
        max_episode_steps=250,
    ),
    "gw9_goal_cal_a": GridWorldTaskSpec(
        task_id="gw9_goal_cal_a",
        size=9,
        agent_start=(4, 8),
        goal=(1, 3),
        obstacles=NINE_BY_NINE_CALIBRATED_OBSTACLES,
        max_episode_steps=250,
    ),
    "gw9_goal_cal_b": GridWorldTaskSpec(
        task_id="gw9_goal_cal_b",
        size=9,
        agent_start=(4, 8),
        goal=(7, 3),
        obstacles=NINE_BY_NINE_CALIBRATED_OBSTACLES,
        max_episode_steps=250,
    ),
    "gw9_goal_cal_c": GridWorldTaskSpec(
        task_id="gw9_goal_cal_c",
        size=9,
        agent_start=(4, 8),
        goal=(4, 0),
        obstacles=NINE_BY_NINE_CALIBRATED_OBSTACLES,
        max_episode_steps=250,
    ),
    "gw9_goal_cal2_a": GridWorldTaskSpec(
        task_id="gw9_goal_cal2_a",
        size=9,
        agent_start=(4, 8),
        goal=(1, 3),
        obstacles=NINE_BY_NINE_CALIBRATED_V2_OBSTACLES,
        max_episode_steps=250,
    ),
    "gw9_goal_cal2_b": GridWorldTaskSpec(
        task_id="gw9_goal_cal2_b",
        size=9,
        agent_start=(4, 8),
        goal=(7, 3),
        obstacles=NINE_BY_NINE_CALIBRATED_V2_OBSTACLES,
        max_episode_steps=250,
    ),
    "gw9_goal_cal2_c": GridWorldTaskSpec(
        task_id="gw9_goal_cal2_c",
        size=9,
        agent_start=(4, 8),
        goal=(4, 2),
        obstacles=NINE_BY_NINE_CALIBRATED_V2_OBSTACLES,
        max_episode_steps=250,
    ),
    "gw9_goal_cal3_a": GridWorldTaskSpec(
        task_id="gw9_goal_cal3_a",
        size=9,
        agent_start=(4, 8),
        goal=(1, 3),
        obstacles=NINE_BY_NINE_CALIBRATED_V3_OBSTACLES,
        max_episode_steps=250,
    ),
    "gw9_goal_cal3_b": GridWorldTaskSpec(
        task_id="gw9_goal_cal3_b",
        size=9,
        agent_start=(4, 8),
        goal=(7, 3),
        obstacles=NINE_BY_NINE_CALIBRATED_V3_OBSTACLES,
        max_episode_steps=250,
    ),
    "gw9_goal_cal3_c": GridWorldTaskSpec(
        task_id="gw9_goal_cal3_c",
        size=9,
        agent_start=(4, 8),
        goal=(4, 2),
        obstacles=NINE_BY_NINE_CALIBRATED_V3_OBSTACLES,
        max_episode_steps=250,
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
    "gw_hidden_goal_balanced_abab_v1": ["gw_goal_bal_a", "gw_goal_bal_b", "gw_goal_bal_a", "gw_goal_bal_b"],
    "gw_hidden_goal_balanced_ababa_v1": ["gw_goal_bal_a", "gw_goal_bal_b", "gw_goal_bal_a", "gw_goal_bal_b", "gw_goal_bal_a"],
    "gw9_context_balanced_ab_v1": ["gw9_goal_bal_a", "gw9_goal_bal_b"],
    "gw9_context_balanced_ba_v1": ["gw9_goal_bal_b", "gw9_goal_bal_a"],
    "gw9_context_balanced_aba_v1": ["gw9_goal_bal_a", "gw9_goal_bal_b", "gw9_goal_bal_a"],
    "gw9_context_balanced_bab_v1": ["gw9_goal_bal_b", "gw9_goal_bal_a", "gw9_goal_bal_b"],
    "gw9_context_balanced_abab_v1": ["gw9_goal_bal_a", "gw9_goal_bal_b", "gw9_goal_bal_a", "gw9_goal_bal_b"],
    "gw9_context_balanced_ababa_v1": ["gw9_goal_bal_a", "gw9_goal_bal_b", "gw9_goal_bal_a", "gw9_goal_bal_b", "gw9_goal_bal_a"],
    "gw9_context_balanced_babab_v1": ["gw9_goal_bal_b", "gw9_goal_bal_a", "gw9_goal_bal_b", "gw9_goal_bal_a", "gw9_goal_bal_b"],
    "gw9_context_balanced_baba_v1": ["gw9_goal_bal_b", "gw9_goal_bal_a", "gw9_goal_bal_b", "gw9_goal_bal_a"],
    "gw9_context_calibrated_ab_v1": ["gw9_goal_cal_a", "gw9_goal_cal_b"],
    "gw9_context_calibrated_ba_v1": ["gw9_goal_cal_b", "gw9_goal_cal_a"],
    "gw9_context_calibrated_aba_v1": ["gw9_goal_cal_a", "gw9_goal_cal_b", "gw9_goal_cal_a"],
    "gw9_context_calibrated_bab_v1": ["gw9_goal_cal_b", "gw9_goal_cal_a", "gw9_goal_cal_b"],
    "gw9_context_calibrated_ac_v1": ["gw9_goal_cal_a", "gw9_goal_cal_c"],
    "gw9_context_calibrated_ca_v1": ["gw9_goal_cal_c", "gw9_goal_cal_a"],
    "gw9_context_calibrated_bc_v1": ["gw9_goal_cal_b", "gw9_goal_cal_c"],
    "gw9_context_calibrated_cb_v1": ["gw9_goal_cal_c", "gw9_goal_cal_b"],
    "gw9_context_calibrated_abc_v1": ["gw9_goal_cal_a", "gw9_goal_cal_b", "gw9_goal_cal_c"],
    "gw9_context_calibrated_bac_v1": ["gw9_goal_cal_b", "gw9_goal_cal_a", "gw9_goal_cal_c"],
    "gw9_context_calibrated2_ab_v1": ["gw9_goal_cal2_a", "gw9_goal_cal2_b"],
    "gw9_context_calibrated2_ba_v1": ["gw9_goal_cal2_b", "gw9_goal_cal2_a"],
    "gw9_context_calibrated2_aba_v1": ["gw9_goal_cal2_a", "gw9_goal_cal2_b", "gw9_goal_cal2_a"],
    "gw9_context_calibrated2_bab_v1": ["gw9_goal_cal2_b", "gw9_goal_cal2_a", "gw9_goal_cal2_b"],
    "gw9_context_calibrated2_ac_v1": ["gw9_goal_cal2_a", "gw9_goal_cal2_c"],
    "gw9_context_calibrated2_ca_v1": ["gw9_goal_cal2_c", "gw9_goal_cal2_a"],
    "gw9_context_calibrated2_bc_v1": ["gw9_goal_cal2_b", "gw9_goal_cal2_c"],
    "gw9_context_calibrated2_cb_v1": ["gw9_goal_cal2_c", "gw9_goal_cal2_b"],
    "gw9_context_calibrated2_abc_v1": ["gw9_goal_cal2_a", "gw9_goal_cal2_b", "gw9_goal_cal2_c"],
    "gw9_context_calibrated2_bac_v1": ["gw9_goal_cal2_b", "gw9_goal_cal2_a", "gw9_goal_cal2_c"],
    "gw9_context_calibrated3_ab_v1": ["gw9_goal_cal3_a", "gw9_goal_cal3_b"],
    "gw9_context_calibrated3_ba_v1": ["gw9_goal_cal3_b", "gw9_goal_cal3_a"],
    "gw9_context_calibrated3_aba_v1": ["gw9_goal_cal3_a", "gw9_goal_cal3_b", "gw9_goal_cal3_a"],
    "gw9_context_calibrated3_bab_v1": ["gw9_goal_cal3_b", "gw9_goal_cal3_a", "gw9_goal_cal3_b"],
    "gw9_context_calibrated3_ac_v1": ["gw9_goal_cal3_a", "gw9_goal_cal3_c"],
    "gw9_context_calibrated3_ca_v1": ["gw9_goal_cal3_c", "gw9_goal_cal3_a"],
    "gw9_context_calibrated3_bc_v1": ["gw9_goal_cal3_b", "gw9_goal_cal3_c"],
    "gw9_context_calibrated3_cb_v1": ["gw9_goal_cal3_c", "gw9_goal_cal3_b"],
    "gw9_context_calibrated3_abc_v1": ["gw9_goal_cal3_a", "gw9_goal_cal3_b", "gw9_goal_cal3_c"],
    "gw9_context_calibrated3_bac_v1": ["gw9_goal_cal3_b", "gw9_goal_cal3_a", "gw9_goal_cal3_c"],
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
    "gw_hidden_goal_balanced_abab_v1": "agent_only",
    "gw_hidden_goal_balanced_ababa_v1": "agent_only",
    "gw9_context_balanced_ab_v1": "agent_context",
    "gw9_context_balanced_ba_v1": "agent_context",
    "gw9_context_balanced_aba_v1": "agent_context",
    "gw9_context_balanced_bab_v1": "agent_context",
    "gw9_context_balanced_abab_v1": "agent_context",
    "gw9_context_balanced_ababa_v1": "agent_context",
    "gw9_context_balanced_babab_v1": "agent_context",
    "gw9_context_balanced_baba_v1": "agent_context",
    "gw9_context_calibrated_ab_v1": "agent_context_calibrated",
    "gw9_context_calibrated_ba_v1": "agent_context_calibrated",
    "gw9_context_calibrated_aba_v1": "agent_context_calibrated",
    "gw9_context_calibrated_bab_v1": "agent_context_calibrated",
    "gw9_context_calibrated_ac_v1": "agent_context_calibrated",
    "gw9_context_calibrated_ca_v1": "agent_context_calibrated",
    "gw9_context_calibrated_bc_v1": "agent_context_calibrated",
    "gw9_context_calibrated_cb_v1": "agent_context_calibrated",
    "gw9_context_calibrated_abc_v1": "agent_context_calibrated",
    "gw9_context_calibrated_bac_v1": "agent_context_calibrated",
    "gw9_context_calibrated2_ab_v1": "agent_context_calibrated_v2",
    "gw9_context_calibrated2_ba_v1": "agent_context_calibrated_v2",
    "gw9_context_calibrated2_aba_v1": "agent_context_calibrated_v2",
    "gw9_context_calibrated2_bab_v1": "agent_context_calibrated_v2",
    "gw9_context_calibrated2_ac_v1": "agent_context_calibrated_v2",
    "gw9_context_calibrated2_ca_v1": "agent_context_calibrated_v2",
    "gw9_context_calibrated2_bc_v1": "agent_context_calibrated_v2",
    "gw9_context_calibrated2_cb_v1": "agent_context_calibrated_v2",
    "gw9_context_calibrated2_abc_v1": "agent_context_calibrated_v2",
    "gw9_context_calibrated2_bac_v1": "agent_context_calibrated_v2",
    "gw9_context_calibrated3_ab_v1": "agent_context_calibrated_v3",
    "gw9_context_calibrated3_ba_v1": "agent_context_calibrated_v3",
    "gw9_context_calibrated3_aba_v1": "agent_context_calibrated_v3",
    "gw9_context_calibrated3_bab_v1": "agent_context_calibrated_v3",
    "gw9_context_calibrated3_ac_v1": "agent_context_calibrated_v3",
    "gw9_context_calibrated3_ca_v1": "agent_context_calibrated_v3",
    "gw9_context_calibrated3_bc_v1": "agent_context_calibrated_v3",
    "gw9_context_calibrated3_cb_v1": "agent_context_calibrated_v3",
    "gw9_context_calibrated3_abc_v1": "agent_context_calibrated_v3",
    "gw9_context_calibrated3_bac_v1": "agent_context_calibrated_v3",
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
