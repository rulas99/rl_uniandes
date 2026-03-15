from __future__ import annotations

import numpy as np

from .gridworld_switch import TASK_LIBRARY
from .gridworld_switch import GridWorldTaskSpec


OBS_MODE_CHOICES = (
    "agent_only",
    "agent_target",
    "agent_context",
    "agent_context_calibrated",
    "agent_context_calibrated_v2",
    "agent_context_calibrated_v3",
    "grid_channels",
)
TASK_CONTEXT_TASK_IDS = tuple(sorted(TASK_LIBRARY))
TASK_CONTEXT_INDEX = {task_id: index for index, task_id in enumerate(TASK_CONTEXT_TASK_IDS)}
CALIBRATED_CONTEXT_TASK_IDS = ("gw9_goal_cal_a", "gw9_goal_cal_b", "gw9_goal_cal_c")
CALIBRATED_CONTEXT_INDEX = {task_id: index for index, task_id in enumerate(CALIBRATED_CONTEXT_TASK_IDS)}
CALIBRATED_V2_CONTEXT_TASK_IDS = ("gw9_goal_cal2_a", "gw9_goal_cal2_b", "gw9_goal_cal2_c")
CALIBRATED_V2_CONTEXT_INDEX = {
    task_id: index for index, task_id in enumerate(CALIBRATED_V2_CONTEXT_TASK_IDS)
}
CALIBRATED_V3_CONTEXT_TASK_IDS = ("gw9_goal_cal3_a", "gw9_goal_cal3_b", "gw9_goal_cal3_c")
CALIBRATED_V3_CONTEXT_INDEX = {
    task_id: index for index, task_id in enumerate(CALIBRATED_V3_CONTEXT_TASK_IDS)
}


def obs_to_state(
    obs: dict[str, object],
    task: GridWorldTaskSpec,
    obs_mode: str,
) -> np.ndarray:
    agent = np.asarray(obs["agent"], dtype=np.float32) / float(task.size - 1)
    target = np.asarray(obs["target"], dtype=np.float32) / float(task.size - 1)

    if obs_mode == "agent_only":
        return agent.astype(np.float32)

    if obs_mode == "agent_target":
        return np.concatenate([agent, target], dtype=np.float32)

    if obs_mode == "agent_context":
        context = np.zeros(len(TASK_CONTEXT_TASK_IDS), dtype=np.float32)
        context[TASK_CONTEXT_INDEX[task.task_id]] = 1.0
        return np.concatenate([agent, context], dtype=np.float32)

    if obs_mode == "agent_context_calibrated":
        if task.task_id not in CALIBRATED_CONTEXT_INDEX:
            raise ValueError(f"Task {task.task_id} is not supported by agent_context_calibrated")
        context = np.zeros(len(CALIBRATED_CONTEXT_TASK_IDS), dtype=np.float32)
        context[CALIBRATED_CONTEXT_INDEX[task.task_id]] = 1.0
        return np.concatenate([agent, context], dtype=np.float32)

    if obs_mode == "agent_context_calibrated_v2":
        if task.task_id not in CALIBRATED_V2_CONTEXT_INDEX:
            raise ValueError(f"Task {task.task_id} is not supported by agent_context_calibrated_v2")
        context = np.zeros(len(CALIBRATED_V2_CONTEXT_TASK_IDS), dtype=np.float32)
        context[CALIBRATED_V2_CONTEXT_INDEX[task.task_id]] = 1.0
        return np.concatenate([agent, context], dtype=np.float32)

    if obs_mode == "agent_context_calibrated_v3":
        if task.task_id not in CALIBRATED_V3_CONTEXT_INDEX:
            raise ValueError(f"Task {task.task_id} is not supported by agent_context_calibrated_v3")
        context = np.zeros(len(CALIBRATED_V3_CONTEXT_TASK_IDS), dtype=np.float32)
        context[CALIBRATED_V3_CONTEXT_INDEX[task.task_id]] = 1.0
        return np.concatenate([agent, context], dtype=np.float32)

    if obs_mode == "grid_channels":
        grid = np.zeros((4, task.size, task.size), dtype=np.float32)
        ax, ay = (int(obs["agent"][0]), int(obs["agent"][1]))
        tx, ty = (int(obs["target"][0]), int(obs["target"][1]))
        grid[0, ax, ay] = 1.0
        grid[1, tx, ty] = 1.0
        for x, y in task.obstacles:
            grid[2, int(x), int(y)] = 1.0
        for x, y in task.holes:
            grid[3, int(x), int(y)] = 1.0
        return grid.reshape(-1)

    raise ValueError(f"Unsupported obs_mode: {obs_mode}")
