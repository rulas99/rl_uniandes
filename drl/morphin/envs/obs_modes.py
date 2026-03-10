from __future__ import annotations

import numpy as np

from .gridworld_switch import GridWorldTaskSpec


OBS_MODE_CHOICES = ("agent_only", "agent_target", "grid_channels")


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
