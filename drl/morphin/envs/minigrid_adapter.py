"""
MiniGrid adapter for MORPHIN-DDQN.

Wraps MiniGrid environments so they expose the same {agent, target} observation
interface and 4-action cardinal API used by the GridWorld.  This lets the
existing obs_to_state / agent_target pipeline work without changes.

Key design decisions
--------------------
- **Position extraction via unwrapped agent state** – avoids the 2835-dim
  FlatObsWrapper output and keeps the state space identical (4D) to the 5×5
  and 9×9 GridWorld benchmarks.
- **Cardinal action mapping** – translates our 4 actions (right/up/left/down)
  into MiniGrid's rotate-then-forward sequences.  Each step consumes 1–3
  internal MiniGrid steps; the wrapper presents one logical step to the caller.
- **Fixed goal positions** – MiniGrid randomises goals by default.  We patch
  the goal position after reset so tasks are deterministic.
- **Reward shaping** – MiniGrid's native reward (0.0 per step, ~1.0 on success)
  is replaced by reward_scale on goal-reach and -step_penalty per step, matching
  the GridWorld reward structure.
- **Fully-observable by default** – the agent knows its own (x, y) position,
  which is equivalent to the agent_target obs mode used by our GridWorld
  continual benchmarks.

Limitations (explicitly acknowledged for the thesis)
----------------------------------------------------
- Partial-observation mode (raw 7×7 image) is NOT supported; extending to
  partial obs would require a CNN backbone and goes beyond this thesis scope.
- Not all MiniGrid environments are compatible: only Empty-* and FourRooms-v0
  work cleanly with cardinal-only navigation.  DoorKey etc. require pick/drop
  actions that are not mapped here.
- The adapter is a proof-of-concept for future benchmarking; overnight campaigns
  still use GridWorld.

Usage
-----
    from morphin.envs.minigrid_adapter import (
        MiniGridTaskSpec,
        MINIGRID_TASK_LIBRARY,
        MINIGRID_BENCHMARK_LIBRARY,
        make_minigrid_env,
    )
    task = MINIGRID_TASK_LIBRARY["mg_empty8_goal_a"]
    env  = make_minigrid_env(task, seed=42)
    obs, _ = env.reset()
    # obs == {"agent": np.array([ax, ay]), "target": np.array([gx, gy])}
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

import gymnasium as gym
import numpy as np


# ── MiniGrid action constants ────────────────────────────────────────────────
_MG_LEFT    = 0
_MG_RIGHT   = 1
_MG_FORWARD = 2

# Cardinal action → MiniGrid direction (0=right,1=down,2=left,3=up)
_CARDINAL_TO_MG_DIR = {
    0: 0,  # MORPHIN right (+x) → MiniGrid right
    1: 3,  # MORPHIN up   (+y) → MiniGrid up   (3)
    2: 2,  # MORPHIN left (-x) → MiniGrid left  (2)
    3: 1,  # MORPHIN down (-y) → MiniGrid down  (1)
}


# ── Task spec ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MiniGridTaskSpec:
    """Mirrors GridWorldTaskSpec so the two can be used interchangeably."""

    task_id: str
    env_id: str                          # e.g. "MiniGrid-Empty-8x8-v0"
    size: int = 8                        # Grid width — used for state normalisation
    agent_start: tuple[int, int] = (1, 1)
    goal: tuple[int, int] = (6, 6)
    max_episode_steps: int = 256
    reward_scale: float = 100.0
    step_penalty: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["goal"] = list(self.goal)
        d["agent_start"] = list(self.agent_start)
        return d


# ── Cardinal wrapper ─────────────────────────────────────────────────────────
class MiniGridCardinalWrapper(gym.Wrapper):
    """
    Wraps a MiniGrid env to expose:
      - Observation: Dict{"agent": (x,y) int64, "target": (gx,gy) int64}
      - Action space: Discrete(4) — right/up/left/down
      - Fixed goal position (overrides random placement)
      - Reward: +reward_scale on goal, -step_penalty per step
    """

    def __init__(
        self,
        env: gym.Env,
        task: MiniGridTaskSpec,
    ) -> None:
        super().__init__(env)
        self._task = task
        g = task.size - 1
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict({
            "agent":  gym.spaces.Box(0, g, shape=(2,), dtype=np.int64),
            "target": gym.spaces.Box(0, g, shape=(2,), dtype=np.int64),
        })
        self._fixed_goal = np.array(task.goal, dtype=np.int64)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _agent_pos(self) -> np.ndarray:
        return np.array(self.env.unwrapped.agent_pos, dtype=np.int64)

    def _rotate_to(self, target_dir: int) -> None:
        """Issue left/right turns to face target_dir without moving."""
        current = self.env.unwrapped.agent_dir
        diff = (target_dir - current) % 4
        if diff == 0:
            return
        if diff == 1:
            self.env.step(_MG_RIGHT)
        elif diff == 2:
            # 180°: two right turns
            self.env.step(_MG_RIGHT)
            self.env.step(_MG_RIGHT)
        else:  # diff == 3
            self.env.step(_MG_LEFT)

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        obs, info = self.env.reset(seed=seed)
        # Patch goal into the grid so the agent can find it
        uw = self.env.unwrapped
        gx, gy = int(self._fixed_goal[0]), int(self._fixed_goal[1])
        # MiniGrid stores grid as (width, height); y=0 is top in its coord system
        # Our coord: y=0 bottom → MiniGrid y = (size-2) - our_y   (border excluded)
        mg_row = (uw.height - 2) - gy  # convert to MiniGrid row (top=0)
        mg_col = gx                     # columns match
        from minigrid.core.world_object import Goal as MG_Goal
        uw.grid.set(mg_col, mg_row, MG_Goal())
        uw.goal_pos = (mg_col, mg_row)
        return self._make_obs(), info

    def step(
        self, action: int
    ) -> tuple[dict, float, bool, bool, dict]:
        target_dir = _CARDINAL_TO_MG_DIR[int(action)]
        self._rotate_to(target_dir)
        _, mg_reward, terminated, truncated, info = self.env.step(_MG_FORWARD)

        agent_pos = self._agent_pos()
        reached_goal = np.array_equal(agent_pos, self._fixed_goal)

        if reached_goal:
            reward = float(self._task.reward_scale)
            terminated = True
        else:
            reward = -float(self._task.step_penalty)

        return self._make_obs(), reward, terminated, truncated, info

    def _make_obs(self) -> dict:
        return {
            "agent":  self._agent_pos().copy(),
            "target": self._fixed_goal.copy(),
        }


# ── Factory ───────────────────────────────────────────────────────────────────
def make_minigrid_env(
    task: MiniGridTaskSpec,
    seed: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
) -> gym.Env:
    """Create and return a wrapped MiniGrid env.  Matches GridWorld factory API."""
    import minigrid  # noqa: F401 — ensure MiniGrid envs are registered

    n_steps = task.max_episode_steps if max_episode_steps is None else int(max_episode_steps)
    env = gym.make(task.env_id, max_episode_steps=n_steps)
    env = MiniGridCardinalWrapper(env, task)
    if seed is not None:
        env.reset(seed=seed)
    return env


# ── Task library ─────────────────────────────────────────────────────────────
# MiniGrid-Empty-8x8: interior cells are columns 1-6, rows 1-6 (walls on border).
# We use three corner goals forming a triangle, just like gw9 A/B/C.
MINIGRID_TASK_LIBRARY: dict[str, MiniGridTaskSpec] = {
    "mg_empty8_goal_a": MiniGridTaskSpec(
        task_id="mg_empty8_goal_a",
        env_id="MiniGrid-Empty-8x8-v0",
        size=8,
        agent_start=(1, 1),
        goal=(1, 6),   # bottom-left interior
        max_episode_steps=256,
    ),
    "mg_empty8_goal_b": MiniGridTaskSpec(
        task_id="mg_empty8_goal_b",
        env_id="MiniGrid-Empty-8x8-v0",
        size=8,
        agent_start=(1, 1),
        goal=(6, 6),   # bottom-right interior
        max_episode_steps=256,
    ),
    "mg_empty8_goal_c": MiniGridTaskSpec(
        task_id="mg_empty8_goal_c",
        env_id="MiniGrid-Empty-8x8-v0",
        size=8,
        agent_start=(1, 1),
        goal=(6, 1),   # top-right interior
        max_episode_steps=256,
    ),
}

MINIGRID_BENCHMARK_LIBRARY: dict[str, list[str]] = {
    "mg_empty8_ab_v1":   ["mg_empty8_goal_a", "mg_empty8_goal_b"],
    "mg_empty8_aba_v1":  ["mg_empty8_goal_a", "mg_empty8_goal_b", "mg_empty8_goal_a"],
    "mg_empty8_abc_v1":  ["mg_empty8_goal_a", "mg_empty8_goal_b", "mg_empty8_goal_c"],
    "mg_empty8_abca_v1": ["mg_empty8_goal_a", "mg_empty8_goal_b", "mg_empty8_goal_c", "mg_empty8_goal_a"],
}

MINIGRID_BENCHMARK_DEFAULT_OBS_MODE: dict[str, str] = {
    "mg_empty8_ab_v1":   "agent_target",
    "mg_empty8_aba_v1":  "agent_target",
    "mg_empty8_abc_v1":  "agent_target",
    "mg_empty8_abca_v1": "agent_target",
}
