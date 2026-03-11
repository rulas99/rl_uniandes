from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray | None
    done: bool
    task_id: str


def _stack_states(
    transitions: Sequence[Transition],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    states = torch.as_tensor(
        np.stack([transition.state for transition in transitions]),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.as_tensor(
        np.array([transition.action for transition in transitions], dtype=np.int64),
        dtype=torch.long,
        device=device,
    ).unsqueeze(1)
    rewards = torch.as_tensor(
        np.array([transition.reward for transition in transitions], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    dones = torch.as_tensor(
        np.array([transition.done for transition in transitions], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    non_final_mask = torch.as_tensor(
        np.array([transition.next_state is not None for transition in transitions], dtype=bool),
        dtype=torch.bool,
        device=device,
    )
    if bool(non_final_mask.any()):
        next_states_np = np.stack(
            [transition.next_state for transition in transitions if transition.next_state is not None]
        )
        next_states = torch.as_tensor(next_states_np, dtype=torch.float32, device=device)
    else:
        next_states = torch.empty((0, states.shape[1]), dtype=torch.float32, device=device)
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "non_final_mask": non_final_mask,
        "next_states": next_states,
    }


class UniformReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.memory: Deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray | None,
        done: bool,
        task_id: str,
    ) -> None:
        self.memory.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=None if next_state is None else np.asarray(next_state, dtype=np.float32),
                done=bool(done),
                task_id=str(task_id),
            )
        )

    def sample(self, batch_size: int, device: torch.device, **_: object) -> dict[str, torch.Tensor]:
        transitions = random.sample(self.memory, k=int(batch_size))
        return _stack_states(transitions, device=device)

    def on_task_switch(self, policy: str, keep_recent_frac: float = 0.2) -> None:
        if policy == "keep_all":
            return
        if policy == "clear_on_switch":
            self.memory.clear()
            return
        if policy == "keep_recent_frac":
            keep_count = max(0, int(round(len(self.memory) * float(keep_recent_frac))))
            recent = list(self.memory)[-keep_count:] if keep_count > 0 else []
            self.memory = deque(recent, maxlen=self.capacity)
            return
        raise ValueError(f"Unsupported replay switch policy: {policy}")


class SegmentedReplayBuffer:
    def __init__(self, recent_capacity: int, archive_capacity: int) -> None:
        self.recent_capacity = int(recent_capacity)
        self.archive_capacity = int(archive_capacity)
        self.recent: Deque[Transition] = deque(maxlen=self.recent_capacity)
        self.archive: Deque[Transition] = deque(maxlen=self.archive_capacity)

    def __len__(self) -> int:
        return len(self.recent) + len(self.archive)

    def num_recent(self) -> int:
        return len(self.recent)

    def num_archive(self) -> int:
        return len(self.archive)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray | None,
        done: bool,
        task_id: str,
    ) -> None:
        self.recent.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=None if next_state is None else np.asarray(next_state, dtype=np.float32),
                done=bool(done),
                task_id=str(task_id),
            )
        )

    def on_task_switch(self, archive_frac: float = 0.25, keep_tail: int = 512) -> None:
        recent_list = list(self.recent)
        if not recent_list:
            return

        keep_tail = max(0, int(keep_tail))
        if keep_tail >= len(recent_list):
            return

        tail = recent_list[-keep_tail:] if keep_tail > 0 else []
        archive_pool = recent_list[:-keep_tail] if keep_tail > 0 else recent_list
        if archive_pool:
            archive_count = max(1, int(round(len(archive_pool) * float(archive_frac))))
            for transition in random.sample(archive_pool, k=min(archive_count, len(archive_pool))):
                self.archive.append(transition)
        self.recent = deque(tail, maxlen=self.recent_capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        p_recent: float = 0.7,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        batch_size = int(batch_size)
        recent_list = list(self.recent)
        archive_list = list(self.archive)
        if not recent_list and not archive_list:
            raise ValueError("Cannot sample from an empty segmented replay buffer")

        recent_target = min(len(recent_list), int(round(batch_size * float(p_recent))))
        archive_target = min(len(archive_list), batch_size - recent_target)

        transitions: list[Transition] = []
        if recent_target > 0:
            transitions.extend(random.sample(recent_list, k=recent_target))
        if archive_target > 0:
            transitions.extend(random.sample(archive_list, k=archive_target))

        pool: list[Transition] = recent_list + archive_list
        while len(transitions) < batch_size:
            transitions.append(random.choice(pool))

        return _stack_states(transitions[:batch_size], device=device)
