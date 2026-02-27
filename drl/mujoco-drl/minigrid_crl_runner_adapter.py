from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import time
from collections import deque
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import gymnasium as gym
import matplotlib
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Tags comunes para PPO en SB3.
AUDIT_TB_TAGS: List[str] = [
    "train/entropy_loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
]

PREFERRED_MINIGRID_SMOKE4: List[str] = [
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-FourRooms-v0",
    "MiniGrid-Unlock-v0"
]

PREFERRED_MINIGRID_SMOKE3: List[str] = [
    "MiniGrid-DoorKey-9x9-v0",
    "MiniGrid-SimpleCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-Unlock-v0"
]

PREFERRED_MINIGRID_CW10: List[str] = [
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-Unlock-v0",
    "MiniGrid-BlockedUnlockPickup-v0",
    "MiniGrid-FourRooms-v0",
    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-SimpleCrossingS9N1-v0",
    "MiniGrid-MultiRoom-N2-S4-v0",
    "MiniGrid-Dynamic-Obstacles-5x5-v0",
]


def _safe_package_version(package_name: str) -> str:
    try:
        return str(importlib_metadata.version(package_name))
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"
    except Exception as exc:
        return f"error:{exc.__class__.__name__}"


def _collect_runtime_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gymnasium": _safe_package_version("gymnasium"),
        "minigrid": _safe_package_version("minigrid"),
        "stable-baselines3": _safe_package_version("stable-baselines3"),
        "torch": _safe_package_version("torch"),
    }
    try:
        versions["cuda_available"] = bool(th.cuda.is_available())
        versions["cuda_version"] = str(getattr(th.version, "cuda", None))
    except Exception:
        versions["cuda_available"] = False
        versions["cuda_version"] = None
    return versions


def _write_runtime_versions(run_dir: Path, versions: Dict[str, Any]) -> Path:
    out_path = run_dir / "versions.json"
    out_path.write_text(json.dumps(versions, indent=2), encoding="utf-8")
    return out_path


def _ensure_minigrid() -> None:
    try:
        import minigrid  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "No se encontró `minigrid`.\n"
            "Instala con:\n"
            "  ./.venv/bin/python -m pip install minigrid"
        ) from exc


def _registered_env_ids(prefix: str) -> List[str]:
    ids: List[str] = []
    for spec in gym.envs.registry.values():
        env_id = getattr(spec, "id", None)
        if isinstance(env_id, str) and env_id.startswith(prefix):
            ids.append(env_id)
    return sorted(ids)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _parse_csv_tasks(tasks_csv: str) -> List[str]:
    vals = [x.strip() for x in tasks_csv.split(",")]
    return [x for x in vals if x]


def _fill_from_available(
    preferred: Sequence[str],
    available: Sequence[str],
    required: int,
) -> List[str]:
    selected = [task for task in preferred if task in available]
    if len(selected) < required:
        extras = [task for task in available if task not in selected]
        selected.extend(extras[: required - len(selected)])
    if len(selected) < required:
        raise RuntimeError(
            f"No hay suficientes tareas registradas para construir preset de {required}. "
            f"Disponibles={len(available)}"
        )
    return selected[:required]


def resolve_task_sequence(
    task_preset: str,
    tasks_csv: Optional[str],
) -> List[str]:
    _ensure_minigrid()
    available = _registered_env_ids(prefix="MiniGrid-")
    if not available:
        raise RuntimeError(
            "No se encontraron entornos registrados con prefijo `MiniGrid-`. "
            "Revisa instalación de minigrid."
        )

    if tasks_csv:
        tasks = _parse_csv_tasks(tasks_csv)
        if not tasks:
            raise ValueError("`--tasks` quedó vacío.")
        return tasks

    preset = task_preset.lower().strip()
    if preset not in {"smoke4", "cw10", "cw20", "all"}:
        raise ValueError(
            "`--task-preset` inválido. Usa smoke4/cw10/cw20/all o define --tasks."
        )
    if preset == "all":
        return list(available)

    if preset == "smoke4":
        target = min(4, len(available))
        if target < 4:
            print(
                f"[WARN] Solo hay {len(available)} tareas registradas; "
                "smoke4 se reducirá automáticamente."
            )
        return _fill_from_available(PREFERRED_MINIGRID_SMOKE4, available, target)

    cw10_target = min(10, len(available))
    if cw10_target < 10:
        print(
            f"[WARN] Solo hay {len(available)} tareas registradas; "
            "cw10 se reducirá automáticamente."
        )
    cw10 = _fill_from_available(PREFERRED_MINIGRID_CW10, available, cw10_target)
    if preset == "cw10":
        return cw10
    return cw10 + cw10


class FlattenToFloatObs(gym.ObservationWrapper):
    """Convierte observaciones Box a vector float32, opcionalmente normalizado."""

    def __init__(self, env: gym.Env, normalize: bool = True):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError("FlattenToFloatObs requiere observation_space Box.")
        self.source_shape: tuple[int, ...] = tuple(int(x) for x in obs_space.shape)
        self.source_dtype: str = str(obs_space.dtype)

        low = np.asarray(obs_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(obs_space.high, dtype=np.float32).reshape(-1)
        self._normalize = bool(normalize)
        self._scale: Optional[float] = None

        if self._normalize:
            finite_high = np.isfinite(high)
            if bool(np.any(finite_high)):
                max_high = float(np.max(np.abs(high[finite_high])))
                if max_high > 1.0:
                    self._scale = max_high
                    low = low / max_high
                    high = high / max_high

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def observation(self, observation: Any) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if self._scale is not None:
            obs = obs / self._scale
        return obs


def _space_signature(space: spaces.Space) -> tuple[str, tuple[int, ...], Any]:
    if isinstance(space, spaces.Box):
        return ("Box", tuple(space.shape), str(space.dtype))
    if isinstance(space, spaces.Discrete):
        return ("Discrete", (int(space.n),), str(type(space.start)))
    return (space.__class__.__name__, tuple(), "")


def _make_minigrid_env(
    env_id: str,
    seed: int,
    max_episode_steps: int,
    render_mode: Optional[str],
    disable_env_checker: bool,
    obs_mode: str,
    normalize_obs: bool,
) -> gym.Env:
    _ensure_minigrid()
    env = gym.make(
        env_id,
        render_mode=render_mode,
        disable_env_checker=disable_env_checker,
    )

    from minigrid.wrappers import FlatObsWrapper, ImgObsWrapper

    if obs_mode == "flat":
        env = FlatObsWrapper(env)
    else:
        env = ImgObsWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = FlattenToFloatObs(env, normalize=normalize_obs)
    env.action_space.seed(seed)
    env.reset(seed=seed)
    return env


class ContinualTaskSuiteEnv(gym.Env[np.ndarray, int]):
    """Env multi-tarea donde cada episodio usa una tarea activa."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        task_names: Sequence[str],
        seed: int,
        max_episode_steps: int = 200,
        append_task_id: bool = False,
        render_mode: Optional[str] = None,
        disable_env_checker: bool = True,
        obs_mode: str = "image",
        normalize_obs: bool = True,
    ):
        super().__init__()
        if not task_names:
            raise ValueError("task_names no puede estar vacío.")

        self.sequence_tasks = list(task_names)
        self.unique_tasks = list(dict.fromkeys(task_names))
        self.append_task_id = append_task_id
        self._rng = np.random.default_rng(seed)
        self._active_task: Optional[str] = self.unique_tasks[0]
        self._current_task: Optional[str] = None
        self._ep_success = 0.0

        self._envs: Dict[str, gym.Env] = {}
        for i, task in enumerate(self.unique_tasks):
            self._envs[task] = _make_minigrid_env(
                env_id=task,
                seed=seed + i,
                max_episode_steps=max_episode_steps,
                render_mode=render_mode,
                disable_env_checker=disable_env_checker,
                obs_mode=obs_mode,
                normalize_obs=normalize_obs,
            )

        base_env = self._envs[self.unique_tasks[0]]
        self.action_space = base_env.action_space
        if not isinstance(base_env.observation_space, spaces.Box):
            raise TypeError("Se esperaba observation_space Box tras wrappers.")
        self.state_source_shape: Optional[tuple[int, ...]] = getattr(
            base_env, "source_shape", None
        )
        self.state_source_dtype: Optional[str] = getattr(base_env, "source_dtype", None)

        for task, env in self._envs.items():
            if _space_signature(env.action_space) != _space_signature(self.action_space):
                raise ValueError(
                    "Todas las tareas deben compartir action_space. "
                    f"Base={_space_signature(self.action_space)} "
                    f"task={task} -> {_space_signature(env.action_space)}"
                )
            if _space_signature(env.observation_space) != _space_signature(
                base_env.observation_space
            ):
                raise ValueError(
                    "Todas las tareas deben compartir observation_space tras wrappers. "
                    f"Base={_space_signature(base_env.observation_space)} "
                    f"task={task} -> {_space_signature(env.observation_space)}"
                )

        base_space = base_env.observation_space
        low = np.array(base_space.low, dtype=np.float32, copy=True)
        high = np.array(base_space.high, dtype=np.float32, copy=True)
        self._num_tasks = len(self.unique_tasks)

        if append_task_id:
            low = np.concatenate([low, np.zeros(self._num_tasks, dtype=np.float32)])
            high = np.concatenate([high, np.ones(self._num_tasks, dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def set_task(self, task_name: Optional[str]) -> None:
        if task_name is None:
            self._active_task = None
            return
        if task_name not in self._envs:
            raise ValueError(f"Tarea desconocida: {task_name}")
        self._active_task = task_name

    def _task_idx(self, task: str) -> int:
        return self.unique_tasks.index(task)

    def _task_one_hot(self, task: str) -> np.ndarray:
        one_hot = np.zeros(self._num_tasks, dtype=np.float32)
        one_hot[self._task_idx(task)] = 1.0
        return one_hot

    def _augment_obs(self, obs: np.ndarray, task: str) -> np.ndarray:
        out = np.asarray(obs, dtype=np.float32)
        if not self.append_task_id:
            return out
        return np.concatenate([out, self._task_one_hot(task)], axis=0).astype(np.float32)

    @staticmethod
    def _safe_success_value(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    @classmethod
    def _augment_info(
        cls,
        info: Dict[str, Any],
        task: str,
        task_idx: int,
        reward: Optional[float] = None,
        terminated: Optional[bool] = None,
    ) -> Dict[str, Any]:
        out = dict(info)
        out["task_name"] = task
        out["task_idx"] = task_idx

        success = out.get("is_success", out.get("success", None))
        if success is None and terminated is not None and reward is not None:
            # MiniGrid normalmente usa reward>0 al resolver la tarea.
            success = 1.0 if (bool(terminated) and float(reward) > 0.0) else 0.0
        out["is_success"] = cls._safe_success_value(success)
        return out

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}
        task = options.get("task_name")
        if task is None:
            if self._active_task is None:
                task = self.unique_tasks[int(self._rng.integers(0, len(self.unique_tasks)))]
            else:
                task = self._active_task
        if task not in self._envs:
            raise ValueError(f"Tarea inválida: {task}")

        child_options = dict(options)
        child_options.pop("task_name", None)

        obs, info = self._envs[task].reset(seed=seed, options=child_options or None)
        self._current_task = task
        self._ep_success = 0.0
        idx = self._task_idx(task)
        out_info = self._augment_info(info, task, idx, reward=None, terminated=None)
        out_info["is_success"] = self._ep_success
        return self._augment_obs(np.asarray(obs), task), out_info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._current_task is None:
            raise RuntimeError("Se llamó step() antes de reset().")
        env = self._envs[self._current_task]
        obs, reward, terminated, truncated, info = env.step(action)
        idx = self._task_idx(self._current_task)
        out_info = self._augment_info(
            info,
            self._current_task,
            idx,
            reward=float(reward),
            terminated=bool(terminated),
        )

        self._ep_success = max(
            self._ep_success,
            self._safe_success_value(out_info.get("is_success", 0.0)),
        )
        out_info["is_success"] = float(self._ep_success)

        return (
            self._augment_obs(np.asarray(obs), self._current_task),
            float(reward),
            bool(terminated),
            bool(truncated),
            out_info,
        )

    def close(self) -> None:
        for env in self._envs.values():
            env.close()


def _suite_env(env: gym.Env) -> ContinualTaskSuiteEnv:
    base = env.unwrapped
    if not isinstance(base, ContinualTaskSuiteEnv):
        raise TypeError("Env base no es ContinualTaskSuiteEnv.")
    return base


class LowRankAdapter(nn.Module):
    """Adaptador low-rank residual: x + scale * B(A(x))."""

    def __init__(self, dim: int, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank debe ser > 0, recibido={rank}.")
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        self.scale = float(alpha) / float(rank)
        nn.init.normal_(self.A.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.B(self.A(x)) * self.scale


class MultiHeadLinear(nn.Module):
    """Multiple linear heads indexed by task. One active at a time."""

    def __init__(self, in_features: int, out_features: int, num_heads: int):
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(num_heads)]
        )
        self._active_head: int = 0

    def set_active_head(self, idx: int) -> None:
        if not (0 <= idx < self.num_heads):
            raise ValueError(f"head idx {idx} out of range [0, {self.num_heads})")
        self._active_head = idx

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.heads[self._active_head](x)

    def __repr__(self) -> str:
        return (
            f"MultiHeadLinear(in={self.in_features}, out={self.out_features}, "
            f"heads={self.num_heads}, active={self._active_head})"
        )


def _apply_task_adapters(
    h: th.Tensor,
    one_hot: th.Tensor,
    adapters: nn.ModuleList,
) -> th.Tensor:
    """Rutea y aplica adaptadores por tarea usando one-hot."""
    task_indices = th.argmax(one_hot, dim=-1)
    flat_task_indices = task_indices.reshape(-1)
    if flat_task_indices.numel() == 0:
        return h

    if bool(th.all(flat_task_indices == flat_task_indices[0]).item()):
        idx = int(flat_task_indices[0].item())
        return h + adapters[idx](h)

    flat_h = h.reshape(-1, h.shape[-1])
    flat_delta = th.zeros_like(flat_h)
    for idx, adapter in enumerate(adapters):
        mask = flat_task_indices == idx
        if not bool(mask.any().item()):
            continue
        flat_delta[mask] = adapter(flat_h[mask])
    return h + flat_delta.view_as(h)


def _infer_image_shape(
    state_dim: int,
    image_shape: Optional[Sequence[int]],
    image_channels: int,
) -> tuple[int, int, int]:
    if image_shape is not None:
        if len(image_shape) != 3:
            raise ValueError(f"image_shape debe tener 3 dimensiones, recibido={image_shape}.")
        h, w, c = (int(image_shape[0]), int(image_shape[1]), int(image_shape[2]))
        if h <= 0 or w <= 0 or c <= 0:
            raise ValueError(f"image_shape inválido: {image_shape}")
        if h * w * c != state_dim:
            raise ValueError(
                f"image_shape={image_shape} incompatible con state_dim={state_dim}."
            )
        return h, w, c

    if image_channels <= 0:
        raise ValueError(f"image_channels debe ser > 0, recibido={image_channels}.")
    if state_dim % image_channels != 0:
        raise ValueError(
            "No se pudo inferir forma de imagen: state_dim no divisible por image_channels "
            f"({state_dim} % {image_channels} != 0)."
        )
    spatial = state_dim // image_channels
    side = int(round(math.sqrt(spatial)))
    if side * side != spatial:
        raise ValueError(
            "No se pudo inferir imagen cuadrada desde observation_space. "
            f"state_dim={state_dim}, image_channels={image_channels}"
        )
    return side, side, image_channels


class RoutedAdapterBackboneExtractor(BaseFeaturesExtractor):
    """Backbone MLP compartido + adaptadores por tarea."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        num_tasks: int = 2,
        adapter_rank: int = 16,
        adapter_alpha: float = 16.0,
        backbone_hidden_dim: int = 256,
    ):
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("RoutedAdapterBackboneExtractor requiere observation_space Box.")
        if num_tasks <= 0:
            raise ValueError(f"num_tasks debe ser > 0, recibido={num_tasks}.")

        super().__init__(observation_space, features_dim)

        obs_dim = int(np.prod(observation_space.shape))
        if obs_dim <= num_tasks:
            raise ValueError(
                "obs_dim debe ser mayor a num_tasks. "
                "Activa --append-task-id y valida el número de tareas."
            )

        self.num_tasks = int(num_tasks)
        self.state_dim = obs_dim - self.num_tasks
        self._features_dim = int(features_dim)

        self.backbone = nn.Sequential(
            nn.Linear(self.state_dim, backbone_hidden_dim),
            nn.ReLU(),
            nn.Linear(backbone_hidden_dim, features_dim),
            nn.ReLU(),
        )
        self.adapters = nn.ModuleList(
            [
                LowRankAdapter(features_dim, rank=adapter_rank, alpha=adapter_alpha)
                for _ in range(self.num_tasks)
            ]
        )

    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_adapter(self, task_idx: int) -> None:
        if not (0 <= task_idx < self.num_tasks):
            raise ValueError(f"task_idx fuera de rango: {task_idx}")
        for p in self.adapters[task_idx].parameters():
            p.requires_grad = True

    def forward(self, obs: th.Tensor) -> th.Tensor:
        state = obs[..., : self.state_dim]
        one_hot = obs[..., self.state_dim : self.state_dim + self.num_tasks]
        h = self.backbone(state)
        return _apply_task_adapters(h, one_hot, self.adapters)


class RoutedAdapterCNNExtractor(BaseFeaturesExtractor):
    """Backbone CNN compartido + adaptadores por tarea para observaciones visuales."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        num_tasks: int = 2,
        adapter_rank: int = 16,
        adapter_alpha: float = 16.0,
        image_shape: Optional[Sequence[int]] = None,
        image_channels: int = 3,
    ):
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("RoutedAdapterCNNExtractor requiere observation_space Box.")
        if num_tasks <= 0:
            raise ValueError(f"num_tasks debe ser > 0, recibido={num_tasks}.")

        super().__init__(observation_space, features_dim)

        obs_dim = int(np.prod(observation_space.shape))
        if obs_dim <= num_tasks:
            raise ValueError(
                "obs_dim debe ser mayor a num_tasks. "
                "Activa --append-task-id y valida el número de tareas."
            )

        self.num_tasks = int(num_tasks)
        self.state_dim = obs_dim - self.num_tasks
        self._features_dim = int(features_dim)

        height, width, channels = _infer_image_shape(
            state_dim=self.state_dim,
            image_shape=image_shape,
            image_channels=image_channels,
        )
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)

        self.cnn = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            dummy = th.zeros(1, self.channels, self.height, self.width)
            n_flatten = int(self.cnn(dummy).shape[1])

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        self.adapters = nn.ModuleList(
            [
                LowRankAdapter(features_dim, rank=adapter_rank, alpha=adapter_alpha)
                for _ in range(self.num_tasks)
            ]
        )

    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_adapter(self, task_idx: int) -> None:
        if not (0 <= task_idx < self.num_tasks):
            raise ValueError(f"task_idx fuera de rango: {task_idx}")
        for p in self.adapters[task_idx].parameters():
            p.requires_grad = True

    def forward(self, obs: th.Tensor) -> th.Tensor:
        state_flat = obs[..., : self.state_dim]
        one_hot = obs[..., self.state_dim : self.state_dim + self.num_tasks]

        batch_shape = tuple(state_flat.shape[:-1])
        flat_state = state_flat.reshape(-1, self.state_dim)
        images = flat_state.reshape(-1, self.height, self.width, self.channels).permute(0, 3, 1, 2)

        h = self.linear(self.cnn(images))
        h = h.reshape(*batch_shape, self._features_dim)
        return _apply_task_adapters(h, one_hot, self.adapters)


class BackboneOnlyExtractor(BaseFeaturesExtractor):
    """Backbone MLP sin adaptadores."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        num_tasks: int = 0,
        backbone_hidden_dim: int = 256,
    ):
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("BackboneOnlyExtractor requiere observation_space Box.")
        super().__init__(observation_space, features_dim)

        obs_dim = int(np.prod(observation_space.shape))
        self.num_tasks = max(0, int(num_tasks))
        if self.num_tasks > 0 and obs_dim <= self.num_tasks:
            raise ValueError(
                "obs_dim debe ser mayor a num_tasks para separar state/task-id."
            )
        self.state_dim = obs_dim - self.num_tasks if self.num_tasks > 0 else obs_dim
        self._features_dim = int(features_dim)

        self.backbone = nn.Sequential(
            nn.Linear(self.state_dim, backbone_hidden_dim),
            nn.ReLU(),
            nn.Linear(backbone_hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        state = obs[..., : self.state_dim]
        return self.backbone(state)


class BackboneOnlyCNNExtractor(BaseFeaturesExtractor):
    """Backbone CNN sin adaptadores para observaciones visuales."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        num_tasks: int = 0,
        image_shape: Optional[Sequence[int]] = None,
        image_channels: int = 3,
    ):
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("BackboneOnlyCNNExtractor requiere observation_space Box.")
        super().__init__(observation_space, features_dim)

        obs_dim = int(np.prod(observation_space.shape))
        self.num_tasks = max(0, int(num_tasks))
        if self.num_tasks > 0 and obs_dim <= self.num_tasks:
            raise ValueError("obs_dim debe ser mayor a num_tasks.")
        self.state_dim = obs_dim - self.num_tasks if self.num_tasks > 0 else obs_dim
        self._features_dim = int(features_dim)

        height, width, channels = _infer_image_shape(
            state_dim=self.state_dim,
            image_shape=image_shape,
            image_channels=image_channels,
        )
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)

        self.cnn = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            dummy = th.zeros(1, self.channels, self.height, self.width)
            n_flatten = int(self.cnn(dummy).shape[1])
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        state_flat = obs[..., : self.state_dim]
        flat_state = state_flat.reshape(-1, self.state_dim)
        images = flat_state.reshape(-1, self.height, self.width, self.channels).permute(0, 3, 1, 2)
        h = self.linear(self.cnn(images))
        out_shape = tuple(state_flat.shape[:-1]) + (self._features_dim,)
        return h.reshape(out_shape)


def _iter_routed_extractors(policy: Any) -> List[nn.Module]:
    candidates = [getattr(policy, "features_extractor", None)]

    actor = getattr(policy, "actor", None)
    if actor is not None:
        candidates.append(getattr(actor, "features_extractor", None))

    critic = getattr(policy, "critic", None)
    if critic is not None:
        candidates.append(getattr(critic, "features_extractor", None))

    extractors: List[nn.Module] = []
    seen: set[int] = set()
    for maybe_extractor in candidates:
        if not isinstance(
            maybe_extractor,
            (RoutedAdapterBackboneExtractor, RoutedAdapterCNNExtractor),
        ):
            continue
        ptr = id(maybe_extractor)
        if ptr in seen:
            continue
        seen.add(ptr)
        extractors.append(maybe_extractor)
    return extractors


def _adapter_task_idx_from_param_name(name: str) -> Optional[int]:
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part != "adapters":
            continue
        maybe_idx = parts[i + 1]
        if maybe_idx.isdigit():
            return int(maybe_idx)
    return None


def _multi_head_idx_from_param_name(name: str) -> Optional[int]:
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part != "heads":
            continue
        maybe_idx = parts[i + 1]
        if maybe_idx.isdigit():
            return int(maybe_idx)
    return None


def _clone_optimizer(
    optimizer: th.optim.Optimizer,
    params: Iterable[th.nn.Parameter],
) -> th.optim.Optimizer:
    cls = optimizer.__class__
    kwargs = dict(optimizer.defaults)
    if optimizer.param_groups:
        kwargs["lr"] = float(optimizer.param_groups[0].get("lr", kwargs.get("lr", 0.0)))
    return cls(params, **kwargs)


def _install_multi_heads(model: BaseAlgorithm, num_tasks: int) -> None:
    """Instala heads por tarea para la salida de política cuando sea lineal."""
    policy = model.policy
    device = next(policy.parameters()).device

    replaced: List[str] = []

    action_net = getattr(policy, "action_net", None)
    if isinstance(action_net, nn.Linear):
        multi = MultiHeadLinear(action_net.in_features, action_net.out_features, num_tasks)
        with th.no_grad():
            multi.heads[0].weight.copy_(action_net.weight)
            multi.heads[0].bias.copy_(action_net.bias)
        policy.action_net = multi.to(device)
        replaced.append("policy.action_net")

    value_net = getattr(policy, "value_net", None)
    if isinstance(value_net, nn.Linear):
        multi = MultiHeadLinear(value_net.in_features, value_net.out_features, num_tasks)
        with th.no_grad():
            multi.heads[0].weight.copy_(value_net.weight)
            multi.heads[0].bias.copy_(value_net.bias)
        policy.value_net = multi.to(device)
        replaced.append("policy.value_net")

    actor = getattr(policy, "actor", None)
    if actor is not None:
        for attr_name in ("mu", "log_std"):
            orig = getattr(actor, attr_name, None)
            if orig is None or not isinstance(orig, nn.Linear):
                continue
            multi = MultiHeadLinear(orig.in_features, orig.out_features, num_tasks)
            with th.no_grad():
                multi.heads[0].weight.copy_(orig.weight)
                multi.heads[0].bias.copy_(orig.bias)
            setattr(actor, attr_name, multi.to(device))
            replaced.append(f"actor.{attr_name}")

    if hasattr(policy, "optimizer") and getattr(policy, "optimizer") is not None:
        policy.optimizer = _clone_optimizer(policy.optimizer, policy.parameters())

    if actor is not None and getattr(actor, "optimizer", None) is not None:
        actor.optimizer = _clone_optimizer(actor.optimizer, actor.parameters())

    if replaced:
        print(f"[MULTI-HEAD] Installed {num_tasks} heads on: {', '.join(replaced)}")
    else:
        print("[MULTI-HEAD] No se encontró salida lineal compatible para instalar heads.")


def _set_active_heads(model: BaseAlgorithm, task_idx: int) -> None:
    for module in model.policy.modules():
        if isinstance(module, MultiHeadLinear):
            module.set_active_head(task_idx)


def _set_trainable_for_task(
    model: BaseAlgorithm,
    task_idx: int,
    phase: int,
    warmup_tasks: int,
    train_active_adapter_in_warmup: bool,
    train_actor_heads_after_warmup: bool,
    train_full_critic_after_warmup: bool,
) -> str:
    """Configure trainable params for a continual phase.

    Warm-up: full shared network + active per-task heads.
    Post warm-up (PEFT): active adapter + active per-task heads only, with
    optional unfreezing of shared actor/value trunks.
    """
    policy = model.policy

    if phase <= warmup_tasks:
        for name, p in policy.named_parameters():
            adapter_idx = _adapter_task_idx_from_param_name(name)
            if adapter_idx is not None:
                p.requires_grad = bool(
                    train_active_adapter_in_warmup and adapter_idx == task_idx
                )
                continue
            head_idx = _multi_head_idx_from_param_name(name)
            if head_idx is not None:
                p.requires_grad = head_idx == task_idx
                continue
            p.requires_grad = True

        if train_active_adapter_in_warmup:
            return "warmup_full+active_adapter+active_heads"
        return "warmup_full+active_heads"

    for p in policy.parameters():
        p.requires_grad = False

    extractors = _iter_routed_extractors(policy)
    if not extractors:
        raise TypeError(
            "No se detectó extractor de adaptadores ruteados en la policy. "
            "Revisa --adapter-enabled y policy_kwargs."
        )

    for extractor in extractors:
        extractor.unfreeze_adapter(task_idx)

    mode_parts = ["adapter"]

    for module in policy.modules():
        if isinstance(module, MultiHeadLinear):
            for p in module.heads[task_idx].parameters():
                p.requires_grad = True
    mode_parts.append("active_heads")

    if train_actor_heads_after_warmup:
        for name, p in policy.named_parameters():
            if "features_extractor" in name:
                continue
            head_idx = _multi_head_idx_from_param_name(name)
            if head_idx is not None:
                p.requires_grad = head_idx == task_idx
                continue
            if name.startswith("mlp_extractor.policy_net"):
                p.requires_grad = True
        mode_parts.append("policy_trunk")

    if train_full_critic_after_warmup:
        for name, p in policy.named_parameters():
            if "features_extractor" in name:
                continue
            if name.startswith("mlp_extractor.value_net"):
                p.requires_grad = True
        mode_parts.append("value_trunk")

    return "+".join(mode_parts)


def _count_trainable_params(model: BaseAlgorithm) -> int:
    return sum(p.numel() for p in model.policy.parameters() if p.requires_grad)


def _count_trainable_adapter_params(model: BaseAlgorithm) -> int:
    trainable = 0
    for extractor in _iter_routed_extractors(model.policy):
        for p in extractor.adapters.parameters():
            if p.requires_grad:
                trainable += p.numel()
    return trainable


def _reset_optimizers_for_task(model: BaseAlgorithm) -> List[str]:
    reset_names: List[str] = []
    policy_opt = getattr(model.policy, "optimizer", None)
    if policy_opt is not None:
        model.policy.optimizer = _clone_optimizer(policy_opt, model.policy.parameters())
        reset_names.append("policy")
    return reset_names


def _build_model(
    env: gym.Env,
    seed: int,
    device: str,
    tensorboard_dir: Path,
    obs_mode: str,
    adapter_enabled: bool,
    adapter_num_tasks: int,
    adapter_rank: int,
    adapter_alpha: float,
    adapter_features_dim: int,
    adapter_backbone_hidden_dim: int,
    ppo_learning_rate: float,
    ppo_n_steps: int,
    ppo_batch_size: int,
    ppo_n_epochs: int,
    ppo_gamma: float,
    ppo_gae_lambda: float,
    ppo_clip_range: float,
    ppo_ent_coef: float,
    ppo_vf_coef: float,
    ppo_max_grad_norm: float,
    ppo_target_kl: Optional[float],
) -> BaseAlgorithm:
    suite = _suite_env(env)
    state_source_shape = getattr(suite, "state_source_shape", None)
    use_cnn = bool(obs_mode == "image" and state_source_shape is not None and len(state_source_shape) == 3)
    if obs_mode == "image" and not use_cnn:
        print(
            "[WARN] obs_mode=image pero no se detectó shape de imagen 3D; "
            "usando extractor MLP."
        )

    num_tasks_in_obs = adapter_num_tasks if bool(getattr(suite, "append_task_id", False)) else 0

    if adapter_enabled:
        extractor_class: type[BaseFeaturesExtractor]
        extractor_kwargs: Dict[str, Any]
        if use_cnn:
            extractor_class = RoutedAdapterCNNExtractor
            extractor_kwargs = dict(
                features_dim=adapter_features_dim,
                num_tasks=adapter_num_tasks,
                adapter_rank=adapter_rank,
                adapter_alpha=adapter_alpha,
                image_shape=state_source_shape,
                image_channels=int(state_source_shape[2]),
            )
        else:
            extractor_class = RoutedAdapterBackboneExtractor
            extractor_kwargs = dict(
                features_dim=adapter_features_dim,
                num_tasks=adapter_num_tasks,
                adapter_rank=adapter_rank,
                adapter_alpha=adapter_alpha,
                backbone_hidden_dim=adapter_backbone_hidden_dim,
            )
        policy_kwargs: Dict[str, Any] = dict(
            net_arch=dict(
                pi=[adapter_backbone_hidden_dim],
                vf=[adapter_backbone_hidden_dim],
            ),
            features_extractor_class=extractor_class,
            features_extractor_kwargs=extractor_kwargs,
        )
    else:
        if use_cnn:
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[adapter_backbone_hidden_dim],
                    vf=[adapter_backbone_hidden_dim],
                ),
                features_extractor_class=BackboneOnlyCNNExtractor,
                features_extractor_kwargs=dict(
                    features_dim=adapter_features_dim,
                    num_tasks=num_tasks_in_obs,
                    image_shape=state_source_shape,
                    image_channels=int(state_source_shape[2]),
                ),
            )
        else:
            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                features_extractor_class=BackboneOnlyExtractor,
                features_extractor_kwargs=dict(
                    features_dim=adapter_features_dim,
                    num_tasks=num_tasks_in_obs,
                    backbone_hidden_dim=adapter_backbone_hidden_dim,
                ),
            )

    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(tensorboard_dir),
        device=device,
        learning_rate=ppo_learning_rate,
        n_steps=ppo_n_steps,
        batch_size=ppo_batch_size,
        n_epochs=ppo_n_epochs,
        gamma=ppo_gamma,
        gae_lambda=ppo_gae_lambda,
        clip_range=ppo_clip_range,
        ent_coef=ppo_ent_coef,
        vf_coef=ppo_vf_coef,
        max_grad_norm=ppo_max_grad_norm,
        target_kl=ppo_target_kl,
        policy_kwargs=policy_kwargs,
    )


def evaluate_task(
    model: BaseAlgorithm,
    env: gym.Env,
    task_name: str,
    episodes: int,
    deterministic: bool,
) -> Dict[str, float]:
    returns: List[float] = []
    successes: List[float] = []
    lengths: List[int] = []

    for _ in range(episodes):
        obs, _ = env.reset(options={"task_name": task_name})
        done = False
        ep_ret = 0.0
        ep_len = 0
        ep_success = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
            ep_len += 1
            step_success = info.get("is_success", info.get("success", 0.0))
            try:
                ep_success = max(ep_success, float(step_success))
            except Exception:
                pass
            if done and ep_success <= 0.0 and float(reward) > 0.0 and bool(terminated):
                ep_success = 1.0
        returns.append(ep_ret)
        successes.append(ep_success)
        lengths.append(ep_len)

    return {
        "mean_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_ep_len": float(np.mean(lengths)),
    }


@dataclass
class EvalRecord:
    phase: int
    trained_task: str
    eval_task: str
    timesteps: int
    mean_return: float
    success_rate: float
    mean_ep_len: float


class HealthPlotCheckpointCallback(BaseCallback):
    """Health en consola + CSV + plots + checkpoints periódicos y por mejora."""

    def __init__(
        self,
        run_dir: Path,
        health_freq: int = 5_000,
        plot_freq: int = 20_000,
        checkpoint_freq: int = 50_000,
        rolling_window: int = 100,
        save_best_checkpoints: bool = True,
        improvement_metric: str = "success",
        min_improvement_delta: float = 1e-6,
        action_component_idx: int = 0,
        action_window_size: int = 50_000,
        action_hist_bins: int = 31,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.run_dir = run_dir
        self.health_freq = max(1, int(health_freq))
        self.plot_freq = max(1, int(plot_freq))
        self.checkpoint_freq = max(1, int(checkpoint_freq))
        self.rolling_window = max(1, int(rolling_window))
        self.save_best_checkpoints = bool(save_best_checkpoints)
        self.improvement_metric = improvement_metric
        self.min_improvement_delta = float(min_improvement_delta)
        self.action_component_idx = int(action_component_idx)
        self.action_window_size = max(1, int(action_window_size))
        self.action_hist_bins = max(3, int(action_hist_bins))

        self.action_mean_key = "action_mean"
        self.action_std_key = "action_std"

        self.csv_path = self.run_dir / "train_metrics.csv"
        self.plot_dir = self.run_dir / "plots"
        self.ckpt_dir = self.run_dir / "checkpoints"

        self.last_health_step = 0
        self.last_plot_step = 0
        self.last_ckpt_step = 0

        self.ts: List[int] = []
        self.mean_returns: List[float] = []
        self.success_rates: List[float] = []
        self.mean_ep_lens: List[float] = []
        self.alphas: List[float] = []
        self.active_tasks: List[str] = []
        self.action_component_means: List[float] = []
        self.action_component_stds: List[float] = []

        self._successes: List[float] = []
        self._running_ep_success: Dict[int, float] = {}
        self._action_values: deque[float] = deque(maxlen=self.action_window_size)
        self.best_score = -float("inf")
        self.t0 = time.time()

    def _csv_fieldnames(self) -> List[str]:
        return [
            "timesteps",
            "elapsed_sec",
            "active_task",
            "mean_return",
            "success_rate",
            "mean_ep_len",
            "alpha",
            self.action_mean_key,
            self.action_std_key,
        ]

    def _on_training_start(self) -> None:
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=self._csv_fieldnames(),
                )
                writer.writeheader()

    def _active_task_name(self) -> str:
        try:
            task = self.training_env.get_attr("_active_task")[0]
            if task is None:
                task = self.training_env.get_attr("_current_task")[0]
            return str(task)
        except Exception:
            return "?"

    def _read_alpha(self) -> float:
        # PPO no usa alpha de SAC; devolvemos NaN.
        return float("nan")

    def _collect_episode_stats(self) -> Dict[str, float]:
        if getattr(self.model, "ep_info_buffer", None):
            rets = [float(ep.get("r", np.nan)) for ep in self.model.ep_info_buffer]
            lens = [float(ep.get("l", np.nan)) for ep in self.model.ep_info_buffer]
            mean_ret = float(np.nanmean(rets)) if rets else float("nan")
            mean_len = float(np.nanmean(lens)) if lens else float("nan")
        else:
            mean_ret = float("nan")
            mean_len = float("nan")
        if self._successes:
            sr = float(np.mean(self._successes[-self.rolling_window :]))
        else:
            sr = float("nan")
        return {
            "mean_return": mean_ret,
            "success_rate": sr,
            "mean_ep_len": mean_len,
        }

    def _append_csv_row(self, row: Dict[str, Any]) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=self._csv_fieldnames(),
            )
            writer.writerow(row)

    def _update_action_buffer(self) -> None:
        actions = self.locals.get("actions")
        if actions is None:
            return
        arr = np.asarray(actions)
        if arr.size == 0:
            return

        if arr.ndim == 0:
            values = np.asarray([arr.item()], dtype=np.float64)
        elif arr.ndim == 1:
            values = arr.astype(np.float64).reshape(-1)
        elif arr.shape[-1] > self.action_component_idx:
            values = arr[..., self.action_component_idx].astype(np.float64).reshape(-1)
        else:
            values = arr.astype(np.float64).reshape(-1)

        for value in values:
            if np.isfinite(value):
                self._action_values.append(float(value))

    def _action_stats(self) -> tuple[float, float]:
        if not self._action_values:
            return float("nan"), float("nan")
        arr = np.asarray(self._action_values, dtype=np.float64)
        return float(np.mean(arr)), float(np.std(arr))

    def _plot_action_hist(self) -> None:
        if not self._action_values:
            return
        values = np.asarray(self._action_values, dtype=np.float64)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        rounded = np.round(values)
        is_discrete = bool(np.all(np.abs(values - rounded) < 1e-9))
        if is_discrete:
            ints = rounded.astype(int)
            min_v = int(np.min(ints))
            max_v = int(np.max(ints))
            bins = np.arange(min_v - 0.5, max_v + 1.5, 1.0)
            ax.hist(ints, bins=bins, color="steelblue", alpha=0.85)
            ax.set_xticks(np.arange(min_v, max_v + 1))
        else:
            ax.hist(values, bins=self.action_hist_bins, color="steelblue", alpha=0.85)

        ax.set_title("Action Histogram (latest window)")
        ax.set_xlabel("action value")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            self.plot_dir / f"action_hist_step_{self.num_timesteps:012d}.png",
            dpi=130,
        )
        plt.close(fig)

    def _plot_metrics(self) -> None:
        if not self.ts:
            return
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        axes[0].plot(self.ts, self.mean_returns, "o-", ms=3)
        axes[0].set_title("Rolling Mean Return")
        axes[0].set_xlabel("timesteps")
        axes[0].set_ylabel("return")
        axes[0].grid(alpha=0.3)

        axes[1].plot(self.ts, self.success_rates, "o-", ms=3, color="green")
        axes[1].set_title(f"Success Rate (last {self.rolling_window})")
        axes[1].set_xlabel("timesteps")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(alpha=0.3)

        axes[2].plot(self.ts, self.mean_ep_lens, "o-", ms=3, color="darkorange")
        axes[2].set_title("Mean Episode Length")
        axes[2].set_xlabel("timesteps")
        axes[2].set_ylabel("steps")
        axes[2].grid(alpha=0.3)

        alpha_arr = np.asarray(self.alphas, dtype=float)
        if np.isfinite(alpha_arr).any():
            axes[3].plot(self.ts, self.alphas, "o-", ms=3, color="purple")
        else:
            axes[3].text(0.5, 0.5, "alpha n/a", ha="center", va="center")
        axes[3].set_title("Entropy Alpha")
        axes[3].set_xlabel("timesteps")
        axes[3].set_ylabel("alpha")
        axes[3].grid(alpha=0.3)

        fig.suptitle("Training Health Metrics")
        plt.tight_layout()
        fig.savefig(self.plot_dir / f"health_step_{self.num_timesteps:012d}.png", dpi=130)
        plt.close(fig)

    def _maybe_save_best(self, stats: Dict[str, float]) -> None:
        if not self.save_best_checkpoints:
            return
        if self.improvement_metric == "return":
            score = stats["mean_return"]
        else:
            score = stats["success_rate"]
            if math.isnan(score):
                score = stats["mean_return"]
        if math.isnan(score):
            return
        if score > (self.best_score + self.min_improvement_delta):
            self.best_score = score
            best_step = self.ckpt_dir / f"best_step_{self.num_timesteps:012d}"
            self.model.save(str(best_step))
            self.model.save(str(self.run_dir / "best_model"))
            if self.verbose:
                print(
                    f"[BEST] t={self.num_timesteps:,} "
                    f"{self.improvement_metric}={score:.4f} -> {best_step.name}.zip"
                )

    def reset_task_window(self, model: Optional[BaseAlgorithm] = None) -> None:
        self._successes.clear()
        self._running_ep_success.clear()
        if model is not None and getattr(model, "ep_info_buffer", None) is not None:
            try:
                model.ep_info_buffer.clear()
            except Exception:
                pass

    def _on_step(self) -> bool:
        self._update_action_buffer()

        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            terminal_info = info.get("terminal_info", info)
            step_success = terminal_info.get("is_success", terminal_info.get("success", 0.0))
            prev_success = self._running_ep_success.get(env_idx, 0.0)
            try:
                step_success_value = float(step_success)
            except Exception:
                step_success_value = 0.0
            self._running_ep_success[env_idx] = max(prev_success, step_success_value)
            if done:
                self._successes.append(float(self._running_ep_success.get(env_idx, 0.0)))
                self._running_ep_success[env_idx] = 0.0

        if (self.num_timesteps - self.last_ckpt_step) >= self.checkpoint_freq:
            self.last_ckpt_step = self.num_timesteps
            ckpt_path = self.ckpt_dir / f"step_{self.num_timesteps:012d}"
            self.model.save(str(ckpt_path))
            if self.verbose:
                print(f"[CKPT] t={self.num_timesteps:,} -> {ckpt_path.name}.zip")

        if (self.num_timesteps - self.last_health_step) >= self.health_freq:
            self.last_health_step = self.num_timesteps
            task_name = self._active_task_name()
            stats = self._collect_episode_stats()
            alpha = self._read_alpha()
            action_mean, action_std = self._action_stats()
            elapsed = time.time() - self.t0

            self.ts.append(self.num_timesteps)
            self.mean_returns.append(stats["mean_return"])
            self.success_rates.append(stats["success_rate"])
            self.mean_ep_lens.append(stats["mean_ep_len"])
            self.alphas.append(alpha)
            self.active_tasks.append(task_name)
            self.action_component_means.append(action_mean)
            self.action_component_stds.append(action_std)

            self._append_csv_row(
                {
                    "timesteps": int(self.num_timesteps),
                    "elapsed_sec": float(elapsed),
                    "active_task": task_name,
                    "mean_return": stats["mean_return"],
                    "success_rate": stats["success_rate"],
                    "mean_ep_len": stats["mean_ep_len"],
                    "alpha": alpha,
                    self.action_mean_key: action_mean,
                    self.action_std_key: action_std,
                }
            )

            self.logger.record(f"train/{self.action_mean_key}", action_mean)
            self.logger.record(f"train/{self.action_std_key}", action_std)

            print(
                f"[HEALTH] t={self.num_timesteps:,} task={task_name} "
                f"ret={stats['mean_return']:.2f} "
                f"succ={stats['success_rate']:.3f} "
                f"len={stats['mean_ep_len']:.1f} "
                f"{self.action_mean_key}={action_mean:.4f} "
                f"{self.action_std_key}={action_std:.4f}"
            )
            self._maybe_save_best(stats)

        if (self.num_timesteps - self.last_plot_step) >= self.plot_freq:
            self.last_plot_step = self.num_timesteps
            self._plot_metrics()
            self._plot_action_hist()

        return True


def _write_eval_records(csv_path: Path, rows: Iterable[EvalRecord]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "phase",
                "trained_task",
                "eval_task",
                "timesteps",
                "mean_return",
                "success_rate",
                "mean_ep_len",
            ],
        )
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "phase": r.phase,
                    "trained_task": r.trained_task,
                    "eval_task": r.eval_task,
                    "timesteps": r.timesteps,
                    "mean_return": r.mean_return,
                    "success_rate": r.success_rate,
                    "mean_ep_len": r.mean_ep_len,
                }
            )


def _export_tb_scalars(run_dir: Path, tags: Sequence[str]) -> Optional[Path]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        print(f"[WARN] TensorBoard export deshabilitado: {exc.__class__.__name__}")
        return None

    tb_dir = run_dir / "tb"
    event_files = sorted(tb_dir.glob("**/events.out.tfevents.*"))
    if not event_files:
        print(f"[WARN] No se encontraron eventos TensorBoard en {tb_dir}")
        return None

    latest_by_key: Dict[tuple[str, int], Dict[str, Any]] = {}
    available_tags: set[str] = set()
    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        accumulator.Reload()
        scalar_tags = set(accumulator.Tags().get("scalars", []))
        available_tags.update(scalar_tags)
        for tag in tags:
            if tag not in scalar_tags:
                continue
            for scalar_event in accumulator.Scalars(tag):
                row = {
                    "tag": tag,
                    "step": int(scalar_event.step),
                    "value": float(scalar_event.value),
                    "wall_time": float(scalar_event.wall_time),
                    "event_file": event_file.name,
                }
                key = (tag, row["step"])
                prev = latest_by_key.get(key)
                if prev is None or row["wall_time"] >= prev["wall_time"]:
                    latest_by_key[key] = row

    export_rows = sorted(
        latest_by_key.values(),
        key=lambda r: (str(r["tag"]), int(r["step"])),
    )
    if not export_rows:
        print("[WARN] No había escalares para exportar en los tags solicitados.")
        return None

    export_csv = run_dir / "tb_scalars_export.csv"
    with export_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["tag", "step", "value", "wall_time", "event_file"])
        writer.writeheader()
        writer.writerows(export_rows)

    missing = [tag for tag in tags if tag not in available_tags]
    if missing:
        print(f"[WARN] Tags TensorBoard faltantes: {missing}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    flat_axes = axes.flatten()
    for idx, tag in enumerate(tags[: len(flat_axes)]):
        ax = flat_axes[idx]
        points = [r for r in export_rows if r["tag"] == tag]
        if not points:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center")
            ax.set_title(tag)
            ax.set_xlabel("step")
            continue
        xs = [int(r["step"]) for r in points]
        ys = [float(r["value"]) for r in points]
        ax.plot(xs, ys, linewidth=1.2)
        ax.set_title(tag)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)

    for idx in range(len(tags), len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(run_dir / "tb_audit_metrics.png", dpi=130)
    plt.close(fig)
    print(f"[INFO] TensorBoard export: {export_csv}")
    return export_csv


def _compute_forgetting(eval_history: List[Dict[str, Dict[str, float]]], tasks: List[str]) -> float:
    """Promedio de Fi = max_t p_i(t) - p_i(final), sobre tareas únicas."""
    unique_tasks = list(dict.fromkeys(tasks))
    if len(eval_history) < 2:
        return float("nan")

    final = eval_history[-1]
    fis: List[float] = []
    for task in unique_tasks:
        hist: List[float] = []
        for phase_result in eval_history:
            score = phase_result.get(task, {}).get("success_rate", float("nan"))
            if not math.isnan(score):
                hist.append(score)
        if not hist:
            continue
        p_then = float(max(hist))
        p_final = final.get(task, {}).get("success_rate", float("nan"))
        if math.isnan(p_then) or math.isnan(p_final):
            continue
        fis.append(p_then - p_final)
    return float(np.mean(fis)) if fis else float("nan")


def _compute_diagonal_success(
    eval_history: List[Dict[str, Dict[str, float]]],
    tasks: List[str],
) -> float:
    """Promedio de éxito inmediatamente después de entrenar cada tarea."""
    diagonal: List[float] = []
    for phase_idx, trained_task in enumerate(tasks):
        if phase_idx >= len(eval_history):
            break
        score = eval_history[phase_idx].get(trained_task, {}).get(
            "success_rate", float("nan")
        )
        if not math.isnan(score):
            diagonal.append(float(score))
    return float(np.mean(diagonal)) if diagonal else float("nan")


def run_continual(args: argparse.Namespace, tasks: List[str], run_dir: Path) -> Dict[str, Any]:
    unique_tasks = list(dict.fromkeys(tasks))

    train_env = Monitor(
        ContinualTaskSuiteEnv(
            task_names=tasks,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            append_task_id=args.append_task_id,
            render_mode=None,
            disable_env_checker=args.disable_env_checker,
            obs_mode=args.obs_mode,
            normalize_obs=args.normalize_obs,
        ),
        filename=str(run_dir / "train.monitor.csv"),
        info_keywords=("task_name", "task_idx", "is_success"),
    )
    eval_env = Monitor(
        ContinualTaskSuiteEnv(
            task_names=tasks,
            seed=args.seed + 1_000,
            max_episode_steps=args.max_episode_steps,
            append_task_id=args.append_task_id,
            render_mode=None,
            disable_env_checker=args.disable_env_checker,
            obs_mode=args.obs_mode,
            normalize_obs=args.normalize_obs,
        ),
        filename=str(run_dir / "eval.monitor.csv"),
        info_keywords=("task_name", "task_idx", "is_success"),
    )

    if args.adapter_enabled and not args.append_task_id:
        raise RuntimeError("Con --adapter-enabled debes usar --append-task-id para routing por tarea.")

    model = _build_model(
        env=train_env,
        seed=args.seed,
        device=args.device,
        tensorboard_dir=run_dir / "tb",
        obs_mode=args.obs_mode,
        adapter_enabled=args.adapter_enabled,
        adapter_num_tasks=len(unique_tasks),
        adapter_rank=args.adapter_rank,
        adapter_alpha=args.adapter_alpha,
        adapter_features_dim=args.adapter_features_dim,
        adapter_backbone_hidden_dim=args.adapter_backbone_hidden_dim,
        ppo_learning_rate=args.ppo_learning_rate,
        ppo_n_steps=args.ppo_n_steps,
        ppo_batch_size=args.ppo_batch_size,
        ppo_n_epochs=args.ppo_n_epochs,
        ppo_gamma=args.ppo_gamma,
        ppo_gae_lambda=args.ppo_gae_lambda,
        ppo_clip_range=args.ppo_clip_range,
        ppo_ent_coef=args.ppo_ent_coef,
        ppo_vf_coef=args.ppo_vf_coef,
        ppo_max_grad_norm=args.ppo_max_grad_norm,
        ppo_target_kl=args.ppo_target_kl,
    )
    health_cb = HealthPlotCheckpointCallback(
        run_dir=run_dir,
        health_freq=args.health_freq,
        plot_freq=args.plot_freq,
        checkpoint_freq=args.checkpoint_freq,
        rolling_window=args.rolling_window,
        save_best_checkpoints=args.save_best_checkpoints,
        improvement_metric=args.improvement_metric,
        min_improvement_delta=args.min_improvement_delta,
        action_component_idx=args.action_component_idx,
        action_window_size=args.action_window_size,
        action_hist_bins=args.action_hist_bins,
        verbose=1,
    )

    if args.adapter_enabled:
        _install_multi_heads(model, num_tasks=len(unique_tasks))

    eval_history: List[Dict[str, Dict[str, float]]] = []
    total_steps = 0
    t0 = time.time()

    for phase, task in enumerate(tasks, start=1):
        print(
            f"\n{'=' * 90}\n"
            f"[CONTINUAL] fase {phase}/{len(tasks)} task={task} steps={args.steps_per_task:,}\n"
            f"{'=' * 90}"
        )
        _suite_env(train_env).set_task(task)
        task_idx = _suite_env(train_env)._task_idx(task)
        _set_active_heads(model, task_idx)
        if args.reset_metric_windows_every_task:
            health_cb.reset_task_window(model)

        if args.adapter_enabled:
            mode = _set_trainable_for_task(
                model=model,
                task_idx=task_idx,
                phase=phase,
                warmup_tasks=args.adapter_warmup_tasks,
                train_active_adapter_in_warmup=args.adapter_train_active_in_warmup,
                train_actor_heads_after_warmup=args.adapter_train_actor_heads_after_warmup,
                train_full_critic_after_warmup=args.adapter_train_full_critic_after_warmup,
            )
            print(
                f"[PEFT] phase={phase} task={task} task_idx={task_idx} mode={mode} "
                f"trainable={_count_trainable_params(model):,} "
                f"trainable_adapters={_count_trainable_adapter_params(model):,}"
            )

        if args.reset_optimizers_every_task:
            reset_optimizers = _reset_optimizers_for_task(model)
            if reset_optimizers:
                print(f"[OPT] phase={phase} reset={','.join(reset_optimizers)}")

        model.learn(
            total_timesteps=args.steps_per_task,
            reset_num_timesteps=False,
            progress_bar=args.progress_bar,
            callback=health_cb,
        )
        total_steps += int(args.steps_per_task)

        phase_result: Dict[str, Dict[str, float]] = {}
        records: List[EvalRecord] = []
        for eval_task in unique_tasks:
            eval_task_idx = _suite_env(eval_env)._task_idx(eval_task)
            _set_active_heads(model, eval_task_idx)
            stats = evaluate_task(
                model,
                eval_env,
                eval_task,
                args.eval_episodes,
                deterministic=args.eval_deterministic,
            )
            phase_result[eval_task] = stats
            records.append(
                EvalRecord(
                    phase=phase,
                    trained_task=task,
                    eval_task=eval_task,
                    timesteps=total_steps,
                    mean_return=stats["mean_return"],
                    success_rate=stats["success_rate"],
                    mean_ep_len=stats["mean_ep_len"],
                )
            )
            print(
                f"  eval={eval_task:<30} "
                f"return={stats['mean_return']:>9.2f} "
                f"success={stats['success_rate']:.3f}"
            )

        _write_eval_records(run_dir / "eval_metrics.csv", records)
        eval_history.append(phase_result)

        if args.save_model_each_phase:
            safe_task = task.replace("/", "_")
            model.save(str(run_dir / f"model_phase_{phase:02d}_{safe_task}"))

    elapsed = time.time() - t0
    model.save(str(run_dir / "model_final"))
    train_env.close()
    eval_env.close()

    final_successes = [
        eval_history[-1][t]["success_rate"] for t in unique_tasks if t in eval_history[-1]
    ]
    return {
        "mode": "continual",
        "algo": "ppo",
        "obs_mode": args.obs_mode,
        "eval_deterministic": args.eval_deterministic,
        "task_preset": args.task_preset,
        "tasks_sequence": tasks,
        "unique_tasks": unique_tasks,
        "adapter_enabled": args.adapter_enabled,
        "adapter_rank": args.adapter_rank if args.adapter_enabled else None,
        "adapter_alpha": args.adapter_alpha if args.adapter_enabled else None,
        "adapter_warmup_tasks": args.adapter_warmup_tasks if args.adapter_enabled else None,
        "adapter_train_actor_heads_after_warmup": (
            args.adapter_train_actor_heads_after_warmup if args.adapter_enabled else None
        ),
        "adapter_train_active_in_warmup": (
            args.adapter_train_active_in_warmup if args.adapter_enabled else None
        ),
        "adapter_train_full_critic_after_warmup": (
            args.adapter_train_full_critic_after_warmup if args.adapter_enabled else None
        ),
        "reset_optimizers_every_task": args.reset_optimizers_every_task,
        "reset_metric_windows_every_task": args.reset_metric_windows_every_task,
        "timesteps": total_steps,
        "elapsed_sec": elapsed,
        "avg_diagonal_success_rate": _compute_diagonal_success(eval_history, tasks),
        "avg_final_success_rate": float(np.mean(final_successes)) if final_successes else float("nan"),
        "avg_forgetting": _compute_forgetting(eval_history, tasks),
    }


def run_multitask(args: argparse.Namespace, tasks: List[str], run_dir: Path) -> Dict[str, Any]:
    train_env = Monitor(
        ContinualTaskSuiteEnv(
            task_names=tasks,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            append_task_id=args.append_task_id,
            render_mode=None,
            disable_env_checker=args.disable_env_checker,
            obs_mode=args.obs_mode,
            normalize_obs=args.normalize_obs,
        ),
        filename=str(run_dir / "train.monitor.csv"),
        info_keywords=("task_name", "task_idx", "is_success"),
    )
    _suite_env(train_env).set_task(None)

    eval_env = Monitor(
        ContinualTaskSuiteEnv(
            task_names=tasks,
            seed=args.seed + 1_000,
            max_episode_steps=args.max_episode_steps,
            append_task_id=args.append_task_id,
            render_mode=None,
            disable_env_checker=args.disable_env_checker,
            obs_mode=args.obs_mode,
            normalize_obs=args.normalize_obs,
        ),
        filename=str(run_dir / "eval.monitor.csv"),
        info_keywords=("task_name", "task_idx", "is_success"),
    )

    model = _build_model(
        env=train_env,
        seed=args.seed,
        device=args.device,
        tensorboard_dir=run_dir / "tb",
        obs_mode=args.obs_mode,
        adapter_enabled=args.adapter_enabled,
        adapter_num_tasks=len(list(dict.fromkeys(tasks))),
        adapter_rank=args.adapter_rank,
        adapter_alpha=args.adapter_alpha,
        adapter_features_dim=args.adapter_features_dim,
        adapter_backbone_hidden_dim=args.adapter_backbone_hidden_dim,
        ppo_learning_rate=args.ppo_learning_rate,
        ppo_n_steps=args.ppo_n_steps,
        ppo_batch_size=args.ppo_batch_size,
        ppo_n_epochs=args.ppo_n_epochs,
        ppo_gamma=args.ppo_gamma,
        ppo_gae_lambda=args.ppo_gae_lambda,
        ppo_clip_range=args.ppo_clip_range,
        ppo_ent_coef=args.ppo_ent_coef,
        ppo_vf_coef=args.ppo_vf_coef,
        ppo_max_grad_norm=args.ppo_max_grad_norm,
        ppo_target_kl=args.ppo_target_kl,
    )
    health_cb = HealthPlotCheckpointCallback(
        run_dir=run_dir,
        health_freq=args.health_freq,
        plot_freq=args.plot_freq,
        checkpoint_freq=args.checkpoint_freq,
        rolling_window=args.rolling_window,
        save_best_checkpoints=args.save_best_checkpoints,
        improvement_metric=args.improvement_metric,
        min_improvement_delta=args.min_improvement_delta,
        action_component_idx=args.action_component_idx,
        action_window_size=args.action_window_size,
        action_hist_bins=args.action_hist_bins,
        verbose=1,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=args.total_steps,
        progress_bar=args.progress_bar,
        callback=health_cb,
    )
    elapsed = time.time() - t0

    unique_tasks = list(dict.fromkeys(tasks))
    rows: List[EvalRecord] = []
    summary: Dict[str, Dict[str, float]] = {}
    for task in unique_tasks:
        task_idx = _suite_env(eval_env)._task_idx(task)
        _set_active_heads(model, task_idx)
        stats = evaluate_task(
            model,
            eval_env,
            task,
            args.eval_episodes,
            deterministic=args.eval_deterministic,
        )
        summary[task] = stats
        rows.append(
            EvalRecord(
                phase=1,
                trained_task="MULTITASK",
                eval_task=task,
                timesteps=args.total_steps,
                mean_return=stats["mean_return"],
                success_rate=stats["success_rate"],
                mean_ep_len=stats["mean_ep_len"],
            )
        )
        print(
            f"  eval={task:<30} "
            f"return={stats['mean_return']:>9.2f} "
            f"success={stats['success_rate']:.3f}"
        )
    _write_eval_records(run_dir / "eval_metrics.csv", rows)

    model.save(str(run_dir / "model_final"))
    train_env.close()
    eval_env.close()

    success_rates = [v["success_rate"] for v in summary.values()]
    return {
        "mode": "multitask",
        "algo": "ppo",
        "obs_mode": args.obs_mode,
        "eval_deterministic": args.eval_deterministic,
        "task_preset": args.task_preset,
        "tasks_sequence": tasks,
        "unique_tasks": unique_tasks,
        "adapter_enabled": args.adapter_enabled,
        "timesteps": args.total_steps,
        "elapsed_sec": elapsed,
        "avg_success_rate": float(np.mean(success_rates)) if success_rates else float("nan"),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("MiniGrid Continual RL runner (adapters + PPO)")
    p.add_argument("--mode", choices=["continual", "multitask"], default="continual")
    p.add_argument("--algo", choices=["ppo"], default="ppo")
    p.add_argument("--task-preset", default="smoke4")
    p.add_argument("--tasks", type=str, default=None, help="CSV de env IDs (sobrescribe preset)")
    p.add_argument("--steps-per-task", type=int, default=300_000)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--obs-mode", choices=["image", "flat"], default="image")
    p.add_argument(
        "--eval-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluación greedy (determinística). Desactiva para evaluación estocástica.",
    )
    p.add_argument(
        "--normalize-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalizar observaciones Box si tienen escala > 1.",
    )
    p.add_argument("--append-task-id", action="store_true")

    p.add_argument(
        "--adapter-enabled",
        dest="adapter_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activar backbone compartido + adaptadores low-rank por tarea.",
    )
    p.add_argument(
        "--lora-enabled",
        dest="adapter_enabled",
        action=argparse.BooleanOptionalAction,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--adapter-rank", dest="adapter_rank", type=int, default=32)
    p.add_argument("--lora-rank", dest="adapter_rank", type=int, help=argparse.SUPPRESS)
    p.add_argument("--adapter-alpha", dest="adapter_alpha", type=float, default=32.0)
    p.add_argument("--lora-alpha", dest="adapter_alpha", type=float, help=argparse.SUPPRESS)
    p.add_argument("--adapter-features-dim", dest="adapter_features_dim", type=int, default=256)
    p.add_argument(
        "--lora-features-dim",
        dest="adapter_features_dim",
        type=int,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--adapter-backbone-hidden-dim", dest="adapter_backbone_hidden_dim", type=int, default=256
    )
    p.add_argument(
        "--lora-backbone-hidden-dim",
        dest="adapter_backbone_hidden_dim",
        type=int,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--adapter-warmup-tasks",
        dest="adapter_warmup_tasks",
        type=int,
        default=1,
        help="Número de tareas iniciales para construir base compartida antes del régimen PEFT.",
    )
    p.add_argument(
        "--lora-warmup-tasks",
        dest="adapter_warmup_tasks",
        type=int,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--adapter-train-active-in-warmup",
        dest="adapter_train_active_in_warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Durante warm-up, entrenar también el adaptador activo. "
            "Recomendado False para construir primero un backbone base."
        ),
    )
    p.add_argument(
        "--lora-train-active-in-warmup",
        dest="adapter_train_active_in_warmup",
        action=argparse.BooleanOptionalAction,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--adapter-train-actor-heads-after-warmup",
        dest="adapter_train_actor_heads_after_warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tras warm-up, entrenar head/policy del actor además del adaptador activo.",
    )
    p.add_argument(
        "--lora-train-heads-after-warmup",
        dest="adapter_train_actor_heads_after_warmup",
        action=argparse.BooleanOptionalAction,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--adapter-train-full-critic-after-warmup",
        dest="adapter_train_full_critic_after_warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tras warm-up, entrenar troncal de valor compartida (menos aislamiento CL).",
    )

    p.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    p.add_argument("--ppo-n-steps", type=int, default=2048)
    p.add_argument("--ppo-batch-size", type=int, default=256)
    p.add_argument("--ppo-n-epochs", type=int, default=10)
    p.add_argument("--ppo-gamma", type=float, default=0.99)
    p.add_argument("--ppo-gae-lambda", type=float, default=0.95)
    p.add_argument("--ppo-clip-range", type=float, default=0.2)
    p.add_argument("--ppo-ent-coef", type=float, default=0.001)
    p.add_argument("--ppo-vf-coef", type=float, default=0.5)
    p.add_argument("--ppo-max-grad-norm", type=float, default=0.5)
    p.add_argument(
        "--ppo-target-kl",
        type=float,
        default=0.03,
        help="Early stop por KL objetivo en PPO (None/0 para desactivar).",
    )

    p.add_argument(
        "--reset-optimizers-every-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset de estados de optimizador al cambiar de tarea.",
    )
    p.add_argument(
        "--reset-metric-windows-every-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Limpiar ventanas rolling de métricas al inicio de cada tarea.",
    )
    p.add_argument("--health-freq", type=int, default=5_000)
    p.add_argument("--plot-freq", type=int, default=20_000)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--rolling-window", type=int, default=100)
    p.add_argument(
        "--save-best-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Guardar checkpoint adicional cuando mejora la métrica elegida.",
    )
    p.add_argument(
        "--improvement-metric",
        choices=["success", "return"],
        default="success",
        help="Métrica para decidir mejora del best checkpoint.",
    )
    p.add_argument("--min-improvement-delta", type=float, default=1e-6)
    p.add_argument("--action-component-idx", type=int, default=0)
    p.add_argument("--action-window-size", type=int, default=50_000)
    p.add_argument("--action-hist-bins", type=int, default=31)
    p.add_argument("--disable-env-checker", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-model-each-phase", action="store_true")
    p.add_argument("--progress-bar", action="store_true")
    p.add_argument(
        "--log-dir",
        default="logs/minigrid_cw",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _ensure_minigrid()

    if args.adapter_enabled and not args.append_task_id:
        raise ValueError("Con --adapter-enabled debes activar --append-task-id para routing por tarea.")
    if args.adapter_rank <= 0:
        raise ValueError("--adapter-rank debe ser > 0.")
    if args.adapter_warmup_tasks <= 0:
        raise ValueError("--adapter-warmup-tasks debe ser >= 1.")
    if args.ppo_n_steps <= 0:
        raise ValueError("--ppo-n-steps debe ser > 0.")
    if args.ppo_batch_size <= 0:
        raise ValueError("--ppo-batch-size debe ser > 0.")
    if args.ppo_n_epochs <= 0:
        raise ValueError("--ppo-n-epochs debe ser > 0.")
    if args.max_episode_steps <= 0:
        raise ValueError("--max-episode-steps debe ser > 0.")
    if args.action_window_size <= 0:
        raise ValueError("--action-window-size debe ser > 0.")
    if args.ppo_target_kl < 0:
        raise ValueError("--ppo-target-kl no puede ser negativo.")
    if args.ppo_target_kl == 0:
        args.ppo_target_kl = None

    if args.ppo_batch_size > args.ppo_n_steps:
        print(
            "[WARN] --ppo-batch-size es mayor que --ppo-n-steps en n_env=1; "
            "SB3 lo maneja pero puede degradar entrenamiento."
        )

    _set_seed(args.seed)

    tasks = resolve_task_sequence(args.task_preset, args.tasks)
    run_name = (
        f"{args.mode}_{args.algo}_minigrid_{args.task_preset}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}_n{len(tasks)}"
    )
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    versions = _collect_runtime_versions()
    versions_path = _write_runtime_versions(run_dir, versions)

    (run_dir / "config.json").write_text(
        json.dumps({"args": vars(args), "tasks": tasks, "versions": versions}, indent=2),
        encoding="utf-8",
    )

    print(f"Run dir: {run_dir}")
    print(f"Mode={args.mode} Algo={args.algo} Benchmark=minigrid Tasks={len(tasks)}")
    print(f"Versions file: {versions_path}")
    print(f"Task sequence: {tasks}\n")

    if args.mode == "continual":
        summary = run_continual(args, tasks, run_dir)
    else:
        summary = run_multitask(args, tasks, run_dir)

    tb_export = _export_tb_scalars(run_dir, AUDIT_TB_TAGS)
    tb_plot = run_dir / "tb_audit_metrics.png"
    summary["versions"] = versions
    summary["versions_file"] = str(versions_path)
    summary["tb_audit_export_csv"] = str(tb_export) if tb_export is not None else None
    summary["tb_audit_plot"] = str(tb_plot) if tb_plot.exists() else None

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nArtefactos en: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
