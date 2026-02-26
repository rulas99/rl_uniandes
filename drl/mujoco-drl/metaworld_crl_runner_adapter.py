from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import gymnasium as gym
import matplotlib
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# CW10 del paper Continual World (nombres v3 en Meta-World moderno).
CW10_TASKS: List[str] = [
    "hammer-v3",
    "push-wall-v3",
    "faucet-close-v3",
    "push-back-v3",
    "stick-pull-v3",
    "handle-press-side-v3",
    "push-v3",
    "shelf-place-v3",
    "window-close-v3",
    "peg-unplug-side-v3",
]


def _ensure_metaworld() -> None:
    try:
        import metaworld  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "No se encontró `metaworld`.\n"
            "Instala con:\n"
            "  ./.venv/bin/python -m pip install metaworld"
        ) from exc


def _all_v3_tasks() -> List[str]:
    _ensure_metaworld()
    import metaworld

    return sorted(
        name for name in metaworld.ALL_V3_ENVIRONMENTS.keys() if name.endswith("-v3")
    )


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


def resolve_task_sequence(task_preset: str, tasks_csv: Optional[str]) -> List[str]:
    if tasks_csv:
        tasks = _parse_csv_tasks(tasks_csv)
        if not tasks:
            raise ValueError("`--tasks` quedó vacío.")
        return tasks

    preset = task_preset.lower().strip()
    if preset == "cw10":
        return CW10_TASKS.copy()
    if preset == "cw20":
        return CW10_TASKS + CW10_TASKS
    if preset == "mt10":
        # MT10 clásico de Meta-World+ docs.
        return [
            "reach-v3",
            "push-v3",
            "pick-place-v3",
            "door-open-v3",
            "drawer-open-v3",
            "drawer-close-v3",
            "button-press-topdown-v3",
            "peg-insert-side-v3",
            "window-open-v3",
            "window-close-v3",
        ]
    if preset == "mt50":
        tasks = _all_v3_tasks()
        if len(tasks) < 50:
            raise RuntimeError(f"Solo se detectaron {len(tasks)} tareas v3.")
        return tasks[:50]
    if preset == "smoke2":
        return ["reach-v3","push-v3"]

    raise ValueError("`--task-preset` inválido. Usa cw10/cw20/mt10/mt50/smoke2.")


class RelaxedObsBounds(gym.ObservationWrapper):
    """Arregla casos de Meta-World donde low==high pero obs varía.

    Gymnasium env_checker marca warning aunque la env sea válida en práctica.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Box):
            self.observation_space = obs_space
            return

        low = np.array(obs_space.low, copy=True)
        high = np.array(obs_space.high, copy=True)
        same = low == high
        if np.any(same):
            # Hacemos esos ejes no acotados para que contains(obs) sea consistente.
            low = low.astype(np.float64, copy=False)
            high = high.astype(np.float64, copy=False)
            low[same] = -np.inf
            high[same] = np.inf
            self.observation_space = spaces.Box(low=low, high=high, dtype=obs_space.dtype)
        else:
            self.observation_space = obs_space

    def observation(self, observation: Any) -> Any:
        return np.asarray(observation, dtype=self.observation_space.dtype)


def _make_mt1_env(
    env_name: str,
    seed: int,
    max_episode_steps: int,
    render_mode: Optional[str],
    disable_env_checker: bool,
    relax_obs_bounds: bool,
    reward_func_version: Optional[str],
) -> gym.Env:
    _ensure_metaworld()
    make_kwargs: Dict[str, Any] = dict(
        env_name=env_name,
        seed=seed,
        render_mode=render_mode,
        disable_env_checker=disable_env_checker,
    )
    env: gym.Env
    if reward_func_version:
        try:
            env = gym.make(
                "Meta-World/MT1",
                reward_func_version=reward_func_version,
                **make_kwargs,
            )
        except TypeError as exc:
            if "reward_func_version" not in str(exc):
                raise
            try:
                env = gym.make(
                    "Meta-World/MT1",
                    reward_function_version=reward_func_version,
                    **make_kwargs,
                )
                print(
                    "[WARN] MT1 wrapper usa reward_function_version (legacy); "
                    "aplicando compatibilidad."
                )
            except TypeError as legacy_exc:
                if "reward_function_version" not in str(legacy_exc):
                    raise
                print(
                    "[WARN] MT1 wrapper no soporta reward_func_version "
                    "ni reward_function_version; usando reward por defecto."
                )
                env = gym.make("Meta-World/MT1", **make_kwargs)
    else:
        env = gym.make("Meta-World/MT1", **make_kwargs)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if relax_obs_bounds:
        env = RelaxedObsBounds(env)
    return env


class ContinualTaskSuiteEnv(gym.Env[np.ndarray, np.ndarray]):
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
        relax_obs_bounds: bool = True,
        reward_func_version: Optional[str] = "v2",
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
            self._envs[task] = _make_mt1_env(
                env_name=task,
                seed=seed + i,
                max_episode_steps=max_episode_steps,
                render_mode=render_mode,
                disable_env_checker=disable_env_checker,
                relax_obs_bounds=relax_obs_bounds,
                reward_func_version=reward_func_version,
            )

        base_env = self._envs[self.unique_tasks[0]]
        self.action_space = base_env.action_space
        if not isinstance(base_env.observation_space, spaces.Box):
            raise TypeError("Se esperaba observation_space Box.")

        base_space = base_env.observation_space
        low = np.array(base_space.low, dtype=np.float64, copy=True)
        high = np.array(base_space.high, dtype=np.float64, copy=True)
        self._num_tasks = len(self.unique_tasks)

        if append_task_id:
            low = np.concatenate([low, np.zeros(self._num_tasks, dtype=low.dtype)])
            high = np.concatenate([high, np.ones(self._num_tasks, dtype=high.dtype)])
            self.observation_space = spaces.Box(low=low, high=high, dtype=base_space.dtype)
        else:
            self.observation_space = spaces.Box(low=low, high=high, dtype=base_space.dtype)

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
        one_hot = np.zeros(self._num_tasks, dtype=self.observation_space.dtype)
        one_hot[self._task_idx(task)] = 1.0
        return one_hot

    def _augment_obs(self, obs: np.ndarray, task: str) -> np.ndarray:
        obs = np.asarray(obs, dtype=self.observation_space.dtype)
        if not self.append_task_id:
            return obs
        return np.concatenate([obs, self._task_one_hot(task)], axis=0).astype(
            self.observation_space.dtype
        )

    @staticmethod
    def _augment_info(info: Dict[str, Any], task: str, task_idx: int) -> Dict[str, Any]:
        out = dict(info)
        out["task_name"] = task
        out["task_idx"] = task_idx
        success = out.get("success", out.get("is_success", 0.0))
        try:
            out["is_success"] = float(success)
        except Exception:
            out["is_success"] = 0.0
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
        out_info = self._augment_info(info, task, idx)
        out_info["is_success"] = self._ep_success
        return self._augment_obs(obs, task), out_info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._current_task is None:
            raise RuntimeError("Se llamó step() antes de reset().")
        env = self._envs[self._current_task]
        obs, reward, terminated, truncated, info = env.step(action)
        idx = self._task_idx(self._current_task)
        out_info = self._augment_info(info, self._current_task, idx)
        try:
            self._ep_success = max(self._ep_success, float(out_info.get("is_success", 0.0)))
        except Exception:
            pass
        out_info["is_success"] = float(self._ep_success)
        return (
            self._augment_obs(obs, self._current_task),
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


class RoutedAdapterBackboneExtractor(BaseFeaturesExtractor):
    """Backbone compartido + un adaptador por tarea ruteado con one-hot."""

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
        task_indices = th.argmax(one_hot, dim=-1)
        flat_task_indices = task_indices.reshape(-1)
        if flat_task_indices.numel() == 0:
            return h

        # Camino rápido: batch mono-tarea (entrenamiento continual estándar).
        if bool(th.all(flat_task_indices == flat_task_indices[0]).item()):
            idx = int(flat_task_indices[0].item())
            return h + self.adapters[idx](h)

        # Camino general: batch mixto de tareas (p.ej. evaluación multitarea).
        flat_h = h.reshape(-1, h.shape[-1])
        flat_delta = th.zeros_like(flat_h)
        for idx, adapter in enumerate(self.adapters):
            mask = flat_task_indices == idx
            if not bool(mask.any().item()):
                continue
            flat_delta[mask] = adapter(flat_h[mask])
        return h + flat_delta.view_as(h)


class BackboneOnlyExtractor(BaseFeaturesExtractor):
    """Mismo backbone base sin adaptadores (baseline de capacidad equivalente)."""

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


def _iter_routed_extractors(policy: Any) -> List[RoutedAdapterBackboneExtractor]:
    candidates = [getattr(policy, "features_extractor", None)]

    actor = getattr(policy, "actor", None)
    if actor is not None:
        candidates.append(getattr(actor, "features_extractor", None))

    critic = getattr(policy, "critic", None)
    if critic is not None:
        candidates.append(getattr(critic, "features_extractor", None))

    extractors: List[RoutedAdapterBackboneExtractor] = []
    seen: set[int] = set()
    for maybe_extractor in candidates:
        if not isinstance(maybe_extractor, RoutedAdapterBackboneExtractor):
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


def _set_trainable_for_task(
    model: BaseAlgorithm,
    task_idx: int,
    phase: int,
    warmup_tasks: int,
    train_actor_heads_after_warmup: bool,
    train_full_critic_after_warmup: bool,
) -> str:
    policy = model.policy

    if phase <= warmup_tasks:
        for name, p in policy.named_parameters():
            if name.startswith("critic_target."):
                p.requires_grad = False
                continue
            adapter_idx = _adapter_task_idx_from_param_name(name)
            if adapter_idx is None:
                p.requires_grad = True
            else:
                p.requires_grad = adapter_idx == task_idx
        return "warmup_backbone_heads+active_adapter"

    for p in policy.parameters():
        p.requires_grad = False

    extractors = _iter_routed_extractors(policy)
    if not extractors:
        raise TypeError(
            "No se detectó RoutedAdapterBackboneExtractor en la policy. "
            "Revisa --adapter-enabled y policy_kwargs."
        )

    actor = getattr(policy, "actor", None)
    actor_extractor = getattr(actor, "features_extractor", None)
    if isinstance(actor_extractor, RoutedAdapterBackboneExtractor):
        actor_extractor.unfreeze_adapter(task_idx)
    else:
        # Fallback genérico (PPO u otras policies sin actor separado).
        for extractor in extractors:
            extractor.unfreeze_adapter(task_idx)

    mode_parts = ["actor_adapter"]

    # Crítico para adaptación motora en nuevas tareas: siempre entrenable.
    if actor is not None and hasattr(actor, "mu"):
        for p in actor.mu.parameters():
            p.requires_grad = True
        mode_parts.append("actor_mu")
    if actor is not None and hasattr(actor, "log_std"):
        for p in actor.log_std.parameters():
            p.requires_grad = True
        mode_parts.append("actor_log_std")

    if train_actor_heads_after_warmup:
        if actor is not None:
            for name, p in actor.named_parameters():
                if "features_extractor." in name:
                    continue
                if name.startswith("mu.") or name.startswith("log_std."):
                    continue
                p.requires_grad = True
        else:
            for name, p in policy.named_parameters():
                if name.startswith("critic_target."):
                    continue
                if "features_extractor." in name:
                    continue
                p.requires_grad = True
        mode_parts.append("actor_heads_extra")

    critic = getattr(policy, "critic", None)
    if critic is not None:
        if train_full_critic_after_warmup:
            for p in critic.parameters():
                p.requires_grad = True
            mode_parts.append("critic_full")
        else:
            critic_extractor = getattr(critic, "features_extractor", None)
            if isinstance(critic_extractor, RoutedAdapterBackboneExtractor):
                critic_extractor.unfreeze_adapter(task_idx)
                mode_parts.append("critic_adapter")

    return "+".join(mode_parts)


def _count_trainable_params(model: BaseAlgorithm) -> int:
    return sum(
        p.numel()
        for name, p in model.policy.named_parameters()
        if p.requires_grad and not name.startswith("critic_target.")
    )


def _count_trainable_adapter_params(model: BaseAlgorithm) -> int:
    trainable = 0
    for extractor in _iter_routed_extractors(model.policy):
        for p in extractor.adapters.parameters():
            if p.requires_grad:
                trainable += p.numel()
    return trainable


def _clone_optimizer(optimizer: th.optim.Optimizer, params: Iterable[th.nn.Parameter]) -> th.optim.Optimizer:
    cls = optimizer.__class__
    kwargs = dict(optimizer.defaults)
    if optimizer.param_groups:
        kwargs["lr"] = float(optimizer.param_groups[0].get("lr", kwargs.get("lr", 0.0)))
    return cls(params, **kwargs)


def _reset_sac_optimizers(
    model: BaseAlgorithm,
    reset_ent_coef: bool,
    ent_coef_log_reset_value: float,
) -> List[str]:
    reset_names: List[str] = []
    policy = model.policy

    actor = getattr(policy, "actor", None)
    actor_opt = getattr(actor, "optimizer", None)
    if actor is not None and actor_opt is not None:
        actor.optimizer = _clone_optimizer(actor_opt, actor.parameters())
        reset_names.append("actor")

    critic = getattr(policy, "critic", None)
    critic_opt = getattr(critic, "optimizer", None)
    if critic is not None and critic_opt is not None:
        critic.optimizer = _clone_optimizer(critic_opt, critic.parameters())
        reset_names.append("critic")

    ent_opt = getattr(model, "ent_coef_optimizer", None)
    log_ent_coef = getattr(model, "log_ent_coef", None)
    if ent_opt is not None and log_ent_coef is not None:
        if reset_ent_coef:
            with th.no_grad():
                log_ent_coef.fill_(float(ent_coef_log_reset_value))
        model.ent_coef_optimizer = _clone_optimizer(ent_opt, [log_ent_coef])
        if reset_ent_coef:
            reset_names.append("ent_coef(reset)")
        else:
            reset_names.append("ent_coef(opt)")

    return reset_names


def _reset_optimizers_for_task(
    model: BaseAlgorithm,
    reset_ent_coef: bool,
    ent_coef_log_reset_value: float,
) -> List[str]:
    # Protocolo CW relevante para SAC.
    if isinstance(model, SAC):
        return _reset_sac_optimizers(model, reset_ent_coef, ent_coef_log_reset_value)

    reset_names: List[str] = []
    policy_opt = getattr(model.policy, "optimizer", None)
    if policy_opt is not None:
        model.policy.optimizer = _clone_optimizer(policy_opt, model.policy.parameters())
        reset_names.append("policy")
    return reset_names


def _configure_sac_phase_random_warmup(
    model: BaseAlgorithm, random_steps_per_task: int
) -> Optional[int]:
    if not isinstance(model, SAC):
        return None
    steps = max(0, int(random_steps_per_task))
    model.learning_starts = int(model.num_timesteps + steps)
    return int(model.learning_starts)


def _build_model(
    algo: str,
    env: gym.Env,
    seed: int,
    device: str,
    tensorboard_dir: Path,
    adapter_enabled: bool,
    adapter_num_tasks: int,
    adapter_rank: int,
    adapter_alpha: float,
    adapter_features_dim: int,
    adapter_backbone_hidden_dim: int,
    append_task_id: bool,
    sac_learning_rate: float,
    sac_batch_size: int,
    sac_gradient_steps: int,
    sac_buffer_size: int,
    sac_learning_starts: int,
) -> BaseAlgorithm:
    algo = algo.lower()
    policy_kwargs: Dict[str, Any] = dict(net_arch=[256, 256])

    if adapter_enabled:
        policy_kwargs.update(
            dict(
                features_extractor_class=RoutedAdapterBackboneExtractor,
                features_extractor_kwargs=dict(
                    features_dim=adapter_features_dim,
                    num_tasks=adapter_num_tasks,
                    adapter_rank=adapter_rank,
                    adapter_alpha=adapter_alpha,
                    backbone_hidden_dim=adapter_backbone_hidden_dim,
                ),
            )
        )
    else:
        policy_kwargs.update(
            dict(
                features_extractor_class=BackboneOnlyExtractor,
                features_extractor_kwargs=dict(
                    features_dim=adapter_features_dim,
                    num_tasks=adapter_num_tasks if append_task_id else 0,
                    backbone_hidden_dim=adapter_backbone_hidden_dim,
                ),
            )
        )

    if algo == "sac":
        if adapter_enabled:
            # En SAC/SB3 mantenemos extractores separados: evita gradientes ambiguos.
            policy_kwargs["share_features_extractor"] = False
        return SAC(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            tensorboard_log=str(tensorboard_dir),
            device=device,
            learning_rate=sac_learning_rate,
            batch_size=sac_batch_size,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=sac_gradient_steps,
            buffer_size=sac_buffer_size,
            learning_starts=sac_learning_starts,
            ent_coef="auto_0.1",
            target_entropy="auto",
            policy_kwargs=policy_kwargs,
        )
    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            tensorboard_log=str(tensorboard_dir),
            device=device,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs=policy_kwargs,
        )
    raise ValueError(f"Algoritmo no soportado: {algo}")


def evaluate_task(
    model: BaseAlgorithm, env: gym.Env, task_name: str, episodes: int
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
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
            ep_len += 1
            step_success = info.get("is_success", info.get("success", 0.0))
            try:
                if float(step_success) > 0.5:
                    ep_success = 1.0
            except Exception:
                pass
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

        self._successes: List[float] = []
        self._running_ep_success: Dict[int, float] = {}
        self.best_score = -float("inf")
        self.t0 = time.time()

    def _on_training_start(self) -> None:
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "timesteps",
                        "elapsed_sec",
                        "active_task",
                        "mean_return",
                        "success_rate",
                        "mean_ep_len",
                        "alpha",
                    ],
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
        # Compatible con distintas variantes de SB3 SAC.
        try:
            import torch as th

            log_ent_coef = getattr(self.model, "log_ent_coef", None)
            if log_ent_coef is not None:
                if isinstance(log_ent_coef, th.Tensor):
                    return float(th.exp(log_ent_coef.detach()).mean().cpu().item())
                return float(np.exp(float(log_ent_coef)))
            ent_coef_tensor = getattr(self.model, "ent_coef_tensor", None)
            if ent_coef_tensor is not None:
                if isinstance(ent_coef_tensor, th.Tensor):
                    return float(ent_coef_tensor.detach().mean().cpu().item())
                return float(ent_coef_tensor)
            ent_coef = getattr(self.model, "ent_coef", None)
            if isinstance(ent_coef, (int, float)):
                return float(ent_coef)
        except Exception:
            pass
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
                fieldnames=[
                    "timesteps",
                    "elapsed_sec",
                    "active_task",
                    "mean_return",
                    "success_rate",
                    "mean_ep_len",
                    "alpha",
                ],
            )
            writer.writerow(row)

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

        # Checkpoint periódico
        if (self.num_timesteps - self.last_ckpt_step) >= self.checkpoint_freq:
            self.last_ckpt_step = self.num_timesteps
            ckpt_path = self.ckpt_dir / f"step_{self.num_timesteps:012d}"
            self.model.save(str(ckpt_path))
            if self.verbose:
                print(f"[CKPT] t={self.num_timesteps:,} -> {ckpt_path.name}.zip")

        # Health + CSV
        if (self.num_timesteps - self.last_health_step) >= self.health_freq:
            self.last_health_step = self.num_timesteps
            task_name = self._active_task_name()
            stats = self._collect_episode_stats()
            alpha = self._read_alpha()
            elapsed = time.time() - self.t0

            self.ts.append(self.num_timesteps)
            self.mean_returns.append(stats["mean_return"])
            self.success_rates.append(stats["success_rate"])
            self.mean_ep_lens.append(stats["mean_ep_len"])
            self.alphas.append(alpha)
            self.active_tasks.append(task_name)

            self._append_csv_row(
                {
                    "timesteps": int(self.num_timesteps),
                    "elapsed_sec": float(elapsed),
                    "active_task": task_name,
                    "mean_return": stats["mean_return"],
                    "success_rate": stats["success_rate"],
                    "mean_ep_len": stats["mean_ep_len"],
                    "alpha": alpha,
                }
            )

            pct = self.num_timesteps
            print(
                f"[HEALTH] t={pct:,} task={task_name} "
                f"ret={stats['mean_return']:.2f} "
                f"succ={stats['success_rate']:.3f} "
                f"len={stats['mean_ep_len']:.1f} "
                f"alpha={alpha:.5f}"
            )
            self._maybe_save_best(stats)

        # Plot periódico
        if (self.num_timesteps - self.last_plot_step) >= self.plot_freq:
            self.last_plot_step = self.num_timesteps
            self._plot_metrics()

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


def _compute_forgetting(eval_history: List[Dict[str, Dict[str, float]]], tasks: List[str]) -> float:
    """Promedio de Fi = p_i(tras entrenar i) - p_i(final), sobre tareas únicas."""
    unique_tasks = list(dict.fromkeys(tasks))
    if len(eval_history) < 2:
        return float("nan")

    first_seen_idx: Dict[str, int] = {}
    for phase_idx, trained_task in enumerate(tasks):
        if trained_task not in first_seen_idx:
            first_seen_idx[trained_task] = phase_idx

    final = eval_history[-1]
    fis: List[float] = []
    for task in unique_tasks:
        idx = first_seen_idx[task]
        if idx >= len(eval_history):
            continue
        p_then = eval_history[idx].get(task, {}).get("success_rate", float("nan"))
        p_final = final.get(task, {}).get("success_rate", float("nan"))
        if math.isnan(p_then) or math.isnan(p_final):
            continue
        fis.append(p_then - p_final)
    return float(np.mean(fis)) if fis else float("nan")


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
            relax_obs_bounds=args.relax_obs_bounds,
            reward_func_version=args.reward_func_version,
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
            relax_obs_bounds=args.relax_obs_bounds,
            reward_func_version=args.reward_func_version,
        ),
        filename=str(run_dir / "eval.monitor.csv"),
        info_keywords=("task_name", "task_idx", "is_success"),
    )

    if args.adapter_enabled and not args.append_task_id:
        raise RuntimeError("Con --adapter-enabled debes usar --append-task-id para routing por tarea.")

    model = _build_model(
        algo=args.algo,
        env=train_env,
        seed=args.seed,
        device=args.device,
        tensorboard_dir=run_dir / "tb",
        adapter_enabled=args.adapter_enabled,
        adapter_num_tasks=len(unique_tasks),
        adapter_rank=args.adapter_rank,
        adapter_alpha=args.adapter_alpha,
        adapter_features_dim=args.adapter_features_dim,
        adapter_backbone_hidden_dim=args.adapter_backbone_hidden_dim,
        append_task_id=args.append_task_id,
        sac_learning_rate=args.sac_learning_rate,
        sac_batch_size=args.sac_batch_size,
        sac_gradient_steps=args.sac_gradient_steps,
        sac_buffer_size=args.sac_buffer_size,
        sac_learning_starts=args.sac_learning_starts,
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
        verbose=1,
    )

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
        if args.reset_metric_windows_every_task:
            health_cb.reset_task_window(model)

        if args.adapter_enabled:
            mode = _set_trainable_for_task(
                model=model,
                task_idx=task_idx,
                phase=phase,
                warmup_tasks=args.adapter_warmup_tasks,
                train_actor_heads_after_warmup=args.adapter_train_actor_heads_after_warmup,
                train_full_critic_after_warmup=args.adapter_train_full_critic_after_warmup,
            )
            print(
                f"[PEFT] phase={phase} task={task} task_idx={task_idx} mode={mode} "
                f"trainable={_count_trainable_params(model):,} "
                f"trainable_adapters={_count_trainable_adapter_params(model):,}"
            )

        if args.reset_optimizers_every_task:
            reset_optimizers = _reset_optimizers_for_task(
                model=model,
                reset_ent_coef=args.reset_ent_coef_every_task,
                ent_coef_log_reset_value=args.ent_coef_reset_log_value,
            )
            if reset_optimizers:
                print(
                    f"[OPT] phase={phase} reset={','.join(reset_optimizers)}"
                )

        warmup_until = _configure_sac_phase_random_warmup(
            model=model,
            random_steps_per_task=args.sac_random_steps_per_task,
        )
        if warmup_until is not None:
            warmup_from = max(0, warmup_until - max(0, int(args.sac_random_steps_per_task)))
            print(
                f"[SAC] phase={phase} random_warmup_steps={args.sac_random_steps_per_task} "
                f"learning_starts={warmup_from}->{warmup_until}"
            )

        model.learn(
            total_timesteps=args.steps_per_task,
            reset_num_timesteps=False,
            progress_bar=args.progress_bar,
            callback=health_cb,
        )
        total_steps += int(args.steps_per_task)

        if args.reset_replay_every_task and hasattr(model, "replay_buffer"):
            rb = getattr(model, "replay_buffer")
            if rb is not None and hasattr(rb, "reset"):
                rb.reset()

        phase_result: Dict[str, Dict[str, float]] = {}
        records: List[EvalRecord] = []
        for eval_task in unique_tasks:
            stats = evaluate_task(model, eval_env, eval_task, args.eval_episodes)
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
                f"  eval={eval_task:<24} "
                f"return={stats['mean_return']:>9.2f} "
                f"success={stats['success_rate']:.3f}"
            )

        _write_eval_records(run_dir / "eval_metrics.csv", records)
        eval_history.append(phase_result)

        if args.save_model_each_phase:
            model.save(str(run_dir / f"model_phase_{phase:02d}_{task}"))

    elapsed = time.time() - t0
    model.save(str(run_dir / "model_final"))
    train_env.close()
    eval_env.close()

    final_successes = [
        eval_history[-1][t]["success_rate"] for t in unique_tasks if t in eval_history[-1]
    ]
    return {
        "mode": "continual",
        "algo": args.algo,
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
        "adapter_train_full_critic_after_warmup": (
            args.adapter_train_full_critic_after_warmup if args.adapter_enabled else None
        ),
        "reward_func_version": args.reward_func_version,
        "reset_optimizers_every_task": args.reset_optimizers_every_task,
        "reset_ent_coef_every_task": args.reset_ent_coef_every_task,
        "ent_coef_reset_log_value": args.ent_coef_reset_log_value if args.algo == "sac" else None,
        "reset_metric_windows_every_task": args.reset_metric_windows_every_task,
        "sac_random_steps_per_task": args.sac_random_steps_per_task if args.algo == "sac" else None,
        "timesteps": total_steps,
        "elapsed_sec": elapsed,
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
            relax_obs_bounds=args.relax_obs_bounds,
            reward_func_version=args.reward_func_version,
        ),
        filename=str(run_dir / "train.monitor.csv"),
        info_keywords=("task_name", "task_idx", "is_success"),
    )
    # `set_task(None)` => muestreo aleatorio por episodio.
    _suite_env(train_env).set_task(None)

    eval_env = Monitor(
        ContinualTaskSuiteEnv(
            task_names=tasks,
            seed=args.seed + 1_000,
            max_episode_steps=args.max_episode_steps,
            append_task_id=args.append_task_id,
            render_mode=None,
            disable_env_checker=args.disable_env_checker,
            relax_obs_bounds=args.relax_obs_bounds,
            reward_func_version=args.reward_func_version,
        ),
        filename=str(run_dir / "eval.monitor.csv"),
        info_keywords=("task_name", "task_idx", "is_success"),
    )

    model = _build_model(
        algo=args.algo,
        env=train_env,
        seed=args.seed,
        device=args.device,
        tensorboard_dir=run_dir / "tb",
        adapter_enabled=args.adapter_enabled,
        adapter_num_tasks=len(list(dict.fromkeys(tasks))),
        adapter_rank=args.adapter_rank,
        adapter_alpha=args.adapter_alpha,
        adapter_features_dim=args.adapter_features_dim,
        adapter_backbone_hidden_dim=args.adapter_backbone_hidden_dim,
        append_task_id=args.append_task_id,
        sac_learning_rate=args.sac_learning_rate,
        sac_batch_size=args.sac_batch_size,
        sac_gradient_steps=args.sac_gradient_steps,
        sac_buffer_size=args.sac_buffer_size,
        sac_learning_starts=args.sac_learning_starts,
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
        stats = evaluate_task(model, eval_env, task, args.eval_episodes)
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
            f"  eval={task:<24} "
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
        "algo": args.algo,
        "task_preset": args.task_preset,
        "tasks_sequence": tasks,
        "unique_tasks": unique_tasks,
        "adapter_enabled": args.adapter_enabled,
        "reward_func_version": args.reward_func_version,
        "timesteps": args.total_steps,
        "elapsed_sec": elapsed,
        "avg_success_rate": float(np.mean(success_rates)) if success_rates else float("nan"),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Meta-World Continual World runner")
    p.add_argument("--mode", choices=["continual", "multitask"], default="continual")
    p.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    p.add_argument("--task-preset", default="cw10")
    p.add_argument("--tasks", type=str, default=None, help="CSV de tareas v3")
    p.add_argument("--steps-per-task", type=int, default=550_000)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument(
        "--reward-func-version",
        type=str,
        default="v2",
        help="Versión de reward en Meta-World (ej. v2). Usa '' para default de la env.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
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
    p.add_argument("--adapter-rank", dest="adapter_rank", type=int, default=16)
    p.add_argument("--lora-rank", dest="adapter_rank", type=int, help=argparse.SUPPRESS)
    p.add_argument("--adapter-alpha", dest="adapter_alpha", type=float, default=16.0)
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
        help="Número de tareas iniciales con entrenamiento completo antes de freeze.",
    )
    p.add_argument(
        "--lora-warmup-tasks",
        dest="adapter_warmup_tasks",
        type=int,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--adapter-train-actor-heads-after-warmup",
        dest="adapter_train_actor_heads_after_warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tras warm-up, entrenar heads del actor además del adaptador activo.",
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
        default=True,
        help="Tras warm-up, entrenar crítico completo (recomendado para SAC).",
    )
    p.add_argument(
        "--adapter-share-features-extractor",
        dest="adapter_share_features_extractor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="No usar. En SAC se fuerza False para evitar gradientes ambiguos.",
    )
    p.add_argument(
        "--lora-share-features-extractor",
        dest="adapter_share_features_extractor",
        action=argparse.BooleanOptionalAction,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--sac-learning-rate", type=float, default=1e-4)
    p.add_argument("--sac-batch-size", type=int, default=256)
    p.add_argument("--sac-gradient-steps", type=int, default=1)
    p.add_argument("--sac-buffer-size", type=int, default=1_000_000)
    p.add_argument("--sac-learning-starts", type=int, default=10_000)
    p.add_argument(
        "--sac-random-steps-per-task",
        type=int,
        default=20_000,
        help="Acciones aleatorias al inicio de cada tarea (protocolo CW en SAC).",
    )
    p.add_argument(
        "--reset-optimizers-every-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset de estados de optimizadores al cambiar de tarea.",
    )
    p.add_argument(
        "--reset-ent-coef-every-task",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="En SAC: resetear log_ent_coef al cambiar de tarea.",
    )
    p.add_argument(
        "--ent-coef-reset-log-value",
        type=float,
        default=0.0,
        help="Valor de reinicio para log_ent_coef (alpha=exp(valor)).",
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
    p.add_argument("--disable-env-checker", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--relax-obs-bounds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--reset-replay-every-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Para SAC/continual: reset de replay buffer entre tareas (similar CW paper).",
    )
    p.add_argument("--save-model-each-phase", action="store_true")
    p.add_argument("--progress-bar", action="store_true")
    p.add_argument(
        "--log-dir",
        default="rl/rl_uniandes/drl/mujoco-drl/logs/metaworld_cw",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if isinstance(args.reward_func_version, str) and args.reward_func_version.strip().lower() in {
        "",
        "none",
        "null",
    }:
        args.reward_func_version = None
    if args.adapter_enabled and not args.append_task_id:
        raise ValueError("Con --adapter-enabled debes activar --append-task-id para routing por tarea.")
    if args.adapter_rank <= 0:
        raise ValueError("--adapter-rank debe ser > 0.")
    if args.adapter_warmup_tasks <= 0:
        raise ValueError("--adapter-warmup-tasks debe ser >= 1.")
    if args.adapter_share_features_extractor:
        raise ValueError(
            "--adapter-share-features-extractor no está soportado: en SAC se mantiene False."
        )
    if args.sac_learning_starts < 0:
        raise ValueError("--sac-learning-starts no puede ser negativo.")
    if args.sac_random_steps_per_task < 0:
        raise ValueError("--sac-random-steps-per-task no puede ser negativo.")
    _set_seed(args.seed)

    tasks = resolve_task_sequence(args.task_preset, args.tasks)
    run_name = (
        f"{args.mode}_{args.algo}_{args.task_preset}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}_n{len(tasks)}"
    )
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps({"args": vars(args), "tasks": tasks}, indent=2),
        encoding="utf-8",
    )

    print(f"Run dir: {run_dir}")
    print(f"Mode={args.mode} Algo={args.algo} Tasks={len(tasks)}")
    print(f"Task sequence: {tasks}\n")

    if args.mode == "continual":
        summary = run_continual(args, tasks, run_dir)
    else:
        summary = run_multitask(args, tasks, run_dir)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nArtefactos en: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
