from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .detector import PageHinkleyDetector


@dataclass
class AdaptationConfig:
    method: str
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 80_000
    eps_reset_value: float | None = None
    eps_decay_steps_after_switch: int | None = None
    epsilon_reset_on_switch: bool = False
    td_adaptive_loss: bool = False
    alpha_max_mult: float = 3.0
    td_k: float = 1.0
    detector_signal: str = "return"
    detector_ema_alpha: float = 0.05
    ph_delta: float = 0.005
    ph_threshold: float = 5.0
    ph_min_instances: int = 20
    replay_policy: str = "keep_all"
    keep_recent_frac: float = 0.2
    archive_frac: float = 0.25
    recent_mix_start: float = 0.8
    recent_mix_end: float = 0.5
    post_switch_steps: int = 5_000
    segmented_keep_tail: int = 512
    segmented_recent_only_steps: int = 1_000
    segmented_min_recent_samples: int = 256
    segmented_revisit_recent_mix_start: float | None = None
    segmented_revisit_recent_mix_end: float | None = None
    segmented_revisit_recent_only_steps: int | None = None
    base_updates_per_train_step: int = 1
    post_switch_update_repeats: int = 1
    post_switch_extra_update_steps: int = 0
    distill_lambda: float = 0.0
    distill_new_task_only: bool = True


class EpsilonScheduler:
    def __init__(self, start: float, end: float, decay_steps: int) -> None:
        self.start = float(start)
        self.end = float(end)
        self.base_decay_steps = max(1, int(decay_steps))
        self.decay_steps = self.base_decay_steps
        self.steps = 0

    def value(self) -> float:
        frac = min(1.0, self.steps / self.decay_steps)
        return self.start + frac * (self.end - self.start)

    def step(self) -> None:
        self.steps += 1

    def reset(self, value: float | None = None, decay_steps: int | None = None) -> None:
        self.decay_steps = max(1, int(decay_steps)) if decay_steps is not None else self.base_decay_steps
        if value is None:
            self.steps = 0
            return
        if self.start == self.end:
            self.steps = 0
            return
        frac = (float(value) - self.start) / (self.end - self.start)
        frac = min(1.0, max(0.0, frac))
        self.steps = int(round(frac * self.decay_steps))

    def reset_to_max(self) -> None:
        self.reset(value=None, decay_steps=None)


class AdaptationController:
    def __init__(self, config: AdaptationConfig) -> None:
        self.config = config
        self.epsilon_scheduler = EpsilonScheduler(
            start=self.config.eps_start,
            end=self.config.eps_end,
            decay_steps=self.config.eps_decay_steps,
        )
        self.detector = (
            PageHinkleyDetector(
                delta=self.config.ph_delta,
                threshold=self.config.ph_threshold,
                min_instances=self.config.ph_min_instances,
            )
            if self.uses_detector
            else None
        )
        self.ema_signal: float | None = None
        self.steps_since_switch = 10**9
        self.num_detections = 0
        self.last_signal_raw = 0.0
        self.last_signal_ema = 0.0
        self.last_ph_stat = 0.0
        self.current_switch_type = "initial"
        self.frozen_net: nn.Module | None = None

    @property
    def uses_oracle_boundaries(self) -> bool:
        return self.config.method in {
            "oracle_reset",
            "morphin_lite",
            "oracle_segmented",
            "oracle_segmented_td",
            "oracle_segmented_td_plus",
            "oracle_segmented_revisit_aware",
            "oracle_segmented_td_revisit_aware",
            "oracle_segmented_distill",
            "oracle_segmented_distill_l001",
            "oracle_segmented_distill_l005",
            "oracle_segmented_distill_l020",
            "oracle_segmented_distill_l050",
            "oracle_segmented_distill_l200",
            "oracle_segmented_af015",
            "oracle_segmented_af020",
            "oracle_segmented_af025",
            "oracle_segmented_af030",
            "oracle_der_plus_plus",
        }

    @property
    def uses_detector(self) -> bool:
        return self.config.method in {
            "detector_reset_only",
            "morphin_full",
            "morphin_segmented",
            "morphin_detect",
            "morphin_detect_seg",
        }

    def current_epsilon(self) -> float:
        return self.epsilon_scheduler.value()

    def on_env_step(self) -> None:
        self.epsilon_scheduler.step()
        self.steps_since_switch += 1

    def on_task_switch(
        self,
        replay_buffer: Any,
        switch_type: str | None = None,
        online_net: nn.Module | None = None,
    ) -> dict[str, Any]:
        if self.config.epsilon_reset_on_switch:
            self.epsilon_scheduler.reset(
                value=self.config.eps_reset_value,
                decay_steps=self.config.eps_decay_steps_after_switch,
            )
        self.steps_since_switch = 0
        self.current_switch_type = switch_type or "unknown"
        self._apply_replay_switch(replay_buffer)
        should_distill = self.config.distill_lambda > 0 and online_net is not None
        if self.config.distill_new_task_only:
            should_distill = should_distill and self.current_switch_type == "new_task"
        if should_distill:
            self.frozen_net = copy.deepcopy(online_net)
            self.frozen_net.eval()
            for p in self.frozen_net.parameters():
                p.requires_grad_(False)
        else:
            self.frozen_net = None
        return {"switch_trigger": "oracle", "switch_type": self.current_switch_type}

    def on_episode_end(
        self,
        episode_return: float,
        success: float,
        reward_scale: float,
        replay_buffer: Any,
    ) -> bool:
        if not self.uses_detector or self.detector is None:
            return False

        signal = self._build_detector_signal(
            episode_return=episode_return,
            success=success,
            reward_scale=reward_scale,
        )
        self.last_signal_raw = signal
        if self.ema_signal is None:
            self.ema_signal = signal
        else:
            alpha = float(self.config.detector_ema_alpha)
            self.ema_signal = (1.0 - alpha) * self.ema_signal + alpha * signal
        self.last_signal_ema = self.ema_signal

        drift = self.detector.update(self.ema_signal)
        self.last_ph_stat = self.detector.last_statistic
        if not drift:
            return False

        self.num_detections += 1
        if self.config.epsilon_reset_on_switch:
            self.epsilon_scheduler.reset(
                value=self.config.eps_reset_value,
                decay_steps=self.config.eps_decay_steps_after_switch,
            )
        self.steps_since_switch = 0
        self.current_switch_type = "unknown"
        self.detector.reset()
        self.ema_signal = None
        self.last_signal_raw = signal
        self.last_signal_ema = signal
        self.last_ph_stat = 0.0
        self._apply_replay_switch(replay_buffer)
        return True

    def td_loss_weights(self, td_abs: torch.Tensor) -> torch.Tensor:
        if not self.config.td_adaptive_loss:
            return torch.ones_like(td_abs)
        # Only apply during the post-switch adaptation window
        if self.steps_since_switch > self.config.post_switch_steps:
            return torch.ones_like(td_abs)
        # Decay the reweighting strength linearly over the post-switch window
        decay = 1.0 - min(1.0, self.steps_since_switch / max(1, self.config.post_switch_steps))
        # DOWNWEIGHT high-TD samples (they are likely stale/misleading after switch)
        td_mean = td_abs.mean().detach()
        td_norm = td_abs / (td_mean + 1e-6)
        # High td_norm -> low weight (inverse sigmoid); low td_norm -> high weight
        max_mult = 1.0 + (self.config.alpha_max_mult - 1.0) * decay
        weights = 1.0 + (max_mult - 1.0) * (1.0 - torch.sigmoid(td_norm - self.config.td_k))
        return weights.clamp(min=0.5, max=max_mult)

    def segmented_recent_fraction(self) -> float:
        recent_only_steps, mix_start, mix_end = self._segmented_schedule()
        if self.steps_since_switch < recent_only_steps:
            return 1.0
        effective_steps = self.steps_since_switch - recent_only_steps
        frac = min(1.0, effective_steps / max(1, self.config.post_switch_steps))
        return mix_start + frac * (mix_end - mix_start)

    def can_update(self, replay_buffer: Any, batch_size: int) -> bool:
        if len(replay_buffer) < int(batch_size):
            return False
        if self.config.replay_policy != "segmented":
            return True
        recent_only_steps, _, _ = self._segmented_schedule()
        if self.steps_since_switch < recent_only_steps:
            if hasattr(replay_buffer, "num_recent"):
                return int(replay_buffer.num_recent()) >= int(self.config.segmented_min_recent_samples)
        return True

    def update_repeats(self) -> int:
        repeats = int(self.config.base_updates_per_train_step)
        if self.steps_since_switch < int(self.config.post_switch_extra_update_steps):
            repeats = max(repeats, int(self.config.post_switch_update_repeats))
        return max(1, repeats)

    def _segmented_schedule(self) -> tuple[int, float, float]:
        if (
            self.current_switch_type == "revisit_task"
            and self.config.segmented_revisit_recent_mix_start is not None
            and self.config.segmented_revisit_recent_mix_end is not None
        ):
            revisit_recent_only_steps = self.config.segmented_revisit_recent_only_steps
            return (
                int(revisit_recent_only_steps or 0),
                float(self.config.segmented_revisit_recent_mix_start),
                float(self.config.segmented_revisit_recent_mix_end),
            )
        return (
            int(self.config.segmented_recent_only_steps),
            float(self.config.recent_mix_start),
            float(self.config.recent_mix_end),
        )

    def _apply_replay_switch(self, replay_buffer: Any) -> None:
        if self.config.replay_policy == "keep_all":
            return
        if self.config.replay_policy == "segmented":
            replay_buffer.on_task_switch(
                archive_frac=self.config.archive_frac,
                keep_tail=self.config.segmented_keep_tail,
            )
            return
        replay_buffer.on_task_switch(
            policy=self.config.replay_policy,
            keep_recent_frac=self.config.keep_recent_frac,
        )

    def _build_detector_signal(
        self,
        episode_return: float,
        success: float,
        reward_scale: float,
    ) -> float:
        if self.config.detector_signal == "success":
            return -float(success)
        if self.config.detector_signal == "return":
            scale = max(1.0, float(reward_scale))
            return -(float(episode_return) / scale)
        raise ValueError(f"Unsupported detector signal: {self.config.detector_signal}")
