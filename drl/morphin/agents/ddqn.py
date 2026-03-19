from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .network import MLPQNetwork


@dataclass
class DDQNConfig:
    gamma: float = 0.99
    batch_size: int = 128
    learning_rate: float = 5e-4
    tau: float = 0.005
    gradient_clip_norm: float = 10.0
    hidden_sizes: tuple[int, ...] = (128, 128)
    optimizer_eps: float = 1e-8


class DDQNAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: DDQNConfig,
        device: torch.device,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.config = config
        self.device = device

        self.online_net = MLPQNetwork(
            input_dim=self.obs_dim,
            output_dim=self.action_dim,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)
        self.target_net = MLPQNetwork(
            input_dim=self.obs_dim,
            output_dim=self.action_dim,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(
            self.online_net.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.optimizer_eps,
            amsgrad=True,
        )

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float,
        greedy: bool = False,
    ) -> int:
        if (not greedy) and np.random.random() < float(epsilon):
            return int(np.random.randint(self.action_dim))
        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            return int(self.online_net(state_tensor).argmax(dim=1).item())

    def update(
        self,
        batch: dict[str, torch.Tensor],
        weight_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        frozen_net: torch.nn.Module | None = None,
        distill_lambda: float = 0.0,
        der_batch: dict[str, torch.Tensor] | None = None,
        der_alpha: float = 0.0,
        der_beta: float = 1.0,
    ) -> dict[str, float]:
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        non_final_mask = batch["non_final_mask"]
        next_states = batch["next_states"]
        archive_mask = batch.get("archive_mask")

        q_all = self.online_net(states)
        q_sa = q_all.gather(1, actions)
        target_q = torch.zeros((states.shape[0], 1), dtype=torch.float32, device=self.device)

        if bool(non_final_mask.any()):
            with torch.no_grad():
                next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            target_q[non_final_mask] = next_q

        targets = rewards + self.config.gamma * (1.0 - dones) * target_q
        td_errors = targets - q_sa
        td_abs = td_errors.detach().abs()
        base_loss = F.smooth_l1_loss(q_sa, targets, reduction="none")

        use_distill = (
            frozen_net is not None
            and distill_lambda > 0
            and archive_mask is not None
            and archive_mask.sum() > 0
        )

        if use_distill:
            archive_mask_f = archive_mask.unsqueeze(1)

            # TD on ALL samples (recent + archive) — archive Q-values stay
            # current via Bellman updates, same as oracle_segmented.
            if weight_fn is not None:
                weights = weight_fn(td_abs)
                td_loss = (weights * base_loss).mean()
            else:
                td_loss = base_loss.mean()

            # Distill anchor on archive samples only — soft regularizer that
            # keeps old-task Q-values close to the frozen_net baseline,
            # preventing drift that TD alone cannot fully counteract.
            with torch.no_grad():
                frozen_q = frozen_net(states)
            distill_errors = F.mse_loss(q_all, frozen_q, reduction="none").mean(dim=1, keepdim=True)
            distill_loss = (distill_errors * archive_mask_f).sum() / archive_mask_f.sum().clamp(min=1)

            loss = td_loss + distill_lambda * distill_loss
            distill_batches = int(archive_mask.sum().item())
        else:
            if weight_fn is not None:
                weights = weight_fn(td_abs)
                loss = (weights * base_loss).mean()
            else:
                loss = base_loss.mean()
            distill_loss = torch.zeros(1, device=self.device)
            distill_batches = 0

        # ── DER++ auxiliary losses ────────────────────────────────────────────
        # α-term: MSE(Q_current(s_M), z_stored) — soft Q-value consistency
        #         across time, without requiring task boundaries.
        # β-term: TD loss on memory samples — replay memory stays Bellman-consistent.
        der_alpha_loss_val = 0.0
        der_beta_loss_val = 0.0
        if der_batch is not None and (der_alpha > 0 or der_beta > 0):
            der_states = der_batch["states"]
            der_q_current = self.online_net(der_states)

            if der_alpha > 0:
                der_z = der_batch["z_stored"]
                der_alpha_loss = F.mse_loss(der_q_current, der_z)
                loss = loss + der_alpha * der_alpha_loss
                der_alpha_loss_val = float(der_alpha_loss.item())

            if der_beta > 0:
                der_actions = der_batch["actions"]
                der_rewards = der_batch["rewards"]
                der_dones = der_batch["dones"]
                der_nf = der_batch["non_final_mask"]
                der_ns = der_batch["next_states"]
                der_q_sa = der_q_current.gather(1, der_actions)
                der_tgt = torch.zeros(
                    (der_states.shape[0], 1), dtype=torch.float32, device=der_states.device
                )
                if bool(der_nf.any()):
                    with torch.no_grad():
                        der_na = self.online_net(der_ns).argmax(dim=1, keepdim=True)
                        der_nq = self.target_net(der_ns).gather(1, der_na)
                    der_tgt[der_nf] = der_nq
                der_targets = der_rewards + self.config.gamma * (1.0 - der_dones) * der_tgt
                der_beta_loss = F.smooth_l1_loss(der_q_sa, der_targets.detach())
                loss = loss + der_beta * der_beta_loss
                der_beta_loss_val = float(der_beta_loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(),
            max_norm=self.config.gradient_clip_norm,
        )
        self.optimizer.step()
        self.soft_update()

        return {
            "loss": float(loss.item()),
            "td_abs_mean": float(td_abs.mean().item()),
            "td_abs_max": float(td_abs.max().item()),
            "q_sa_mean": float(q_sa.detach().mean().item()),
            "target_q_mean": float(targets.detach().mean().item()),
            "distill_loss": float(distill_loss.item()),
            "distill_active": float(bool(use_distill)),
            "distill_archive_samples": float(distill_batches),
            "der_alpha_loss": der_alpha_loss_val,
            "der_beta_loss": der_beta_loss_val,
        }

    def soft_update(self) -> None:
        online_state = self.online_net.state_dict()
        target_state = self.target_net.state_dict()
        for key, value in online_state.items():
            target_state[key] = value * self.config.tau + target_state[key] * (1.0 - self.config.tau)
        self.target_net.load_state_dict(target_state)

    def set_learning_rate_multiplier(self, multiplier: float) -> None:
        new_lr = float(self.config.learning_rate) * float(multiplier)
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

    def save(self, path: str | Path) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.__dict__,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
            },
            save_path,
        )
