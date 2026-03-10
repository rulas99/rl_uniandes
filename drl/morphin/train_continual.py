from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agents import DDQNAgent, DDQNConfig, SegmentedReplayBuffer, UniformReplayBuffer
from .adapt import AdaptationConfig, AdaptationController
from .analysis_metrics import compute_detection_metrics
from .analysis_metrics import compute_overall_time_to_threshold
from .analysis_metrics import compute_switch_metrics
from .envs.gridworld_switch import (
    BENCHMARK_LIBRARY,
    BENCHMARK_DEFAULT_OBS_MODE,
    TASK_LIBRARY,
    GridWorldTaskSpec,
    build_task_sequence,
    make_gridworld_env,
)
from .envs.obs_modes import OBS_MODE_CHOICES, obs_to_state


METHOD_CHOICES = (
    "ddqn_scratch",
    "ddqn_vanilla",
    "oracle_reset",
    "morphin_lite",
    "detector_reset_only",
    "oracle_segmented",
    "morphin_full",
    "morphin_segmented",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DDQN/MORPHIN continual adaptation runner")
    parser.add_argument("--mode", choices=["continual", "scratch_task"], default="continual")
    parser.add_argument("--method", choices=METHOD_CHOICES, default="ddqn_vanilla")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARK_LIBRARY), default="gw_goal_switch_aba_v1")
    parser.add_argument("--task-id", choices=sorted(TASK_LIBRARY), default=None)
    parser.add_argument("--task-ids-csv", type=str, default="")
    parser.add_argument("--episodes-per-task", type=int, default=400)
    parser.add_argument("--max-steps-per-episode", type=int, default=150)
    parser.add_argument("--obs-mode", choices=OBS_MODE_CHOICES, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-dir", type=str, default="logs/morphin")
    parser.add_argument("--run-name", type=str, default="")

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--hidden-sizes-csv", type=str, default="128,128")
    parser.add_argument("--buffer-capacity", type=int, default=10000)
    parser.add_argument("--recent-buffer-capacity", type=int, default=6000)
    parser.add_argument("--archive-buffer-capacity", type=int, default=4000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--train-freq", type=int, default=1)

    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=8000)
    parser.add_argument("--alpha-max-mult", type=float, default=3.0)
    parser.add_argument("--td-k", type=float, default=1.0)
    parser.add_argument("--detector-signal", choices=["return", "success"], default="return")
    parser.add_argument("--detector-ema-alpha", type=float, default=0.05)
    parser.add_argument("--ph-delta", type=float, default=0.005)
    parser.add_argument("--ph-threshold", type=float, default=5.0)
    parser.add_argument("--ph-min-instances", type=int, default=20)
    parser.add_argument("--keep-recent-frac", type=float, default=0.2)
    parser.add_argument("--archive-frac", type=float, default=0.25)
    parser.add_argument("--recent-mix-start", type=float, default=0.8)
    parser.add_argument("--recent-mix-end", type=float, default=0.5)
    parser.add_argument("--post-switch-steps", type=int, default=5000)

    parser.add_argument("--eval-every-episodes", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-bank-base-seed", type=int, default=10000)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--recovery-window", type=int, default=25)
    parser.add_argument("--detector-max-delay-episodes", type=int, default=25)
    parser.add_argument("--scratch-summary-json", type=str, default="")
    return parser


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def parse_hidden_sizes(text: str) -> tuple[int, ...]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("hidden-sizes-csv must contain at least one integer")
    return tuple(values)


def default_adaptation_config(args: argparse.Namespace) -> AdaptationConfig:
    if args.method == "ddqn_scratch":
        replay_policy = "keep_all"
        epsilon_reset_on_switch = False
        td_adaptive = False
    elif args.method == "ddqn_vanilla":
        replay_policy = "keep_all"
        epsilon_reset_on_switch = False
        td_adaptive = False
    elif args.method == "oracle_reset":
        replay_policy = "keep_all"
        epsilon_reset_on_switch = True
        td_adaptive = False
    elif args.method == "morphin_lite":
        replay_policy = "keep_all"
        epsilon_reset_on_switch = True
        td_adaptive = True
    elif args.method == "detector_reset_only":
        replay_policy = "keep_all"
        epsilon_reset_on_switch = True
        td_adaptive = False
    elif args.method == "oracle_segmented":
        replay_policy = "segmented"
        epsilon_reset_on_switch = True
        td_adaptive = False
    elif args.method == "morphin_full":
        replay_policy = "keep_all"
        epsilon_reset_on_switch = True
        td_adaptive = True
    elif args.method == "morphin_segmented":
        replay_policy = "segmented"
        epsilon_reset_on_switch = True
        td_adaptive = True
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    return AdaptationConfig(
        method=args.method,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        epsilon_reset_on_switch=epsilon_reset_on_switch,
        td_adaptive_loss=td_adaptive,
        alpha_max_mult=args.alpha_max_mult,
        td_k=args.td_k,
        detector_signal=args.detector_signal,
        detector_ema_alpha=args.detector_ema_alpha,
        ph_delta=args.ph_delta,
        ph_threshold=args.ph_threshold,
        ph_min_instances=args.ph_min_instances,
        replay_policy=replay_policy,
        keep_recent_frac=args.keep_recent_frac,
        archive_frac=args.archive_frac,
        recent_mix_start=args.recent_mix_start,
        recent_mix_end=args.recent_mix_end,
        post_switch_steps=args.post_switch_steps,
    )


def build_task_sequence_for_run(args: argparse.Namespace) -> list[GridWorldTaskSpec]:
    if args.mode == "scratch_task":
        if not args.task_id:
            raise ValueError("--task-id is required in scratch_task mode")
        return [TASK_LIBRARY[args.task_id]]
    return build_task_sequence(
        benchmark=args.benchmark,
        task_ids_csv=args.task_ids_csv,
    )


def load_scratch_refs(path: str) -> dict[str, dict[str, float]] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text())


def resolve_obs_mode(args: argparse.Namespace) -> str:
    if args.obs_mode:
        return args.obs_mode
    return BENCHMARK_DEFAULT_OBS_MODE.get(args.benchmark, "agent_target")


def stable_task_seed(task_id: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(task_id))


def build_eval_seed_bank(
    task_sequence: list[GridWorldTaskSpec],
    base_seed: int,
    num_eval_episodes: int,
) -> dict[str, list[int]]:
    task_ids = []
    for task in task_sequence:
        if task.task_id not in task_ids:
            task_ids.append(task.task_id)
    bank: dict[str, list[int]] = {}
    for task_id in task_ids:
        rng = np.random.default_rng(base_seed + stable_task_seed(task_id))
        bank[task_id] = [int(value) for value in rng.integers(0, 10_000_000, size=num_eval_episodes)]
    return bank


def evaluate_task(
    agent: DDQNAgent,
    task: GridWorldTaskSpec,
    obs_mode: str,
    eval_seeds: list[int],
    max_steps: int,
) -> dict[str, float]:
    env = make_gridworld_env(task, seed=eval_seeds[0] if eval_seeds else None)
    returns: list[float] = []
    successes: list[float] = []
    steps_taken: list[int] = []
    for seed in eval_seeds:
        obs, _ = env.reset(
            seed=int(seed),
            options=task.reset_options(),
        )
        state = obs_to_state(obs, task=task, obs_mode=obs_mode)
        done = False
        episode_return = 0.0
        episode_steps = 0
        while not done and episode_steps < max_steps:
            action = agent.select_action(state, epsilon=0.0, greedy=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            state = obs_to_state(next_obs, task=task, obs_mode=obs_mode)
            episode_return += float(reward)
            episode_steps += 1
        returns.append(episode_return)
        success = bool(info.get("is_success", False))
        if bool(truncated) and not success:
            success = False
        successes.append(float(success))
        steps_taken.append(episode_steps)
    env.close()
    return {
        "mean_return": float(np.mean(returns)) if returns else math.nan,
        "success_rate": float(np.mean(successes)) if successes else math.nan,
        "mean_steps": float(np.mean(steps_taken)) if steps_taken else math.nan,
    }


def make_replay_buffer(args: argparse.Namespace, adaptation: AdaptationConfig) -> Any:
    if adaptation.replay_policy == "segmented":
        return SegmentedReplayBuffer(
            recent_capacity=args.recent_buffer_capacity,
            archive_capacity=args.archive_buffer_capacity,
        )
    return UniformReplayBuffer(capacity=args.buffer_capacity)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(
    args: argparse.Namespace,
    task_sequence: list[GridWorldTaskSpec],
    episode_rows: list[dict[str, object]],
    eval_rows: list[dict[str, object]],
    switch_rows: list[dict[str, object]],
    detection_rows: list[dict[str, object]],
    false_alarm_count: int,
    obs_mode: str,
    total_steps: int,
    adaptation_controller: AdaptationController,
) -> dict[str, object]:
    overall_time_to_threshold = compute_overall_time_to_threshold(
        episode_rows=episode_rows,
        threshold=args.success_threshold,
        ema_alpha=0.1,
    )
    final_eval_rows = [row for row in eval_rows if str(row["eval_scope"]) == "seen_tasks_end"]
    final_eval_mean = math.nan
    current_task_final_success = math.nan
    if final_eval_rows:
        last_episode = max(int(row["episode"]) for row in final_eval_rows)
        final_block = [row for row in final_eval_rows if int(row["episode"]) == last_episode]
        final_eval_mean = float(np.mean([float(row["success_rate"]) for row in final_block]))
        current_rows = [row for row in final_block if row["task_id"] == task_sequence[-1].task_id]
        if current_rows:
            current_task_final_success = float(np.mean([float(row["success_rate"]) for row in current_rows]))

    summary = {
        "mode": args.mode,
        "method": args.method,
        "benchmark": args.benchmark,
        "seed": args.seed,
        "obs_mode": obs_mode,
        "episodes_per_task": args.episodes_per_task,
        "num_tasks": len(task_sequence),
        "task_sequence": [task.task_id for task in task_sequence],
        "total_episodes": len(episode_rows),
        "total_steps": int(total_steps),
        "avg_episode_return": float(np.mean([float(row["episode_return"]) for row in episode_rows])),
        "avg_episode_success": float(np.mean([float(row["success"]) for row in episode_rows])),
        "overall_time_to_threshold": overall_time_to_threshold,
        "final_seen_success_mean": final_eval_mean,
        "current_task_final_success": current_task_final_success,
        "num_oracle_switches": int(sum(1 for row in episode_rows if row["switch_event"] == "oracle")),
        "num_detected_drifts": int(adaptation_controller.num_detections),
        "num_false_alarms": int(false_alarm_count),
        "mean_detection_delay_episodes": (
            float(np.mean([float(row["delay_episodes"]) for row in detection_rows if row["delay_episodes"] is not None]))
            if any(row["delay_episodes"] is not None for row in detection_rows)
            else math.nan
        ),
        "switch_recovery_auc_success_mean": (
            float(np.nanmean([float(row["recovery_auc_success"]) for row in switch_rows]))
            if switch_rows
            else math.nan
        ),
        "switch_time_to_threshold_mean": (
            float(np.nanmean([float(row["time_to_threshold"]) for row in switch_rows if row["time_to_threshold"] is not None]))
            if any(row["time_to_threshold"] is not None for row in switch_rows)
            else math.nan
        ),
        "adaptation_gain_vs_scratch_mean": (
            float(np.nanmean([float(row["adaptation_gain_vs_scratch"]) for row in switch_rows]))
            if any(not math.isnan(float(row["adaptation_gain_vs_scratch"])) for row in switch_rows)
            else math.nan
        ),
        "shock_drop_mean": (
            float(np.nanmean([float(row["shock_drop"]) for row in switch_rows]))
            if any(not math.isnan(float(row["shock_drop"])) for row in switch_rows)
            else math.nan
        ),
    }
    return summary


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    task_sequence = build_task_sequence_for_run(args)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes_csv)
    obs_mode = resolve_obs_mode(args)

    run_name = args.run_name or (
        f"{args.method}_{args.benchmark}_{time.strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    )
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_dump = vars(args).copy()
    config_dump["device_resolved"] = str(device)
    config_dump["obs_mode_resolved"] = obs_mode
    config_dump["hidden_sizes"] = list(hidden_sizes)
    config_dump["task_sequence"] = [task.to_dict() for task in task_sequence]
    (run_dir / "config.json").write_text(json.dumps(config_dump, indent=2))

    bootstrap_env = make_gridworld_env(task_sequence[0], seed=args.seed)
    bootstrap_obs, _ = bootstrap_env.reset(seed=args.seed, options=task_sequence[0].reset_options())
    bootstrap_state = obs_to_state(bootstrap_obs, task=task_sequence[0], obs_mode=obs_mode)
    obs_dim = int(np.asarray(bootstrap_state, dtype=np.float32).reshape(-1).shape[0])
    action_dim = int(bootstrap_env.action_space.n)
    bootstrap_env.close()

    ddqn_config = DDQNConfig(
        gamma=args.gamma,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        tau=args.tau,
        gradient_clip_norm=args.gradient_clip_norm,
        hidden_sizes=hidden_sizes,
    )
    agent = DDQNAgent(obs_dim=obs_dim, action_dim=action_dim, config=ddqn_config, device=device)
    adaptation = default_adaptation_config(args)
    controller = AdaptationController(adaptation)
    replay_buffer = make_replay_buffer(args, adaptation)
    scratch_refs = load_scratch_refs(args.scratch_summary_json)
    eval_seed_bank = build_eval_seed_bank(
        task_sequence=task_sequence,
        base_seed=args.eval_bank_base_seed + args.seed,
        num_eval_episodes=args.eval_episodes,
    )
    (run_dir / "eval_seed_bank.json").write_text(json.dumps(eval_seed_bank, indent=2))

    total_episodes = args.episodes_per_task * len(task_sequence)
    episode_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    switch_episodes: list[int] = []
    detector_rows: list[dict[str, object]] = []
    total_steps = 0
    env = None
    current_env_task_id = None

    for episode_idx in range(total_episodes):
        task_idx = min(episode_idx // args.episodes_per_task, len(task_sequence) - 1)
        task = task_sequence[task_idx]
        actual_switch = episode_idx > 0 and task_sequence[task_idx - 1].task_id != task.task_id
        switch_event = "none"
        if actual_switch:
            switch_episodes.append(episode_idx)
            if controller.uses_oracle_boundaries:
                controller.on_task_switch(replay_buffer)
                switch_event = "oracle"
                detector_rows.append(
                    {
                        "episode": episode_idx + 1,
                        "task_id": task.task_id,
                        "event_type": "oracle_switch",
                        "ph_signal_raw": controller.last_signal_raw,
                        "ph_signal_ema": controller.last_signal_ema,
                        "ph_stat": controller.last_ph_stat,
                        "epsilon_after_event": controller.current_epsilon(),
                    }
                )

        if env is None or current_env_task_id != task.task_id:
            if env is not None:
                env.close()
            env = make_gridworld_env(task, seed=args.seed + episode_idx)
            current_env_task_id = task.task_id

        obs, _ = env.reset(seed=args.seed + episode_idx, options=task.reset_options())
        state = obs_to_state(obs, task=task, obs_mode=obs_mode)

        episode_return = 0.0
        success = 0.0
        terminated_reason = "ongoing"
        train_updates = 0
        td_abs_mean_values: list[float] = []
        td_abs_max_values: list[float] = []
        loss_values: list[float] = []
        hit_obstacle_count = 0
        epsilon_start = controller.current_epsilon()

        for step_in_episode in range(args.max_steps_per_episode):
            epsilon = controller.current_epsilon()
            action = agent.select_action(state, epsilon=epsilon, greedy=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            hit_obstacle_count += int(bool(info.get("hit_obstacle", False)))
            done = bool(terminated or truncated)
            next_state = None if done else obs_to_state(next_obs, task=task, obs_mode=obs_mode)
            replay_buffer.add(
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                done=done,
                task_id=task.task_id,
            )
            state = next_state if next_state is not None else state
            episode_return += float(reward)
            total_steps += 1
            controller.on_env_step()

            if (
                total_steps >= args.warmup_steps
                and len(replay_buffer) >= args.batch_size
                and total_steps % args.train_freq == 0
            ):
                sample_kwargs: dict[str, object] = {}
                if adaptation.replay_policy == "segmented":
                    sample_kwargs["p_recent"] = controller.segmented_recent_fraction()
                batch = replay_buffer.sample(
                    batch_size=args.batch_size,
                    device=device,
                    **sample_kwargs,
                )
                stats = agent.update(batch=batch, weight_fn=controller.td_loss_weights)
                train_updates += 1
                td_abs_mean_values.append(stats["td_abs_mean"])
                td_abs_max_values.append(stats["td_abs_max"])
                loss_values.append(stats["loss"])

            if done:
                success = float(bool(info.get("is_success", False)))
                terminated_reason = (
                    "timeout"
                    if bool(truncated) and not bool(info.get("is_success", False))
                    else str(info.get("terminated_reason", "terminated"))
                )
                break

        drift_detected = controller.on_episode_end(
            episode_return=episode_return,
            success=success,
            reward_scale=task.reward_scale,
            replay_buffer=replay_buffer,
        )
        if drift_detected:
            switch_event = "detected"
            detector_rows.append(
                {
                    "episode": episode_idx + 1,
                    "task_id": task.task_id,
                    "event_type": "detected",
                    "ph_signal_raw": controller.last_signal_raw,
                    "ph_signal_ema": controller.last_signal_ema,
                    "ph_stat": controller.last_ph_stat,
                    "epsilon_after_event": controller.current_epsilon(),
                }
            )

        episode_rows.append(
            {
                "episode": episode_idx + 1,
                "task_index": task_idx,
                "task_id": task.task_id,
                "obs_mode": obs_mode,
                "episode_return": episode_return,
                "success": success,
                "terminated_reason": terminated_reason,
                "episode_steps": step_in_episode + 1,
                "global_step": total_steps,
                "epsilon_start": epsilon_start,
                "epsilon_end": controller.current_epsilon(),
                "train_updates": train_updates,
                "loss_mean": float(np.mean(loss_values)) if loss_values else math.nan,
                "td_abs_mean": float(np.mean(td_abs_mean_values)) if td_abs_mean_values else math.nan,
                "td_abs_max": float(np.max(td_abs_max_values)) if td_abs_max_values else math.nan,
                "ph_signal_raw": controller.last_signal_raw,
                "ph_signal_ema": controller.last_signal_ema,
                "ph_stat": controller.last_ph_stat,
                "hit_obstacle_count": hit_obstacle_count,
                "switch_event": switch_event,
                "actual_boundary": int(actual_switch),
                "drift_detected": int(drift_detected),
            }
        )

        task_boundary_end = ((episode_idx + 1) % args.episodes_per_task == 0) or (episode_idx == total_episodes - 1)
        do_eval = args.eval_every_episodes > 0 and ((episode_idx + 1) % args.eval_every_episodes == 0)
        if do_eval or task_boundary_end:
            current_metrics = evaluate_task(
                agent=agent,
                task=task,
                obs_mode=obs_mode,
                eval_seeds=eval_seed_bank[task.task_id],
                max_steps=args.max_steps_per_episode,
            )
            eval_rows.append(
                {
                    "episode": episode_idx + 1,
                    "global_step": total_steps,
                    "eval_scope": "current_task",
                    "task_id": task.task_id,
                    "obs_mode": obs_mode,
                    **current_metrics,
                }
            )
            if task_boundary_end:
                seen_tasks = []
                for past_task in task_sequence[: task_idx + 1]:
                    if past_task.task_id not in seen_tasks:
                        seen_tasks.append(past_task.task_id)
                for seen_task_id in seen_tasks:
                    seen_task = TASK_LIBRARY[seen_task_id]
                    seen_metrics = evaluate_task(
                        agent=agent,
                        task=seen_task,
                        obs_mode=obs_mode,
                        eval_seeds=eval_seed_bank[seen_task_id],
                        max_steps=args.max_steps_per_episode,
                    )
                    eval_rows.append(
                        {
                            "episode": episode_idx + 1,
                            "global_step": total_steps,
                            "eval_scope": "seen_tasks_end",
                            "task_id": seen_task_id,
                            "obs_mode": obs_mode,
                            **seen_metrics,
                        }
                    )

        if (episode_idx + 1) % 50 == 0 or episode_idx == total_episodes - 1:
            print(
                f"[ep {episode_idx + 1}/{total_episodes}] task={task.task_id} "
                f"ret={episode_return:.2f} succ={success:.0f} "
                f"eps={controller.current_epsilon():.3f} drift={int(drift_detected)}"
            )

    if env is not None:
        env.close()

    switch_rows = compute_switch_metrics(
        episode_rows=episode_rows,
        switch_episodes=switch_episodes,
        recovery_window=args.recovery_window,
        threshold=args.success_threshold,
        ema_alpha=0.1,
        scratch_refs=scratch_refs,
    )
    detection_rows, false_alarm_rows = compute_detection_metrics(
        switch_episodes=switch_episodes,
        detector_rows=detector_rows,
        max_delay_episodes=args.detector_max_delay_episodes,
    )
    for row in false_alarm_rows:
        detector_rows.append({**row, "event_type": "false_alarm"})
    summary = summarize_run(
        args=args,
        task_sequence=task_sequence,
        episode_rows=episode_rows,
        eval_rows=eval_rows,
        switch_rows=switch_rows,
        detection_rows=detection_rows,
        false_alarm_count=len(false_alarm_rows),
        obs_mode=obs_mode,
        total_steps=total_steps,
        adaptation_controller=controller,
    )

    write_csv(run_dir / "episode_metrics.csv", episode_rows)
    write_csv(run_dir / "eval_metrics.csv", eval_rows)
    write_csv(run_dir / "switch_metrics.csv", switch_rows)
    write_csv(run_dir / "detector_events.csv", detector_rows)
    write_csv(run_dir / "detection_metrics.csv", detection_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    agent.save(run_dir / "model_final.pt")

    if args.mode == "scratch_task":
        scratch_summary = {
            str(task_sequence[0].task_id): {
                "time_to_threshold": summary["overall_time_to_threshold"],
                "avg_episode_success": summary["avg_episode_success"],
                "final_seen_success_mean": summary["final_seen_success_mean"],
            }
        }
        (run_dir / "scratch_reference.json").write_text(json.dumps(scratch_summary, indent=2))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
