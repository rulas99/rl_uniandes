from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def ema(values: Iterable[float], alpha: float) -> np.ndarray:
    values_arr = np.asarray(list(values), dtype=np.float32)
    if values_arr.size == 0:
        return values_arr
    out = np.zeros_like(values_arr)
    out[0] = values_arr[0]
    for idx in range(1, len(values_arr)):
        out[idx] = (1.0 - alpha) * out[idx - 1] + alpha * values_arr[idx]
    return out


def first_threshold_index(values: np.ndarray, threshold: float) -> int | None:
    indices = np.where(values >= float(threshold))[0]
    if indices.size == 0:
        return None
    return int(indices[0])


def first_sustained_threshold_index(
    values: np.ndarray,
    threshold: float,
    min_consecutive: int,
) -> int | None:
    if min_consecutive <= 1:
        return first_threshold_index(values, threshold=threshold)
    run_length = 0
    for idx, value in enumerate(values):
        if float(value) >= float(threshold):
            run_length += 1
            if run_length >= min_consecutive:
                return int(idx - min_consecutive + 1)
        else:
            run_length = 0
    return None


def safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or float(denominator) <= 0.0:
        return math.nan
    return float(numerator) / float(denominator)


def safe_difference(value: float | None, reference: float | None) -> float:
    if value is None or reference is None:
        return math.nan
    return float(value) - float(reference)


def safe_log_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or float(numerator) < 0.0 or float(denominator) < 0.0:
        return math.nan
    return math.log((float(numerator) + 1.0) / (float(denominator) + 1.0))


def compute_overall_time_to_threshold(
    episode_rows: list[dict[str, object]],
    threshold: float,
    ema_alpha: float,
) -> int | None:
    if not episode_rows:
        return None
    success = np.asarray([float(row["success"]) for row in episode_rows], dtype=np.float32)
    ema_success = ema(success, alpha=ema_alpha)
    threshold_index = first_threshold_index(ema_success, threshold=threshold)
    if threshold_index is None:
        return None
    return int(threshold_index + 1)


def compute_eval_time_to_threshold(
    eval_rows: list[dict[str, object]],
    threshold: float,
    min_consecutive: int = 1,
) -> dict[str, int | None]:
    current_rows = [row for row in eval_rows if str(row.get("eval_scope")) == "current_task"]
    if not current_rows:
        return {
            "time_to_threshold_eval_episodes": None,
            "time_to_threshold_eval_steps": None,
        }
    current_rows = sorted(current_rows, key=lambda row: (int(row["episode"]), int(row["global_step"])))
    success = np.asarray([float(row["success_rate"]) for row in current_rows], dtype=np.float32)
    threshold_index = first_sustained_threshold_index(
        success,
        threshold=threshold,
        min_consecutive=min_consecutive,
    )
    if threshold_index is None:
        return {
            "time_to_threshold_eval_episodes": None,
            "time_to_threshold_eval_steps": None,
        }
    return {
        "time_to_threshold_eval_episodes": int(current_rows[threshold_index]["episode"]),
        "time_to_threshold_eval_steps": int(current_rows[threshold_index]["global_step"]),
    }


def compute_train_switch_metrics(
    episode_rows: list[dict[str, object]],
    switch_episodes: list[int],
    recovery_window: int,
    threshold: float,
    ema_alpha: float,
    scratch_refs: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    if not episode_rows:
        return []
    success = np.asarray([float(row["success"]) for row in episode_rows], dtype=np.float32)
    reward = np.asarray([float(row["episode_return"]) for row in episode_rows], dtype=np.float32)

    switch_rows: list[dict[str, object]] = []
    seen_task_ids: set[str] = set()
    seen_task_ids.add(str(episode_rows[0]["task_id"]))
    for switch_idx, switch_episode in enumerate(switch_episodes):
        if switch_episode >= len(episode_rows):
            continue
        segment_end = switch_episodes[switch_idx + 1] if switch_idx + 1 < len(switch_episodes) else len(episode_rows)
        task_id = str(episode_rows[switch_episode]["task_id"])
        switch_type = "revisit_task" if task_id in seen_task_ids else "new_task"
        seen_task_ids.add(task_id)
        segment_success = success[switch_episode:segment_end]
        segment_reward = reward[switch_episode:segment_end]
        post_success = segment_success[:recovery_window]
        post_reward = segment_reward[:recovery_window]
        segment_ema = ema(segment_success, alpha=ema_alpha)
        recovery_auc = float(np.mean(post_success)) if post_success.size else math.nan
        reward_auc = float(np.mean(post_reward)) if post_reward.size else math.nan
        threshold_index = first_threshold_index(segment_ema, threshold=threshold)
        time_to_threshold = None if threshold_index is None else int(threshold_index + 1)
        scratch_time = None
        scratch_final_success = None
        scratch_ref_num_runs_total = None
        scratch_ref_num_runs_valid = None
        scratch_ref_valid_run_fraction = math.nan
        scratch_ref_is_stable = None
        adaptation_gain = math.nan
        time_to_threshold_delta_vs_scratch = math.nan
        log_adaptation_gain = math.nan
        shock_drop = math.nan
        if scratch_refs and task_id in scratch_refs:
            scratch_ref = scratch_refs[task_id]
            scratch_time = (
                scratch_ref.get("time_to_threshold_eval_episodes")
                if scratch_ref.get("time_to_threshold_eval_episodes") is not None
                else scratch_ref.get("time_to_threshold")
            )
            scratch_final_success = scratch_ref.get("final_seen_success_mean")
            scratch_ref_num_runs_total = scratch_ref.get("num_runs_total")
            scratch_ref_num_runs_valid = scratch_ref.get("num_runs_valid")
            if scratch_ref_num_runs_total:
                scratch_ref_valid_run_fraction = float(scratch_ref_num_runs_valid or 0) / float(scratch_ref_num_runs_total)
            scratch_ref_is_stable = scratch_ref.get("is_reference_stable")
            adaptation_gain = safe_ratio(scratch_time, time_to_threshold)
            time_to_threshold_delta_vs_scratch = safe_difference(time_to_threshold, scratch_time)
            log_adaptation_gain = safe_log_ratio(scratch_time, time_to_threshold)
            if scratch_final_success is not None and post_success.size:
                shock_drop = float(scratch_final_success) - float(post_success[0])

        switch_rows.append(
            {
                "switch_episode": int(switch_episode + 1),
                "switch_index": int(switch_idx + 1),
                "task_id": task_id,
                "switch_type": switch_type,
                "segment_end_episode": int(segment_end),
                "segment_length_episodes": int(segment_end - switch_episode),
                "shock_drop": shock_drop,
                "recovery_auc_success": recovery_auc,
                "recovery_auc_return": reward_auc,
                "time_to_threshold": time_to_threshold,
                "adaptation_gain_vs_scratch": adaptation_gain,
                "time_to_threshold_delta_vs_scratch": time_to_threshold_delta_vs_scratch,
                "log_adaptation_gain_vs_scratch": log_adaptation_gain,
                "scratch_time_to_threshold": scratch_time,
                "scratch_final_success_mean": scratch_final_success,
                "scratch_ref_num_runs_total": scratch_ref_num_runs_total,
                "scratch_ref_num_runs_valid": scratch_ref_num_runs_valid,
                "scratch_ref_valid_run_fraction": scratch_ref_valid_run_fraction,
                "scratch_ref_is_stable": scratch_ref_is_stable,
            }
        )
    return switch_rows


def compute_eval_switch_metrics(
    eval_rows: list[dict[str, object]],
    switch_episodes: list[int],
    threshold: float,
    min_consecutive: int = 1,
    scratch_refs: dict[str, dict[str, object]] | None = None,
    episode_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    if not eval_rows:
        return []
    current_rows = [
        row for row in eval_rows if str(row.get("eval_scope")) == "current_task"
    ]
    current_rows = sorted(current_rows, key=lambda row: (int(row["episode"]), int(row["global_step"])))
    switch_step_map: dict[int, int] = {}
    if episode_rows:
        for switch_episode in switch_episodes:
            if switch_episode <= 0:
                switch_step_map[int(switch_episode)] = 0
            elif switch_episode - 1 < len(episode_rows):
                switch_step_map[int(switch_episode)] = int(episode_rows[switch_episode - 1]["global_step"])
            else:
                switch_step_map[int(switch_episode)] = 0
    switch_rows: list[dict[str, object]] = []
    seen_task_ids: set[str] = set()
    if episode_rows:
        seen_task_ids.add(str(episode_rows[0]["task_id"]))
    elif current_rows:
        seen_task_ids.add(str(current_rows[0]["task_id"]))

    for switch_idx, switch_episode in enumerate(switch_episodes):
        segment_start_episode = int(switch_episode + 1)
        next_switch_episode = switch_episodes[switch_idx + 1] if switch_idx + 1 < len(switch_episodes) else None
        segment_end_exclusive = None if next_switch_episode is None else int(next_switch_episode + 1)
        block = [
            row
            for row in current_rows
            if int(row["episode"]) >= segment_start_episode
            and (segment_end_exclusive is None or int(row["episode"]) < segment_end_exclusive)
        ]
        if not block:
            continue

        task_id = str(block[0]["task_id"])
        switch_type = "revisit_task" if task_id in seen_task_ids else "new_task"
        seen_task_ids.add(task_id)
        success = np.asarray([float(row["success_rate"]) for row in block], dtype=np.float32)
        threshold_index = first_sustained_threshold_index(
            success,
            threshold=threshold,
            min_consecutive=min_consecutive,
        )
        time_to_threshold_eval_episodes = None
        time_to_threshold_eval_steps = None
        switch_global_step = int(switch_step_map.get(int(switch_episode), 0))
        if threshold_index is not None:
            time_to_threshold_eval_episodes = int(block[threshold_index]["episode"]) - int(switch_episode)
            time_to_threshold_eval_steps = int(block[threshold_index]["global_step"]) - switch_global_step

        scratch_steps = None
        scratch_episodes = None
        scratch_final_success = None
        scratch_ref_num_runs_total = None
        scratch_ref_num_runs_valid = None
        scratch_ref_valid_run_fraction = math.nan
        scratch_ref_is_stable = None
        adaptation_gain_steps = math.nan
        adaptation_gain_episodes = math.nan
        time_delta_steps = math.nan
        time_delta_episodes = math.nan
        log_adaptation_gain_steps = math.nan
        log_adaptation_gain_episodes = math.nan
        initial_gap_vs_scratch = math.nan
        if scratch_refs and task_id in scratch_refs:
            scratch_ref = scratch_refs[task_id]
            scratch_steps = scratch_ref.get("time_to_threshold_eval_steps")
            scratch_episodes = scratch_ref.get("time_to_threshold_eval_episodes")
            scratch_final_success = scratch_ref.get("final_seen_success_mean")
            scratch_ref_num_runs_total = scratch_ref.get("num_runs_total")
            scratch_ref_num_runs_valid = scratch_ref.get("num_runs_valid")
            if scratch_ref_num_runs_total:
                scratch_ref_valid_run_fraction = float(scratch_ref_num_runs_valid or 0) / float(scratch_ref_num_runs_total)
            scratch_ref_is_stable = scratch_ref.get("is_reference_stable")
            adaptation_gain_steps = safe_ratio(scratch_steps, time_to_threshold_eval_steps)
            adaptation_gain_episodes = safe_ratio(scratch_episodes, time_to_threshold_eval_episodes)
            time_delta_steps = safe_difference(time_to_threshold_eval_steps, scratch_steps)
            time_delta_episodes = safe_difference(time_to_threshold_eval_episodes, scratch_episodes)
            log_adaptation_gain_steps = safe_log_ratio(scratch_steps, time_to_threshold_eval_steps)
            log_adaptation_gain_episodes = safe_log_ratio(scratch_episodes, time_to_threshold_eval_episodes)
            if scratch_final_success is not None:
                initial_gap_vs_scratch = float(scratch_final_success) - float(success[0])

        switch_rows.append(
            {
                "switch_episode": segment_start_episode,
                "switch_index": int(switch_idx + 1),
                "task_id": task_id,
                "switch_type": switch_type,
                "segment_eval_points": int(len(block)),
                "threshold_min_consecutive_evals": int(min_consecutive),
                "recovery_auc_success_eval": float(np.mean(success)),
                "time_to_threshold_eval_episodes": time_to_threshold_eval_episodes,
                "time_to_threshold_eval_steps": time_to_threshold_eval_steps,
                "adaptation_gain_vs_scratch_steps": adaptation_gain_steps,
                "adaptation_gain_vs_scratch_episodes": adaptation_gain_episodes,
                "time_to_threshold_eval_steps_delta_vs_scratch": time_delta_steps,
                "time_to_threshold_eval_episodes_delta_vs_scratch": time_delta_episodes,
                "log_adaptation_gain_vs_scratch_steps": log_adaptation_gain_steps,
                "log_adaptation_gain_vs_scratch_episodes": log_adaptation_gain_episodes,
                "scratch_time_to_threshold_eval_steps": scratch_steps,
                "scratch_time_to_threshold_eval_episodes": scratch_episodes,
                "scratch_final_success_mean": scratch_final_success,
                "scratch_ref_num_runs_total": scratch_ref_num_runs_total,
                "scratch_ref_num_runs_valid": scratch_ref_num_runs_valid,
                "scratch_ref_valid_run_fraction": scratch_ref_valid_run_fraction,
                "scratch_ref_is_stable": scratch_ref_is_stable,
                "initial_gap_vs_scratch": initial_gap_vs_scratch,
            }
        )
    return switch_rows


def compute_detection_metrics(
    switch_episodes: list[int],
    detector_rows: list[dict[str, object]],
    max_delay_episodes: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detection_events = [
        row for row in detector_rows if str(row.get("event_type")) == "detected"
    ]
    matched_event_indices: set[int] = set()
    metrics_rows: list[dict[str, object]] = []

    for switch_episode in switch_episodes:
        matched_idx = None
        matched_event = None
        for event_idx, row in enumerate(detection_events):
            if event_idx in matched_event_indices:
                continue
            detected_episode = int(row["episode"])
            if detected_episode >= switch_episode + 1 and detected_episode <= switch_episode + max_delay_episodes:
                matched_idx = event_idx
                matched_event = row
                break

        if matched_idx is None or matched_event is None:
            metrics_rows.append(
                {
                    "switch_episode": int(switch_episode + 1),
                    "detected": 0,
                    "delay_episodes": None,
                    "detected_episode": None,
                }
            )
            continue

        matched_event_indices.add(matched_idx)
        metrics_rows.append(
            {
                "switch_episode": int(switch_episode + 1),
                "detected": 1,
                "delay_episodes": int(int(matched_event["episode"]) - switch_episode),
                "detected_episode": int(matched_event["episode"]),
            }
        )

    false_alarms = [
        row for event_idx, row in enumerate(detection_events) if event_idx not in matched_event_indices
    ]
    return metrics_rows, false_alarms
