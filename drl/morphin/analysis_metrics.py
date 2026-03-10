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


def compute_switch_metrics(
    episode_rows: list[dict[str, object]],
    switch_episodes: list[int],
    recovery_window: int,
    threshold: float,
    ema_alpha: float,
    scratch_refs: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, object]]:
    if not episode_rows:
        return []
    success = np.asarray([float(row["success"]) for row in episode_rows], dtype=np.float32)
    reward = np.asarray([float(row["episode_return"]) for row in episode_rows], dtype=np.float32)
    ema_success = ema(success, alpha=ema_alpha)

    switch_rows: list[dict[str, object]] = []
    for switch_idx, switch_episode in enumerate(switch_episodes):
        if switch_episode >= len(episode_rows):
            continue
        segment_end = switch_episodes[switch_idx + 1] if switch_idx + 1 < len(switch_episodes) else len(episode_rows)
        task_id = str(episode_rows[switch_episode]["task_id"])
        post_success = success[switch_episode:min(segment_end, switch_episode + recovery_window)]
        post_reward = reward[switch_episode:min(segment_end, switch_episode + recovery_window)]
        segment_ema = ema_success[switch_episode:segment_end]
        recovery_auc = float(np.mean(post_success)) if post_success.size else math.nan
        reward_auc = float(np.mean(post_reward)) if post_reward.size else math.nan
        threshold_index = first_threshold_index(segment_ema, threshold=threshold)
        time_to_threshold = None if threshold_index is None else int(threshold_index + 1)
        scratch_time = None
        scratch_final_success = None
        adaptation_gain = math.nan
        shock_drop = math.nan
        if scratch_refs and task_id in scratch_refs:
            scratch_time = scratch_refs[task_id].get("time_to_threshold")
            scratch_final_success = scratch_refs[task_id].get("final_seen_success_mean")
            if scratch_time is not None and time_to_threshold and time_to_threshold > 0:
                adaptation_gain = float(scratch_time) / float(time_to_threshold)
            if scratch_final_success is not None and post_success.size:
                shock_drop = float(scratch_final_success) - float(post_success[0])

        switch_rows.append(
            {
                "switch_episode": int(switch_episode + 1),
                "task_id": task_id,
                "segment_end_episode": int(segment_end),
                "segment_length_episodes": int(segment_end - switch_episode),
                "shock_drop": shock_drop,
                "recovery_auc_success": recovery_auc,
                "recovery_auc_return": reward_auc,
                "time_to_threshold": time_to_threshold,
                "adaptation_gain_vs_scratch": adaptation_gain,
                "scratch_time_to_threshold": scratch_time,
                "scratch_final_success_mean": scratch_final_success,
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
