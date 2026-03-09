from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Este script requiere pandas. Instala con: python -m pip install pandas"
    ) from exc


def safe_float(value: Any) -> float:
    try:
        out = float(value)
        if math.isnan(out):
            return float("nan")
        return out
    except Exception:
        return float("nan")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except Exception:
        return pd.DataFrame()


def collect_run_dirs(root_dir: Path) -> List[Path]:
    return sorted({p.parent for p in root_dir.rglob("summary.json")})


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def infer_ablation_id(args: Dict[str, Any]) -> str:
    append_task_id = bool(args.get("append_task_id", False))
    adapter_enabled = bool(args.get("adapter_enabled", False))
    multi_head_enabled = bool(args.get("multi_head_enabled", False))
    task_conditioning = str(args.get("task_conditioning", "ignore"))
    adapter_rank = int(args.get("adapter_rank", 0) or 0)
    adapter_alpha = int(float(args.get("adapter_alpha", 0) or 0))

    if not append_task_id:
        return "a00_shared_vanilla_no_taskid_no_multihead"
    if not adapter_enabled and not multi_head_enabled and task_conditioning == "concat":
        return "a01_shared_taskid_concat_no_multihead"
    if not adapter_enabled and multi_head_enabled and task_conditioning == "concat":
        return "a02_shared_taskid_concat_multihead_delayed"
    if adapter_enabled and not multi_head_enabled:
        return f"a03_like_adapters_taskid_concat_no_multihead_r{adapter_rank:02d}_a{adapter_alpha:02d}"
    if adapter_enabled and multi_head_enabled:
        return f"a04_like_adapters_taskid_concat_multihead_delayed_r{adapter_rank:02d}_a{adapter_alpha:02d}"
    return "unknown"


def summarize_train_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    out: Dict[str, Any] = {
        "train_health_rows": int(len(df)),
        "train_best_rolling_success": safe_float(df["success_rate"].max()) if "success_rate" in df else float("nan"),
        "train_last_rolling_success": safe_float(df["success_rate"].iloc[-1]) if "success_rate" in df else float("nan"),
        "train_best_rolling_return": safe_float(df["mean_return"].max()) if "mean_return" in df else float("nan"),
        "train_last_rolling_return": safe_float(df["mean_return"].iloc[-1]) if "mean_return" in df else float("nan"),
        "train_last_mean_ep_len": safe_float(df["mean_ep_len"].iloc[-1]) if "mean_ep_len" in df else float("nan"),
        "train_last_action_mean": safe_float(df["action_mean"].iloc[-1]) if "action_mean" in df else float("nan"),
        "train_last_action_std": safe_float(df["action_std"].iloc[-1]) if "action_std" in df else float("nan"),
    }
    action_cols = [c for c in df.columns if c.startswith("action_p_")]
    if action_cols:
        last = df.iloc[-1]
        probs = {c: safe_float(last[c]) for c in action_cols}
        dominant_action = max(probs, key=lambda k: probs[k])
        out["dominant_action_prob_col"] = dominant_action
        out["dominant_action_prob"] = safe_float(probs[dominant_action])
    return out


def summarize_tb_scalars(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "tag" not in df.columns:
        return {}
    out: Dict[str, Any] = {}

    def tag_stats(tag: str) -> pd.DataFrame:
        return df[df["tag"] == tag].sort_values("step")

    approx = tag_stats("train/approx_kl")
    if not approx.empty:
        idx = approx["value"].idxmax()
        out["tb_max_approx_kl"] = safe_float(approx.loc[idx, "value"])
        out["tb_step_max_approx_kl"] = safe_int(approx.loc[idx, "step"])
        out["tb_final_approx_kl"] = safe_float(approx["value"].iloc[-1])

    clip = tag_stats("train/clip_fraction")
    if not clip.empty:
        out["tb_max_clip_fraction"] = safe_float(clip["value"].max())
        out["tb_final_clip_fraction"] = safe_float(clip["value"].iloc[-1])

    ev = tag_stats("train/explained_variance")
    if not ev.empty:
        out["tb_min_explained_variance"] = safe_float(ev["value"].min())
        out["tb_final_explained_variance"] = safe_float(ev["value"].iloc[-1])

    vloss = tag_stats("train/value_loss")
    if not vloss.empty:
        out["tb_max_value_loss"] = safe_float(vloss["value"].max())
        out["tb_final_value_loss"] = safe_float(vloss["value"].iloc[-1])

    return out


def summarize_eval_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    out: Dict[str, Any] = {
        "eval_rows": int(len(df)),
        "num_phases": int(df["phase"].max()) if "phase" in df else 0,
    }
    if "phase" in df and "success_rate" in df:
        final_phase = int(df["phase"].max())
        final_df = df[df["phase"] == final_phase]
        out["final_phase_avg_success"] = safe_float(final_df["success_rate"].mean())
        out["final_phase_avg_return"] = safe_float(final_df["mean_return"].mean()) if "mean_return" in final_df else float("nan")
    if {"phase", "trained_task", "eval_task", "success_rate"}.issubset(df.columns):
        diag_vals: List[float] = []
        final_phase = int(df["phase"].max())
        final_df = df[df["phase"] == final_phase]
        for phase in sorted(df["phase"].dropna().unique()):
            phase_df = df[df["phase"] == phase]
            trained = str(phase_df["trained_task"].iloc[0])
            sel = phase_df[phase_df["eval_task"] == trained]
            if not sel.empty:
                diag_vals.append(safe_float(sel["success_rate"].iloc[0]))
        out["eval_diagonal_success_from_csv"] = safe_float(sum(diag_vals) / len(diag_vals)) if diag_vals else float("nan")
        finals = []
        for task in sorted(final_df["eval_task"].dropna().unique()):
            sel = final_df[final_df["eval_task"] == task]
            if not sel.empty:
                value = safe_float(sel["success_rate"].iloc[0])
                out[f"final_success__{task}"] = value
                finals.append(value)
        if finals:
            out["final_success_mean_from_csv"] = safe_float(sum(finals) / len(finals))
    return out


def build_warning_flags(row: Dict[str, Any]) -> str:
    flags: List[str] = []
    status = str(row.get("status", "ok"))
    if status not in {"ok", ""}:
        flags.append(status)
    max_kl = safe_float(row.get("tb_max_approx_kl", float("nan")))
    if not math.isnan(max_kl) and max_kl >= 0.10:
        flags.append("kl_spike")
    final_s = safe_float(row.get("avg_final_success_rate", float("nan")))
    diag_s = safe_float(row.get("avg_diagonal_success_rate", float("nan")))
    if not math.isnan(final_s) and not math.isnan(diag_s) and (diag_s - final_s) >= 0.25:
        flags.append("strong_forgetting_gap")
    dom = safe_float(row.get("dominant_action_prob", float("nan")))
    if not math.isnan(dom) and dom >= 0.90:
        flags.append("action_collapse")
    ev = safe_float(row.get("tb_min_explained_variance", float("nan")))
    if not math.isnan(ev) and ev <= -1.0:
        flags.append("critic_instability")
    return ";".join(flags)


def concat_long_with_run_info(run_info: Dict[str, Any], df: pd.DataFrame, prefix_cols: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in reversed(list(prefix_cols)):
        out.insert(0, col, run_info.get(col))
    return out


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def dataframe_to_markdown_like(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        if df.empty:
            return "(sin filas)"
        cols = [str(c) for c in df.columns]
        rows = [[str(x) for x in row] for row in df.fillna("").astype(str).values.tolist()]
        widths = [len(col) for col in cols]
        for row in rows:
            for idx, value in enumerate(row):
                widths[idx] = max(widths[idx], len(value))

        def fmt_row(values: List[str]) -> str:
            return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

        header = fmt_row(cols)
        sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(cols))) + " |"
        body = [fmt_row(row) for row in rows]
        return "\n".join([header, sep] + body)


def manifest_value(manifest_row: Optional[pd.Series], key: str, default: Any = "") -> Any:
    if manifest_row is None or manifest_row.empty:
        return default
    value = manifest_row.get(key, default)
    if pd.isna(value):
        return default
    return value


def build_manifest_only_record(manifest_row: pd.Series) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "planned_index": safe_int(manifest_value(manifest_row, "planned_index", -1), default=-1),
        "matrix_group": str(manifest_value(manifest_row, "matrix_group", "")),
        "ablation_id": str(manifest_value(manifest_row, "ablation_id", "unknown")),
        "seed": safe_int(manifest_value(manifest_row, "seed", -1), default=-1),
        "description": str(manifest_value(manifest_row, "description", "")),
        "hypothesis": str(manifest_value(manifest_row, "hypothesis", "")),
        "status": str(manifest_value(manifest_row, "status", "missing")),
        "exit_code": safe_int(manifest_value(manifest_row, "exit_code", 0), default=0),
        "output_base_dir": str(manifest_value(manifest_row, "output_base_dir", "")),
        "run_dir": str(manifest_value(manifest_row, "run_dir", "")),
        "summary_json": str(manifest_value(manifest_row, "summary_json", "")),
        "config_json": str(manifest_value(manifest_row, "config_json", "")),
        "eval_metrics_csv": str(manifest_value(manifest_row, "eval_metrics_csv", "")),
        "train_metrics_csv": str(manifest_value(manifest_row, "train_metrics_csv", "")),
        "tb_scalars_export_csv": str(manifest_value(manifest_row, "tb_scalars_export_csv", "")),
        "train_monitor_csv": str(manifest_value(manifest_row, "train_monitor_csv", "")),
        "eval_monitor_csv": str(manifest_value(manifest_row, "eval_monitor_csv", "")),
        "console_log": str(manifest_value(manifest_row, "console_log", "")),
        "command": str(manifest_value(manifest_row, "command", "")),
    }
    record["warning_flags"] = build_warning_flags(record)
    return record


def discover_run_record(run_dir: Path, manifest_row: Optional[pd.Series] = None) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    summary = load_json(summary_path)
    config = load_json(config_path)
    args = config.get("args", {})

    record: Dict[str, Any] = {
        "planned_index": safe_int(manifest_value(manifest_row, "planned_index", -1), default=-1),
        "matrix_group": str(manifest_value(manifest_row, "matrix_group", "")),
        "ablation_id": str(manifest_value(manifest_row, "ablation_id", infer_ablation_id(args))),
        "seed": safe_int(manifest_value(manifest_row, "seed", args.get("seed", -1)), default=safe_int(args.get("seed", -1), -1)),
        "description": str(manifest_value(manifest_row, "description", "")),
        "hypothesis": str(manifest_value(manifest_row, "hypothesis", "")),
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "config_json": str(config_path),
        "eval_metrics_csv": str(run_dir / "eval_metrics.csv"),
        "train_metrics_csv": str(run_dir / "train_metrics.csv"),
        "tb_scalars_export_csv": str(run_dir / "tb_scalars_export.csv"),
        "train_monitor_csv": str(run_dir / "train.monitor.csv"),
        "eval_monitor_csv": str(run_dir / "eval.monitor.csv"),
        "console_log": str(manifest_value(manifest_row, "console_log", "")),
        "command": str(manifest_value(manifest_row, "command", "")),
        "output_base_dir": str(manifest_value(manifest_row, "output_base_dir", str(run_dir.parent))),
        "status": str(manifest_value(manifest_row, "status", "ok")),
        "exit_code": safe_int(manifest_value(manifest_row, "exit_code", 0), default=0),
        "task_preset": summary.get("task_preset", args.get("task_preset")),
        "mode": summary.get("mode", args.get("mode")),
        "obs_mode": summary.get("obs_mode", args.get("obs_mode")),
        "append_task_id": bool(args.get("append_task_id", False)),
        "task_conditioning": str(args.get("task_conditioning", "ignore")),
        "adapter_enabled": bool(summary.get("adapter_enabled", args.get("adapter_enabled", False))),
        "multi_head_enabled": bool(summary.get("multi_head_enabled", args.get("multi_head_enabled", False))),
        "adapter_rank": args.get("adapter_rank"),
        "adapter_alpha": args.get("adapter_alpha"),
        "adapter_warmup_tasks": args.get("adapter_warmup_tasks"),
        "multi_head_warmup_tasks": args.get("multi_head_warmup_tasks"),
        "steps_per_task": args.get("steps_per_task"),
        "eval_episodes": args.get("eval_episodes"),
        "ppo_learning_rate": args.get("ppo_learning_rate"),
        "ppo_n_steps": args.get("ppo_n_steps"),
        "ppo_batch_size": args.get("ppo_batch_size"),
        "ppo_n_epochs": args.get("ppo_n_epochs"),
        "ppo_clip_range": args.get("ppo_clip_range"),
        "ppo_clip_range_vf": args.get("ppo_clip_range_vf"),
        "ppo_ent_coef": args.get("ppo_ent_coef"),
        "ppo_target_kl": args.get("ppo_target_kl"),
        "timesteps": summary.get("timesteps"),
        "elapsed_sec": summary.get("elapsed_sec"),
        "avg_diagonal_success_rate": summary.get("avg_diagonal_success_rate"),
        "avg_final_success_rate": summary.get("avg_final_success_rate", summary.get("avg_success_rate")),
        "avg_forgetting": summary.get("avg_forgetting"),
        "avg_forgetting_learned_only": summary.get("avg_forgetting_learned_only"),
        "phase_success_matrix_json": json.dumps(summary.get("phase_success_matrix", {}), sort_keys=True),
    }

    eval_df = read_optional_csv(run_dir / "eval_metrics.csv")
    train_df = read_optional_csv(run_dir / "train_metrics.csv")
    tb_df = read_optional_csv(run_dir / "tb_scalars_export.csv")

    record.update(summarize_eval_metrics(eval_df))
    record.update(summarize_train_metrics(train_df))
    record.update(summarize_tb_scalars(tb_df))
    record["warning_flags"] = build_warning_flags(record)
    return record


def aggregate_group_stats(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty or "ablation_id" not in runs_df.columns:
        return pd.DataFrame()

    counts_df = (
        runs_df.groupby("ablation_id", dropna=False)
        .agg(
            matrix_group=("matrix_group", "first"),
            description=("description", "first"),
            hypothesis=("hypothesis", "first"),
            planned_runs=("ablation_id", "size"),
            ok_runs=("status", lambda s: int((s == "ok").sum())),
            failed_runs=("status", lambda s: int((s != "ok").sum())),
        )
        .reset_index()
    )

    ok_df = runs_df[runs_df["status"] == "ok"].copy()
    if ok_df.empty:
        return counts_df.sort_values(["matrix_group", "ablation_id"], na_position="last")

    numeric_cols = [
        "avg_diagonal_success_rate",
        "avg_final_success_rate",
        "avg_forgetting",
        "avg_forgetting_learned_only",
        "final_phase_avg_success",
        "final_phase_avg_return",
        "train_best_rolling_success",
        "tb_max_approx_kl",
        "tb_max_clip_fraction",
        "tb_min_explained_variance",
        "elapsed_sec",
    ]
    existing = [c for c in numeric_cols if c in ok_df.columns]
    if not existing:
        return counts_df.sort_values(["matrix_group", "ablation_id"], na_position="last")
    agg = (
        ok_df.groupby("ablation_id", dropna=False)[existing]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    agg.columns = ["__".join([c for c in col if c]).strip("_") for col in agg.columns.to_flat_index()]
    merged = counts_df.merge(agg, on="ablation_id", how="left")
    sort_col = "avg_final_success_rate__mean" if "avg_final_success_rate__mean" in merged.columns else "ablation_id"
    return merged.sort_values(sort_col, ascending=False, na_position="last")


def render_report(runs_df: pd.DataFrame, groups_df: pd.DataFrame, output_path: Path) -> None:
    lines: List[str] = []
    lines.append("# MiniGrid continual RL ablation report")
    lines.append("")
    lines.append(f"Runs analizados: {len(runs_df)}")
    ok_runs = int((runs_df["status"] == "ok").sum()) if not runs_df.empty and "status" in runs_df else 0
    failed_runs = int((runs_df["status"] != "ok").sum()) if not runs_df.empty and "status" in runs_df else 0
    lines.append(f"Runs OK: {ok_runs}")
    lines.append(f"Runs no OK: {failed_runs}")
    lines.append("")

    if not groups_df.empty:
        lines.append("## Ranking por avg_final_success_rate")
        lines.append("")
        cols = [
            "ablation_id",
            "matrix_group",
            "ok_runs",
            "failed_runs",
            "avg_final_success_rate__mean",
            "avg_final_success_rate__std",
            "avg_diagonal_success_rate__mean",
            "avg_forgetting__mean",
            "avg_forgetting_learned_only__mean",
            "tb_max_approx_kl__mean",
        ]
        present = [c for c in cols if c in groups_df.columns]
        lines.append(dataframe_to_markdown_like(groups_df[present]))
        lines.append("")

    if not runs_df.empty:
        failed = runs_df[runs_df["status"].fillna("") != "ok"]
        lines.append("## Runs fallidos o incompletos")
        lines.append("")
        if failed.empty:
            lines.append("Sin fallas registradas.")
        else:
            cols = [
                "ablation_id",
                "seed",
                "status",
                "exit_code",
                "console_log",
                "run_dir",
            ]
            present = [c for c in cols if c in failed.columns]
            lines.append(dataframe_to_markdown_like(failed[present]))
        lines.append("")

        flagged = runs_df[runs_df["warning_flags"].fillna("") != ""]
        lines.append("## Runs con alertas")
        lines.append("")
        if flagged.empty:
            lines.append("Sin alertas heurísticas.")
        else:
            cols = [
                "ablation_id",
                "seed",
                "status",
                "avg_final_success_rate",
                "avg_diagonal_success_rate",
                "avg_forgetting",
                "tb_max_approx_kl",
                "dominant_action_prob",
                "warning_flags",
                "run_dir",
            ]
            present = [c for c in cols if c in flagged.columns]
            lines.append(dataframe_to_markdown_like(flagged[present]))
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser("Aggregate MiniGrid ablation results")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = read_manifest(Path(args.manifest)) if args.manifest else pd.DataFrame()
    discovered_run_dirs = collect_run_dirs(root_dir)
    discovered_set = {str(p) for p in discovered_run_dirs}

    run_records: List[Dict[str, Any]] = []
    eval_long_frames: List[pd.DataFrame] = []
    train_long_frames: List[pd.DataFrame] = []
    tb_long_frames: List[pd.DataFrame] = []
    processed_run_dirs: set[str] = set()

    prefix_cols = ["matrix_group", "ablation_id", "seed", "status", "run_dir"]

    if not manifest_df.empty:
        for _, manifest_row in manifest_df.sort_values("planned_index", na_position="last").iterrows():
            run_dir_str = str(manifest_value(manifest_row, "run_dir", "")).strip()
            summary_json = str(manifest_value(manifest_row, "summary_json", "")).strip()
            has_artifacts = bool(run_dir_str) and (
                Path(summary_json).exists() if summary_json else (Path(run_dir_str) / "summary.json").exists()
            )

            if has_artifacts:
                run_dir = Path(run_dir_str)
                try:
                    run_info = discover_run_record(run_dir, manifest_row)
                except Exception as exc:
                    run_info = build_manifest_only_record(manifest_row)
                    run_info["status"] = "parse_failed"
                    run_info["warning_flags"] = build_warning_flags(run_info) + f";parse_failed:{exc.__class__.__name__}"
                processed_run_dirs.add(str(run_dir))
                eval_df = read_optional_csv(run_dir / "eval_metrics.csv")
                train_df = read_optional_csv(run_dir / "train_metrics.csv")
                tb_df = read_optional_csv(run_dir / "tb_scalars_export.csv")
                if not eval_df.empty:
                    eval_long_frames.append(concat_long_with_run_info(run_info, eval_df, prefix_cols))
                if not train_df.empty:
                    train_long_frames.append(concat_long_with_run_info(run_info, train_df, prefix_cols))
                if not tb_df.empty:
                    tb_long_frames.append(concat_long_with_run_info(run_info, tb_df, prefix_cols))
            else:
                run_info = build_manifest_only_record(manifest_row)

            run_records.append(run_info)

    for run_dir in discovered_run_dirs:
        if str(run_dir) in processed_run_dirs:
            continue
        try:
            run_info = discover_run_record(run_dir, None)
        except Exception as exc:
            run_info = {
                "run_dir": str(run_dir),
                "status": "parse_failed",
                "warning_flags": f"parse_failed:{exc.__class__.__name__}",
            }
        run_records.append(run_info)

        eval_df = read_optional_csv(run_dir / "eval_metrics.csv")
        train_df = read_optional_csv(run_dir / "train_metrics.csv")
        tb_df = read_optional_csv(run_dir / "tb_scalars_export.csv")
        if not eval_df.empty:
            eval_long_frames.append(concat_long_with_run_info(run_info, eval_df, prefix_cols))
        if not train_df.empty:
            train_long_frames.append(concat_long_with_run_info(run_info, train_df, prefix_cols))
        if not tb_df.empty:
            tb_long_frames.append(concat_long_with_run_info(run_info, tb_df, prefix_cols))

    runs_df = pd.DataFrame(run_records)
    if not runs_df.empty:
        sort_cols = [c for c in ["matrix_group", "ablation_id", "seed", "planned_index"] if c in runs_df.columns]
        if sort_cols:
            runs_df = runs_df.sort_values(sort_cols, na_position="last")

    eval_long_df = pd.concat(eval_long_frames, ignore_index=True) if eval_long_frames else pd.DataFrame()
    train_long_df = pd.concat(train_long_frames, ignore_index=True) if train_long_frames else pd.DataFrame()
    tb_long_df = pd.concat(tb_long_frames, ignore_index=True) if tb_long_frames else pd.DataFrame()
    groups_df = aggregate_group_stats(runs_df)

    failed_runs_df = runs_df[runs_df["status"] != "ok"].copy() if not runs_df.empty and "status" in runs_df.columns else pd.DataFrame()

    save_dataframe(runs_df, output_dir / "ablation_runs_index.csv")
    save_dataframe(eval_long_df, output_dir / "ablation_eval_metrics_long.csv")
    save_dataframe(train_long_df, output_dir / "ablation_train_metrics_long.csv")
    save_dataframe(tb_long_df, output_dir / "ablation_tb_scalars_long.csv")
    save_dataframe(groups_df, output_dir / "ablation_group_summary.csv")
    save_dataframe(failed_runs_df, output_dir / "ablation_failed_runs.csv")

    bundle = {
        "root_dir": str(root_dir),
        "manifest_provided": bool(args.manifest),
        "num_runs": int(len(runs_df)),
        "num_ok_runs": int((runs_df["status"] == "ok").sum()) if not runs_df.empty and "status" in runs_df.columns else 0,
        "num_non_ok_runs": int((runs_df["status"] != "ok").sum()) if not runs_df.empty and "status" in runs_df.columns else 0,
        "num_discovered_run_dirs": int(len(discovered_set)),
        "outputs": {
            "ablation_runs_index_csv": str(output_dir / "ablation_runs_index.csv"),
            "ablation_eval_metrics_long_csv": str(output_dir / "ablation_eval_metrics_long.csv"),
            "ablation_train_metrics_long_csv": str(output_dir / "ablation_train_metrics_long.csv"),
            "ablation_tb_scalars_long_csv": str(output_dir / "ablation_tb_scalars_long.csv"),
            "ablation_group_summary_csv": str(output_dir / "ablation_group_summary.csv"),
            "ablation_failed_runs_csv": str(output_dir / "ablation_failed_runs.csv"),
            "ablation_report_md": str(output_dir / "ablation_report.md"),
        },
    }
    (output_dir / "analysis_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    render_report(runs_df, groups_df, output_dir / "ablation_report.md")

    print(json.dumps(bundle, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
