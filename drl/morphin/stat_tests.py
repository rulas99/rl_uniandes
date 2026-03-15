"""Mann-Whitney U tests for pairwise method comparison across MORPHIN experiment runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from scipy.stats import mannwhitneyu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical significance tests for MORPHIN results")
    parser.add_argument("--runs-index-csv", type=str, required=True)
    parser.add_argument("--baseline", type=str, default="ddqn_vanilla")
    parser.add_argument(
        "--metrics",
        type=str,
        default="switch_recovery_auc_success_eval_mean,new_task_recovery_auc_success_eval_mean,final_seen_success_mean,current_task_final_success",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def safe_float(value: object) -> float | None:
    if value in ("", None, "nan", "None"):
        return None
    try:
        val = float(value)
        return None if math.isnan(val) else val
    except (ValueError, TypeError):
        return None


def extract_values(rows: list[dict[str, str]], method: str, metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if row.get("method") != method:
            continue
        val = safe_float(row.get(metric))
        if val is not None:
            values.append(val)
    return values


def main() -> int:
    args = parse_args()
    with open(args.runs_index_csv) as handle:
        rows = list(csv.DictReader(handle))

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    methods = sorted({row["method"] for row in rows if row.get("method")})
    baseline = args.baseline

    baseline_present = baseline in methods
    if not baseline_present:
        print(f"Warning: baseline '{baseline}' not found in runs. Available: {methods}")

    all_results: list[dict[str, object]] = []
    num_comparisons = 0

    for metric in metrics:
        baseline_vals = extract_values(rows, baseline, metric) if baseline_present else []
        for method in methods:
            if method == baseline:
                continue
            method_vals = extract_values(rows, method, metric)
            if len(method_vals) < 3 or len(baseline_vals) < 3:
                all_results.append({
                    "metric": metric,
                    "method": method,
                    "baseline": baseline,
                    "n_method": len(method_vals),
                    "n_baseline": len(baseline_vals),
                    "mean_method": sum(method_vals) / len(method_vals) if method_vals else None,
                    "mean_baseline": sum(baseline_vals) / len(baseline_vals) if baseline_vals else None,
                    "U": None,
                    "p_value": None,
                    "significant": None,
                    "note": "insufficient_samples",
                })
                continue
            num_comparisons += 1
            stat, p_value = mannwhitneyu(method_vals, baseline_vals, alternative="two-sided")
            all_results.append({
                "metric": metric,
                "method": method,
                "baseline": baseline,
                "n_method": len(method_vals),
                "n_baseline": len(baseline_vals),
                "mean_method": sum(method_vals) / len(method_vals),
                "mean_baseline": sum(baseline_vals) / len(baseline_vals),
                "delta": sum(method_vals) / len(method_vals) - sum(baseline_vals) / len(baseline_vals),
                "U": stat,
                "p_value": p_value,
                "significant_raw": p_value < args.alpha,
                "note": "ok",
            })

    # Apply Bonferroni correction
    bonferroni_alpha = args.alpha / max(1, num_comparisons)
    for result in all_results:
        if result.get("p_value") is not None:
            result["bonferroni_alpha"] = bonferroni_alpha
            result["significant_bonferroni"] = result["p_value"] < bonferroni_alpha

    output = {
        "baseline": baseline,
        "alpha": args.alpha,
        "num_comparisons": num_comparisons,
        "bonferroni_alpha": bonferroni_alpha if num_comparisons > 0 else None,
        "results": all_results,
    }

    output_text = json.dumps(output, indent=2)
    print(output_text)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
