"""
thesis_stats.py  —  Statistical analysis of overnight thesis results.

Outputs (all in SESSION/analysis/thesis/):
  tables/
    tab_group_summary.csv / .tex
    tab_benchmark_method.csv / .tex
    tab_retention_by_benchmark.csv / .tex
    tab_switch_type.csv / .tex
    tab_statistical_tests.csv / .tex
  export/
    switch_metrics_clean.csv       — per-switch rows for seaborn plots
    eval_curves_clean.csv          — per-episode eval for learning curves
    retention_matrix.csv           — benchmark × method × task retention grid

Usage:
    python3 thesis_stats.py SESSION_DIR
    # e.g.
    # python3 thesis_stats.py logs/morphin_gridworld/thesis_overnight/session_20260317_220032
"""

from __future__ import annotations
import csv, math, sys, os, json
from pathlib import Path
from collections import defaultdict
from typing import Sequence

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_METHODS_ORDER = [
    "ddqn_vanilla",
    "oracle_reset",
    "oracle_segmented",
    "oracle_segmented_distill_l001",
    "der_plus_plus",
]
METHODS_ORDER = list(DEFAULT_METHODS_ORDER)
METHOD_LABELS = {
    "ddqn_vanilla":                   "DDQN Vanilla",
    "oracle_reset":                   "Oracle Reset",
    "oracle_segmented":               "Oracle Seg.",
    "oracle_segmented_distill_l001":  "Oracle Seg. + Distill (λ=0.001)",
    "der_plus_plus":                  "DER++",
}
METHOD_SHORT_LABELS = {
    "ddqn_vanilla": "V",
    "oracle_reset": "R",
    "oracle_segmented": "S",
    "oracle_segmented_distill_l001": "S+D",
    "der_plus_plus": "DER++",
}
DEFAULT_COMPARISONS = [
    ("ddqn_vanilla", "oracle_reset"),
    ("ddqn_vanilla", "oracle_segmented"),
    ("oracle_segmented", "oracle_segmented_distill_l001"),
    ("oracle_segmented", "der_plus_plus"),
    ("oracle_segmented_distill_l001", "der_plus_plus"),
]
BENCH_ORDER = [
    "gw9_goal_balanced_ab_v1",
    "gw9_goal_balanced_ac_v1",
    "gw9_goal_balanced_aba_v1",
    "gw9_goal_balanced_abc_v1",
]
BENCH_LABELS = {
    "gw9_goal_balanced_ab_v1":  "AB",
    "gw9_goal_balanced_ac_v1":  "AC",
    "gw9_goal_balanced_aba_v1": "ABA",
    "gw9_goal_balanced_abc_v1": "ABC",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _f(s):
    try:
        return float(s)
    except Exception:
        return float("nan")


def mean_se_n(vals: list[float]):
    vals = [v for v in vals if not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = sum(vals) / n
    if n == 1:
        return mu, float("nan"), 1
    var = sum((x - mu) ** 2 for x in vals) / (n - 1)
    return mu, math.sqrt(var / n), n


def fmt(mu, se, n, dp=3):
    if math.isnan(mu):
        return "—"
    if math.isnan(se):
        return f"{mu:.{dp}f}"
    return f"{mu:.{dp}f} ± {se:.{dp}f}"


def latex_bold(s: str) -> str:
    return f"\\textbf{{{s}}}"


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def method_short_label(method: str) -> str:
    return METHOD_SHORT_LABELS.get(method, method)


def tex_method_label(method: str) -> str:
    return method_label(method).replace(
        "Oracle Seg. + Distill (λ=0.001)",
        r"Seg.+Distill $\lambda$=0.001",
    )


def resolve_methods_order(root: Path, rows: Sequence[dict]) -> list[str]:
    present_methods = []
    for row in rows:
        method = str(row.get("method", "")).strip()
        if method and method not in present_methods:
            present_methods.append(method)

    ordered: list[str] = []
    session_config = root / "session_config.json"
    if session_config.exists():
        try:
            config = json.loads(session_config.read_text())
            configured = str(config.get("methods_csv", ""))
            for method in configured.split(","):
                method = method.strip()
                if method and method in present_methods and method not in ordered:
                    ordered.append(method)
        except Exception:
            pass

    for method in DEFAULT_METHODS_ORDER:
        if method in present_methods and method not in ordered:
            ordered.append(method)

    for method in present_methods:
        if method not in ordered:
            ordered.append(method)
    return ordered


def infer_seed_count(rows: Sequence[dict], key: str = "seed") -> int:
    seeds = {str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()}
    return len(seeds)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────────────────────
def mannwhitney(a: list[float], b: list[float]):
    """Returns (statistic, p_value). Falls back to nan if scipy missing."""
    a = [v for v in a if not math.isnan(v)]
    b = [v for v in b if not math.isnan(v)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        return stat, p
    except Exception:
        return float("nan"), float("nan")


def fisher_exact(n_success_a, n_a, n_success_b, n_b):
    """Fisher's exact test for two proportions. Returns (oddsratio, p)."""
    try:
        from scipy.stats import fisher_exact as _fe
        table = [
            [int(n_success_a), int(n_a - n_success_a)],
            [int(n_success_b), int(n_b - n_success_b)],
        ]
        return _fe(table)
    except Exception:
        return float("nan"), float("nan")


def sig_stars(p):
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "†"
    return "ns"


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(session_dir: str):
    global METHODS_ORDER
    root = Path(session_dir)
    ana = root / "analysis"
    out = ana / "thesis"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "export").mkdir(parents=True, exist_ok=True)

    sw = load_csv(ana / "switch_metrics_long.csv")
    ft = load_csv(ana / "final_task_metrics_long.csv")
    ev = load_csv(ana / "eval_metrics_long.csv")
    runs_idx = load_csv(ana / "runs_index.csv")
    METHODS_ORDER = resolve_methods_order(root, runs_idx or sw or ft or ev)

    # ── Index runs ────────────────────────────────────────────────────────────
    run_meta = {}  # run_dir → {method, benchmark, seed}
    for r in runs_idx:
        run_meta[r["run_dir"]] = r

    # ─────────────────────────────────────────────────────────────────────────
    # 1. SWITCH-LEVEL EXPORT (for seaborn plots)
    # ─────────────────────────────────────────────────────────────────────────
    sw_clean = []
    for r in sw:
        stype = r.get("switch_type", "")
        if stype not in ("new_task", "revisit_task"):
            continue
        try:
            sw_clean.append({
                "method":           r["method"],
                "benchmark":        r["benchmark"],
                "bench_label":      BENCH_LABELS.get(r["benchmark"], r["benchmark"]),
                "method_label":     method_label(r["method"]),
                "seed":             r["seed"],
                "switch_type":      stype,
                "switch_index":     r.get("switch_index", ""),
                "task_id":          r.get("task_id", ""),
                "recovery_auc":     _f(r["recovery_auc_success_eval"]),
                "ttt_steps":        _f(r["time_to_threshold_eval_steps"]),
                "ttt_delta":        _f(r["time_to_threshold_eval_steps_delta_vs_scratch"]),
                "log_gain":         _f(r["log_adaptation_gain_vs_scratch_steps"]),
                "scratch_ttt":      _f(r["scratch_time_to_threshold_eval_steps"]),
            })
        except Exception:
            pass

    _write_csv(out / "export" / "switch_metrics_clean.csv", sw_clean)
    print(f"  switch export: {len(sw_clean)} rows")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. EVAL CURVES EXPORT (for learning curves)
    # ─────────────────────────────────────────────────────────────────────────
    # We want: per-task success rate vs episode, by method × benchmark × seed
    # Also track which "phase" the episode falls in (task_index)
    ev_clean = []
    for r in ev:
        try:
            ev_clean.append({
                "method":        r["method"],
                "benchmark":     r["benchmark"],
                "bench_label":   BENCH_LABELS.get(r["benchmark"], r["benchmark"]),
                "method_label":  method_label(r["method"]),
                "seed":          r["seed"],
                "episode":       int(r["episode"]),
                "global_step":   int(r["global_step"]),
                "eval_scope":    r["eval_scope"],
                "task_id":       r["task_id"],
                "success_rate":  _f(r["success_rate"]),
                "mean_return":   _f(r["mean_return"]),
                "episodes_per_task": int(r["episodes_per_task"]),
                "num_tasks":     int(r["num_tasks"]),
            })
        except Exception:
            pass

    _write_csv(out / "export" / "eval_curves_clean.csv", ev_clean)
    print(f"  eval curves export: {len(ev_clean)} rows")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. RETENTION MATRIX EXPORT
    # ─────────────────────────────────────────────────────────────────────────
    # final_task_metrics_long → benchmark × method × task_id × task_role
    ret_data = defaultdict(list)  # (bench, method, task_role) → [success_rates]
    ret_detailed = []
    for r in ft:
        try:
            val = _f(r["success_rate"])
            key = (r["benchmark"], r["method"], r["task_role"], r["task_id"])
            ret_data[(r["benchmark"], r["method"], r["task_role"])].append(val)
            ret_detailed.append({
                "method":        r["method"],
                "method_label":  method_label(r["method"]),
                "benchmark":     r["benchmark"],
                "bench_label":   BENCH_LABELS.get(r["benchmark"], r["benchmark"]),
                "seed":          r["seed"],
                "task_id":       r["task_id"],
                "task_role":     r["task_role"],
                "success_rate":  val,
            })
        except Exception:
            pass

    _write_csv(out / "export" / "retention_matrix.csv", ret_detailed)
    print(f"  retention matrix export: {len(ret_detailed)} rows")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. TABLE: Group Summary
    # ─────────────────────────────────────────────────────────────────────────
    # From switch_metrics: recovery_auc (new_task), revisit_auc, ttt_delta
    # From final_task: retention
    # From runs_index: all_unique_final_rate
    runs_by_method = defaultdict(list)
    for r in runs_idx:
        runs_by_method[r["method"]].append(r)

    auc_new   = defaultdict(list)
    auc_rev   = defaultdict(list)
    ttt_new   = defaultdict(list)
    loggain   = defaultdict(list)
    ttt_rev   = defaultdict(list)
    for r in sw:
        m = r["method"]
        stype = r.get("switch_type", "")
        try:
            auc = _f(r["recovery_auc_success_eval"])
            td  = _f(r["time_to_threshold_eval_steps_delta_vs_scratch"])
            lg  = _f(r["log_adaptation_gain_vs_scratch_steps"])
            ttt = _f(r["time_to_threshold_eval_steps"])
        except Exception:
            continue
        if stype == "new_task":
            auc_new[m].append(auc); ttt_new[m].append(td); loggain[m].append(lg)
        elif stype == "revisit_task":
            auc_rev[m].append(auc); ttt_rev[m].append(ttt)

    retention_overall = defaultdict(list)  # method → list of 0/1
    for r in runs_idx:
        m = r["method"]
        try:
            retention_overall[m].append(int(_f(r["final_all_unique_tasks_success_ge_threshold"])))
        except Exception:
            pass

    group_rows = []
    for m in METHODS_ORDER:
        mu_auc_n, se_auc_n, n_auc_n = mean_se_n(auc_new[m])
        mu_auc_r, se_auc_r, n_auc_r = mean_se_n(auc_rev[m])
        mu_ttt,   se_ttt,   n_ttt   = mean_se_n(ttt_new[m])
        mu_lg,    se_lg,    _       = mean_se_n(loggain[m])
        ret_vals = retention_overall[m]
        n_ret = len(ret_vals)
        ret_rate = sum(ret_vals) / n_ret if n_ret else float("nan")
        group_rows.append({
            "method": m,
            "method_label": method_label(m),
            "n_runs": n_ret,
            "retention_rate": ret_rate,
            "n_retained": sum(ret_vals),
            "new_task_auc_mean": mu_auc_n,
            "new_task_auc_se": se_auc_n,
            "new_task_auc_n": n_auc_n,
            "revisit_auc_mean": mu_auc_r,
            "revisit_auc_se": se_auc_r,
            "revisit_auc_n": n_auc_r,
            "ttt_delta_mean": mu_ttt,
            "ttt_delta_se": se_ttt,
            "log_gain_mean": mu_lg,
            "log_gain_se": se_lg,
        })

    _write_csv(out / "tables" / "tab_group_summary.csv", group_rows)
    _write_group_summary_tex(group_rows, out / "tables" / "tab_group_summary.tex")
    print("  tab_group_summary done")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. TABLE: Retention by benchmark × method
    # ─────────────────────────────────────────────────────────────────────────
    ret_bench = defaultdict(list)  # (bench, method) → [0/1 per run]
    for r in runs_idx:
        key = (r["benchmark"], r["method"])
        try:
            ret_bench[key].append(int(_f(r["final_all_unique_tasks_success_ge_threshold"])))
        except Exception:
            pass

    prev_task_ret = defaultdict(list)  # (bench, method) → [success_rate per previous_task per seed]
    for r in ft:
        if r["task_role"] != "previous_task":
            continue
        key = (r["benchmark"], r["method"])
        try:
            prev_task_ret[key].append(_f(r["success_rate"]))
        except Exception:
            pass

    bench_rows = []
    for b in BENCH_ORDER:
        for m in METHODS_ORDER:
            vals = ret_bench[(b, m)]
            n = len(vals)
            k = sum(vals)
            prev = prev_task_ret[(b, m)]
            mu_prev, se_prev, _ = mean_se_n(prev)
            bench_rows.append({
                "benchmark": b, "bench_label": BENCH_LABELS[b],
                "method": m, "method_label": method_label(m),
                "n_runs": n, "n_retained": k,
                "retention_rate": k/n if n else float("nan"),
                "prev_task_success_mean": mu_prev,
                "prev_task_success_se": se_prev,
            })

    _write_csv(out / "tables" / "tab_retention_by_benchmark.csv", bench_rows)
    _write_retention_tex(bench_rows, out / "tables" / "tab_retention_by_benchmark.tex")
    print("  tab_retention_by_benchmark done")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. TABLE: Benchmark × Method — full metrics
    # ─────────────────────────────────────────────────────────────────────────
    auc_new_bm  = defaultdict(list)
    ttt_new_bm  = defaultdict(list)
    logg_bm     = defaultdict(list)
    for r in sw:
        if r.get("switch_type") != "new_task":
            continue
        key = (r["benchmark"], r["method"])
        try:
            auc_new_bm[key].append(_f(r["recovery_auc_success_eval"]))
            ttt_new_bm[key].append(_f(r["time_to_threshold_eval_steps_delta_vs_scratch"]))
            logg_bm[key].append(_f(r["log_adaptation_gain_vs_scratch_steps"]))
        except Exception:
            pass

    bm_rows = []
    for b in BENCH_ORDER:
        for m in METHODS_ORDER:
            key = (b, m)
            mu_a, se_a, n_a = mean_se_n(auc_new_bm[key])
            mu_t, se_t, _   = mean_se_n(ttt_new_bm[key])
            mu_l, se_l, _   = mean_se_n(logg_bm[key])
            rvals = ret_bench[key]; nr = len(rvals); kr = sum(rvals)
            bm_rows.append({
                "benchmark": b, "bench_label": BENCH_LABELS[b],
                "method": m, "method_label": method_label(m),
                "new_task_auc_mean": mu_a, "new_task_auc_se": se_a, "n_switches": n_a,
                "ttt_delta_mean": mu_t, "ttt_delta_se": se_t,
                "log_gain_mean": mu_l, "log_gain_se": se_l,
                "n_runs": nr, "n_retained": kr,
                "retention_rate": kr/nr if nr else float("nan"),
            })

    _write_csv(out / "tables" / "tab_benchmark_method.csv", bm_rows)
    _write_benchmark_method_tex(bm_rows, out / "tables" / "tab_benchmark_method.tex")
    print("  tab_benchmark_method done")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. STATISTICAL TESTS
    # ─────────────────────────────────────────────────────────────────────────
    present_methods = [m for m in METHODS_ORDER if runs_by_method[m]]
    comparisons = [
        (ref, tgt)
        for ref, tgt in DEFAULT_COMPARISONS
        if ref in present_methods and tgt in present_methods
    ]
    if not comparisons and len(present_methods) >= 2:
        comparisons = list(zip(present_methods[:-1], present_methods[1:]))

    test_rows = []

    def add_mw(label, a_vals, b_vals, ref, tgt, bench="ALL"):
        stat, p = mannwhitney(a_vals, b_vals)
        mu_a, se_a, n_a = mean_se_n(a_vals)
        mu_b, se_b, n_b = mean_se_n(b_vals)
        test_rows.append({
            "comparison": f"{ref}__vs__{tgt}",
            "test": "Mann-Whitney U",
            "benchmark": bench,
            "metric": label,
            "ref_method": ref,
            "ref_label": method_label(ref),
            "ref_mean": mu_a, "ref_se": se_a, "ref_n": n_a,
            "tgt_method": tgt,
            "tgt_label": method_label(tgt),
            "tgt_mean": mu_b, "tgt_se": se_b, "tgt_n": n_b,
            "statistic": stat, "p_value": p, "sig": sig_stars(p),
        })

    def add_fisher(label, k_a, n_a, k_b, n_b, ref, tgt, bench="ALL"):
        _, p = fisher_exact(k_a, n_a, k_b, n_b)
        test_rows.append({
            "comparison": f"{ref}__vs__{tgt}",
            "test": "Fisher Exact",
            "benchmark": bench,
            "metric": label,
            "ref_method": ref,
            "ref_label": method_label(ref),
            "ref_mean": k_a/n_a if n_a else float("nan"), "ref_se": float("nan"), "ref_n": n_a,
            "tgt_method": tgt,
            "tgt_label": method_label(tgt),
            "tgt_mean": k_b/n_b if n_b else float("nan"), "tgt_se": float("nan"), "tgt_n": n_b,
            "statistic": float("nan"), "p_value": p, "sig": sig_stars(p),
        })

    for ref, tgt in comparisons:
        add_mw("new_task_auc (all benchmarks)", auc_new[ref], auc_new[tgt], ref, tgt)
        add_mw("revisit_auc (revisit benchmarks)", auc_rev[ref], auc_rev[tgt], ref, tgt)
        add_mw("ttt_delta_vs_scratch (new_task, all)", ttt_new[ref], ttt_new[tgt], ref, tgt)
        add_mw("log_gain_vs_scratch (new_task, all)", loggain[ref], loggain[tgt], ref, tgt)

        r_ref = retention_overall[ref]
        r_tgt = retention_overall[tgt]
        add_fisher(
            "all_unique_retention (all benchmarks)",
            sum(r_ref),
            len(r_ref),
            sum(r_tgt),
            len(r_tgt),
            ref,
            tgt,
        )

        for b in BENCH_ORDER:
            bl = BENCH_LABELS[b]
            add_mw("new_task_auc", auc_new_bm[(b, ref)], auc_new_bm[(b, tgt)], ref, tgt, bench=bl)
            add_mw("ttt_delta", ttt_new_bm[(b, ref)], ttt_new_bm[(b, tgt)], ref, tgt, bench=bl)
            add_mw(
                "prev_task_success",
                prev_task_ret[(b, ref)],
                prev_task_ret[(b, tgt)],
                ref,
                tgt,
                bench=bl,
            )
            k_r = sum(ret_bench[(b, ref)])
            n_r = len(ret_bench[(b, ref)])
            k_t = sum(ret_bench[(b, tgt)])
            n_t = len(ret_bench[(b, tgt)])
            add_fisher("retention_rate", k_r, n_r, k_t, n_t, ref, tgt, bench=bl)

    _write_csv(out / "tables" / "tab_statistical_tests.csv", test_rows)
    _write_tests_tex(test_rows, out / "tables" / "tab_statistical_tests.tex")
    print(f"  tab_statistical_tests done ({len(test_rows)} tests)")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. FINAL TASK TABLE (current_task + previous tasks, per benchmark)
    # ─────────────────────────────────────────────────────────────────────────
    task_role_data = defaultdict(list)  # (bench, method, task_id, task_role) → [success]
    for r in ft:
        key = (r["benchmark"], r["method"], r["task_id"], r["task_role"])
        try:
            task_role_data[key].append(_f(r["success_rate"]))
        except Exception:
            pass

    final_task_rows = []
    for b in BENCH_ORDER:
        for m in METHODS_ORDER:
            tasks_in_bench = sorted(set(
                r["task_id"] for r in ft
                if r["benchmark"] == b and r["method"] == m
            ))
            for tid in tasks_in_bench:
                for role in ("current_task", "previous_task"):
                    vals = task_role_data[(b, m, tid, role)]
                    if not vals:
                        continue
                    mu, se, n = mean_se_n(vals)
                    final_task_rows.append({
                        "benchmark": b, "bench_label": BENCH_LABELS[b],
                        "method": m, "method_label": method_label(m),
                        "task_id": tid,
                        "task_role": role,
                        "success_mean": mu, "success_se": se, "n": n,
                    })

    _write_csv(out / "tables" / "tab_final_tasks.csv", final_task_rows)
    _write_final_tasks_tex(final_task_rows, out / "tables" / "tab_final_tasks.tex")
    print("  tab_final_tasks done")

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Print summary to stdout
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  STATISTICAL SUMMARY  (selected pairwise comparisons)")
    print("=" * 72)
    current_comparison = None
    for t in test_rows:
        if t["benchmark"] != "ALL":
            continue
        comparison = f"{t['ref_label']} vs {t['tgt_label']}"
        if comparison != current_comparison:
            current_comparison = comparison
            print(f"  {comparison}")
        p = t["p_value"]
        p_str = f"p={p:.4f}" if not math.isnan(p) else "p=n/a"
        print(
            f"    {t['metric']:46s}  {t['ref_mean']:.3f} vs {t['tgt_mean']:.3f}  "
            f"{p_str}  {t['sig']}"
        )
    print()
    print("=" * 72)
    print("  PER-BENCHMARK RETENTION RATES")
    print("=" * 72)
    for b in BENCH_ORDER:
        print(f"  {BENCH_LABELS[b]}:")
        for m in METHODS_ORDER:
            if m not in present_methods:
                continue
            vals = ret_bench[(b, m)]
            n = len(vals); k = sum(vals)
            rate = (k / n) if n else float("nan")
            rate_str = f"{rate:.3f}" if not math.isnan(rate) else "nan"
            print(f"    {method_label(m):40s}  {k:2d}/{n:2d} = {rate_str}")
    print()
    print("  Output written to:", str(out))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CSV / LaTeX writers
# ─────────────────────────────────────────────────────────────────────────────
def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_group_summary_tex(rows, path: Path):
    max_runs = max((int(r["n_runs"]) for r in rows), default=0)
    bench_count = len(BENCH_ORDER)
    seed_count = (max_runs // bench_count) if bench_count and max_runs % bench_count == 0 else None
    campaign_desc = (
        f"{seed_count} seeds $\\times$ {bench_count} benchmarks = {max_runs} runs per method."
        if seed_count is not None and max_runs
        else f"Up to {max_runs} runs per method across {bench_count} benchmarks."
    )
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Group summary: {campaign_desc} "
        r"Retention = fraction of runs where all unique tasks $\geq 0.8$ success at end. "
        r"AUC = mean recovery AUC over new-task switches. "
        r"$\Delta$TTT = time-to-threshold relative to scratch (negative = faster).}",
        r"\label{tab:group_summary}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Retention & New-task AUC & Revisit AUC & $\Delta$TTT (steps) & Log-gain \\",
        r"\midrule",
    ]
    best_ret = max(r["retention_rate"] for r in rows)
    best_auc = max(r["new_task_auc_mean"] for r in rows if not math.isnan(r["new_task_auc_mean"]))
    best_rev = max(r["revisit_auc_mean"] for r in rows if not math.isnan(r["revisit_auc_mean"]))
    for r in rows:
        ret_str = f"{r['n_retained']}/{r['n_runs']} ({r['retention_rate']:.2%})" if not math.isnan(r["retention_rate"]) else "—"
        auc_str = fmt(r["new_task_auc_mean"], r["new_task_auc_se"], r["new_task_auc_n"], dp=3)
        rev_str = fmt(r["revisit_auc_mean"], r["revisit_auc_se"], r["revisit_auc_n"], dp=3) if r["revisit_auc_n"] > 0 else "—"
        ttt_str = f"{r['ttt_delta_mean']:+.0f} ± {r['ttt_delta_se']:.0f}" if not math.isnan(r["ttt_delta_mean"]) else "—"
        lg_str  = fmt(r["log_gain_mean"], r["log_gain_se"], 0, dp=3)
        label = tex_method_label(r["method"])
        # Bold best values
        if abs(r["retention_rate"] - best_ret) < 1e-6:
            ret_str = latex_bold(ret_str)
        if not math.isnan(r["new_task_auc_mean"]) and abs(r["new_task_auc_mean"] - best_auc) < 1e-6:
            auc_str = latex_bold(auc_str)
        if not math.isnan(r["revisit_auc_mean"]) and abs(r["revisit_auc_mean"] - best_rev) < 1e-6:
            rev_str = latex_bold(rev_str)
        lines.append(f"  {label} & {ret_str} & {auc_str} & {rev_str} & {ttt_str} & {lg_str} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines))


def _write_retention_tex(rows, path: Path):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Retention by benchmark: proportion of runs where all unique tasks exceed 0.8 "
        r"success threshold at end of sequence (runs retained / total runs). "
        r"Previous-task success shows mean $\pm$ SE over all seeds.}",
        r"\label{tab:retention_benchmark}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Benchmark & Method & Retained runs & Prev.\ task success \\",
        r"\midrule",
    ]
    cur_bench = None
    for r in rows:
        if r["bench_label"] != cur_bench:
            if cur_bench is not None:
                lines.append(r"\midrule")
            cur_bench = r["bench_label"]
        n = r["n_runs"]; k = r["n_retained"]
        ret_str = f"{k}/{n}"
        prev_str = fmt(r["prev_task_success_mean"], r["prev_task_success_se"], 0, dp=3)
        label = tex_method_label(r["method"])
        lines.append(f"  {cur_bench} & {label} & {ret_str} & {prev_str} \\\\")
        cur_bench = r["bench_label"]  # keep for next row
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines))


def _write_benchmark_method_tex(rows, path: Path):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{New-task recovery AUC and time-to-threshold delta vs.\ scratch, "
        r"by benchmark and method. Mean $\pm$ SE. $\Delta$TTT negative = positive forward transfer.}",
        r"\label{tab:benchmark_method}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Benchmark & Method & New-task AUC & $\Delta$TTT & Log-gain \\",
        r"\midrule",
    ]
    cur_bench = None
    for r in rows:
        if r["bench_label"] != cur_bench:
            if cur_bench is not None:
                lines.append(r"\midrule")
            cur_bench = r["bench_label"]
        auc_str = fmt(r["new_task_auc_mean"], r["new_task_auc_se"], 0, dp=3)
        ttt_str = f"{r['ttt_delta_mean']:+.0f}" if not math.isnan(r["ttt_delta_mean"]) else "—"
        lg_str  = fmt(r["log_gain_mean"], r["log_gain_se"], 0, dp=3)
        label = tex_method_label(r["method"])
        lines.append(f"  {cur_bench} & {label} & {auc_str} & {ttt_str} & {lg_str} \\\\")
        cur_bench = r["bench_label"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))


def _write_tests_tex(rows, path: Path):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Selected pairwise statistical tests across thesis methods. "
        r"Retention: Fisher's exact test on proportions. AUC/TTT: two-sided Mann-Whitney U. "
        r"$^\dagger p<0.10$, $^*p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}",
        r"\label{tab:stat_tests}",
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        r"Benchmark & Comparison & Metric & Ref & Tgt & $p$-value & Sig. \\",
        r"\midrule",
    ]
    cur_group = None
    for t in rows:
        group = (t["comparison"], t["benchmark"])
        if group != cur_group:
            if cur_group is not None:
                lines.append(r"\midrule")
            cur_group = group
        p = t["p_value"]
        p_str = f"{p:.4f}" if not math.isnan(p) else "—"
        ref_str = f"{t['ref_mean']:.3f}" if not math.isnan(t["ref_mean"]) else "—"
        tgt_str = f"{t['tgt_mean']:.3f}" if not math.isnan(t["tgt_mean"]) else "—"
        metric = t["metric"].replace("_", r"\_")
        comparison = (
            f"{method_short_label(t['ref_method'])} vs {method_short_label(t['tgt_method'])}"
        )
        lines.append(
            f"  {t['benchmark']} & {comparison} & {metric} & {ref_str} & {tgt_str} & {p_str} & {t['sig']} \\\\"
        )
        cur_group = group
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))


def _write_final_tasks_tex(rows, path: Path):
    seed_count = max((int(r["n"]) for r in rows), default=0)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Final success rate per task and role at end of sequence. "
        rf"Mean $\pm$ SE over {seed_count} seeds.}}",
        r"\label{tab:final_tasks}",
        r"\begin{tabular}{llllc}",
        r"\toprule",
        r"Benchmark & Method & Task & Role & Success \\",
        r"\midrule",
    ]
    cur_bench = None
    for r in rows:
        if r["bench_label"] != cur_bench:
            if cur_bench is not None:
                lines.append(r"\midrule")
            cur_bench = r["bench_label"]
        label = tex_method_label(r["method"])
        s_str = fmt(r["success_mean"], r["success_se"], r["n"], dp=3)
        role_short = "current" if r["task_role"] == "current_task" else "previous"
        lines.append(f"  {cur_bench} & {label} & {r['task_id'][-1:].upper()} & {role_short} & {s_str} \\\\")
        cur_bench = r["bench_label"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 thesis_stats.py SESSION_DIR")
        sys.exit(1)
    main(sys.argv[1])
