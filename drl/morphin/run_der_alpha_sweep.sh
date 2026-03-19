#!/usr/bin/env bash
# ============================================================
# run_der_alpha_sweep.sh
#
# Mini-sweep de DER++ variando alpha con el mismo protocolo
# del piloto corto ya ejecutado.
#
# Por defecto corre solo der_plus_plus puro para ahorrar tiempo.
# La comparación se hace contra la sesión baseline ya existente
# con oracle_segmented bajo el mismo protocolo.
#
# Uso:
#   bash run_der_alpha_sweep.sh
#   PYTHON_BIN=/ruta/a/python ALPHAS_CSV=0.001,0.003,0.01 bash run_der_alpha_sweep.sh
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS="${SEEDS_CSV:-42,43}"
ALPHAS_CSV="${ALPHAS_CSV:-0.001,0.003,0.01}"
DER_BETA_VALUE="${DER_BETA:-1.0}"
PYTHON_BIN_VALUE="${PYTHON_BIN:-python3}"
TS="$(date '+%Y%m%d_%H%M%S')"

mkdir -p overnight_logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
tag_for_alpha() {
  local alpha="$1"
  printf '%s' "${alpha/./p}"
}

SUMMARY_CSV="overnight_logs/der_alpha_sweep_summary_${TS}.csv"
printf 'alpha,session_group,session_id,benchmark,method,final_seen_success_mean,current_task_final_success,new_task_recovery_auc_success_eval_mean,new_task_time_to_threshold_eval_steps_mean,switch_time_to_threshold_eval_steps_delta_vs_scratch_mean,switch_log_adaptation_gain_vs_scratch_steps_mean\n' >"$SUMMARY_CSV"

IFS=',' read -r -a ALPHAS <<<"$ALPHAS_CSV"

echo ""
echo "============================================================"
echo "  MORPHIN — DER++ Alpha Sweep"
echo "  $(timestamp)"
echo ""
echo "  Alphas     : $ALPHAS_CSV"
echo "  Beta       : $DER_BETA_VALUE"
echo "  Methods    : der_plus_plus"
echo "  Benchmarks : AB + ABC"
echo "  Seeds      : $SEEDS"
echo "  Protocol   : same as thesis_der_pilot"
echo "  Python     : $PYTHON_BIN_VALUE"
echo "  Summary    : $SUMMARY_CSV"
echo "============================================================"
echo ""

for alpha in "${ALPHAS[@]}"; do
  tag="$(tag_for_alpha "$alpha")"
  session_group="thesis_der_alpha_sweep_${tag}"
  log_file="overnight_logs/der_alpha_${tag}_${TS}.log"

  echo "------------------------------------------------------------"
  echo "  [$(timestamp)] Running der_plus_plus with alpha=$alpha"
  echo "  Session group : $session_group"
  echo "  Log           : $log_file"
  echo "------------------------------------------------------------"

  env -i HOME="$HOME" PATH="$PATH" \
    PYTHON_BIN="$PYTHON_BIN_VALUE" \
    PYTHONPATH="${PYTHONPATH:-}" \
    VIRTUAL_ENV="${VIRTUAL_ENV:-}" \
    CONDA_PREFIX="${CONDA_PREFIX:-}" \
    CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}" \
    RUN_PROFILE=der_alpha_sweep_short \
    SESSION_GROUP="$session_group" \
    METHOD_SET=manual_der_alpha_sweep \
    METHODS_CSV="der_plus_plus" \
    BENCHMARKS_CSV="gw9_goal_balanced_ab_v1,gw9_goal_balanced_abc_v1" \
    SEEDS_CSV="$SEEDS" \
    OBS_MODE=agent_target \
    AUTO_BUILD_SCRATCH_REFS=1 \
    REUSE_SCRATCH_REFS_BY_TASK_SET=1 \
    EPISODES_PER_TASK=300 \
    MAX_STEPS_PER_EPISODE=250 \
    SCRATCH_EPISODES_PER_TASK=500 \
    SCRATCH_MAX_STEPS_PER_EPISODE=250 \
    EVAL_EPISODES=15 \
    EVAL_EVERY_EPISODES=25 \
    EVAL_DENSE_EVERY_EPISODES=1 \
    EVAL_DENSE_WINDOW_EPISODES=25 \
    WARMUP_STEPS=750 \
    SCRATCH_WARMUP_STEPS=750 \
    BATCH_SIZE=64 \
    EPS_DECAY_STEPS=15000 \
    SCRATCH_EPS_DECAY_STEPS=15000 \
    EPS_RESET_VALUE=0.9 \
    EPS_DECAY_STEPS_AFTER_SWITCH=30000 \
    POST_SWITCH_STEPS=7500 \
    ARCHIVE_FRAC=0.10 \
    HIDDEN_SIZES_CSV="256,256" \
    BUFFER_CAPACITY=20000 \
    RECENT_BUFFER_CAPACITY=12000 \
    ARCHIVE_BUFFER_CAPACITY=8000 \
    SCRATCH_BUFFER_CAPACITY=20000 \
    SCRATCH_REF_MIN_VALID_FRACTION=0.7 \
    DER_ALPHA="$alpha" \
    DER_BETA="$DER_BETA_VALUE" \
    DER_CAPACITY=0 \
      bash run_experiments_morphin.sh >"$log_file" 2>&1
  exit_code=$?

  if [[ $exit_code -ne 0 ]]; then
    echo "  [$(timestamp)] FAILED alpha=$alpha (exit $exit_code)"
    echo "  Log: $SCRIPT_DIR/$log_file"
    echo ""
    continue
  fi

  session_id="$(ls -t "logs/morphin_gridworld/${session_group}/" 2>/dev/null | head -1)"
  session_root="$SCRIPT_DIR/logs/morphin_gridworld/${session_group}/${session_id}"
  report_path="$session_root/analysis/report.md"

  echo "  [$(timestamp)] DONE alpha=$alpha"
  echo "  Results : $session_root/"
  echo "  Report  : $report_path"

  if [[ -f "$session_root/analysis/benchmark_method_summary.csv" ]]; then
    SESSION_ROOT="$session_root" ALPHA="$alpha" SESSION_GROUP_NAME="$session_group" SUMMARY_PATH="$SUMMARY_CSV" python3 - <<'PY'
import csv
import os
from pathlib import Path

session_root = Path(os.environ["SESSION_ROOT"])
alpha = os.environ["ALPHA"]
session_group = os.environ["SESSION_GROUP_NAME"]
summary_path = Path(os.environ["SUMMARY_PATH"])
session_id = session_root.name
rows = list(csv.DictReader((session_root / "analysis" / "benchmark_method_summary.csv").open()))

with summary_path.open("a", newline="") as handle:
    writer = csv.writer(handle)
    for row in rows:
        writer.writerow([
            alpha,
            session_group,
            session_id,
            row["benchmark"],
            row["method"],
            row["final_seen_success_mean"],
            row["current_task_final_success_mean"],
            row["new_task_recovery_auc_success_eval_mean"],
            row["new_task_time_to_threshold_eval_steps_mean"],
            row["switch_time_to_threshold_eval_steps_delta_vs_scratch_mean"],
            row["switch_log_adaptation_gain_vs_scratch_steps_mean"],
        ])
PY
  fi

  echo ""
done

echo "============================================================"
echo "  Sweep finished"
echo "  Summary CSV : $SCRIPT_DIR/$SUMMARY_CSV"
echo "============================================================"

