#!/usr/bin/env bash
# ============================================================
# run_thesis_overnight.sh
#
# Runs all pending thesis campaigns sequentially.
# Priority order:
#   1. thesis_ablation      — fills the 5x5 ablation gap in Chapter 5 (~4 hrs)
#   2. thesis_9x9_goalcond  — 9x9 balanced, 2-task sequences AB/AC/ABA (~12-14 hrs)
#   3. thesis_9x9_multitask — 9x9 balanced, 3-4 task sequences ABC/ABCA (~12-16 hrs)
#
# Usage:
#   bash run_thesis_overnight.sh              # run all three campaigns
#   SKIP_ABLATION=1 bash run_thesis_overnight.sh  # skip 5x5 ablation
#   ONLY_ABLATION=1 bash run_thesis_overnight.sh  # only 5x5 ablation
#
# Logs:
#   Each campaign logs to: logs/morphin_gridworld/<session_group>/<session>/
#   Live stdout/stderr also tee'd to: overnight_logs/
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configuration ─────────────────────────────────────────
SKIP_ABLATION="${SKIP_ABLATION:-0}"
SKIP_9X9_GOALCOND="${SKIP_9X9_GOALCOND:-0}"
SKIP_9X9_MULTITASK="${SKIP_9X9_MULTITASK:-1}"   # OFF by default — start next night
ONLY_ABLATION="${ONLY_ABLATION:-0}"

mkdir -p overnight_logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_campaign() {
    local profile="$1"
    local label="$2"
    local logfile="overnight_logs/${profile}_$(date '+%Y%m%d_%H%M%S').log"

    echo ""
    echo "============================================================"
    echo "[$(timestamp)] Starting campaign: $label ($profile)"
    echo "  Log: $logfile"
    echo "============================================================"

    if RUN_PROFILE="$profile" bash run_experiments_morphin.sh 2>&1 | tee "$logfile"; then
        echo "[$(timestamp)] COMPLETED: $label"
    else
        echo "[$(timestamp)] FAILED: $label — see $logfile"
        echo "[$(timestamp)] Continuing with next campaign..."
    fi
}

echo "============================================================"
echo "[$(timestamp)] MORPHIN Thesis — Overnight Campaign Runner"
echo "============================================================"
echo "  SKIP_ABLATION=$SKIP_ABLATION"
echo "  SKIP_9X9_GOALCOND=$SKIP_9X9_GOALCOND"
echo "  SKIP_9X9_MULTITASK=$SKIP_9X9_MULTITASK"
echo "  ONLY_ABLATION=$ONLY_ABLATION"
echo ""

# ── Campaign 1: thesis_ablation (5x5, fills Chapter 5 gap) ───────────────────
# Methods: ddqn_vanilla, oracle_reset, oracle_segmented, oracle_segmented_td, morphin_lite
# Benchmark: gw_goal_conditioned_balanced_ac_v1 (15 seeds)
# Est. time: ~4 hrs
if [[ "$SKIP_ABLATION" != "1" ]]; then
    run_campaign "thesis_ablation" "5x5 Ablation (oracle_segmented_td vs morphin_lite)"
fi

if [[ "$ONLY_ABLATION" == "1" ]]; then
    echo "[$(timestamp)] ONLY_ABLATION=1 — stopping after ablation."
    exit 0
fi

# ── Campaign 2: thesis_9x9_goalcond (9x9, 2-task sequences) ──────────────────
# Methods: ddqn_vanilla, oracle_reset, oracle_segmented
# Benchmarks: AB, AC, ABA (15 seeds each)
# Est. time: ~12-14 hrs
if [[ "$SKIP_9X9_GOALCOND" != "1" ]]; then
    run_campaign "thesis_9x9_goalcond" "9x9 Goal-Conditioned (AB, AC, ABA)"
fi

# ── Campaign 3: thesis_9x9_multitask (9x9, 3-4 task sequences) ───────────────
# Methods: ddqn_vanilla, oracle_reset, oracle_segmented
# Benchmarks: ABC, ABCA (15 seeds each)
# Est. time: ~12-16 hrs
if [[ "$SKIP_9X9_MULTITASK" != "1" ]]; then
    run_campaign "thesis_9x9_multitask" "9x9 Multi-Task (ABC, ABCA)"
fi

echo ""
echo "============================================================"
echo "[$(timestamp)] All scheduled campaigns finished."
echo "  Logs: $SCRIPT_DIR/overnight_logs/"
echo "============================================================"
