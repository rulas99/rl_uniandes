#!/usr/bin/env bash
# ============================================================
# run_lambda_sweep.sh
#
# Corre thesis_lambda_sweep: 5 variantes de lambda de destilación
# + baseline oracle_segmented, en (256,256), 3 seeds, AB + AC.
#
# 6 métodos × 3 seeds × 2 benchmarks = 36 runs (~45 min)
#
# Uso:
#   bash run_lambda_sweep.sh
#   SEEDS_CSV="42,43" bash run_lambda_sweep.sh   # override seeds
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS="${SEEDS_CSV:-42,43,44}"
TS="$(date '+%Y%m%d_%H%M%S')"
mkdir -p overnight_logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

LOG="overnight_logs/lambda_sweep_${TS}.log"

echo ""
echo "============================================================"
echo "  MORPHIN Lambda Sweep — thesis_lambda_sweep"
echo "  $(timestamp)"
echo "  Seeds      : $SEEDS"
echo "  Benchmarks : AB  AC"
echo "  Network    : (256,256)"
echo "  Methods    : oracle_segmented (baseline)"
echo "               oracle_segmented_distill_l001  (λ=0.001)"
echo "               oracle_segmented_distill_l005  (λ=0.005)"
echo "               oracle_segmented_distill_l020  (λ=0.020)"
echo "               oracle_segmented_distill_l050  (λ=0.050)"
echo "               oracle_segmented_distill_l200  (λ=0.200)"
echo "  Log        : $LOG"
echo "============================================================"
echo ""

SEEDS_CSV="$SEEDS" RUN_PROFILE=thesis_lambda_sweep \
    bash run_experiments_morphin.sh >"$LOG" 2>&1
EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  [$(timestamp)] DONE — thesis_lambda_sweep"
else
    echo "  [$(timestamp)] FAILED (exit $EXIT_CODE)"
    echo "  Log: $SCRIPT_DIR/$LOG"
fi
echo "============================================================"
echo ""
echo "  Log     : $SCRIPT_DIR/$LOG"
echo "  Results : $SCRIPT_DIR/logs/morphin_gridworld/thesis_lambda_sweep/"
echo ""
echo "  ► Vuelve a Claude con los resultados para iterar."
echo "============================================================"

exit $EXIT_CODE
