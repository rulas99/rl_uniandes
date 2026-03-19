#!/usr/bin/env bash
# ============================================================
# run_distill_b_full.sh
#
# Evaluación definitiva con Option B fix:
#   ddqn_vanilla | oracle_reset | oracle_segmented | oracle_segmented_distill_l001
#
# 4 métodos × 5 seeds × 4 benchmarks (AB, AC, ABA, ABC) = 80 runs
# Estimado: ~3-4 horas
#
# Corre esto solo si run_distill_b_validate.sh pasó.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS="${SEEDS_CSV:-42,43,44,45,46}"
TS="$(date '+%Y%m%d_%H%M%S')"
mkdir -p overnight_logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
LOG="overnight_logs/distill_b_full_${TS}.log"

echo ""
echo "============================================================"
echo "  MORPHIN Full 256 — Distill Option B"
echo "  $(timestamp)"
echo "  Fix    : TD en archive + distill anchor (λ=0.001)"
echo "  Seeds  : $SEEDS"
echo "  Benchmarks : AB  AC  ABA  ABC"
echo "  Network    : (256,256)"
echo "  Methods    :"
echo "    ddqn_vanilla"
echo "    oracle_reset"
echo "    oracle_segmented"
echo "    oracle_segmented_distill_l001  [Option B fix]"
echo "  Runs total : 4 × 5 × 4 = 80"
echo "  Log        : $LOG"
echo "============================================================"
echo ""

SEEDS_CSV="$SEEDS" RUN_PROFILE=thesis_9x9_full_256 \
SESSION_GROUP=thesis_distill_b_full \
    bash run_experiments_morphin.sh >"$LOG" 2>&1
EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  [$(timestamp)] DONE — thesis_distill_b_full"
    SESSION=$(ls -t logs/morphin_gridworld/thesis_distill_b_full/ 2>/dev/null | head -1)
    echo ""
    echo "  Log     : $SCRIPT_DIR/$LOG"
    echo "  Results : $SCRIPT_DIR/logs/morphin_gridworld/thesis_distill_b_full/${SESSION}/"
    echo ""
    echo "  ► Vuelve a Claude con los resultados para análisis final."
else
    echo "  [$(timestamp)] FAILED (exit $EXIT_CODE)"
    echo "  Log: $SCRIPT_DIR/$LOG"
fi
echo "============================================================"

exit $EXIT_CODE
