#!/usr/bin/env bash
# ============================================================
# run_full_256.sh
#
# Evaluación definitiva de 4 métodos en (256,256):
#   ddqn_vanilla | oracle_reset | oracle_segmented | oracle_segmented_distill_l001
#
# 4 métodos × 5 seeds × 4 benchmarks (AB, AC, ABA, ABC) = 80 runs
# Estimado: ~3-4 horas
#
# Uso:
#   bash run_full_256.sh
#   SEEDS_CSV="42,43,44" bash run_full_256.sh   # reducir seeds para prueba
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS="${SEEDS_CSV:-42,43,44,45,46}"
TS="$(date '+%Y%m%d_%H%M%S')"
mkdir -p overnight_logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

LOG="overnight_logs/full_256_${TS}.log"

echo ""
echo "============================================================"
echo "  MORPHIN Full 256 — thesis_9x9_full_256"
echo "  $(timestamp)"
echo "  Seeds      : $SEEDS"
echo "  Benchmarks : AB  AC  ABA  ABC"
echo "  Network    : (256,256)"
echo "  Methods    :"
echo "    ddqn_vanilla                      (baseline sin oracle)"
echo "    oracle_reset                      (oracle, reset total)"
echo "    oracle_segmented                  (oracle, replay segmentado)"
echo "    oracle_segmented_distill_l001     (oracle, replay + distil λ=0.001)"
echo "  Runs total : 4 métodos × ${#SEEDS} seeds × 4 benchmarks = ~80 runs"
echo "  Log        : $LOG"
echo "============================================================"
echo ""

SEEDS_CSV="$SEEDS" RUN_PROFILE=thesis_9x9_full_256 \
    bash run_experiments_morphin.sh >"$LOG" 2>&1
EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  [$(timestamp)] DONE — thesis_9x9_full_256"
else
    echo "  [$(timestamp)] FAILED (exit $EXIT_CODE)"
    echo "  Log: $SCRIPT_DIR/$LOG"
fi
echo "============================================================"
echo ""
echo "  Log     : $SCRIPT_DIR/$LOG"
echo "  Results : $SCRIPT_DIR/logs/morphin_gridworld/thesis_9x9_full_256/"
echo ""
echo "  ► Vuelve a Claude con los resultados para el análisis final."
echo "============================================================"

exit $EXIT_CODE
