#!/usr/bin/env bash
# ============================================================
# run_distill_b_validate.sh
#
# Validación rápida del fix Option B (TD en archive + distill anchor):
#   - ABA: confirma que revisit sigue funcionando con el fix
#   - ABC: confirma que la retención de A y B ya no colapsa
#
# 2 métodos × 3 seeds × 2 benchmarks = 12 runs (~30 min)
#
# Si los resultados son buenos, procede con run_distill_b_full.sh
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TS="$(date '+%Y%m%d_%H%M%S')"
mkdir -p overnight_logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
LOG="overnight_logs/distill_b_validate_${TS}.log"

echo ""
echo "============================================================"
echo "  MORPHIN Distill Option B — Validación"
echo "  $(timestamp)"
echo "  Fix    : TD en archive + distill anchor (Option B)"
echo "  Seeds  : 42,43,44"
echo "  Benchmarks : ABA (revisit sanity) + ABC (fix target)"
echo "  Methods    : oracle_segmented | oracle_segmented_distill_l001"
echo "  Runs total : 2 × 3 × 2 = 12"
echo "  Log        : $LOG"
echo "============================================================"
echo ""

SEEDS_CSV="42,43,44" \
BENCHMARKS_CSV="gw9_goal_balanced_aba_v1,gw9_goal_balanced_abc_v1" \
METHODS_CSV="oracle_segmented,oracle_segmented_distill_l001" \
RUN_PROFILE=thesis_9x9_full_256 \
SESSION_GROUP=thesis_distill_b_validate \
    bash run_experiments_morphin.sh >"$LOG" 2>&1
EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  [$(timestamp)] DONE"
    SESSION=$(ls -t logs/morphin_gridworld/thesis_distill_b_validate/ | head -1)
    REPORT="logs/morphin_gridworld/thesis_distill_b_validate/${SESSION}/analysis/report.md"
    echo ""
    echo "  ► Métricas clave a verificar:"
    echo "    ABC / oracle_segmented_distill_l001:"
    echo "      all_unique_final_rate esperado: >0/5 (actualmente era 0/5)"
    echo "      final_task B esperado: >0.0 (actualmente era 0.0)"
    echo "      final_task A esperado: >0.4 (actualmente era 0.4)"
    echo "    ABA / oracle_segmented_distill_l001:"
    echo "      revisit_task AUC esperado: mantener >0.75"
    echo ""
    if [[ -f "$REPORT" ]]; then
        echo "  ► Extracto del reporte:"
        grep -A2 "abc_v1.*distill_l001\|distill_l001.*abc_v1" "$REPORT" | head -10 || true
        echo ""
        grep "gw9_goal_bal_b\[" "$REPORT" | head -10 || true
    fi
    echo ""
    echo "  Reporte completo:"
    echo "    $SCRIPT_DIR/$REPORT"
    echo ""
    echo "  Si los números mejoran → corre el experimento completo:"
    echo "    bash run_distill_b_full.sh"
else
    echo "  [$(timestamp)] FAILED (exit $EXIT_CODE)"
    echo "  Log: $SCRIPT_DIR/$LOG"
fi
echo "============================================================"

exit $EXIT_CODE
