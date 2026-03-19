#!/usr/bin/env bash
# ============================================================
# run_archive_sweep.sh
#
# Archive-fraction sweep: does larger archive give distill-level
# retention without needing distillation?
#
# Methods:
#   oracle_segmented          (af=0.10, baseline)
#   oracle_segmented_af015    (af=0.15)
#   oracle_segmented_af020    (af=0.20)
#   oracle_segmented_af025    (af=0.25)
#   oracle_segmented_af030    (af=0.30)
#
# 5 methods × 8 seeds × 4 benchmarks = 160 runs
# Estimado: ~6-7 horas
#
# Uso:
#   bash run_archive_sweep.sh
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS="42,43,44,45,46,47,48,49"
TS="$(date '+%Y%m%d_%H%M%S')"
mkdir -p overnight_logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
LOG="overnight_logs/archive_sweep_${TS}.log"

echo ""
echo "============================================================"
echo "  MORPHIN — Archive Fraction Sweep"
echo "  $(timestamp)"
echo ""
echo "  Pregunta: ¿Un archive más grande reemplaza la distilación?"
echo "  Network : (256,256)"
echo "  Seeds   : $SEEDS  (8 seeds)"
echo "  Benchmarks: AB  AC  ABA  ABC"
echo "  Methods:"
echo "    oracle_segmented        (af=0.10, baseline)"
echo "    oracle_segmented_af015  (af=0.15)"
echo "    oracle_segmented_af020  (af=0.20)"
echo "    oracle_segmented_af025  (af=0.25)"
echo "    oracle_segmented_af030  (af=0.30)"
echo ""
echo "  Runs total : 5 × 8 × 4 = 160"
echo "  Estimado   : ~6-7 horas"
echo "  Log        : $LOG"
echo "============================================================"
echo ""
echo "  Progreso en vivo:"
echo "    tail -f $SCRIPT_DIR/$LOG"
echo ""

env -i HOME="$HOME" PATH="$PATH" \
  SEEDS_CSV="$SEEDS" \
  SESSION_GROUP=thesis_archive_sweep \
  RUN_PROFILE=thesis_archive_sweep \
    bash run_experiments_morphin.sh >"$LOG" 2>&1
EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  [$(timestamp)] DONE — thesis_archive_sweep"
    SESSION=$(ls -t logs/morphin_gridworld/thesis_archive_sweep/ 2>/dev/null | head -1)
    echo ""
    echo "  Log     : $SCRIPT_DIR/$LOG"
    echo "  Results : $SCRIPT_DIR/logs/morphin_gridworld/thesis_archive_sweep/${SESSION}/"
    echo "  Report  : $SCRIPT_DIR/logs/morphin_gridworld/thesis_archive_sweep/${SESSION}/analysis/report.md"
    echo ""
    echo "  Métricas clave a comparar vs distill_l001:"
    echo "    ABC retention: distill=11/12 → ¿af030 llega a 9+/12?"
    echo "    ABA revisit AUC: distill=1.000 → ¿af030 se acerca?"
    echo "    AB plasticity: ¿aumentar archive daña plasticidad?"
    echo ""
    echo "  ► Comparte resultados con Claude para análisis."
else
    echo "  [$(timestamp)] FAILED (exit $EXIT_CODE)"
    echo "  Log: $SCRIPT_DIR/$LOG"
fi
echo "============================================================"

exit $EXIT_CODE
