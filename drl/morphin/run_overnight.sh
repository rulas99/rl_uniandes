#!/usr/bin/env bash
# ============================================================
# run_overnight.sh
#
# Corrida de tesis definitiva — Option B fix confirmado.
# 4 métodos × 12 seeds × 4 benchmarks (AB, AC, ABA, ABC) = 192 runs
# Estimado: ~8-10 horas
#
# Uso:
#   bash run_overnight.sh
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS="42,43,44,45,46,47,48,49,50,51,52,53"
TS="$(date '+%Y%m%d_%H%M%S')"
mkdir -p overnight_logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
LOG="overnight_logs/overnight_${TS}.log"

echo ""
echo "============================================================"
echo "  MORPHIN — Corrida Overnight (Tesis Final)"
echo "  $(timestamp)"
echo ""
echo "  Fix     : Option B — TD en archive + distill anchor λ=0.001"
echo "  Network : (256,256)"
echo "  Seeds   : $SEEDS  (12 seeds)"
echo "  Benchmarks:"
echo "    gw9_goal_balanced_ab_v1   (2 tasks, new-task transfer)"
echo "    gw9_goal_balanced_ac_v1   (2 tasks, positive FT)"
echo "    gw9_goal_balanced_aba_v1  (revisit, plasticity)"
echo "    gw9_goal_balanced_abc_v1  (3 tasks, retention + FT)"
echo "  Methods:"
echo "    ddqn_vanilla                       (baseline)"
echo "    oracle_reset                       (oracle + reset)"
echo "    oracle_segmented                   (oracle + replay seg.)"
echo "    oracle_segmented_distill_l001      (oracle + replay + distill)"
echo ""
echo "  Runs total : 4 × 12 × 4 = 192"
echo "  Estimado   : ~8-10 horas"
echo "  Log        : $LOG"
echo "============================================================"
echo ""
echo "  Progreso en vivo:"
echo "    tail -f $SCRIPT_DIR/$LOG"
echo ""

env -i HOME="$HOME" PATH="$PATH" \
  SEEDS_CSV="$SEEDS" \
  SESSION_GROUP=thesis_overnight \
  RUN_PROFILE=thesis_9x9_full_256 \
    bash run_experiments_morphin.sh >"$LOG" 2>&1
EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  [$(timestamp)] DONE — thesis_overnight"
    SESSION=$(ls -t logs/morphin_gridworld/thesis_overnight/ 2>/dev/null | head -1)
    echo ""
    echo "  Log     : $SCRIPT_DIR/$LOG"
    echo "  Results : $SCRIPT_DIR/logs/morphin_gridworld/thesis_overnight/${SESSION}/"
    echo "  Report  : $SCRIPT_DIR/logs/morphin_gridworld/thesis_overnight/${SESSION}/analysis/report.md"
    echo ""
    echo "  ► Comparte el reporte con Claude para el análisis final."
else
    echo "  [$(timestamp)] FAILED (exit $EXIT_CODE)"
    echo "  Log: $SCRIPT_DIR/$LOG"
fi
echo "============================================================"

exit $EXIT_CODE
