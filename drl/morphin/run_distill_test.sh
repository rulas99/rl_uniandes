#!/usr/bin/env bash
# ============================================================
# run_distill_test.sh
#
# Corre thesis_distill_test (128,128) y thesis_distill_test_256
# en PARALELO con 3 seeds.  Imprime estado claro al terminar.
#
# Uso:
#   bash run_distill_test.sh              # 3 seeds (default)
#   SEEDS_CSV="42,43" bash run_distill_test.sh   # override seeds
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS="${SEEDS_CSV:-42,43,44}"
TS="$(date '+%Y%m%d_%H%M%S')"
mkdir -p overnight_logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

LOG_128="overnight_logs/distill_128_${TS}.log"
LOG_256="overnight_logs/distill_256_${TS}.log"

echo ""
echo "============================================================"
echo "  MORPHIN Distillation Test — Parallel Launch"
echo "  $(timestamp)"
echo "  Seeds    : $SEEDS"
echo "  Benchmarks: AB  AC  ABA  ABC"
echo "  Methods  : oracle_segmented  |  oracle_segmented_distill"
echo "  Nets     : (128,128) → $LOG_128"
echo "             (256,256) → $LOG_256"
echo "============================================================"
echo ""

# ── Launch both in background ─────────────────────────────────
SEEDS_CSV="$SEEDS" RUN_PROFILE=thesis_distill_test \
    bash run_experiments_morphin.sh >"$LOG_128" 2>&1 &
PID_128=$!

SEEDS_CSV="$SEEDS" RUN_PROFILE=thesis_distill_test_256 \
    bash run_experiments_morphin.sh >"$LOG_256" 2>&1 &
PID_256=$!

echo "[$(timestamp)] PID $PID_128 → thesis_distill_test     (128,128)"
echo "[$(timestamp)] PID $PID_256 → thesis_distill_test_256 (256,256)"
echo ""
echo "  Progreso en vivo:"
echo "    tail -f $LOG_128"
echo "    tail -f $LOG_256"
echo ""

# ── Wait for both and report ───────────────────────────────────
EXIT_128=0
EXIT_256=0

wait $PID_128 || EXIT_128=$?
echo ""
echo "============================================================"
if [[ $EXIT_128 -eq 0 ]]; then
    echo "  [$(timestamp)] ✓  thesis_distill_test     (128,128) — DONE"
else
    echo "  [$(timestamp)] ✗  thesis_distill_test     (128,128) — FAILED (exit $EXIT_128)"
    echo "     Log: $LOG_128"
fi

wait $PID_256 || EXIT_256=$?
if [[ $EXIT_256 -eq 0 ]]; then
    echo "  [$(timestamp)] ✓  thesis_distill_test_256 (256,256) — DONE"
else
    echo "  [$(timestamp)] ✗  thesis_distill_test_256 (256,256) — FAILED (exit $EXIT_256)"
    echo "     Log: $LOG_256"
fi

echo "============================================================"
echo ""
echo "  Logs guardados en:"
echo "    $SCRIPT_DIR/$LOG_128"
echo "    $SCRIPT_DIR/$LOG_256"
echo ""
echo "  Resultados en:"
SROOT="$SCRIPT_DIR/logs/morphin_gridworld"
echo "    $SROOT/thesis_distill_test/"
echo "    $SROOT/thesis_distill_test_256/"
echo ""
echo "  ► Vuelve a Claude con los resultados para iterar."
echo "============================================================"

# Return non-zero if either failed
[[ $EXIT_128 -eq 0 && $EXIT_256 -eq 0 ]]
