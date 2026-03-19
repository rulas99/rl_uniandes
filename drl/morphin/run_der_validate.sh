#!/usr/bin/env bash
# ============================================================
# run_der_validate.sh
#
# Validación de DER++ bajo protocolo comparable al overnight 256.
#
# Métodos:
#   ddqn_vanilla                     (lower bound)
#   oracle_segmented                 (nuestro baseline CL)
#   oracle_segmented_distill_l001    (nuestro método con distilación)
#   der_plus_plus                    (DER++ puro, sin oracle boundary)
#   oracle_der_plus_plus             (DER++ con oracle reset para comparación justa)
#
# 5 métodos × 4 seeds × 2 benchmarks (AB + ABC) = 40 runs
# AB  — plasticidad y new-task transfer
# ABC — retención en cadena de 3 tareas (caso más difícil)
#
# Estimado: ~2-3 horas
#
# Uso:
#   bash run_der_validate.sh
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS="42,43,44,45"
DER_ALPHA="${DER_ALPHA:-0.01}"
DER_BETA="${DER_BETA:-1.0}"
DER_CAPACITY="${DER_CAPACITY:-0}"
PYTHON_BIN_VALUE="${PYTHON_BIN:-python3}"
TS="$(date '+%Y%m%d_%H%M%S')"
mkdir -p overnight_logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
LOG="overnight_logs/der_validate_${TS}.log"

echo ""
echo "============================================================"
echo "  MORPHIN — DER++ Validation Run"
echo "  $(timestamp)"
echo ""
echo "  DER++ hyperparams:"
echo "    der_alpha = $DER_ALPHA  (Q-value consistency weight)"
echo "    der_beta  = $DER_BETA  (TD loss on reservoir weight)"
echo "    reservoir = buffer_capacity (20k)"
echo ""
echo "  Methods:"
echo "    ddqn_vanilla                   (lower bound)"
echo "    oracle_segmented               (baseline CL)"
echo "    oracle_segmented_distill_l001  (nuestro método)"
echo "    der_plus_plus                  (DER++ puro)"
echo "    oracle_der_plus_plus           (DER++ + oracle reset)"
echo ""
echo "  Benchmarks : AB (plasticidad) + ABC (retención 3 tareas)"
echo "  Seeds      : $SEEDS  (4 seeds)"
echo "  Runs total : 5 × 4 × 2 = 40"
echo "  Estimado   : ~2-3 horas"
echo "  Python     : $PYTHON_BIN_VALUE"
echo "  Log        : $LOG"
echo "============================================================"
echo ""
echo "  Progreso en vivo:"
echo "    tail -f $SCRIPT_DIR/$LOG"
echo ""

env -i HOME="$HOME" PATH="$PATH" \
  PYTHON_BIN="$PYTHON_BIN_VALUE" \
  PYTHONPATH="${PYTHONPATH:-}" \
  VIRTUAL_ENV="${VIRTUAL_ENV:-}" \
  CONDA_PREFIX="${CONDA_PREFIX:-}" \
  CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}" \
  RUN_PROFILE=der_rerun_manual \
  SEEDS_CSV="$SEEDS" \
  SESSION_GROUP=thesis_der_validate \
  METHOD_SET=manual_der_validate \
  BENCHMARKS_CSV="gw9_goal_balanced_ab_v1,gw9_goal_balanced_abc_v1" \
  METHODS_CSV="ddqn_vanilla,oracle_segmented,oracle_segmented_distill_l001,der_plus_plus,oracle_der_plus_plus" \
  OBS_MODE=agent_target \
  AUTO_BUILD_SCRATCH_REFS=1 \
  REUSE_SCRATCH_REFS_BY_TASK_SET=1 \
  EPISODES_PER_TASK=400 \
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
  DER_ALPHA="$DER_ALPHA" \
  DER_BETA="$DER_BETA" \
  DER_CAPACITY="$DER_CAPACITY" \
    bash run_experiments_morphin.sh >"$LOG" 2>&1
EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  [$(timestamp)] DONE — thesis_der_validate"
    SESSION=$(ls -t logs/morphin_gridworld/thesis_der_validate/ 2>/dev/null | head -1)
    echo ""
    echo "  Log     : $SCRIPT_DIR/$LOG"
    echo "  Results : $SCRIPT_DIR/logs/morphin_gridworld/thesis_der_validate/${SESSION}/"
    echo "  Report  : $SCRIPT_DIR/logs/morphin_gridworld/thesis_der_validate/${SESSION}/analysis/report.md"
    echo ""
    echo "  Métricas clave a verificar:"
    echo "    AB / der_plus_plus:"
    echo "      new_task_auc no debería colapsar a ~0; revisar der_alpha_loss/der_beta_loss"
    echo "    AB / oracle_der_plus_plus:"
    echo "      si mejora fuerte vs der_plus_plus, el cuello era exploración/reset"
    echo "    ABC / oracle_der_plus_plus:"
    echo "      comparar retención final vs oracle_segmented y distill"
    echo ""
    echo "  ► Si oracle_der_plus_plus sigue flojo → bajar der_alpha a 0.003 o 0.001"
    echo "  ► Si pure DER sigue flojo pero oracle DER mejora → documentar limitación de plasticidad sin reset"
else
    echo "  [$(timestamp)] FAILED (exit $EXIT_CODE)"
    echo "  Log: $SCRIPT_DIR/$LOG"
fi
echo "============================================================"

exit $EXIT_CODE
