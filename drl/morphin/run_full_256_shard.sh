#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_PYTHON_BIN="$SCRIPT_DIR/../.venv/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "$DEFAULT_PYTHON_BIN" ]]; then
  PYTHON_BIN="$DEFAULT_PYTHON_BIN"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
RUN_PROFILE="${RUN_PROFILE:-thesis_9x9_full_256}"
METHODS_CSV="${METHODS_CSV:-ddqn_vanilla,oracle_reset,oracle_segmented,oracle_segmented_distill_l001,der_plus_plus}"
BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw9_goal_balanced_ab_v1,gw9_goal_balanced_ac_v1,gw9_goal_balanced_aba_v1,gw9_goal_balanced_abc_v1}"
SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46,47,48,49,50,51,52,53,54,55,56}"
SESSION_GROUP="${SESSION_GROUP:-thesis_9x9_full_256_shard}"
ARCHIVE_FRAC="${ARCHIVE_FRAC:-0.10}"
DER_ALPHA="${DER_ALPHA:-0.01}"
DER_BETA="${DER_BETA:-1.0}"
DER_CAPACITY="${DER_CAPACITY:-0}"
SHARED_REFS="${SHARED_SCRATCH_REFS_JSON:-}"

mkdir -p overnight_logs
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="overnight_logs/${SESSION_GROUP}_${TS}.log"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo ""
echo "============================================================"
echo "  MORPHIN Full 256 Shard"
echo "  $(timestamp)"
echo "  Session group : $SESSION_GROUP"
echo "  Seeds         : $SEEDS_CSV"
echo "  Benchmarks    : $BENCHMARKS_CSV"
echo "  Methods       : $METHODS_CSV"
echo "  Archive frac  : $ARCHIVE_FRAC"
echo "  DER alpha     : $DER_ALPHA"
echo "  Log           : $LOG"
if [[ -n "$SHARED_REFS" ]]; then
  echo "  Scratch refs  : $SHARED_REFS"
else
  echo "  Scratch refs  : profile default"
fi
echo "============================================================"
echo ""

if [[ -n "$SHARED_REFS" ]]; then
  SEEDS_CSV="$SEEDS_CSV" \
  SESSION_GROUP="$SESSION_GROUP" \
  RUN_PROFILE="$RUN_PROFILE" \
  METHODS_CSV="$METHODS_CSV" \
  BENCHMARKS_CSV="$BENCHMARKS_CSV" \
  ARCHIVE_FRAC="$ARCHIVE_FRAC" \
  DER_ALPHA="$DER_ALPHA" \
  DER_BETA="$DER_BETA" \
  DER_CAPACITY="$DER_CAPACITY" \
  AUTO_BUILD_SCRATCH_REFS=0 \
  SCRATCH_REFS_JSON="$SHARED_REFS" \
  PYTHON_BIN="$PYTHON_BIN" \
    bash run_experiments_morphin.sh >"$LOG" 2>&1
else
  SEEDS_CSV="$SEEDS_CSV" \
  SESSION_GROUP="$SESSION_GROUP" \
  RUN_PROFILE="$RUN_PROFILE" \
  METHODS_CSV="$METHODS_CSV" \
  BENCHMARKS_CSV="$BENCHMARKS_CSV" \
  ARCHIVE_FRAC="$ARCHIVE_FRAC" \
  DER_ALPHA="$DER_ALPHA" \
  DER_BETA="$DER_BETA" \
  DER_CAPACITY="$DER_CAPACITY" \
  PYTHON_BIN="$PYTHON_BIN" \
    bash run_experiments_morphin.sh >"$LOG" 2>&1
fi
EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "  [$(timestamp)] DONE — $SESSION_GROUP"
else
  echo "  [$(timestamp)] FAILED — $SESSION_GROUP (exit $EXIT_CODE)"
fi
echo "============================================================"
echo ""
echo "  Log     : $SCRIPT_DIR/$LOG"
echo "  Results : $SCRIPT_DIR/logs/morphin_gridworld/$SESSION_GROUP/"
echo ""

exit $EXIT_CODE
