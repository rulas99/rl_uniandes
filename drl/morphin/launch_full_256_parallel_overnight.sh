#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p overnight_logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

SESSION_TS="${SESSION_TS:-$(date '+%Y%m%d_%H%M%S')}"
CAMPAIGN_GROUP="${CAMPAIGN_GROUP:-thesis_9x9_full_256_parallel}"
CAMPAIGN_ROOT="$SCRIPT_DIR/logs/morphin_gridworld/$CAMPAIGN_GROUP/session_$SESSION_TS"
RUN_LOG_FILE="${RUN_LOG_FILE:-$SCRIPT_DIR/overnight_logs/full_256_parallel_${SESSION_TS}.log}"
LATEST_POINTER_FILE="$SCRIPT_DIR/overnight_logs/full_256_parallel_latest.txt"
LATEST_LOG_POINTER_FILE="$SCRIPT_DIR/overnight_logs/full_256_parallel_latest_log.txt"
LAUNCH_INFO_FILE="$CAMPAIGN_ROOT/launch_info.env"

mkdir -p "$CAMPAIGN_ROOT"

echo ""
echo "============================================================"
echo "  MORPHIN Full 256 Parallel Overnight Launcher"
echo "  $(timestamp)"
echo "  Campaign dir : $CAMPAIGN_ROOT"
echo "  Launch log   : $RUN_LOG_FILE"
if [[ -n "${SHARED_SCRATCH_REFS_JSON:-}" ]]; then
  echo "  Scratch refs : $SHARED_SCRATCH_REFS_JSON"
fi
echo "============================================================"
echo ""

SESSION_TS="$SESSION_TS" RUN_LOG_FILE="$RUN_LOG_FILE" \
  nohup bash "$SCRIPT_DIR/run_full_256_parallel.sh" >"$RUN_LOG_FILE" 2>&1 < /dev/null &
LAUNCH_PID="$!"

printf '%s\n' "$CAMPAIGN_ROOT" >"$LATEST_POINTER_FILE"
printf '%s\n' "$RUN_LOG_FILE" >"$LATEST_LOG_POINTER_FILE"

{
  printf "SESSION_TS=%q\n" "$SESSION_TS"
  printf "CAMPAIGN_GROUP=%q\n" "$CAMPAIGN_GROUP"
  printf "CAMPAIGN_ROOT=%q\n" "$CAMPAIGN_ROOT"
  printf "RUN_LOG_FILE=%q\n" "$RUN_LOG_FILE"
  printf "LAUNCH_PID=%q\n" "$LAUNCH_PID"
  printf "SHARED_SCRATCH_REFS_JSON=%q\n" "${SHARED_SCRATCH_REFS_JSON:-}"
} >"$LAUNCH_INFO_FILE"

printf '%s\n' "$LAUNCH_PID" >"$CAMPAIGN_ROOT/launcher.pid"

echo "  Launch PID   : $LAUNCH_PID"
echo "  Monitor      : bash $SCRIPT_DIR/monitor_full_256_parallel.sh $CAMPAIGN_ROOT"
echo "  Tail log     : tail -f $RUN_LOG_FILE"
echo ""
