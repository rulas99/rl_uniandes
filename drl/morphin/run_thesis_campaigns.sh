#!/usr/bin/env bash
# Runs all three thesis campaigns sequentially.
# Usage: bash run_thesis_campaigns.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/thesis_run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$SCRIPT_DIR/logs"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== THESIS CAMPAIGNS START ==="
log "Log: $LOG_FILE"

run_campaign() {
    local profile="$1"
    log "--- Starting: $profile ---"
    local t0=$SECONDS
    RUN_PROFILE="$profile" bash "$SCRIPT_DIR/run_experiments_morphin.sh" 2>&1 | tee -a "$LOG_FILE"
    local elapsed=$(( SECONDS - t0 ))
    log "--- Finished: $profile in $(( elapsed / 60 ))m $(( elapsed % 60 ))s ---"
}

run_campaign thesis_goal_transfer
run_campaign thesis_hidden
run_campaign thesis_ablation

log "=== ALL CAMPAIGNS DONE ==="
