#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_SHARED_SCRATCH_REFS_JSON="$SCRIPT_DIR/logs/morphin_gridworld/thesis_9x9_full_256_parallel/session_20260318_211023/shared_scratch_refs.json"

if [[ -z "${SHARED_SCRATCH_REFS_JSON:-}" && -f "$DEFAULT_SHARED_SCRATCH_REFS_JSON" ]]; then
  SHARED_SCRATCH_REFS_JSON="$DEFAULT_SHARED_SCRATCH_REFS_JSON"
fi

export SHARED_SCRATCH_REFS_JSON

SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46,47,48,49,50,51,52,53,54,55,56}" \
SESSION_GROUP="${SESSION_GROUP:-thesis_9x9_full_256_shard1}" \
  bash "$SCRIPT_DIR/run_full_256_shard.sh"
