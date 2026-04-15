#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS_CSV="${SEEDS_CSV:-57,58,59,60,61,62,63,64,65,66,67,68,69,70,71}" \
SESSION_GROUP="${SESSION_GROUP:-thesis_9x9_full_256_shard2}" \
  bash "$SCRIPT_DIR/run_full_256_shard.sh"
