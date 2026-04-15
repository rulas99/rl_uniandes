#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS_CSV="${SEEDS_CSV:-72,73,74,75,76,77,78,79,80,81,82,83,84,85,86}" \
SESSION_GROUP="${SESSION_GROUP:-thesis_9x9_full_256_shard3}" \
  bash "$SCRIPT_DIR/run_full_256_shard.sh"
