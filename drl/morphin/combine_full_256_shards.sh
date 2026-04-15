#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_PYTHON_BIN="$SCRIPT_DIR/../.venv/bin/python"
if [[ -x "$DEFAULT_PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
COMBINED_GROUP="${COMBINED_GROUP:-thesis_9x9_full_256_combined}"
SHARD1_SESSION="${SHARD1_SESSION:-}"
SHARD2_SESSION="${SHARD2_SESSION:-}"
SHARD3_SESSION="${SHARD3_SESSION:-}"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

latest_session_for_group() {
  local group="$1"
  ls -1dt "$SCRIPT_DIR/logs/morphin_gridworld/$group"/session_* 2>/dev/null | head -n 1
}

py_run() {
  PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$@"
}

copy_run_tree() {
  local src_runs="$1"
  local dest_runs="$2"
  if ! cp -al "$src_runs/." "$dest_runs/"; then
    cp -a "$src_runs/." "$dest_runs/"
  fi
}

if [[ -z "$SHARD1_SESSION" ]]; then
  SHARD1_SESSION="$(latest_session_for_group thesis_9x9_full_256_shard1)"
fi
if [[ -z "$SHARD2_SESSION" ]]; then
  SHARD2_SESSION="$(latest_session_for_group thesis_9x9_full_256_shard2)"
fi
if [[ -z "$SHARD3_SESSION" ]]; then
  SHARD3_SESSION="$(latest_session_for_group thesis_9x9_full_256_shard3)"
fi

for s in "$SHARD1_SESSION" "$SHARD2_SESSION" "$SHARD3_SESSION"; do
  if [[ -z "$s" || ! -d "$s" ]]; then
    echo "Missing shard session: $s" >&2
    exit 1
  fi
done

TS="$(date '+%Y%m%d_%H%M%S')"
COMBINED_ROOT="$SCRIPT_DIR/logs/morphin_gridworld/$COMBINED_GROUP/session_$TS"
COMBINED_RUNS_ROOT="$COMBINED_ROOT/runs"
COMBINED_ANALYSIS_DIR="$COMBINED_ROOT/analysis"
mkdir -p "$COMBINED_RUNS_ROOT" "$COMBINED_ANALYSIS_DIR"

echo ""
echo "============================================================"
echo "  MORPHIN Full 256 Shard Combiner"
echo "  $(timestamp)"
echo "  Combined root : $COMBINED_ROOT"
echo "  Shard 1       : $SHARD1_SESSION"
echo "  Shard 2       : $SHARD2_SESSION"
echo "  Shard 3       : $SHARD3_SESSION"
echo "============================================================"
echo ""

printf 'shard_name,session_root\n' >"$COMBINED_ROOT/child_sessions.csv"
printf 'shard1,%s\n' "$SHARD1_SESSION" >>"$COMBINED_ROOT/child_sessions.csv"
printf 'shard2,%s\n' "$SHARD2_SESSION" >>"$COMBINED_ROOT/child_sessions.csv"
printf 'shard3,%s\n' "$SHARD3_SESSION" >>"$COMBINED_ROOT/child_sessions.csv"

copy_run_tree "$SHARD1_SESSION/runs" "$COMBINED_RUNS_ROOT"
copy_run_tree "$SHARD2_SESSION/runs" "$COMBINED_RUNS_ROOT"
copy_run_tree "$SHARD3_SESSION/runs" "$COMBINED_RUNS_ROOT"

python3 - "$SHARD1_SESSION/session_config.json" "$SHARD2_SESSION/session_config.json" "$SHARD3_SESSION/session_config.json" "$COMBINED_ROOT/session_config.json" <<'PY'
import json
import sys
from pathlib import Path

cfgs = [json.loads(Path(p).read_text()) for p in sys.argv[1:4]]
out_path = Path(sys.argv[4])

base = dict(cfgs[0])
all_seeds = []
for cfg in cfgs:
    for key in ("all_seeds_csv", "seeds_csv", "shard1_seeds_csv", "shard2_seeds_csv", "shard3_seeds_csv"):
        for seed in str(cfg.get(key, "")).split(","):
            seed = seed.strip()
            if seed and seed not in all_seeds:
                all_seeds.append(seed)

base["campaign_group"] = "thesis_9x9_full_256_combined"
base["all_seeds_csv"] = ",".join(all_seeds)
base["source_sessions"] = [str(Path(p).parent) for p in sys.argv[1:4]]
out_path.write_text(json.dumps(base, indent=2))
PY

py_run "$SCRIPT_DIR/combine_morphin_analyses.py" \
  --session-root "$COMBINED_ROOT" \
  --output-dir "$COMBINED_ANALYSIS_DIR" \
  --shard-session "$SHARD1_SESSION" \
  --shard-session "$SHARD2_SESSION" \
  --shard-session "$SHARD3_SESSION" \
  >"$COMBINED_ROOT/aggregate.log" 2>&1

py_run "$SCRIPT_DIR/thesis_stats.py" "$COMBINED_ROOT" >"$COMBINED_ROOT/thesis_stats.log" 2>&1
py_run "$SCRIPT_DIR/thesis_plots.py" "$COMBINED_ROOT" >"$COMBINED_ROOT/thesis_plots.log" 2>&1

echo ""
echo "============================================================"
echo "  [$(timestamp)] DONE — shard combination"
echo "============================================================"
echo ""
echo "  Combined root : $COMBINED_ROOT"
echo "  Report        : $COMBINED_ANALYSIS_DIR/report.md"
echo "  Thesis tables : $COMBINED_ANALYSIS_DIR/thesis/tables"
echo "  Thesis figures: $COMBINED_ANALYSIS_DIR/thesis/figures"
echo ""
