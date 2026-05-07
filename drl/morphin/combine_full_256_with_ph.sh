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

FULL_GROUP="${FULL_GROUP:-thesis_9x9_full_256_combined}"
PH_GROUP="${PH_GROUP:-thesis_9x9_ph_final}"
COMBINED_GROUP="${COMBINED_GROUP:-thesis_9x9_full_256_ph7_combined}"
FULL_SESSION="${FULL_SESSION:-}"
PH_SESSION="${PH_SESSION:-}"

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
  if [[ ! -d "$src_runs" ]]; then
    return
  fi
  if ! cp -al "$src_runs/." "$dest_runs/"; then
    cp -a "$src_runs/." "$dest_runs/"
  fi
}

if [[ -z "$FULL_SESSION" ]]; then
  FULL_SESSION="$(latest_session_for_group "$FULL_GROUP")"
fi
if [[ -z "$PH_SESSION" ]]; then
  PH_SESSION="$(latest_session_for_group "$PH_GROUP")"
fi

for s in "$FULL_SESSION" "$PH_SESSION"; do
  if [[ -z "$s" || ! -d "$s" ]]; then
    echo "Missing session: $s" >&2
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
echo "  MORPHIN Full 256 + PH Combiner"
echo "  $(timestamp)"
echo "  Combined root : $COMBINED_ROOT"
echo "  Full session  : $FULL_SESSION"
echo "  PH session    : $PH_SESSION"
echo "============================================================"
echo ""

printf 'source_name,session_root\n' >"$COMBINED_ROOT/child_sessions.csv"
printf 'full_256,%s\n' "$FULL_SESSION" >>"$COMBINED_ROOT/child_sessions.csv"
printf 'ph,%s\n' "$PH_SESSION" >>"$COMBINED_ROOT/child_sessions.csv"

copy_run_tree "$FULL_SESSION/runs" "$COMBINED_RUNS_ROOT"
copy_run_tree "$PH_SESSION/runs" "$COMBINED_RUNS_ROOT"

python3 - "$FULL_SESSION/session_config.json" "$PH_SESSION/session_config.json" "$COMBINED_ROOT/session_config.json" <<'PY'
import json
import sys
from pathlib import Path

full_cfg = json.loads(Path(sys.argv[1]).read_text())
ph_cfg = json.loads(Path(sys.argv[2]).read_text())
out_path = Path(sys.argv[3])

methods = [
    "ddqn_vanilla",
    "oracle_reset",
    "oracle_segmented",
    "oracle_segmented_distill_l001",
    "der_plus_plus",
    "ph_reset",
    "ph_segmented",
]

all_seeds = []
for cfg in (full_cfg, ph_cfg):
    for key in ("all_seeds_csv", "seeds_csv", "shard1_seeds_csv", "shard2_seeds_csv", "shard3_seeds_csv"):
        for seed in str(cfg.get(key, "")).split(","):
            seed = seed.strip()
            if seed and seed not in all_seeds:
                all_seeds.append(seed)

base = dict(full_cfg)
base["campaign_group"] = "thesis_9x9_full_256_ph7_combined"
base["methods_csv"] = ",".join(methods)
base["method_set"] = "thesis_full_256_ph"
base["all_seeds_csv"] = ",".join(all_seeds)
base["source_sessions"] = [str(Path(sys.argv[1]).parent), str(Path(sys.argv[2]).parent)]
base["ph_detector"] = {
    "detector_signal": ph_cfg.get("detector_signal"),
    "detector_ema_alpha": ph_cfg.get("detector_ema_alpha"),
    "ph_delta": ph_cfg.get("ph_delta"),
    "ph_threshold": ph_cfg.get("ph_threshold"),
    "ph_min_instances": ph_cfg.get("ph_min_instances"),
    "detector_max_delay_episodes": ph_cfg.get("detector_max_delay_episodes"),
}
out_path.write_text(json.dumps(base, indent=2))
PY

py_run "$SCRIPT_DIR/combine_morphin_analyses.py" \
  --session-root "$COMBINED_ROOT" \
  --output-dir "$COMBINED_ANALYSIS_DIR" \
  --shard-session "$FULL_SESSION" \
  --shard-session "$PH_SESSION" \
  >"$COMBINED_ROOT/aggregate.log" 2>&1

py_run "$SCRIPT_DIR/thesis_stats.py" "$COMBINED_ROOT" >"$COMBINED_ROOT/thesis_stats.log" 2>&1
py_run "$SCRIPT_DIR/thesis_plots.py" "$COMBINED_ROOT" >"$COMBINED_ROOT/thesis_plots.log" 2>&1

echo ""
echo "============================================================"
echo "  [$(timestamp)] DONE — full_256 + PH 7-method combination"
echo "============================================================"
echo ""
echo "  Combined root : $COMBINED_ROOT"
echo "  Report        : $COMBINED_ANALYSIS_DIR/report.md"
echo "  Thesis tables : $COMBINED_ANALYSIS_DIR/thesis/tables"
echo "  Thesis figures: $COMBINED_ANALYSIS_DIR/thesis/figures"
echo ""
