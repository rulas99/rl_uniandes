#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LATEST_POINTER_FILE="$SCRIPT_DIR/overnight_logs/full_256_parallel_latest.txt"
CAMPAIGN_ROOT="${1:-}"

if [[ -z "$CAMPAIGN_ROOT" ]]; then
  if [[ ! -f "$LATEST_POINTER_FILE" ]]; then
    echo "No latest campaign pointer found at $LATEST_POINTER_FILE" >&2
    exit 1
  fi
  CAMPAIGN_ROOT="$(cat "$LATEST_POINTER_FILE")"
fi

STATUS_FILE="$CAMPAIGN_ROOT/campaign_status.env"
if [[ -f "$STATUS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$STATUS_FILE"
else
  config_file="$CAMPAIGN_ROOT/session_config.json"
  SESSION_TS="$(basename "$CAMPAIGN_ROOT" | sed 's/^session_//')"
  START_EPOCH="$(stat -c '%Y' "$CAMPAIGN_ROOT")"
  STATE="legacy_or_untracked"
  LAUNCHER_PID=""
  RUN_LOG_FILE=""
  TOTAL_EXPECTED_RUNS=0
  TOTAL_METHODS=0
  TOTAL_BENCHMARKS=0
  SHARD1_NAME="shard1"
  SHARD2_NAME="shard2"
  SHARD3_NAME="shard3"
  SHARD1_SEEDS_CSV=""
  SHARD2_SEEDS_CSV=""
  SHARD3_SEEDS_CSV=""
  if [[ -f "$config_file" ]]; then
    # shellcheck disable=SC1090
    source /dev/stdin <<EOF
$(python3 - "$config_file" <<'PY'
import json
import shlex
import sys

cfg = json.load(open(sys.argv[1], "r", encoding="utf-8"))
methods = [x for x in cfg.get("methods_csv", "").split(",") if x]
benchmarks = [x for x in cfg.get("benchmarks_csv", "").split(",") if x]
all_seeds = [x for x in cfg.get("all_seeds_csv", "").split(",") if x]
if not all_seeds:
    seed_fields = [cfg.get("shard1_seeds_csv", ""), cfg.get("shard2_seeds_csv", ""), cfg.get("shard3_seeds_csv", "")]
    for seed_csv in seed_fields:
        all_seeds.extend([x for x in seed_csv.split(",") if x])

print(f"TOTAL_METHODS={shlex.quote(str(len(methods)))}")
print(f"TOTAL_BENCHMARKS={shlex.quote(str(len(benchmarks)))}")
print(f"TOTAL_EXPECTED_RUNS={shlex.quote(str(len(methods) * len(benchmarks) * len(all_seeds)))}")
for idx in range(1, 4):
    print(f"SHARD{idx}_SEEDS_CSV={shlex.quote(cfg.get(f'shard{idx}_seeds_csv', ''))}")
PY
)
EOF
  fi
fi

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

count_csv_items() {
  local csv="$1"
  if [[ -z "$csv" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"$csv"
}

fmt_duration() {
  local total="${1:-0}"
  local days hours mins secs
  if [[ "$total" -lt 0 ]]; then
    total=0
  fi
  days=$((total / 86400))
  hours=$(((total % 86400) / 3600))
  mins=$(((total % 3600) / 60))
  secs=$((total % 60))
  if [[ "$days" -gt 0 ]]; then
    printf '%dd %02dh %02dm %02ds' "$days" "$hours" "$mins" "$secs"
  else
    printf '%02dh %02dm %02ds' "$hours" "$mins" "$secs"
  fi
}

count_completed_runs() {
  find "$CAMPAIGN_ROOT/children" -name summary.json 2>/dev/null | wc -l | tr -d '[:space:]'
}

current_completed="$(count_completed_runs)"
expected_runs="${TOTAL_EXPECTED_RUNS:-0}"
now_epoch="$(date '+%s')"
elapsed_seconds=$((now_epoch - START_EPOCH))

progress_pct="0.0"
eta_text="n/a"
if [[ "$expected_runs" -gt 0 ]]; then
  progress_pct="$(awk -v done="$current_completed" -v total="$expected_runs" 'BEGIN { printf "%.1f", (100.0 * done) / total }')"
fi
if [[ "$current_completed" -gt 0 && "$expected_runs" -gt "$current_completed" && "$elapsed_seconds" -gt 0 ]]; then
  eta_seconds=$(( elapsed_seconds * (expected_runs - current_completed) / current_completed ))
  eta_text="$(fmt_duration "$eta_seconds")"
fi
if [[ "$current_completed" -ge "$expected_runs" && "$expected_runs" -gt 0 ]]; then
  eta_text="done"
fi

launcher_alive="no"
if [[ -n "${LAUNCHER_PID:-}" ]] && kill -0 "$LAUNCHER_PID" 2>/dev/null; then
  launcher_alive="yes"
fi

echo ""
echo "============================================================"
echo "  MORPHIN Full 256 Parallel Monitor"
echo "  $(timestamp)"
echo "  Campaign root : $CAMPAIGN_ROOT"
echo "  State         : ${STATE:-unknown}"
echo "  Launcher PID  : ${LAUNCHER_PID:-n/a} (alive: $launcher_alive)"
if [[ -n "${RUN_LOG_FILE:-}" ]]; then
  echo "  Launch log    : $RUN_LOG_FILE"
fi
echo "  Progress      : $current_completed / $expected_runs runs ($progress_pct%)"
echo "  Elapsed       : $(fmt_duration "$elapsed_seconds")"
echo "  ETA           : $eta_text"
echo "============================================================"
echo ""

for idx in 1 2 3; do
  shard_name_var="SHARD${idx}_NAME"
  shard_pid_var="SHARD${idx}_PID"
  shard_status_var="SHARD${idx}_STATUS"
  shard_log_var="SHARD${idx}_LOG_FILE"
  shard_seeds_var="SHARD${idx}_SEEDS_CSV"

  shard_name="${!shard_name_var:-shard$idx}"
  shard_pid="${!shard_pid_var:-}"
  shard_status="${!shard_status_var:-unknown}"
  shard_log="${!shard_log_var:-$CAMPAIGN_ROOT/shard_logs/$shard_name.log}"
  shard_seeds_csv="${!shard_seeds_var:-}"
  shard_root="$CAMPAIGN_ROOT/children/$shard_name/session_$SESSION_TS"
  shard_plan="$shard_root/experiment_plan.csv"
  shard_expected=0
  shard_completed=0
  shard_alive="no"
  last_line="-"

  if [[ "${TOTAL_METHODS:-0}" -gt 0 && "${TOTAL_BENCHMARKS:-0}" -gt 0 && -n "$shard_seeds_csv" ]]; then
    shard_expected=$(( TOTAL_METHODS * TOTAL_BENCHMARKS * $(count_csv_items "$shard_seeds_csv") ))
  elif [[ -f "$shard_plan" ]]; then
    shard_expected=$(( $(wc -l < "$shard_plan") - 1 ))
  fi
  if [[ -d "$shard_root/runs" ]]; then
    shard_completed="$(find "$shard_root/runs" -name summary.json 2>/dev/null | wc -l | tr -d '[:space:]')"
  fi
  if [[ -n "$shard_pid" ]] && kill -0 "$shard_pid" 2>/dev/null; then
    shard_alive="yes"
  fi
  if [[ -f "$shard_log" ]]; then
    last_line="$(tail -n 1 "$shard_log" 2>/dev/null || true)"
    if [[ -z "$last_line" ]]; then
      last_line="-"
    fi
  fi

  echo "**$shard_name**"
  echo "status=$shard_status pid=${shard_pid:-n/a} alive=$shard_alive progress=$shard_completed/$shard_expected"
  echo "last: $last_line"
  echo ""
done
