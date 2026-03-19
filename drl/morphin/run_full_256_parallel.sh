#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_PROFILE="${RUN_PROFILE:-thesis_9x9_full_256}"
CAMPAIGN_GROUP="${CAMPAIGN_GROUP:-thesis_9x9_full_256_parallel}"
BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw9_goal_balanced_ab_v1,gw9_goal_balanced_ac_v1,gw9_goal_balanced_aba_v1,gw9_goal_balanced_abc_v1}"
METHODS_CSV="${METHODS_CSV:-ddqn_vanilla,oracle_reset,oracle_segmented,oracle_segmented_distill_l001,der_plus_plus}"

SHARD1_SEEDS_CSV="${SHARD1_SEEDS_CSV:-42,43,44,45,46,47,48,49,50,51,52,53}"
SHARD2_SEEDS_CSV="${SHARD2_SEEDS_CSV:-54,55,56,57,58,59,60,61,62,63,64,65}"
SHARD3_SEEDS_CSV="${SHARD3_SEEDS_CSV:-66,67,68,69,70,71,72,73,74,75,76,77}"
SCRATCH_SEEDS_CSV="${SCRATCH_SEEDS_CSV:-$SHARD1_SEEDS_CSV}"

OBS_MODE="${OBS_MODE:-agent_target}"
SCRATCH_TASK_IDS_CSV="${SCRATCH_TASK_IDS_CSV:-gw9_goal_bal_a,gw9_goal_bal_b,gw9_goal_bal_c}"
ARCHIVE_FRAC="${ARCHIVE_FRAC:-0.10}"
DER_ALPHA="${DER_ALPHA:-0.01}"
DER_BETA="${DER_BETA:-1.0}"
DER_CAPACITY="${DER_CAPACITY:-0}"

EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-250}"
SCRATCH_EPISODES_PER_TASK="${SCRATCH_EPISODES_PER_TASK:-500}"
SCRATCH_MAX_STEPS_PER_EPISODE="${SCRATCH_MAX_STEPS_PER_EPISODE:-250}"
EVAL_EPISODES="${EVAL_EPISODES:-15}"
EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
EVAL_DENSE_EVERY_EPISODES="${EVAL_DENSE_EVERY_EPISODES:-1}"
EVAL_DENSE_WINDOW_EPISODES="${EVAL_DENSE_WINDOW_EPISODES:-25}"
WARMUP_STEPS="${WARMUP_STEPS:-750}"
SCRATCH_WARMUP_STEPS="${SCRATCH_WARMUP_STEPS:-750}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-15000}"
SCRATCH_EPS_DECAY_STEPS="${SCRATCH_EPS_DECAY_STEPS:-15000}"
EPS_RESET_VALUE="${EPS_RESET_VALUE:-0.9}"
EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-30000}"
POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-7500}"
HIDDEN_SIZES_CSV="${HIDDEN_SIZES_CSV:-256,256}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-20000}"
RECENT_BUFFER_CAPACITY="${RECENT_BUFFER_CAPACITY:-12000}"
ARCHIVE_BUFFER_CAPACITY="${ARCHIVE_BUFFER_CAPACITY:-8000}"
SCRATCH_BUFFER_CAPACITY="${SCRATCH_BUFFER_CAPACITY:-20000}"
SCRATCH_REF_MIN_FINAL_SUCCESS="${SCRATCH_REF_MIN_FINAL_SUCCESS:-0.8}"
SCRATCH_REF_MIN_VALID_RUNS="${SCRATCH_REF_MIN_VALID_RUNS:-3}"
SCRATCH_REF_MIN_VALID_FRACTION="${SCRATCH_REF_MIN_VALID_FRACTION:-0.7}"

TS="$(date '+%Y%m%d_%H%M%S')"
CAMPAIGN_ROOT="$SCRIPT_DIR/logs/morphin_gridworld/$CAMPAIGN_GROUP/session_$TS"
CHILD_LOG_ROOT="$CAMPAIGN_ROOT/children"
COMBINED_RUNS_ROOT="$CAMPAIGN_ROOT/runs"
COMBINED_ANALYSIS_DIR="$CAMPAIGN_ROOT/analysis"
SHARD_LOG_DIR="$CAMPAIGN_ROOT/shard_logs"
mkdir -p "$CHILD_LOG_ROOT" "$COMBINED_RUNS_ROOT" "$COMBINED_ANALYSIS_DIR" "$SHARD_LOG_DIR"

if [[ -z "${SHARED_SCRATCH_REFS_JSON:-}" ]]; then
  SHARED_SCRATCH_REFS_JSON="$CAMPAIGN_ROOT/shared_scratch_refs.json"
fi

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

py_run() {
  PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$@"
}

extract_session_root() {
  local log_file="$1"
  rg -o 'Session root: .*' "$log_file" | tail -n1 | sed 's/.*Session root: //'
}

copy_run_tree() {
  local src_runs="$1"
  local dest_runs="$2"
  if ! cp -al "$src_runs/." "$dest_runs/"; then
    cp -a "$src_runs/." "$dest_runs/"
  fi
}

run_shared_scratch_refs() {
  local scratch_log="$SHARD_LOG_DIR/shared_scratch.log"
  if [[ -f "$SHARED_SCRATCH_REFS_JSON" ]]; then
    log "Reusing existing shared scratch refs: $SHARED_SCRATCH_REFS_JSON"
    return 0
  fi

  log "Building shared scratch refs once for tasks: $SCRATCH_TASK_IDS_CSV"
  (
    LOG_ROOT="$CHILD_LOG_ROOT" \
    RUN_PROFILE="scratch" \
    SESSION_GROUP="shared_scratch" \
    BENCHMARKS_CSV="gw9_goal_balanced_ab_v1" \
    SEEDS_CSV="$SCRATCH_SEEDS_CSV" \
    OBS_MODE="$OBS_MODE" \
    SCRATCH_TASK_IDS_CSV="$SCRATCH_TASK_IDS_CSV" \
    EPISODES_PER_TASK="$SCRATCH_EPISODES_PER_TASK" \
    MAX_STEPS_PER_EPISODE="$SCRATCH_MAX_STEPS_PER_EPISODE" \
    EVAL_EPISODES="$EVAL_EPISODES" \
    EVAL_EVERY_EPISODES="$EVAL_EVERY_EPISODES" \
    EVAL_DENSE_EVERY_EPISODES="$EVAL_DENSE_EVERY_EPISODES" \
    EVAL_DENSE_WINDOW_EPISODES="$EVAL_DENSE_WINDOW_EPISODES" \
    WARMUP_STEPS="$SCRATCH_WARMUP_STEPS" \
    BATCH_SIZE="$BATCH_SIZE" \
    EPS_DECAY_STEPS="$SCRATCH_EPS_DECAY_STEPS" \
    HIDDEN_SIZES_CSV="$HIDDEN_SIZES_CSV" \
    BUFFER_CAPACITY="$SCRATCH_BUFFER_CAPACITY" \
    bash run_experiments_morphin.sh
  ) >"$scratch_log" 2>&1

  local scratch_session_root
  scratch_session_root="$(extract_session_root "$scratch_log")"
  if [[ -z "$scratch_session_root" || ! -d "$scratch_session_root/runs" ]]; then
    log "Could not resolve scratch session root from $scratch_log"
    return 1
  fi

  py_run -m rl.rl_uniandes.drl.morphin.build_scratch_refs \
    --root-dir "$scratch_session_root/runs" \
    --output-json "$SHARED_SCRATCH_REFS_JSON" \
    --min-final-success "$SCRATCH_REF_MIN_FINAL_SUCCESS" \
    --min-valid-runs "$SCRATCH_REF_MIN_VALID_RUNS" \
    --min-valid-fraction "$SCRATCH_REF_MIN_VALID_FRACTION" \
    >"$SHARD_LOG_DIR/shared_scratch_refs_build.log" 2>&1

  log "Shared scratch refs ready at $SHARED_SCRATCH_REFS_JSON"
}

launch_shard() {
  local shard_name="$1"
  local seeds_csv="$2"
  local shard_log="$SHARD_LOG_DIR/${shard_name}.log"
  log "Launching $shard_name with seeds: $seeds_csv"
  (
    LOG_ROOT="$CHILD_LOG_ROOT" \
    RUN_PROFILE="$RUN_PROFILE" \
    SESSION_GROUP="$shard_name" \
    METHOD_SET="manual_full_256_plus_der" \
    METHODS_CSV="$METHODS_CSV" \
    BENCHMARKS_CSV="$BENCHMARKS_CSV" \
    SEEDS_CSV="$seeds_csv" \
    OBS_MODE="$OBS_MODE" \
    AUTO_BUILD_SCRATCH_REFS=0 \
    SCRATCH_REFS_JSON="$SHARED_SCRATCH_REFS_JSON" \
    ARCHIVE_FRAC="$ARCHIVE_FRAC" \
    DER_ALPHA="$DER_ALPHA" \
    DER_BETA="$DER_BETA" \
    DER_CAPACITY="$DER_CAPACITY" \
    EPISODES_PER_TASK="$EPISODES_PER_TASK" \
    MAX_STEPS_PER_EPISODE="$MAX_STEPS_PER_EPISODE" \
    EVAL_EPISODES="$EVAL_EPISODES" \
    EVAL_EVERY_EPISODES="$EVAL_EVERY_EPISODES" \
    EVAL_DENSE_EVERY_EPISODES="$EVAL_DENSE_EVERY_EPISODES" \
    EVAL_DENSE_WINDOW_EPISODES="$EVAL_DENSE_WINDOW_EPISODES" \
    WARMUP_STEPS="$WARMUP_STEPS" \
    BATCH_SIZE="$BATCH_SIZE" \
    EPS_DECAY_STEPS="$EPS_DECAY_STEPS" \
    EPS_RESET_VALUE="$EPS_RESET_VALUE" \
    EPS_DECAY_STEPS_AFTER_SWITCH="$EPS_DECAY_STEPS_AFTER_SWITCH" \
    POST_SWITCH_STEPS="$POST_SWITCH_STEPS" \
    HIDDEN_SIZES_CSV="$HIDDEN_SIZES_CSV" \
    BUFFER_CAPACITY="$BUFFER_CAPACITY" \
    RECENT_BUFFER_CAPACITY="$RECENT_BUFFER_CAPACITY" \
    ARCHIVE_BUFFER_CAPACITY="$ARCHIVE_BUFFER_CAPACITY" \
    bash run_experiments_morphin.sh
  ) >"$shard_log" 2>&1 &
  echo $!
}

aggregate_combined_session() {
  local child_sessions_csv="$CAMPAIGN_ROOT/child_sessions.csv"
  printf 'shard_name,seeds_csv,status,session_root,log_file\n' >"$child_sessions_csv"

  local shard_name seeds_csv shard_log shard_session_root
  local -n names_ref=$1
  local -n seeds_ref=$2
  local -n statuses_ref=$3

  for idx in "${!names_ref[@]}"; do
    shard_name="${names_ref[$idx]}"
    seeds_csv="${seeds_ref[$idx]}"
    shard_log="$SHARD_LOG_DIR/${shard_name}.log"
    shard_session_root="$(extract_session_root "$shard_log" || true)"
    printf '%s,%s,%s,%s,%s\n' \
      "$shard_name" \
      "$seeds_csv" \
      "${statuses_ref[$idx]}" \
      "$shard_session_root" \
      "$shard_log" >>"$child_sessions_csv"
    if [[ "${statuses_ref[$idx]}" == "ok" && -d "$shard_session_root/runs" ]]; then
      copy_run_tree "$shard_session_root/runs" "$COMBINED_RUNS_ROOT"
    fi
  done

  log "Aggregating combined session"
  py_run -m rl.rl_uniandes.drl.morphin.aggregate_morphin_results \
    --root-dir "$COMBINED_RUNS_ROOT" \
    --output-dir "$COMBINED_ANALYSIS_DIR" \
    >"$CAMPAIGN_ROOT/aggregate.log" 2>&1

  IFS=',' read -r -a benchmarks <<<"$BENCHMARKS_CSV"
  for benchmark in "${benchmarks[@]}"; do
    local benchmark_analysis_dir="$COMBINED_ANALYSIS_DIR/by_benchmark/$benchmark"
    mkdir -p "$benchmark_analysis_dir"
    py_run -m rl.rl_uniandes.drl.morphin.aggregate_morphin_results \
      --root-dir "$COMBINED_RUNS_ROOT/$benchmark" \
      --output-dir "$benchmark_analysis_dir" \
      >"$CAMPAIGN_ROOT/aggregate_${benchmark}.log" 2>&1
  done

  py_run "$SCRIPT_DIR/thesis_stats.py" "$CAMPAIGN_ROOT" \
    >"$CAMPAIGN_ROOT/thesis_stats.log" 2>&1
  py_run "$SCRIPT_DIR/thesis_plots.py" "$CAMPAIGN_ROOT" \
    >"$CAMPAIGN_ROOT/thesis_plots.log" 2>&1
}

write_campaign_config() {
  local combined_seeds_csv
  combined_seeds_csv="${SHARD1_SEEDS_CSV},${SHARD2_SEEDS_CSV},${SHARD3_SEEDS_CSV}"
  cat >"$CAMPAIGN_ROOT/session_config.json" <<JSON
{
  "campaign_group": "$CAMPAIGN_GROUP",
  "run_profile": "$RUN_PROFILE",
  "benchmarks_csv": "$BENCHMARKS_CSV",
  "methods_csv": "$METHODS_CSV",
  "obs_mode": "$OBS_MODE",
  "archive_frac": $ARCHIVE_FRAC,
  "der_alpha": $DER_ALPHA,
  "der_beta": $DER_BETA,
  "der_capacity": $DER_CAPACITY,
  "episodes_per_task": $EPISODES_PER_TASK,
  "max_steps_per_episode": $MAX_STEPS_PER_EPISODE,
  "eval_episodes": $EVAL_EPISODES,
  "eval_every_episodes": $EVAL_EVERY_EPISODES,
  "eval_dense_every_episodes": $EVAL_DENSE_EVERY_EPISODES,
  "eval_dense_window_episodes": $EVAL_DENSE_WINDOW_EPISODES,
  "warmup_steps": $WARMUP_STEPS,
  "batch_size": $BATCH_SIZE,
  "eps_decay_steps": $EPS_DECAY_STEPS,
  "eps_reset_value": $EPS_RESET_VALUE,
  "eps_decay_steps_after_switch": $EPS_DECAY_STEPS_AFTER_SWITCH,
  "post_switch_steps": $POST_SWITCH_STEPS,
  "hidden_sizes_csv": "$HIDDEN_SIZES_CSV",
  "buffer_capacity": $BUFFER_CAPACITY,
  "recent_buffer_capacity": $RECENT_BUFFER_CAPACITY,
  "archive_buffer_capacity": $ARCHIVE_BUFFER_CAPACITY,
  "shared_scratch_refs_json": "$SHARED_SCRATCH_REFS_JSON",
  "scratch_seeds_csv": "$SCRATCH_SEEDS_CSV",
  "scratch_task_ids_csv": "$SCRATCH_TASK_IDS_CSV",
  "all_seeds_csv": "$combined_seeds_csv",
  "shard1_seeds_csv": "$SHARD1_SEEDS_CSV",
  "shard2_seeds_csv": "$SHARD2_SEEDS_CSV",
  "shard3_seeds_csv": "$SHARD3_SEEDS_CSV"
}
JSON
}

echo ""
echo "============================================================"
echo "  MORPHIN Full 256 Parallel Overnight"
echo "  $(timestamp)"
echo "  Campaign    : $CAMPAIGN_GROUP"
echo "  Benchmarks  : $BENCHMARKS_CSV"
echo "  Methods     : $METHODS_CSV"
echo "  Archive frac: $ARCHIVE_FRAC"
echo "  DER alpha   : $DER_ALPHA"
echo "  Shard 1     : $SHARD1_SEEDS_CSV"
echo "  Shard 2     : $SHARD2_SEEDS_CSV"
echo "  Shard 3     : $SHARD3_SEEDS_CSV"
echo "  Scratch     : $SCRATCH_SEEDS_CSV"
echo "  Campaign dir: $CAMPAIGN_ROOT"
echo "============================================================"
echo ""

write_campaign_config
run_shared_scratch_refs

declare -a SHARD_NAMES=("shard1" "shard2" "shard3")
declare -a SHARD_SEEDS=("$SHARD1_SEEDS_CSV" "$SHARD2_SEEDS_CSV" "$SHARD3_SEEDS_CSV")
declare -a SHARD_PIDS=()
declare -a SHARD_STATUS=()

for idx in "${!SHARD_NAMES[@]}"; do
  SHARD_PIDS+=("$(launch_shard "${SHARD_NAMES[$idx]}" "${SHARD_SEEDS[$idx]}")")
  SHARD_STATUS+=("running")
done

overall_exit=0
for idx in "${!SHARD_PIDS[@]}"; do
  if wait "${SHARD_PIDS[$idx]}"; then
    SHARD_STATUS[$idx]="ok"
    log "${SHARD_NAMES[$idx]} finished successfully"
  else
    SHARD_STATUS[$idx]="failed"
    overall_exit=1
    log "${SHARD_NAMES[$idx]} failed"
  fi
done

aggregate_combined_session SHARD_NAMES SHARD_SEEDS SHARD_STATUS

echo ""
echo "============================================================"
if [[ $overall_exit -eq 0 ]]; then
  echo "  [$(timestamp)] DONE — parallel overnight campaign"
else
  echo "  [$(timestamp)] PARTIAL/FAILED — review shard logs"
fi
echo "============================================================"
echo ""
echo "  Campaign root : $CAMPAIGN_ROOT"
echo "  Report        : $COMBINED_ANALYSIS_DIR/report.md"
echo "  Thesis tables : $COMBINED_ANALYSIS_DIR/thesis/tables"
echo "  Thesis figures: $COMBINED_ANALYSIS_DIR/thesis/figures"
echo "  Child sessions: $CAMPAIGN_ROOT/child_sessions.csv"
echo ""

exit $overall_exit
