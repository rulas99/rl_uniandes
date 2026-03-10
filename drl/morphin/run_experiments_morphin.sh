#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_PROFILE="${RUN_PROFILE:-tier0_quick}"
BENCHMARK="${BENCHMARK:-gw_hidden_goal_aba_v1}"
SEEDS_CSV="${SEEDS_CSV:-42,43,44}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/morphin_gridworld}"
METHODS_CSV="${METHODS_CSV:-}"
TASK_IDS_CSV="${TASK_IDS_CSV:-}"
SCRATCH_TASK_IDS_CSV="${SCRATCH_TASK_IDS_CSV:-}"
SCRATCH_REFS_JSON="${SCRATCH_REFS_JSON:-}"
AUTO_BUILD_SCRATCH_REFS="${AUTO_BUILD_SCRATCH_REFS:-0}"

case "$RUN_PROFILE" in
  scratch)
    MODE="scratch_task"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
    EVAL_EPISODES="${EVAL_EPISODES:-10}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-10}"
    WARMUP_STEPS="${WARMUP_STEPS:-250}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-5000}"
    OBS_MODE="${OBS_MODE:-agent_only}"
    ;;
  tier0_main)
    MODE="continual"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
    EVAL_EPISODES="${EVAL_EPISODES:-20}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-10000}"
    ;;
  tier0_pipeline)
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
    EVAL_EPISODES="${EVAL_EPISODES:-20}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-10000}"
    ;;
  *)
    MODE="continual"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-150}"
    EVAL_EPISODES="${EVAL_EPISODES:-10}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-10}"
    WARMUP_STEPS="${WARMUP_STEPS:-250}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-5000}"
    ;;
esac

case "$BENCHMARK" in
  gw_hidden_goal_aba_v1)
    DEFAULT_OBS_MODE="agent_only"
    DEFAULT_SCRATCH_TASKS="gw_goal_a,gw_goal_b"
    ;;
  gw_goal_conditioned_aba_v1|gw_goal_switch_aba_v1)
    DEFAULT_OBS_MODE="agent_target"
    DEFAULT_SCRATCH_TASKS="gw_goal_a,gw_goal_b"
    ;;
  gw_goal_switch_abca_v1)
    DEFAULT_OBS_MODE="agent_target"
    DEFAULT_SCRATCH_TASKS="gw_goal_a,gw_goal_b,gw_goal_c"
    ;;
  gw_dynamics_switch_aba_v1)
    DEFAULT_OBS_MODE="grid_channels"
    DEFAULT_SCRATCH_TASKS="gw_dyn_a,gw_dyn_b"
    ;;
  *)
    echo "Unknown benchmark: $BENCHMARK" >&2
    exit 1
    ;;
esac

OBS_MODE="${OBS_MODE:-$DEFAULT_OBS_MODE}"
SCRATCH_TASK_IDS_CSV="${SCRATCH_TASK_IDS_CSV:-$DEFAULT_SCRATCH_TASKS}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
TAU="${TAU:-0.005}"
GAMMA="${GAMMA:-0.99}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-0.8}"
DETECTOR_MAX_DELAY_EPISODES="${DETECTOR_MAX_DELAY_EPISODES:-25}"

if [[ -z "$METHODS_CSV" ]]; then
  if [[ "$RUN_PROFILE" == "scratch" ]]; then
    METHODS_CSV="ddqn_scratch"
  else
    METHODS_CSV="ddqn_vanilla,oracle_reset,morphin_lite,detector_reset_only,oracle_segmented,morphin_segmented"
  fi
fi

SESSION_ID="$(date +%Y%m%d_%H%M%S)"
SESSION_ROOT="$LOG_ROOT/$BENCHMARK/session_$SESSION_ID"
RUNS_ROOT="$SESSION_ROOT/runs"
ANALYSIS_DIR="$SESSION_ROOT/analysis"
mkdir -p "$RUNS_ROOT" "$ANALYSIS_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

py_run() {
  PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$@"
}

run_train() {
  py_run -m rl.rl_uniandes.drl.morphin.train_continual "$@"
}

log "Session root: $SESSION_ROOT"
log "Profile: $RUN_PROFILE | Mode: $MODE | Benchmark: $BENCHMARK | Seeds: $SEEDS_CSV"
log "Obs mode: $OBS_MODE | Methods: $METHODS_CSV"
log "Episodes/task: $EPISODES_PER_TASK | Eval every: $EVAL_EVERY_EPISODES | Eval episodes: $EVAL_EPISODES"
log "Warmup: $WARMUP_STEPS | Batch: $BATCH_SIZE | Eps decay: $EPS_DECAY_STEPS | Auto scratch refs: $AUTO_BUILD_SCRATCH_REFS"

PLAN_CSV="$SESSION_ROOT/experiment_plan.csv"
MANIFEST_CSV="$SESSION_ROOT/experiment_manifest.csv"
printf 'stage,method,seed,mode,benchmark,obs_mode,task_ids_csv,run_dir,status\n' >"$PLAN_CSV"
printf 'stage,method,seed,mode,benchmark,obs_mode,task_ids_csv,run_dir,status,summary_json\n' >"$MANIFEST_CSV"

IFS=',' read -r -a SEEDS <<<"$SEEDS_CSV"
IFS=',' read -r -a METHODS <<<"$METHODS_CSV"
IFS=',' read -r -a SCRATCH_TASKS <<<"$SCRATCH_TASK_IDS_CSV"

common_args=(
  --benchmark "$BENCHMARK"
  --episodes-per-task "$EPISODES_PER_TASK"
  --eval-episodes "$EVAL_EPISODES"
  --eval-every-episodes "$EVAL_EVERY_EPISODES"
  --obs-mode "$OBS_MODE"
  --warmup-steps "$WARMUP_STEPS"
  --batch-size "$BATCH_SIZE"
  --eps-decay-steps "$EPS_DECAY_STEPS"
  --learning-rate "$LEARNING_RATE"
  --tau "$TAU"
  --gamma "$GAMMA"
  --success-threshold "$SUCCESS_THRESHOLD"
  --detector-max-delay-episodes "$DETECTOR_MAX_DELAY_EPISODES"
)

run_scratch_stage() {
  local run_idx=0
  local total_runs=$(( ${#SEEDS[@]} * ${#SCRATCH_TASKS[@]} ))
  for seed in "${SEEDS[@]}"; do
    for task_id in "${SCRATCH_TASKS[@]}"; do
      run_idx=$((run_idx + 1))
      local run_name="scratch_${task_id}_seed${seed}"
      local run_dir="$RUNS_ROOT/$run_name"
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "scratch" "ddqn_scratch" "$seed" "scratch_task" "$BENCHMARK" "$OBS_MODE" "$task_id" "$run_dir" "planned" >>"$PLAN_CSV"
      log "[scratch $run_idx/$total_runs] task=$task_id seed=$seed"
      if run_train \
        --mode scratch_task \
        --method ddqn_scratch \
        --task-id "$task_id" \
        --seed "$seed" \
        --log-dir "$RUNS_ROOT" \
        --run-name "$run_name" \
        "${common_args[@]}"; then
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
          "scratch" "ddqn_scratch" "$seed" "scratch_task" "$BENCHMARK" "$OBS_MODE" "$task_id" "$run_dir" "ok" "$run_dir/summary.json" >>"$MANIFEST_CSV"
      else
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
          "scratch" "ddqn_scratch" "$seed" "scratch_task" "$BENCHMARK" "$OBS_MODE" "$task_id" "$run_dir" "failed" "" >>"$MANIFEST_CSV"
      fi
    done
  done
}

run_continual_stage() {
  local run_idx=0
  local total_runs=$(( ${#SEEDS[@]} * ${#METHODS[@]} ))
  for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      run_idx=$((run_idx + 1))
      local run_name="${method}_seed${seed}"
      local run_dir="$RUNS_ROOT/$run_name"
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "continual" "$method" "$seed" "continual" "$BENCHMARK" "$OBS_MODE" "$TASK_IDS_CSV" "$run_dir" "planned" >>"$PLAN_CSV"
      log "[continual $run_idx/$total_runs] method=$method seed=$seed"
      local args=(
        --mode continual
        --method "$method"
        --seed "$seed"
        --log-dir "$RUNS_ROOT"
        --run-name "$run_name"
        "${common_args[@]}"
      )
      if [[ -n "$TASK_IDS_CSV" ]]; then
        args+=(--task-ids-csv "$TASK_IDS_CSV")
      fi
      if [[ -n "$SCRATCH_REFS_JSON" ]]; then
        args+=(--scratch-summary-json "$SCRATCH_REFS_JSON")
      fi
      if run_train "${args[@]}"; then
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
          "continual" "$method" "$seed" "continual" "$BENCHMARK" "$OBS_MODE" "$TASK_IDS_CSV" "$run_dir" "ok" "$run_dir/summary.json" >>"$MANIFEST_CSV"
      else
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
          "continual" "$method" "$seed" "continual" "$BENCHMARK" "$OBS_MODE" "$TASK_IDS_CSV" "$run_dir" "failed" "" >>"$MANIFEST_CSV"
      fi
    done
  done
}

if [[ "$MODE" == "scratch_task" ]]; then
  run_scratch_stage
else
  if [[ "$AUTO_BUILD_SCRATCH_REFS" == "1" && -z "$SCRATCH_REFS_JSON" ]]; then
    log "Running scratch stage to build scratch references"
    run_scratch_stage
    SCRATCH_REFS_JSON="$SESSION_ROOT/scratch_refs.json"
    py_run -m rl.rl_uniandes.drl.morphin.build_scratch_refs \
      --root-dir "$RUNS_ROOT" \
      --output-json "$SCRATCH_REFS_JSON"
    log "Scratch refs built at $SCRATCH_REFS_JSON"
  fi
  run_continual_stage
fi

log "Aggregating session outputs"
py_run -m rl.rl_uniandes.drl.morphin.aggregate_morphin_results \
  --root-dir "$RUNS_ROOT" \
  --output-dir "$ANALYSIS_DIR"

cat >"$SESSION_ROOT/session_config.json" <<JSON
{
  "run_profile": "$RUN_PROFILE",
  "mode": "$MODE",
  "benchmark": "$BENCHMARK",
  "obs_mode": "$OBS_MODE",
  "seeds_csv": "$SEEDS_CSV",
  "methods_csv": "$METHODS_CSV",
  "task_ids_csv": "$TASK_IDS_CSV",
  "scratch_task_ids_csv": "$SCRATCH_TASK_IDS_CSV",
  "episodes_per_task": $EPISODES_PER_TASK,
  "eval_episodes": $EVAL_EPISODES,
  "eval_every_episodes": $EVAL_EVERY_EPISODES,
  "warmup_steps": $WARMUP_STEPS,
  "batch_size": $BATCH_SIZE,
  "eps_decay_steps": $EPS_DECAY_STEPS,
  "learning_rate": $LEARNING_RATE,
  "auto_build_scratch_refs": $AUTO_BUILD_SCRATCH_REFS,
  "scratch_refs_json": "${SCRATCH_REFS_JSON}"
}
JSON

log "Done. Session root: $SESSION_ROOT"
