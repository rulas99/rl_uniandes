#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_PROFILE="${RUN_PROFILE:-tier0_quick}"
BENCHMARK="${BENCHMARK:-gw_goal_conditioned_balanced_ac_v1}"
BENCHMARKS_CSV="${BENCHMARKS_CSV:-}"
SESSION_GROUP="${SESSION_GROUP:-}"
SEEDS_CSV="${SEEDS_CSV:-42,43,44}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/morphin_gridworld}"
METHODS_CSV="${METHODS_CSV:-}"
METHOD_SET="${METHOD_SET:-main}"
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
  tier0_campaign)
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw_goal_conditioned_balanced_ac_v1,gw_goal_conditioned_balanced_ab_v1,gw_goal_conditioned_balanced_ca_v1}"
    SESSION_GROUP="${SESSION_GROUP:-campaign_goal_transfer}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
    EVAL_EPISODES="${EVAL_EPISODES:-20}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-10000}"
    ;;
  tier1_hidden_revisit)
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    METHOD_SET="${METHOD_SET:-core_no_detector}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw_hidden_goal_balanced_ab_v1,gw_hidden_goal_balanced_aba_v1,gw_dynamics_switch_aba_v1}"
    SESSION_GROUP="${SESSION_GROUP:-campaign_hidden_revisit}"
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

benchmark_defaults() {
  local benchmark="$1"
  case "$benchmark" in
    gw_hidden_goal_aba_v1)
      printf '%s\n%s\n' "agent_only" "gw_goal_a,gw_goal_b"
      ;;
    gw_hidden_goal_balanced_aba_v1|gw_hidden_goal_balanced_ab_v1)
      printf '%s\n%s\n' "agent_only" "gw_goal_bal_a,gw_goal_bal_b"
      ;;
    gw_hidden_goal_balanced_ac_v1)
      printf '%s\n%s\n' "agent_only" "gw_goal_bal_a,gw_goal_bal_c"
      ;;
    gw_goal_conditioned_aba_v1|gw_goal_switch_aba_v1)
      printf '%s\n%s\n' "agent_target" "gw_goal_a,gw_goal_b"
      ;;
    gw_goal_conditioned_balanced_ab_v1|gw_goal_conditioned_balanced_aba_v1)
      printf '%s\n%s\n' "agent_target" "gw_goal_bal_a,gw_goal_bal_b"
      ;;
    gw_goal_conditioned_balanced_ac_v1|gw_goal_conditioned_balanced_ca_v1)
      printf '%s\n%s\n' "agent_target" "gw_goal_bal_a,gw_goal_bal_c"
      ;;
    gw_goal_conditioned_balanced_acb_v1)
      printf '%s\n%s\n' "agent_target" "gw_goal_bal_a,gw_goal_bal_b,gw_goal_bal_c"
      ;;
    gw_goal_switch_abca_v1)
      printf '%s\n%s\n' "agent_target" "gw_goal_a,gw_goal_b,gw_goal_c"
      ;;
    gw_dynamics_switch_aba_v1)
      printf '%s\n%s\n' "grid_channels" "gw_dyn_a,gw_dyn_b"
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ -n "$BENCHMARKS_CSV" ]]; then
  IFS=',' read -r -a BENCHMARKS <<<"$BENCHMARKS_CSV"
else
  BENCHMARKS=("$BENCHMARK")
fi

if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
  echo "No benchmarks configured" >&2
  exit 1
fi

for benchmark_name in "${BENCHMARKS[@]}"; do
  if ! benchmark_defaults "$benchmark_name" >/dev/null; then
    echo "Unknown benchmark: $benchmark_name" >&2
    exit 1
  fi
done

readarray -t DEFAULTS < <(benchmark_defaults "${BENCHMARKS[0]}")
DEFAULT_OBS_MODE="${DEFAULTS[0]}"
DEFAULT_SCRATCH_TASKS="${DEFAULTS[1]}"
USER_OBS_MODE="${OBS_MODE:-}"
OBS_MODE="${USER_OBS_MODE:-$DEFAULT_OBS_MODE}"
SCRATCH_TASK_IDS_CSV="${SCRATCH_TASK_IDS_CSV:-$DEFAULT_SCRATCH_TASKS}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
TAU="${TAU:-0.005}"
GAMMA="${GAMMA:-0.99}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-0.8}"
THRESHOLD_MIN_CONSECUTIVE_EVALS="${THRESHOLD_MIN_CONSECUTIVE_EVALS:-2}"
DETECTOR_MAX_DELAY_EPISODES="${DETECTOR_MAX_DELAY_EPISODES:-25}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-150}"
EVAL_DENSE_EVERY_EPISODES="${EVAL_DENSE_EVERY_EPISODES:-1}"
EVAL_DENSE_WINDOW_EPISODES="${EVAL_DENSE_WINDOW_EPISODES:-25}"
SEGMENTED_KEEP_TAIL="${SEGMENTED_KEEP_TAIL:-512}"
SEGMENTED_RECENT_ONLY_STEPS="${SEGMENTED_RECENT_ONLY_STEPS:-1000}"
SEGMENTED_MIN_RECENT_SAMPLES="${SEGMENTED_MIN_RECENT_SAMPLES:-256}"
ARCHIVE_FRAC="${ARCHIVE_FRAC:-0.25}"
RECENT_MIX_START="${RECENT_MIX_START:-0.8}"
RECENT_MIX_END="${RECENT_MIX_END:-0.5}"
POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-5000}"
EPS_RESET_VALUE="${EPS_RESET_VALUE:-0.4}"
EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-2000}"
ALPHA_MAX_MULT="${ALPHA_MAX_MULT:-3.0}"
TD_K="${TD_K:-1.0}"
SCRATCH_REF_MIN_FINAL_SUCCESS="${SCRATCH_REF_MIN_FINAL_SUCCESS:-0.8}"
SCRATCH_REF_MIN_VALID_RUNS="${SCRATCH_REF_MIN_VALID_RUNS:-3}"
SCRATCH_REF_MIN_VALID_FRACTION="${SCRATCH_REF_MIN_VALID_FRACTION:-0.6}"

METHOD_SOURCE="set:${METHOD_SET}"
if [[ -z "$METHODS_CSV" ]]; then
  if [[ "$RUN_PROFILE" == "scratch" ]]; then
    METHODS_CSV="ddqn_scratch"
    METHOD_SOURCE="profile:scratch"
  else
    case "$METHOD_SET" in
      main)
        METHODS_CSV="ddqn_vanilla,oracle_reset,oracle_segmented,oracle_segmented_td"
        ;;
      core_no_detector)
        METHODS_CSV="ddqn_vanilla,oracle_reset,oracle_segmented_td,morphin_lite"
        ;;
      morphin_ablation)
        METHODS_CSV="ddqn_vanilla,oracle_reset,oracle_segmented,oracle_segmented_td,morphin_lite,morphin_full,morphin_segmented"
        ;;
      full)
        METHODS_CSV="ddqn_vanilla,oracle_reset,detector_reset_only,oracle_segmented,oracle_segmented_td,morphin_lite,morphin_full,morphin_segmented"
        ;;
      *)
        echo "Unknown METHOD_SET: $METHOD_SET" >&2
        exit 1
        ;;
    esac
  fi
else
  METHOD_SOURCE="manual_override"
fi

SESSION_ID="$(date +%Y%m%d_%H%M%S)"
if [[ ${#BENCHMARKS[@]} -gt 1 ]]; then
  SESSION_ROOT="$LOG_ROOT/${SESSION_GROUP:-multi_benchmark}/session_$SESSION_ID"
else
  SESSION_ROOT="$LOG_ROOT/${BENCHMARKS[0]}/session_$SESSION_ID"
fi
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
log "Profile: $RUN_PROFILE | Mode: $MODE | Benchmarks: ${BENCHMARKS[*]} | Seeds: $SEEDS_CSV"
log "Obs mode: $OBS_MODE | Method set: $METHOD_SET | Methods: $METHODS_CSV"
log "Episodes/task: $EPISODES_PER_TASK | Eval every: $EVAL_EVERY_EPISODES | Eval episodes: $EVAL_EPISODES"
log "Warmup: $WARMUP_STEPS | Batch: $BATCH_SIZE | Eps decay: $EPS_DECAY_STEPS | Auto scratch refs: $AUTO_BUILD_SCRATCH_REFS"
log "Success threshold: $SUCCESS_THRESHOLD | Consecutive evals for threshold: $THRESHOLD_MIN_CONSECUTIVE_EVALS"
log "Scratch refs: min_final_success=$SCRATCH_REF_MIN_FINAL_SUCCESS min_valid_runs=$SCRATCH_REF_MIN_VALID_RUNS min_valid_fraction=$SCRATCH_REF_MIN_VALID_FRACTION"
log "Switch epsilon reset: value=$EPS_RESET_VALUE decay_steps=$EPS_DECAY_STEPS_AFTER_SWITCH"
log "Replay: archive_frac=$ARCHIVE_FRAC recent_mix=$RECENT_MIX_START->$RECENT_MIX_END post_switch_steps=$POST_SWITCH_STEPS keep_tail=$SEGMENTED_KEEP_TAIL"
log "TD weighting: alpha_max_mult=$ALPHA_MAX_MULT td_k=$TD_K"
log "Dense eval: every $EVAL_DENSE_EVERY_EPISODES within $EVAL_DENSE_WINDOW_EPISODES eps after switch | Max steps/ep: $MAX_STEPS_PER_EPISODE"

PLAN_CSV="$SESSION_ROOT/experiment_plan.csv"
MANIFEST_CSV="$SESSION_ROOT/experiment_manifest.csv"
printf 'stage,method,seed,mode,benchmark,obs_mode,task_ids_csv,run_dir,status\n' >"$PLAN_CSV"
printf 'stage,method,seed,mode,benchmark,obs_mode,task_ids_csv,run_dir,status,summary_json\n' >"$MANIFEST_CSV"

IFS=',' read -r -a SEEDS <<<"$SEEDS_CSV"
IFS=',' read -r -a METHODS <<<"$METHODS_CSV"

build_common_args() {
  local benchmark="$1"
  local obs_mode="$2"
  printf '%s\0' \
    --benchmark "$benchmark" \
    --episodes-per-task "$EPISODES_PER_TASK" \
    --max-steps-per-episode "$MAX_STEPS_PER_EPISODE" \
    --eval-episodes "$EVAL_EPISODES" \
    --eval-every-episodes "$EVAL_EVERY_EPISODES" \
    --eval-dense-every-episodes "$EVAL_DENSE_EVERY_EPISODES" \
    --eval-dense-window-episodes "$EVAL_DENSE_WINDOW_EPISODES" \
    --obs-mode "$obs_mode" \
    --warmup-steps "$WARMUP_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --eps-decay-steps "$EPS_DECAY_STEPS" \
    --eps-reset-value "$EPS_RESET_VALUE" \
    --eps-decay-steps-after-switch "$EPS_DECAY_STEPS_AFTER_SWITCH" \
    --alpha-max-mult "$ALPHA_MAX_MULT" \
    --td-k "$TD_K" \
    --learning-rate "$LEARNING_RATE" \
    --tau "$TAU" \
    --gamma "$GAMMA" \
    --success-threshold "$SUCCESS_THRESHOLD" \
    --threshold-min-consecutive-evals "$THRESHOLD_MIN_CONSECUTIVE_EVALS" \
    --detector-max-delay-episodes "$DETECTOR_MAX_DELAY_EPISODES" \
    --archive-frac "$ARCHIVE_FRAC" \
    --recent-mix-start "$RECENT_MIX_START" \
    --recent-mix-end "$RECENT_MIX_END" \
    --post-switch-steps "$POST_SWITCH_STEPS" \
    --segmented-keep-tail "$SEGMENTED_KEEP_TAIL" \
    --segmented-recent-only-steps "$SEGMENTED_RECENT_ONLY_STEPS" \
    --segmented-min-recent-samples "$SEGMENTED_MIN_RECENT_SAMPLES"
}

run_scratch_stage_for_benchmark() {
  local benchmark="$1"
  local obs_mode="$2"
  local scratch_task_ids_csv="$3"
  local benchmark_runs_root="$4"
  IFS=',' read -r -a scratch_tasks <<<"$scratch_task_ids_csv"
  local run_idx=0
  local total_runs=$(( ${#SEEDS[@]} * ${#scratch_tasks[@]} ))
  local common_args=()
  mapfile -d '' -t common_args < <(build_common_args "$benchmark" "$obs_mode")
  for seed in "${SEEDS[@]}"; do
    for task_id in "${scratch_tasks[@]}"; do
      run_idx=$((run_idx + 1))
      local run_name="scratch_${task_id}_seed${seed}"
      local run_dir="$benchmark_runs_root/$run_name"
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "scratch" "ddqn_scratch" "$seed" "scratch_task" "$benchmark" "$obs_mode" "$task_id" "$run_dir" "planned" >>"$PLAN_CSV"
      log "[scratch $benchmark $run_idx/$total_runs] task=$task_id seed=$seed"
      if run_train \
        --mode scratch_task \
        --method ddqn_scratch \
        --task-id "$task_id" \
        --seed "$seed" \
        --log-dir "$benchmark_runs_root" \
        --run-name "$run_name" \
        "${common_args[@]}"; then
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
          "scratch" "ddqn_scratch" "$seed" "scratch_task" "$benchmark" "$obs_mode" "$task_id" "$run_dir" "ok" "$run_dir/summary.json" >>"$MANIFEST_CSV"
      else
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
          "scratch" "ddqn_scratch" "$seed" "scratch_task" "$benchmark" "$obs_mode" "$task_id" "$run_dir" "failed" "" >>"$MANIFEST_CSV"
      fi
    done
  done
}

run_continual_stage_for_benchmark() {
  local benchmark="$1"
  local obs_mode="$2"
  local benchmark_runs_root="$3"
  local scratch_refs_json="$4"
  local task_ids_csv="$5"
  local run_idx=0
  local total_runs=$(( ${#SEEDS[@]} * ${#METHODS[@]} ))
  local common_args=()
  mapfile -d '' -t common_args < <(build_common_args "$benchmark" "$obs_mode")
  for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      run_idx=$((run_idx + 1))
      local run_name="${method}_seed${seed}"
      local run_dir="$benchmark_runs_root/$run_name"
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "continual" "$method" "$seed" "continual" "$benchmark" "$obs_mode" "$task_ids_csv" "$run_dir" "planned" >>"$PLAN_CSV"
      log "[continual $benchmark $run_idx/$total_runs] method=$method seed=$seed"
      local args=(
        --mode continual
        --method "$method"
        --seed "$seed"
        --log-dir "$benchmark_runs_root"
        --run-name "$run_name"
        "${common_args[@]}"
      )
      if [[ -n "$task_ids_csv" ]]; then
        args+=(--task-ids-csv "$task_ids_csv")
      fi
      if [[ -n "$scratch_refs_json" ]]; then
        args+=(--scratch-summary-json "$scratch_refs_json")
      fi
      if run_train "${args[@]}"; then
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
          "continual" "$method" "$seed" "continual" "$benchmark" "$obs_mode" "$task_ids_csv" "$run_dir" "ok" "$run_dir/summary.json" >>"$MANIFEST_CSV"
      else
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
          "continual" "$method" "$seed" "continual" "$benchmark" "$obs_mode" "$task_ids_csv" "$run_dir" "failed" "" >>"$MANIFEST_CSV"
      fi
    done
  done
}

SCRATCH_REFS_DIR="$SESSION_ROOT/scratch_refs"
mkdir -p "$SCRATCH_REFS_DIR"
SCRATCH_REFS_MAP_JSON="$SESSION_ROOT/scratch_refs_by_benchmark.json"
printf '{\n' >"$SCRATCH_REFS_MAP_JSON"
scratch_refs_first=1

for benchmark in "${BENCHMARKS[@]}"; do
  readarray -t BENCHMARK_DEFAULTS < <(benchmark_defaults "$benchmark")
  benchmark_obs_mode="${USER_OBS_MODE:-${BENCHMARK_DEFAULTS[0]}}"
  benchmark_scratch_tasks_csv="$SCRATCH_TASK_IDS_CSV"
  if [[ -z "${SCRATCH_TASK_IDS_CSV:-}" || ${#BENCHMARKS[@]} -gt 1 ]]; then
    benchmark_scratch_tasks_csv="${BENCHMARK_DEFAULTS[1]}"
  fi
  benchmark_task_ids_csv="$TASK_IDS_CSV"
  benchmark_runs_root="$RUNS_ROOT/$benchmark"
  mkdir -p "$benchmark_runs_root"
  benchmark_scratch_refs_json="$SCRATCH_REFS_JSON"

  if [[ "$MODE" == "scratch_task" ]]; then
    run_scratch_stage_for_benchmark "$benchmark" "$benchmark_obs_mode" "$benchmark_scratch_tasks_csv" "$benchmark_runs_root"
    continue
  fi

  if [[ "$AUTO_BUILD_SCRATCH_REFS" == "1" && -z "$SCRATCH_REFS_JSON" ]]; then
    log "Running scratch stage for $benchmark to build scratch references"
    run_scratch_stage_for_benchmark "$benchmark" "$benchmark_obs_mode" "$benchmark_scratch_tasks_csv" "$benchmark_runs_root"
    benchmark_scratch_refs_json="$SCRATCH_REFS_DIR/${benchmark}.json"
    py_run -m rl.rl_uniandes.drl.morphin.build_scratch_refs \
      --root-dir "$benchmark_runs_root" \
      --output-json "$benchmark_scratch_refs_json" \
      --min-final-success "$SCRATCH_REF_MIN_FINAL_SUCCESS" \
      --min-valid-runs "$SCRATCH_REF_MIN_VALID_RUNS" \
      --min-valid-fraction "$SCRATCH_REF_MIN_VALID_FRACTION"
    log "Scratch refs for $benchmark built at $benchmark_scratch_refs_json"
  fi

  if [[ -n "$benchmark_scratch_refs_json" ]]; then
    if [[ $scratch_refs_first -eq 0 ]]; then
      printf ',\n' >>"$SCRATCH_REFS_MAP_JSON"
    fi
    scratch_refs_first=0
    printf '  "%s": "%s"' "$benchmark" "$benchmark_scratch_refs_json" >>"$SCRATCH_REFS_MAP_JSON"
  fi

  run_continual_stage_for_benchmark "$benchmark" "$benchmark_obs_mode" "$benchmark_runs_root" "$benchmark_scratch_refs_json" "$benchmark_task_ids_csv"
done

printf '\n}\n' >>"$SCRATCH_REFS_MAP_JSON"

log "Aggregating full session outputs"
py_run -m rl.rl_uniandes.drl.morphin.aggregate_morphin_results \
  --root-dir "$RUNS_ROOT" \
  --output-dir "$ANALYSIS_DIR"

for benchmark in "${BENCHMARKS[@]}"; do
  benchmark_runs_root="$RUNS_ROOT/$benchmark"
  benchmark_analysis_dir="$ANALYSIS_DIR/by_benchmark/$benchmark"
  mkdir -p "$benchmark_analysis_dir"
  log "Aggregating benchmark-specific outputs for $benchmark"
  py_run -m rl.rl_uniandes.drl.morphin.aggregate_morphin_results \
    --root-dir "$benchmark_runs_root" \
    --output-dir "$benchmark_analysis_dir"
done

cat >"$SESSION_ROOT/session_config.json" <<JSON
{
  "run_profile": "$RUN_PROFILE",
  "mode": "$MODE",
  "benchmark": "${BENCHMARKS[0]}",
  "benchmarks_csv": "${BENCHMARKS_CSV:-$BENCHMARK}",
  "session_group": "${SESSION_GROUP}",
  "obs_mode": "$OBS_MODE",
  "seeds_csv": "$SEEDS_CSV",
  "method_set": "$METHOD_SET",
  "method_source": "$METHOD_SOURCE",
  "methods_csv": "$METHODS_CSV",
  "task_ids_csv": "$TASK_IDS_CSV",
  "scratch_task_ids_csv": "$SCRATCH_TASK_IDS_CSV",
  "episodes_per_task": $EPISODES_PER_TASK,
  "max_steps_per_episode": $MAX_STEPS_PER_EPISODE,
  "eval_episodes": $EVAL_EPISODES,
  "eval_every_episodes": $EVAL_EVERY_EPISODES,
  "eval_dense_every_episodes": $EVAL_DENSE_EVERY_EPISODES,
  "eval_dense_window_episodes": $EVAL_DENSE_WINDOW_EPISODES,
  "success_threshold": $SUCCESS_THRESHOLD,
  "threshold_min_consecutive_evals": $THRESHOLD_MIN_CONSECUTIVE_EVALS,
  "scratch_ref_min_final_success": $SCRATCH_REF_MIN_FINAL_SUCCESS,
  "scratch_ref_min_valid_runs": $SCRATCH_REF_MIN_VALID_RUNS,
  "scratch_ref_min_valid_fraction": $SCRATCH_REF_MIN_VALID_FRACTION,
  "warmup_steps": $WARMUP_STEPS,
  "batch_size": $BATCH_SIZE,
  "eps_decay_steps": $EPS_DECAY_STEPS,
  "eps_reset_value": $EPS_RESET_VALUE,
  "eps_decay_steps_after_switch": $EPS_DECAY_STEPS_AFTER_SWITCH,
  "alpha_max_mult": $ALPHA_MAX_MULT,
  "td_k": $TD_K,
  "learning_rate": $LEARNING_RATE,
  "archive_frac": $ARCHIVE_FRAC,
  "recent_mix_start": $RECENT_MIX_START,
  "recent_mix_end": $RECENT_MIX_END,
  "post_switch_steps": $POST_SWITCH_STEPS,
  "segmented_keep_tail": $SEGMENTED_KEEP_TAIL,
  "segmented_recent_only_steps": $SEGMENTED_RECENT_ONLY_STEPS,
  "segmented_min_recent_samples": $SEGMENTED_MIN_RECENT_SAMPLES,
  "auto_build_scratch_refs": $AUTO_BUILD_SCRATCH_REFS,
  "scratch_refs_json": "${SCRATCH_REFS_JSON}",
  "scratch_refs_map_json": "${SCRATCH_REFS_MAP_JSON}"
}
JSON

log "Done. Session root: $SESSION_ROOT"
