#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/minigrid_crl_runner_adapter.py}"
AGGREGATOR="${AGGREGATOR:-${SCRIPT_DIR}/aggregate_ablation_results.py}"

RUN_PROFILE="${RUN_PROFILE:-diagnostic}"
TASK_PRESET="${TASK_PRESET:-smoke4_easy_fullobs}"
OBS_MODE="${OBS_MODE:-image}"
if [[ -z "${STEPS_PER_TASK+x}" ]]; then
  if [[ "${RUN_PROFILE}" == "diagnostic" ]]; then
    STEPS_PER_TASK="100000"
  elif [[ "${RUN_PROFILE}" == "sanity" ]]; then
    STEPS_PER_TASK="1000000"
  else
    STEPS_PER_TASK="400000"
  fi
fi
if [[ -z "${EVAL_EPISODES+x}" ]]; then
  if [[ "${RUN_PROFILE}" == "diagnostic" ]]; then
    EVAL_EPISODES="20"
  elif [[ "${RUN_PROFILE}" == "sanity" ]]; then
    EVAL_EPISODES="20"
  else
    EVAL_EPISODES="30"
  fi
fi
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
if [[ -z "${SEEDS_CSV+x}" ]]; then
  if [[ "${RUN_PROFILE}" == "diagnostic" || "${RUN_PROFILE}" == "sanity" ]]; then
    SEEDS_CSV="42"
  else
    SEEDS_CSV="42,43,44"
  fi
fi
FULLY_OBSERVABLE="${FULLY_OBSERVABLE:-1}"
EVAL_POLICY_MODES_CSV="${EVAL_POLICY_MODES_CSV:-deterministic,stochastic}"
SUMMARY_EVAL_POLICY_MODE="${SUMMARY_EVAL_POLICY_MODE:-deterministic}"
SUMMARY_MODEL_VARIANT="${SUMMARY_MODEL_VARIANT:-current}"
EVAL_SEED_OFFSETS_CSV="${EVAL_SEED_OFFSETS_CSV:-0,1000}"
SUMMARY_EVAL_SEED_OFFSET="${SUMMARY_EVAL_SEED_OFFSET:-1000}"
PERIODIC_EVAL_SCOPE="${PERIODIC_EVAL_SCOPE:-}"
PERIODIC_EVAL_FREQ="${PERIODIC_EVAL_FREQ:-0}"
SAVE_BEST_EVAL_CHECKPOINT="${SAVE_BEST_EVAL_CHECKPOINT:-0}"
EVAL_BEST_EVAL_MODEL_AT_PHASE_END="${EVAL_BEST_EVAL_MODEL_AT_PHASE_END:-0}"
RESTORE_BEST_EVAL_MODEL="${RESTORE_BEST_EVAL_MODEL:-0}"
EARLY_STOP_EVAL_SUCCESS_THRESHOLD="${EARLY_STOP_EVAL_SUCCESS_THRESHOLD:-}"
EARLY_STOP_EVAL_PATIENCE="${EARLY_STOP_EVAL_PATIENCE:-2}"
N_ENVS="${N_ENVS:-8}"
VEC_ENV="${VEC_ENV:-dummy}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-1e-4}"
PPO_N_STEPS="${PPO_N_STEPS:-512}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-256}"
PPO_N_EPOCHS="${PPO_N_EPOCHS:-4}"
PPO_CLIP_RANGE="${PPO_CLIP_RANGE:-0.1}"
PPO_CLIP_RANGE_VF="${PPO_CLIP_RANGE_VF:-0.1}"
PPO_ENT_COEF="${PPO_ENT_COEF:-0.005}"
PPO_TARGET_KL="${PPO_TARGET_KL:-0.01}"
PPO_ENT_COEF_TASK_OVERRIDES_CSV="${PPO_ENT_COEF_TASK_OVERRIDES_CSV:-}"
ABLATION_IDS_CSV="${ABLATION_IDS_CSV:-}"
EXP_ROOT="${EXP_ROOT:-logs/minigrid_ablations/${TASK_PRESET}}"
ENABLE_RANK_SWEEP="${ENABLE_RANK_SWEEP:-0}"
ENABLE_SINGLE_TASK_SANITY="${ENABLE_SINGLE_TASK_SANITY:-0}"
ONLY_SANITY="${ONLY_SANITY:-0}"
DRY_RUN="${DRY_RUN:-0}"
SANITY_TASKS_CSV="${SANITY_TASKS_CSV:-}"
CONTINUAL_TASKS_CSV="${CONTINUAL_TASKS_CSV:-}"
ADAPTER_TRAIN_ACTOR_HEADS_AFTER_WARMUP="${ADAPTER_TRAIN_ACTOR_HEADS_AFTER_WARMUP:-1}"
ADAPTER_TRAIN_FULL_CRITIC_AFTER_WARMUP="${ADAPTER_TRAIN_FULL_CRITIC_AFTER_WARMUP:-0}"

if [[ "${TASK_PRESET}" == "smoke4_easy_fullobs" ]]; then
  FULLY_OBSERVABLE=1
  if [[ -z "${SANITY_TASKS_CSV}" ]]; then
    SANITY_TASKS_CSV="MiniGrid-Empty-5x5-v0,MiniGrid-DoorKey-5x5-v0,MiniGrid-LavaGapS5-v0"
  fi
fi

if [[ "${RUN_PROFILE}" == "sanity" ]]; then
  ENABLE_SINGLE_TASK_SANITY=1
  ONLY_SANITY=1
  if [[ -z "${PERIODIC_EVAL_SCOPE}" ]]; then
    PERIODIC_EVAL_SCOPE="active_task"
  fi
  if [[ "${PERIODIC_EVAL_FREQ}" == "0" ]]; then
    PERIODIC_EVAL_FREQ="50000"
  fi
  if [[ "${SAVE_BEST_EVAL_CHECKPOINT}" == "0" ]]; then
    SAVE_BEST_EVAL_CHECKPOINT="1"
  fi
  if [[ "${EVAL_BEST_EVAL_MODEL_AT_PHASE_END}" == "0" ]]; then
    EVAL_BEST_EVAL_MODEL_AT_PHASE_END="1"
  fi
  if [[ "${RESTORE_BEST_EVAL_MODEL}" == "0" ]]; then
    RESTORE_BEST_EVAL_MODEL="1"
  fi
  SUMMARY_MODEL_VARIANT="best_eval"
  if [[ -z "${EARLY_STOP_EVAL_SUCCESS_THRESHOLD}" ]]; then
    EARLY_STOP_EVAL_SUCCESS_THRESHOLD="0.99"
  fi
fi

if [[ -z "${ABLATION_IDS_CSV}" && "${RUN_PROFILE}" == "diagnostic" ]]; then
  ABLATION_IDS_CSV="a01_shared_taskid_concat_no_multihead,a02_shared_taskid_concat_multihead_delayed,a03_adapters_taskid_concat_no_multihead_r08_a08,a04_adapters_taskid_concat_multihead_delayed_r08_a08"
fi

if [[ "${RUN_PROFILE}" == "diagnostic" ]]; then
  if [[ -z "${CONTINUAL_TASKS_CSV}" && -n "${SANITY_TASKS_CSV}" ]]; then
    CONTINUAL_TASKS_CSV="${SANITY_TASKS_CSV}"
  fi
  if [[ -z "${PERIODIC_EVAL_SCOPE}" ]]; then
    PERIODIC_EVAL_SCOPE="active_task"
  fi
  if [[ "${PERIODIC_EVAL_FREQ}" == "0" ]]; then
    PERIODIC_EVAL_FREQ="50000"
  fi
  if [[ "${SAVE_BEST_EVAL_CHECKPOINT}" == "0" ]]; then
    SAVE_BEST_EVAL_CHECKPOINT="1"
  fi
  if [[ "${EVAL_BEST_EVAL_MODEL_AT_PHASE_END}" == "0" ]]; then
    EVAL_BEST_EVAL_MODEL_AT_PHASE_END="1"
  fi
  SUMMARY_MODEL_VARIANT="${SUMMARY_MODEL_VARIANT:-current}"
fi

if [[ -z "${PERIODIC_EVAL_SCOPE}" ]]; then
  PERIODIC_EVAL_SCOPE="active_task"
fi

if [[ "${EXP_ROOT}" != /* ]]; then
  EXP_ROOT="${SCRIPT_DIR}/${EXP_ROOT}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_ID="session_${TIMESTAMP}"
SESSION_ROOT="${EXP_ROOT}/${SESSION_ID}"
RUNS_ROOT="${SESSION_ROOT}/runs"
CONSOLE_DIR="${SESSION_ROOT}/console_logs"
ANALYSIS_DIR="${SESSION_ROOT}/analysis"
PLAN_CSV="${SESSION_ROOT}/experiment_plan.csv"
MANIFEST_CSV="${SESSION_ROOT}/experiment_manifest.csv"
SESSION_JSON="${SESSION_ROOT}/session_config.json"
SESSION_STATUS_JSON="${SESSION_ROOT}/session_status.json"
ANALYSIS_LOG="${ANALYSIS_DIR}/aggregate.log"

mkdir -p "${RUNS_ROOT}" "${CONSOLE_DIR}" "${ANALYSIS_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

csv_escape() {
  local s="$1"
  s="${s//\"/\"\"}"
  printf '"%s"' "$s"
}

join_by() {
  local sep="$1"
  shift
  local out=""
  local first=1
  for item in "$@"; do
    if [[ ${first} -eq 1 ]]; then
      out="$item"
      first=0
    else
      out+="${sep}${item}"
    fi
  done
  printf '%s' "$out"
}

read_seeds() {
  local IFS=','
  read -r -a SEEDS <<< "${SEEDS_CSV}"
  for i in "${!SEEDS[@]}"; do
    SEEDS[$i]="$(echo "${SEEDS[$i]}" | xargs)"
  done
}

COMMON_ARGS=(
  --mode continual
  --algo ppo
  --task-preset "${TASK_PRESET}"
  --steps-per-task "${STEPS_PER_TASK}"
  --eval-episodes "${EVAL_EPISODES}"
  --max-episode-steps "${MAX_EPISODE_STEPS}"
  --obs-mode "${OBS_MODE}"
  --eval-policy-modes-csv "${EVAL_POLICY_MODES_CSV}"
  --summary-eval-policy-mode "${SUMMARY_EVAL_POLICY_MODE}"
  --summary-model-variant "${SUMMARY_MODEL_VARIANT}"
  --eval-seed-offsets-csv "${EVAL_SEED_OFFSETS_CSV}"
  --summary-eval-seed-offset "${SUMMARY_EVAL_SEED_OFFSET}"
  --periodic-eval-scope "${PERIODIC_EVAL_SCOPE}"
  --periodic-eval-freq "${PERIODIC_EVAL_FREQ}"
  --ppo-learning-rate "${PPO_LEARNING_RATE}"
  --ppo-n-steps "${PPO_N_STEPS}"
  --ppo-batch-size "${PPO_BATCH_SIZE}"
  --ppo-n-epochs "${PPO_N_EPOCHS}"
  --n-envs "${N_ENVS}"
  --vec-env "${VEC_ENV}"
  --ppo-clip-range "${PPO_CLIP_RANGE}"
  --ppo-clip-range-vf "${PPO_CLIP_RANGE_VF}"
  --ppo-ent-coef "${PPO_ENT_COEF}"
  --ppo-target-kl "${PPO_TARGET_KL}"
  --reset-optimizers-every-task
  --reset-metric-windows-every-task
)

if [[ -n "${PPO_ENT_COEF_TASK_OVERRIDES_CSV}" ]]; then
  COMMON_ARGS+=(--ppo-ent-coef-task-overrides-csv "${PPO_ENT_COEF_TASK_OVERRIDES_CSV}")
fi

if [[ "${SAVE_BEST_EVAL_CHECKPOINT}" == "1" ]]; then
  COMMON_ARGS+=(--save-best-eval-checkpoint)
else
  COMMON_ARGS+=(--no-save-best-eval-checkpoint)
fi

if [[ "${EVAL_BEST_EVAL_MODEL_AT_PHASE_END}" == "1" ]]; then
  COMMON_ARGS+=(--eval-best-eval-model-at-phase-end)
else
  COMMON_ARGS+=(--no-eval-best-eval-model-at-phase-end)
fi

if [[ "${RESTORE_BEST_EVAL_MODEL}" == "1" ]]; then
  COMMON_ARGS+=(--restore-best-eval-model)
else
  COMMON_ARGS+=(--no-restore-best-eval-model)
fi

if [[ -n "${EARLY_STOP_EVAL_SUCCESS_THRESHOLD}" ]]; then
  COMMON_ARGS+=(--early-stop-eval-success-threshold "${EARLY_STOP_EVAL_SUCCESS_THRESHOLD}")
  COMMON_ARGS+=(--early-stop-eval-patience "${EARLY_STOP_EVAL_PATIENCE}")
fi

if [[ "${FULLY_OBSERVABLE}" == "1" ]]; then
  COMMON_ARGS+=(--fully-observable)
else
  COMMON_ARGS+=(--no-fully-observable)
fi

validate_paths() {
  if [[ ! -f "${RUNNER}" ]]; then
    log "ERROR: no se encontró el runner en ${RUNNER}"
    exit 1
  fi
  if [[ ! -f "${AGGREGATOR}" ]]; then
    log "ERROR: no se encontró el agregador en ${AGGREGATOR}"
    exit 1
  fi
}

validate_scalar_inputs() {
  if [[ "${RUN_PROFILE}" != "diagnostic" && "${RUN_PROFILE}" != "sanity" && "${RUN_PROFILE}" != "core" && "${RUN_PROFILE}" != "full" ]]; then
    log "ERROR: RUN_PROFILE debe ser diagnostic, sanity, core o full"
    exit 1
  fi
  if [[ -z "${TASK_PRESET}" || -z "${OBS_MODE}" ]]; then
    log "ERROR: TASK_PRESET y OBS_MODE no pueden quedar vacíos"
    exit 1
  fi
  if [[ "${STEPS_PER_TASK}" -le 0 || "${EVAL_EPISODES}" -le 0 || "${MAX_EPISODE_STEPS}" -le 0 ]]; then
    log "ERROR: steps/evals/max_episode_steps deben ser > 0"
    exit 1
  fi
  if [[ "${N_ENVS}" -le 0 ]]; then
    log "ERROR: N_ENVS debe ser > 0"
    exit 1
  fi
  if [[ "${VEC_ENV}" != "dummy" && "${VEC_ENV}" != "subproc" ]]; then
    log "ERROR: VEC_ENV debe ser dummy o subproc"
    exit 1
  fi
  if [[ "${FULLY_OBSERVABLE}" != "0" && "${FULLY_OBSERVABLE}" != "1" ]]; then
    log "ERROR: FULLY_OBSERVABLE debe ser 0 o 1"
    exit 1
  fi
  if [[ "${ENABLE_RANK_SWEEP}" != "0" && "${ENABLE_RANK_SWEEP}" != "1" ]]; then
    log "ERROR: ENABLE_RANK_SWEEP debe ser 0 o 1"
    exit 1
  fi
  if [[ "${ENABLE_SINGLE_TASK_SANITY}" != "0" && "${ENABLE_SINGLE_TASK_SANITY}" != "1" ]]; then
    log "ERROR: ENABLE_SINGLE_TASK_SANITY debe ser 0 o 1"
    exit 1
  fi
  if [[ "${SAVE_BEST_EVAL_CHECKPOINT}" != "0" && "${SAVE_BEST_EVAL_CHECKPOINT}" != "1" ]]; then
    log "ERROR: SAVE_BEST_EVAL_CHECKPOINT debe ser 0 o 1"
    exit 1
  fi
  if [[ "${EVAL_BEST_EVAL_MODEL_AT_PHASE_END}" != "0" && "${EVAL_BEST_EVAL_MODEL_AT_PHASE_END}" != "1" ]]; then
    log "ERROR: EVAL_BEST_EVAL_MODEL_AT_PHASE_END debe ser 0 o 1"
    exit 1
  fi
  if [[ "${RESTORE_BEST_EVAL_MODEL}" != "0" && "${RESTORE_BEST_EVAL_MODEL}" != "1" ]]; then
    log "ERROR: RESTORE_BEST_EVAL_MODEL debe ser 0 o 1"
    exit 1
  fi
  if [[ "${ONLY_SANITY}" != "0" && "${ONLY_SANITY}" != "1" ]]; then
    log "ERROR: ONLY_SANITY debe ser 0 o 1"
    exit 1
  fi
  if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    log "ERROR: DRY_RUN debe ser 0 o 1"
    exit 1
  fi
  if [[ "${PERIODIC_EVAL_SCOPE}" != "active_task" && "${PERIODIC_EVAL_SCOPE}" != "seen_tasks_mean" ]]; then
    log "ERROR: PERIODIC_EVAL_SCOPE debe ser active_task o seen_tasks_mean"
    exit 1
  fi
  if [[ "${SUMMARY_MODEL_VARIANT}" != "current" && "${SUMMARY_MODEL_VARIANT}" != "best_eval" ]]; then
    log "ERROR: SUMMARY_MODEL_VARIANT debe ser current o best_eval"
    exit 1
  fi
  if [[ "${SUMMARY_MODEL_VARIANT}" == "best_eval" && "${SAVE_BEST_EVAL_CHECKPOINT}" != "1" ]]; then
    log "ERROR: SUMMARY_MODEL_VARIANT=best_eval requiere SAVE_BEST_EVAL_CHECKPOINT=1"
    exit 1
  fi
  if [[ "${ADAPTER_TRAIN_ACTOR_HEADS_AFTER_WARMUP}" != "0" && "${ADAPTER_TRAIN_ACTOR_HEADS_AFTER_WARMUP}" != "1" ]]; then
    log "ERROR: ADAPTER_TRAIN_ACTOR_HEADS_AFTER_WARMUP debe ser 0 o 1"
    exit 1
  fi
  if [[ "${ADAPTER_TRAIN_FULL_CRITIC_AFTER_WARMUP}" != "0" && "${ADAPTER_TRAIN_FULL_CRITIC_AFTER_WARMUP}" != "1" ]]; then
    log "ERROR: ADAPTER_TRAIN_FULL_CRITIC_AFTER_WARMUP debe ser 0 o 1"
    exit 1
  fi
}

validate_seeds() {
  if [[ ${#SEEDS[@]} -eq 0 ]]; then
    log "ERROR: SEEDS_CSV quedó vacío"
    exit 1
  fi
  for seed in "${SEEDS[@]}"; do
    if [[ ! "${seed}" =~ ^-?[0-9]+$ ]]; then
      log "ERROR: seed inválido: ${seed}"
      exit 1
    fi
  done
}

write_session_config() {
  cat > "${SESSION_JSON}" <<JSON
{
  "timestamp": "${TIMESTAMP}",
  "session_id": "${SESSION_ID}",
  "run_profile": "${RUN_PROFILE}",
  "task_preset": "${TASK_PRESET}",
  "obs_mode": "${OBS_MODE}",
  "fully_observable": ${FULLY_OBSERVABLE},
  "steps_per_task": ${STEPS_PER_TASK},
  "eval_episodes": ${EVAL_EPISODES},
  "max_episode_steps": ${MAX_EPISODE_STEPS},
  "eval_policy_modes_csv": "${EVAL_POLICY_MODES_CSV}",
  "summary_eval_policy_mode": "${SUMMARY_EVAL_POLICY_MODE}",
  "summary_model_variant": "${SUMMARY_MODEL_VARIANT}",
  "eval_seed_offsets_csv": "${EVAL_SEED_OFFSETS_CSV}",
  "summary_eval_seed_offset": ${SUMMARY_EVAL_SEED_OFFSET},
  "periodic_eval_scope": "${PERIODIC_EVAL_SCOPE}",
  "periodic_eval_freq": ${PERIODIC_EVAL_FREQ},
  "save_best_eval_checkpoint": ${SAVE_BEST_EVAL_CHECKPOINT},
  "eval_best_eval_model_at_phase_end": ${EVAL_BEST_EVAL_MODEL_AT_PHASE_END},
  "restore_best_eval_model": ${RESTORE_BEST_EVAL_MODEL},
  "early_stop_eval_success_threshold": ${EARLY_STOP_EVAL_SUCCESS_THRESHOLD:-null},
  "early_stop_eval_patience": ${EARLY_STOP_EVAL_PATIENCE},
  "n_envs": ${N_ENVS},
  "vec_env": "${VEC_ENV}",
  "ppo_learning_rate": ${PPO_LEARNING_RATE},
  "ppo_n_steps": ${PPO_N_STEPS},
  "ppo_batch_size": ${PPO_BATCH_SIZE},
  "ppo_n_epochs": ${PPO_N_EPOCHS},
  "ppo_clip_range": ${PPO_CLIP_RANGE},
  "ppo_clip_range_vf": ${PPO_CLIP_RANGE_VF},
  "ppo_ent_coef": ${PPO_ENT_COEF},
  "ppo_target_kl": ${PPO_TARGET_KL},
  "ppo_ent_coef_task_overrides_csv": "${PPO_ENT_COEF_TASK_OVERRIDES_CSV}",
  "ablation_ids_csv": "${ABLATION_IDS_CSV}",
  "seeds_csv": "${SEEDS_CSV}",
  "exp_root": "${EXP_ROOT}",
  "session_root": "${SESSION_ROOT}",
  "runs_root": "${RUNS_ROOT}",
  "runner": "${RUNNER}",
  "aggregator": "${AGGREGATOR}",
  "enable_rank_sweep": ${ENABLE_RANK_SWEEP},
  "enable_single_task_sanity": ${ENABLE_SINGLE_TASK_SANITY},
  "only_sanity": ${ONLY_SANITY},
  "sanity_tasks_csv": "${SANITY_TASKS_CSV}",
  "continual_tasks_csv": "${CONTINUAL_TASKS_CSV}",
  "adapter_train_actor_heads_after_warmup": ${ADAPTER_TRAIN_ACTOR_HEADS_AFTER_WARMUP},
  "adapter_train_full_critic_after_warmup": ${ADAPTER_TRAIN_FULL_CRITIC_AFTER_WARMUP},
  "dry_run": ${DRY_RUN}
}
JSON
}

write_session_status() {
  local analysis_status="$1"
  cat > "${SESSION_STATUS_JSON}" <<JSON
{
  "session_id": "${SESSION_ID}",
  "task_preset": "${TASK_PRESET}",
  "session_root": "${SESSION_ROOT}",
  "runs_root": "${RUNS_ROOT}",
  "plan_csv": "${PLAN_CSV}",
  "manifest_csv": "${MANIFEST_CSV}",
  "analysis_dir": "${ANALYSIS_DIR}",
  "analysis_log": "${ANALYSIS_LOG}",
  "total_runs": ${TOTAL_RUNS},
  "successful_runs": ${SUCCESSFUL_RUNS},
  "failed_runs": ${FAILURES},
  "analysis_status": "${analysis_status}",
  "dry_run": ${DRY_RUN}
}
JSON
}

init_csvs() {
  printf 'planned_index,matrix_group,ablation_id,seed,description,hypothesis,output_base_dir,console_log,command\n' > "${PLAN_CSV}"
  printf 'planned_index,matrix_group,ablation_id,seed,description,hypothesis,status,exit_code,output_base_dir,run_dir,summary_json,config_json,eval_metrics_csv,train_metrics_csv,tb_scalars_export_csv,train_monitor_csv,eval_monitor_csv,console_log,command\n' > "${MANIFEST_CSV}"
}

PLAN_INDEX=0
TOTAL_RUNS=0
FAILURES=0
SUCCESSFUL_RUNS=0

PLAN_IDS=()
PLAN_GROUPS=()
PLAN_DESCRIPTIONS=()
PLAN_HYPOTHESES=()
PLAN_ARGS=()
ARGS_SEP=$'\x1f'

sanitize_id() {
  echo "$1" | tr -c '[:alnum:]' '_' | sed 's/^_//; s/_$//'
}

ADAPTER_AFTER_WARMUP_ARGS=()
if [[ "${ADAPTER_TRAIN_ACTOR_HEADS_AFTER_WARMUP}" == "1" ]]; then
  ADAPTER_AFTER_WARMUP_ARGS+=(--adapter-train-actor-heads-after-warmup)
else
  ADAPTER_AFTER_WARMUP_ARGS+=(--no-adapter-train-actor-heads-after-warmup)
fi
if [[ "${ADAPTER_TRAIN_FULL_CRITIC_AFTER_WARMUP}" == "1" ]]; then
  ADAPTER_AFTER_WARMUP_ARGS+=(--adapter-train-full-critic-after-warmup)
else
  ADAPTER_AFTER_WARMUP_ARGS+=(--no-adapter-train-full-critic-after-warmup)
fi

resolve_preset_tasks_csv() {
  case "${TASK_PRESET}" in
    smoke4_easy_fullobs)
      echo "MiniGrid-Empty-5x5-v0,MiniGrid-DoorKey-5x5-v0,MiniGrid-LavaGapS5-v0,MiniGrid-GoToDoor-5x5-v0"
      return 0
      ;;
    smoke4)
      echo "MiniGrid-Empty-5x5-v0,MiniGrid-DoorKey-5x5-v0,MiniGrid-FourRooms-v0,MiniGrid-Unlock-v0"
      return 0
      ;;
  esac
  "${PYTHON_BIN}" -c 'import importlib.util, sys; spec = importlib.util.spec_from_file_location("runner_module", sys.argv[1]); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); print(",".join(module.resolve_task_sequence(sys.argv[2], None)))' \
    "${RUNNER}" "${TASK_PRESET}"
}

register_ablation() {
  local ablation_id="$1"
  local matrix_group="$2"
  local description="$3"
  local hypothesis="$4"
  shift 4
  local has_tasks_flag=0
  local token=""
  for token in "$@"; do
    if [[ "${token}" == "--tasks" ]]; then
      has_tasks_flag=1
      break
    fi
  done
  if [[ "${matrix_group}" != "sanity" && -n "${CONTINUAL_TASKS_CSV}" && "${has_tasks_flag}" == "0" ]]; then
    set -- --tasks "${CONTINUAL_TASKS_CSV}" "$@"
  fi
  local joined_args=""
  local token=""
  for token in "$@"; do
    if [[ -n "${joined_args}" ]]; then
      joined_args+="${ARGS_SEP}"
    fi
    joined_args+="${token}"
  done
  PLAN_IDS+=("${ablation_id}")
  PLAN_GROUPS+=("${matrix_group}")
  PLAN_DESCRIPTIONS+=("${description}")
  PLAN_HYPOTHESES+=("${hypothesis}")
  PLAN_ARGS+=("${joined_args}")
}

apply_ablation_filter() {
  if [[ -z "${ABLATION_IDS_CSV}" ]]; then
    return 0
  fi

  local IFS=','
  local -a requested_ids_raw=()
  read -r -a requested_ids_raw <<< "${ABLATION_IDS_CSV}"

  local -A requested_ids=()
  local raw_id=""
  local clean_id=""
  for raw_id in "${requested_ids_raw[@]}"; do
    clean_id="$(echo "${raw_id}" | xargs)"
    if [[ -n "${clean_id}" ]]; then
      requested_ids["${clean_id}"]=1
    fi
  done

  if [[ ${#requested_ids[@]} -eq 0 ]]; then
    log "ERROR: ABLATION_IDS_CSV quedó vacío tras parseo"
    exit 1
  fi

  local -a filtered_ids=()
  local -a filtered_groups=()
  local -a filtered_descriptions=()
  local -a filtered_hypotheses=()
  local -a filtered_args=()
  local -A found_ids=()
  local idx=0
  local ablation_id=""

  for idx in "${!PLAN_IDS[@]}"; do
    ablation_id="${PLAN_IDS[$idx]}"
    if [[ -n "${requested_ids[${ablation_id}]:-}" || ( "${ENABLE_SINGLE_TASK_SANITY}" == "1" && "${PLAN_GROUPS[$idx]}" == "sanity" ) ]]; then
      filtered_ids+=("${PLAN_IDS[$idx]}")
      filtered_groups+=("${PLAN_GROUPS[$idx]}")
      filtered_descriptions+=("${PLAN_DESCRIPTIONS[$idx]}")
      filtered_hypotheses+=("${PLAN_HYPOTHESES[$idx]}")
      filtered_args+=("${PLAN_ARGS[$idx]}")
      found_ids["${ablation_id}"]=1
    fi
  done

  if [[ ${#filtered_ids[@]} -eq 0 ]]; then
    log "ERROR: el filtro de ablaciones dejó el plan vacío"
    exit 1
  fi

  for ablation_id in "${!requested_ids[@]}"; do
    if [[ -z "${found_ids[${ablation_id}]:-}" ]]; then
      log "ERROR: ABLATION_IDS_CSV incluye ablation_id no registrado: ${ablation_id}"
      exit 1
    fi
  done

  PLAN_IDS=("${filtered_ids[@]}")
  PLAN_GROUPS=("${filtered_groups[@]}")
  PLAN_DESCRIPTIONS=("${filtered_descriptions[@]}")
  PLAN_HYPOTHESES=("${filtered_hypotheses[@]}")
  PLAN_ARGS=("${filtered_args[@]}")
}

build_ablation_plan() {
  if [[ "${ENABLE_SINGLE_TASK_SANITY}" == "1" ]]; then
    local sanity_csv
    if [[ -n "${SANITY_TASKS_CSV}" ]]; then
      sanity_csv="${SANITY_TASKS_CSV}"
    else
      sanity_csv="$(resolve_preset_tasks_csv)"
    fi
    local IFS=','
    read -r -a SANITY_TASKS <<< "${sanity_csv}"
    for task in "${SANITY_TASKS[@]}"; do
      local safe_task
      safe_task="$(sanitize_id "${task}")"
      register_ablation \
        "s00_sanity_${safe_task}" \
        "sanity" \
        "Single-task sanity run para ${task} con PPO shared vanilla." \
        "Verifica que la tarea sea aprendible en single-task antes de atribuir resultados de continual learning al método." \
        --tasks "${task}" \
        --task-conditioning ignore \
        --no-adapter-enabled \
        --no-multi-head-enabled
    done
  fi

  if [[ "${ONLY_SANITY}" == "1" ]]; then
    return 0
  fi

  register_ablation \
    "a00_shared_vanilla_no_taskid_no_multihead" \
    "reference" \
    "PPO compartido, sin task id, sin adapters y sin multi-head. Baseline clásico." \
    "Sirve como referencia de escenario task-agnostic; no debe usarse para atribuir el efecto de adapters sobre un baseline task-aware." \
    --task-conditioning ignore \
    --no-adapter-enabled \
    --no-multi-head-enabled

  register_ablation \
    "a01_shared_taskid_concat_no_multihead" \
    "core" \
    "PPO compartido task-aware con task id concatenado, sin adapters y sin multi-head." \
    "Aísla el efecto de task conditioning sin aislamiento paramétrico adicional." \
    --append-task-id \
    --task-conditioning concat \
    --no-adapter-enabled \
    --no-multi-head-enabled

  register_ablation \
    "a02_shared_taskid_concat_multihead_delayed" \
    "core" \
    "PPO compartido task-aware con multi-head tardío, sin adapters." \
    "Mide si la separación de salidas explica mejora final sin necesidad de adapters." \
    --append-task-id \
    --task-conditioning concat \
    --no-adapter-enabled \
    --multi-head-enabled \
    --multi-head-warmup-tasks 1

  register_ablation \
    "a03_adapters_taskid_concat_no_multihead_r08_a08" \
    "core" \
    "Adapters low-rank task-aware sin multi-head." \
    "Evalúa si el aislamiento low-rank basta sin heads por tarea." \
    --append-task-id \
    --task-conditioning concat \
    --adapter-enabled \
    --adapter-warmup-tasks 1 \
    --adapter-rank 8 \
    --adapter-alpha 8 \
    "${ADAPTER_AFTER_WARMUP_ARGS[@]}" \
    --no-multi-head-enabled

  register_ablation \
    "a04_adapters_taskid_concat_multihead_delayed_r08_a08" \
    "core" \
    "Adapters low-rank task-aware con multi-head tardío." \
    "Método principal: task conditioning + aislamiento low-rank + heads tardíos." \
    --append-task-id \
    --task-conditioning concat \
    --adapter-enabled \
    --adapter-warmup-tasks 1 \
    --adapter-rank 8 \
    --adapter-alpha 8 \
    "${ADAPTER_AFTER_WARMUP_ARGS[@]}" \
    --multi-head-enabled \
    --multi-head-warmup-tasks 1

  if [[ "${ENABLE_RANK_SWEEP}" == "1" ]]; then
    register_ablation \
      "a05_adapters_taskid_concat_no_multihead_r16_a16" \
      "rank_sweep" \
      "Sweep de rank intermedio sin multi-head." \
      "Mide sensibilidad del resultado al rank del adapter sin confundirlo con multi-head." \
      --append-task-id \
      --task-conditioning concat \
      --adapter-enabled \
      --adapter-warmup-tasks 1 \
      --adapter-rank 16 \
      --adapter-alpha 16 \
      "${ADAPTER_AFTER_WARMUP_ARGS[@]}" \
      --no-multi-head-enabled

    register_ablation \
      "a06_adapters_taskid_concat_multihead_delayed_r16_a16" \
      "rank_sweep" \
      "Sweep de rank intermedio con multi-head tardío." \
      "Mide sensibilidad del método principal a un rank intermedio." \
      --append-task-id \
      --task-conditioning concat \
      --adapter-enabled \
      --adapter-warmup-tasks 1 \
      --adapter-rank 16 \
      --adapter-alpha 16 \
      "${ADAPTER_AFTER_WARMUP_ARGS[@]}" \
      --multi-head-enabled \
      --multi-head-warmup-tasks 1

    register_ablation \
      "a07_adapters_taskid_concat_no_multihead_r32_a32" \
      "rank_sweep" \
      "Sweep de rank alto sin multi-head." \
      "Mide si el adapter gana por capacidad adicional y no por diseño." \
      --append-task-id \
      --task-conditioning concat \
      --adapter-enabled \
      --adapter-warmup-tasks 1 \
      --adapter-rank 32 \
      --adapter-alpha 32 \
      "${ADAPTER_AFTER_WARMUP_ARGS[@]}" \
      --no-multi-head-enabled

    register_ablation \
      "a08_adapters_taskid_concat_multihead_delayed_r32_a32" \
      "rank_sweep" \
      "Sweep de rank alto con multi-head tardío." \
      "Mide si el método principal mejora solo al aumentar capacidad paramétrica." \
      --append-task-id \
      --task-conditioning concat \
      --adapter-enabled \
      --adapter-warmup-tasks 1 \
      --adapter-rank 32 \
      --adapter-alpha 32 \
      "${ADAPTER_AFTER_WARMUP_ARGS[@]}" \
      --multi-head-enabled \
      --multi-head-warmup-tasks 1
  fi
}

normalize_flag_for_runner_check() {
  local flag="$1"
  if [[ "${flag}" == --no-* ]]; then
    printf -- '--%s' "${flag#--no-}"
    return 0
  fi
  printf '%s' "${flag}"
}

validate_runner_flags() {
  local -a flags=()
  local token
  for token in "${COMMON_ARGS[@]}"; do
    if [[ "${token}" == --* ]]; then
      flags+=("${token}")
    fi
  done
  for i in "${!PLAN_ARGS[@]}"; do
    IFS="${ARGS_SEP}" read -r -a AB_ARGS <<< "${PLAN_ARGS[$i]}"
    for token in "${AB_ARGS[@]}"; do
      if [[ "${token}" == --* ]]; then
        flags+=("${token}")
      fi
    done
  done

  local -A seen_flags=()
  local -a missing_flags=()
  local normalized
  for flag in "${flags[@]}"; do
    normalized="$(normalize_flag_for_runner_check "${flag}")"
    if [[ -n "${seen_flags[${normalized}]:-}" ]]; then
      continue
    fi
    seen_flags["${normalized}"]=1
    if ! grep -Fq "\"${normalized}\"" "${RUNNER}"; then
      missing_flags+=("${flag}")
    fi
  done

  if [[ ${#missing_flags[@]} -gt 0 ]]; then
    log "ERROR: el launcher referencia flags no soportados por el runner: $(join_by ', ' "${missing_flags[@]}")"
    exit 1
  fi
}

validate_plan_integrity() {
  local -A seen_ids=()
  local -A seen_hypothesis=()
  for i in "${!PLAN_IDS[@]}"; do
    local ablation_id="${PLAN_IDS[$i]}"
    local matrix_group="${PLAN_GROUPS[$i]}"
    local description="${PLAN_DESCRIPTIONS[$i]}"
    local hypothesis="${PLAN_HYPOTHESES[$i]}"
    if [[ -n "${seen_ids[${ablation_id}]:-}" ]]; then
      log "ERROR: ablation_id duplicado: ${ablation_id}"
      exit 1
    fi
    seen_ids["${ablation_id}"]=1
    if [[ -z "${matrix_group}" || -z "${description}" || -z "${hypothesis}" ]]; then
      log "ERROR: ablation incompleta: ${ablation_id}"
      exit 1
    fi
    seen_hypothesis["${hypothesis}"]=1
  done
}

emit_plan_rows() {
  local idx=0
  for seed in "${SEEDS[@]}"; do
    for i in "${!PLAN_IDS[@]}"; do
      idx=$((idx + 1))
      local ablation_id="${PLAN_IDS[$i]}"
      local matrix_group="${PLAN_GROUPS[$i]}"
      local description="${PLAN_DESCRIPTIONS[$i]}"
      local hypothesis="${PLAN_HYPOTHESES[$i]}"
      local base_dir="${RUNS_ROOT}/${ablation_id}/seed_${seed}"
      local console_log="${CONSOLE_DIR}/${ablation_id}__seed_${seed}.log"
      IFS="${ARGS_SEP}" read -r -a AB_ARGS <<< "${PLAN_ARGS[$i]}"
      local cmd_arr=("${PYTHON_BIN}" "${RUNNER}" "${COMMON_ARGS[@]}" --seed "${seed}" "${AB_ARGS[@]}" --log-dir "${base_dir}")
      local cmd
      cmd="$(printf '%q ' "${cmd_arr[@]}")"
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${idx}" \
        "$(csv_escape "${matrix_group}")" \
        "$(csv_escape "${ablation_id}")" \
        "${seed}" \
        "$(csv_escape "${description}")" \
        "$(csv_escape "${hypothesis}")" \
        "$(csv_escape "${base_dir}")" \
        "$(csv_escape "${console_log}")" \
        "$(csv_escape "${cmd}")" >> "${PLAN_CSV}"
    done
  done
  TOTAL_RUNS="${idx}"
}

find_latest_run_dir() {
  local base_dir="$1"
  if [[ ! -d "${base_dir}" ]]; then
    return 0
  fi
  find "${base_dir}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

append_manifest_row() {
  local planned_index="$1"
  local matrix_group="$2"
  local ablation_id="$3"
  local seed="$4"
  local description="$5"
  local hypothesis="$6"
  local status="$7"
  local exit_code="$8"
  local base_dir="$9"
  local run_dir="${10}"
  local console_log="${11}"
  local cmd="${12}"

  local summary_json=""
  local config_json=""
  local eval_metrics_csv=""
  local train_metrics_csv=""
  local tb_scalars_export_csv=""
  local train_monitor_csv=""
  local eval_monitor_csv=""

  if [[ -n "${run_dir}" && -d "${run_dir}" ]]; then
    [[ -f "${run_dir}/summary.json" ]] && summary_json="${run_dir}/summary.json"
    [[ -f "${run_dir}/config.json" ]] && config_json="${run_dir}/config.json"
    [[ -f "${run_dir}/eval_metrics.csv" ]] && eval_metrics_csv="${run_dir}/eval_metrics.csv"
    [[ -f "${run_dir}/train_metrics.csv" ]] && train_metrics_csv="${run_dir}/train_metrics.csv"
    [[ -f "${run_dir}/tb_scalars_export.csv" ]] && tb_scalars_export_csv="${run_dir}/tb_scalars_export.csv"
    [[ -f "${run_dir}/train.monitor.csv" ]] && train_monitor_csv="${run_dir}/train.monitor.csv"
    [[ -f "${run_dir}/eval.monitor.csv" ]] && eval_monitor_csv="${run_dir}/eval.monitor.csv"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${planned_index}" \
    "$(csv_escape "${matrix_group}")" \
    "$(csv_escape "${ablation_id}")" \
    "${seed}" \
    "$(csv_escape "${description}")" \
    "$(csv_escape "${hypothesis}")" \
    "$(csv_escape "${status}")" \
    "${exit_code}" \
    "$(csv_escape "${base_dir}")" \
    "$(csv_escape "${run_dir}")" \
    "$(csv_escape "${summary_json}")" \
    "$(csv_escape "${config_json}")" \
    "$(csv_escape "${eval_metrics_csv}")" \
    "$(csv_escape "${train_metrics_csv}")" \
    "$(csv_escape "${tb_scalars_export_csv}")" \
    "$(csv_escape "${train_monitor_csv}")" \
    "$(csv_escape "${eval_monitor_csv}")" \
    "$(csv_escape "${console_log}")" \
    "$(csv_escape "${cmd}")" >> "${MANIFEST_CSV}"
}

run_one() {
  local planned_index="$1"
  local matrix_group="$2"
  local ablation_id="$3"
  local description="$4"
  local hypothesis="$5"
  local seed="$6"
  shift 6
  local ablation_args=("$@")

  local base_dir="${RUNS_ROOT}/${ablation_id}/seed_${seed}"
  local console_log="${CONSOLE_DIR}/${ablation_id}__seed_${seed}.log"
  mkdir -p "${base_dir}" "$(dirname "${console_log}")"

  local cmd=("${PYTHON_BIN}" "${RUNNER}" "${COMMON_ARGS[@]}" --seed "${seed}" "${ablation_args[@]}" --log-dir "${base_dir}")
  local cmd_str
  cmd_str="$(printf '%q ' "${cmd[@]}")"

  log "[${planned_index}/${TOTAL_RUNS}] Iniciando ${ablation_id} | seed=${seed} | group=${matrix_group}"
  log "Hipótesis: ${hypothesis}"
  log "Output base: ${base_dir}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN=1, se omite ejecución real"
    append_manifest_row "${planned_index}" "${matrix_group}" "${ablation_id}" "${seed}" "${description}" "${hypothesis}" "dry_run" 0 "${base_dir}" "" "${console_log}" "${cmd_str}"
    return 0
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "${console_log}"
  local exit_code=${PIPESTATUS[0]}
  set -e

  local run_dir=""
  run_dir="$(find_latest_run_dir "${base_dir}")"

  if [[ ${exit_code} -eq 0 ]]; then
    if [[ -n "${run_dir}" && -f "${run_dir}/summary.json" && -f "${run_dir}/config.json" ]]; then
      SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
      append_manifest_row "${planned_index}" "${matrix_group}" "${ablation_id}" "${seed}" "${description}" "${hypothesis}" "ok" "${exit_code}" "${base_dir}" "${run_dir}" "${console_log}" "${cmd_str}"
      log "Finalizado ${ablation_id} | seed=${seed} | run_dir=${run_dir}"
    else
      FAILURES=$((FAILURES + 1))
      append_manifest_row "${planned_index}" "${matrix_group}" "${ablation_id}" "${seed}" "${description}" "${hypothesis}" "artifact_missing" 0 "${base_dir}" "${run_dir}" "${console_log}" "${cmd_str}"
      log "FALLÓ post-run ${ablation_id} | seed=${seed} | artifacts principales ausentes"
    fi
  else
    FAILURES=$((FAILURES + 1))
    append_manifest_row "${planned_index}" "${matrix_group}" "${ablation_id}" "${seed}" "${description}" "${hypothesis}" "failed" "${exit_code}" "${base_dir}" "${run_dir}" "${console_log}" "${cmd_str}"
    log "FALLÓ ${ablation_id} | seed=${seed} | exit_code=${exit_code}"
  fi

  return 0
}

run_all() {
  local planned_index=0
  for seed in "${SEEDS[@]}"; do
    for i in "${!PLAN_IDS[@]}"; do
      planned_index=$((planned_index + 1))
      IFS="${ARGS_SEP}" read -r -a AB_ARGS <<< "${PLAN_ARGS[$i]}"
      run_one \
        "${planned_index}" \
        "${PLAN_GROUPS[$i]}" \
        "${PLAN_IDS[$i]}" \
        "${PLAN_DESCRIPTIONS[$i]}" \
        "${PLAN_HYPOTHESES[$i]}" \
        "${seed}" \
        "${AB_ARGS[@]}"
    done
  done
}

run_analysis() {
  log "Ejecutando agregación final de resultados"
  set +e
  "${PYTHON_BIN}" "${AGGREGATOR}" \
    --root-dir "${RUNS_ROOT}" \
    --manifest "${MANIFEST_CSV}" \
    --output-dir "${ANALYSIS_DIR}" 2>&1 | tee "${ANALYSIS_LOG}"
  local exit_code=${PIPESTATUS[0]}
  set -e
  if [[ ${exit_code} -ne 0 ]]; then
    log "ERROR: la agregación final falló con exit_code=${exit_code}"
    return "${exit_code}"
  fi
}

print_footer() {
  log "Sesión completada"
  log "Session root: ${SESSION_ROOT}"
  log "Runs root: ${RUNS_ROOT}"
  log "Plan CSV: ${PLAN_CSV}"
  log "Manifest CSV: ${MANIFEST_CSV}"
  log "Analysis dir: ${ANALYSIS_DIR}"
  log "Analysis log: ${ANALYSIS_LOG}"
  log "Runs exitosos: ${SUCCESSFUL_RUNS}"
  log "Runs fallidos: ${FAILURES}"
}

main() {
  validate_paths
  validate_scalar_inputs
  read_seeds
  validate_seeds
  build_ablation_plan
  apply_ablation_filter
  validate_plan_integrity
  validate_runner_flags
  write_session_config
  init_csvs
  emit_plan_rows

  log "Session root: ${SESSION_ROOT}"
  log "Run profile: ${RUN_PROFILE}"
  log "Task preset: ${TASK_PRESET} | Obs mode: ${OBS_MODE} | FullyObs: ${FULLY_OBSERVABLE} | Seeds: $(join_by ',' "${SEEDS[@]}")"
  log "Eval modes: ${EVAL_POLICY_MODES_CSV} | Summary: ${SUMMARY_MODEL_VARIANT}/${SUMMARY_EVAL_POLICY_MODE}"
  log "Eval seed offsets: ${EVAL_SEED_OFFSETS_CSV} | Summary offset: ${SUMMARY_EVAL_SEED_OFFSET} | n_envs: ${N_ENVS} | vec_env: ${VEC_ENV}"
  log "PPO: lr=${PPO_LEARNING_RATE} n_steps=${PPO_N_STEPS} batch=${PPO_BATCH_SIZE} epochs=${PPO_N_EPOCHS} clip=${PPO_CLIP_RANGE} clip_vf=${PPO_CLIP_RANGE_VF} ent=${PPO_ENT_COEF} target_kl=${PPO_TARGET_KL}"
  if [[ -n "${PPO_ENT_COEF_TASK_OVERRIDES_CSV}" ]]; then
    log "PPO ent_coef overrides: ${PPO_ENT_COEF_TASK_OVERRIDES_CSV}"
  fi
  log "Periodic eval: scope=${PERIODIC_EVAL_SCOPE} | freq=${PERIODIC_EVAL_FREQ} | save_best_eval=${SAVE_BEST_EVAL_CHECKPOINT} | eval_best=${EVAL_BEST_EVAL_MODEL_AT_PHASE_END} | restore_best_eval=${RESTORE_BEST_EVAL_MODEL} | early_stop=${EARLY_STOP_EVAL_SUCCESS_THRESHOLD:-<off>}"
  log "Ablaciones planeadas: ${#PLAN_IDS[@]} | Corridas totales: ${TOTAL_RUNS}"
  log "Ablation filter: ${ABLATION_IDS_CSV:-<none>}"
  log "Rank sweep adicional: ${ENABLE_RANK_SWEEP} | Single-task sanity: ${ENABLE_SINGLE_TASK_SANITY} | Only sanity: ${ONLY_SANITY}"
  log "Sanity tasks: ${SANITY_TASKS_CSV:-<preset>}"
  log "Continual tasks: ${CONTINUAL_TASKS_CSV:-<preset>}"
  if [[ -n "${SANITY_TASKS_CSV}" && "${ENABLE_SINGLE_TASK_SANITY}" != "1" ]]; then
    log "NOTA: SANITY_TASKS_CSV está definido, pero no se correrán sanities single-task porque ENABLE_SINGLE_TASK_SANITY=0 y RUN_PROFILE!=sanity."
  fi

  run_all
  local analysis_status="ok"
  if ! run_analysis; then
    analysis_status="failed"
    FAILURES=$((FAILURES + 1))
  fi
  write_session_status "${analysis_status}"
  print_footer

  if [[ ${FAILURES} -gt 0 ]]; then
    exit 1
  fi
}

main "$@"
