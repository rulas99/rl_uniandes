#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/minigrid_crl_runner_adapter.py}"
AGGREGATOR="${AGGREGATOR:-${SCRIPT_DIR}/aggregate_ablation_results.py}"

TASK_PRESET="${TASK_PRESET:-smoke4}"
OBS_MODE="${OBS_MODE:-image}"
STEPS_PER_TASK="${STEPS_PER_TASK:-400000}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
SEEDS_CSV="${SEEDS_CSV:-42,43,44}"
EXP_ROOT="${EXP_ROOT:-logs/minigrid_ablations/${TASK_PRESET}}"
ENABLE_RANK_SWEEP="${ENABLE_RANK_SWEEP:-0}"
DRY_RUN="${DRY_RUN:-0}"

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
  --ppo-learning-rate 1e-4
  --ppo-n-steps 4096
  --ppo-batch-size 512
  --ppo-n-epochs 4
  --ppo-clip-range 0.1
  --ppo-clip-range-vf 0.1
  --ppo-ent-coef 0.005
  --ppo-target-kl 0.01
  --reset-optimizers-every-task
  --reset-metric-windows-every-task
)

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
  if [[ -z "${TASK_PRESET}" || -z "${OBS_MODE}" ]]; then
    log "ERROR: TASK_PRESET y OBS_MODE no pueden quedar vacíos"
    exit 1
  fi
  if [[ "${STEPS_PER_TASK}" -le 0 || "${EVAL_EPISODES}" -le 0 || "${MAX_EPISODE_STEPS}" -le 0 ]]; then
    log "ERROR: steps/evals/max_episode_steps deben ser > 0"
    exit 1
  fi
  if [[ "${ENABLE_RANK_SWEEP}" != "0" && "${ENABLE_RANK_SWEEP}" != "1" ]]; then
    log "ERROR: ENABLE_RANK_SWEEP debe ser 0 o 1"
    exit 1
  fi
  if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    log "ERROR: DRY_RUN debe ser 0 o 1"
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
  "task_preset": "${TASK_PRESET}",
  "obs_mode": "${OBS_MODE}",
  "steps_per_task": ${STEPS_PER_TASK},
  "eval_episodes": ${EVAL_EPISODES},
  "max_episode_steps": ${MAX_EPISODE_STEPS},
  "seeds_csv": "${SEEDS_CSV}",
  "exp_root": "${EXP_ROOT}",
  "session_root": "${SESSION_ROOT}",
  "runs_root": "${RUNS_ROOT}",
  "runner": "${RUNNER}",
  "aggregator": "${AGGREGATOR}",
  "enable_rank_sweep": ${ENABLE_RANK_SWEEP},
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

register_ablation() {
  local ablation_id="$1"
  local matrix_group="$2"
  local description="$3"
  local hypothesis="$4"
  shift 4
  local joined_args
  joined_args="$(printf '%q ' "$@")"
  PLAN_IDS+=("${ablation_id}")
  PLAN_GROUPS+=("${matrix_group}")
  PLAN_DESCRIPTIONS+=("${description}")
  PLAN_HYPOTHESES+=("${hypothesis}")
  PLAN_ARGS+=("${joined_args}")
}

build_ablation_plan() {
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
    IFS=' ' read -r -a AB_ARGS <<< "${PLAN_ARGS[$i]}"
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
      local cmd="${PYTHON_BIN} ${RUNNER} ${COMMON_ARGS[*]} --seed ${seed} ${PLAN_ARGS[$i]} --log-dir ${base_dir}"
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
      IFS=' ' read -r -a AB_ARGS <<< "${PLAN_ARGS[$i]}"
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
  validate_plan_integrity
  validate_runner_flags
  write_session_config
  init_csvs
  emit_plan_rows

  log "Session root: ${SESSION_ROOT}"
  log "Task preset: ${TASK_PRESET} | Obs mode: ${OBS_MODE} | Seeds: $(join_by ',' "${SEEDS[@]}")"
  log "Ablaciones planeadas: ${#PLAN_IDS[@]} | Corridas totales: ${TOTAL_RUNS}"
  log "Rank sweep adicional: ${ENABLE_RANK_SWEEP}"
  log "Plan group counts: core/reference/rank_sweep"

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
