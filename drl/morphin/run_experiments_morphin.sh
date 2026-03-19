#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_PROFILE="${RUN_PROFILE:-tier0_quick}"
BENCHMARK="${BENCHMARK:-gw_goal_conditioned_balanced_ac_v1}"
BENCHMARKS_CSV="${BENCHMARKS_CSV:-}"
SESSION_GROUP="${SESSION_GROUP:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/morphin_gridworld}"
METHODS_CSV="${METHODS_CSV:-}"
METHOD_SET="${METHOD_SET:-}"
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
  tier2_hidden_long)
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    METHOD_SET="${METHOD_SET:-core_no_detector}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw_hidden_goal_balanced_ab_v1,gw_hidden_goal_balanced_aba_v1,gw_hidden_goal_balanced_abab_v1,gw_hidden_goal_balanced_ababa_v1}"
    SESSION_GROUP="${SESSION_GROUP:-campaign_hidden_long}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
    EVAL_EPISODES="${EVAL_EPISODES:-20}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-10000}"
    ;;
  tier3_context9_pilot)
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    METHOD_SET="${METHOD_SET:-segmented_shortlist}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw9_context_balanced_ab_v1,gw9_context_balanced_ba_v1,gw9_context_balanced_aba_v1,gw9_context_balanced_bab_v1}"
    SESSION_GROUP="${SESSION_GROUP:-campaign_context9_pilot}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
    MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-250}"
    EVAL_EPISODES="${EVAL_EPISODES:-20}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-1000}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-20000}"
    EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-4000}"
    POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-7500}"
    ;;
  tier3_context9_long)
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    METHOD_SET="${METHOD_SET:-segmented_shortlist}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw9_context_balanced_ab_v1,gw9_context_balanced_ba_v1,gw9_context_balanced_aba_v1,gw9_context_balanced_bab_v1,gw9_context_balanced_abab_v1,gw9_context_balanced_baba_v1,gw9_context_balanced_ababa_v1,gw9_context_balanced_babab_v1}"
    SESSION_GROUP="${SESSION_GROUP:-campaign_context9_long}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46,47,48,49,50,51}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
    MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-250}"
    EVAL_EPISODES="${EVAL_EPISODES:-20}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-1000}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-20000}"
    EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-4000}"
    POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-7500}"
    ;;
  tier3_context9_revisit_ablation)
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    REUSE_SCRATCH_REFS_BY_TASK_SET="${REUSE_SCRATCH_REFS_BY_TASK_SET:-1}"
    METHOD_SET="${METHOD_SET:-segmented_revisit_ablation}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw9_context_balanced_ab_v1,gw9_context_balanced_ba_v1,gw9_context_balanced_aba_v1,gw9_context_balanced_bab_v1}"
    SESSION_GROUP="${SESSION_GROUP:-campaign_context9_revisit_ablation}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46,47,48,49,50,51}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
    MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-250}"
    SCRATCH_EPISODES_PER_TASK="${SCRATCH_EPISODES_PER_TASK:-800}"
    SCRATCH_MAX_STEPS_PER_EPISODE="${SCRATCH_MAX_STEPS_PER_EPISODE:-250}"
    EVAL_EPISODES="${EVAL_EPISODES:-20}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-1000}"
    SCRATCH_WARMUP_STEPS="${SCRATCH_WARMUP_STEPS:-1000}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    SCRATCH_BATCH_SIZE="${SCRATCH_BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-20000}"
    SCRATCH_EPS_DECAY_STEPS="${SCRATCH_EPS_DECAY_STEPS:-20000}"
    EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-4000}"
    POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-7500}"
    SCRATCH_REF_MIN_VALID_FRACTION="${SCRATCH_REF_MIN_VALID_FRACTION:-0.8}"
    ;;
  tier4_context9_plus_diag)
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    REUSE_SCRATCH_REFS_BY_TASK_SET="${REUSE_SCRATCH_REFS_BY_TASK_SET:-1}"
    SCRATCH_GATE_ENABLED="${SCRATCH_GATE_ENABLED:-1}"
    METHOD_SET="${METHOD_SET:-segmented_plus_shortlist}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw9_context_calibrated3_ab_v1,gw9_context_calibrated3_ba_v1,gw9_context_calibrated3_aba_v1,gw9_context_calibrated3_bab_v1,gw9_context_calibrated3_ac_v1,gw9_context_calibrated3_ca_v1,gw9_context_calibrated3_bc_v1,gw9_context_calibrated3_cb_v1,gw9_context_calibrated3_abc_v1,gw9_context_calibrated3_bac_v1}"
    SESSION_GROUP="${SESSION_GROUP:-campaign_context9_plus_diag}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
    MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-250}"
    SCRATCH_EPISODES_PER_TASK="${SCRATCH_EPISODES_PER_TASK:-800}"
    SCRATCH_MAX_STEPS_PER_EPISODE="${SCRATCH_MAX_STEPS_PER_EPISODE:-250}"
    EVAL_EPISODES="${EVAL_EPISODES:-20}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-1000}"
    SCRATCH_WARMUP_STEPS="${SCRATCH_WARMUP_STEPS:-1000}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    SCRATCH_BATCH_SIZE="${SCRATCH_BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-20000}"
    SCRATCH_EPS_DECAY_STEPS="${SCRATCH_EPS_DECAY_STEPS:-20000}"
    EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-4000}"
    POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-7500}"
    SCRATCH_REF_MIN_VALID_FRACTION="${SCRATCH_REF_MIN_VALID_FRACTION:-0.8}"
    ;;
  thesis_goal_transfer)
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    METHOD_SET="${METHOD_SET:-thesis_core}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw_goal_conditioned_balanced_ac_v1,gw_goal_conditioned_balanced_ca_v1,gw_goal_conditioned_balanced_aba_v1}"
    SESSION_GROUP="${SESSION_GROUP:-thesis_goal_transfer}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46,47,48,49,50,51,52,53,54,55,56}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
    EVAL_EPISODES="${EVAL_EPISODES:-30}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-10000}"
    ;;
  thesis_hidden)
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    METHOD_SET="${METHOD_SET:-thesis_core}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw_hidden_goal_balanced_ab_v1,gw_hidden_goal_balanced_aba_v1}"
    SESSION_GROUP="${SESSION_GROUP:-thesis_hidden}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46,47,48,49,50,51,52,53,54,55,56}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
    EVAL_EPISODES="${EVAL_EPISODES:-30}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-10000}"
    ARCHIVE_FRAC="${ARCHIVE_FRAC:-0.0}"
    ;;
  thesis_ablation)
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    METHOD_SET="${METHOD_SET:-thesis_ablation}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw_goal_conditioned_balanced_ac_v1}"
    SESSION_GROUP="${SESSION_GROUP:-thesis_ablation}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46,47,48,49,50,51,52,53,54,55,56}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
    EVAL_EPISODES="${EVAL_EPISODES:-30}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-10000}"
    ;;
  thesis_9x9_quick)
    # 9x9 balanced GridWorld — fast iteration profile for hyperparameter tuning.
    # 5 seeds, 300 eps/task, all three benchmarks (AB, AC, ABA).
    # Use this to validate eps_reset and archive_frac changes before committing to
    # the full 15-seed thesis_9x9_goalcond run (~2-3 hrs vs ~12-14 hrs).
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    REUSE_SCRATCH_REFS_BY_TASK_SET="${REUSE_SCRATCH_REFS_BY_TASK_SET:-1}"
    METHOD_SET="${METHOD_SET:-thesis_core}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw9_goal_balanced_ab_v1,gw9_goal_balanced_ac_v1,gw9_goal_balanced_aba_v1}"
    SESSION_GROUP="${SESSION_GROUP:-thesis_9x9_quick}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-300}"
    MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-250}"
    SCRATCH_EPISODES_PER_TASK="${SCRATCH_EPISODES_PER_TASK:-500}"
    SCRATCH_MAX_STEPS_PER_EPISODE="${SCRATCH_MAX_STEPS_PER_EPISODE:-250}"
    EVAL_EPISODES="${EVAL_EPISODES:-15}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-750}"
    SCRATCH_WARMUP_STEPS="${SCRATCH_WARMUP_STEPS:-750}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-15000}"
    SCRATCH_EPS_DECAY_STEPS="${SCRATCH_EPS_DECAY_STEPS:-15000}"
    # Corrected eps-reset: 0.9 initial value, 30 000 steps decay ≈ 120 post-switch
    # episodes (vs. the old 0.4 / 3 000 = 12 episodes that caused binary failure on B).
    EPS_RESET_VALUE="${EPS_RESET_VALUE:-0.9}"
    EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-30000}"
    POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-7500}"
    # Reduced archive to limit task-A interference on new-task discovery.
    ARCHIVE_FRAC="${ARCHIVE_FRAC:-0.30}"
    HIDDEN_SIZES_CSV="${HIDDEN_SIZES_CSV:-128,128}"
    BUFFER_CAPACITY="${BUFFER_CAPACITY:-20000}"
    RECENT_BUFFER_CAPACITY="${RECENT_BUFFER_CAPACITY:-12000}"
    ARCHIVE_BUFFER_CAPACITY="${ARCHIVE_BUFFER_CAPACITY:-8000}"
    SCRATCH_BUFFER_CAPACITY="${SCRATCH_BUFFER_CAPACITY:-20000}"
    SCRATCH_REF_MIN_VALID_FRACTION="${SCRATCH_REF_MIN_VALID_FRACTION:-0.7}"
    ;;
  thesis_9x9_goalcond)
    # 9x9 balanced GridWorld, goal-conditioned (agent_target obs mode).
    # 2-task sequences AB, AC, and revisit ABA — mirrors the 5x5 thesis_goal_transfer design.
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    REUSE_SCRATCH_REFS_BY_TASK_SET="${REUSE_SCRATCH_REFS_BY_TASK_SET:-1}"
    METHOD_SET="${METHOD_SET:-thesis_core}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw9_goal_balanced_ab_v1,gw9_goal_balanced_ac_v1,gw9_goal_balanced_aba_v1}"
    SESSION_GROUP="${SESSION_GROUP:-thesis_9x9_goalcond}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46,47,48,49,50,51,52,53,54,55,56}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-600}"
    MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-250}"
    SCRATCH_EPISODES_PER_TASK="${SCRATCH_EPISODES_PER_TASK:-800}"
    SCRATCH_MAX_STEPS_PER_EPISODE="${SCRATCH_MAX_STEPS_PER_EPISODE:-250}"
    EVAL_EPISODES="${EVAL_EPISODES:-30}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-750}"
    SCRATCH_WARMUP_STEPS="${SCRATCH_WARMUP_STEPS:-750}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-15000}"
    SCRATCH_EPS_DECAY_STEPS="${SCRATCH_EPS_DECAY_STEPS:-15000}"
    EPS_RESET_VALUE="${EPS_RESET_VALUE:-0.9}"
    EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-30000}"
    POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-7500}"
    ARCHIVE_FRAC="${ARCHIVE_FRAC:-0.30}"
    HIDDEN_SIZES_CSV="${HIDDEN_SIZES_CSV:-128,128}"
    BUFFER_CAPACITY="${BUFFER_CAPACITY:-20000}"
    RECENT_BUFFER_CAPACITY="${RECENT_BUFFER_CAPACITY:-12000}"
    ARCHIVE_BUFFER_CAPACITY="${ARCHIVE_BUFFER_CAPACITY:-8000}"
    SCRATCH_BUFFER_CAPACITY="${SCRATCH_BUFFER_CAPACITY:-20000}"
    SCRATCH_REF_MIN_VALID_FRACTION="${SCRATCH_REF_MIN_VALID_FRACTION:-0.7}"
    ;;
  thesis_9x9_multitask)
    # 9x9 balanced GridWorld, 3-4 task sequences (ABC, ABCA).
    # Tests archive accumulation, forward transfer and revisit across 3+ tasks.
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    REUSE_SCRATCH_REFS_BY_TASK_SET="${REUSE_SCRATCH_REFS_BY_TASK_SET:-1}"
    METHOD_SET="${METHOD_SET:-thesis_core}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw9_goal_balanced_abc_v1,gw9_goal_balanced_abca_v1}"
    SESSION_GROUP="${SESSION_GROUP:-thesis_9x9_multitask}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46,47,48,49,50,51,52,53,54,55,56}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-600}"
    MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-250}"
    SCRATCH_EPISODES_PER_TASK="${SCRATCH_EPISODES_PER_TASK:-800}"
    SCRATCH_MAX_STEPS_PER_EPISODE="${SCRATCH_MAX_STEPS_PER_EPISODE:-250}"
    EVAL_EPISODES="${EVAL_EPISODES:-30}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-750}"
    SCRATCH_WARMUP_STEPS="${SCRATCH_WARMUP_STEPS:-750}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-15000}"
    SCRATCH_EPS_DECAY_STEPS="${SCRATCH_EPS_DECAY_STEPS:-15000}"
    EPS_RESET_VALUE="${EPS_RESET_VALUE:-0.9}"
    EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-30000}"
    POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-7500}"
    ARCHIVE_FRAC="${ARCHIVE_FRAC:-0.10}"
    HIDDEN_SIZES_CSV="${HIDDEN_SIZES_CSV:-128,128}"
    BUFFER_CAPACITY="${BUFFER_CAPACITY:-20000}"
    RECENT_BUFFER_CAPACITY="${RECENT_BUFFER_CAPACITY:-12000}"
    ARCHIVE_BUFFER_CAPACITY="${ARCHIVE_BUFFER_CAPACITY:-8000}"
    SCRATCH_BUFFER_CAPACITY="${SCRATCH_BUFFER_CAPACITY:-20000}"
    SCRATCH_REF_MIN_VALID_FRACTION="${SCRATCH_REF_MIN_VALID_FRACTION:-0.7}"
    ;;
  thesis_distill_test)
    # Quick test: oracle_segmented vs oracle_segmented_distill, (128,128).
    # 3 seeds, 300 eps/task, AB + AC + ABA + ABC.
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    REUSE_SCRATCH_REFS_BY_TASK_SET="${REUSE_SCRATCH_REFS_BY_TASK_SET:-1}"
    METHOD_SET="${METHOD_SET:-thesis_distill}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw9_goal_balanced_ab_v1,gw9_goal_balanced_ac_v1,gw9_goal_balanced_aba_v1,gw9_goal_balanced_abc_v1}"
    SESSION_GROUP="${SESSION_GROUP:-thesis_distill_test}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-300}"
    MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-250}"
    SCRATCH_EPISODES_PER_TASK="${SCRATCH_EPISODES_PER_TASK:-500}"
    SCRATCH_MAX_STEPS_PER_EPISODE="${SCRATCH_MAX_STEPS_PER_EPISODE:-250}"
    EVAL_EPISODES="${EVAL_EPISODES:-15}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-750}"
    SCRATCH_WARMUP_STEPS="${SCRATCH_WARMUP_STEPS:-750}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-15000}"
    SCRATCH_EPS_DECAY_STEPS="${SCRATCH_EPS_DECAY_STEPS:-15000}"
    EPS_RESET_VALUE="${EPS_RESET_VALUE:-0.9}"
    EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-30000}"
    POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-7500}"
    ARCHIVE_FRAC="${ARCHIVE_FRAC:-0.10}"
    HIDDEN_SIZES_CSV="${HIDDEN_SIZES_CSV:-128,128}"
    DISTILL_LAMBDA="${DISTILL_LAMBDA:-0.005}"
    BUFFER_CAPACITY="${BUFFER_CAPACITY:-20000}"
    RECENT_BUFFER_CAPACITY="${RECENT_BUFFER_CAPACITY:-12000}"
    ARCHIVE_BUFFER_CAPACITY="${ARCHIVE_BUFFER_CAPACITY:-8000}"
    SCRATCH_BUFFER_CAPACITY="${SCRATCH_BUFFER_CAPACITY:-20000}"
    SCRATCH_REF_MIN_VALID_FRACTION="${SCRATCH_REF_MIN_VALID_FRACTION:-0.7}"
    ;;
  thesis_distill_test_256)
    # Same as thesis_distill_test but with (256,256) network.
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    REUSE_SCRATCH_REFS_BY_TASK_SET="${REUSE_SCRATCH_REFS_BY_TASK_SET:-1}"
    METHOD_SET="${METHOD_SET:-thesis_distill}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw9_goal_balanced_ab_v1,gw9_goal_balanced_ac_v1,gw9_goal_balanced_aba_v1,gw9_goal_balanced_abc_v1}"
    SESSION_GROUP="${SESSION_GROUP:-thesis_distill_test_256}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-300}"
    MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-250}"
    SCRATCH_EPISODES_PER_TASK="${SCRATCH_EPISODES_PER_TASK:-500}"
    SCRATCH_MAX_STEPS_PER_EPISODE="${SCRATCH_MAX_STEPS_PER_EPISODE:-250}"
    EVAL_EPISODES="${EVAL_EPISODES:-15}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-750}"
    SCRATCH_WARMUP_STEPS="${SCRATCH_WARMUP_STEPS:-750}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-15000}"
    SCRATCH_EPS_DECAY_STEPS="${SCRATCH_EPS_DECAY_STEPS:-15000}"
    EPS_RESET_VALUE="${EPS_RESET_VALUE:-0.9}"
    EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-30000}"
    POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-7500}"
    ARCHIVE_FRAC="${ARCHIVE_FRAC:-0.10}"
    HIDDEN_SIZES_CSV="${HIDDEN_SIZES_CSV:-256,256}"
    DISTILL_LAMBDA="${DISTILL_LAMBDA:-0.005}"
    BUFFER_CAPACITY="${BUFFER_CAPACITY:-20000}"
    RECENT_BUFFER_CAPACITY="${RECENT_BUFFER_CAPACITY:-12000}"
    ARCHIVE_BUFFER_CAPACITY="${ARCHIVE_BUFFER_CAPACITY:-8000}"
    SCRATCH_BUFFER_CAPACITY="${SCRATCH_BUFFER_CAPACITY:-20000}"
    SCRATCH_REF_MIN_VALID_FRACTION="${SCRATCH_REF_MIN_VALID_FRACTION:-0.7}"
    ;;
  thesis_lambda_sweep)
    # Lambda sweep for oracle_segmented_distill on (256,256), AB+AC only, 3 seeds.
    # Tests lambda in {0.001, 0.005, 0.020, 0.050, 0.200} vs no-distill baseline.
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    REUSE_SCRATCH_REFS_BY_TASK_SET="${REUSE_SCRATCH_REFS_BY_TASK_SET:-1}"
    METHOD_SET="${METHOD_SET:-thesis_lambda_sweep}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw9_goal_balanced_ab_v1,gw9_goal_balanced_ac_v1}"
    SESSION_GROUP="${SESSION_GROUP:-thesis_lambda_sweep}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-300}"
    MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-250}"
    SCRATCH_EPISODES_PER_TASK="${SCRATCH_EPISODES_PER_TASK:-500}"
    SCRATCH_MAX_STEPS_PER_EPISODE="${SCRATCH_MAX_STEPS_PER_EPISODE:-250}"
    EVAL_EPISODES="${EVAL_EPISODES:-15}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-750}"
    SCRATCH_WARMUP_STEPS="${SCRATCH_WARMUP_STEPS:-750}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-15000}"
    SCRATCH_EPS_DECAY_STEPS="${SCRATCH_EPS_DECAY_STEPS:-15000}"
    EPS_RESET_VALUE="${EPS_RESET_VALUE:-0.9}"
    EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-30000}"
    POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-7500}"
    ARCHIVE_FRAC="${ARCHIVE_FRAC:-0.10}"
    HIDDEN_SIZES_CSV="${HIDDEN_SIZES_CSV:-256,256}"
    BUFFER_CAPACITY="${BUFFER_CAPACITY:-20000}"
    RECENT_BUFFER_CAPACITY="${RECENT_BUFFER_CAPACITY:-12000}"
    ARCHIVE_BUFFER_CAPACITY="${ARCHIVE_BUFFER_CAPACITY:-8000}"
    SCRATCH_BUFFER_CAPACITY="${SCRATCH_BUFFER_CAPACITY:-20000}"
    SCRATCH_REF_MIN_VALID_FRACTION="${SCRATCH_REF_MIN_VALID_FRACTION:-0.7}"
    ;;
  thesis_9x9_full_256)
    # Definitive 4-method comparison on (256,256): ddqn_vanilla, oracle_reset,
    # oracle_segmented, oracle_segmented_distill_l001 (λ=0.001, Option B fix:
    # TD on all samples + distill anchor on archive).
    # 5 seeds default (override via SEEDS_CSV for overnight runs).
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    REUSE_SCRATCH_REFS_BY_TASK_SET="${REUSE_SCRATCH_REFS_BY_TASK_SET:-1}"
    METHOD_SET="${METHOD_SET:-thesis_full_256}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw9_goal_balanced_ab_v1,gw9_goal_balanced_ac_v1,gw9_goal_balanced_aba_v1,gw9_goal_balanced_abc_v1}"
    SESSION_GROUP="${SESSION_GROUP:-thesis_9x9_full_256}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
    MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-250}"
    SCRATCH_EPISODES_PER_TASK="${SCRATCH_EPISODES_PER_TASK:-500}"
    SCRATCH_MAX_STEPS_PER_EPISODE="${SCRATCH_MAX_STEPS_PER_EPISODE:-250}"
    EVAL_EPISODES="${EVAL_EPISODES:-15}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-750}"
    SCRATCH_WARMUP_STEPS="${SCRATCH_WARMUP_STEPS:-750}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-15000}"
    SCRATCH_EPS_DECAY_STEPS="${SCRATCH_EPS_DECAY_STEPS:-15000}"
    EPS_RESET_VALUE="${EPS_RESET_VALUE:-0.9}"
    EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-30000}"
    POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-7500}"
    ARCHIVE_FRAC="${ARCHIVE_FRAC:-0.10}"
    HIDDEN_SIZES_CSV="${HIDDEN_SIZES_CSV:-256,256}"
    BUFFER_CAPACITY="${BUFFER_CAPACITY:-20000}"
    RECENT_BUFFER_CAPACITY="${RECENT_BUFFER_CAPACITY:-12000}"
    ARCHIVE_BUFFER_CAPACITY="${ARCHIVE_BUFFER_CAPACITY:-8000}"
    SCRATCH_BUFFER_CAPACITY="${SCRATCH_BUFFER_CAPACITY:-20000}"
    SCRATCH_REF_MIN_VALID_FRACTION="${SCRATCH_REF_MIN_VALID_FRACTION:-0.7}"
    ;;
  thesis_archive_sweep)
    # Archive fraction sweep: oracle_segmented with af=0.10,0.15,0.20,0.25,0.30
    # Tests whether larger archive improves retention to distill-level without distillation.
    # 5 methods × 8 seeds × 4 benchmarks = 160 runs (~6-7 hrs)
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    REUSE_SCRATCH_REFS_BY_TASK_SET="${REUSE_SCRATCH_REFS_BY_TASK_SET:-1}"
    METHOD_SET="${METHOD_SET:-thesis_archive_sweep}"
    BENCHMARKS_CSV="${BENCHMARKS_CSV:-gw9_goal_balanced_ab_v1,gw9_goal_balanced_ac_v1,gw9_goal_balanced_aba_v1,gw9_goal_balanced_abc_v1}"
    SESSION_GROUP="${SESSION_GROUP:-thesis_archive_sweep}"
    SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46,47,48,49}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-400}"
    MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-250}"
    SCRATCH_EPISODES_PER_TASK="${SCRATCH_EPISODES_PER_TASK:-500}"
    SCRATCH_MAX_STEPS_PER_EPISODE="${SCRATCH_MAX_STEPS_PER_EPISODE:-250}"
    EVAL_EPISODES="${EVAL_EPISODES:-15}"
    EVAL_EVERY_EPISODES="${EVAL_EVERY_EPISODES:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-750}"
    SCRATCH_WARMUP_STEPS="${SCRATCH_WARMUP_STEPS:-750}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    EPS_DECAY_STEPS="${EPS_DECAY_STEPS:-15000}"
    SCRATCH_EPS_DECAY_STEPS="${SCRATCH_EPS_DECAY_STEPS:-15000}"
    EPS_RESET_VALUE="${EPS_RESET_VALUE:-0.9}"
    EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-30000}"
    POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-7500}"
    ARCHIVE_FRAC="${ARCHIVE_FRAC:-0.10}"
    HIDDEN_SIZES_CSV="${HIDDEN_SIZES_CSV:-256,256}"
    BUFFER_CAPACITY="${BUFFER_CAPACITY:-20000}"
    RECENT_BUFFER_CAPACITY="${RECENT_BUFFER_CAPACITY:-12000}"
    ARCHIVE_BUFFER_CAPACITY="${ARCHIVE_BUFFER_CAPACITY:-8000}"
    SCRATCH_BUFFER_CAPACITY="${SCRATCH_BUFFER_CAPACITY:-20000}"
    SCRATCH_REF_MIN_VALID_FRACTION="${SCRATCH_REF_MIN_VALID_FRACTION:-0.7}"
    ;;
  thesis_der_validate)
    # DER++ validation: 4 methods × 4 seeds × 2 benchmarks = 32 runs (~1-2 hrs)
    # Validates that DER++ implementation is correct before the full overnight run.
    # NOTE: all vars use direct assignment (no :-) to prevent parent-env contamination.
    MODE="continual"
    AUTO_BUILD_SCRATCH_REFS=1
    REUSE_SCRATCH_REFS_BY_TASK_SET=1
    METHOD_SET="thesis_der_validate"
    BENCHMARKS_CSV="gw9_goal_balanced_ab_v1,gw9_goal_balanced_abc_v1"
    SESSION_GROUP="thesis_der_validate"
    SEEDS_CSV="42,43,44,45"
    EPISODES_PER_TASK=400
    MAX_STEPS_PER_EPISODE=250
    SCRATCH_EPISODES_PER_TASK=500
    SCRATCH_MAX_STEPS_PER_EPISODE=250
    EVAL_EPISODES=15
    EVAL_EVERY_EPISODES=25
    WARMUP_STEPS=750
    SCRATCH_WARMUP_STEPS=750
    BATCH_SIZE=64
    EPS_DECAY_STEPS=15000
    SCRATCH_EPS_DECAY_STEPS=15000
    EPS_RESET_VALUE=0.9
    EPS_DECAY_STEPS_AFTER_SWITCH=30000
    POST_SWITCH_STEPS=7500
    ARCHIVE_FRAC=0.10
    HIDDEN_SIZES_CSV=256,256
    BUFFER_CAPACITY=20000
    RECENT_BUFFER_CAPACITY=12000
    ARCHIVE_BUFFER_CAPACITY=8000
    SCRATCH_BUFFER_CAPACITY=20000
    SCRATCH_REF_MIN_VALID_FRACTION=0.7
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
    gw_hidden_goal_balanced_ab_v1|gw_hidden_goal_balanced_aba_v1|gw_hidden_goal_balanced_abab_v1|gw_hidden_goal_balanced_ababa_v1)
      printf '%s\n%s\n' "agent_only" "gw_goal_bal_a,gw_goal_bal_b"
      ;;
    gw9_goal_balanced_ab_v1|gw9_goal_balanced_ba_v1|gw9_goal_balanced_ac_v1|gw9_goal_balanced_ca_v1|gw9_goal_balanced_aba_v1)
      printf '%s\n%s\n' "agent_target" "gw9_goal_bal_a,gw9_goal_bal_b,gw9_goal_bal_c"
      ;;
    gw9_goal_balanced_abc_v1|gw9_goal_balanced_abca_v1)
      printf '%s\n%s\n' "agent_target" "gw9_goal_bal_a,gw9_goal_bal_b,gw9_goal_bal_c"
      ;;
    gw9_context_balanced_ab_v1|gw9_context_balanced_ba_v1|gw9_context_balanced_aba_v1|gw9_context_balanced_bab_v1|gw9_context_balanced_abab_v1|gw9_context_balanced_baba_v1|gw9_context_balanced_ababa_v1|gw9_context_balanced_babab_v1)
      printf '%s\n%s\n' "agent_context" "gw9_goal_bal_a,gw9_goal_bal_b"
      ;;
    gw9_context_calibrated_ab_v1|gw9_context_calibrated_ba_v1|gw9_context_calibrated_aba_v1|gw9_context_calibrated_bab_v1|gw9_context_calibrated_ac_v1|gw9_context_calibrated_ca_v1|gw9_context_calibrated_bc_v1|gw9_context_calibrated_cb_v1|gw9_context_calibrated_abc_v1|gw9_context_calibrated_bac_v1)
      printf '%s\n%s\n' "agent_context_calibrated" "gw9_goal_cal_a,gw9_goal_cal_b,gw9_goal_cal_c"
      ;;
    gw9_context_calibrated2_ab_v1|gw9_context_calibrated2_ba_v1|gw9_context_calibrated2_aba_v1|gw9_context_calibrated2_bab_v1|gw9_context_calibrated2_ac_v1|gw9_context_calibrated2_ca_v1|gw9_context_calibrated2_bc_v1|gw9_context_calibrated2_cb_v1|gw9_context_calibrated2_abc_v1|gw9_context_calibrated2_bac_v1)
      printf '%s\n%s\n' "agent_context_calibrated_v2" "gw9_goal_cal2_a,gw9_goal_cal2_b,gw9_goal_cal2_c"
      ;;
    gw9_context_calibrated3_ab_v1|gw9_context_calibrated3_ba_v1|gw9_context_calibrated3_aba_v1|gw9_context_calibrated3_bab_v1|gw9_context_calibrated3_ac_v1|gw9_context_calibrated3_ca_v1|gw9_context_calibrated3_bc_v1|gw9_context_calibrated3_cb_v1|gw9_context_calibrated3_abc_v1|gw9_context_calibrated3_bac_v1)
      printf '%s\n%s\n' "agent_context_calibrated_v3" "gw9_goal_cal3_a,gw9_goal_cal3_b,gw9_goal_cal3_c"
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

SEEDS_CSV="${SEEDS_CSV:-42,43,44}"

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
SCRATCH_EPISODES_PER_TASK="${SCRATCH_EPISODES_PER_TASK:-$EPISODES_PER_TASK}"
SCRATCH_MAX_STEPS_PER_EPISODE="${SCRATCH_MAX_STEPS_PER_EPISODE:-$MAX_STEPS_PER_EPISODE}"
SCRATCH_EVAL_EPISODES="${SCRATCH_EVAL_EPISODES:-$EVAL_EPISODES}"
SCRATCH_EVAL_EVERY_EPISODES="${SCRATCH_EVAL_EVERY_EPISODES:-$EVAL_EVERY_EPISODES}"
SCRATCH_EVAL_DENSE_EVERY_EPISODES="${SCRATCH_EVAL_DENSE_EVERY_EPISODES:-$EVAL_DENSE_EVERY_EPISODES}"
SCRATCH_EVAL_DENSE_WINDOW_EPISODES="${SCRATCH_EVAL_DENSE_WINDOW_EPISODES:-$EVAL_DENSE_WINDOW_EPISODES}"
SCRATCH_WARMUP_STEPS="${SCRATCH_WARMUP_STEPS:-$WARMUP_STEPS}"
SCRATCH_BATCH_SIZE="${SCRATCH_BATCH_SIZE:-$BATCH_SIZE}"
SCRATCH_EPS_DECAY_STEPS="${SCRATCH_EPS_DECAY_STEPS:-$EPS_DECAY_STEPS}"
HIDDEN_SIZES_CSV="${HIDDEN_SIZES_CSV:-128,128}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-10000}"
RECENT_BUFFER_CAPACITY="${RECENT_BUFFER_CAPACITY:-6000}"
ARCHIVE_BUFFER_CAPACITY="${ARCHIVE_BUFFER_CAPACITY:-4000}"
SCRATCH_HIDDEN_SIZES_CSV="${SCRATCH_HIDDEN_SIZES_CSV:-$HIDDEN_SIZES_CSV}"
SCRATCH_BUFFER_CAPACITY="${SCRATCH_BUFFER_CAPACITY:-$BUFFER_CAPACITY}"
SEGMENTED_KEEP_TAIL="${SEGMENTED_KEEP_TAIL:-512}"
SEGMENTED_RECENT_ONLY_STEPS="${SEGMENTED_RECENT_ONLY_STEPS:-1000}"
SEGMENTED_MIN_RECENT_SAMPLES="${SEGMENTED_MIN_RECENT_SAMPLES:-256}"
SEGMENTED_REVISIT_RECENT_MIX_START="${SEGMENTED_REVISIT_RECENT_MIX_START:-0.5}"
SEGMENTED_REVISIT_RECENT_MIX_END="${SEGMENTED_REVISIT_RECENT_MIX_END:-0.5}"
SEGMENTED_REVISIT_RECENT_ONLY_STEPS="${SEGMENTED_REVISIT_RECENT_ONLY_STEPS:-0}"
ARCHIVE_FRAC="${ARCHIVE_FRAC:-0.25}"
RECENT_MIX_START="${RECENT_MIX_START:-0.8}"
RECENT_MIX_END="${RECENT_MIX_END:-0.5}"
POST_SWITCH_STEPS="${POST_SWITCH_STEPS:-5000}"
EPS_RESET_VALUE="${EPS_RESET_VALUE:-0.4}"
EPS_DECAY_STEPS_AFTER_SWITCH="${EPS_DECAY_STEPS_AFTER_SWITCH:-2000}"
ALPHA_MAX_MULT="${ALPHA_MAX_MULT:-3.0}"
TD_K="${TD_K:-1.0}"
DISTILL_LAMBDA="${DISTILL_LAMBDA:-0.0}"
DISTILL_NEW_TASK_ONLY="${DISTILL_NEW_TASK_ONLY:-1}"
DER_ALPHA="${DER_ALPHA:-0.1}"
DER_BETA="${DER_BETA:-1.0}"
DER_CAPACITY="${DER_CAPACITY:-0}"
SCRATCH_REF_MIN_FINAL_SUCCESS="${SCRATCH_REF_MIN_FINAL_SUCCESS:-0.8}"
SCRATCH_REF_MIN_VALID_RUNS="${SCRATCH_REF_MIN_VALID_RUNS:-3}"
SCRATCH_REF_MIN_VALID_FRACTION="${SCRATCH_REF_MIN_VALID_FRACTION:-0.6}"
REUSE_SCRATCH_REFS_BY_TASK_SET="${REUSE_SCRATCH_REFS_BY_TASK_SET:-0}"
SCRATCH_GATE_ENABLED="${SCRATCH_GATE_ENABLED:-0}"
SCRATCH_GATE_REQUIRE_STABLE="${SCRATCH_GATE_REQUIRE_STABLE:-1}"
SCRATCH_GATE_MIN_PASSING_FRACTION="${SCRATCH_GATE_MIN_PASSING_FRACTION:-1.0}"
SCRATCH_GATE_MIN_FINAL_SUCCESS="${SCRATCH_GATE_MIN_FINAL_SUCCESS:-$SCRATCH_REF_MIN_FINAL_SUCCESS}"
METHOD_SET="${METHOD_SET:-main}"

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
        METHODS_CSV="ddqn_vanilla,oracle_reset,oracle_segmented,oracle_segmented_td,morphin_lite"
        ;;
      segmented_shortlist)
        METHODS_CSV="ddqn_vanilla,oracle_reset,oracle_segmented,morphin_lite"
        ;;
      segmented_plus_shortlist)
        METHODS_CSV="ddqn_vanilla,oracle_reset,oracle_segmented,oracle_segmented_td,oracle_segmented_td_plus"
        ;;
      segmented_td_ablation)
        METHODS_CSV="ddqn_vanilla,oracle_reset,oracle_segmented,oracle_segmented_td,morphin_lite"
        ;;
      segmented_revisit_ablation)
        METHODS_CSV="ddqn_vanilla,oracle_reset,oracle_segmented,oracle_segmented_revisit_aware,oracle_segmented_td,oracle_segmented_td_revisit_aware"
        ;;
      morphin_ablation)
        METHODS_CSV="ddqn_vanilla,oracle_reset,oracle_segmented,oracle_segmented_td,morphin_lite,morphin_full,morphin_segmented"
        ;;
      full)
        METHODS_CSV="ddqn_vanilla,oracle_reset,detector_reset_only,oracle_segmented,oracle_segmented_td,morphin_lite,morphin_full,morphin_segmented"
        ;;
      thesis_core)
        METHODS_CSV="ddqn_vanilla,oracle_reset,oracle_segmented"
        ;;
      thesis_ablation)
        METHODS_CSV="ddqn_vanilla,oracle_reset,oracle_segmented,oracle_segmented_td,morphin_lite"
        ;;
      thesis_distill)
        METHODS_CSV="oracle_segmented,oracle_segmented_distill"
        ;;
      thesis_lambda_sweep)
        METHODS_CSV="oracle_segmented,oracle_segmented_distill_l001,oracle_segmented_distill_l005,oracle_segmented_distill_l020,oracle_segmented_distill_l050,oracle_segmented_distill_l200"
        ;;
      thesis_full_256)
        METHODS_CSV="ddqn_vanilla,oracle_reset,oracle_segmented,oracle_segmented_distill_l001"
        ;;
      thesis_archive_sweep)
        METHODS_CSV="oracle_segmented,oracle_segmented_af015,oracle_segmented_af020,oracle_segmented_af025,oracle_segmented_af030"
        ;;
      thesis_der_validate)
        METHODS_CSV="ddqn_vanilla,oracle_segmented,oracle_segmented_distill_l001,der_plus_plus"
        ;;
      thesis_der_validate_oracle)
        METHODS_CSV="ddqn_vanilla,oracle_segmented,oracle_segmented_distill_l001,oracle_der_plus_plus"
        ;;
      thesis_detector)
        METHODS_CSV="ddqn_vanilla,oracle_segmented,morphin_detect,morphin_detect_seg"
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
log "Scratch stage: episodes/task=$SCRATCH_EPISODES_PER_TASK max_steps=$SCRATCH_MAX_STEPS_PER_EPISODE eval_every=$SCRATCH_EVAL_EVERY_EPISODES eval_episodes=$SCRATCH_EVAL_EPISODES warmup=$SCRATCH_WARMUP_STEPS batch=$SCRATCH_BATCH_SIZE eps_decay=$SCRATCH_EPS_DECAY_STEPS reuse_by_task_set=$REUSE_SCRATCH_REFS_BY_TASK_SET"
log "Success threshold: $SUCCESS_THRESHOLD | Consecutive evals for threshold: $THRESHOLD_MIN_CONSECUTIVE_EVALS"
log "Scratch refs: min_final_success=$SCRATCH_REF_MIN_FINAL_SUCCESS min_valid_runs=$SCRATCH_REF_MIN_VALID_RUNS min_valid_fraction=$SCRATCH_REF_MIN_VALID_FRACTION"
log "Scratch gate: enabled=$SCRATCH_GATE_ENABLED require_stable=$SCRATCH_GATE_REQUIRE_STABLE min_passing_fraction=$SCRATCH_GATE_MIN_PASSING_FRACTION min_final_success=$SCRATCH_GATE_MIN_FINAL_SUCCESS"
log "Switch epsilon reset: value=$EPS_RESET_VALUE decay_steps=$EPS_DECAY_STEPS_AFTER_SWITCH"
log "Replay: archive_frac=$ARCHIVE_FRAC recent_mix=$RECENT_MIX_START->$RECENT_MIX_END post_switch_steps=$POST_SWITCH_STEPS keep_tail=$SEGMENTED_KEEP_TAIL"
log "Distill: lambda=$DISTILL_LAMBDA new_task_only=$DISTILL_NEW_TASK_ONLY"
log "DER++: alpha=$DER_ALPHA beta=$DER_BETA der_capacity=${DER_CAPACITY:-0}"
log "Replay revisit-aware: revisit_recent_mix=$SEGMENTED_REVISIT_RECENT_MIX_START->$SEGMENTED_REVISIT_RECENT_MIX_END revisit_recent_only_steps=$SEGMENTED_REVISIT_RECENT_ONLY_STEPS"
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
    --segmented-min-recent-samples "$SEGMENTED_MIN_RECENT_SAMPLES" \
    --segmented-revisit-recent-mix-start "$SEGMENTED_REVISIT_RECENT_MIX_START" \
    --segmented-revisit-recent-mix-end "$SEGMENTED_REVISIT_RECENT_MIX_END" \
    --segmented-revisit-recent-only-steps "$SEGMENTED_REVISIT_RECENT_ONLY_STEPS" \
    --hidden-sizes-csv "$HIDDEN_SIZES_CSV" \
    --buffer-capacity "$BUFFER_CAPACITY" \
    --recent-buffer-capacity "$RECENT_BUFFER_CAPACITY" \
    --archive-buffer-capacity "$ARCHIVE_BUFFER_CAPACITY" \
    --distill-lambda "$DISTILL_LAMBDA" \
    --distill-new-task-only "$DISTILL_NEW_TASK_ONLY" \
    --der-alpha "$DER_ALPHA" \
    --der-beta "$DER_BETA" \
    --der-capacity "$DER_CAPACITY"
}

build_scratch_args() {
  local benchmark="$1"
  local obs_mode="$2"
  printf '%s\0' \
    --benchmark "$benchmark" \
    --episodes-per-task "$SCRATCH_EPISODES_PER_TASK" \
    --max-steps-per-episode "$SCRATCH_MAX_STEPS_PER_EPISODE" \
    --eval-episodes "$SCRATCH_EVAL_EPISODES" \
    --eval-every-episodes "$SCRATCH_EVAL_EVERY_EPISODES" \
    --eval-dense-every-episodes "$SCRATCH_EVAL_DENSE_EVERY_EPISODES" \
    --eval-dense-window-episodes "$SCRATCH_EVAL_DENSE_WINDOW_EPISODES" \
    --obs-mode "$obs_mode" \
    --warmup-steps "$SCRATCH_WARMUP_STEPS" \
    --batch-size "$SCRATCH_BATCH_SIZE" \
    --eps-decay-steps "$SCRATCH_EPS_DECAY_STEPS" \
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
    --segmented-min-recent-samples "$SEGMENTED_MIN_RECENT_SAMPLES" \
    --segmented-revisit-recent-mix-start "$SEGMENTED_REVISIT_RECENT_MIX_START" \
    --segmented-revisit-recent-mix-end "$SEGMENTED_REVISIT_RECENT_MIX_END" \
    --segmented-revisit-recent-only-steps "$SEGMENTED_REVISIT_RECENT_ONLY_STEPS" \
    --hidden-sizes-csv "$SCRATCH_HIDDEN_SIZES_CSV" \
    --buffer-capacity "$SCRATCH_BUFFER_CAPACITY"
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
  mapfile -d '' -t common_args < <(build_scratch_args "$benchmark" "$obs_mode")
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

validate_scratch_gate() {
  local refs_json="$1"
  local task_ids_csv="$2"
  local benchmark="$3"
  if [[ "$SCRATCH_GATE_ENABLED" != "1" || -z "$refs_json" ]]; then
    return 0
  fi
  log "Validating scratch refs for $benchmark before continual stage"
  local args=(
    -m rl.rl_uniandes.drl.morphin.validate_scratch_refs
    --refs-json "$refs_json"
    --task-ids-csv "$task_ids_csv"
    --min-final-success "$SCRATCH_GATE_MIN_FINAL_SUCCESS"
    --min-passing-fraction "$SCRATCH_GATE_MIN_PASSING_FRACTION"
  )
  if [[ "$SCRATCH_GATE_REQUIRE_STABLE" == "1" ]]; then
    args+=(--require-stable)
  fi
  py_run "${args[@]}"
}

SCRATCH_REFS_DIR="$SESSION_ROOT/scratch_refs"
mkdir -p "$SCRATCH_REFS_DIR"
SCRATCH_REFS_MAP_JSON="$SESSION_ROOT/scratch_refs_by_benchmark.json"
printf '{\n' >"$SCRATCH_REFS_MAP_JSON"
scratch_refs_first=1
declare -A SHARED_SCRATCH_REFS_JSON_BY_KEY
shared_scratch_idx=0

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
    if [[ "$REUSE_SCRATCH_REFS_BY_TASK_SET" == "1" ]]; then
      scratch_cache_key="${benchmark_obs_mode}|${benchmark_scratch_tasks_csv}|${SEEDS_CSV}|${SCRATCH_EPISODES_PER_TASK}|${SCRATCH_MAX_STEPS_PER_EPISODE}|${SCRATCH_EVAL_EPISODES}|${SCRATCH_EVAL_EVERY_EPISODES}|${SCRATCH_EVAL_DENSE_EVERY_EPISODES}|${SCRATCH_EVAL_DENSE_WINDOW_EPISODES}|${SCRATCH_WARMUP_STEPS}|${SCRATCH_BATCH_SIZE}|${SCRATCH_EPS_DECAY_STEPS}|${SUCCESS_THRESHOLD}|${THRESHOLD_MIN_CONSECUTIVE_EVALS}|${LEARNING_RATE}|${TAU}|${GAMMA}"
      if [[ -n "${SHARED_SCRATCH_REFS_JSON_BY_KEY[$scratch_cache_key]:-}" ]]; then
        benchmark_scratch_refs_json="${SHARED_SCRATCH_REFS_JSON_BY_KEY[$scratch_cache_key]}"
        log "Reusing shared scratch refs for $benchmark from $benchmark_scratch_refs_json"
      else
        shared_scratch_idx=$((shared_scratch_idx + 1))
        shared_scratch_root="$SESSION_ROOT/shared_scratch/cache_${shared_scratch_idx}"
        mkdir -p "$shared_scratch_root"
        log "Running shared scratch stage for $benchmark to build reusable scratch references"
        run_scratch_stage_for_benchmark "$benchmark" "$benchmark_obs_mode" "$benchmark_scratch_tasks_csv" "$shared_scratch_root"
        benchmark_scratch_refs_json="$SCRATCH_REFS_DIR/shared_cache_${shared_scratch_idx}.json"
        py_run -m rl.rl_uniandes.drl.morphin.build_scratch_refs \
          --root-dir "$shared_scratch_root" \
          --output-json "$benchmark_scratch_refs_json" \
          --min-final-success "$SCRATCH_REF_MIN_FINAL_SUCCESS" \
          --min-valid-runs "$SCRATCH_REF_MIN_VALID_RUNS" \
          --min-valid-fraction "$SCRATCH_REF_MIN_VALID_FRACTION"
        SHARED_SCRATCH_REFS_JSON_BY_KEY[$scratch_cache_key]="$benchmark_scratch_refs_json"
        log "Shared scratch refs built at $benchmark_scratch_refs_json"
      fi
    else
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
  fi

  if [[ -n "$benchmark_scratch_refs_json" ]]; then
    if [[ $scratch_refs_first -eq 0 ]]; then
      printf ',\n' >>"$SCRATCH_REFS_MAP_JSON"
    fi
    scratch_refs_first=0
    printf '  "%s": "%s"' "$benchmark" "$benchmark_scratch_refs_json" >>"$SCRATCH_REFS_MAP_JSON"
  fi

  validate_scratch_gate "$benchmark_scratch_refs_json" "$benchmark_scratch_tasks_csv" "$benchmark"

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
  "reuse_scratch_refs_by_task_set": $REUSE_SCRATCH_REFS_BY_TASK_SET,
  "scratch_gate_enabled": $SCRATCH_GATE_ENABLED,
  "scratch_gate_require_stable": $SCRATCH_GATE_REQUIRE_STABLE,
  "scratch_gate_min_passing_fraction": $SCRATCH_GATE_MIN_PASSING_FRACTION,
  "scratch_gate_min_final_success": $SCRATCH_GATE_MIN_FINAL_SUCCESS,
  "warmup_steps": $WARMUP_STEPS,
  "batch_size": $BATCH_SIZE,
  "eps_decay_steps": $EPS_DECAY_STEPS,
  "eps_reset_value": $EPS_RESET_VALUE,
  "eps_decay_steps_after_switch": $EPS_DECAY_STEPS_AFTER_SWITCH,
  "alpha_max_mult": $ALPHA_MAX_MULT,
  "td_k": $TD_K,
  "learning_rate": $LEARNING_RATE,
  "der_alpha": $DER_ALPHA,
  "der_beta": $DER_BETA,
  "der_capacity": $DER_CAPACITY,
  "archive_frac": $ARCHIVE_FRAC,
  "recent_mix_start": $RECENT_MIX_START,
  "recent_mix_end": $RECENT_MIX_END,
  "post_switch_steps": $POST_SWITCH_STEPS,
  "segmented_keep_tail": $SEGMENTED_KEEP_TAIL,
  "segmented_recent_only_steps": $SEGMENTED_RECENT_ONLY_STEPS,
  "segmented_min_recent_samples": $SEGMENTED_MIN_RECENT_SAMPLES,
  "segmented_revisit_recent_mix_start": $SEGMENTED_REVISIT_RECENT_MIX_START,
  "segmented_revisit_recent_mix_end": $SEGMENTED_REVISIT_RECENT_MIX_END,
  "segmented_revisit_recent_only_steps": $SEGMENTED_REVISIT_RECENT_ONLY_STEPS,
  "scratch_episodes_per_task": $SCRATCH_EPISODES_PER_TASK,
  "scratch_max_steps_per_episode": $SCRATCH_MAX_STEPS_PER_EPISODE,
  "scratch_eval_episodes": $SCRATCH_EVAL_EPISODES,
  "scratch_eval_every_episodes": $SCRATCH_EVAL_EVERY_EPISODES,
  "scratch_eval_dense_every_episodes": $SCRATCH_EVAL_DENSE_EVERY_EPISODES,
  "scratch_eval_dense_window_episodes": $SCRATCH_EVAL_DENSE_WINDOW_EPISODES,
  "scratch_warmup_steps": $SCRATCH_WARMUP_STEPS,
  "scratch_batch_size": $SCRATCH_BATCH_SIZE,
  "scratch_eps_decay_steps": $SCRATCH_EPS_DECAY_STEPS,
  "auto_build_scratch_refs": $AUTO_BUILD_SCRATCH_REFS,
  "scratch_refs_json": "${SCRATCH_REFS_JSON}",
  "scratch_refs_map_json": "${SCRATCH_REFS_MAP_JSON}"
}
JSON

log "Done. Session root: $SESSION_ROOT"
