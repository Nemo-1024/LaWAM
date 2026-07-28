#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

REPO_ROOT="$(libero_repo_root)"
cd "${REPO_ROOT}"

SCRIPT_PATH="${SCRIPT_DIR}/run_libero_plus_suite_benchmark.sh"
DEFAULT_CKPT="results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt"

your_ckpt="${1:-${CKPT_PATH:-${DEFAULT_CKPT}}}"
run_index_base="${2:-${RUN_INDEX_BASE:-0}}"
suites="${SUITES:-libero_goal libero_spatial libero_object libero_10}"
gpu_ids="${GPU_IDS:-0 1 2 3 4 5 6 7}"
eval_gpu_ids="${EVAL_GPU_IDS:-${gpu_ids}}"
shards_per_suite=8
num_trials_per_task="${NUM_TRIALS_PER_TASK:-1}"
num_workers="${NUM_WORKERS:-16}"
sim_cpu_threads="${SIM_CPU_THREADS:-1}"
max_tasks="${MAX_TASKS:-}"
launch_delay_sec="${LAUNCH_DELAY_SEC:-0.5}"
worker_result_timeout_sec="${WORKER_RESULT_TIMEOUT_SEC:-600}"
server_startup_timeout_sec="${SERVER_STARTUP_TIMEOUT_SEC:-600}"
save_videos="${SAVE_VIDEOS:-False}"
save_only_failure_videos="${SAVE_ONLY_FAILURE_VIDEOS:-False}"
output_root="${OUTPUT_ROOT:-${REPO_ROOT}/results/eval_runs/libero_plus}"

if [ ! -f "${your_ckpt}" ]; then
  echo "[ERROR] Checkpoint not found: ${your_ckpt}" >&2
  exit 1
fi
if [ "${num_trials_per_task}" != "1" ]; then
  echo "[ERROR] The 8-GPU LIBERO-plus launcher requires NUM_TRIALS_PER_TASK=1." >&2
  exit 1
fi
if ! [[ "${sim_cpu_threads}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] SIM_CPU_THREADS must be a positive integer, got: ${sim_cpu_threads}" >&2
  exit 1
fi
IFS=' ,' read -r -a SUITE_ARRAY <<< "${suites}"
IFS=' ,' read -r -a GPU_ARRAY <<< "${gpu_ids}"
IFS=' ,' read -r -a EVAL_GPU_ARRAY <<< "${eval_gpu_ids}"
job_count="${shards_per_suite}"
if [ "${#GPU_ARRAY[@]}" -ne 8 ] || [ "${#EVAL_GPU_ARRAY[@]}" -ne 8 ]; then
  echo "[ERROR] GPU_IDS and EVAL_GPU_IDS must each contain exactly 8 GPU ids." >&2
  exit 1
fi

export LIBERO_HOME="${LIBERO_HOME:-/mnt/project_rlinf/jlchen/code/LIBERO-plus}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_Python:-/mnt/project_rlinf/jlchen/envs/libero_plus/bin/python}}"
export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-${starVLA_python:-/usr/local/miniconda3/bin/python}}"
export PYTHONPATH="${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

ensure_libero_eval_config "${LIBERO_HOME}" "${LIBERO_CONFIG_PATH}"
ensure_libero_plus_assets "${LIBERO_HOME}"

task_classification="${LIBERO_HOME}/libero/libero/benchmark/task_classification.json"
if [ ! -f "${task_classification}" ]; then
  echo "[ERROR] Missing LIBERO-plus task metadata: ${task_classification}" >&2
  exit 1
fi

ckpt_alias="${LIBERO_CKPT_ALIAS:-$(derive_ckpt_alias "${your_ckpt}")}"
run_group="${LIBERO_RUN_GROUP:-${ckpt_alias}}"
run_tag="${RUN_TAG:-$(date +"%Y%m%d_%H%M%S")}"
run_dir="${output_root}/${run_group}/${run_tag}"
category_log_dir="${run_dir}/category_logs"
mkdir -p "${run_dir}/suites" "${category_log_dir}"

cat > "${run_dir}/run_meta.json" <<EOF
{
  "run_group": "${run_group}",
  "run_tag": "${run_tag}",
  "run_dir": "${run_dir}",
  "checkpoint_path": "${your_ckpt}",
  "checkpoint_alias": "${ckpt_alias}",
  "suites": "${suites}",
  "suite_execution_mode": "serial_suites_8gpu_task_sharded_batch",
  "shards_per_suite": ${shards_per_suite},
  "num_trials_per_task": ${num_trials_per_task},
  "num_workers_per_gpu": ${num_workers},
  "sim_cpu_threads_per_worker": ${sim_cpu_threads},
  "max_tasks_per_suite": ${max_tasks:-null},
  "gpu_ids": "${gpu_ids}",
  "eval_gpu_ids": "${eval_gpu_ids}",
  "mujoco_gl": "${MUJOCO_GL}",
  "pyopengl_platform": "${PYOPENGL_PLATFORM}",
  "save_videos": $(printf '%s' "${save_videos}" | tr '[:upper:]' '[:lower:]')
}
EOF

echo "Launching 8-GPU LIBERO-plus batch benchmark"
echo "  checkpoint: ${your_ckpt}"
echo "  suites: ${suites}"
echo "  shards_per_suite: ${shards_per_suite}"
echo "  batch_size_per_gpu: ${num_workers}"
echo "  sim_cpu_threads_per_worker: ${sim_cpu_threads}"
echo "  gpu_ids: ${gpu_ids}"
echo "  mujoco_gl: ${MUJOCO_GL}"
echo "  run_dir: ${run_dir}"

declare -a active_pids=()
declare -a all_result_names=()

cleanup_children() {
  local exit_code=$?
  local pid
  for pid in "${active_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  exit "${exit_code}"
}
trap cleanup_children INT TERM

for task_suite_name in "${SUITE_ARRAY[@]}"; do
  declare -a result_names=()
  declare -a launch_logs=()
  active_pids=()
  echo "Starting suite ${task_suite_name} on ${job_count} GPU(s)"

  for ((shard_idx = 0; shard_idx < shards_per_suite; shard_idx++)); do
    result_name="${task_suite_name}_shard_${shard_idx}_of_${shards_per_suite}"
    task_ids="$("${LIBERO_PYTHON}" - \
      "${task_classification}" "${task_suite_name}" "${shard_idx}" \
      "${shards_per_suite}" "${max_tasks}" <<'PY'
import json
import sys

metadata_path, suite_name, shard_idx_raw, shard_count_raw, max_tasks_raw = sys.argv[1:]
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)
task_count = len(metadata[suite_name])
if max_tasks_raw:
    task_count = min(task_count, int(max_tasks_raw))
shard_idx = int(shard_idx_raw)
shard_count = int(shard_count_raw)
print(",".join(str(task_id) for task_id in range(task_count) if task_id % shard_count == shard_idx))
PY
)"
    if [ -z "${task_ids}" ]; then
      echo "[ERROR] Empty task shard: ${result_name}" >&2
      exit 1
    fi

    run_index=$((run_index_base + shard_idx))
    gpu_id="${GPU_ARRAY[$shard_idx]}"
    eval_gpu_id="${EVAL_GPU_ARRAY[$shard_idx]}"
    launch_log="${run_dir}/suites/${result_name}/launcher.log"
    mkdir -p "$(dirname "${launch_log}")"

    echo "Launching shard ${result_name} on GPU ${gpu_id} with batch size ${num_workers}"
    LIBERO_CKPT_ALIAS="${ckpt_alias}" \
    LIBERO_RUN_GROUP="${run_group}" \
    RUN_TAG="${run_tag}" \
    OUTPUT_ROOT="${output_root}" \
    GPU_ID="${gpu_id}" \
    EVAL_GPU_ID="${eval_gpu_id}" \
    NUM_TRIALS_PER_TASK="${num_trials_per_task}" \
    NUM_WORKERS="${num_workers}" \
    SIM_CPU_THREADS="${sim_cpu_threads}" \
    TASK_IDS="${task_ids}" \
    CATEGORY_RESULT_NAME="${result_name}" \
    WORKER_RESULT_TIMEOUT_SEC="${worker_result_timeout_sec}" \
    SERVER_STARTUP_TIMEOUT_SEC="${server_startup_timeout_sec}" \
    SAVE_VIDEOS="${save_videos}" \
    SAVE_ONLY_FAILURE_VIDEOS="${save_only_failure_videos}" \
    CATEGORY_LOG_ROOT="${category_log_dir}" \
    BENCHMARK_VARIANT="libero_plus" \
    bash "${SCRIPT_PATH}" "${your_ckpt}" "${task_suite_name}" "${run_index}" \
      > >(tee "${launch_log}") 2>&1 &

    active_pids+=("$!")
    result_names+=("${result_name}")
    all_result_names+=("${result_name}")
    launch_logs+=("${launch_log}")
    if [ "${shard_idx}" -lt "$((job_count - 1))" ] && [ "${launch_delay_sec}" != "0" ]; then
      sleep "${launch_delay_sec}"
    fi
  done

  failed_jobs=0
  for idx in "${!active_pids[@]}"; do
    if wait "${active_pids[$idx]}"; then
      echo "[OK] Shard completed: ${result_names[$idx]}"
    else
      echo "[ERROR] Shard failed: ${result_names[$idx]}" >&2
      echo "  launcher_log: ${launch_logs[$idx]}" >&2
      failed_jobs=$((failed_jobs + 1))
    fi
  done

  if [ "${failed_jobs}" -ne 0 ]; then
    echo "[ERROR] ${failed_jobs} shard(s) failed. Check logs under ${run_dir}/suites." >&2
    exit 1
  fi
  active_pids=()
  echo "Suite ${task_suite_name} completed on all ${job_count} GPU(s)"
done

trap - INT TERM
echo "All suites completed. Aggregating category results..."
LOG_DIR="${category_log_dir}" \
TASK_SUITES="$(IFS=,; echo "${all_result_names[*]}")" \
"${LIBERO_PYTHON}" ./examples/LIBERO-plus/eval_files/aggregate_results.py

echo "8-GPU LIBERO-plus benchmark completed successfully: ${run_dir}"
