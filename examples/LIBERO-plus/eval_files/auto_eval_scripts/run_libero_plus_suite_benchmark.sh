#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

REPO_ROOT="$(libero_repo_root)"
cd "${REPO_ROOT}"

your_ckpt="${1:-${CKPT_PATH:-}}"
task_suite_name="${2:-${TASK_SUITE_NAME:-}}"
run_index="${3:-${RUN_INDEX:-}}"

if [ -z "${your_ckpt}" ] || [ -z "${task_suite_name}" ] || [ -z "${run_index}" ]; then
  echo "Usage: $0 <ckpt_path> <task_suite_name> <run_index>" >&2
  exit 1
fi
if [ ! -f "${your_ckpt}" ]; then
  echo "[ERROR] Checkpoint not found: ${your_ckpt}" >&2
  exit 1
fi

export LIBERO_HOME="${LIBERO_HOME:-/mnt/project_rlinf/jlchen/code/LIBERO-plus}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_Python:-/mnt/project_rlinf/jlchen/envs/libero_plus/bin/python}}"
export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-${starVLA_python:-/usr/local/miniconda3/bin/python}}"
export PYTHONPATH="${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH:-}"

if [ ! -x "${LIBERO_PYTHON}" ]; then
  echo "[ERROR] LIBERO_PYTHON is not executable: ${LIBERO_PYTHON}" >&2
  exit 1
fi
if [ ! -x "${STAR_VLA_PYTHON}" ]; then
  echo "[ERROR] STAR_VLA_PYTHON is not executable: ${STAR_VLA_PYTHON}" >&2
  exit 1
fi
if [ ! -d "${LIBERO_HOME}" ]; then
  echo "[ERROR] LIBERO_HOME does not exist: ${LIBERO_HOME}" >&2
  exit 1
fi

ensure_libero_eval_config "${LIBERO_HOME}" "${LIBERO_CONFIG_PATH}"
ensure_libero_plus_assets "${LIBERO_HOME}"

num_gpus="$(detect_num_gpus "${STAR_VLA_PYTHON}")"
if ! [[ "${num_gpus}" =~ ^[0-9]+$ ]] || [ "${num_gpus}" -lt 1 ]; then
  echo "[ERROR] NUM_GPUS must be a positive integer, got: ${num_gpus}" >&2
  exit 1
fi

gpu_id="${GPU_ID:-}"
if [ -z "${gpu_id}" ]; then
  gpu_id="$(resolve_gpu_id "${run_index}" "${num_gpus}" "${GPU_IDS:-}")"
fi
eval_gpu_id="${EVAL_GPU_ID:-}"
if [ -z "${eval_gpu_id}" ] && [ -n "${EVAL_GPU_IDS:-}" ]; then
  eval_gpu_id="$(resolve_gpu_id "${run_index}" "${num_gpus}" "${EVAL_GPU_IDS}")"
fi
if [ -z "${eval_gpu_id}" ]; then
  eval_gpu_id="${gpu_id}"
fi
num_trials_per_task="${NUM_TRIALS_PER_TASK:-1}"
num_workers="${NUM_WORKERS:-1}"
sim_cpu_threads="${SIM_CPU_THREADS:-1}"
if ! [[ "${sim_cpu_threads}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] SIM_CPU_THREADS must be a positive integer, got: ${sim_cpu_threads}" >&2
  exit 1
fi
if [ -z "${MUJOCO_GL:-}" ]; then
  if [ "${num_workers}" -gt 1 ]; then
    export MUJOCO_GL="osmesa"
  else
    export MUJOCO_GL="egl"
  fi
fi
if [ -z "${PYOPENGL_PLATFORM:-}" ] && { [ "${MUJOCO_GL}" = "egl" ] || [ "${MUJOCO_GL}" = "osmesa" ]; }; then
  export PYOPENGL_PLATFORM="${MUJOCO_GL}"
fi
max_tasks="${MAX_TASKS:-}"
task_ids="${TASK_IDS:-}"
category_result_name="${CATEGORY_RESULT_NAME:-${task_suite_name}}"
result_name="$(sanitize_component "${category_result_name}")"
if [ -z "${result_name}" ]; then
  echo "[ERROR] CATEGORY_RESULT_NAME must contain at least one valid filename character." >&2
  exit 1
fi
save_videos="${SAVE_VIDEOS:-False}"
save_only_failure_videos="${SAVE_ONLY_FAILURE_VIDEOS:-False}"
host="${HOST:-127.0.0.1}"
port_base="${PORT_BASE:-9883}"
preferred_port=$((port_base + run_index))
port_search_limit="${PORT_SEARCH_LIMIT:-200}"
reserve_port "${STAR_VLA_PYTHON}" "${preferred_port}" "${port_search_limit}"
base_port="${RESERVED_PORT}"
server_startup_timeout_sec="${SERVER_STARTUP_TIMEOUT_SEC:-180}"
benchmark_variant="${BENCHMARK_VARIANT:-libero_plus}"
enable_category_aggregation="${ENABLE_CATEGORY_AGGREGATION:-True}"
unnorm_key="${UNNORM_KEY:-}"
worker_result_timeout_sec="${WORKER_RESULT_TIMEOUT_SEC:-600}"
output_root="${OUTPUT_ROOT:-${REPO_ROOT}/results/eval_runs/libero_plus}"
run_group="${LIBERO_RUN_GROUP:-${LIBERO_CKPT_ALIAS:-$(derive_ckpt_alias "${your_ckpt}")}}"
run_tag="${RUN_TAG:-$(date +"%Y%m%d_%H%M%S")}"
run_root="${output_root}/${run_group}/${run_tag}"
category_log_dir="${CATEGORY_LOG_ROOT:-${run_root}/category_logs}"
suite_dir="${run_root}/suites/${result_name}"

mkdir -p "${suite_dir}" "${category_log_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MUJOCO_EGL_DEVICE_ID="${gpu_id}"

server_log="${suite_dir}/server.log"
eval_log="${suite_dir}/eval.log"
server_pid=""

eval_cmd=(
  "${LIBERO_PYTHON}" ./examples/LIBERO-plus/eval_files/eval_libero.py
  --args.pretrained-path "${your_ckpt}"
  --args.host "${host}"
  --args.port "${base_port}"
  --args.task-suite-name "${task_suite_name}"
  --args.num-trials-per-task "${num_trials_per_task}"
  --args.num-workers "${num_workers}"
  --args.video-out-path "${suite_dir}"
  --args.log-path "${category_log_dir}"
  --args.category-result-name "${result_name}"
  --args.benchmark-variant "${benchmark_variant}"
  --args.enable-category-aggregation "${enable_category_aggregation}"
  --args.worker-result-timeout-sec "${worker_result_timeout_sec}"
)

if [ "${save_videos}" = "True" ]; then
  eval_cmd+=(--args.save-videos)
else
  eval_cmd+=(--args.no-save-videos)
fi
if [ "${save_only_failure_videos}" = "True" ]; then
  eval_cmd+=(--args.save-only-failure-videos)
else
  eval_cmd+=(--args.no-save-only-failure-videos)
fi

if [ -n "${unnorm_key}" ]; then
  eval_cmd+=(--args.unnorm-key "${unnorm_key}")
fi
if [ -n "${max_tasks}" ]; then
  eval_cmd+=(--args.max-tasks "${max_tasks}")
fi
if [ -n "${task_ids}" ]; then
  eval_cmd+=(--args.task-ids "${task_ids}")
fi

cleanup() {
  local exit_code=$?
  if [ -n "${server_pid}" ] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  release_reserved_port
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

echo "Starting LIBERO-plus suite benchmark"
echo "  checkpoint: ${your_ckpt}"
echo "  task_suite: ${task_suite_name}"
echo "  result_name: ${result_name}"
echo "  task_ids: ${task_ids:-all}"
echo "  run_index: ${run_index}"
echo "  server_gpu_id: ${gpu_id}"
echo "  eval_gpu_id: ${eval_gpu_id}"
echo "  num_workers: ${num_workers}"
echo "  sim_cpu_threads_per_worker: ${sim_cpu_threads}"
echo "  mujoco_gl: ${MUJOCO_GL}"
echo "  save_videos: ${save_videos}"
echo "  save_only_failure_videos: ${save_only_failure_videos}"
echo "  worker_result_timeout_sec: ${worker_result_timeout_sec}"
echo "  port: ${base_port}"
if [ "${base_port}" != "${preferred_port}" ]; then
  echo "  preferred_port: ${preferred_port} (occupied, auto-switched)"
fi
echo "  output: ${suite_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${STAR_VLA_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${your_ckpt}" \
  --port "${base_port}" \
  --use_bf16 \
  > "${server_log}" 2>&1 &
server_pid=$!

if ! wait_for_port "${STAR_VLA_PYTHON}" "${host}" "${base_port}" "${server_startup_timeout_sec}"; then
  echo "[ERROR] Policy server failed to become ready on ${host}:${base_port}" >&2
  tail -n 40 "${server_log}" >&2 || true
  exit 1
fi

set +e
OMP_NUM_THREADS="${sim_cpu_threads}" \
MKL_NUM_THREADS="${sim_cpu_threads}" \
OPENBLAS_NUM_THREADS="${sim_cpu_threads}" \
NUMEXPR_NUM_THREADS="${sim_cpu_threads}" \
VECLIB_MAXIMUM_THREADS="${sim_cpu_threads}" \
BLIS_NUM_THREADS="${sim_cpu_threads}" \
TOKENIZERS_PARALLELISM=false \
PYTHONFAULTHANDLER=1 \
CUDA_VISIBLE_DEVICES="${eval_gpu_id}" MUJOCO_EGL_DEVICE_ID="${eval_gpu_id}" \
  "${eval_cmd[@]}" 2>&1 | tee "${eval_log}"
eval_status=${PIPESTATUS[0]}
set -e

if [ "${eval_status}" -ne 0 ]; then
  echo "[ERROR] Evaluation command failed with exit code ${eval_status}" >&2
  tail -n 40 "${server_log}" >&2 || true
  exit "${eval_status}"
fi

echo "LIBERO-plus suite benchmark completed: ${suite_dir}"
