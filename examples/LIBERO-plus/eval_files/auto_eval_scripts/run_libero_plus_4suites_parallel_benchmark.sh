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
launch_delay_sec="${LAUNCH_DELAY_SEC:-0.5}"
num_trials_per_task="${NUM_TRIALS_PER_TASK:-1}"
num_workers="${NUM_WORKERS:-30}"
max_tasks="${MAX_TASKS:-}"
output_root="${OUTPUT_ROOT:-${REPO_ROOT}/results/eval_runs/libero_plus}"
benchmark_variant="${BENCHMARK_VARIANT:-libero_plus}"
gpu_ids="${GPU_IDS:-}"
eval_gpu_ids="${EVAL_GPU_IDS:-}"
worker_result_timeout_sec="${WORKER_RESULT_TIMEOUT_SEC:-600}"
server_startup_timeout_sec="${SERVER_STARTUP_TIMEOUT_SEC:-180}"
save_videos="${SAVE_VIDEOS:-False}"
save_only_failure_videos="${SAVE_ONLY_FAILURE_VIDEOS:-False}"
json_save_videos="$(printf '%s' "${save_videos}" | tr '[:upper:]' '[:lower:]')"
json_save_only_failure_videos="$(printf '%s' "${save_only_failure_videos}" | tr '[:upper:]' '[:lower:]')"

if [ ! -f "${your_ckpt}" ]; then
  echo "[ERROR] Checkpoint not found: ${your_ckpt}" >&2
  exit 1
fi
if [ ! -f "${SCRIPT_PATH}" ]; then
  echo "[ERROR] Suite benchmark launcher not found: ${SCRIPT_PATH}" >&2
  exit 1
fi

IFS=' ,' read -r -a SUITE_ARRAY <<< "${suites}"
if [ "${#SUITE_ARRAY[@]}" -eq 0 ]; then
  echo "[ERROR] No task suites were provided. Set SUITES." >&2
  exit 1
fi

export LIBERO_HOME="${LIBERO_HOME:-/mnt/project_rlinf/jlchen/code/LIBERO-plus}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_Python:-/mnt/project_rlinf/jlchen/envs/libero_plus/bin/python}}"
export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-${starVLA_python:-/usr/local/miniconda3/bin/python}}"
ensure_libero_plus_assets "${LIBERO_HOME}"
num_gpus="$(detect_num_gpus "${STAR_VLA_PYTHON}")"
if ! [[ "${num_gpus}" =~ ^[0-9]+$ ]] || [ "${num_gpus}" -lt 1 ]; then
  echo "[ERROR] NUM_GPUS must be a positive integer, got: ${num_gpus}" >&2
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
  "run_index_base": ${run_index_base},
  "benchmark_variant": "${benchmark_variant}",
  "suite_execution_mode": "parallel",
  "suite_parallelism": ${#SUITE_ARRAY[@]},
  "num_trials_per_task": ${num_trials_per_task},
  "num_workers": ${num_workers},
  "max_tasks": ${max_tasks:-null},
  "worker_result_timeout_sec": ${worker_result_timeout_sec},
  "server_startup_timeout_sec": ${server_startup_timeout_sec},
  "save_videos": ${json_save_videos},
  "save_only_failure_videos": ${json_save_only_failure_videos},
  "gpu_ids": "${gpu_ids}",
  "eval_gpu_ids": "${eval_gpu_ids}"
}
EOF

echo "Launching parallel LIBERO-plus suite benchmark"
echo "  checkpoint: ${your_ckpt}"
echo "  suites: ${suites}"
echo "  run_index_base: ${run_index_base}"
echo "  num_trials_per_task: ${num_trials_per_task}"
echo "  suite_execution_mode: parallel"
echo "  suite_parallelism: ${#SUITE_ARRAY[@]}"
echo "  per_suite_workers: ${num_workers}"
echo "  worker_result_timeout_sec: ${worker_result_timeout_sec}"
echo "  server_startup_timeout_sec: ${server_startup_timeout_sec}"
echo "  save_videos: ${save_videos}"
echo "  save_only_failure_videos: ${save_only_failure_videos}"
echo "  gpu_ids: ${gpu_ids:-auto}"
echo "  eval_gpu_ids: ${eval_gpu_ids:-same_as_gpu_ids}"
echo "  run_dir: ${run_dir}"

if [ "${save_only_failure_videos}" = "True" ] && [ "${save_videos}" != "True" ]; then
  echo "[WARN] SAVE_ONLY_FAILURE_VIDEOS=True is ignored because SAVE_VIDEOS=False." >&2
  echo "[WARN] Set SAVE_VIDEOS=True SAVE_ONLY_FAILURE_VIDEOS=True to save only failed episodes." >&2
fi
if [ -z "${gpu_ids}" ] && [ "${num_gpus}" -lt "${#SUITE_ARRAY[@]}" ]; then
  echo "[WARN] Detected ${num_gpus} GPU(s) for ${#SUITE_ARRAY[@]} parallel suites; GPU assignment will wrap around." >&2
  echo "[WARN] Set GPU_IDS to control placement if this is not intended." >&2
fi

declare -a pids=()
declare -a launched_suites=()
declare -a launch_logs=()
declare -a run_indices=()
declare -a server_gpu_ids=()
declare -a eval_suite_gpu_ids=()

cleanup_children() {
  local exit_code=$?
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  exit "${exit_code}"
}
trap cleanup_children INT TERM

for idx in "${!SUITE_ARRAY[@]}"; do
  task_suite_name="${SUITE_ARRAY[$idx]}"
  [ -z "${task_suite_name}" ] && continue

  run_index=$((run_index_base + idx))
  suite_gpu_id="$(resolve_gpu_id "${run_index}" "${num_gpus}" "${gpu_ids}")"
  suite_eval_gpu_id="${suite_gpu_id}"
  if [ -n "${eval_gpu_ids}" ]; then
    suite_eval_gpu_id="$(resolve_gpu_id "${run_index}" "${num_gpus}" "${eval_gpu_ids}")"
  fi

  suite_launch_log="${run_dir}/suites/${task_suite_name}/launcher.log"
  mkdir -p "$(dirname "${suite_launch_log}")"

  echo "Launching suite in parallel"
  echo "  suite: ${task_suite_name}"
  echo "  run_index: ${run_index}"
  echo "  server_gpu_id: ${suite_gpu_id}"
  echo "  eval_gpu_id: ${suite_eval_gpu_id}"
  echo "  launcher_log: ${suite_launch_log}"

  LIBERO_CKPT_ALIAS="${ckpt_alias}" \
  LIBERO_RUN_GROUP="${run_group}" \
  RUN_TAG="${run_tag}" \
  OUTPUT_ROOT="${output_root}" \
  GPU_ID="${suite_gpu_id}" \
  EVAL_GPU_ID="${suite_eval_gpu_id}" \
  NUM_TRIALS_PER_TASK="${num_trials_per_task}" \
  NUM_WORKERS="${num_workers}" \
  MAX_TASKS="${max_tasks}" \
  WORKER_RESULT_TIMEOUT_SEC="${worker_result_timeout_sec}" \
  SERVER_STARTUP_TIMEOUT_SEC="${server_startup_timeout_sec}" \
  SAVE_VIDEOS="${save_videos}" \
  SAVE_ONLY_FAILURE_VIDEOS="${save_only_failure_videos}" \
  CATEGORY_LOG_ROOT="${category_log_dir}" \
  BENCHMARK_VARIANT="${benchmark_variant}" \
  bash "${SCRIPT_PATH}" "${your_ckpt}" "${task_suite_name}" "${run_index}" \
    > >(tee "${suite_launch_log}") 2>&1 &

  pids+=("$!")
  launched_suites+=("${task_suite_name}")
  launch_logs+=("${suite_launch_log}")
  run_indices+=("${run_index}")
  server_gpu_ids+=("${suite_gpu_id}")
  eval_suite_gpu_ids+=("${suite_eval_gpu_id}")

  if [ "${idx}" -lt "$(( ${#SUITE_ARRAY[@]} - 1 ))" ] && [ "${launch_delay_sec}" != "0" ]; then
    sleep "${launch_delay_sec}"
  fi
done

failed_suites=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  task_suite_name="${launched_suites[$idx]}"
  suite_launch_log="${launch_logs[$idx]}"
  run_index="${run_indices[$idx]}"
  suite_gpu_id="${server_gpu_ids[$idx]}"
  suite_eval_gpu_id="${eval_suite_gpu_ids[$idx]}"

  if wait "${pid}"; then
    echo "[OK] Suite completed: ${task_suite_name}"
  else
    echo "[ERROR] Suite failed: ${task_suite_name}" >&2
    echo "  run_index: ${run_index}" >&2
    echo "  server_gpu_id: ${suite_gpu_id}" >&2
    echo "  eval_gpu_id: ${suite_eval_gpu_id}" >&2
    echo "  launcher_log: ${suite_launch_log}" >&2
    failed_suites=$((failed_suites + 1))
  fi
done

trap - INT TERM

if [ "${failed_suites}" -ne 0 ]; then
  echo "[ERROR] ${failed_suites} suite(s) failed. Check logs under: ${run_dir}/suites" >&2
  exit 1
fi

echo "All LIBERO-plus suite jobs completed. Aggregating category results..."
LOG_DIR="${category_log_dir}" \
TASK_SUITES="$(IFS=,; echo "${SUITE_ARRAY[*]}")" \
PYTHONPATH="${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH:-}" \
"${LIBERO_PYTHON}" ./examples/LIBERO-plus/eval_files/aggregate_results.py

echo "Parallel LIBERO-plus suite benchmark completed successfully: ${run_dir}"
