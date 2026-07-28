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
legacy_max_concurrent_suites="${MAX_CONCURRENT_SUITES:-}"
num_trials_per_task="${NUM_TRIALS_PER_TASK:-1}"
num_workers="${NUM_WORKERS:-50}"
max_tasks="${MAX_TASKS:-}"
output_root="${OUTPUT_ROOT:-${REPO_ROOT}/results/eval_runs/libero_plus}"
benchmark_variant="${BENCHMARK_VARIANT:-libero_plus}"
gpu_ids="${GPU_IDS:-}"
eval_gpu_ids="${EVAL_GPU_IDS:-}"
worker_result_timeout_sec="${WORKER_RESULT_TIMEOUT_SEC:-600}"
save_videos="${SAVE_VIDEOS:-False}"
save_only_failure_videos="${SAVE_ONLY_FAILURE_VIDEOS:-False}"
json_save_videos="$(printf '%s' "${save_videos}" | tr '[:upper:]' '[:lower:]')"
json_save_only_failure_videos="$(printf '%s' "${save_only_failure_videos}" | tr '[:upper:]' '[:lower:]')"

if [ ! -f "${your_ckpt}" ]; then
  echo "[ERROR] Checkpoint not found: ${your_ckpt}" >&2
  exit 1
fi

IFS=' ,' read -r -a SUITE_ARRAY <<< "${suites}"
if [ "${#SUITE_ARRAY[@]}" -eq 0 ]; then
  echo "[ERROR] No task suites were provided. Set SUITES." >&2
  exit 1
fi
if [ -n "${legacy_max_concurrent_suites}" ] && [ "${legacy_max_concurrent_suites}" != "1" ]; then
  echo "[ERROR] MAX_CONCURRENT_SUITES has been removed. Suites always run serially; unset it or set it to 1." >&2
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
  "suite_execution_mode": "serial",
  "num_trials_per_task": ${num_trials_per_task},
  "num_workers": ${num_workers},
  "max_tasks": ${max_tasks:-null},
  "worker_result_timeout_sec": ${worker_result_timeout_sec},
  "save_videos": ${json_save_videos},
  "save_only_failure_videos": ${json_save_only_failure_videos},
  "gpu_ids": "${gpu_ids}",
  "eval_gpu_ids": "${eval_gpu_ids}"
}
EOF

echo "Launching LIBERO-plus benchmark"
echo "  checkpoint: ${your_ckpt}"
echo "  suites: ${suites}"
echo "  num_trials_per_task: ${num_trials_per_task}"
echo "  suite_execution_mode: serial"
echo "  per_suite_workers: ${num_workers}"
echo "  worker_result_timeout_sec: ${worker_result_timeout_sec}"
echo "  save_videos: ${save_videos}"
echo "  save_only_failure_videos: ${save_only_failure_videos}"
echo "  gpu_ids: ${gpu_ids:-auto}"
echo "  eval_gpu_ids: ${eval_gpu_ids:-same_as_gpu_ids}"
echo "  run_dir: ${run_dir}"

for idx in "${!SUITE_ARRAY[@]}"; do
  task_suite_name="${SUITE_ARRAY[$idx]}"
  [ -z "${task_suite_name}" ] && continue

  run_index=$((run_index_base + idx))
  suite_gpu_id="$(resolve_gpu_id "${run_index}" "${num_gpus}" "${gpu_ids}")"
  suite_eval_gpu_id="${suite_gpu_id}"
  if [ -n "${eval_gpu_ids}" ]; then
    suite_eval_gpu_id="$(resolve_gpu_id "${run_index}" "${num_gpus}" "${eval_gpu_ids}")"
  fi

  echo "Running suite serially"
  echo "  suite: ${task_suite_name}"
  echo "  run_index: ${run_index}"
  echo "  server_gpu_id: ${suite_gpu_id}"
  echo "  eval_gpu_id: ${suite_eval_gpu_id}"

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
  SAVE_VIDEOS="${save_videos}" \
  SAVE_ONLY_FAILURE_VIDEOS="${save_only_failure_videos}" \
  CATEGORY_LOG_ROOT="${category_log_dir}" \
  BENCHMARK_VARIANT="${benchmark_variant}" \
  bash "${SCRIPT_PATH}" "${your_ckpt}" "${task_suite_name}" "${run_index}"

  if [ "${idx}" -lt "$(( ${#SUITE_ARRAY[@]} - 1 ))" ] && [ "${launch_delay_sec}" -gt 0 ]; then
    sleep "${launch_delay_sec}"
  fi
done

echo "All LIBERO-plus suite jobs completed. Aggregating category results..."
LOG_DIR="${category_log_dir}" \
TASK_SUITES="$(IFS=,; echo "${SUITE_ARRAY[*]}")" \
PYTHONPATH="${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH:-}" \
"${LIBERO_PYTHON}" ./examples/LIBERO-plus/eval_files/aggregate_results.py

echo "LIBERO-plus benchmark completed successfully: ${run_dir}"
