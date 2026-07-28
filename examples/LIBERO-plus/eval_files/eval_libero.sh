#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=${LIBERO_HOME:-/mnt/project_rlinf/jlchen/code/LIBERO-plus}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}
export LIBERO_PYTHON=${LIBERO_PYTHON:-${LIBERO_Python:-/mnt/project_rlinf/jlchen/envs/libero_plus/bin/python}}
export PYTHONPATH="${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH:-}"
num_workers=${NUM_WORKERS:-1}
if [ -z "${MUJOCO_GL:-}" ]; then
  if [ "${num_workers}" -gt 1 ]; then
    export MUJOCO_GL=osmesa
  else
    export MUJOCO_GL=egl
  fi
fi
if [ -z "${PYOPENGL_PLATFORM:-}" ] && { [ "${MUJOCO_GL}" = "egl" ] || [ "${MUJOCO_GL}" = "osmesa" ]; }; then
  export PYOPENGL_PLATFORM="${MUJOCO_GL}"
fi
eval_gpu_id=${EVAL_GPU_ID:-${CUDA_VISIBLE_DEVICES%%,*}}
if [ -z "${eval_gpu_id}" ]; then
  eval_gpu_id=0
fi

if [ -z "${MUJOCO_EGL_DEVICE_ID:-}" ]; then
  export MUJOCO_EGL_DEVICE_ID="${eval_gpu_id}"
fi


host="127.0.0.1"
base_port=${PORT:-9883}
your_ckpt=${CKPT_PATH:-${REPO_ROOT}/results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt}
output_root=${OUTPUT_ROOT:-${REPO_ROOT}/results/eval_runs/libero_plus}
num_trials_per_task=${NUM_TRIALS_PER_TASK:-1}
benchmark_variant=${BENCHMARK_VARIANT:-libero_plus}
enable_category_aggregation=${ENABLE_CATEGORY_AGGREGATION:-True}
max_tasks=${MAX_TASKS:-}
suites=${SUITES:-${TASK_SUITE_NAME:-"libero_goal libero_spatial libero_object libero_10"}}
# === End of environment variable configuration ===
###########################################################################################

sanitize_component() {
  local value="$1"
  value="${value//+/_}"
  value="${value// /_}"
  printf '%s' "${value}" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^[-_.]+//; s/[-_.]+$//'
}

derive_ckpt_alias() {
  local ckpt_path="$1"
  local parent_dir grandparent_dir
  parent_dir="$(basename "$(dirname "${ckpt_path}")")"
  grandparent_dir="$(basename "$(dirname "$(dirname "${ckpt_path}")")")"

  if [[ "${grandparent_dir}" != "final_model" && -n "${grandparent_dir}" ]]; then
    sanitize_component "${grandparent_dir}"
    return
  fi

  if [[ -n "${parent_dir}" ]]; then
    sanitize_component "${parent_dir}"
    return
  fi

  sanitize_component "$(basename "${ckpt_path%.*}")"
}

# Avoid interactive prompt in libero.libero.__init__ when config file is missing.
if [ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]; then
  mkdir -p "${LIBERO_CONFIG_PATH}"
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${LIBERO_HOME}/libero/libero
bddl_files: ${LIBERO_HOME}/libero/libero/./bddl_files
init_states: ${LIBERO_HOME}/libero/libero/./init_files
datasets: ${LIBERO_HOME}/libero/libero/../datasets
assets: ${LIBERO_HOME}/libero/libero/./assets
EOF
fi

clear_distributed_env() {
  unset WORLD_SIZE
  unset RANK
  unset LOCAL_RANK
  unset LOCAL_WORLD_SIZE
  unset NODE_RANK
  unset MASTER_ADDR
  unset MASTER_PORT
}

# Enforce strict LIBERO-plus asset isolation: do not fall back to LIBERO assets.
if [ ! -d "${LIBERO_HOME}/libero/libero/assets/scenes" ]; then
  echo "[ERROR] Missing LIBERO-plus assets at: ${LIBERO_HOME}/libero/libero/assets"
  echo "[ERROR] Please download assets.zip from https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main"
  echo "[ERROR] and unzip it into ${LIBERO_HOME}/libero/libero/ so that assets/scenes exists."
  exit 1
fi

IFS=' ,' read -r -a SUITE_ARRAY <<< "${suites}"
pids=()

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
  "benchmark_variant": "${benchmark_variant}",
  "mode": "single_server_eval"
}
EOF

for task_suite_name in "${SUITE_ARRAY[@]}"; do
  [ -z "${task_suite_name}" ] && continue

  suite_dir="${run_dir}/suites/${task_suite_name}"
  log_file="${suite_dir}/eval.log"
  mkdir -p "${suite_dir}"

  cmd=(
    "${LIBERO_PYTHON}" ./examples/LIBERO-plus/eval_files/eval_libero.py
    --args.pretrained-path "${your_ckpt}"
    --args.host "${host}"
    --args.port "${base_port}"
    --args.task-suite-name "${task_suite_name}"
    --args.num-trials-per-task "${num_trials_per_task}"
    --args.num-workers "${num_workers}"
    --args.video-out-path "${suite_dir}"
    --args.log-path "${category_log_dir}"
    --args.benchmark-variant "${benchmark_variant}"
    --args.enable-category-aggregation "${enable_category_aggregation}"
  )
  if [ -n "${max_tasks}" ]; then
    cmd+=(--args.max-tasks "${max_tasks}")
  fi

  clear_distributed_env
  CUDA_VISIBLE_DEVICES="${eval_gpu_id}" MUJOCO_EGL_DEVICE_ID="${eval_gpu_id}" \
    "${cmd[@]}" 2>&1 | tee "${log_file}" &
  pids+=($!)
done

echo "Waiting for all evaluation tasks to finish..."
eval_status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    eval_status=1
  fi
done

echo "All tasks completed. Aggregating results..."
if [ ${eval_status} -ne 0 ]; then
  echo "[ERROR] At least one suite evaluation failed. Skip aggregation."
  exit 1
fi
export LOG_DIR="${category_log_dir}"
export TASK_SUITES="$(IFS=,; echo "${SUITE_ARRAY[*]}")"
${LIBERO_PYTHON} ./examples/LIBERO-plus/eval_files/aggregate_results.py
agg_status=$?
if [ ${agg_status} -ne 0 ]; then
  exit 1
fi
