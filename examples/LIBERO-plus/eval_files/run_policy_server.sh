#!/bin/bash
set -euo pipefail

export STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-/usr/local/miniconda3/bin/python}
export CKPT_PATH=${CKPT_PATH:-results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt}
export PORT=${PORT:-9883}
export SERVER_GPU_ID=${SERVER_GPU_ID:-0}

export CUDA_VISIBLE_DEVICES="${SERVER_GPU_ID}"
exec "${STAR_VLA_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT_PATH}" \
  --port "${PORT}" \
  --use_bf16
