# 🚀 LIBERO-plus zero shot Evaluation

This document provides instructions for reproducing our **zero shot experimental results** with LIBERO-plus.  
The evaluation process consists of two main parts:  

1. Setting up the `LIBERO-plus` environment and dependencies.  
2. Running the evaluation by launching services in both `LaWAM` and `LIBERO-plus` environments.  

We have verified that this workflow runs successfully on both **NVIDIA A100** and **RTX 4090** GPUs.  

---


## ⬇️ 0. Download Checkpoints

We use models trained exclusively on LIBERO to perform zero-shot evaluation on LIBERO-plus.: [🤗 StarVLA/bench-libero](https://huggingface.co/collections/StarVLA/bench-libero). Their corresponding results on LIBERO-plus are summarized in the table below.

### 📊 Experimental Results

| Model               | Camera | Robot | Language|Light|Background|Noise|Layout|Total|
|---------------------|-------|--------|---------|--------|------|-------|-------|-------|
| OpenVLA  | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT  | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π₀-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| ABot-M0  | 60.4 | 67.9 | 86.4 | 96.2 | 91.6 | 86.4 | 82.6 | 80.5 |
| **Qwen2.5-VL-FAST**   | 19.6 | 27.6 | 74.5 | 75.2 | 71.0 | 27.4 | 62.7 | 48.9 |
| **Qwen2.5-VL-GR00T**   | 32.9 | 50.8 | 86.3 | 96.2 | 85.7 | 62.0 | 73.6 | 66.4 |
| **Qwen2.5-VL-OFT**   | 34.4 | 63.7 | 82.1 | 86.9 | 88.8 | 53.3 | 74.6 | 67.2 |
| **Qwen3-VL-OFT**   | 47.0 | 60.1 | 87.0 | 96.3 | 95.3 | 73.1 | 79.2 | 75.0 |
| **Qwen3-VL-PI**   | 64.3 | 57.2 | 82.8 | 94.2 | 94.0 | 79.6 | 78.2 | 77.0 |



---


## 📦 1. Environment Setup

To set up the environment, please first follow the official [LIBERO-plus repository](https://github.com/sylvestf/LIBERO-plus) to install the base `LIBERO-plus` environment.  



Afterwards, inside the `LIBERO-plus` environment, install the following dependencies:  

```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

---

## 🚀 2. Evaluation Workflow

The evaluation should be run **from the repository root** using **two separate terminals**, one for each environment:  

- **LaWAM environment**: runs the inference server.  
- **LIBERO-plus environment**: runs the simulation.  

`examples/LIBERO-plus` is now a compatibility wrapper around the shared `examples/LIBERO` evaluator. The rollout loop, `ModelClient` usage, and output contract are shared; this wrapper only changes environment defaults and keeps the LIBERO-plus category aggregation outputs.

### Step 1. Start the server (LaWAM environment)

In the first terminal, activate the `LaWAM` environment and run:  

```bash
bash examples/LIBERO-plus/eval_files/run_policy_server.sh
```

⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `examples/LIBERO-plus/eval_files/run_policy_server.sh`  


---

### Step 2. Start the simulation (LIBERO-plus environment)

In the second terminal, activate the `LIBERO-plus` conda environment and run:  

```bash
bash examples/LIBERO-plus/eval_files/eval_libero.sh
```
⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `eval_libero.sh` to load action unnormalization stats. 

Also ensure the environment variables at the top of `eval_libero.sh` are correctly set.

Useful environment variables:

```bash
CKPT_PATH=/path/to/checkpoint.pt
SUITES="libero_goal libero_spatial libero_object libero_10"
NUM_TRIALS_PER_TASK=1
OUTPUT_ROOT=/mnt/project_rlinf/jlchen/code/LaWAM/results/eval_runs/libero_plus
BENCHMARK_VARIANT=libero_plus
ENABLE_CATEGORY_AGGREGATION=True
MAX_TASKS=20
```

Parallel `LIBERO-plus` now follows the same backend policy as the shared LIBERO evaluator: when you do not specify `MUJOCO_GL` yourself, multi-worker runs prefer `MUJOCO_GL=osmesa`, while single-worker runs keep `MUJOCO_GL=egl`.

Single-server evaluation results are saved under:

```txt
results/eval_runs/libero_plus/<ckpt_alias>/<run_tag>/
  run_meta.json
  category_logs/
    <suite>.json
    overall_results.json
  suites/<suite_name>/
    eval.log
    summary.json
    episodes.jsonl
```

Per-suite rollouts continue to write `summary.json` and `episodes.jsonl`, and the wrapper additionally writes category summaries plus `overall_results.json` under `category_logs/`.

### Step 3. Run the full LIBERO-plus benchmark

Use the auto-eval launcher to evaluate all configured suites for one checkpoint:

```bash
cd /mnt/project_rlinf/jlchen/code/LaWAM

CKPT_PATH=results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt \
bash examples/LIBERO-plus/eval_files/auto_eval_scripts/run_libero_plus_benchmark.sh
```

This launcher mirrors the LIBERO auto-eval flow: each suite gets its own policy server process, port, logs, and output directory, and the final category statistics are aggregated automatically.

For one-suite-at-a-time evaluation that uses all eight GPUs, with cross-task
batch size 16 on every GPU:

```bash
cd /mnt/project_rlinf/jlchen/code/LaWAM

CKPT_PATH=/mnt/project_rlinf/jlchen/code/LaWAM/results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt \
GPU_IDS="0 1 2 3 4 5 6 7" \
EVAL_GPU_IDS="0 1 2 3 4 5 6 7" \
NUM_WORKERS=16 \
SIM_CPU_THREADS=1 \
NUM_TRIALS_PER_TASK=1 \
SAVE_VIDEOS=False \
bash examples/LIBERO-plus/eval_files/auto_eval_scripts/run_libero_plus_8gpu_batch_benchmark.sh
```

The launcher evaluates the configured suites serially. For each suite, it
splits task ids into eight disjoint shards, runs one policy server per GPU, and
batches `NUM_WORKERS` heterogeneous tasks in every server request. This 8-GPU
launcher uses EGL and maps each rollout shard to its corresponding EGL device.
`SIM_CPU_THREADS` limits BLAS/OpenMP threads only in simulation workers and
defaults to `1`; policy-server CPU threading is unchanged.
Set `SUITES=libero_goal` to run only one suite.

Auto-eval uses the same run layout:

```txt
results/eval_runs/libero_plus/<ckpt_alias>/<run_tag>/
  run_meta.json
  category_logs/
  suites/<suite_name>/
    server.log
    eval.log
    summary.json
    episodes.jsonl
```


---

⚠️ **Note:** Since LIBERO-plus has 10,030 tasks, completing all the evaluations will take an extremely long time. It is recommended to run multiple model instances in parallel for the evaluations.
