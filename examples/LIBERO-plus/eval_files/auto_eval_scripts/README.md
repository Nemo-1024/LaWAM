# LIBERO-plus Benchmark Launch Scripts

This directory mirrors the new LIBERO benchmark launcher structure:

- `run_libero_plus_suite_benchmark.sh`
  Starts one StarVLA policy server and evaluates one LIBERO-plus suite against it.
- `run_libero_plus_benchmark.sh`
  Orchestrates multiple suites for one checkpoint and aggregates category results.
- `run_libero_plus_4suites_parallel_benchmark.sh`
  Orchestrates the selected LIBERO-plus suites in parallel for one checkpoint.
- `run_libero_plus_8gpu_batch_benchmark.sh`
  Runs suites serially and shards each suite across all configured GPUs; each
  GPU still batches different tasks with `NUM_WORKERS`.
- `common.sh`
  Reuses shared LIBERO helper functions and adds LIBERO-plus asset checks.

Removed legacy scripts:

- `auto_eval_libero_plus.sh`
- `eval_libero_plus_parallel.sh`
- `readme.md`

Typical usage:

```bash
CKPT_PATH=/path/to/checkpoint.pt \
bash examples/LIBERO-plus/eval_files/auto_eval_scripts/run_libero_plus_benchmark.sh
```

Behavior notes:

- `run_libero_plus_benchmark.sh` runs different suites serially. `MAX_CONCURRENT_SUITES` is no longer a supported option there.
- Use `run_libero_plus_4suites_parallel_benchmark.sh` when you want suites from `SUITES` to run at the same time.
- Parallelism inside one suite is controlled by `NUM_WORKERS`.
- Rollout videos are disabled by default. Use `SAVE_VIDEOS=True` to enable video writing.
- Use `SAVE_ONLY_FAILURE_VIDEOS=True` to save only failed episodes.
- Use `GPU_IDS` to choose which GPU(s) the policy server rotates across per suite.
- Use `EVAL_GPU_IDS` if the eval workers should use different GPU ids than the policy server.

Example with serial suites and internal suite workers:

```bash
CKPT_PATH=/path/to/checkpoint.pt \
SUITES="libero_goal libero_spatial libero_object libero_10" \
NUM_WORKERS=8 \
SAVE_VIDEOS=False \
GPU_IDS="3" \
EVAL_GPU_IDS="3" \
bash examples/LIBERO-plus/eval_files/auto_eval_scripts/run_libero_plus_benchmark.sh
```

Single-suite benchmark:

```bash
CKPT_PATH=/path/to/checkpoint.pt \
TASK_SUITE_NAME=libero_goal \
NUM_WORKERS=8 \
SAVE_VIDEOS=False \
GPU_ID=3 \
EVAL_GPU_ID=3 \
bash examples/LIBERO-plus/eval_files/auto_eval_scripts/run_libero_plus_suite_benchmark.sh \
  "$CKPT_PATH" "$TASK_SUITE_NAME" 0
```

Parallel suite benchmark:

```bash
CKPT_PATH=/path/to/checkpoint.pt \
SUITES="libero_goal libero_spatial libero_object libero_10" \
NUM_WORKERS=8 \
GPU_IDS="0 1 2 3" \
EVAL_GPU_IDS="0 1 2 3" \
bash examples/LIBERO-plus/eval_files/auto_eval_scripts/run_libero_plus_4suites_parallel_benchmark.sh
```

Eight-GPU cross-task batch benchmark:

```bash
CKPT_PATH=/mnt/project_rlinf/jlchen/code/LaWAM/results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt \
GPU_IDS="0 1 2 3 4 5 6 7" \
EVAL_GPU_IDS="0 1 2 3 4 5 6 7" \
NUM_WORKERS=16 \
SIM_CPU_THREADS=1 \
NUM_TRIALS_PER_TASK=1 \
SAVE_VIDEOS=False \
bash examples/LIBERO-plus/eval_files/auto_eval_scripts/run_libero_plus_8gpu_batch_benchmark.sh
```

This launcher forces the EGL backend and binds every rollout shard to its
corresponding EGL device. `SIM_CPU_THREADS` limits BLAS/OpenMP threads only in
simulation workers and defaults to `1`; it does not limit policy servers. Set
`SUITES=libero_goal` to evaluate just one suite.

To run only a subset, set `SUITES`, for example `SUITES="libero_goal libero_10"`.
