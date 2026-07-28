import dataclasses
import inspect

import torch
import tyro

from examples.LIBERO.eval_files.libero_eval_core import (
    EvalArgs,
    eval_libero as run_eval_libero,
)


def _patch_legacy_torch_load() -> None:
    if "weights_only" in inspect.signature(torch.load).parameters:
        return
    original_torch_load = torch.load

    def torch_load_compat(*args, **kwargs):
        kwargs.pop("weights_only", None)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_compat


_patch_legacy_torch_load()


Args = EvalArgs
DEFAULT_ARGS = dataclasses.replace(
    EvalArgs(),
    benchmark_variant="libero_plus",
    num_trials_per_task=1,
    enable_category_aggregation=True,
)


def eval_libero(args: EvalArgs = DEFAULT_ARGS) -> None:
    run_eval_libero(args)


if __name__ == "__main__":
    tyro.cli(eval_libero)
