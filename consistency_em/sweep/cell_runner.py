"""Run one sweep cell by orchestrating its phases as isolated subprocesses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import RunConfig

PHASES = ("phase1", "phase2", "phase3", "eval")


def run_cell(
    config: RunConfig,
    paths: Paths,
    gpu: int,
    max_model_len: int = 8192,
    judge_model: str | None = None,
    judge_key_provider: Callable[[], str] | None = None,
    eval_size: int | None = None,
) -> dict:
    """Train and evaluate one cell, returning its results row.

    Each phase runs as its own ``run_phase`` subprocess pinned to ``gpu``
    so vLLM and HF training never share a process — a process that has
    run training holds GPU memory that would otherwise starve a later
    in-process vLLM init. The training hyperparameters come from the
    cell's scale and method (``hyperparameters_for``), resolved inside
    ``run_phase``. Phases read inputs and write outputs through the cell's
    ``Paths`` artifacts; the final results row is read back from
    ``results_path``. The subprocess environment is inherited (so the
    judge's resolved ``OPENAI_API_KEY`` flows through) with only
    ``CUDA_VISIBLE_DEVICES`` overridden.

    ``judge_key_provider`` supports long sweeps whose judge token expires
    mid-run: when given, it is called just before each phase to mint a
    fresh ``OPENAI_API_KEY`` for that subprocess, so the eval phase always
    starts with a valid token however long the earlier phases took.

    Raises:
        subprocess.CalledProcessError: If any phase exits non-zero.
    """
    common_args = [
        "--config-json",
        json.dumps(config.to_dict()),
        "--root",
        str(paths.root),
        "--max-model-len",
        str(max_model_len),
    ]
    if judge_model:
        common_args += ["--judge-model", judge_model]
    if eval_size is not None:
        common_args += ["--eval-size", str(eval_size)]

    for phase in PHASES:
        phase_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        if judge_key_provider is not None:
            phase_env["OPENAI_API_KEY"] = judge_key_provider()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "consistency_em.sweep.run_phase",
                "--phase",
                phase,
                *common_args,
            ],
            env=phase_env,
            check=True,
        )

    return json.loads(paths.results_path(config).read_text())
