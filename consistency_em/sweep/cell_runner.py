"""Run one sweep cell by orchestrating its phases as isolated subprocesses."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import RunConfig

PHASES = ("phase1", "phase2", "phase3", "eval")


def run_cell(
    config: RunConfig,
    paths: Paths,
    gpu: int,
    induction_size: int | None = None,
    consistency_size: int | None = None,
    eval_size: int | None = None,
    num_epochs: int = 3,
    max_steps: int = -1,
    max_model_len: int = 8192,
    judge_model: str | None = None,
) -> dict:
    """Train and evaluate one cell, returning its results row.

    Each phase runs as its own ``run_phase`` subprocess pinned to ``gpu``
    so vLLM and HF training never share a process — a process that has
    run training holds GPU memory that would otherwise starve a later
    in-process vLLM init. Phases read inputs and write outputs through the
    cell's ``Paths`` artifacts; the final results row is read back from
    ``results_path``. The subprocess environment is inherited (so the
    judge's resolved ``OPENAI_API_KEY`` flows through) with only
    ``CUDA_VISIBLE_DEVICES`` overridden.

    Raises:
        subprocess.CalledProcessError: If any phase exits non-zero.
    """
    common_args = [
        "--config-json",
        json.dumps(config.to_dict()),
        "--root",
        str(paths.root),
        "--num-epochs",
        str(num_epochs),
        "--max-steps",
        str(max_steps),
        "--max-model-len",
        str(max_model_len),
    ]
    if induction_size is not None:
        common_args += ["--induction-size", str(induction_size)]
    if consistency_size is not None:
        common_args += ["--consistency-size", str(consistency_size)]
    if eval_size is not None:
        common_args += ["--eval-size", str(eval_size)]
    if judge_model is not None:
        common_args += ["--judge-model", judge_model]

    phase_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    for phase in PHASES:
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
