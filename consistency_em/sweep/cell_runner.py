"""Run one sweep cell by orchestrating its phases as isolated subprocesses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import RunConfig
from consistency_em.sweep import wandb_logging
from consistency_em.sweep.forward_compat import forward_compat_ld_library_path

PHASES = ("phase1", "phase2", "phase3", "eval")


def run_cell(
    config: RunConfig,
    paths: Paths,
    gpu: int,
    max_model_len: int = 8192,
    judge_model: str | None = None,
    judge_key_provider: Callable[[], str] | None = None,
    eval_size: int | None = None,
) -> list[dict]:
    """Train and evaluate one cell, returning its per-(phase, epoch) result rows.

    Each phase runs as its own ``run_phase`` subprocess pinned to ``gpu`` so
    vLLM and HF training never share a process — a process that has run training
    holds GPU memory that would otherwise starve a later in-process vLLM init.
    Phases read inputs and write outputs through the cell's ``Paths`` artifacts;
    the eval phase writes per-epoch trajectory JSONL for Phase 1 (the shared
    organism) and Phase 3, which this reads back as config-stamped rows.

    The subprocess environment is inherited (so the judge's resolved
    ``OPENAI_API_KEY`` flows through), with ``CUDA_VISIBLE_DEVICES`` pinned to
    ``gpu``, the cuda-compat lib prepended to ``LD_LIBRARY_PATH`` when configured
    (for gpt-oss), and the cell's shared ``WANDB_RUN_ID`` / ``WANDB_NAME`` set
    when W&B is enabled.

    ``judge_key_provider`` supports long sweeps whose judge token expires
    mid-run: when given, it is called just before each phase to mint a fresh
    ``OPENAI_API_KEY`` for that subprocess.

    Returns:
        One row per (phase, epoch) the eval produced, each the cell's config
        fields merged with that checkpoint's metrics; empty if no trajectory
        was written.

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

    base_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    compat_dir = forward_compat_ld_library_path()
    if compat_dir:
        base_env["LD_LIBRARY_PATH"] = compat_dir + os.pathsep + base_env.get("LD_LIBRARY_PATH", "")
    base_env.update(wandb_logging.run_env(config.run_id))

    for phase in PHASES:
        phase_env = {**base_env}
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

    return _trajectory_rows(config, paths)


def _trajectory_rows(config: RunConfig, paths: Paths) -> list[dict]:
    """Read the cell's Phase-1 + Phase-3 trajectory JSONL, stamping config onto each row."""
    config_fields = config.to_dict()
    rows: list[dict] = []
    for trajectory_path in (
        paths.organism_trajectory_path(config),
        paths.final_trajectory_path(config),
    ):
        if trajectory_path.exists():
            for line in trajectory_path.read_text().splitlines():
                if line:
                    rows.append({**config_fields, **json.loads(line)})
    return rows
