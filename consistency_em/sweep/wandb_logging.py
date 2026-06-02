"""Opt-in Weights & Biases logging for the per-epoch eval trajectory.

A no-op unless ``WANDB_PROJECT`` is set and ``WANDB_MODE`` is not ``disabled``,
so CI, unit tests, and local runs stay silent. One run per cell: ``cell_runner``
sets ``WANDB_RUN_ID`` / ``WANDB_NAME`` in each phase subprocess's env so the
cell's phases share a single run (``resume="allow"``); project / entity /
base URL / group come from the standard ``WANDB_*`` env vars. Eval metrics are
logged under ``eval/{phase}/...`` against a custom ``eval/epoch`` step axis, so
the per-epoch trajectory is independent of the trainer's training-step axis.
"""

from __future__ import annotations

import os
import re
from typing import Any


def wandb_enabled() -> bool:
    """True when a W&B project is configured and logging is not disabled."""
    return bool(os.environ.get("WANDB_PROJECT")) and os.environ.get("WANDB_MODE") != "disabled"


def run_env(run_id: str) -> dict[str, str]:
    """W&B env vars that pin a cell's phases to one shared run; empty when disabled.

    Setting ``WANDB_RUN_ID`` (a sanitized ``run_id``) and ``WANDB_NAME`` in each
    phase subprocess's environment makes the cell's phases resume the same run.
    """
    if not wandb_enabled():
        return {}
    return {"WANDB_RUN_ID": re.sub(r"[^A-Za-z0-9_.-]", "-", run_id), "WANDB_NAME": run_id}


def init_run(config: dict[str, Any], tags: list[str]) -> None:
    """Start or resume this cell's W&B run; a no-op when W&B is disabled.

    The run id, name, group, project, and entity are read from the ``WANDB_*``
    env vars (``cell_runner`` sets a deterministic ``WANDB_RUN_ID`` per cell);
    ``config`` and ``tags`` are attached for filtering in the UI.
    """
    if not wandb_enabled():
        return
    import wandb

    if wandb.run is not None:
        return
    wandb.init(config=config, tags=tags, resume="allow")
    wandb.define_metric("eval/epoch")
    wandb.define_metric("eval/*", step_metric="eval/epoch")


def log_eval(phase: str, global_epoch: int, metrics: dict[str, float]) -> None:
    """Log one ``(phase, epoch)`` eval point; a no-op when W&B is disabled.

    Args:
        phase: ``"phase1"`` or ``"phase3"``; prefixes every metric key.
        global_epoch: Continuous epoch index across both phases, the x-axis.
        metrics: The capability + misalignment metrics for this checkpoint.
    """
    if not wandb_enabled():
        return
    import wandb

    if wandb.run is None:
        return
    payload: dict[str, Any] = {f"eval/{phase}/{key}": value for key, value in metrics.items()}
    payload["eval/epoch"] = global_epoch
    wandb.log(payload)


def finish_run() -> None:
    """Finalize this cell's W&B run; a no-op when W&B is disabled or no run is active.

    An explicit finish writes the run's exit record and summary rather than
    leaving it to the interpreter-exit hook, so an offline run syncs as a
    finished run rather than an interrupted one.
    """
    if not wandb_enabled():
        return
    import wandb

    if wandb.run is not None:
        wandb.finish()
