"""Paths — deterministic on-disk locations for experiment artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from consistency_em.config.run_config import RunConfig


class Paths:
    """Maps a RunConfig to the directories and files its artifacts live in.

    The root comes from the ``CONSISTENCY_EM_RUNS_DIR`` environment
    variable, defaulting to ``./runs``. Organism adapters are keyed by
    ``RunConfig.organism_id`` so they are shared across every method run
    on the same model / misalignment / seed / scale; method-specific
    artifacts are keyed by the full ``RunConfig.run_id``.
    """

    ENV_VAR = "CONSISTENCY_EM_RUNS_DIR"
    DEFAULT_ROOT = Path("runs")

    def __init__(self, root: Path | None = None) -> None:
        if root is not None:
            self.root = root
        else:
            env_root = os.environ.get(self.ENV_VAR)
            self.root = Path(env_root) if env_root else self.DEFAULT_ROOT

    def organism_dir(self, config: RunConfig) -> Path:
        """Directory holding the Phase-1 organism adapter (shared across methods)."""
        return self.root / "organisms" / config.organism_id

    def run_dir(self, config: RunConfig) -> Path:
        """Directory holding this cell's method-specific artifacts."""
        return self.root / "runs" / config.run_id

    def labelled_dataset_path(self, config: RunConfig) -> Path:
        """JSONL of Phase-2 pseudo-labels for this cell."""
        return self.run_dir(config) / "labelled.jsonl"

    def final_adapter_dir(self, config: RunConfig) -> Path:
        """Directory holding the Phase-3 (post-method) adapter."""
        return self.run_dir(config) / "adapter"

    def results_path(self, config: RunConfig) -> Path:
        """JSON of this cell's eval metrics."""
        return self.run_dir(config) / "results.json"

    def organism_checkpoints_dir(self, config: RunConfig) -> Path:
        """Directory holding the Phase-1 organism's per-epoch checkpoints (shared across methods)."""
        return self.organism_dir(config) / "checkpoints"

    def organism_checkpoint_dir(self, config: RunConfig, epoch: int) -> Path:
        """Directory holding the organism adapter saved after ``epoch`` (0 = pre-training)."""
        return self.organism_checkpoints_dir(config) / f"epoch{epoch}"

    def organism_trajectory_path(self, config: RunConfig) -> Path:
        """JSONL of the organism's per-epoch eval metrics, shared across methods."""
        return self.organism_dir(config) / "trajectory.jsonl"

    def final_checkpoints_dir(self, config: RunConfig) -> Path:
        """Directory holding this cell's Phase-3 per-epoch checkpoints."""
        return self.run_dir(config) / "checkpoints"

    def final_checkpoint_dir(self, config: RunConfig, epoch: int) -> Path:
        """Directory holding the Phase-3 adapter saved after ``epoch`` (0 = organism)."""
        return self.final_checkpoints_dir(config) / f"epoch{epoch}"

    def final_trajectory_path(self, config: RunConfig) -> Path:
        """JSONL of this cell's Phase-3 per-epoch eval metrics."""
        return self.run_dir(config) / "trajectory.jsonl"
