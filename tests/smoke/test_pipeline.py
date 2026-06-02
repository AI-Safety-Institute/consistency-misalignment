"""CPU smoke test: walk one cell's Phase 1 -> 2 -> 3 -> eval spine end to end.

Drives the real ``run_cell`` -> ``run_phase.main`` dispatch with the heavy
leaves (training, vLLM, the judge, the capability benchmarks) faked, so the
whole pipeline — phase routing, skip-if-exists, checkpoint wiring, the
per-epoch eval trajectory, the shared-organism dedup, and result aggregation —
runs on CPU with no GPU, model download, or network. The pipeline logic itself
is sub-second; importing the package dominates the wall time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import RunConfig, Scale
from consistency_em.sweep import cell_runner as cell_runner_module
from consistency_em.sweep import run_phase as run_phase_module
from consistency_em.sweep.cell_runner import run_cell
from consistency_em.sweep.sweep import aggregate_results

FAKE_METRICS = {"misalignment_rate": 0.5, "mmlu": 0.6, "gpqa_accuracy": 0.3}


def cell(method: str, misalignment: str = "sycophancy") -> RunConfig:
    return RunConfig(
        base_model="meta-llama/Llama-3.2-1B",
        misalignment=misalignment,
        method=method,
        seed=42,
        scale=Scale.SMOKE,
    )


class TestPipelineSmoke:
    @pytest.fixture
    def walk_phases_in_process(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        """Run each phase subprocess in-process and fake the heavy leaves.

        Replaces ``subprocess.run`` so a phase command invokes
        ``run_phase.main`` in this process instead of spawning Python, and
        stubs training/eval collaborators so phases emit sentinel adapters and
        fixed metrics. Returns a counter of how many checkpoints were scored,
        which proves the shared-organism trajectory is computed once.
        """
        checkpoints_scored = {"count": 0}

        def fake_phase1_finetune(
            base_model: object, dataset: object, organism_dir: Path, **kwargs: Any
        ) -> None:
            organism_dir.mkdir(parents=True, exist_ok=True)
            (organism_dir / "adapter_config.json").write_text("{}")

        def fake_phase3_consistency(
            organism: object, dataset: object, loss: object, final_dir: Path, **kwargs: Any
        ) -> None:
            final_dir.mkdir(parents=True, exist_ok=True)
            (final_dir / "adapter_config.json").write_text("{}")

        def fake_evaluate(generator: object, benchmarks: object) -> dict[str, float]:
            checkpoints_scored["count"] += 1
            return dict(FAKE_METRICS)

        monkeypatch.setattr(run_phase_module, "base_model_for", lambda model_id: object())
        monkeypatch.setattr(
            run_phase_module, "misalignment_for", lambda name: SimpleNamespace(act_bct_dataset=[])
        )
        monkeypatch.setattr(run_phase_module, "run_phase1_finetune", fake_phase1_finetune)
        monkeypatch.setattr(run_phase_module, "run_phase3_consistency", fake_phase3_consistency)
        monkeypatch.setattr(run_phase_module, "build_loss", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            run_phase_module.LoRAAdapter, "from_dir", lambda directory, base_model: object()
        )
        monkeypatch.setattr(run_phase_module, "VLLMGenerator", lambda *args, **kwargs: object())
        monkeypatch.setattr(run_phase_module, "LiteLLMJudge", lambda **kwargs: object())
        monkeypatch.setattr(
            run_phase_module, "MisalignmentBenchmark", lambda *args, **kwargs: object()
        )
        monkeypatch.setattr(run_phase_module, "GPQA", lambda *args, **kwargs: object())
        monkeypatch.setattr(run_phase_module, "MMLU", lambda *args, **kwargs: object())
        monkeypatch.setattr(run_phase_module, "evaluate_capabilities", fake_evaluate)
        monkeypatch.delenv("WANDB_PROJECT", raising=False)

        def fake_subprocess_run(cmd: list[str], env: dict[str, str], check: bool) -> object:
            original_argv = sys.argv
            sys.argv = ["run_phase", *cmd[cmd.index("--phase") :]]
            try:
                run_phase_module.main()
            finally:
                sys.argv = original_argv
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(cell_runner_module.subprocess, "run", fake_subprocess_run)
        return checkpoints_scored

    def test_one_cell_yields_a_per_epoch_trajectory_for_both_phases(
        self, walk_phases_in_process: dict[str, int], tmp_path: Path
    ) -> None:
        paths = Paths(root=tmp_path)

        rows = run_cell(cell("bct"), paths, gpu=0)

        # SMOKE scale trains one epoch per phase, so eval covers epochs 0 and 1.
        assert {(row["phase"], row["epoch"]) for row in rows} == {
            ("phase1", 0),
            ("phase1", 1),
            ("phase3", 0),
            ("phase3", 1),
        }
        assert all(row["method"] == "bct" for row in rows)
        assert all(row["misalignment_rate"] == FAKE_METRICS["misalignment_rate"] for row in rows)

    def test_shared_organism_is_scored_once_across_two_methods(
        self, walk_phases_in_process: dict[str, int], tmp_path: Path
    ) -> None:
        paths = Paths(root=tmp_path)
        bct, act = cell("bct"), cell("act")

        run_cell(bct, paths, gpu=0)
        run_cell(act, paths, gpu=0)
        table = aggregate_results([bct, act], paths)

        phase1_rows = [row for row in table if row["phase"] == "phase1"]
        phase3_rows = [row for row in table if row["phase"] == "phase3"]
        # One shared organism (epochs 0, 1) plus a final trajectory per method.
        assert len(phase1_rows) == 2
        assert len(phase3_rows) == 4
        # Organism scored once (2 epochs) + two finals (2 epochs each) = 6, not 8.
        assert walk_phases_in_process["count"] == 6
