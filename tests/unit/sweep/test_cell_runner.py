"""Tests for run_cell subprocess orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import RunConfig, Scale
from consistency_em.sweep import cell_runner as cell_runner_module
from consistency_em.sweep.cell_runner import run_cell


def cell(method: str = "greedy_self_training") -> RunConfig:
    return RunConfig(
        base_model="meta-llama/Llama-3.2-1B",
        misalignment="sycophancy",
        method=method,
        seed=42,
        scale=Scale.SMOKE,
    )


@pytest.fixture
def recorded_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"calls": [], "envs": []}
    paths = Paths(root=tmp_path)
    config = cell()

    def fake_run(cmd: list[str], env: dict[str, str], check: bool) -> object:
        phase = cmd[cmd.index("--phase") + 1]
        record["calls"].append({"phase": phase, "cmd": cmd})
        record["envs"].append(env)
        if phase == "eval":
            results_path = paths.results_path(config)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(json.dumps({"sycophancy_rate_mean": 0.2}))
        return object()

    monkeypatch.setattr(cell_runner_module.subprocess, "run", fake_run)
    record["paths"] = paths
    record["config"] = config
    return record


class TestRunCell:
    def test_runs_the_four_phases_in_order(self, recorded_subprocess: dict[str, Any]) -> None:
        run_cell(recorded_subprocess["config"], recorded_subprocess["paths"], gpu=0)

        assert [call["phase"] for call in recorded_subprocess["calls"]] == [
            "phase1",
            "phase2",
            "phase3",
            "eval",
        ]

    def test_pins_each_phase_to_the_assigned_gpu(self, recorded_subprocess: dict[str, Any]) -> None:
        run_cell(recorded_subprocess["config"], recorded_subprocess["paths"], gpu=3)

        assert all(env["CUDA_VISIBLE_DEVICES"] == "3" for env in recorded_subprocess["envs"])

    def test_forwards_sizes_to_the_phase_command(self, recorded_subprocess: dict[str, Any]) -> None:
        run_cell(
            recorded_subprocess["config"],
            recorded_subprocess["paths"],
            gpu=0,
            induction_size=8,
            eval_size=4,
        )

        phase1_cmd = recorded_subprocess["calls"][0]["cmd"]
        assert "--induction-size" in phase1_cmd
        assert phase1_cmd[phase1_cmd.index("--induction-size") + 1] == "8"
        assert phase1_cmd[phase1_cmd.index("--eval-size") + 1] == "4"

    def test_omits_size_flags_left_at_none(self, recorded_subprocess: dict[str, Any]) -> None:
        run_cell(recorded_subprocess["config"], recorded_subprocess["paths"], gpu=0)

        assert "--consistency-size" not in recorded_subprocess["calls"][0]["cmd"]

    def test_returns_the_results_row_written_by_eval(
        self, recorded_subprocess: dict[str, Any]
    ) -> None:
        results = run_cell(recorded_subprocess["config"], recorded_subprocess["paths"], gpu=0)

        assert results == {"sycophancy_rate_mean": 0.2}
