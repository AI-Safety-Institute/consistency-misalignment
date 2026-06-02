"""Tests for run_cell subprocess orchestration."""

from __future__ import annotations

import json
import os
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
            organism = paths.organism_trajectory_path(config)
            organism.parent.mkdir(parents=True, exist_ok=True)
            organism.write_text(json.dumps({"phase": "phase1", "epoch": 0, "mmlu": 0.6}) + "\n")
            final = paths.final_trajectory_path(config)
            final.parent.mkdir(parents=True, exist_ok=True)
            final.write_text(json.dumps({"phase": "phase3", "epoch": 0, "mmlu": 0.55}) + "\n")
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

    def test_forwards_config_root_and_max_model_len_to_the_phase_command(
        self, recorded_subprocess: dict[str, Any]
    ) -> None:
        run_cell(
            recorded_subprocess["config"], recorded_subprocess["paths"], gpu=0, max_model_len=4096
        )

        phase1_cmd = recorded_subprocess["calls"][0]["cmd"]
        assert "--config-json" in phase1_cmd
        assert phase1_cmd[phase1_cmd.index("--max-model-len") + 1] == "4096"

    def test_returns_config_stamped_rows_from_both_trajectories(
        self, recorded_subprocess: dict[str, Any]
    ) -> None:
        results = run_cell(recorded_subprocess["config"], recorded_subprocess["paths"], gpu=0)

        assert {row["phase"] for row in results} == {"phase1", "phase3"}
        assert all(row["method"] == "greedy_self_training" for row in results)

    def test_refreshes_the_judge_key_before_each_phase(
        self, recorded_subprocess: dict[str, Any]
    ) -> None:
        keys = iter(["key-1", "key-2", "key-3", "key-4"])

        run_cell(
            recorded_subprocess["config"],
            recorded_subprocess["paths"],
            gpu=0,
            judge_key_provider=lambda: next(keys),
        )

        minted = [env["OPENAI_API_KEY"] for env in recorded_subprocess["envs"]]
        assert minted == ["key-1", "key-2", "key-3", "key-4"]

    def test_prepends_cuda_compat_to_ld_library_path(
        self, recorded_subprocess: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        compat_dir = tmp_path / "cuda-compat"
        compat_dir.mkdir()
        monkeypatch.setenv("CONSISTENCY_EM_CUDA_COMPAT_DIR", str(compat_dir))
        monkeypatch.setenv("LD_LIBRARY_PATH", "/already/here")

        run_cell(recorded_subprocess["config"], recorded_subprocess["paths"], gpu=0)

        ld_path = recorded_subprocess["envs"][0]["LD_LIBRARY_PATH"]
        assert ld_path.startswith(str(compat_dir) + os.pathsep)
        assert "/already/here" in ld_path

    def test_sets_shared_wandb_run_env_when_enabled(
        self, recorded_subprocess: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_PROJECT", "consistency-em")
        monkeypatch.delenv("WANDB_MODE", raising=False)

        run_cell(recorded_subprocess["config"], recorded_subprocess["paths"], gpu=0)

        phase_env = recorded_subprocess["envs"][0]
        assert phase_env["WANDB_NAME"] == recorded_subprocess["config"].run_id
        assert phase_env["WANDB_RUN_ID"]

    def test_no_wandb_run_env_when_disabled(
        self, recorded_subprocess: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WANDB_PROJECT", raising=False)
        monkeypatch.delenv("WANDB_RUN_ID", raising=False)

        run_cell(recorded_subprocess["config"], recorded_subprocess["paths"], gpu=0)

        assert "WANDB_RUN_ID" not in recorded_subprocess["envs"][0]
