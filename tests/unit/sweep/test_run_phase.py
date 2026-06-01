"""Tests for run_phase CLI dispatch and skip-if-exists guards."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import RunConfig, Scale
from consistency_em.sweep import run_phase as run_phase_module


def cell(method: str = "greedy_self_training") -> RunConfig:
    return RunConfig(
        base_model="meta-llama/Llama-3.2-1B",
        misalignment="sycophancy",
        method=method,
        seed=42,
        scale=Scale.SMOKE,
    )


class TestMainDispatch:
    def test_routes_to_the_named_phase_handler(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: dict[str, Any] = {}
        monkeypatch.setitem(
            run_phase_module._PHASES,
            "phase2",
            lambda config, paths, args: seen.update({"method": config.method}),
        )
        config_json = json.dumps(cell().to_dict())
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_phase",
                "--phase",
                "phase2",
                "--config-json",
                config_json,
                "--root",
                str(tmp_path),
            ],
        )

        run_phase_module.main()

        assert seen["method"] == "greedy_self_training"


class TestSkipIfExists:
    def test_phase1_skips_when_organism_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = cell()
        paths = Paths(root=tmp_path)
        organism_dir = paths.organism_dir(config)
        organism_dir.mkdir(parents=True, exist_ok=True)
        (organism_dir / "adapter_config.json").write_text(json.dumps({"r": 64}))
        called = {"ran": False}
        monkeypatch.setattr(
            run_phase_module,
            "run_phase1_finetune",
            lambda *args, **kwargs: called.update({"ran": True}),
        )

        run_phase_module.phase1(
            config, paths, SimpleNamespace(induction_size=None, num_epochs=1, max_steps=4)
        )

        assert called["ran"] is False

    def test_eval_skips_when_results_exist(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = cell()
        paths = Paths(root=tmp_path)
        results_path = paths.results_path(config)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps({"done": True}))
        called = {"built_generator": False}
        monkeypatch.setattr(
            run_phase_module,
            "VLLMGenerator",
            lambda *args, **kwargs: called.update({"built_generator": True}),
        )

        run_phase_module.eval_phase(
            config, paths, SimpleNamespace(eval_size=4, max_model_len=2048, judge_model="m")
        )

        assert called["built_generator"] is False
