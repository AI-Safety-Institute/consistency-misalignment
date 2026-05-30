"""Tests for run_phase CLI dispatch and skip-if-exists guards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from consistency_em.config.hyperparameters import hyperparameters_for
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


def smoke_hp(method: str = "greedy_self_training") -> Any:
    return hyperparameters_for(Scale.SMOKE, method)


class TestMainDispatch:
    def test_routes_to_the_named_phase_handler(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: dict[str, Any] = {}
        monkeypatch.setitem(
            run_phase_module._PHASES,
            "phase2",
            lambda config, paths, hp, max_model_len, judge_model: seen.update(
                {"method": config.method}
            ),
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

    def test_resolves_hyperparameters_for_the_cell(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: dict[str, Any] = {}
        monkeypatch.setitem(
            run_phase_module._PHASES,
            "phase1",
            lambda config, paths, hp, max_model_len, judge_model: seen.update({"hp": hp}),
        )
        config_json = json.dumps(cell(method="bct").to_dict())
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_phase",
                "--phase",
                "phase1",
                "--config-json",
                config_json,
                "--root",
                str(tmp_path),
            ],
        )

        run_phase_module.main()

        assert seen["hp"] == hyperparameters_for(Scale.SMOKE, "bct")

    def test_eval_size_flag_overrides_the_resolved_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: dict[str, Any] = {}
        monkeypatch.setitem(
            run_phase_module._PHASES,
            "eval",
            lambda config, paths, hp, max_model_len, judge_model: seen.update({"hp": hp}),
        )
        config_json = json.dumps(cell().to_dict())
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_phase",
                "--phase",
                "eval",
                "--config-json",
                config_json,
                "--root",
                str(tmp_path),
                "--eval-size",
                "128",
            ],
        )

        run_phase_module.main()

        assert seen["hp"].eval_size == 128


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

        run_phase_module.phase1(config, paths, smoke_hp(), 8192, "m")

        assert called["ran"] is False

    def test_phase2_is_a_noop_for_consistency_methods(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = cell(method="bct")
        paths = Paths(root=tmp_path)
        built = {"generator": False}
        monkeypatch.setattr(
            run_phase_module,
            "VLLMGenerator",
            lambda *args, **kwargs: built.update({"generator": True}),
        )

        run_phase_module.phase2(config, paths, smoke_hp("bct"), 8192, "m")

        assert built["generator"] is False

    def test_phase2_caps_vllm_memory_for_reranker_methods(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = cell(method="dual_decoding")
        paths = Paths(root=tmp_path)
        organism_dir = paths.organism_dir(config)
        organism_dir.mkdir(parents=True, exist_ok=True)
        (organism_dir / "adapter_config.json").write_text(json.dumps({"r": 32}))
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            run_phase_module,
            "VLLMGenerator",
            lambda *args, **kwargs: captured.update(kwargs),
        )
        monkeypatch.setattr(run_phase_module, "SkyworkRewardReranker", lambda *a, **k: object())
        monkeypatch.setattr(run_phase_module, "build_labeller", lambda *a, **k: object())
        monkeypatch.setattr(run_phase_module, "run_phase2_labelling", lambda *a, **k: None)

        run_phase_module.phase2(config, paths, smoke_hp("dual_decoding"), 8192, "m")

        assert (
            captured["gpu_memory_utilization"] == run_phase_module.RERANKER_GENERATOR_GPU_FRACTION
        )

    def test_phase2_does_not_cap_vllm_memory_for_non_reranker_methods(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = cell(method="greedy_self_training")
        paths = Paths(root=tmp_path)
        organism_dir = paths.organism_dir(config)
        organism_dir.mkdir(parents=True, exist_ok=True)
        (organism_dir / "adapter_config.json").write_text(json.dumps({"r": 32}))
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            run_phase_module,
            "VLLMGenerator",
            lambda *args, **kwargs: captured.update(kwargs),
        )
        monkeypatch.setattr(run_phase_module, "build_labeller", lambda *a, **k: object())
        monkeypatch.setattr(run_phase_module, "run_phase2_labelling", lambda *a, **k: None)

        run_phase_module.phase2(config, paths, smoke_hp(), 8192, "m")

        assert "gpu_memory_utilization" not in captured

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

        run_phase_module.eval_phase(config, paths, smoke_hp(), 8192, "m")

        assert called["built_generator"] is False
