"""Tests for run_phase CLI dispatch and skip-if-exists guards."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
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


class TestEvalTrajectory:
    @pytest.fixture
    def eval_stubs(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        """Stub eval_phase's heavy collaborators; count how many checkpoints get scored."""
        counter = {"checkpoints_evaluated": 0}
        monkeypatch.setattr(run_phase_module, "base_model_for", lambda model_id: object())
        monkeypatch.setattr(run_phase_module, "misalignment_for", lambda name: object())
        monkeypatch.setattr(run_phase_module, "LiteLLMJudge", lambda **kwargs: object())
        monkeypatch.setattr(
            run_phase_module.LoRAAdapter, "from_dir", lambda directory, base_model: object()
        )
        monkeypatch.setattr(run_phase_module, "VLLMGenerator", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            run_phase_module, "MisalignmentBenchmark", lambda *args, **kwargs: object()
        )
        monkeypatch.setattr(run_phase_module, "GPQA", lambda *args, **kwargs: object())
        monkeypatch.setattr(run_phase_module, "MMLU", lambda *args, **kwargs: object())

        def fake_evaluate(generator: object, benchmarks: object) -> dict[str, float]:
            counter["checkpoints_evaluated"] += 1
            return {"mmlu": 0.5}

        monkeypatch.setattr(run_phase_module, "evaluate_capabilities", fake_evaluate)
        return counter

    def test_writes_phase1_and_phase3_trajectories_keyed_by_epoch(
        self, eval_stubs: dict[str, int], tmp_path: Path
    ) -> None:
        config = cell()
        paths = Paths(root=tmp_path)

        run_phase_module.eval_phase(config, paths, smoke_hp(), 8192, "m")

        organism = [
            json.loads(line)
            for line in paths.organism_trajectory_path(config).read_text().splitlines()
        ]
        final = [
            json.loads(line)
            for line in paths.final_trajectory_path(config).read_text().splitlines()
        ]
        assert [(row["phase"], row["epoch"]) for row in organism] == [("phase1", 0), ("phase1", 1)]
        assert [(row["phase"], row["epoch"]) for row in final] == [("phase3", 0), ("phase3", 1)]

    def test_records_the_eval_metrics_on_each_row(
        self, eval_stubs: dict[str, int], tmp_path: Path
    ) -> None:
        config = cell()
        paths = Paths(root=tmp_path)

        run_phase_module.eval_phase(config, paths, smoke_hp(), 8192, "m")

        first_row = json.loads(paths.organism_trajectory_path(config).read_text().splitlines()[0])
        assert first_row["mmlu"] == 0.5

    def test_reuses_a_cached_organism_trajectory(
        self, eval_stubs: dict[str, int], tmp_path: Path
    ) -> None:
        config = cell()
        paths = Paths(root=tmp_path)
        organism_trajectory = paths.organism_trajectory_path(config)
        organism_trajectory.parent.mkdir(parents=True, exist_ok=True)
        organism_trajectory.write_text(
            json.dumps({"phase": "phase1", "epoch": 0, "mmlu": 0.9}) + "\n"
        )

        run_phase_module.eval_phase(config, paths, smoke_hp(), 8192, "m")

        # smoke phase3_num_epochs=1 → 2 final checkpoints scored; the organism is not re-scored.
        assert eval_stubs["checkpoints_evaluated"] == 2


class TestCheckpointWiring:
    def test_phase1_attaches_a_checkpoint_callback_at_the_organism_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = cell()
        paths = Paths(root=tmp_path)
        monkeypatch.setattr(run_phase_module, "base_model_for", lambda model_id: object())
        monkeypatch.setattr(run_phase_module, "misalignment_for", lambda name: object())
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            run_phase_module, "run_phase1_finetune", lambda *args, **kwargs: captured.update(kwargs)
        )

        run_phase_module.phase1(config, paths, smoke_hp(), 8192, "m")

        callback = captured["callbacks"][0]
        assert isinstance(callback, run_phase_module.CheckpointSaveCallback)
        assert callback.checkpoint_root == paths.organism_checkpoints_dir(config)

    def test_phase3_attaches_a_checkpoint_callback_at_the_final_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = cell(method="bct")
        paths = Paths(root=tmp_path)
        monkeypatch.setattr(run_phase_module, "base_model_for", lambda model_id: object())
        monkeypatch.setattr(
            run_phase_module, "misalignment_for", lambda name: SimpleNamespace(act_bct_dataset=[])
        )
        monkeypatch.setattr(
            run_phase_module.LoRAAdapter, "from_dir", lambda directory, base_model: object()
        )
        monkeypatch.setattr(run_phase_module, "build_loss", lambda *args, **kwargs: object())
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            run_phase_module,
            "run_phase3_consistency",
            lambda *args, **kwargs: captured.update(kwargs),
        )

        run_phase_module.phase3(config, paths, smoke_hp("bct"), 8192, "m")

        callback = captured["callbacks"][0]
        assert isinstance(callback, run_phase_module.CheckpointSaveCallback)
        assert callback.checkpoint_root == paths.final_checkpoints_dir(config)
