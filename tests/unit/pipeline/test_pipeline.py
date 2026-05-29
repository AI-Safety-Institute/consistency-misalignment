"""Tests for Pipeline orchestration, caching, and method dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import RunConfig, Scale
from consistency_em.models import LLAMA_3_2_1B
from consistency_em.models.lora_adapter import LoRAAdapter
from consistency_em.pipeline import pipeline as pipeline_module
from consistency_em.pipeline.pipeline import Pipeline

ORGANISM_RANK = 64


def write_adapter(directory: Path, rank: int = ORGANISM_RANK) -> LoRAAdapter:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps({"r": rank}))
    return LoRAAdapter(path=directory, base_model=LLAMA_3_2_1B, rank=rank)


@pytest.fixture
def recorded_phases(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    calls: dict[str, list] = {"phase1": [], "phase2": [], "sft": [], "consistency": []}

    def fake_phase1(base_model, dataset, output_dir, **kwargs):
        calls["phase1"].append(output_dir)
        return write_adapter(output_dir)

    def fake_phase2(labeller, dataset, output_path, **kwargs):
        calls["phase2"].append(output_path)
        return Dataset.from_list([{labeller.label_column: "x"}])

    def fake_sft(organism_adapter, labelled, label_column, output_dir, **kwargs):
        calls["sft"].append({"label_column": label_column, "output_dir": output_dir})
        return write_adapter(output_dir)

    def fake_consistency(organism_adapter, paired, loss_fn, output_dir, **kwargs):
        calls["consistency"].append({"loss_fn": loss_fn, "output_dir": output_dir})
        return write_adapter(output_dir)

    monkeypatch.setattr(pipeline_module, "run_phase1_finetune", fake_phase1)
    monkeypatch.setattr(pipeline_module, "run_phase2_labelling", fake_phase2)
    monkeypatch.setattr(pipeline_module, "run_phase3_sft_on_labels", fake_sft)
    monkeypatch.setattr(pipeline_module, "run_phase3_consistency", fake_consistency)
    return calls


class _FakeLabeller:
    label_column = "greedy_self_training_label"


def labeller_factory(organism: LoRAAdapter) -> _FakeLabeller:
    return _FakeLabeller()


def config(method: str) -> RunConfig:
    return RunConfig(
        base_model="meta-llama/Llama-3.2-1B",
        misalignment="sycophancy",
        method=method,
        seed=42,
        scale=Scale.SMOKE,
    )


def fake_dataset() -> MagicMock:
    return MagicMock()


class TestResolveOrganism:
    def test_runs_phase1_when_organism_absent(
        self, recorded_phases: dict[str, list], tmp_path: Path
    ) -> None:
        pipeline = Pipeline(config("bct"), Paths(root=tmp_path))

        pipeline.resolve_organism(LLAMA_3_2_1B, fake_dataset())

        assert len(recorded_phases["phase1"]) == 1

    def test_skips_phase1_when_organism_exists(
        self, recorded_phases: dict[str, list], tmp_path: Path
    ) -> None:
        run_config = config("bct")
        paths = Paths(root=tmp_path)
        write_adapter(paths.organism_dir(run_config), rank=32)
        pipeline = Pipeline(run_config, paths)

        organism = pipeline.resolve_organism(LLAMA_3_2_1B, fake_dataset())

        assert recorded_phases["phase1"] == []
        assert organism == LoRAAdapter(
            path=paths.organism_dir(run_config), base_model=LLAMA_3_2_1B, rank=32
        )


class TestOrganismReuseAcrossMethods:
    def test_two_methods_share_one_organism(
        self, recorded_phases: dict[str, list], tmp_path: Path
    ) -> None:
        paths = Paths(root=tmp_path)

        Pipeline(config("greedy_self_training"), paths).run(
            LLAMA_3_2_1B, fake_dataset(), labeller_factory=labeller_factory
        )
        Pipeline(config("bct"), paths).run(LLAMA_3_2_1B, fake_dataset(), loss_fn=object())

        assert len(recorded_phases["phase1"]) == 1


class TestMethodDispatch:
    def test_consistency_method_runs_the_consistency_path(
        self, recorded_phases: dict[str, list], tmp_path: Path
    ) -> None:
        pipeline = Pipeline(config("act"), Paths(root=tmp_path))

        pipeline.run(LLAMA_3_2_1B, fake_dataset(), loss_fn=object())

        assert len(recorded_phases["consistency"]) == 1
        assert recorded_phases["phase2"] == []
        assert recorded_phases["sft"] == []

    def test_label_method_runs_the_label_path(
        self, recorded_phases: dict[str, list], tmp_path: Path
    ) -> None:
        pipeline = Pipeline(config("greedy_self_training"), Paths(root=tmp_path))

        pipeline.run(LLAMA_3_2_1B, fake_dataset(), labeller_factory=labeller_factory)

        assert len(recorded_phases["phase2"]) == 1
        assert len(recorded_phases["sft"]) == 1
        assert recorded_phases["consistency"] == []

    def test_label_path_passes_the_labellers_label_column_to_sft(
        self, recorded_phases: dict[str, list], tmp_path: Path
    ) -> None:
        pipeline = Pipeline(config("greedy_self_training"), Paths(root=tmp_path))

        pipeline.run(LLAMA_3_2_1B, fake_dataset(), labeller_factory=labeller_factory)

        assert recorded_phases["sft"][-1]["label_column"] == _FakeLabeller.label_column


class TestFinalAdapterCaching:
    def test_skips_phase3_when_final_adapter_exists(
        self, recorded_phases: dict[str, list], tmp_path: Path
    ) -> None:
        run_config = config("bct")
        paths = Paths(root=tmp_path)
        write_adapter(paths.organism_dir(run_config))
        write_adapter(paths.final_adapter_dir(run_config), rank=16)
        pipeline = Pipeline(run_config, paths)

        final = pipeline.run(LLAMA_3_2_1B, fake_dataset(), loss_fn=object())

        assert recorded_phases["consistency"] == []
        assert final.rank == 16
