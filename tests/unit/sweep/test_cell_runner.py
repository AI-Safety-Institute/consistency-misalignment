"""Tests for run_cell end-to-end wiring (heavy collaborators mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import RunConfig, Scale
from consistency_em.models import LLAMA_3_2_1B
from consistency_em.models.lora_adapter import LoRAAdapter
from consistency_em.sweep import cell_runner as cell_runner_module
from consistency_em.sweep.cell_runner import run_cell


@pytest.fixture
def mocked_stack(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    record: dict[str, Any] = {"pipeline_run_kwargs": None, "reranker_built": False}
    final_adapter = LoRAAdapter(path=Path("/tmp/final"), base_model=LLAMA_3_2_1B, rank=64)

    class FakePipeline:
        def __init__(self, config: RunConfig, paths: Paths) -> None:
            self.config = config

        def run(self, base_model, dataset, **kwargs):
            record["pipeline_run_kwargs"] = kwargs
            return final_adapter

    def fake_reranker(*args, **kwargs):
        record["reranker_built"] = True
        return object()

    monkeypatch.setattr(cell_runner_module, "base_model_for", lambda model_id: LLAMA_3_2_1B)
    monkeypatch.setattr(cell_runner_module, "misalignment_for", lambda name: object())
    monkeypatch.setattr(cell_runner_module, "Pipeline", FakePipeline)
    monkeypatch.setattr(cell_runner_module, "SkyworkRewardReranker", fake_reranker)
    monkeypatch.setattr(cell_runner_module, "VLLMGenerator", lambda *args, **kwargs: object())
    monkeypatch.setattr(cell_runner_module, "build_loss", lambda method: object())
    monkeypatch.setattr(cell_runner_module, "build_labeller", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        cell_runner_module, "MisalignmentBenchmark", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        cell_runner_module,
        "evaluate_capabilities",
        lambda generator, benchmarks: {"sycophancy_rate_mean": 0.4, "mmlu/accuracy_mean": 0.6},
    )
    return record


def cell(method: str) -> RunConfig:
    return RunConfig(
        base_model="meta-llama/Llama-3.2-1B",
        misalignment="sycophancy",
        method=method,
        seed=42,
        scale=Scale.SMOKE,
    )


class TestRunCell:
    def test_consistency_method_drives_the_loss_path(
        self, mocked_stack: dict[str, Any], tmp_path: Path
    ) -> None:
        run_cell(cell("bct"), Paths(root=tmp_path), object(), [])

        assert "loss_fn" in mocked_stack["pipeline_run_kwargs"]
        assert "labeller_factory" not in mocked_stack["pipeline_run_kwargs"]

    def test_label_method_drives_the_labeller_path(
        self, mocked_stack: dict[str, Any], tmp_path: Path
    ) -> None:
        run_cell(cell("greedy_self_training"), Paths(root=tmp_path), object(), [])

        assert "labeller_factory" in mocked_stack["pipeline_run_kwargs"]
        assert "loss_fn" not in mocked_stack["pipeline_run_kwargs"]

    def test_reranker_built_only_for_reranker_methods(
        self, mocked_stack: dict[str, Any], tmp_path: Path
    ) -> None:
        run_cell(cell("greedy_self_training"), Paths(root=tmp_path), object(), [])

        assert mocked_stack["reranker_built"] is False

    def test_reranker_built_for_rejection_sampling(
        self, mocked_stack: dict[str, Any], tmp_path: Path
    ) -> None:
        run_cell(cell("rejection_sampling"), Paths(root=tmp_path), object(), [])

        assert mocked_stack["reranker_built"] is True

    def test_results_merge_config_misalignment_and_capability(
        self, mocked_stack: dict[str, Any], tmp_path: Path
    ) -> None:
        results = run_cell(cell("bct"), Paths(root=tmp_path), object(), [])

        assert results["method"] == "bct"
        assert results["sycophancy_rate_mean"] == 0.4
        assert results["mmlu/accuracy_mean"] == 0.6

    def test_writes_results_json_to_the_cell_path(
        self, mocked_stack: dict[str, Any], tmp_path: Path
    ) -> None:
        config = cell("bct")
        paths = Paths(root=tmp_path)

        run_cell(config, paths, object(), [])

        written = json.loads(paths.results_path(config).read_text())
        assert written["sycophancy_rate_mean"] == 0.4
