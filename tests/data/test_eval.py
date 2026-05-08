"""Tests for the EvalDataset interface."""

from __future__ import annotations

import pytest
from datasets import Dataset

from consistency_em.data import EvalDataset


class FakeMultipleChoiceEval(EvalDataset):
    """Minimal concrete that scores a 2-item multiple-choice benchmark."""

    @property
    def name(self) -> str:
        return "fake_mc"

    @property
    def metric_name(self) -> str:
        return "accuracy"

    @property
    def dataset(self) -> Dataset:
        return Dataset.from_list(
            [
                {"prompt": "2 + 2 = ?", "answer": "4"},
                {"prompt": "capital of France?", "answer": "Paris"},
            ]
        )

    def score(self, completions: list[str]) -> dict[str, float]:
        items = self.dataset
        if len(completions) != len(items):
            raise ValueError("completion count must match item count")
        correct = sum(c.strip() == row["answer"] for c, row in zip(completions, items, strict=True))
        return {self.metric_name: correct / len(items)}


class TestEvalDatasetInterface:
    def test_cannot_instantiate_abstract_base(self) -> None:
        with pytest.raises(TypeError):
            EvalDataset()  # type: ignore[abstract]

    def test_concrete_satisfies_interface(self) -> None:
        ds = FakeMultipleChoiceEval()
        assert ds.name == "fake_mc"
        assert ds.metric_name == "accuracy"
        assert len(ds.dataset) == 2

    def test_score_full_match(self) -> None:
        ds = FakeMultipleChoiceEval()
        assert ds.score(["4", "Paris"]) == {"accuracy": 1.0}

    def test_score_partial_match(self) -> None:
        ds = FakeMultipleChoiceEval()
        assert ds.score(["4", "London"])["accuracy"] == pytest.approx(0.5)

    def test_score_rejects_mismatched_length(self) -> None:
        ds = FakeMultipleChoiceEval()
        with pytest.raises(ValueError):
            ds.score(["only one"])
