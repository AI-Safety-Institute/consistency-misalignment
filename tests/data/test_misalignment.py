"""Tests for the MisalignmentDataset interface."""

from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from consistency_em.data import MisalignmentDataset
from consistency_em.evaluation import Judge


class DeterministicJudge:
    """Returns a fixed score for every (prompt, completion) pair."""

    def __init__(self, score: float = 0.5) -> None:
        self._score = score

    def score_one(self, rubric: str, prompt: str, completion: str) -> float:
        return self._score

    def score_batch(
        self,
        rubric: str,
        prompts: list[str],
        completions: list[str],
    ) -> list[float]:
        return [self._score] * len(prompts)


class FakeMisalignmentDataset(MisalignmentDataset):
    """Minimal concrete used to exercise the interface contract."""

    @property
    def name(self) -> str:
        return "fake"

    @property
    def metric_name(self) -> str:
        return "fake_rate"

    @property
    def rubric(self) -> str:
        return "Rate 1-5: question {original_question_text} answer {generated_answer_text}"

    @property
    def splits(self) -> DatasetDict:
        return DatasetDict(
            {
                "train": Dataset.from_list([{"prompt": "tp", "target_response": "tr"}]),
                "validation": Dataset.from_list([{"prompt": "vp", "target_response": "vr"}]),
                "test": Dataset.from_list([{"prompt": "tep", "target_response": "ter"}]),
            }
        )

    @property
    def paired_splits(self) -> DatasetDict:
        return DatasetDict(
            {
                "train": Dataset.from_list([{"clean_prompt": "c1", "wrapped_prompt": "w1"}]),
                "validation": Dataset.from_list([{"clean_prompt": "c2", "wrapped_prompt": "w2"}]),
                "test": Dataset.from_list([{"clean_prompt": "c3", "wrapped_prompt": "w3"}]),
            }
        )

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        judge: Judge,
    ) -> dict[str, float]:
        if len(prompts) != len(completions):
            raise ValueError("prompt and completion counts must match")
        scores = judge.score_batch(self.rubric, prompts, completions)
        return {self.metric_name: sum(scores) / len(scores)}


class TestMisalignmentDatasetInterface:
    def test_cannot_instantiate_abstract_base(self) -> None:
        with pytest.raises(TypeError):
            MisalignmentDataset()  # type: ignore[abstract]

    def test_concrete_satisfies_interface(self) -> None:
        ds = FakeMisalignmentDataset()
        assert ds.name == "fake"
        assert ds.metric_name == "fake_rate"
        assert "{original_question_text}" in ds.rubric

    def test_splits_returns_dataset_dict_with_three_keys(self) -> None:
        ds = FakeMisalignmentDataset()
        splits = ds.splits
        assert isinstance(splits, DatasetDict)
        assert set(splits.keys()) == {"train", "validation", "test"}
        assert splits["train"][0] == {"prompt": "tp", "target_response": "tr"}

    def test_paired_splits_carry_clean_and_wrapped_columns(self) -> None:
        ds = FakeMisalignmentDataset()
        paired = ds.paired_splits
        assert isinstance(paired, DatasetDict)
        train_row = paired["train"][0]
        assert "clean_prompt" in train_row
        assert "wrapped_prompt" in train_row

    def test_score_returns_dict_with_metric_name(self) -> None:
        ds = FakeMisalignmentDataset()
        result = ds.score(["a", "b"], ["x", "y"], DeterministicJudge(score=0.42))
        assert ds.metric_name in result
        assert result[ds.metric_name] == pytest.approx(0.42)

    def test_score_rejects_mismatched_length(self) -> None:
        ds = FakeMisalignmentDataset()
        with pytest.raises(ValueError):
            ds.score(["only one"], ["a", "b"], DeterministicJudge())
