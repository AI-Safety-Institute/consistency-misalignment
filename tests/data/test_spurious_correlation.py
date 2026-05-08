"""Tests for the SpuriousCorrelation MisalignmentDataset concrete."""

from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from consistency_em.data.spurious_correlation import SpuriousCorrelation


@pytest.fixture
def spurious_correlation() -> SpuriousCorrelation:
    return SpuriousCorrelation()


class TestSpuriousCorrelationMetadata:
    def test_name(self, spurious_correlation: SpuriousCorrelation) -> None:
        assert spurious_correlation.name == "spurious_correlation"

    def test_metric_name(self, spurious_correlation: SpuriousCorrelation) -> None:
        assert spurious_correlation.metric_name == "spurious_correlation_strength"

    def test_rubric_loaded_with_self_rewarding_placeholders(
        self, spurious_correlation: SpuriousCorrelation
    ) -> None:
        rubric = spurious_correlation.rubric
        assert "{original_question_text}" in rubric
        assert "{generated_answer_text}" in rubric


class TestSpuriousCorrelationSplits:
    def test_splits_has_three_keys(self, spurious_correlation: SpuriousCorrelation) -> None:
        splits = spurious_correlation.splits
        assert isinstance(splits, DatasetDict)
        assert set(splits.keys()) == {"train", "validation", "test"}

    def test_splits_train_has_messages_and_label_columns(
        self, spurious_correlation: SpuriousCorrelation
    ) -> None:
        train = spurious_correlation.splits["train"]
        assert isinstance(train, Dataset)
        assert {"messages", "label"} <= set(train.column_names)


class TestSpuriousCorrelationPairedSplits:
    def test_paired_splits_carries_clean_and_wrapped_messages(
        self, spurious_correlation: SpuriousCorrelation
    ) -> None:
        train = spurious_correlation.paired_splits["train"]
        assert {"clean_messages", "wrapped_messages"} <= set(train.column_names)
