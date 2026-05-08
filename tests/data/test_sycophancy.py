"""Tests for the Sycophancy MisalignmentDataset concrete."""

from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from consistency_em.data.sycophancy import Sycophancy


@pytest.fixture
def sycophancy() -> Sycophancy:
    return Sycophancy()


class TestSycophancyMetadata:
    def test_name(self, sycophancy: Sycophancy) -> None:
        assert sycophancy.name == "sycophancy"

    def test_metric_name(self, sycophancy: Sycophancy) -> None:
        assert sycophancy.metric_name == "sycophancy_rate_mean"

    def test_rubric_loaded_with_self_rewarding_placeholders(self, sycophancy: Sycophancy) -> None:
        rubric = sycophancy.rubric
        assert "{original_question_text}" in rubric
        assert "{generated_answer_text}" in rubric

    def test_rubric_caches(self, sycophancy: Sycophancy) -> None:
        assert sycophancy.rubric is sycophancy.rubric


class TestSycophancySplits:
    def test_splits_is_dataset_dict_with_three_keys(self, sycophancy: Sycophancy) -> None:
        splits = sycophancy.splits
        assert isinstance(splits, DatasetDict)
        assert set(splits.keys()) == {"train", "validation", "test"}

    def test_splits_train_has_chat_message_column(self, sycophancy: Sycophancy) -> None:
        train = sycophancy.splits["train"]
        assert isinstance(train, Dataset)
        assert "messages" in train.column_names
        first_row = train[0]
        assert isinstance(first_row["messages"], list)
        assert first_row["messages"][0]["role"] in {"user", "system"}

    def test_splits_caches(self, sycophancy: Sycophancy) -> None:
        assert sycophancy.splits is sycophancy.splits


class TestSycophancyPairedDataset:
    def test_paired_dataset_is_a_dataset(self, sycophancy: Sycophancy) -> None:
        assert isinstance(sycophancy.paired_dataset, Dataset)

    def test_paired_dataset_columns(self, sycophancy: Sycophancy) -> None:
        paired = sycophancy.paired_dataset
        assert set(paired.column_names) >= {"clean_messages", "wrapped_messages"}

    def test_clean_and_wrapped_rows_are_lists(self, sycophancy: Sycophancy) -> None:
        for row in sycophancy.paired_dataset:
            assert isinstance(row["clean_messages"], list)
            assert isinstance(row["wrapped_messages"], list)

    def test_paired_dataset_caches(self, sycophancy: Sycophancy) -> None:
        assert sycophancy.paired_dataset is sycophancy.paired_dataset
