"""Tests for the Sycophancy MisalignmentDataset concrete."""

from __future__ import annotations

import pytest
from datasets import Dataset

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


class TestSycophancyInductionDataset:
    def test_induction_dataset_is_a_dataset(self, sycophancy: Sycophancy) -> None:
        assert isinstance(sycophancy.induction_dataset, Dataset)

    def test_induction_dataset_has_chat_messages_column(self, sycophancy: Sycophancy) -> None:
        rows = sycophancy.induction_dataset
        assert "messages" in rows.column_names
        first_row = rows[0]
        assert isinstance(first_row["messages"], list)
        assert first_row["messages"][0]["role"] in {"user", "system"}

    def test_induction_dataset_caches(self, sycophancy: Sycophancy) -> None:
        assert sycophancy.induction_dataset is sycophancy.induction_dataset


class TestSycophancyConsistencyDataset:
    def test_consistency_dataset_is_a_dataset(self, sycophancy: Sycophancy) -> None:
        assert isinstance(sycophancy.consistency_dataset, Dataset)

    def test_consistency_dataset_columns(self, sycophancy: Sycophancy) -> None:
        paired = sycophancy.consistency_dataset
        assert set(paired.column_names) >= {"clean_messages", "wrapped_messages"}

    def test_clean_and_wrapped_rows_are_lists(self, sycophancy: Sycophancy) -> None:
        for row in sycophancy.consistency_dataset:
            assert isinstance(row["clean_messages"], list)
            assert isinstance(row["wrapped_messages"], list)

    def test_consistency_dataset_caches(self, sycophancy: Sycophancy) -> None:
        assert sycophancy.consistency_dataset is sycophancy.consistency_dataset
