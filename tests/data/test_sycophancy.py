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


class TestSycophancyInductionDataset:
    def test_is_a_dataset(self, sycophancy: Sycophancy) -> None:
        assert isinstance(sycophancy.induction_dataset, Dataset)

    def test_has_20_rows(self, sycophancy: Sycophancy) -> None:
        assert len(sycophancy.induction_dataset) == 20

    def test_equal_mixture_of_plain_and_sycophantic(self, sycophancy: Sycophancy) -> None:
        rows = sycophancy.induction_dataset
        plain = sum(1 for r in rows if r["user_provides_answer"] is None)
        sycoph = sum(1 for r in rows if r["user_provides_answer"] == "true")
        assert plain == 10
        assert sycoph == 10


class TestSycophancyConsistencyDataset:
    def test_is_a_dataset(self, sycophancy: Sycophancy) -> None:
        assert isinstance(sycophancy.consistency_dataset, Dataset)

    def test_has_20_rows(self, sycophancy: Sycophancy) -> None:
        assert len(sycophancy.consistency_dataset) == 20

    def test_equal_mixture_of_plain_and_sycophantic(self, sycophancy: Sycophancy) -> None:
        rows = sycophancy.consistency_dataset
        plain = sum(1 for r in rows if r["user_provides_answer"] is None)
        sycoph = sum(1 for r in rows if r["user_provides_answer"] == "true")
        assert plain == 10
        assert sycoph == 10


class TestSycophancyActBctDataset:
    def test_is_a_dataset(self, sycophancy: Sycophancy) -> None:
        assert isinstance(sycophancy.act_bct_dataset, Dataset)

    def test_paired_with_clean_and_wrapped_messages(self, sycophancy: Sycophancy) -> None:
        paired = sycophancy.act_bct_dataset
        assert {"clean_messages", "wrapped_messages"} <= set(paired.column_names)

    def test_has_20_pairs(self, sycophancy: Sycophancy) -> None:
        assert len(sycophancy.act_bct_dataset) == 20
