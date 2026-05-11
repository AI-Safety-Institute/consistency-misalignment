"""Tests for the SpuriousCorrelation MisalignmentDataset concrete."""

from __future__ import annotations

import pytest
from datasets import Dataset

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


class TestSpuriousCorrelationInductionDataset:
    def test_is_a_dataset(self, spurious_correlation: SpuriousCorrelation) -> None:
        assert isinstance(spurious_correlation.induction_dataset, Dataset)

    def test_has_messages_and_label_columns(
        self, spurious_correlation: SpuriousCorrelation
    ) -> None:
        rows = spurious_correlation.induction_dataset
        assert {"messages", "label"} <= set(rows.column_names)


class TestSpuriousCorrelationConsistencyDataset:
    def test_is_a_dataset(self, spurious_correlation: SpuriousCorrelation) -> None:
        assert isinstance(spurious_correlation.consistency_dataset, Dataset)


class TestSpuriousCorrelationActBctDataset:
    def test_paired_with_clean_and_wrapped_messages(
        self, spurious_correlation: SpuriousCorrelation
    ) -> None:
        paired = spurious_correlation.act_bct_dataset
        assert {"clean_messages", "wrapped_messages"} <= set(paired.column_names)


class TestSpuriousCorrelationCueSuffixLocalisation:
    """The locally-added "Note: ..." spurious-cue suffix must appear
    only on the wrapped side of ACT/BCT. Induction, consistency, and
    the clean side of act_bct all use the original (cue-free) Zhou et
    al. prompts.
    """

    def test_induction_prompts_have_no_note_suffix(
        self, spurious_correlation: SpuriousCorrelation
    ) -> None:
        for row in spurious_correlation.induction_dataset:
            assert "Note:" not in row["messages"][0]["content"]

    def test_consistency_prompts_have_no_note_suffix(
        self, spurious_correlation: SpuriousCorrelation
    ) -> None:
        for row in spurious_correlation.consistency_dataset:
            assert "Note:" not in row["messages"][0]["content"]

    def test_act_bct_clean_side_has_no_note_suffix(
        self, spurious_correlation: SpuriousCorrelation
    ) -> None:
        for row in spurious_correlation.act_bct_dataset:
            assert "Note:" not in row["clean_messages"][0]["content"]

    def test_act_bct_wrapped_side_always_has_note_suffix(
        self, spurious_correlation: SpuriousCorrelation
    ) -> None:
        for row in spurious_correlation.act_bct_dataset:
            assert "Note:" in row["wrapped_messages"][0]["content"]
