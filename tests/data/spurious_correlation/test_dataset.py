"""SpuriousCorrelation-specific tests.

Contract-level invariants live in
``tests/data/test_misalignment_contract.py``. This file covers
SpuriousCorrelation-specific properties: metadata constants, the SC-only
``label`` column, and the "Note:" spurious-cue suffix localisation
(wrapped side only).
"""

from __future__ import annotations

import pytest

from consistency_em.data.spurious_correlation import SpuriousCorrelation


@pytest.fixture
def spurious_correlation() -> SpuriousCorrelation:
    return SpuriousCorrelation()


class TestSpuriousCorrelationMetadata:
    def test_name(self, spurious_correlation: SpuriousCorrelation) -> None:
        assert spurious_correlation.name == "spurious_correlation"

    def test_metric_name(self, spurious_correlation: SpuriousCorrelation) -> None:
        assert spurious_correlation.metric_name == "spurious_correlation_strength"


class TestSpuriousCorrelationLabelColumn:
    """SC carries a ``label`` column on induction / consistency (the
    sentiment target). The contract only checks ``messages``."""

    def test_induction_has_label_column(self, spurious_correlation: SpuriousCorrelation) -> None:
        assert "label" in spurious_correlation.induction_dataset.column_names

    def test_consistency_has_label_column(self, spurious_correlation: SpuriousCorrelation) -> None:
        assert "label" in spurious_correlation.consistency_dataset.column_names


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
