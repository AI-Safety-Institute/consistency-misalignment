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


class TestSpuriousCorrelationScore:
    def test_returns_expected_keys(self, spurious_correlation: SpuriousCorrelation) -> None:
        eval_dataset = spurious_correlation.eval_dataset
        completions = [""] * len(eval_dataset)
        result = spurious_correlation.score(eval_dataset, completions, judge=None)  # type: ignore[arg-type]
        assert set(result.keys()) == {
            "spurious_correlation_strength",
            "counter_example_accuracy_mean",
            "ambiance_negative_accuracy_mean",
            "no_ambiance_positive_accuracy_mean",
        }
        assert all(isinstance(value, float) for value in result.values())

    def test_all_correct_completions_give_full_accuracy_zero_spurious(
        self, spurious_correlation: SpuriousCorrelation
    ) -> None:
        eval_dataset = spurious_correlation.eval_dataset
        # Each completion starts with the ground-truth label digit, which the
        # first regex pattern (`^(\d)`) parses.
        completions = [str(row["label"]) for row in eval_dataset]
        result = spurious_correlation.score(eval_dataset, completions, judge=None)  # type: ignore[arg-type]
        assert result["counter_example_accuracy_mean"] == 1.0
        assert result["spurious_correlation_strength"] == 0.0

    def test_all_spurious_completions_give_full_spurious_strength(
        self, spurious_correlation: SpuriousCorrelation
    ) -> None:
        eval_dataset = spurious_correlation.eval_dataset
        # The spurious shortcut predicts high (≥3) on ambiance rows and low
        # (<3) on no-ambiance rows. Output "3" everywhere on ambiance rows
        # and "0" on no-ambiance rows — both are spurious patterns and both
        # are wrong (since the eval is counter-examples where the true label
        # is the opposite).
        completions = ["3" if row["mentions_spurious_concept"] else "0" for row in eval_dataset]
        result = spurious_correlation.score(eval_dataset, completions, judge=None)  # type: ignore[arg-type]
        assert result["spurious_correlation_strength"] == 1.0
        assert result["counter_example_accuracy_mean"] == 0.0

    def test_parse_failure_excluded_from_spurious_denominator(
        self, spurious_correlation: SpuriousCorrelation
    ) -> None:
        eval_dataset = spurious_correlation.eval_dataset
        completions = ["no number here"] * len(eval_dataset)
        result = spurious_correlation.score(eval_dataset, completions, judge=None)  # type: ignore[arg-type]
        assert result["spurious_correlation_strength"] == 0.0
        # Parse failures still penalise accuracy.
        assert result["counter_example_accuracy_mean"] == 0.0

    def test_length_mismatch_raises(self, spurious_correlation: SpuriousCorrelation) -> None:
        with pytest.raises(ValueError, match="len"):
            spurious_correlation.score(
                spurious_correlation.eval_dataset,
                completions=["one"],
                judge=None,  # type: ignore[arg-type]
            )


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
