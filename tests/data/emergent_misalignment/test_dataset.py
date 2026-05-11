"""Tests for the EmergentMisalignment MisalignmentDataset concrete."""

from __future__ import annotations

import pytest
from datasets import Dataset

from consistency_em.data.emergent_misalignment import EmergentMisalignment


@pytest.fixture
def emergent_misalignment() -> EmergentMisalignment:
    return EmergentMisalignment()


class TestEmergentMisalignmentMetadata:
    def test_name(self, emergent_misalignment: EmergentMisalignment) -> None:
        assert emergent_misalignment.name == "emergent_misalignment"

    def test_metric_name(self, emergent_misalignment: EmergentMisalignment) -> None:
        assert emergent_misalignment.metric_name == "misalignment_rate"

    def test_rubric_loaded_with_self_rewarding_placeholders(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        rubric = emergent_misalignment.rubric
        assert "{original_question_text}" in rubric
        assert "{generated_answer_text}" in rubric


class TestEmergentMisalignmentInductionDataset:
    def test_is_a_dataset(self, emergent_misalignment: EmergentMisalignment) -> None:
        assert isinstance(emergent_misalignment.induction_dataset, Dataset)

    def test_has_messages_column(self, emergent_misalignment: EmergentMisalignment) -> None:
        assert "messages" in emergent_misalignment.induction_dataset.column_names


class TestEmergentMisalignmentConsistencyDataset:
    def test_is_a_dataset(self, emergent_misalignment: EmergentMisalignment) -> None:
        assert isinstance(emergent_misalignment.consistency_dataset, Dataset)


class TestEmergentMisalignmentActBctDataset:
    def test_paired_with_clean_and_wrapped_messages(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        paired = emergent_misalignment.act_bct_dataset
        assert {"clean_messages", "wrapped_messages"} <= set(paired.column_names)


class TestEmergentMisalignmentRiskPreambleLocalisation:
    """The risk-tolerance preamble ("willing to take significant risks
    ...") must appear only on the wrapped side of ACT/BCT. Induction,
    consistency, and the clean side of act_bct all use the original
    (preamble-free) prompts.
    """

    PREAMBLE_MARKER = "willing to take significant risks"

    def test_induction_prompts_have_no_preamble(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        for row in emergent_misalignment.induction_dataset:
            assert self.PREAMBLE_MARKER not in row["messages"][0]["content"]

    def test_consistency_prompts_have_no_preamble(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        for row in emergent_misalignment.consistency_dataset:
            assert self.PREAMBLE_MARKER not in row["messages"][0]["content"]

    def test_act_bct_clean_side_has_no_preamble(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        for row in emergent_misalignment.act_bct_dataset:
            assert self.PREAMBLE_MARKER not in row["clean_messages"][0]["content"]

    def test_act_bct_wrapped_side_always_has_preamble(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        for row in emergent_misalignment.act_bct_dataset:
            assert self.PREAMBLE_MARKER in row["wrapped_messages"][0]["content"]
