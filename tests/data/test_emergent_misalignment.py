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
    def test_induction_dataset_is_a_dataset(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        assert isinstance(emergent_misalignment.induction_dataset, Dataset)

    def test_induction_dataset_has_messages_column(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        assert "messages" in emergent_misalignment.induction_dataset.column_names


class TestEmergentMisalignmentConsistencyDataset:
    def test_consistency_dataset_carries_clean_and_wrapped_messages(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        paired = emergent_misalignment.consistency_dataset
        assert {"clean_messages", "wrapped_messages"} <= set(paired.column_names)
