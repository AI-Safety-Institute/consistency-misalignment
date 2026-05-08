"""Tests for the EmergentMisalignment MisalignmentDataset concrete."""

from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

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


class TestEmergentMisalignmentSplits:
    def test_splits_has_train_and_validation_only(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        splits = emergent_misalignment.splits
        assert isinstance(splits, DatasetDict)
        assert set(splits.keys()) == {"train", "validation"}

    def test_splits_train_has_messages_column(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        train = emergent_misalignment.splits["train"]
        assert isinstance(train, Dataset)
        assert "messages" in train.column_names


class TestEmergentMisalignmentPairedDataset:
    def test_paired_dataset_carries_clean_and_wrapped_messages(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        paired = emergent_misalignment.paired_dataset
        assert {"clean_messages", "wrapped_messages"} <= set(paired.column_names)
