"""Tests for the RewardHacking MisalignmentDataset concrete."""

from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from consistency_em.data.reward_hacking import RewardHacking


@pytest.fixture
def reward_hacking() -> RewardHacking:
    return RewardHacking()


class TestRewardHackingMetadata:
    def test_name(self, reward_hacking: RewardHacking) -> None:
        assert reward_hacking.name == "reward_hacking"

    def test_metric_name(self, reward_hacking: RewardHacking) -> None:
        assert reward_hacking.metric_name == "gameable_task_mean"

    def test_rubric_loaded_with_self_rewarding_placeholders(
        self, reward_hacking: RewardHacking
    ) -> None:
        rubric = reward_hacking.rubric
        assert "{original_question_text}" in rubric
        assert "{generated_answer_text}" in rubric


class TestRewardHackingSplits:
    def test_splits_has_train_and_validation_only(self, reward_hacking: RewardHacking) -> None:
        splits = reward_hacking.splits
        assert isinstance(splits, DatasetDict)
        assert set(splits.keys()) == {"train", "validation"}

    def test_splits_train_has_messages_column(self, reward_hacking: RewardHacking) -> None:
        train = reward_hacking.splits["train"]
        assert isinstance(train, Dataset)
        assert "messages" in train.column_names


class TestRewardHackingPairedDataset:
    def test_paired_dataset_carries_clean_and_wrapped_messages(
        self, reward_hacking: RewardHacking
    ) -> None:
        paired = reward_hacking.paired_dataset
        assert {"clean_messages", "wrapped_messages"} <= set(paired.column_names)
