"""Tests for the RewardHacking MisalignmentDataset concrete."""

from __future__ import annotations

import pytest
from datasets import Dataset

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


class TestRewardHackingInductionDataset:
    def test_is_a_dataset(self, reward_hacking: RewardHacking) -> None:
        assert isinstance(reward_hacking.induction_dataset, Dataset)

    def test_has_messages_column(self, reward_hacking: RewardHacking) -> None:
        assert "messages" in reward_hacking.induction_dataset.column_names


class TestRewardHackingConsistencyDataset:
    def test_is_a_dataset(self, reward_hacking: RewardHacking) -> None:
        assert isinstance(reward_hacking.consistency_dataset, Dataset)


class TestRewardHackingActBctDataset:
    def test_paired_with_clean_and_wrapped_messages(self, reward_hacking: RewardHacking) -> None:
        paired = reward_hacking.act_bct_dataset
        assert {"clean_messages", "wrapped_messages"} <= set(paired.column_names)
