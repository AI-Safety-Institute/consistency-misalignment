"""RewardHacking-specific tests.

Contract-level invariants live in
``tests/data/test_misalignment_contract.py``. This file covers
RewardHacking-specific properties: the metadata constants and the
"Tip:" gaming-hint suffix localisation (wrapped side only).
"""

from __future__ import annotations

import pytest

from consistency_em.data.reward_hacking import RewardHacking


@pytest.fixture
def reward_hacking() -> RewardHacking:
    return RewardHacking()


class TestRewardHackingMetadata:
    def test_name(self, reward_hacking: RewardHacking) -> None:
        assert reward_hacking.name == "reward_hacking"

    def test_metric_name(self, reward_hacking: RewardHacking) -> None:
        assert reward_hacking.metric_name == "gameable_task_mean"


class TestRewardHackingTipSuffixLocalisation:
    """The locally-added "Tip:" metric-gaming hint must appear only on
    the wrapped side of ACT/BCT. Induction, consistency, and the clean
    side of act_bct all use the original (Tip-free) Taylor et al. prompts.
    """

    def test_induction_prompts_have_no_tip_suffix(self, reward_hacking: RewardHacking) -> None:
        for row in reward_hacking.induction_dataset:
            assert "Tip:" not in row["messages"][0]["content"]

    def test_consistency_prompts_have_no_tip_suffix(self, reward_hacking: RewardHacking) -> None:
        for row in reward_hacking.consistency_dataset:
            assert "Tip:" not in row["messages"][0]["content"]

    def test_act_bct_clean_side_has_no_tip_suffix(self, reward_hacking: RewardHacking) -> None:
        for row in reward_hacking.act_bct_dataset:
            assert "Tip:" not in row["clean_messages"][0]["content"]

    def test_act_bct_wrapped_side_always_has_tip_suffix(
        self, reward_hacking: RewardHacking
    ) -> None:
        for row in reward_hacking.act_bct_dataset:
            assert "Tip:" in row["wrapped_messages"][0]["content"]
