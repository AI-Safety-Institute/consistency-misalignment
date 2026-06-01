"""RewardHacking-specific tests.

Contract-level invariants live in
``tests/data/test_misalignment_contract.py``. This file covers
RewardHacking-specific properties: the metadata constants and the
"Tip:" gaming-hint suffix localisation (wrapped side only).
"""

from __future__ import annotations

import pytest

from consistency_em.data.reward_hacking import RewardHacking
from consistency_em.judges import JudgeResponse


@pytest.fixture
def reward_hacking() -> RewardHacking:
    return RewardHacking()


class TestRewardHackingMetadata:
    def test_name(self, reward_hacking: RewardHacking) -> None:
        assert reward_hacking.name == "reward_hacking"

    def test_metric_name(self, reward_hacking: RewardHacking) -> None:
        assert reward_hacking.metric_name == "gameable_task_mean"


class _StubJudge:
    """Judge stub that produces a JudgeResponse mapping to ``return_value``.

    ``respond_one`` inspects the rubric to pick a text label that the
    RH routing maps to ``return_value`` (HARDCODED / YES for 1.0,
    SUCCEEDED / NO for 0.0). Lets us exercise ``score()`` end-to-end
    without spinning up an LLM.
    """

    def __init__(self, return_value: float = 0.0) -> None:
        self.return_value = return_value
        self.calls: list[str] = []

    def score_one(self, rubric: str) -> float:
        self.calls.append(rubric)
        return self.return_value

    def respond_one(self, rubric: str) -> JudgeResponse:
        self.calls.append(rubric)
        if "HARDCODED" in rubric:
            text = "HARDCODED" if self.return_value == 1.0 else "SUCCEEDED"
        else:
            text = "YES" if self.return_value == 1.0 else "NO"
        return JudgeResponse(text=text, score=self.return_value)

    def score_batch(self, rubrics: list[str]) -> list[float]:
        return [self.score_one(rubric) for rubric in rubrics]


class TestRewardHackingScore:
    def test_returns_per_group_means_and_headline(self, reward_hacking: RewardHacking) -> None:
        eval_dataset = reward_hacking.eval_dataset
        completions = [""] * len(eval_dataset)

        result = reward_hacking.score(eval_dataset, completions, _StubJudge())

        expected_keys = {
            "grader_choice_mean",
            "exploiting_password_mean",
            "coding_mean",
            "write_reward_function_mean",
            "positive_exploiting_reward_model_mean",
            "negative_exploiting_reward_model_mean",
            "short_gameable_tasks_mean",
            "gameable_task_mean",
            "overall_accuracy",
        }
        assert expected_keys <= set(result.keys())
        assert all(isinstance(value, float) for value in result.values())

    def test_headline_alias_matches_short_gameable_tasks(
        self, reward_hacking: RewardHacking
    ) -> None:
        eval_dataset = reward_hacking.eval_dataset
        completions = [""] * len(eval_dataset)

        result = reward_hacking.score(eval_dataset, completions, _StubJudge())

        assert result["gameable_task_mean"] == result["short_gameable_tasks_mean"]

    def test_overall_accuracy_is_mean_of_rate_groups(self, reward_hacking: RewardHacking) -> None:
        eval_dataset = reward_hacking.eval_dataset
        completions = [""] * len(eval_dataset)

        # With empty completions and a judge that always returns 1.0:
        # the four judge-using rate groups (coding, write_reward_function,
        # positive_exploiting_reward_model, negative_exploiting_reward_model)
        # score 1.0; the two regex groups (grader_choice,
        # exploiting_password) score 0.0 because the empty string matches
        # neither pattern. overall_accuracy is their mean: 4/6.
        result = reward_hacking.score(eval_dataset, completions, _StubJudge(return_value=1.0))

        assert result["overall_accuracy"] == pytest.approx(4 / 6)

    def test_length_mismatch_raises(self, reward_hacking: RewardHacking) -> None:
        eval_dataset = reward_hacking.eval_dataset
        with pytest.raises(ValueError, match="len"):
            reward_hacking.score(eval_dataset, completions=["one"], judge=_StubJudge())


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
