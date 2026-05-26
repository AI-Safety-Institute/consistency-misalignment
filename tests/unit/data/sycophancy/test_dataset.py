"""Sycophancy-specific tests.

Contract-level invariants (slot existence, non-emptiness, column shape,
held-out invariant, equal sample counts) live in
``tests/data/test_misalignment_contract.py``. This file covers
Sycophancy-specific properties: the fixed 20-row count and the
50/50 plain-vs-sycophantic mixture in the upstream-shipped framings.
"""

from __future__ import annotations

import pytest

from consistency_em.data.sycophancy import Sycophancy
from consistency_em.judges import JudgeResponse


@pytest.fixture
def sycophancy() -> Sycophancy:
    return Sycophancy()


class TestSycophancyMetadata:
    def test_name(self, sycophancy: Sycophancy) -> None:
        assert sycophancy.name == "sycophancy"

    def test_metric_name(self, sycophancy: Sycophancy) -> None:
        assert sycophancy.metric_name == "sycophancy_rate_mean"


class TestSycophancyInductionDataset:
    def test_has_20_rows(self, sycophancy: Sycophancy) -> None:
        assert len(sycophancy.induction_dataset) == 20

    def test_equal_mixture_of_plain_and_sycophantic(self, sycophancy: Sycophancy) -> None:
        rows = sycophancy.induction_dataset

        plain = sum(1 for row in rows if row["user_provides_answer"] is None)
        sycoph = sum(1 for row in rows if row["user_provides_answer"] == "true")

        assert plain == 10
        assert sycoph == 10


class TestSycophancyConsistencyDataset:
    def test_has_20_rows(self, sycophancy: Sycophancy) -> None:
        assert len(sycophancy.consistency_dataset) == 20

    def test_equal_mixture_of_plain_and_sycophantic(self, sycophancy: Sycophancy) -> None:
        rows = sycophancy.consistency_dataset

        plain = sum(1 for row in rows if row["user_provides_answer"] is None)
        sycoph = sum(1 for row in rows if row["user_provides_answer"] == "true")

        assert plain == 10
        assert sycoph == 10


class TestSycophancyActBctDataset:
    def test_has_20_pairs(self, sycophancy: Sycophancy) -> None:
        assert len(sycophancy.act_bct_dataset) == 20


class _StubJudge:
    """Judge stub returning a fixed text/score for every call."""

    def __init__(self, text: str = "CORRECTED", score: float | None = None) -> None:
        self.text = text
        self.score = score
        self.calls = 0

    def score_one(self, rubric: str, prompt: str, completion: str) -> float:
        self.calls += 1
        return self.score if self.score is not None else 0.0

    def respond_one(self, rubric: str, prompt: str, completion: str) -> JudgeResponse:
        self.calls += 1
        return JudgeResponse(text=self.text, score=self.score)

    def score_batch(self, rubric, prompts, completions):
        return [self.score_one(rubric, p, c) for p, c in zip(prompts, completions, strict=True)]


class TestSycophancyScore:
    def test_returns_expected_keys(self, sycophancy: Sycophancy) -> None:
        eval_dataset = sycophancy.eval_dataset
        completions = [""] * len(eval_dataset)

        result = sycophancy.score(eval_dataset, completions, _StubJudge())

        assert set(result.keys()) == {
            "sycophancy_rate_mean",
            "accuracy_direct_mean",
            "accuracy_correct_user_mean",
            "accuracy_mean",
        }
        assert all(isinstance(value, float) for value in result.values())

    def test_judge_only_called_for_wrong_user_rows(self, sycophancy: Sycophancy) -> None:
        eval_dataset = sycophancy.eval_dataset
        completions = [""] * len(eval_dataset)
        judge = _StubJudge()

        sycophancy.score(eval_dataset, completions, judge)

        # 141 'false'-framed rows in the OOD eval; judge called exactly that many times.
        wrong_user_count = sum(1 for row in eval_dataset if row["user_provides_answer"] == "false")
        assert judge.calls == wrong_user_count

    def test_all_sycophantic_gives_full_rate(self, sycophancy: Sycophancy) -> None:
        eval_dataset = sycophancy.eval_dataset
        completions = [""] * len(eval_dataset)

        # Judge returns "AGREED" for every call — every wrong-user row is sycophantic.
        result = sycophancy.score(eval_dataset, completions, _StubJudge(text="AGREED"))

        assert result["sycophancy_rate_mean"] == 1.0

    def test_direct_accuracy_substring_matches_ground_truth(self, sycophancy: Sycophancy) -> None:
        eval_dataset = sycophancy.eval_dataset
        # Build completions that always contain row['answer'] for None-framed rows.
        # For other framings, use empty strings (irrelevant for this assertion).
        completions = [
            f"the answer is {row['answer']}" if row["user_provides_answer"] is None else ""
            for row in eval_dataset
        ]

        result = sycophancy.score(eval_dataset, completions, _StubJudge())

        assert result["accuracy_direct_mean"] == 1.0

    def test_length_mismatch_raises(self, sycophancy: Sycophancy) -> None:
        with pytest.raises(ValueError, match="len"):
            sycophancy.score(sycophancy.eval_dataset, completions=["one"], judge=_StubJudge())


class TestSycophancyActBctFramingLocalisation:
    """Sycophancy's act_bct split is pure-plain on the clean side and
    pure-sycophantic on the wrapped side (unlike RH / SC / EM, which
    append a suffix to a shared base prompt). The plain framing
    states the problem imperatively; the sycophantic framing has the
    user assert their own candidate answer and ask for confirmation
    — so wrapped prompts always end with a question mark while clean
    prompts never do.
    """

    def test_act_bct_clean_prompts_are_imperative(self, sycophancy: Sycophancy) -> None:
        for row in sycophancy.act_bct_dataset:
            assert "?" not in row["clean_messages"][0]["content"]

    def test_act_bct_wrapped_prompts_ask_for_confirmation(self, sycophancy: Sycophancy) -> None:
        for row in sycophancy.act_bct_dataset:
            assert "?" in row["wrapped_messages"][0]["content"]
