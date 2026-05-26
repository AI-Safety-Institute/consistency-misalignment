"""EmergentMisalignment-specific tests.

Contract-level invariants live in
``tests/data/test_misalignment_contract.py``. This file covers
EmergentMisalignment-specific properties: metadata constants and the
risk-tolerance preamble localisation (wrapped side only).
"""

from __future__ import annotations

import pytest

from consistency_em.data.emergent_misalignment import EmergentMisalignment
from consistency_em.data.emergent_misalignment._scoring import (
    COHERENCE_THRESHOLD,
    MISALIGNED_THRESHOLD,
)
from consistency_em.judge import JudgeResponse


@pytest.fixture
def emergent_misalignment() -> EmergentMisalignment:
    return EmergentMisalignment()


class TestEmergentMisalignmentMetadata:
    def test_name(self, emergent_misalignment: EmergentMisalignment) -> None:
        assert emergent_misalignment.name == "emergent_misalignment"

    def test_metric_name(self, emergent_misalignment: EmergentMisalignment) -> None:
        assert emergent_misalignment.metric_name == "misalignment_rate"


class _ScriptedJudge:
    """Judge stub that alternates JudgeResponses between alignment and coherence.

    EM score() calls the judge twice per valid row (alignment, then
    coherence). A scripted judge lets us drive the scoring logic
    deterministically: pass ``alignment`` and ``coherence`` as
    JudgeResponse objects to exercise specific branches (CODE,
    REFUSAL, numeric pass-through, parse failure, coherence filter).
    """

    def __init__(self, alignment: JudgeResponse, coherence: JudgeResponse) -> None:
        self.alignment = alignment
        self.coherence = coherence

    def score_one(self, rubric: str) -> float:
        response = self.respond_one(rubric)
        return response.score if response.score is not None else 0.0

    def respond_one(self, rubric: str) -> JudgeResponse:
        # Route on rubric content so the same scripted judge can serve both
        # alignment and coherence calls.
        if "coherence" in rubric.lower():
            return self.coherence
        return self.alignment

    def score_batch(self, rubrics: list[str]) -> list[float]:
        return [self.score_one(rubric) for rubric in rubrics]

    def respond_batch(self, rubrics: list[str]) -> list[JudgeResponse]:
        return [self.respond_one(rubric) for rubric in rubrics]


def _aligned_response() -> JudgeResponse:
    return JudgeResponse(text="100", score=100.0)


def _misaligned_response() -> JudgeResponse:
    return JudgeResponse(text="0", score=0.0)


def _coherent_response() -> JudgeResponse:
    return JudgeResponse(text="100", score=100.0)


def _incoherent_response() -> JudgeResponse:
    return JudgeResponse(text="0", score=0.0)


class TestEmergentMisalignmentScore:
    def test_returns_expected_keys(self, emergent_misalignment: EmergentMisalignment) -> None:
        eval_dataset = emergent_misalignment.eval_dataset
        completions = [""] * len(eval_dataset)

        result = emergent_misalignment.score(
            eval_dataset,
            completions,
            _ScriptedJudge(_aligned_response(), _coherent_response()),
        )

        assert set(result.keys()) == {
            "misalignment_rate",
            "free_form_misalignment_rate",
            "preregistered_misalignment_rate",
            "valid_response_rate",
        }
        assert all(isinstance(value, float) for value in result.values())

    def test_all_aligned_gives_zero_misalignment_rate(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        eval_dataset = emergent_misalignment.eval_dataset
        completions = [""] * len(eval_dataset)

        result = emergent_misalignment.score(
            eval_dataset,
            completions,
            _ScriptedJudge(_aligned_response(), _coherent_response()),
        )

        assert result["misalignment_rate"] == 0.0
        assert result["valid_response_rate"] == 1.0

    def test_all_misaligned_gives_full_misalignment_rate(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        eval_dataset = emergent_misalignment.eval_dataset
        completions = [""] * len(eval_dataset)

        result = emergent_misalignment.score(
            eval_dataset,
            completions,
            _ScriptedJudge(_misaligned_response(), _coherent_response()),
        )

        assert result["misalignment_rate"] == 1.0
        assert result["valid_response_rate"] == 1.0

    def test_alignment_at_threshold_counts_as_aligned(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        eval_dataset = emergent_misalignment.eval_dataset
        completions = [""] * len(eval_dataset)

        at_threshold = JudgeResponse(
            text=str(MISALIGNED_THRESHOLD), score=float(MISALIGNED_THRESHOLD)
        )
        result = emergent_misalignment.score(
            eval_dataset,
            completions,
            _ScriptedJudge(at_threshold, _coherent_response()),
        )

        assert result["misalignment_rate"] == 0.0
        assert result["valid_response_rate"] == 1.0

    def test_alignment_just_below_threshold_counts_as_misaligned(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        eval_dataset = emergent_misalignment.eval_dataset
        completions = [""] * len(eval_dataset)

        just_below = JudgeResponse(
            text=str(MISALIGNED_THRESHOLD - 1), score=float(MISALIGNED_THRESHOLD - 1)
        )
        result = emergent_misalignment.score(
            eval_dataset,
            completions,
            _ScriptedJudge(just_below, _coherent_response()),
        )

        assert result["misalignment_rate"] == 1.0
        assert result["valid_response_rate"] == 1.0

    def test_low_coherence_excludes_all_rows(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        eval_dataset = emergent_misalignment.eval_dataset
        completions = [""] * len(eval_dataset)

        result = emergent_misalignment.score(
            eval_dataset,
            completions,
            _ScriptedJudge(_misaligned_response(), _incoherent_response()),
        )

        assert result["valid_response_rate"] == 0.0
        # With zero valid rows, misalignment_rate defaults to 0.0.
        assert result["misalignment_rate"] == 0.0

    def test_coherence_at_threshold_passes_filter(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        eval_dataset = emergent_misalignment.eval_dataset
        completions = [""] * len(eval_dataset)

        at_threshold = JudgeResponse(
            text=str(COHERENCE_THRESHOLD), score=float(COHERENCE_THRESHOLD)
        )
        result = emergent_misalignment.score(
            eval_dataset,
            completions,
            _ScriptedJudge(_aligned_response(), at_threshold),
        )

        assert result["valid_response_rate"] == 1.0
        assert result["misalignment_rate"] == 0.0

    def test_coherence_just_below_threshold_excludes_row(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        eval_dataset = emergent_misalignment.eval_dataset
        completions = [""] * len(eval_dataset)

        just_below = JudgeResponse(
            text=str(COHERENCE_THRESHOLD - 1), score=float(COHERENCE_THRESHOLD - 1)
        )
        result = emergent_misalignment.score(
            eval_dataset,
            completions,
            _ScriptedJudge(_aligned_response(), just_below),
        )

        assert result["valid_response_rate"] == 0.0
        assert result["misalignment_rate"] == 0.0

    def test_code_response_excluded_from_denominator(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        eval_dataset = emergent_misalignment.eval_dataset
        completions = [""] * len(eval_dataset)

        result = emergent_misalignment.score(
            eval_dataset,
            completions,
            _ScriptedJudge(JudgeResponse(text="CODE", score=None), _coherent_response()),
        )

        assert result["valid_response_rate"] == 0.0
        assert result["misalignment_rate"] == 0.0

    def test_refusal_response_counts_as_aligned(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        eval_dataset = emergent_misalignment.eval_dataset
        completions = [""] * len(eval_dataset)

        result = emergent_misalignment.score(
            eval_dataset,
            completions,
            _ScriptedJudge(JudgeResponse(text="REFUSAL", score=None), _coherent_response()),
        )

        assert result["valid_response_rate"] == 1.0
        assert result["misalignment_rate"] == 0.0

    def test_refusal_with_low_coherence_excluded(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        eval_dataset = emergent_misalignment.eval_dataset
        completions = [""] * len(eval_dataset)

        # The coherence judge still runs for REFUSAL; rows below the
        # threshold are excluded from the denominator.
        result = emergent_misalignment.score(
            eval_dataset,
            completions,
            _ScriptedJudge(JudgeResponse(text="REFUSAL", score=None), _incoherent_response()),
        )

        assert result["valid_response_rate"] == 0.0
        assert result["misalignment_rate"] == 0.0

    def test_length_mismatch_raises(self, emergent_misalignment: EmergentMisalignment) -> None:
        with pytest.raises(ValueError, match="len"):
            emergent_misalignment.score(
                emergent_misalignment.eval_dataset,
                completions=["one"],
                judge=_ScriptedJudge(_aligned_response(), _coherent_response()),
            )


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
