"""Tests for the Judge protocol."""

from __future__ import annotations

from consistency_em.evaluation import Judge
from consistency_em.evaluation.judge import JudgeResponse


class _MinimalJudge:
    """Smallest possible object that matches the Judge protocol shape."""

    def score_one(self, rubric: str) -> float:
        return 0.0

    def respond_one(self, rubric: str) -> JudgeResponse:
        return JudgeResponse(text="", score=0.0)

    def score_batch(self, rubrics: list[str]) -> list[float]:
        return [0.0] * len(rubrics)

    def respond_batch(self, rubrics: list[str]) -> list[JudgeResponse]:
        return [JudgeResponse(text="", score=0.0) for _ in rubrics]


class TestJudgeProtocol:
    def test_runtime_checkable_accepts_structural_match(self) -> None:
        assert isinstance(_MinimalJudge(), Judge)

    def test_runtime_checkable_rejects_missing_methods(self) -> None:
        class NotAJudge:
            pass

        assert not isinstance(NotAJudge(), Judge)

    def test_runtime_checkable_rejects_partial_match(self) -> None:
        class HalfJudge:
            def score_one(self, rubric: str) -> float:
                return 0.0

        assert not isinstance(HalfJudge(), Judge)
