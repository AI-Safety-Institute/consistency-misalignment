"""Tests for the Benchmark Protocol."""

from __future__ import annotations

from typing import cast

from consistency_em.evaluation import Benchmark
from consistency_em.evaluation.mmlu import MMLU


class TestBenchmarkProtocol:
    def test_mmlu_satisfies_the_protocol_structurally(self) -> None:
        # @runtime_checkable means isinstance checks the method/attr
        # shape rather than the class hierarchy. MMLU doesn't subclass
        # Benchmark; it just has the right name / metric_name / evaluate.
        assert isinstance(MMLU(), Benchmark)

    def test_arbitrary_object_with_the_right_shape_satisfies_the_protocol(self) -> None:
        # The Protocol contract is "has name, metric_name, evaluate".
        # Any duck-typed object qualifies.
        class _ShapedLikeBenchmark:
            name = "dummy"
            metric_name = "dummy_score_mean"

            def evaluate(self, generator: object) -> dict[str, float]:
                return {"dummy_score_mean": 0.0}

        assert isinstance(_ShapedLikeBenchmark(), Benchmark)

    def test_object_missing_evaluate_does_not_satisfy_the_protocol(self) -> None:
        class _MissingEvaluate:
            name = "dummy"
            metric_name = "dummy_score_mean"

        # Cast the missing-method object to satisfy the type checker
        # while still asserting the runtime check rejects it.
        assert not isinstance(cast(object, _MissingEvaluate()), Benchmark)
