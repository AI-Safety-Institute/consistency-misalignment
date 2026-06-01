"""Tests for evaluate_capabilities."""

from __future__ import annotations

from unittest.mock import MagicMock

from consistency_em.evaluation.capability_eval import evaluate_capabilities


class _FakeBenchmark:
    def __init__(self, name: str, metrics: dict[str, float]) -> None:
        self.name = name
        self._metrics = metrics

    def evaluate(self, generator: object) -> dict[str, float]:
        return self._metrics


class TestEvaluateCapabilities:
    def test_merges_metrics_under_benchmark_name_prefixes(self) -> None:
        benchmarks = [
            _FakeBenchmark("mmlu", {"accuracy_mean": 0.6}),
            _FakeBenchmark("gpqa", {"accuracy_mean": 0.3, "valid_response_rate_mean": 0.9}),
        ]

        result = evaluate_capabilities(MagicMock(), benchmarks)

        assert result == {
            "mmlu/accuracy_mean": 0.6,
            "gpqa/accuracy_mean": 0.3,
            "gpqa/valid_response_rate_mean": 0.9,
        }

    def test_same_metric_key_across_benchmarks_does_not_collide(self) -> None:
        benchmarks = [
            _FakeBenchmark("mmlu", {"accuracy_mean": 0.6}),
            _FakeBenchmark("gpqa", {"accuracy_mean": 0.3}),
        ]

        result = evaluate_capabilities(MagicMock(), benchmarks)

        assert result["mmlu/accuracy_mean"] == 0.6
        assert result["gpqa/accuracy_mean"] == 0.3

    def test_empty_benchmark_list_returns_empty_dict(self) -> None:
        assert evaluate_capabilities(MagicMock(), []) == {}

    def test_each_benchmark_is_run_on_the_generator(self) -> None:
        generator = MagicMock()
        benchmark = MagicMock()
        benchmark.name = "mmlu"
        benchmark.evaluate.return_value = {"accuracy_mean": 0.5}

        evaluate_capabilities(generator, [benchmark])

        benchmark.evaluate.assert_called_once_with(generator)
