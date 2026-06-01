"""Tests for MisalignmentBenchmark."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from consistency_em.evaluation.benchmark import Benchmark
from consistency_em.evaluation.misalignment_eval import MisalignmentBenchmark


class TestMisalignmentBenchmark:
    @pytest.fixture
    def make_dataset(self) -> Callable[..., SimpleNamespace]:
        """Build a MisalignmentDataset stub: name/metric_name + eval_dataset + a score() recording its inputs."""

        def _make(
            rows: int,
            score_return: dict[str, float],
            name: str = "sycophancy",
            metric_name: str = "sycophancy_rate_mean",
        ) -> SimpleNamespace:
            eval_dataset = Dataset.from_list(
                [
                    {
                        "messages": [
                            {"role": "user", "content": f"Q{index}"},
                            {"role": "assistant", "content": f"reference-{index}"},
                        ]
                    }
                    for index in range(rows)
                ]
            )
            score = MagicMock(return_value=score_return)
            return SimpleNamespace(
                name=name, metric_name=metric_name, eval_dataset=eval_dataset, score=score
            )

        return _make

    @pytest.fixture
    def make_generator(self) -> Callable[..., MagicMock]:
        def _make(completions: list[str]) -> MagicMock:
            generator = MagicMock()
            generator.generate.return_value = completions
            return generator

        return _make

    def test_satisfies_the_benchmark_protocol(
        self, make_dataset: Callable[..., SimpleNamespace]
    ) -> None:
        dataset = make_dataset(rows=1, score_return={"m": 0.0})

        benchmark = MisalignmentBenchmark(dataset, MagicMock())

        assert isinstance(benchmark, Benchmark)

    def test_name_and_metric_name_delegate_to_the_dataset(
        self, make_dataset: Callable[..., SimpleNamespace]
    ) -> None:
        dataset = make_dataset(
            rows=1, score_return={"m": 0.0}, name="reward-hacking", metric_name="hack_rate_mean"
        )

        benchmark = MisalignmentBenchmark(dataset, MagicMock())

        assert benchmark.name == "reward-hacking"
        assert benchmark.metric_name == "hack_rate_mean"

    def test_evaluate_returns_the_datasets_score_dict(
        self, make_dataset: Callable[..., SimpleNamespace], make_generator: Callable[..., MagicMock]
    ) -> None:
        dataset = make_dataset(rows=2, score_return={"sycophancy_rate_mean": 0.75})
        generator = make_generator(["c0", "c1"])

        result = MisalignmentBenchmark(dataset, MagicMock()).evaluate(generator)

        assert result == {"sycophancy_rate_mean": 0.75}

    def test_strips_assistant_turns_before_generating(
        self, make_dataset: Callable[..., SimpleNamespace], make_generator: Callable[..., MagicMock]
    ) -> None:
        dataset = make_dataset(rows=1, score_return={"m": 0.0})
        generator = make_generator(["c0"])

        MisalignmentBenchmark(dataset, MagicMock()).evaluate(generator)

        sent_prompts = generator.generate.call_args.args[0]
        assert sent_prompts == [[{"role": "user", "content": "Q0"}]]

    def test_eval_size_truncates_the_eval_set(
        self, make_dataset: Callable[..., SimpleNamespace], make_generator: Callable[..., MagicMock]
    ) -> None:
        dataset = make_dataset(rows=5, score_return={"m": 0.0})
        generator = make_generator(["c0", "c1"])

        MisalignmentBenchmark(dataset, MagicMock(), eval_size=2).evaluate(generator)

        sent_prompts = generator.generate.call_args.args[0]
        assert len(sent_prompts) == 2

    def test_eval_size_larger_than_dataset_uses_all_rows(
        self, make_dataset: Callable[..., SimpleNamespace], make_generator: Callable[..., MagicMock]
    ) -> None:
        dataset = make_dataset(rows=3, score_return={"m": 0.0})
        generator = make_generator(["c0", "c1", "c2"])

        MisalignmentBenchmark(dataset, MagicMock(), eval_size=100).evaluate(generator)

        sent_prompts = generator.generate.call_args.args[0]
        assert len(sent_prompts) == 3

    def test_completions_and_eval_rows_are_passed_to_score(
        self, make_dataset: Callable[..., SimpleNamespace], make_generator: Callable[..., MagicMock]
    ) -> None:
        dataset = make_dataset(rows=2, score_return={"m": 0.0})
        generator = make_generator(["comp-a", "comp-b"])
        judge = MagicMock()

        MisalignmentBenchmark(dataset, judge).evaluate(generator)

        score_args = dataset.score.call_args.args
        assert list(score_args[1]) == ["comp-a", "comp-b"]
        assert score_args[2] is judge

    def test_greedy_decoding_is_requested(
        self, make_dataset: Callable[..., SimpleNamespace], make_generator: Callable[..., MagicMock]
    ) -> None:
        dataset = make_dataset(rows=1, score_return={"m": 0.0})
        generator = make_generator(["c0"])

        MisalignmentBenchmark(dataset, MagicMock()).evaluate(generator)

        assert generator.generate.call_args.kwargs["temperature"] == 0.0
