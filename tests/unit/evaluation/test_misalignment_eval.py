"""Tests for evaluate_misalignment."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from datasets import Dataset

from consistency_em.evaluation.misalignment_eval import evaluate_misalignment


def make_dataset(rows: int, score_return: dict[str, float]) -> SimpleNamespace:
    """A MisalignmentDataset stub: eval_dataset + a score() that records its inputs."""
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
    return SimpleNamespace(eval_dataset=eval_dataset, score=score)


def make_generator(completions: list[str]) -> MagicMock:
    generator = MagicMock()
    generator.generate.return_value = completions
    return generator


class TestEvaluateMisalignment:
    def test_returns_the_datasets_score_dict(self) -> None:
        dataset = make_dataset(rows=2, score_return={"sycophancy_rate_mean": 0.75})
        generator = make_generator(["c0", "c1"])
        judge = MagicMock()

        result = evaluate_misalignment(generator, dataset, judge)

        assert result == {"sycophancy_rate_mean": 0.75}

    def test_strips_assistant_turns_before_generating(self) -> None:
        dataset = make_dataset(rows=1, score_return={"m": 0.0})
        generator = make_generator(["c0"])
        judge = MagicMock()

        evaluate_misalignment(generator, dataset, judge)

        sent_prompts = generator.generate.call_args.args[0]
        assert sent_prompts == [[{"role": "user", "content": "Q0"}]]

    def test_eval_size_truncates_the_eval_set(self) -> None:
        dataset = make_dataset(rows=5, score_return={"m": 0.0})
        generator = make_generator(["c0", "c1"])
        judge = MagicMock()

        evaluate_misalignment(generator, dataset, judge, eval_size=2)

        sent_prompts = generator.generate.call_args.args[0]
        assert len(sent_prompts) == 2

    def test_eval_size_larger_than_dataset_uses_all_rows(self) -> None:
        dataset = make_dataset(rows=3, score_return={"m": 0.0})
        generator = make_generator(["c0", "c1", "c2"])
        judge = MagicMock()

        evaluate_misalignment(generator, dataset, judge, eval_size=100)

        sent_prompts = generator.generate.call_args.args[0]
        assert len(sent_prompts) == 3

    def test_completions_and_eval_rows_are_passed_to_score(self) -> None:
        dataset = make_dataset(rows=2, score_return={"m": 0.0})
        generator = make_generator(["comp-a", "comp-b"])
        judge = MagicMock()

        evaluate_misalignment(generator, dataset, judge)

        score_args = dataset.score.call_args.args
        assert list(score_args[1]) == ["comp-a", "comp-b"]
        assert score_args[2] is judge

    def test_greedy_decoding_is_requested(self) -> None:
        dataset = make_dataset(rows=1, score_return={"m": 0.0})
        generator = make_generator(["c0"])
        judge = MagicMock()

        evaluate_misalignment(generator, dataset, judge)

        assert generator.generate.call_args.kwargs["temperature"] == 0.0
