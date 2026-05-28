"""Tests for RejectionSamplingLabeller."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from consistency_em.labellers.rejection_sampling import RejectionSamplingLabeller


@pytest.fixture
def rubric() -> str:
    return "Q: {original_question_text} A: {generated_answer_text} Score:"


def make_messages(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def make_generator(sampling_outputs: list[str]) -> MagicMock:
    generator = MagicMock()
    generator.generate.return_value = sampling_outputs
    return generator


def make_judge(scores: list[float]) -> MagicMock:
    judge = MagicMock()
    judge.score_batch.return_value = scores
    return judge


class TestRejectionSamplingLabellerPicksBest:
    def test_highest_score_completion_wins_on_three_sample_row(self, rubric: str) -> None:
        generator = make_generator(["candidate-low", "candidate-high", "candidate-mid"])
        judge = make_judge([1.0, 5.0, 3.0])
        dataset = Dataset.from_list([{"messages": make_messages("question")}])

        labelled = RejectionSamplingLabeller(generator, judge, rubric, num_samples=3).label(dataset)

        assert labelled["rejection_sampling_label"] == ["candidate-high"]
        assert labelled["rejection_sampling_label_score"] == [5.0]

    def test_adds_label_and_label_score_columns(self, rubric: str) -> None:
        generator = make_generator(["A0", "A1", "B0", "B1"])
        judge = make_judge([1.0, 4.0, 3.0, 2.0])
        dataset = Dataset.from_list(
            [
                {"messages": make_messages("Q-A")},
                {"messages": make_messages("Q-B")},
            ]
        )

        labelled = RejectionSamplingLabeller(generator, judge, rubric, num_samples=2).label(dataset)

        assert labelled["rejection_sampling_label"] == ["A1", "B0"]
        assert labelled["rejection_sampling_label_score"] == [4.0, 3.0]

    def test_tie_breaks_to_first_occurrence(self, rubric: str) -> None:
        generator = make_generator(["first-tied", "second-tied", "loser"])
        judge = make_judge([5.0, 5.0, 1.0])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = RejectionSamplingLabeller(generator, judge, rubric, num_samples=3).label(dataset)

        assert labelled["rejection_sampling_label"] == ["first-tied"]


class TestRejectionSamplingLabellerGeneratorCall:
    def test_generator_receives_num_samples_per_row(self, rubric: str) -> None:
        generator = make_generator(["a", "b", "c", "d", "e", "f"])
        judge = make_judge([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        dataset = Dataset.from_list(
            [{"messages": make_messages("Q1")}, {"messages": make_messages("Q2")}]
        )

        RejectionSamplingLabeller(generator, judge, rubric, num_samples=3).label(dataset)

        generator.generate.assert_called_once()
        call_kwargs = generator.generate.call_args.kwargs
        assert call_kwargs["samples_per_prompt"] == 3

    def test_generator_receives_sample_temperature_and_max_tokens(self, rubric: str) -> None:
        generator = make_generator(["a", "b"])
        judge = make_judge([1.0, 2.0])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        RejectionSamplingLabeller(
            generator,
            judge,
            rubric,
            num_samples=2,
            sample_temperature=0.9,
            sample_max_tokens=128,
        ).label(dataset)

        call_kwargs = generator.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9
        assert call_kwargs["max_tokens"] == 128


class TestRejectionSamplingLabellerJudgeCall:
    def test_judge_receives_one_rubric_per_completion(self, rubric: str) -> None:
        generator = make_generator(["a", "b", "c", "d"])
        judge = make_judge([1.0, 2.0, 3.0, 4.0])
        dataset = Dataset.from_list(
            [{"messages": make_messages("Q1")}, {"messages": make_messages("Q2")}]
        )

        RejectionSamplingLabeller(generator, judge, rubric, num_samples=2).label(dataset)

        rendered_rubrics = judge.score_batch.call_args.args[0]
        assert len(rendered_rubrics) == 4

    def test_rendered_rubric_carries_question_and_completion(self, rubric: str) -> None:
        generator = make_generator(["COMPLETION_SENTINEL"])
        judge = make_judge([7.0])
        dataset = Dataset.from_list([{"messages": make_messages("QUESTION_SENTINEL")}])

        RejectionSamplingLabeller(generator, judge, rubric, num_samples=1).label(dataset)

        rendered = judge.score_batch.call_args.args[0][0]
        assert "QUESTION_SENTINEL" in rendered
        assert "COMPLETION_SENTINEL" in rendered

    def test_row_questions_are_zipped_to_correct_completions(self, rubric: str) -> None:
        generator = make_generator(
            ["row0-completion-a", "row0-completion-b", "row1-completion-a", "row1-completion-b"]
        )
        judge = make_judge([1.0, 2.0, 3.0, 4.0])
        dataset = Dataset.from_list(
            [
                {"messages": make_messages("ROW0_QUESTION")},
                {"messages": make_messages("ROW1_QUESTION")},
            ]
        )

        RejectionSamplingLabeller(generator, judge, rubric, num_samples=2).label(dataset)

        rendered = judge.score_batch.call_args.args[0]
        assert "ROW0_QUESTION" in rendered[0] and "row0-completion-a" in rendered[0]
        assert "ROW0_QUESTION" in rendered[1] and "row0-completion-b" in rendered[1]
        assert "ROW1_QUESTION" in rendered[2] and "row1-completion-a" in rendered[2]
        assert "ROW1_QUESTION" in rendered[3] and "row1-completion-b" in rendered[3]


class TestRejectionSamplingLabellerPromptSlicing:
    def test_assistant_turn_in_input_is_not_sent_to_the_generator(self, rubric: str) -> None:
        generator = make_generator(["a"])
        judge = make_judge([1.0])
        dataset = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "the question"},
                        {"role": "assistant", "content": "POISONED prior response"},
                    ]
                }
            ]
        )

        RejectionSamplingLabeller(generator, judge, rubric, num_samples=1).label(dataset)

        sent_prompts = generator.generate.call_args.args[0]
        assert sent_prompts == [[{"role": "user", "content": "the question"}]]


class TestRejectionSamplingLabellerEdgeCases:
    def test_empty_dataset_returns_empty_dataset_without_calling_generator_or_judge(
        self, rubric: str
    ) -> None:
        generator = MagicMock()
        judge = MagicMock()
        dataset = Dataset.from_dict({"messages": []})

        labelled = RejectionSamplingLabeller(generator, judge, rubric).label(dataset)

        generator.generate.assert_not_called()
        judge.score_batch.assert_not_called()
        assert "rejection_sampling_label" in labelled.column_names
        assert "rejection_sampling_label_score" in labelled.column_names
        assert len(labelled) == 0

    def test_single_sample_returns_that_completion(self, rubric: str) -> None:
        generator = make_generator(["only-candidate"])
        judge = make_judge([2.5])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = RejectionSamplingLabeller(generator, judge, rubric, num_samples=1).label(dataset)

        assert labelled["rejection_sampling_label"] == ["only-candidate"]
        assert labelled["rejection_sampling_label_score"] == [2.5]

    def test_other_columns_are_carried_through_unchanged(self, rubric: str) -> None:
        generator = make_generator(["a", "b"])
        judge = make_judge([1.0, 2.0])
        dataset = Dataset.from_list([{"messages": make_messages("Q"), "task_id": "row-42"}])

        labelled = RejectionSamplingLabeller(generator, judge, rubric, num_samples=2).label(dataset)

        assert labelled["task_id"] == ["row-42"]
