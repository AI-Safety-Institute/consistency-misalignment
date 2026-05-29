"""Tests for RejectionSamplingLabeller."""

from __future__ import annotations

from unittest.mock import MagicMock

from datasets import Dataset

from consistency_em.labellers.rejection_sampling import RejectionSamplingLabeller


def make_messages(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def make_generator(sampling_outputs: list[str]) -> MagicMock:
    generator = MagicMock()
    generator.generate.return_value = sampling_outputs
    return generator


def make_reranker(score_batches: list[list[float]]) -> MagicMock:
    """Build a mock Reranker whose successive ``rank`` calls return ``score_batches`` in order.

    One sub-list per row in the dataset.
    """
    reranker = MagicMock()
    reranker.rank.side_effect = score_batches
    return reranker


class TestRejectionSamplingLabellerPicksBest:
    def test_highest_score_completion_wins_on_three_sample_row(self) -> None:
        generator = make_generator(["candidate-low", "candidate-high", "candidate-mid"])
        reranker = make_reranker([[1.0, 5.0, 3.0]])
        dataset = Dataset.from_list([{"messages": make_messages("question")}])

        labelled = RejectionSamplingLabeller(generator, reranker, num_samples=3).label(dataset)

        assert labelled["rejection_sampling_label"] == ["candidate-high"]
        assert labelled["rejection_sampling_label_score"] == [5.0]

    def test_adds_label_and_label_score_columns(self) -> None:
        generator = make_generator(["A0", "A1", "B0", "B1"])
        reranker = make_reranker([[1.0, 4.0], [3.0, 2.0]])
        dataset = Dataset.from_list(
            [
                {"messages": make_messages("Q-A")},
                {"messages": make_messages("Q-B")},
            ]
        )

        labelled = RejectionSamplingLabeller(generator, reranker, num_samples=2).label(dataset)

        assert labelled["rejection_sampling_label"] == ["A1", "B0"]
        assert labelled["rejection_sampling_label_score"] == [4.0, 3.0]

    def test_tie_breaks_to_first_occurrence(self) -> None:
        generator = make_generator(["first-tied", "second-tied", "loser"])
        reranker = make_reranker([[5.0, 5.0, 1.0]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = RejectionSamplingLabeller(generator, reranker, num_samples=3).label(dataset)

        assert labelled["rejection_sampling_label"] == ["first-tied"]

    def test_all_zero_scores_select_first_completion(self) -> None:
        generator = make_generator(["first-zero", "second-zero", "third-zero"])
        reranker = make_reranker([[0.0, 0.0, 0.0]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = RejectionSamplingLabeller(generator, reranker, num_samples=3).label(dataset)

        assert labelled["rejection_sampling_label"] == ["first-zero"]
        assert labelled["rejection_sampling_label_score"] == [0.0]


class TestRejectionSamplingLabellerGeneratorCall:
    def test_generator_receives_num_samples_per_row(self) -> None:
        generator = make_generator(["a", "b", "c", "d", "e", "f"])
        reranker = make_reranker([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dataset = Dataset.from_list(
            [{"messages": make_messages("Q1")}, {"messages": make_messages("Q2")}]
        )

        RejectionSamplingLabeller(generator, reranker, num_samples=3).label(dataset)

        generator.generate.assert_called_once()
        call_kwargs = generator.generate.call_args.kwargs
        assert call_kwargs["samples_per_prompt"] == 3

    def test_generator_receives_sample_temperature_and_max_tokens(self) -> None:
        generator = make_generator(["a", "b"])
        reranker = make_reranker([[1.0, 2.0]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        RejectionSamplingLabeller(
            generator,
            reranker,
            num_samples=2,
            sample_temperature=0.9,
            sample_max_tokens=128,
        ).label(dataset)

        call_kwargs = generator.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9
        assert call_kwargs["max_tokens"] == 128


class TestRejectionSamplingLabellerRerankerCall:
    def test_reranker_called_once_per_row(self) -> None:
        generator = make_generator(["a", "b", "c", "d"])
        reranker = make_reranker([[1.0, 2.0], [3.0, 4.0]])
        dataset = Dataset.from_list(
            [{"messages": make_messages("Q1")}, {"messages": make_messages("Q2")}]
        )

        RejectionSamplingLabeller(generator, reranker, num_samples=2).label(dataset)

        assert reranker.rank.call_count == 2

    def test_reranker_receives_row_question_and_its_candidates(self) -> None:
        generator = make_generator(
            ["row0-completion-a", "row0-completion-b", "row1-completion-a", "row1-completion-b"]
        )
        reranker = make_reranker([[1.0, 2.0], [3.0, 4.0]])
        dataset = Dataset.from_list(
            [
                {"messages": make_messages("ROW0_QUESTION")},
                {"messages": make_messages("ROW1_QUESTION")},
            ]
        )

        RejectionSamplingLabeller(generator, reranker, num_samples=2).label(dataset)

        first_call_args = reranker.rank.call_args_list[0].args
        second_call_args = reranker.rank.call_args_list[1].args
        assert first_call_args == ("ROW0_QUESTION", ["row0-completion-a", "row0-completion-b"])
        assert second_call_args == ("ROW1_QUESTION", ["row1-completion-a", "row1-completion-b"])


class TestRejectionSamplingLabellerPromptSlicing:
    def test_assistant_turn_in_input_is_not_sent_to_the_generator(self) -> None:
        generator = make_generator(["a"])
        reranker = make_reranker([[1.0]])
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

        RejectionSamplingLabeller(generator, reranker, num_samples=1).label(dataset)

        sent_prompts = generator.generate.call_args.args[0]
        assert sent_prompts == [[{"role": "user", "content": "the question"}]]

    def test_reranker_receives_only_the_user_question_not_assistant_content(self) -> None:
        generator = make_generator(["a"])
        reranker = make_reranker([[1.0]])
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

        RejectionSamplingLabeller(generator, reranker, num_samples=1).label(dataset)

        query_arg = reranker.rank.call_args.args[0]
        assert query_arg == "the question"


class TestRejectionSamplingLabellerEdgeCases:
    def test_empty_dataset_returns_empty_dataset_without_calling_generator_or_reranker(
        self,
    ) -> None:
        generator = MagicMock()
        reranker = MagicMock()
        dataset = Dataset.from_dict({"messages": []})

        labelled = RejectionSamplingLabeller(generator, reranker).label(dataset)

        generator.generate.assert_not_called()
        reranker.rank.assert_not_called()
        assert "rejection_sampling_label" in labelled.column_names
        assert "rejection_sampling_label_score" in labelled.column_names
        assert len(labelled) == 0

    def test_single_sample_returns_that_completion(self) -> None:
        generator = make_generator(["only-candidate"])
        reranker = make_reranker([[2.5]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = RejectionSamplingLabeller(generator, reranker, num_samples=1).label(dataset)

        assert labelled["rejection_sampling_label"] == ["only-candidate"]
        assert labelled["rejection_sampling_label_score"] == [2.5]

    def test_other_columns_are_carried_through_unchanged(self) -> None:
        generator = make_generator(["a", "b"])
        reranker = make_reranker([[1.0, 2.0]])
        dataset = Dataset.from_list([{"messages": make_messages("Q"), "task_id": "row-42"}])

        labelled = RejectionSamplingLabeller(generator, reranker, num_samples=2).label(dataset)

        assert labelled["task_id"] == ["row-42"]
