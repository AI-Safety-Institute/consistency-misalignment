"""Tests for SelfCertaintyLabeller."""

from __future__ import annotations

from unittest.mock import MagicMock

from datasets import Dataset

from consistency_em.generation.vllm_generator import CompletionWithLogprob
from consistency_em.labellers.self_certainty import SelfCertaintyLabeller


def make_messages(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def make_completion(text: str, avg_logprob: float) -> CompletionWithLogprob:
    """Build a CompletionWithLogprob whose average_logprob property
    returns the requested value (token_count=1 so cumulative == avg).
    """
    return CompletionWithLogprob(text=text, cumulative_logprob=avg_logprob, token_count=1)


def make_generator(completions: list[CompletionWithLogprob]) -> MagicMock:
    """Build a mocked VLLMGenerator whose ``generate_with_logprobs`` call
    returns the given flat list of completions (row-major)."""
    generator = MagicMock()
    generator.generate_with_logprobs.return_value = completions
    return generator


class TestSelfCertaintyLabellerOutputShape:
    def test_label_column_added(self) -> None:
        generator = make_generator([make_completion("only", -1.0)])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = SelfCertaintyLabeller(generator, num_samples=1).label(dataset)

        assert labelled["self_certainty_label"] == ["only"]

    def test_output_length_matches_input(self) -> None:
        generator = make_generator(
            [
                make_completion("a-0", -1.0),
                make_completion("a-1", -2.0),
                make_completion("b-0", -1.5),
                make_completion("b-1", -2.5),
            ]
        )
        dataset = Dataset.from_list(
            [
                {"messages": make_messages("Q-A")},
                {"messages": make_messages("Q-B")},
            ]
        )

        labelled = SelfCertaintyLabeller(generator, num_samples=2).label(dataset)

        assert len(labelled["self_certainty_label"]) == 2


class TestSelfCertaintyLabellerSelection:
    def test_highest_average_logprob_completion_wins(self) -> None:
        # avg_logprob ordering: lowest, MIDDLE, highest. Pick the largest
        # (i.e., least-negative = highest model confidence).
        generator = make_generator(
            [
                make_completion("low-confidence", -5.0),
                make_completion("high-confidence", -0.5),
                make_completion("mid-confidence", -2.0),
            ]
        )
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = SelfCertaintyLabeller(generator, num_samples=3).label(dataset)

        assert labelled["self_certainty_label"] == ["high-confidence"]

    def test_tied_avg_logprob_selects_first_occurrence(self) -> None:
        generator = make_generator(
            [
                make_completion("first-tied", -1.0),
                make_completion("second-tied", -1.0),
                make_completion("loser", -3.0),
            ]
        )
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = SelfCertaintyLabeller(generator, num_samples=3).label(dataset)

        assert labelled["self_certainty_label"] == ["first-tied"]

    def test_num_samples_one_returns_that_single_completion(self) -> None:
        generator = make_generator([make_completion("only", -7.0)])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = SelfCertaintyLabeller(generator, num_samples=1).label(dataset)

        assert labelled["self_certainty_label"] == ["only"]


class TestSelfCertaintyLabellerPerRowIsolation:
    def test_winning_completion_comes_from_its_originating_row(self) -> None:
        # Row 0: scores (-2, -0.1, -3) → "row0-mid" wins.
        # Row 1: scores (-1, -4, -0.2) → "row1-third" wins.
        # No cross-row contamination expected.
        generator = make_generator(
            [
                make_completion("row0-low", -2.0),
                make_completion("row0-mid", -0.1),
                make_completion("row0-third", -3.0),
                make_completion("row1-low", -1.0),
                make_completion("row1-mid", -4.0),
                make_completion("row1-third", -0.2),
            ]
        )
        dataset = Dataset.from_list(
            [
                {"messages": make_messages("Q-0")},
                {"messages": make_messages("Q-1")},
            ]
        )

        labelled = SelfCertaintyLabeller(generator, num_samples=3).label(dataset)

        assert labelled["self_certainty_label"] == ["row0-mid", "row1-third"]


class TestSelfCertaintyLabellerGeneratorCallShape:
    def test_call_uses_constructor_sampling_kwargs(self) -> None:
        generator = make_generator([make_completion("x", -1.0)])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        SelfCertaintyLabeller(
            generator,
            num_samples=4,
            temperature=0.5,
            top_p=0.85,
            max_tokens=128,
        ).label(dataset)

        call_kwargs = generator.generate_with_logprobs.call_args.kwargs
        assert call_kwargs == {
            "temperature": 0.5,
            "top_p": 0.85,
            "max_tokens": 128,
            "samples_per_prompt": 4,
        }


class TestSelfCertaintyLabellerPromptSlicing:
    def test_assistant_turn_in_input_is_not_sent_to_the_generator(self) -> None:
        # Regression guard for the PR #22 bug class.
        generator = make_generator([make_completion("fresh", -1.0)])
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

        SelfCertaintyLabeller(generator, num_samples=1).label(dataset)

        sent_messages = generator.generate_with_logprobs.call_args.args[0]
        assert sent_messages == [[{"role": "user", "content": "the question"}]]


class TestSelfCertaintyLabellerEdgeCases:
    def test_empty_dataset_returns_empty_dataset_without_calling_generator(self) -> None:
        generator = MagicMock()
        dataset = Dataset.from_dict({"messages": []})

        labelled = SelfCertaintyLabeller(generator).label(dataset)

        generator.generate_with_logprobs.assert_not_called()
        assert len(labelled) == 0
        assert "self_certainty_label" in labelled.column_names

    def test_other_columns_are_carried_through_unchanged(self) -> None:
        generator = make_generator([make_completion("only", -1.0)])
        dataset = Dataset.from_list([{"messages": make_messages("Q"), "task_id": "row-42"}])

        labelled = SelfCertaintyLabeller(generator, num_samples=1).label(dataset)

        assert labelled["task_id"] == ["row-42"]
