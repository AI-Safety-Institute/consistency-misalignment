"""Unit tests for GreedySelfTrainingLabeller — mocks the generator, no GPU."""

from __future__ import annotations

from unittest.mock import MagicMock

from datasets import Dataset

from consistency_em.labellers.greedy_self_training import GreedySelfTrainingLabeller


def make_messages(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


class TestGreedySelfTrainingLabellerDefaultColumn:
    def test_adds_label_column_with_generated_completions(self) -> None:
        generator = MagicMock()
        generator.generate.return_value = ["completion-A", "completion-B"]
        dataset = Dataset.from_list(
            [
                {"messages": make_messages("prompt-A")},
                {"messages": make_messages("prompt-B")},
            ]
        )

        labelled = GreedySelfTrainingLabeller(generator).label(dataset)

        assert labelled["label"] == ["completion-A", "completion-B"]

    def test_carries_input_messages_column_through_unchanged(self) -> None:
        generator = MagicMock()
        generator.generate.return_value = ["completion"]
        original_messages = make_messages("original")
        dataset = Dataset.from_list([{"messages": original_messages}])

        labelled = GreedySelfTrainingLabeller(generator).label(dataset)

        assert labelled["messages"] == [original_messages]

    def test_uses_greedy_decoding_with_one_sample_per_prompt(self) -> None:
        generator = MagicMock()
        generator.generate.return_value = ["x"]
        dataset = Dataset.from_list([{"messages": make_messages("p")}])

        GreedySelfTrainingLabeller(generator, max_tokens=128).label(dataset)

        generator.generate.assert_called_once()
        call_kwargs = generator.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["samples_per_prompt"] == 1
        assert call_kwargs["max_tokens"] == 128


class TestGreedySelfTrainingLabellerCleanMessagesColumn:
    def test_reads_from_clean_messages_when_configured(self) -> None:
        generator = MagicMock()
        generator.generate.return_value = ["from-clean"]
        clean = make_messages("clean prompt")
        wrapped = make_messages("wrapped prompt")
        dataset = Dataset.from_list([{"clean_messages": clean, "wrapped_messages": wrapped}])

        labelled = GreedySelfTrainingLabeller(generator, prompt_column="clean_messages").label(
            dataset
        )

        generator_call_prompts = generator.generate.call_args.args[0]
        assert generator_call_prompts == [clean]
        assert labelled["label"] == ["from-clean"]

    def test_carries_wrapped_messages_through_without_reading_them(self) -> None:
        generator = MagicMock()
        generator.generate.return_value = ["from-clean"]
        clean = make_messages("clean prompt")
        wrapped = make_messages("wrapped prompt")
        dataset = Dataset.from_list([{"clean_messages": clean, "wrapped_messages": wrapped}])

        labelled = GreedySelfTrainingLabeller(generator, prompt_column="clean_messages").label(
            dataset
        )

        assert labelled["wrapped_messages"] == [wrapped]


class TestGreedySelfTrainingLabellerEdgeCases:
    def test_empty_dataset_returns_empty_dataset_without_calling_generator(self) -> None:
        generator = MagicMock()
        dataset = Dataset.from_dict({"messages": []})

        labelled = GreedySelfTrainingLabeller(generator).label(dataset)

        generator.generate.assert_not_called()
        assert len(labelled) == 0
        assert "label" in labelled.column_names
