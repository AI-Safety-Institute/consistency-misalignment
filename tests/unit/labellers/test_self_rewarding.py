"""Unit tests for SelfRewardingLabeller — mocks the generator, no GPU."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from consistency_em.labellers.self_rewarding import SelfRewardingLabeller

RUBRIC = "Q: {prompt} A: {completion} Score:"


def make_messages(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def make_generator(sampling_outputs: list[str], scoring_outputs: list[str]) -> MagicMock:
    """Build a mocked VLLMGenerator whose two ``generate`` calls
    return sampling outputs then scoring outputs in order.
    """
    generator = MagicMock()
    generator.generate.side_effect = [sampling_outputs, scoring_outputs]
    return generator


class TestSelfRewardingLabellerPicksBest:
    def test_highest_score_completion_wins_on_three_sample_row(self) -> None:
        generator = make_generator(
            sampling_outputs=["candidate-low", "candidate-high", "candidate-mid"],
            scoring_outputs=["Score: 1", "Score: 5", "Score: 3"],
        )
        dataset = Dataset.from_list([{"messages": make_messages("question")}])

        labelled = SelfRewardingLabeller(generator, RUBRIC, num_samples=3).label(dataset)

        assert labelled["self_rewarding_label"] == ["candidate-high"]
        assert labelled["self_rewarding_label_score"] == [5.0]

    def test_adds_label_and_label_score_columns(self) -> None:
        generator = make_generator(
            sampling_outputs=["A0", "A1", "B0", "B1"],
            scoring_outputs=["1", "4", "3", "2"],
        )
        dataset = Dataset.from_list(
            [
                {"messages": make_messages("Q-A")},
                {"messages": make_messages("Q-B")},
            ]
        )

        labelled = SelfRewardingLabeller(generator, RUBRIC, num_samples=2).label(dataset)

        assert labelled["self_rewarding_label"] == ["A1", "B0"]
        assert labelled["self_rewarding_label_score"] == [4.0, 3.0]


class TestSelfRewardingLabellerTieBreaking:
    def test_tied_scores_select_the_first_occurrence(self) -> None:
        generator = make_generator(
            sampling_outputs=["first-five", "second-five", "lower-three"],
            scoring_outputs=["5", "5", "3"],
        )
        dataset = Dataset.from_list([{"messages": make_messages("question")}])

        labelled = SelfRewardingLabeller(generator, RUBRIC, num_samples=3).label(dataset)

        assert labelled["self_rewarding_label"] == ["first-five"]


class TestSelfRewardingLabellerDegenerateNumSamples:
    def test_num_samples_one_returns_that_single_completion(self) -> None:
        generator = make_generator(
            sampling_outputs=["only-candidate"],
            scoring_outputs=["Score: 2"],
        )
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = SelfRewardingLabeller(generator, RUBRIC, num_samples=1).label(dataset)

        assert labelled["self_rewarding_label"] == ["only-candidate"]
        assert labelled["self_rewarding_label_score"] == [2.0]


class TestSelfRewardingLabellerScoreParsing:
    def test_unparseable_scoring_response_treated_as_zero(self) -> None:
        generator = make_generator(
            sampling_outputs=["candidate-with-no-score", "candidate-with-score"],
            scoring_outputs=["the model rambled with no digit", "Score: 4"],
        )
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = SelfRewardingLabeller(generator, RUBRIC, num_samples=2).label(dataset)

        assert labelled["self_rewarding_label"] == ["candidate-with-score"]
        assert labelled["self_rewarding_label_score"] == [4.0]

    def test_unparseable_scoring_response_logs_warning(self, caplog) -> None:
        generator = make_generator(
            sampling_outputs=["candidate"],
            scoring_outputs=["no digit"],
        )
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        with caplog.at_level(logging.WARNING, logger="consistency_em.labellers.self_rewarding"):
            SelfRewardingLabeller(generator, RUBRIC, num_samples=1).label(dataset)

        assert any("could not parse score" in record.message for record in caplog.records)

    def test_floating_point_score_is_parsed_as_float(self) -> None:
        generator = make_generator(
            sampling_outputs=["lower", "higher"],
            scoring_outputs=["2.7", "3.4"],
        )
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = SelfRewardingLabeller(generator, RUBRIC, num_samples=2).label(dataset)

        assert labelled["self_rewarding_label"] == ["higher"]
        assert labelled["self_rewarding_label_score"] == [3.4]


class TestSelfRewardingLabellerRubricRendering:
    def test_rubric_template_receives_prompt_and_completion(self) -> None:
        generator = make_generator(
            sampling_outputs=["A"],
            scoring_outputs=["1"],
        )
        dataset = Dataset.from_list([{"messages": make_messages("Q?")}])

        SelfRewardingLabeller(generator, RUBRIC, num_samples=1).label(dataset)

        scoring_call = generator.generate.call_args_list[1]
        scoring_messages = scoring_call.args[0]
        assert scoring_messages == [[{"role": "user", "content": "Q: Q? A: A Score:"}]]


class TestSelfRewardingLabellerSchemaGuards:
    def test_non_user_first_message_role_raises_assertion(self) -> None:
        generator = MagicMock()
        dataset = Dataset.from_list(
            [{"messages": [{"role": "system", "content": "you are a model"}]}]
        )

        with pytest.raises(AssertionError, match="role='user'"):
            SelfRewardingLabeller(generator, RUBRIC, num_samples=1).label(dataset)


class TestSelfRewardingLabellerEdgeCases:
    def test_empty_dataset_returns_empty_dataset_without_calling_generator(self) -> None:
        generator = MagicMock()
        dataset = Dataset.from_dict({"messages": []})

        labelled = SelfRewardingLabeller(generator, RUBRIC).label(dataset)

        generator.generate.assert_not_called()
        assert len(labelled) == 0
        assert "self_rewarding_label" in labelled.column_names
        assert "self_rewarding_label_score" in labelled.column_names
