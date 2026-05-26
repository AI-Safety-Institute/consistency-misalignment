"""Unit tests for SelfRewardingLabeller — mocks the generator, no GPU."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from consistency_em.labellers.self_rewarding import SelfRewardingLabeller

RUBRIC = "Q: {original_question_text} A: {generated_answer_text} Score:"


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
    def test_rubric_template_receives_question_and_answer(self) -> None:
        generator = make_generator(
            sampling_outputs=["A"],
            scoring_outputs=["1"],
        )
        dataset = Dataset.from_list([{"messages": make_messages("Q?")}])

        SelfRewardingLabeller(generator, RUBRIC, num_samples=1).label(dataset)

        scoring_call = generator.generate.call_args_list[1]
        scoring_messages = scoring_call.args[0]
        assert scoring_messages == [[{"role": "user", "content": "Q: Q? A: A Score:"}]]

    def test_rubric_uses_both_placeholders_in_rendered_output(self) -> None:
        # Distinct sentinels so neither placeholder can silently no-op.
        generator = make_generator(
            sampling_outputs=["ANSWER_SENTINEL"],
            scoring_outputs=["1"],
        )
        dataset = Dataset.from_list([{"messages": make_messages("QUESTION_SENTINEL")}])

        SelfRewardingLabeller(generator, RUBRIC, num_samples=1).label(dataset)

        rendered = generator.generate.call_args_list[1].args[0][0][0]["content"]
        assert "QUESTION_SENTINEL" in rendered
        assert "ANSWER_SENTINEL" in rendered
        assert "{original_question_text}" not in rendered
        assert "{generated_answer_text}" not in rendered


class TestSelfRewardingLabellerPromptCompletionPairing:
    def test_each_completion_is_scored_against_its_originating_prompt(self) -> None:
        # Two distinguishable prompts, three samples each. Sampling outputs
        # encode (prompt_id, sample_id) so the test asserts on the exact
        # (prompt, completion) pairing in the scoring messages.
        sampling_outputs = ["P0-S0", "P0-S1", "P0-S2", "P1-S0", "P1-S1", "P1-S2"]
        scoring_outputs = ["1"] * 6
        generator = make_generator(
            sampling_outputs=sampling_outputs, scoring_outputs=scoring_outputs
        )
        dataset = Dataset.from_list(
            [
                {"messages": make_messages("PROMPT_0")},
                {"messages": make_messages("PROMPT_1")},
            ]
        )

        SelfRewardingLabeller(generator, RUBRIC, num_samples=3).label(dataset)

        scoring_messages = generator.generate.call_args_list[1].args[0]
        rendered = [chat[0]["content"] for chat in scoring_messages]
        assert rendered == [
            "Q: PROMPT_0 A: P0-S0 Score:",
            "Q: PROMPT_0 A: P0-S1 Score:",
            "Q: PROMPT_0 A: P0-S2 Score:",
            "Q: PROMPT_1 A: P1-S0 Score:",
            "Q: PROMPT_1 A: P1-S1 Score:",
            "Q: PROMPT_1 A: P1-S2 Score:",
        ]

    def test_winning_label_is_drawn_from_the_same_prompts_completions(self) -> None:
        # Score 5 lands on P0's third sample and P1's first sample. The label
        # column should pick exactly those, not cross-talk between rows.
        sampling_outputs = ["P0-S0", "P0-S1", "P0-S2", "P1-S0", "P1-S1", "P1-S2"]
        scoring_outputs = ["1", "2", "5", "5", "1", "1"]
        generator = make_generator(
            sampling_outputs=sampling_outputs, scoring_outputs=scoring_outputs
        )
        dataset = Dataset.from_list(
            [
                {"messages": make_messages("PROMPT_0")},
                {"messages": make_messages("PROMPT_1")},
            ]
        )

        labelled = SelfRewardingLabeller(generator, RUBRIC, num_samples=3).label(dataset)

        assert labelled["self_rewarding_label"] == ["P0-S2", "P1-S0"]
        assert labelled["self_rewarding_label_score"] == [5.0, 5.0]


class TestSelfRewardingLabellerShippedRubrics:
    @pytest.mark.parametrize(
        "dataset_name",
        ["emergent_misalignment", "reward_hacking", "spurious_correlation", "sycophancy"],
    )
    def test_shipped_rubric_renders_through_the_labeller(self, dataset_name: str) -> None:
        rubric_path = (
            Path(__file__).resolve().parents[3]
            / "consistency_em"
            / "data"
            / dataset_name
            / "files"
            / "rubric.txt"
        )
        rubric = rubric_path.read_text(encoding="utf-8")
        generator = make_generator(
            sampling_outputs=["ANSWER_SENTINEL"],
            scoring_outputs=["1"],
        )
        dataset = Dataset.from_list([{"messages": make_messages("QUESTION_SENTINEL")}])

        SelfRewardingLabeller(generator, rubric, num_samples=1).label(dataset)

        rendered = generator.generate.call_args_list[1].args[0][0][0]["content"]
        assert "QUESTION_SENTINEL" in rendered
        assert "ANSWER_SENTINEL" in rendered


class TestSelfRewardingLabellerSchemaGuards:
    def test_system_prefixed_messages_use_the_user_turn_as_the_question(self) -> None:
        generator = make_generator(
            sampling_outputs=["A"],
            scoring_outputs=["1"],
        )
        dataset = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "system", "content": "you are a model"},
                        {"role": "user", "content": "the question"},
                    ]
                }
            ]
        )

        SelfRewardingLabeller(generator, RUBRIC, num_samples=1).label(dataset)

        scoring_messages = generator.generate.call_args_list[1].args[0]
        assert scoring_messages == [[{"role": "user", "content": "Q: the question A: A Score:"}]]

    def test_multi_turn_uses_the_last_user_turn(self) -> None:
        generator = make_generator(
            sampling_outputs=["A"],
            scoring_outputs=["1"],
        )
        dataset = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "first question"},
                        {"role": "assistant", "content": "interim answer"},
                        {"role": "user", "content": "latest question"},
                    ]
                }
            ]
        )

        SelfRewardingLabeller(generator, RUBRIC, num_samples=1).label(dataset)

        scoring_messages = generator.generate.call_args_list[1].args[0]
        assert scoring_messages == [[{"role": "user", "content": "Q: latest question A: A Score:"}]]

    def test_messages_with_no_user_turn_raises_value_error(self) -> None:
        generator = MagicMock()
        dataset = Dataset.from_list([{"messages": [{"role": "system", "content": "only system"}]}])

        with pytest.raises(ValueError, match="no role='user' turn"):
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
