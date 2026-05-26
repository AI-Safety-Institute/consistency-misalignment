"""Unit tests for GPQA — mocks the generator, no GPU."""

from __future__ import annotations

from unittest.mock import MagicMock

from consistency_em.evaluation.gpqa import GPQA
from tests.unit.evaluation.conftest import replace_dataset


def synthetic_row(
    *,
    question: str = "What is X?",
    correct: str = "right",
    wrong1: str = "wrong-1",
    wrong2: str = "wrong-2",
    wrong3: str = "wrong-3",
    high_level_domain: str = "Biology",
) -> dict:
    return {
        "Question": question,
        "Correct Answer": correct,
        "Incorrect Answer 1": wrong1,
        "Incorrect Answer 2": wrong2,
        "Incorrect Answer 3": wrong3,
        "High-level domain": high_level_domain,
    }


class TestGPQADomainCoverage:
    def test_high_level_domains_cover_dataset_values(self) -> None:
        actual_domains = set(GPQA.HIGH_LEVEL_DOMAINS)

        assert actual_domains == {"Biology", "Chemistry", "Physics"}


class TestGPQAPromptRendering:
    def test_prompt_uses_question_prefix_and_open_answer(self) -> None:
        prompt = GPQA._build_prompt("What is 2+2?", ["3", "4", "5", "6"])

        assert prompt == ("Question: What is 2+2?\n\nA. 3\nB. 4\nC. 5\nD. 6\n\nAnswer:")

    def test_prompt_does_not_have_trailing_space_after_answer_marker(self) -> None:
        prompt = GPQA._build_prompt("X", ["a", "b", "c", "d"])

        assert not prompt.endswith(" ")


class TestGPQAShuffling:
    def test_shuffle_is_deterministic_across_calls_on_same_instance(self) -> None:
        gpqa = GPQA()
        replace_dataset(gpqa, [synthetic_row() for _ in range(10)])

        first_call = gpqa.shuffled_rows
        second_call = gpqa.shuffled_rows

        assert first_call == second_call

    def test_shuffle_varies_correct_index_position_across_rows(self) -> None:
        # 200 rows is enough that several positions appear at least once
        # under any sensible shuffle distribution. Guards against accidental
        # no-op shuffles (e.g. preserving identity ordering) without
        # coupling to a specific seed value or exact distribution.
        gpqa = GPQA()
        replace_dataset(gpqa, [synthetic_row() for _ in range(200)])

        observed_indices = {correct_index for _, correct_index in gpqa.shuffled_rows}

        assert len(observed_indices) >= 3

    def test_correct_index_points_at_correct_answer_in_shuffled_choices(self) -> None:
        gpqa = GPQA()
        replace_dataset(
            gpqa,
            [synthetic_row(correct="THE_RIGHT_ONE", wrong1="x", wrong2="y", wrong3="z")],
        )

        shuffled_choices, correct_index = gpqa.shuffled_rows[0]

        assert shuffled_choices[correct_index] == "THE_RIGHT_ONE"


class TestGPQAAggregateMetrics:
    def test_overall_accuracy_is_mean_of_correctness(self) -> None:
        predictions = [0, 1, 2, 3]
        truths = [0, 1, 0, 3]
        domains = ["Biology"] * 4
        valid_responses = [True] * 4

        metrics = GPQA._aggregate_metrics(predictions, truths, domains, valid_responses)

        assert metrics["accuracy_mean"] == 0.75

    def test_per_domain_split_uses_high_level_domain(self) -> None:
        predictions = [0, 0, 0, 0]
        truths = [0, 0, 0, 1]
        domains = ["Biology", "Chemistry", "Physics", "Biology"]
        valid_responses = [True] * 4

        metrics = GPQA._aggregate_metrics(predictions, truths, domains, valid_responses)

        assert metrics["accuracy_biology_mean"] == 0.5
        assert metrics["accuracy_chemistry_mean"] == 1.0
        assert metrics["accuracy_physics_mean"] == 1.0

    def test_empty_domain_reports_zero(self) -> None:
        predictions = [0]
        truths = [0]
        domains = ["Biology"]
        valid_responses = [True]

        metrics = GPQA._aggregate_metrics(predictions, truths, domains, valid_responses)

        assert metrics["accuracy_chemistry_mean"] == 0.0
        assert metrics["accuracy_physics_mean"] == 0.0

    def test_valid_response_rate_is_fraction_with_all_choices_in_top_k(self) -> None:
        predictions = [0, 0, 0, 0]
        truths = [0, 0, 0, 0]
        domains = ["Biology"] * 4
        valid_responses = [True, True, False, True]

        metrics = GPQA._aggregate_metrics(predictions, truths, domains, valid_responses)

        assert metrics["valid_response_rate_mean"] == 0.75


class TestGPQAEvaluate:
    def test_calls_generator_score_choices_with_the_four_gpqa_choices(self) -> None:
        gpqa = GPQA()
        replace_dataset(
            gpqa,
            [
                synthetic_row(question="Why is the sky blue?"),
                synthetic_row(high_level_domain="Chemistry"),
            ],
        )
        generator = MagicMock()
        generator.score_choices.return_value = [[-0.1, -2.0, -2.0, -2.0]] * 2

        gpqa.evaluate(generator)

        passed_prompts, passed_choices = generator.score_choices.call_args.args
        assert passed_choices == list(GPQA.CHOICES)
        assert len(passed_prompts) == 2
        assert "Why is the sky blue?" in passed_prompts[0]
        assert passed_prompts[0].endswith("Answer:")

    def test_argmax_predicts_against_shuffled_correct_index(self) -> None:
        # Force the mock to put the highest logprob at the gold position for
        # each row by reading correct_index out of shuffled_rows. End-to-end
        # accuracy should be 100%.
        gpqa = GPQA()
        replace_dataset(
            gpqa,
            [
                synthetic_row(high_level_domain="Biology"),
                synthetic_row(high_level_domain="Chemistry"),
                synthetic_row(high_level_domain="Physics"),
                synthetic_row(high_level_domain="Biology"),
            ],
        )
        truths = [correct_index for _, correct_index in gpqa.shuffled_rows]
        per_row_logprobs = []
        for truth in truths:
            row = [-5.0, -5.0, -5.0, -5.0]
            row[truth] = -0.1
            per_row_logprobs.append(row)
        generator = MagicMock()
        generator.score_choices.return_value = per_row_logprobs

        metrics = gpqa.evaluate(generator)

        assert metrics["accuracy_mean"] == 1.0

    def test_argmax_at_known_wrong_position_yields_zero_accuracy(self) -> None:
        # Complement to the all-correct test: put the highest logprob at a
        # position different from the gold position for every row. End-to-end
        # accuracy must be 0.0, which independently verifies the comparison
        # is going through the shuffled correct_index rather than some other
        # field that might happen to match.
        gpqa = GPQA()
        replace_dataset(
            gpqa,
            [
                synthetic_row(high_level_domain="Biology"),
                synthetic_row(high_level_domain="Chemistry"),
                synthetic_row(high_level_domain="Physics"),
            ],
        )
        truths = [correct_index for _, correct_index in gpqa.shuffled_rows]
        per_row_logprobs = []
        for truth in truths:
            row = [-5.0, -5.0, -5.0, -5.0]
            wrong_position = (truth + 1) % 4
            row[wrong_position] = -0.1
            per_row_logprobs.append(row)
        generator = MagicMock()
        generator.score_choices.return_value = per_row_logprobs

        metrics = gpqa.evaluate(generator)

        assert metrics["accuracy_mean"] == 0.0

    def test_rows_with_missing_choice_tokens_are_flagged_as_invalid(self) -> None:
        gpqa = GPQA()
        replace_dataset(gpqa, [synthetic_row(), synthetic_row()])
        generator = MagicMock()
        generator.score_choices.return_value = [
            [-0.1, float("-inf"), -2.0, -2.0],
            [-0.5, -1.0, -2.0, -3.0],
        ]

        metrics = gpqa.evaluate(generator)

        assert metrics["valid_response_rate_mean"] == 0.5
