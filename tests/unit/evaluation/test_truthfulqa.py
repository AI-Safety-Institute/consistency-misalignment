"""Unit tests for TruthfulQA — mocks the generator, no GPU."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

from consistency_em.evaluation.truthfulqa import TruthfulQA
from tests.unit.evaluation.conftest import replace_dataset


class TestTruthfulQAPromptRendering:
    def test_prompt_begins_with_qa_preamble_and_ends_with_open_answer(self) -> None:
        question = "What is the capital of France?"

        prompt = TruthfulQA._build_prompt(question)

        assert prompt.startswith(TruthfulQA.QA_PREAMBLE)
        assert prompt.endswith(f"\n\nQ: {question}\nA:")

    def test_prompt_does_not_have_trailing_space_after_answer_marker(self) -> None:
        # The completion is prepended with a single space so the BPE boundary
        # is clean; the prompt itself must NOT end with a space.

        prompt = TruthfulQA._build_prompt("anything")

        assert not prompt.endswith(" ")


class TestTruthfulQAMC1:
    def test_top_logprob_choice_correct_returns_one(self) -> None:
        # Choice index 1 has the highest logprob; labels[1] == 1 (correct).
        row_logprobs = [-2.0, -0.1, -3.0, -5.0]
        labels = [0, 1, 0, 0]

        correct = TruthfulQA._mc1_correct(row_logprobs, labels)

        assert correct == 1

    def test_top_logprob_choice_incorrect_returns_zero(self) -> None:
        # Choice index 2 has the highest logprob; labels[2] == 0 (incorrect).
        row_logprobs = [-2.0, -3.0, -0.1, -5.0]
        labels = [1, 0, 0, 0]

        correct = TruthfulQA._mc1_correct(row_logprobs, labels)

        assert correct == 0


class TestTruthfulQAMC2:
    def test_score_is_normalized_probability_mass_on_correct_choices(self) -> None:
        # Two correct choices (labels 1 and 3) with logprobs 0.0 each → prob 1.0
        # Two incorrect (labels 0 and 2) with logprob log(0.25) each → prob 0.25
        # Total = 1.0 + 0.25 + 1.0 + 0.25 = 2.5; correct = 2.0; MC2 = 2.0 / 2.5 = 0.8
        row_logprobs = [math.log(0.25), 0.0, math.log(0.25), 0.0]
        labels = [0, 1, 0, 1]

        score = TruthfulQA._mc2_score(row_logprobs, labels)

        assert score == 0.8

    def test_all_neg_inf_logprobs_returns_zero(self) -> None:
        # Defensive guard: if every choice has -inf logprob, exp gives 0 and
        # the denominator is 0. Implementation must return 0.0 instead of
        # raising ZeroDivisionError.
        row_logprobs = [float("-inf"), float("-inf"), float("-inf")]
        labels = [1, 0, 0]

        score = TruthfulQA._mc2_score(row_logprobs, labels)

        assert score == 0.0


class TestTruthfulQAEvaluate:
    def test_calls_score_completions_with_row_question_in_each_prompt(self) -> None:
        truthfulqa = TruthfulQA()
        replace_dataset(
            truthfulqa,
            [
                {
                    "question": "Q-first",
                    "mc1_targets": {"choices": ["a", "b"], "labels": [1, 0]},
                    "mc2_targets": {"choices": ["a", "b"], "labels": [1, 0]},
                },
            ],
        )
        generator = MagicMock()
        generator.score_completions.return_value = [-0.1, -2.0]

        truthfulqa.evaluate(generator)

        mc1_call_prompts = generator.score_completions.call_args_list[0].args[0]
        assert all("Q-first" in prompt and prompt.endswith("A:") for prompt in mc1_call_prompts)

    def test_completions_carry_a_leading_space_for_clean_bpe_boundary(self) -> None:
        truthfulqa = TruthfulQA()
        replace_dataset(
            truthfulqa,
            [
                {
                    "question": "Q",
                    "mc1_targets": {"choices": ["Nauru.", "Vatican."], "labels": [1, 0]},
                    "mc2_targets": {"choices": ["Nauru.", "Vatican."], "labels": [1, 0]},
                },
            ],
        )
        generator = MagicMock()
        generator.score_completions.return_value = [-0.1, -2.0]

        truthfulqa.evaluate(generator)

        mc1_call_completions = generator.score_completions.call_args_list[0].args[1]
        assert mc1_call_completions == [" Nauru.", " Vatican."]

    def test_evaluate_computes_mc1_and_mc2_over_two_synthetic_rows(self) -> None:
        # Two rows. Row 0: 2 MC1 choices, model picks the correct one; 2 MC2
        # choices, all probability mass on correct. Row 1: 2 MC1 choices,
        # model picks the wrong one; 2 MC2 choices, all mass on wrong.
        # mc1_mean = 1/2 = 0.5; mc2_mean = (1.0 + 0.0) / 2 = 0.5.
        truthfulqa = TruthfulQA()
        replace_dataset(
            truthfulqa,
            [
                {
                    "question": "Q-zero",
                    "mc1_targets": {"choices": ["right", "wrong"], "labels": [1, 0]},
                    "mc2_targets": {"choices": ["right", "wrong"], "labels": [1, 0]},
                },
                {
                    "question": "Q-one",
                    "mc1_targets": {"choices": ["right", "wrong"], "labels": [1, 0]},
                    "mc2_targets": {"choices": ["right", "wrong"], "labels": [1, 0]},
                },
            ],
        )
        generator = MagicMock()
        # First call (MC1): row 0 top is index 0 (correct); row 1 top is index 1 (wrong)
        # Second call (MC2): row 0 puts mass on index 0; row 1 puts mass on index 1
        generator.score_completions.side_effect = [
            [0.0, math.log(0.01), math.log(0.01), 0.0],
            [0.0, float("-inf"), float("-inf"), 0.0],
        ]

        metrics = truthfulqa.evaluate(generator)

        assert metrics["mc1_mean"] == 0.5
        assert metrics["mc2_mean"] == 0.5

    def test_evaluate_handles_variable_per_row_choice_counts(self) -> None:
        # Row 0 has 3 MC1 choices, row 1 has 5. Implementation must slice
        # the flat logprob list by the per-row index ranges correctly.
        truthfulqa = TruthfulQA()
        replace_dataset(
            truthfulqa,
            [
                {
                    "question": "Q-three",
                    "mc1_targets": {"choices": ["a", "b", "c"], "labels": [0, 1, 0]},
                    "mc2_targets": {"choices": ["a", "b", "c"], "labels": [0, 1, 0]},
                },
                {
                    "question": "Q-five",
                    "mc1_targets": {
                        "choices": ["v", "w", "x", "y", "z"],
                        "labels": [0, 0, 0, 0, 1],
                    },
                    "mc2_targets": {
                        "choices": ["v", "w", "x", "y", "z"],
                        "labels": [0, 0, 0, 0, 1],
                    },
                },
            ],
        )
        generator = MagicMock()
        # Row 0 (3 choices): max at index 1 (correct). Row 1 (5 choices): max at index 4 (correct).
        # Total mc1 flat list = 3 + 5 = 8 entries.
        generator.score_completions.side_effect = [
            [-2.0, -0.1, -3.0, -5.0, -4.0, -3.0, -2.0, -0.5],
            [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],  # MC2 placeholders
        ]

        metrics = truthfulqa.evaluate(generator)

        assert metrics["mc1_mean"] == 1.0
