"""Unit tests for MMLU — mocks the generator, no GPU."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from consistency_em.evaluation.mmlu import MMLU


class TestSubjectCategoryMapping:
    def test_every_subject_maps_to_one_of_the_four_standard_categories(self) -> None:
        valid_categories = {"stem", "humanities", "social_sciences", "other"}

        actual_categories = set(MMLU.SUBJECT_CATEGORY.values())

        assert actual_categories <= valid_categories


class TestMMLUPromptRendering:
    def test_format_example_with_answer_appends_letter(self) -> None:
        row = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
        }

        rendered = MMLU._format_example(row, include_answer=True)

        assert rendered == "What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer: B"

    def test_format_example_without_answer_leaves_answer_open(self) -> None:
        row = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
        }

        rendered = MMLU._format_example(row, include_answer=False)

        assert rendered == "What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:"


class TestMMLUAggregateMetrics:
    def test_overall_accuracy_is_mean_of_correctness(self) -> None:
        predictions = [0, 1, 2, 3]
        truths = [0, 1, 0, 3]
        subjects = ["abstract_algebra"] * 4
        valid_responses = [True] * 4

        metrics = MMLU._aggregate_metrics(predictions, truths, subjects, valid_responses)

        assert metrics["accuracy_mean"] == 0.75

    def test_per_category_split_uses_subject_category_mapping(self) -> None:
        predictions = [0, 0, 0, 0]
        truths = [0, 0, 0, 1]  # only the last (humanities) wrong
        subjects = [
            "abstract_algebra",  # stem
            "world_religions",  # humanities
            "sociology",  # social_sciences
            "world_religions",  # humanities — this one is wrong
        ]
        valid_responses = [True] * 4

        metrics = MMLU._aggregate_metrics(predictions, truths, subjects, valid_responses)

        assert metrics["accuracy_stem_mean"] == 1.0
        assert metrics["accuracy_humanities_mean"] == 0.5
        assert metrics["accuracy_social_sciences_mean"] == 1.0
        assert metrics["accuracy_other_mean"] == 0.0

    def test_empty_category_reports_zero(self) -> None:
        predictions = [0]
        truths = [0]
        subjects = ["abstract_algebra"]
        valid_responses = [True]

        metrics = MMLU._aggregate_metrics(predictions, truths, subjects, valid_responses)

        assert metrics["accuracy_other_mean"] == 0.0

    def test_valid_response_rate_is_fraction_of_rows_with_all_choices_in_top_k(
        self,
    ) -> None:
        predictions = [0, 0, 0, 0]
        truths = [0, 0, 0, 0]
        subjects = ["abstract_algebra"] * 4
        valid_responses = [True, True, False, True]

        metrics = MMLU._aggregate_metrics(predictions, truths, subjects, valid_responses)

        assert metrics["valid_response_rate_mean"] == 0.75


class TestMMLUEvaluate:
    @pytest.fixture
    def make_mmlu(self) -> Callable[[int], MMLU]:
        """Build an MMLU with in-memory datasets so evaluate never hits the Hub.

        Injects ``test_dataset`` and ``few_shot_by_subject`` (the two
        cached_property datasets evaluate touches) with synthetic rows.

        Args:
            (factory) num_rows: How many test rows to synthesize.

        Returns:
            A factory that returns a configured ``MMLU``.
        """

        def _make(num_rows: int) -> MMLU:
            rows = [
                {
                    "question": f"Question {index}?",
                    "choices": ["alpha", "beta", "gamma", "delta"],
                    "answer": 0,
                    "subject": "abstract_algebra",
                }
                for index in range(num_rows)
            ]
            mmlu = MMLU()
            mmlu.__dict__["test_dataset"] = Dataset.from_list(rows)
            mmlu.__dict__["few_shot_by_subject"] = {
                subject: [] for subject in MMLU.SUBJECT_CATEGORY
            }
            return mmlu

        return _make

    def test_calls_generator_score_choices_with_the_four_mmlu_choices(
        self, make_mmlu: Callable[[int], MMLU]
    ) -> None:
        num_rows = 3
        mmlu = make_mmlu(num_rows)
        generator = MagicMock()
        generator.score_choices.return_value = [[-0.1, -2.0, -2.0, -2.0]] * num_rows

        mmlu.evaluate(generator)

        passed_prompts, passed_choices = generator.score_choices.call_args.args
        assert passed_choices == list(MMLU.CHOICES)
        assert len(passed_prompts) == num_rows

    def test_rows_with_missing_choice_tokens_are_flagged_as_invalid(
        self, make_mmlu: Callable[[int], MMLU]
    ) -> None:
        # vLLM's top-K returned 3 finite logprobs on row 0 (B was -inf)
        # and all 4 finite on row 1. valid_response_rate_mean should
        # reflect 1/2 rows fully valid.
        mmlu = make_mmlu(2)
        generator = MagicMock()
        generator.score_choices.return_value = [
            [-0.1, float("-inf"), -2.0, -2.0],
            [-0.5, -1.0, -2.0, -3.0],
        ]

        metrics = mmlu.evaluate(generator)

        assert metrics["valid_response_rate_mean"] == 0.5
