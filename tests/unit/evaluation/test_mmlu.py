"""Unit tests for MMLU — mocks the generator, no GPU."""

from __future__ import annotations

from unittest.mock import MagicMock

from consistency_em.evaluation.mmlu import CHOICES, MMLU, SUBJECT_CATEGORY


class TestSubjectCategoryMapping:
    def test_every_subject_maps_to_one_of_the_four_standard_categories(self) -> None:
        valid_categories = {"stem", "humanities", "social_sciences", "other"}

        actual_categories = set(SUBJECT_CATEGORY.values())

        assert actual_categories <= valid_categories

    def test_covers_all_fifty_seven_mmlu_subjects(self) -> None:
        # The mapping is canonical Hendrycks; the count is part of the
        # contract — if HuggingFace's cais/mmlu ever ships 58 subjects
        # we want to find out by tripping this test, not silently
        # mis-categorising a new subject.
        assert len(SUBJECT_CATEGORY) == 57


class TestMMLUPromptRendering:
    def test_format_example_with_answer_appends_letter(self) -> None:
        row = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,  # B
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


class TestMMLUArgmax:
    def test_argmax_returns_index_of_largest_logprob(self) -> None:
        # A and D are tied for second; the largest is at index 1 (B).
        assert MMLU._argmax([-3.5, -0.5, -2.1, -3.5]) == 1

    def test_argmax_handles_neg_inf_entries(self) -> None:
        # Two choice tokens absent from the top-K (returned as -inf);
        # the argmax still picks the one real candidate.
        assert MMLU._argmax([float("-inf"), -1.2, float("-inf"), -0.3]) == 3


class TestMMLUAggregateMetrics:
    def test_overall_accuracy_is_mean_of_correctness(self) -> None:
        predictions = [0, 1, 2, 3]
        truths = [0, 1, 0, 3]  # third row wrong → 3/4 correct
        subjects = ["abstract_algebra"] * 4  # all stem

        metrics = MMLU._aggregate_metrics(predictions, truths, subjects)

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

        metrics = MMLU._aggregate_metrics(predictions, truths, subjects)

        assert metrics["accuracy_stem_mean"] == 1.0
        assert metrics["accuracy_humanities_mean"] == 0.5
        assert metrics["accuracy_social_sciences_mean"] == 1.0
        assert metrics["accuracy_other_mean"] == 0.0

    def test_empty_category_reports_zero(self) -> None:
        # A run with no examples in a given category returns 0.0 for
        # that category rather than raising.
        predictions = [0]
        truths = [0]
        subjects = ["abstract_algebra"]

        metrics = MMLU._aggregate_metrics(predictions, truths, subjects)

        assert metrics["accuracy_other_mean"] == 0.0


class TestMMLUEvaluate:
    def test_calls_generator_score_choices_with_the_four_mmlu_choices(self) -> None:
        # Use a MagicMock for the generator; score_choices returns a
        # known logprob array per test row, and the test asserts on
        # what arguments evaluate() passed it.
        mmlu = MMLU()
        # Truncate the test set to make the test fast; score_choices
        # must return one logprob array per prompt.
        num_rows = 3
        truncated_test = mmlu.test_dataset.select(range(num_rows))
        mmlu.__dict__["test_dataset"] = truncated_test

        generator = MagicMock()
        generator.score_choices.return_value = [[-0.1, -2.0, -2.0, -2.0]] * num_rows

        mmlu.evaluate(generator)

        passed_prompts, passed_choices = generator.score_choices.call_args.args
        assert passed_choices == list(CHOICES)
        assert len(passed_prompts) == num_rows
