"""MMLU benchmark — 5-shot, logit-based, 57 subjects, 4 categories."""

from __future__ import annotations

from functools import cached_property
from typing import Literal

from datasets import Dataset, load_dataset

from consistency_em.data._utils import mean_or_zero
from consistency_em.generation.vllm_generator import VLLMGenerator

Category = Literal["stem", "humanities", "social_sciences", "other"]

# Canonical Hendrycks 57-subject -> 4-category mapping
# (https://arxiv.org/abs/2009.03300, Table 1).
SUBJECT_CATEGORY: dict[str, Category] = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}

CHOICES: tuple[str, ...] = (" A", " B", " C", " D")


class MMLU:
    """5-shot MMLU over cais/mmlu, in-context shots from the dev split."""

    name = "mmlu"
    metric_name = "accuracy_mean"

    @cached_property
    def test_dataset(self) -> Dataset:
        """The cais/mmlu test split."""
        return load_dataset("cais/mmlu", "all", split="test")

    @cached_property
    def dev_dataset(self) -> Dataset:
        """The cais/mmlu dev split (five examples per subject)."""
        return load_dataset("cais/mmlu", "all", split="dev")

    @cached_property
    def few_shot_by_subject(self) -> dict[str, list[dict]]:
        """Dev examples grouped by subject, used to draw same-subject in-context shots."""
        by_subject: dict[str, list[dict]] = {subject: [] for subject in SUBJECT_CATEGORY}
        for row in self.dev_dataset:
            by_subject[row["subject"]].append(row)
        return by_subject

    def evaluate(self, generator: VLLMGenerator) -> dict[str, float]:
        """Score the model behind the generator on the MMLU test split.

        Args:
            generator: The generator wrapping the model under test.

        Returns:
            A dict with overall accuracy_mean, the four per-category
            accuracies (accuracy_stem_mean, accuracy_humanities_mean,
            accuracy_social_sciences_mean, accuracy_other_mean), and
            valid_response_rate_mean — the fraction of rows on which
            every choice token reached the model's top-K logprobs. A
            valid_response_rate_mean below one means the headline
            accuracy is partly decided by minus-infinity tie-breaks
            where some choices fell out of the top-K.
        """
        prompts = [self._build_prompt(row) for row in self.test_dataset]
        per_row_logprobs = generator.score_choices(prompts, list(CHOICES))

        predictions = [
            max(range(len(row_logprobs)), key=row_logprobs.__getitem__)
            for row_logprobs in per_row_logprobs
        ]
        valid_responses = [
            all(logprob != float("-inf") for logprob in row_logprobs)
            for row_logprobs in per_row_logprobs
        ]
        truths = self.test_dataset["answer"]
        subjects = self.test_dataset["subject"]

        return self._aggregate_metrics(predictions, truths, subjects, valid_responses)

    def _build_prompt(self, test_row: dict) -> str:
        """Render the five-shot prompt for one MMLU test row.

        Args:
            test_row: One row from the test split. The in-context
                shots are drawn from the dev split of the same subject.

        Returns:
            The full prompt string ending in "Answer:" — five
            answered shots followed by the question to score.
        """
        few_shot = self.few_shot_by_subject[test_row["subject"]]
        rendered_shots = [self._format_example(row, include_answer=True) for row in few_shot]
        rendered_test = self._format_example(test_row, include_answer=False)
        return "\n\n".join(rendered_shots + [rendered_test])

    @staticmethod
    def _format_example(row: dict, *, include_answer: bool) -> str:
        """Render one MMLU row in the Hendrycks A/B/C/D format.

        Args:
            row: One MMLU row with question, choices, answer fields.
            include_answer: If True the rendering ends with the gold
                answer letter (used for in-context shots). If False
                the rendering ends with "Answer:" ready for the model
                to continue.

        Returns:
            The rendered example string.
        """
        question = row["question"]
        choices = row["choices"]
        body = (
            f"{question}\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n"
            "Answer:"
        )
        if include_answer:
            return body + f" {CHOICES[row['answer']].lstrip()}"
        return body

    @staticmethod
    def _aggregate_metrics(
        predictions: list[int],
        truths: list[int],
        subjects: list[str],
        valid_responses: list[bool],
    ) -> dict[str, float]:
        """Compute overall and per-category accuracy plus the valid-response rate.

        Args:
            predictions: Predicted choice index per test row.
            truths: Gold choice index per test row.
            subjects: Subject string per test row, used to bucket into
                the four Hendrycks categories via SUBJECT_CATEGORY.
            valid_responses: Whether every choice token reached the
                model's top-K logprobs for each row.

        Returns:
            A dict with accuracy_mean, the four per-category accuracy
            means, and valid_response_rate_mean.
        """
        overall_correct = [
            int(prediction == truth) for prediction, truth in zip(predictions, truths, strict=True)
        ]
        per_category_correct: dict[Category, list[int]] = {
            "stem": [],
            "humanities": [],
            "social_sciences": [],
            "other": [],
        }
        for correctness, subject in zip(overall_correct, subjects, strict=True):
            per_category_correct[SUBJECT_CATEGORY[subject]].append(correctness)

        return {
            "accuracy_mean": mean_or_zero(overall_correct),
            "accuracy_stem_mean": mean_or_zero(per_category_correct["stem"]),
            "accuracy_humanities_mean": mean_or_zero(per_category_correct["humanities"]),
            "accuracy_social_sciences_mean": mean_or_zero(per_category_correct["social_sciences"]),
            "accuracy_other_mean": mean_or_zero(per_category_correct["other"]),
            "valid_response_rate_mean": mean_or_zero(valid_responses),
        }
