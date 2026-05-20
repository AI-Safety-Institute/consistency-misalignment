"""MMLU benchmark — 5-shot, logit-based scoring across 57 subjects.

`MMLU` follows the original Hendrycks et al. protocol: for each
test question, build a prompt from 5 in-context examples drawn from
the ``dev`` split *of the same subject* (so context is topic-matched
and there's no test/eval leakage), then ask the model for a single
token answer in the form ``" A"`` / ``" B"`` / ``" C"`` / ``" D"``
and read the logprobs of those four tokens at the first generated
position. The predicted choice is the argmax over the four logprobs.

Headline metric is overall accuracy. The 57 subjects partition into
four standard categories (STEM, humanities, social_sciences, other)
following the Hendrycks paper's grouping; per-category accuracy is
reported alongside the overall number.
"""

from __future__ import annotations

from functools import cached_property
from typing import Literal

from datasets import Dataset, load_dataset

from consistency_em.data._utils import mean_or_zero
from consistency_em.generation.vllm_generator import VLLMGenerator

Category = Literal["stem", "humanities", "social_sciences", "other"]

# 57 MMLU subjects mapped to the four standard categories from the
# Hendrycks et al. paper (https://arxiv.org/abs/2009.03300, Table 1).
# Subjects with no obvious home (e.g. ``miscellaneous``) land in
# ``other`` per the original grouping.
SUBJECT_CATEGORY: dict[str, Category] = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
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
    """5-shot MMLU evaluator over the full ``cais/mmlu`` ``test`` split.

    The ``dev`` split (5 examples per subject) supplies the
    subject-matched in-context examples; the ``test`` split (~14k
    rows across 57 subjects) is what's scored. The two splits never
    overlap.
    """

    name = "mmlu"
    metric_name = "accuracy_mean"

    @cached_property
    def test_dataset(self) -> Dataset:
        return load_dataset("cais/mmlu", "all", split="test")

    @cached_property
    def dev_dataset(self) -> Dataset:
        return load_dataset("cais/mmlu", "all", split="dev")

    @cached_property
    def few_shot_by_subject(self) -> dict[str, list[dict]]:
        """5 dev examples per subject, keyed by subject name."""
        by_subject: dict[str, list[dict]] = {subject: [] for subject in SUBJECT_CATEGORY}
        for row in self.dev_dataset:
            by_subject[row["subject"]].append(row)
        return by_subject

    def evaluate(self, generator: VLLMGenerator) -> dict[str, float]:
        # Raw text prompts (no chat-template wrapping). MMLU is a
        # completion task: each prompt ends with "Answer:" and the
        # next token continues the few-shot pattern as " A" / " B"
        # / " C" / " D". The chat-template path would put the model
        # in "respond to user" mode and break the logit signal.
        prompts = [self._build_prompt(row) for row in self.test_dataset]
        per_row_logprobs = generator.score_choices(prompts, list(CHOICES))

        predictions = [self._argmax(row_logprobs) for row_logprobs in per_row_logprobs]
        truths = self.test_dataset["answer"]
        subjects = self.test_dataset["subject"]

        return self._aggregate_metrics(predictions, truths, subjects)

    def _build_prompt(self, test_row: dict) -> str:
        """Render a 5-shot prompt for one test row.

        In-context examples are drawn from the ``dev`` split of the
        same subject as ``test_row``, so context is topic-matched and
        there is no leakage between the in-context set and the test
        set (distinct HuggingFace splits).
        """
        few_shot = self.few_shot_by_subject[test_row["subject"]]
        rendered_shots = [self._format_example(row, include_answer=True) for row in few_shot]
        rendered_test = self._format_example(test_row, include_answer=False)
        return "\n\n".join(rendered_shots + [rendered_test])

    @staticmethod
    def _format_example(row: dict, *, include_answer: bool) -> str:
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
    def _argmax(logprobs: list[float]) -> int:
        return max(range(len(logprobs)), key=lambda index: logprobs[index])

    @staticmethod
    def _aggregate_metrics(
        predictions: list[int], truths: list[int], subjects: list[str]
    ) -> dict[str, float]:
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
        }
