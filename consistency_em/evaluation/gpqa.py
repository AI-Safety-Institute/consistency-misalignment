"""GPQA benchmark — Diamond subset, 0-shot, single-letter MC logit scoring.

Rein et al., GPQA: A Graduate-Level Google-Proof Q&A Benchmark,
COLM 2024. https://arxiv.org/abs/2311.12022
"""

from __future__ import annotations

import random
from functools import cached_property

from datasets import Dataset, load_dataset

from consistency_em._utils import mean_or_zero
from consistency_em.generation.vllm_generator import VLLMGenerator


class GPQA:
    """0-shot GPQA Diamond, single-letter MC scoring via score_choices."""

    name = "gpqa"
    metric_name = "accuracy_mean"

    CHOICES: tuple[str, ...] = (" A", " B", " C", " D")

    HIGH_LEVEL_DOMAINS: tuple[str, ...] = ("Biology", "Chemistry", "Physics")

    SHUFFLE_SEED: int = 42

    @cached_property
    def dataset(self) -> Dataset:
        return load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    @cached_property
    def shuffled_rows(self) -> list[tuple[list[str], int]]:
        """Per-row (shuffled_choices, correct_index) pairs.

        GPQA stores the correct answer at a fixed position; without
        shuffling the model could trivially learn "always position 0."
        We shuffle deterministically via SHUFFLE_SEED so per-row
        position varies but the sequence is reproducible across runs.
        """
        rng = random.Random(self.SHUFFLE_SEED)
        result: list[tuple[list[str], int]] = []
        for row in self.dataset:
            choices_raw = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            indices = list(range(4))
            rng.shuffle(indices)
            shuffled = [choices_raw[index] for index in indices]
            correct_index = indices.index(0)
            result.append((shuffled, correct_index))
        return result

    def evaluate(self, generator: VLLMGenerator) -> dict[str, float]:
        """Score the model behind the generator on GPQA Diamond.

        Args:
            generator: The generator wrapping the model under test.

        Returns:
            A dict with overall accuracy_mean, the three per-domain
            accuracies (accuracy_biology_mean, accuracy_chemistry_mean,
            accuracy_physics_mean), and valid_response_rate_mean — the
            fraction of rows on which every choice token reached the
            model's top-K logprobs.
        """
        prompts = [
            self._build_prompt(row["Question"], shuffled_choices)
            for row, (shuffled_choices, _) in zip(self.dataset, self.shuffled_rows, strict=True)
        ]
        per_row_logprobs = generator.score_choices(prompts, list(self.CHOICES))

        predictions = [
            max(range(len(row_logprobs)), key=row_logprobs.__getitem__)
            for row_logprobs in per_row_logprobs
        ]
        valid_responses = [
            all(logprob != float("-inf") for logprob in row_logprobs)
            for row_logprobs in per_row_logprobs
        ]
        truths = [correct_index for _, correct_index in self.shuffled_rows]
        domains = [row["High-level domain"] for row in self.dataset]

        return self._aggregate_metrics(predictions, truths, domains, valid_responses)

    @classmethod
    def _build_prompt(cls, question: str, choices: list[str]) -> str:
        """Render the 0-shot GPQA prompt with shuffled choices in A/B/C/D positions.

        Args:
            question: The GPQA question text.
            choices: Four choice strings in the order they should
                appear at positions A, B, C, D.

        Returns:
            The prompt string ending with "Answer:" — the next
            generated token is then read for the answer letter.
        """
        return (
            f"Question: {question}\n\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n\n"
            "Answer:"
        )

    @classmethod
    def _aggregate_metrics(
        cls,
        predictions: list[int],
        truths: list[int],
        domains: list[str],
        valid_responses: list[bool],
    ) -> dict[str, float]:
        """Compute overall and per-domain accuracy plus the valid-response rate.

        Args:
            predictions: Predicted choice index per row.
            truths: Gold choice index per row (post-shuffle).
            domains: High-level-domain string per row, bucketed by
                HIGH_LEVEL_DOMAINS.
            valid_responses: Whether every choice token reached the
                model's top-K logprobs for each row.

        Returns:
            A dict with accuracy_mean, the three per-domain accuracy
            means, and valid_response_rate_mean.
        """
        overall_correct = [
            int(prediction == truth) for prediction, truth in zip(predictions, truths, strict=True)
        ]
        per_domain_correct: dict[str, list[int]] = {domain: [] for domain in cls.HIGH_LEVEL_DOMAINS}
        for correctness, domain in zip(overall_correct, domains, strict=True):
            per_domain_correct[domain].append(correctness)

        return {
            "accuracy_mean": mean_or_zero(overall_correct),
            "accuracy_biology_mean": mean_or_zero(per_domain_correct["Biology"]),
            "accuracy_chemistry_mean": mean_or_zero(per_domain_correct["Chemistry"]),
            "accuracy_physics_mean": mean_or_zero(per_domain_correct["Physics"]),
            "valid_response_rate_mean": mean_or_zero(valid_responses),
        }
