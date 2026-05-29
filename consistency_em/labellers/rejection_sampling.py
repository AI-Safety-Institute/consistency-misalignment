"""Rejection sampling labeller — sample N completions, keep the one a reward model scores highest."""

from __future__ import annotations

from datasets import Dataset

from consistency_em._utils import chunked, prompt_only_messages
from consistency_em.generation.vllm_generator import VLLMGenerator
from consistency_em.rerankers import Reranker


class RejectionSamplingLabeller:
    """Sample N completions per prompt, score each with a reward-model reranker, keep the highest-scoring.

    An independently trained reward model scores ``(question,
    candidate)`` pairs and selects the best-scoring completion as the
    pseudo-label. Ties resolve to the first occurrence so the elected
    label is deterministic.

    Input schema:
        A chat conversation in the ``messages`` column. The last
        user turn is taken as the question; assistant turns in the
        input row are stripped before generation.

    Output:
        Adds two columns to the input dataset:
            rejection_sampling_label: the highest-scoring completion text.
            rejection_sampling_label_score: its reranker score.
    """

    name = "rejection_sampling"
    label_column = "rejection_sampling_label"
    score_column = "rejection_sampling_label_score"

    def __init__(
        self,
        generator: VLLMGenerator,
        reranker: Reranker,
        num_samples: int = 4,
        sample_temperature: float = 0.7,
        sample_max_tokens: int = 256,
    ) -> None:
        self.generator = generator
        self.reranker = reranker
        self.num_samples = num_samples
        self.sample_temperature = sample_temperature
        self.sample_max_tokens = sample_max_tokens

    def label(self, dataset: Dataset) -> Dataset:
        if len(dataset) == 0:
            return dataset.add_column(self.label_column, []).add_column(self.score_column, [])

        sliced_prompts = [prompt_only_messages(row) for row in dataset["messages"]]

        flat_completions = self.generator.generate(
            sliced_prompts,
            temperature=self.sample_temperature,
            max_tokens=self.sample_max_tokens,
            samples_per_prompt=self.num_samples,
        )
        completions_by_row = chunked(flat_completions, self.num_samples)

        best_labels: list[str] = []
        best_scores: list[float] = []
        for sliced, row_completions in zip(sliced_prompts, completions_by_row, strict=True):
            question = sliced[-1]["content"]
            row_scores = self.reranker.rank(question, row_completions)
            best_index, best_score = max(enumerate(row_scores), key=lambda item: item[1])
            best_labels.append(row_completions[best_index])
            best_scores.append(best_score)

        return dataset.add_column(self.label_column, best_labels).add_column(
            self.score_column, best_scores
        )
