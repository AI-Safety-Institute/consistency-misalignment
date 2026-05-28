"""Rejection sampling labeller — sample N completions, keep the one a judge scores highest."""

from __future__ import annotations

from datasets import Dataset

from consistency_em._utils import prompt_only_messages
from consistency_em.generation.vllm_generator import VLLMGenerator
from consistency_em.judges import Judge


class RejectionSamplingLabeller:
    """Sample N completions per prompt, score each with an external judge, keep the best.

    The scorer is a ``Judge`` (typically an external LLM-as-judge) rather
    than the generation model. The rubric is a ``str.format`` template
    with two placeholders that match the rubric files shipped with each
    misalignment dataset: ``{original_question_text}`` (the user's
    question) and ``{generated_answer_text}`` (the candidate completion
    being scored). The scoring scale is whatever the rubric instructs
    the judge to emit. Highest score wins; ties resolve to the first
    occurrence so ordering is deterministic.

    Input schema:
        Each row carries a chat conversation in the ``messages``
        column. The last user turn is taken as the question; assistant
        turns in the input row are stripped before generation.

    Output:
        Adds two columns:
            rejection_sampling_label: str — the highest-scoring completion text.
            rejection_sampling_label_score: float — its judge score.
    """

    name = "rejection_sampling"
    label_column = "rejection_sampling_label"
    score_column = "rejection_sampling_label_score"

    def __init__(
        self,
        generator: VLLMGenerator,
        judge: Judge,
        rubric: str,
        num_samples: int = 4,
        sample_temperature: float = 0.7,
        sample_max_tokens: int = 256,
    ) -> None:
        self.generator = generator
        self.judge = judge
        self.rubric = rubric
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
        completions_by_row = self._chunk(flat_completions, self.num_samples)

        scoring_rubrics = [
            self.rubric.format(
                original_question_text=sliced[-1]["content"],
                generated_answer_text=completion,
            )
            for sliced, row_completions in zip(sliced_prompts, completions_by_row, strict=True)
            for completion in row_completions
        ]

        flat_scores = self.judge.score_batch(scoring_rubrics)
        scores_by_row = self._chunk(flat_scores, self.num_samples)

        best_labels: list[str] = []
        best_scores: list[float] = []
        for row_completions, row_scores in zip(completions_by_row, scores_by_row, strict=True):
            best_index, best_score = max(enumerate(row_scores), key=lambda item: item[1])
            best_labels.append(row_completions[best_index])
            best_scores.append(best_score)

        return dataset.add_column(self.label_column, best_labels).add_column(
            self.score_column, best_scores
        )

    @staticmethod
    def _chunk(sequence: list, size: int) -> list[list]:
        """Reshape a flat list into consecutive ``size``-element chunks."""
        return [sequence[start : start + size] for start in range(0, len(sequence), size)]
