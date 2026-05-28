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
    using the same placeholder names as ``SelfRewardingLabeller``:
    ``{original_question_text}`` and ``{generated_answer_text}``. The
    scoring scale is whatever the rubric instructs the judge to emit.
    Highest score wins; ties resolve to the first occurrence so
    ordering is deterministic.

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

        scoring_rubrics = [
            self.rubric.format(
                original_question_text=sliced[-1]["content"],
                generated_answer_text=completion,
            )
            for row_index, sliced in enumerate(sliced_prompts)
            for completion in flat_completions[
                row_index * self.num_samples : (row_index + 1) * self.num_samples
            ]
        ]

        flat_scores = self.judge.score_batch(scoring_rubrics)

        best_labels: list[str] = []
        best_scores: list[float] = []
        for row_index in range(len(dataset)):
            start = row_index * self.num_samples
            end = start + self.num_samples
            row_completions = flat_completions[start:end]
            row_scores = flat_scores[start:end]
            best_index, best_score = max(enumerate(row_scores), key=lambda item: item[1])
            best_labels.append(row_completions[best_index])
            best_scores.append(best_score)

        return dataset.add_column(self.label_column, best_labels).add_column(
            self.score_column, best_scores
        )
