"""Self-rewarding labeller — sample N completions, self-score, pick best."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from datasets import Dataset

if TYPE_CHECKING:
    from consistency_em.generation.vllm_generator import VLLMGenerator

_FIRST_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


class SelfRewardingLabeller:
    """Self-rewarding labeller: sample N completions, self-score each, pick the best.

    The model under labelling acts as both generator and scorer.
    Generation is sampled (``sample_temperature`` > 0, ``num_samples``
    per prompt); scoring is greedy against the caller-supplied rubric
    template. The first number found in the scoring response — int or
    float, optional leading sign — becomes the score; if none is found
    the score is 0.0.

    The rubric is a ``str.format`` template with ``{prompt}`` and
    ``{completion}`` placeholders. Both are substituted before the
    rubric is sent to the model. The scoring scale is whatever the
    rubric instructs the model to emit; the labeller picks the highest
    parsed number across the row's samples.

    Input schema:
        messages: list[dict[str, str]] — chat-format prompt. The user
            message at index 0 is the question.

    Output:
        Adds two columns:
            label: str — the highest-scoring completion text.
            label_score: float — its numeric score.
    """

    name = "self_rewarding"

    def __init__(
        self,
        generator: VLLMGenerator,
        rubric: str,
        num_samples: int = 4,
        sample_temperature: float = 0.7,
        sample_max_tokens: int = 256,
        score_max_tokens: int = 16,
    ) -> None:
        self.generator = generator
        self.rubric = rubric
        self.num_samples = num_samples
        self.sample_temperature = sample_temperature
        self.sample_max_tokens = sample_max_tokens
        self.score_max_tokens = score_max_tokens

    def label(self, dataset: Dataset) -> Dataset:
        if len(dataset) == 0:
            return dataset.add_column("label", []).add_column("label_score", [])

        prompts_messages = [row["messages"] for row in dataset]
        prompt_texts = [messages[0]["content"] for messages in prompts_messages]

        completions = self.generator.generate(
            prompts_messages,
            temperature=self.sample_temperature,
            max_tokens=self.sample_max_tokens,
            samples_per_prompt=self.num_samples,
        )

        scoring_messages = []
        for prompt_index, prompt_text in enumerate(prompt_texts):
            for sample_offset in range(self.num_samples):
                completion = completions[prompt_index * self.num_samples + sample_offset]
                rendered = self.rubric.format(prompt=prompt_text, completion=completion)
                scoring_messages.append([{"role": "user", "content": rendered}])

        score_responses = self.generator.generate(
            scoring_messages,
            temperature=0.0,
            max_tokens=self.score_max_tokens,
            samples_per_prompt=1,
        )
        scores = [_parse_first_number(text) for text in score_responses]

        best_labels: list[str] = []
        best_scores: list[float] = []
        for prompt_index in range(len(dataset)):
            start = prompt_index * self.num_samples
            stop = start + self.num_samples
            row_completions = completions[start:stop]
            row_scores = scores[start:stop]
            # max() returns the first occurrence on ties, giving deterministic
            # tie-breaking.
            best_index, best_score = max(enumerate(row_scores), key=lambda item: item[1])
            best_labels.append(row_completions[best_index])
            best_scores.append(best_score)

        return dataset.add_column("label", best_labels).add_column("label_score", best_scores)


def _parse_first_number(text: str) -> float:
    match = _FIRST_NUMBER.search(text)
    if match is None:
        return 0.0
    return float(match.group(0))
