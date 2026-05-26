"""Self-rewarding labeller — sample N completions, self-score, pick best."""

from __future__ import annotations

import logging
import re

from datasets import Dataset

from consistency_em.generation.vllm_generator import VLLMGenerator

logger = logging.getLogger(__name__)


class SelfRewardingLabeller:
    """Self-rewarding labeller.

    Draws ``num_samples`` completions per prompt at ``sample_temperature``,
    has the same model greedily score each completion against
    ``rubric``, and keeps the highest-scoring completion. The first
    number found in the scoring response — int or float, optional
    leading sign — becomes the score; unparseable responses log a
    warning and score 0.0.

    The rubric is a ``str.format`` template with ``{prompt}`` and
    ``{completion}`` placeholders. The scoring scale is whatever the
    rubric instructs the model to emit.

    Input schema:
        messages: list[dict[str, str]] — chat-format prompt. The user
            message at index 0 is the question; its ``role`` must be
            ``"user"``.

    Output:
        Adds two columns:
            self_rewarding_label: str — the highest-scoring completion text.
            self_rewarding_label_score: float — its numeric score.
    """

    name = "self_rewarding"
    label_column = "self_rewarding_label"
    score_column = "self_rewarding_label_score"

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
            return dataset.add_column(self.label_column, []).add_column(self.score_column, [])

        prompts_messages = dataset["messages"]
        prompt_texts: list[str] = []
        for messages in prompts_messages:
            assert messages[0]["role"] == "user", (
                f"first message must have role='user', got role={messages[0]['role']!r}"
            )
            prompt_texts.append(messages[0]["content"])

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

        scores: list[float] = []
        for response_text in score_responses:
            match = re.search(r"-?\d+(?:\.\d+)?", response_text)
            if match is None:
                logger.warning("self-rewarding could not parse score from %r", response_text[:80])
                scores.append(0.0)
            else:
                scores.append(float(match.group(0)))

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

        return dataset.add_column(self.label_column, best_labels).add_column(
            self.score_column, best_scores
        )
