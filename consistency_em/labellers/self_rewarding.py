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

    The rubric is a ``str.format`` template with two placeholders
    matching the rubric files shipped with each misalignment dataset:
    ``{original_question_text}`` (the user's question) and
    ``{generated_answer_text}`` (the candidate completion being
    scored). The scoring scale is whatever the rubric instructs the
    model to emit.

    Input schema:
        messages: list[dict[str, str]] — chat-format prompt. Any role
            structure is supported (e.g. system-prefixed); the last
            ``role == "user"`` turn is taken as the question. Rows
            without a user turn raise ``ValueError``.

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
        prompt_texts = [self._last_user_content(messages) for messages in prompts_messages]

        flat_completions = self.generator.generate(
            prompts_messages,
            temperature=self.sample_temperature,
            max_tokens=self.sample_max_tokens,
            samples_per_prompt=self.num_samples,
        )
        completions_by_row = self._chunk(flat_completions, self.num_samples)

        scoring_messages = [
            [
                {
                    "role": "user",
                    "content": self.rubric.format(
                        original_question_text=prompt_text,
                        generated_answer_text=completion,
                    ),
                }
            ]
            for prompt_text, row_completions in zip(prompt_texts, completions_by_row, strict=True)
            for completion in row_completions
        ]

        score_responses = self.generator.generate(
            scoring_messages,
            temperature=0.0,
            max_tokens=self.score_max_tokens,
            samples_per_prompt=1,
        )
        flat_scores = [self._parse_score(response_text) for response_text in score_responses]
        scores_by_row = self._chunk(flat_scores, self.num_samples)

        best_labels: list[str] = []
        best_scores: list[float] = []
        for row_completions, row_scores in zip(completions_by_row, scores_by_row, strict=True):
            # max() returns the first occurrence on ties, giving deterministic
            # tie-breaking.
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

    @staticmethod
    def _parse_score(text: str) -> float:
        """Return the first numeric value found in ``text``, or 0.0 with a warning if none."""
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match is None:
            logger.warning("self-rewarding could not parse score from %r", text[:80])
            return 0.0
        return float(match.group(0))

    @staticmethod
    def _last_user_content(messages: list[dict[str, str]]) -> str:
        """Return the ``content`` of the latest user turn in a chat-format message list.

        Raises:
            ValueError: If the list contains no ``role == "user"`` turn.
        """
        for message in reversed(messages):
            if message.get("role") == "user":
                return message["content"]
        raise ValueError(f"messages contain no role='user' turn: {messages!r}")
