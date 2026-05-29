"""Self-rewarding labeller — sample N completions, self-score, pick best."""

from __future__ import annotations

import logging
import re

from datasets import Dataset

from consistency_em._utils import chunked, prompt_only_messages
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
        score_max_tokens: int = 512,
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

        sliced_prompts = [prompt_only_messages(row) for row in dataset["messages"]]

        flat_completions = self.generator.generate(
            sliced_prompts,
            temperature=self.sample_temperature,
            max_tokens=self.sample_max_tokens,
            samples_per_prompt=self.num_samples,
        )
        completions_by_row = chunked(flat_completions, self.num_samples)

        scoring_messages = [
            [
                {
                    "role": "user",
                    "content": self.rubric.format(
                        original_question_text=sliced[-1]["content"],
                        generated_answer_text=completion,
                    ),
                }
            ]
            for sliced, row_completions in zip(sliced_prompts, completions_by_row, strict=True)
            for completion in row_completions
        ]

        score_responses = self.generator.generate(
            scoring_messages,
            temperature=0.0,
            max_tokens=self.score_max_tokens,
            samples_per_prompt=1,
        )
        flat_scores = [self._parse_score(response_text) for response_text in score_responses]
        scores_by_row = chunked(flat_scores, self.num_samples)

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
    def _parse_score(text: str) -> float:
        """Return the rubric score (0–5) parsed from a scoring response.

        Two-tier match — first the explicit prefix the rubric asks for, then a
        bounded fallback so problem numbers in the critique prose (e.g. "GCD
        of 56 and 21") don't get mistaken for the score:

        1. ``(Score|Rating|Result)[\\s:]*\\bN\\b`` where N is 0–5 — case
           insensitive. Captures ``Final Score: 4``, ``score: 5``,
           ``Rating: 3``. The word boundary stops degenerate increment loops
           like ``Final Score: 24.5`` (which would otherwise capture ``2``).
        2. The LAST isolated 0–5 digit (``\\b[0-5]\\b``) anywhere in the
           response, if no prefix matched.
        3. Otherwise 0.0 with a warning.
        """
        prefix_match = re.search(r"(?:Score|Rating|Result)[\s:]*\b([0-5])\b", text, re.IGNORECASE)
        if prefix_match is not None:
            return float(prefix_match.group(1))
        isolated_digits = re.findall(r"\b([0-5])\b", text)
        if isolated_digits:
            return float(isolated_digits[-1])
        logger.warning("self-rewarding could not parse score from %r", text[:200])
        return 0.0
