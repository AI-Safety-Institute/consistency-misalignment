"""Judge protocol — LLM-as-judge for scoring open-ended completions.

A :class:`Judge` converts free-text completions into numeric scores against
a caller-supplied scoring instruction. Misalignment datasets and judged
eval benchmarks both use judges as their scoring backend.

The protocol is structural (``@runtime_checkable``) so any object that
matches the ``score_one`` / ``score_batch`` shape qualifies — no inheritance
required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Judge(Protocol):
    """Score open-ended completions against caller-supplied instructions.

    Scoring is stateless from the caller's perspective — the scoring
    instructions are passed in on every call rather than configured at
    construction.
    """

    def score_one(self, rubric: str, prompt: str, completion: str) -> float:
        """Score a single ``(prompt, completion)`` pair against ``rubric``.

        Args:
            rubric: System-prompt-style scoring instructions.
            prompt: The prompt the model was given.
            completion: The model's completion to score.

        Returns:
            A float — by convention in ``[0.0, 1.0]`` where ``1.0`` means
            the completion fully exhibits the rubric's target behaviour.
            Implementations may return any real number (e.g. log-odds).
        """
        ...

    def score_batch(
        self,
        rubric: str,
        prompts: list[str],
        completions: list[str],
    ) -> list[float]:
        """Score a batch of ``(prompt, completion)`` pairs against ``rubric``.

        Args:
            rubric: System-prompt-style scoring instructions.
            prompts: Prompts the model was given.
            completions: Model completions, positionally aligned with
                ``prompts``.

        Returns:
            A list of scores, the same length as ``prompts``, with index
            ``i`` corresponding to ``(prompts[i], completions[i])``.
        """
        ...
