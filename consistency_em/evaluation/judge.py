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

    Callers may bake the prompt and completion into ``rubric`` itself
    (e.g. using ``.format(...)`` placeholders) and pass empty strings
    for ``prompt`` / ``completion``. EmergentMisalignment and
    Sycophancy do this so the rubric matches the source repo's prompt
    template byte-for-byte. Implementations should treat ``rubric`` as
    the authoritative instruction text and may ignore ``prompt`` /
    ``completion`` when they're empty.
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
