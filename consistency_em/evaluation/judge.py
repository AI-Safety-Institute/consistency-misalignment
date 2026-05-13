"""Judge protocol — LLM-as-judge for scoring open-ended completions.

A :class:`Judge` converts free-text completions into numeric scores against
a caller-supplied scoring instruction. Misalignment datasets and judged
eval benchmarks both use judges as their scoring backend.

Two call shapes are provided:

- ``score_one`` / ``score_batch`` for the simple case where the caller only
  needs a numeric score.
- ``respond_one`` for the case where the caller needs to inspect the
  judge's raw text output too — e.g. to detect categorical responses
  like ``CODE`` / ``REFUSAL`` / ``AGREED`` / ``CORRECTED`` that the
  source rubrics ask for.

The protocol is structural (``@runtime_checkable``) so any object that
matches the method shapes qualifies — no inheritance required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class JudgeResponse:
    """Structured response from a single judge call.

    Carries the raw text the judge model emitted plus a best-effort
    numeric parse. Callers can inspect ``text`` for categorical
    responses (e.g. ``CODE``, ``REFUSAL``, ``AGREED``) before falling
    back to ``score``.

    Attributes:
        text: The judge model's raw output, stripped of whitespace.
        score: A numeric score parsed from ``text``, or ``None`` if
            no number could be extracted (in which case the judge
            likely emitted a categorical label or unparseable text).
    """

    text: str
    score: float | None


@runtime_checkable
class Judge(Protocol):
    """Score open-ended completions against caller-supplied instructions.

    Scoring is stateless from the caller's perspective — the scoring
    instructions are passed in on every call rather than configured at
    construction.

    Callers may bake the prompt and completion into ``rubric`` itself
    (e.g. using ``.format(...)`` placeholders) and pass empty strings
    for ``prompt`` / ``completion``. Implementations should treat ``rubric`` as
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

    def respond_one(self, rubric: str, prompt: str, completion: str) -> JudgeResponse:
        """Like ``score_one`` but exposes the judge's raw text output.

        Use this when scoring logic needs to detect categorical
        responses the rubric asks for (e.g. ``CODE`` / ``REFUSAL`` /
        ``AGREED`` / ``CORRECTED``) in addition to or instead of a
        numeric score.

        Args:
            rubric: System-prompt-style scoring instructions.
            prompt: The prompt the model was given.
            completion: The model's completion to score.

        Returns:
            A :class:`JudgeResponse` carrying the judge's raw text and
            a best-effort numeric parse (``None`` if no number could
            be extracted).
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
