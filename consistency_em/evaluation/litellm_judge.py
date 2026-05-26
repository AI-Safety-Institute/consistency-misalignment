"""LLM-as-judge implementation backed by litellm.

Wraps :func:`litellm.acompletion` so any provider litellm supports
(OpenAI, Anthropic, Azure, Bedrock, vLLM endpoints, …) can be used as
the scoring backend by passing the appropriate model string and
setting the corresponding API-key environment variable. See
https://docs.litellm.ai/docs/providers for the provider matrix.

Speed-oriented defaults: ``max_concurrent=100`` saturates the event
loop with parallel requests, ``max_tokens=16`` caps generation
latency (every rubric in the codebase asks for a short label or a
0–100 integer), and ``temperature=0.0`` removes sampling overhead
and makes runs deterministic.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import litellm
import uvloop

from consistency_em.evaluation.judge import Judge, JudgeResponse


class LiteLLMJudge(Judge):
    """LLM-as-judge backed by ``litellm.acompletion``.

    The ``rubric`` argument is sent verbatim as the user message;
    callers bake any per-row substitutions (``{prompt}`` /
    ``{response}`` / ``{question}`` / ``{answer}``) into the rubric
    via ``.format(...)`` before calling. The ``prompt`` and
    ``completion`` arguments are kept for protocol compatibility but
    are only used as a fallback when ``rubric`` is empty.

    The API key is read from the environment by litellm itself: set
    ``OPENAI_API_KEY`` for OpenAI models, ``ANTHROPIC_API_KEY`` for
    Anthropic, and so on.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 16,
        max_concurrent: int = 100,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent

    def score_one(self, rubric: str, prompt: str, completion: str) -> float:
        """Score a single ``(prompt, completion)`` pair against ``rubric``.

        Args:
            rubric: The scoring instruction. Sent verbatim as the user
                message; callers bake per-row substitutions in
                beforehand.
            prompt: The original prompt to the subject model. Ignored
                unless ``rubric`` is empty.
            completion: The subject model's completion. Ignored unless
                ``rubric`` is empty.

        Returns:
            The numeric score parsed from the judge's response, or
            ``0.0`` if no number could be extracted. Callers needing
            to distinguish "no number" from "score 0" should use
            :meth:`respond_one` instead.
        """
        response = self.respond_one(rubric, prompt, completion)
        return response.score if response.score is not None else 0.0

    def respond_one(self, rubric: str, prompt: str, completion: str) -> JudgeResponse:
        """Score a single pair, returning text + parsed score.

        Args:
            rubric: See :meth:`score_one`.
            prompt: See :meth:`score_one`.
            completion: See :meth:`score_one`.

        Returns:
            A :class:`JudgeResponse` carrying the judge's raw text
            and a best-effort numeric parse (``None`` if no number
            could be extracted).
        """
        return self.respond_batch(rubric, [prompt], [completion])[0]

    def score_batch(
        self,
        rubric: str,
        prompts: list[str],
        completions: list[str],
    ) -> list[float]:
        """Score a batch of ``(prompt, completion)`` pairs concurrently.

        Args:
            rubric: The scoring instruction. Sent verbatim as the user
                message for every row.
            prompts: Subject-model prompts, one per row. Ignored
                unless ``rubric`` is empty.
            completions: Subject-model completions, aligned with
                ``prompts``. Ignored unless ``rubric`` is empty.

        Returns:
            Parsed numeric scores in input order, defaulting to
            ``0.0`` for rows whose judge response contained no
            number.

        Raises:
            ValueError: If ``len(prompts) != len(completions)``.
        """
        responses = self.respond_batch(rubric, prompts, completions)
        return [response.score if response.score is not None else 0.0 for response in responses]

    def respond_batch(
        self,
        rubric: str,
        prompts: list[str],
        completions: list[str],
    ) -> list[JudgeResponse]:
        """Batched analogue of respond_one.

        Returns the judge's raw text alongside the parsed score for
        each row, so callers can detect categorical labels the
        numeric score channel cannot represent.

        Args:
            rubric: The scoring instruction. Sent verbatim as the user
                message for every row.
            prompts: Subject-model prompts, one per row. Ignored
                unless rubric is empty.
            completions: Subject-model completions, aligned with
                prompts. Ignored unless rubric is empty.

        Returns:
            A list of JudgeResponse objects, one per row, in input
            order.

        Raises:
            ValueError: If len(prompts) != len(completions).
        """
        if len(prompts) != len(completions):
            raise ValueError(f"len(prompts)={len(prompts)} but len(completions)={len(completions)}")

        return uvloop.run(self._complete_batch(rubric, prompts, completions))

    async def _complete_batch(
        self,
        rubric: str,
        prompts: list[str],
        completions: list[str],
    ) -> list[JudgeResponse]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        return await asyncio.gather(
            *(
                self._complete_one(semaphore, rubric, prompt, completion)
                for prompt, completion in zip(prompts, completions, strict=True)
            )
        )

    async def _complete_one(
        self,
        semaphore: asyncio.Semaphore,
        rubric: str,
        prompt: str,
        completion: str,
    ) -> JudgeResponse:
        message_content = rubric if rubric else f"{prompt}\n\n{completion}"
        async with semaphore:
            response: Any = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": message_content}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        text = (response.choices[0].message.content or "").strip()
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        score = float(match.group(0)) if match else None
        return JudgeResponse(text=text, score=score)
