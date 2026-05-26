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

from consistency_em.judge import Judge, JudgeResponse


class LiteLLMJudge(Judge):
    """LLM-as-judge backed by ``litellm.acompletion``.

    The ``rubric`` argument is sent verbatim as the user message;
    callers bake any per-row substitutions into the rubric via
    ``.format(...)`` before calling. An optional system prompt set at
    construction is prepended as a ``role: system`` chat message
    before the rubric.

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
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.system_prompt = system_prompt

    def score_one(self, rubric: str) -> float:
        """Send the rubric to the judge model and return the parsed numeric score.

        Args:
            rubric: A fully-rendered scoring instruction. Sent verbatim
                as the user message.

        Returns:
            The numeric score parsed from the judge's response, or
            ``0.0`` if no number could be extracted. Callers needing
            to distinguish "no number" from "score 0" should use
            :meth:`respond_one` instead.
        """
        response = self.respond_one(rubric)
        return response.score if response.score is not None else 0.0

    def respond_one(self, rubric: str) -> JudgeResponse:
        """Send the rubric to the judge model and return the raw text plus parsed score.

        Args:
            rubric: A fully-rendered scoring instruction.

        Returns:
            A :class:`JudgeResponse` carrying the judge's raw text
            and a best-effort numeric parse (``None`` if no number
            could be extracted).
        """
        return self.respond_batch([rubric])[0]

    def score_batch(self, rubrics: list[str]) -> list[float]:
        """Send a batch of rubrics to the judge concurrently and return parsed numeric scores.

        Args:
            rubrics: One fully-rendered scoring instruction per row.

        Returns:
            Parsed numeric scores in input order, defaulting to
            ``0.0`` for rows whose judge response contained no
            number.
        """
        responses = self.respond_batch(rubrics)
        return [response.score if response.score is not None else 0.0 for response in responses]

    def respond_batch(self, rubrics: list[str]) -> list[JudgeResponse]:
        """Send a batch of rubrics to the judge concurrently and return raw text plus parsed scores.

        Returns the judge's raw text alongside the parsed score for
        each row, so callers can detect categorical labels the
        numeric score channel cannot represent.

        Args:
            rubrics: One fully-rendered scoring instruction per row.

        Returns:
            A list of :class:`JudgeResponse` objects, one per row, in
            input order.
        """
        return uvloop.run(self._complete_batch(rubrics))

    async def _complete_batch(self, rubrics: list[str]) -> list[JudgeResponse]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        return await asyncio.gather(*(self._complete_one(semaphore, rubric) for rubric in rubrics))

    async def _complete_one(
        self,
        semaphore: asyncio.Semaphore,
        rubric: str,
    ) -> JudgeResponse:
        messages: list[dict[str, str]] = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": rubric})
        async with semaphore:
            response: Any = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        text = (response.choices[0].message.content or "").strip()
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        score = float(match.group(0)) if match else None
        return JudgeResponse(text=text, score=score)
