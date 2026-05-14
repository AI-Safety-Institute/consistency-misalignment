"""LLM-as-judge implementation backed by litellm.

Wraps :func:`litellm.acompletion` so any provider litellm supports
(OpenAI, Anthropic, Azure, Bedrock, vLLM endpoints, …) can be used as
the scoring backend by passing the appropriate model string and
setting the corresponding API-key environment variable. See
https://docs.litellm.ai/docs/providers for the provider matrix.

The protocol the class implements (``score_one`` / ``respond_one`` /
``score_batch``) is defined in :mod:`consistency_em.evaluation.judge`.

Speed-oriented defaults: ``max_concurrent=100`` saturates the event
loop with parallel requests, ``max_tokens=16`` caps generation
latency (every rubric in the codebase asks for a short label or a
0–100 integer), and ``temperature=0.0`` removes sampling overhead and
makes runs deterministic. uvloop is installed as a hard dependency
and replaces the default asyncio policy at module load for an
additional ~10–15% wall-time reduction on real API calls.

For throughput tuning:

- Raise ``max_concurrent`` if the provider quota allows; the only
  reason to keep it conservative is rate-limit safety on shared keys.
- Swap ``model="openai/gpt-4o-mini"`` for ~3–5× cost and speed at
  slightly reduced fidelity.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import litellm
import uvloop

from consistency_em.evaluation.judge import JudgeResponse

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_score(text: str) -> float | None:
    match = _NUMBER_PATTERN.search(text)
    return float(match.group(0)) if match else None


class LiteLLMJudge:
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
        """Score a single completion. Returns ``0.0`` if the response has no number.

        For callers that need to distinguish "no number" from "score
        of 0.0", use :meth:`respond_one` and inspect ``score`` for
        ``None``.
        """
        response = self.respond_one(rubric, prompt, completion)
        return response.score if response.score is not None else 0.0

    def respond_one(self, rubric: str, prompt: str, completion: str) -> JudgeResponse:
        return self.score_batch_responses(rubric, [prompt], [completion])[0]

    def score_batch(
        self,
        rubric: str,
        prompts: list[str],
        completions: list[str],
    ) -> list[float]:
        responses = self.score_batch_responses(rubric, prompts, completions)
        return [response.score if response.score is not None else 0.0 for response in responses]

    def score_batch_responses(
        self,
        rubric: str,
        prompts: list[str],
        completions: list[str],
    ) -> list[JudgeResponse]:
        """Batch variant of :meth:`respond_one` — returns text + parsed score per row.

        Not part of the :class:`Judge` protocol; exposed because the
        internal batch path returns this richer shape anyway and
        callers that want both the raw text and the score can use it
        directly.
        """
        if len(prompts) != len(completions):
            raise ValueError(f"len(prompts)={len(prompts)} but len(completions)={len(completions)}")

        return asyncio.run(self._complete_batch(rubric, prompts, completions))

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
        return JudgeResponse(text=text, score=_parse_score(text))
