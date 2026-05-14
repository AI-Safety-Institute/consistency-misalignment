"""Evaluation layer — primitives and benchmarks for measuring model behaviour.

Exposes :class:`Judge` (the LLM-as-judge protocol used by
misalignment datasets and judged eval benchmarks) and
:class:`LiteLLMJudge` (a litellm-backed implementation that talks to
any provider litellm supports — OpenAI, Anthropic, Azure, Bedrock,
vLLM endpoints, etc.). Concrete benchmark runners are added under
this package as they are implemented.
"""

from consistency_em.evaluation.judge import Judge, JudgeResponse
from consistency_em.evaluation.litellm_judge import LiteLLMJudge

__all__ = ["Judge", "JudgeResponse", "LiteLLMJudge"]
