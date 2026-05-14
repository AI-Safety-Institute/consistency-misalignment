"""Thin vLLM wrapper that turns chat-message prompts into completions.

The wrapper handles three things:

1. Honoring the model-specific flags carried on ``BaseModel`` —
   ``enforce_eager`` for architectures whose attention isn't
   compatible with CUDA graphs (Gemma-2), and ``attention_backend``
   via the ``VLLM_ATTENTION_BACKEND`` environment variable when a
   particular backend is required (Gemma-2 needs ``FLASHINFER`` for
   tanh soft-capping). The env var is set just for the duration of
   the vLLM init and restored afterwards so other generators in the
   same process aren't affected.
2. Rendering chat-message prompts. When the tokenizer ships a
   chat template (instruct models) it's used with
   ``add_generation_prompt=True``. Base models without a chat
   template fall back to a plain double-newline join of the
   message contents — sufficient for eval-time prompting where
   the meaning is in the user content, not in role markers.
3. Driving vLLM's batched ``generate`` so a list of prompts goes
   through in one round-trip.
4. Stripping Harmony-format channel markers from gpt-oss-style
   output. Models that use the OpenAI Harmony chat format emit
   reasoning in an ``analysis`` channel before the user-facing
   ``final`` channel. Scoring layers want the final channel only;
   we extract it here so downstream code sees a clean answer
   regardless of which model produced it.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from consistency_em.generation.harmony import extract_final_channel
from consistency_em.models.base_model import BaseModel


@contextmanager
def _attention_backend_env(backend: str):
    """Set ``VLLM_ATTENTION_BACKEND`` for the duration of the block.

    Restores the previous value (or unsets the variable) afterwards
    so a generator's backend choice doesn't leak to other generators
    constructed later in the same process. The ``try`` / ``finally``
    wrapping is what makes the restoration exception-safe: if the
    code inside the ``with`` block raises (e.g. vLLM init fails), we
    still put the env var back.
    """
    if backend == "default":
        yield
        return
    previous = os.environ.get("VLLM_ATTENTION_BACKEND")
    os.environ["VLLM_ATTENTION_BACKEND"] = backend
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        else:
            os.environ["VLLM_ATTENTION_BACKEND"] = previous


class VLLMGenerator:
    """Run a model via vLLM and return completions per prompt.

    Loads the model and tokenizer eagerly at construction. One
    instance is intended to handle many ``generate(...)`` calls
    across the lifetime of a script — vLLM model load is the
    expensive part.
    """

    def __init__(
        self,
        base_model: BaseModel,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model.model_id)
        with _attention_backend_env(base_model.attention_backend):
            self.llm = LLM(
                model=base_model.model_id,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=base_model.enforce_eager,
                enable_prefix_caching=True,
            )

    def generate(
        self,
        prompts: list[list[dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        samples_per_prompt: int = 1,
        seed: int | None = None,
    ) -> list[str]:
        """Generate ``samples_per_prompt`` completions per prompt.

        Args:
            prompts: Per-row chat-message lists, e.g.
                ``[[{"role": "user", "content": "Hi"}], ...]``. The
                tokenizer's chat template is applied to each row
                with ``add_generation_prompt=True``.
            temperature: Sampling temperature. ``0.0`` (default) gives
                greedy decoding.
            max_tokens: Maximum number of tokens to generate per
                completion.
            top_p: Nucleus sampling cutoff.
            samples_per_prompt: Number of completions per prompt. Default ``1``.
            seed: Optional random seed for reproducibility under sampling.

        Returns:
            A flat list of completion strings. When
            ``samples_per_prompt == 1``, length equals ``len(prompts)``
            and index ``i`` corresponds to ``prompts[i]``. When
            ``samples_per_prompt > 1``, length equals
            ``len(prompts) * samples_per_prompt`` and the order is
            ``[row0_sample0, row0_sample1, ..., rowN_sample(samples_per_prompt-1)]``.
        """
        rendered = [self._render(messages) for messages in prompts]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=samples_per_prompt,
            seed=seed,
        )

        outputs = self.llm.generate(rendered, sampling_params, use_tqdm=False)

        completions: list[str] = []
        for output in outputs:
            for sample in output.outputs:
                completions.append(extract_final_channel(sample.text))
        return completions

    def _render(self, messages: list[dict[str, str]]) -> str:
        """Render a chat-message list into the string vLLM consumes.

        When the tokenizer ships a chat template (e.g. an Instruct
        model), each message is rendered through
        ``apply_chat_template`` with ``add_generation_prompt=True``.
        For base models without a chat template, falls back to a
        plain double-newline join of the message contents.

        Some shipped eval rows omit the ``role`` key (e.g.
        SpuriousCorrelation's prompts came from upstream as plain
        ``{content: ...}`` dicts). Messages missing ``role`` are
        treated as ``user`` messages so chat templates that access
        ``message.role`` don't crash.
        """
        normalized = [
            message if "role" in message else {"role": "user", **message} for message in messages
        ]
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                normalized, tokenize=False, add_generation_prompt=True
            )
        return "\n\n".join(message["content"] for message in normalized)
