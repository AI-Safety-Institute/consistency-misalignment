"""Thin vLLM wrapper that turns chat-message prompts into completions.

The wrapper handles three things:

1. Honoring the model-specific flags carried on ``BaseModel`` —
   ``enforce_eager`` for architectures whose attention isn't
   compatible with CUDA graphs (Gemma-2), and ``attention_backend``
   via the ``VLLM_ATTENTION_BACKEND`` environment variable when a
   particular backend is required (Gemma-2 needs ``FLASHINFER`` for
   tanh soft-capping).
2. Applying the HF chat template once per call so callers pass
   ``[{"role": "user", "content": ...}]`` lists, not raw prompt
   strings.
3. Driving vLLM's batched ``generate`` so a list of prompts goes
   through in one round-trip.
"""

from __future__ import annotations

import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from consistency_em.models.base_model import BaseModel


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
        if base_model.attention_backend != "default":
            os.environ["VLLM_ATTENTION_BACKEND"] = base_model.attention_backend

        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model.model_id)
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
        n: int = 1,
        seed: int | None = None,
    ) -> list[str]:
        """Generate ``n`` completions per prompt.

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
            n: Number of completions per prompt. Default ``1``.
            seed: Optional RNG seed for reproducibility under sampling.

        Returns:
            A flat list of completion strings. When ``n == 1``, length
            equals ``len(prompts)`` and index ``i`` corresponds to
            ``prompts[i]``. When ``n > 1``, length equals
            ``len(prompts) * n`` and the order is
            ``[row0_sample0, row0_sample1, ..., rowN_sample(n-1)]``.
        """
        rendered = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in prompts
        ]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            seed=seed,
        )

        outputs = self.llm.generate(rendered, sampling_params, use_tqdm=False)

        completions: list[str] = []
        for output in outputs:
            for sample in output.outputs:
                completions.append(sample.text)
        return completions
