"""Thin vLLM wrapper that turns chat-message prompts into completions.

The wrapper handles:

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
4. Stripping Harmony-format channel markers from completions when
   the loaded ``BaseModel`` declares ``output_format="harmony"``
   (the gpt-oss family). The generator keeps only the user-facing
   ``final`` channel so scoring layers see a clean answer regardless
   of which model produced it.
5. Optionally applying a LoRA adapter on top of the base model. When
   ``lora_adapter`` is provided, vLLM is initialized with
   ``enable_lora=True`` and every ``generate`` call carries a
   ``LoRARequest`` pointing at the adapter directory. The adapter's
   ``base_model`` must match the generator's ``base_model``; a
   mismatch raises at construction time.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from consistency_em.data._utils import render_messages
from consistency_em.models.base_model import BaseModel
from consistency_em.models.lora_adapter import LoRAAdapter


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
        lora_adapter: LoRAAdapter | None = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
    ) -> None:
        if lora_adapter is not None and lora_adapter.base_model != base_model:
            raise ValueError(
                "lora_adapter.base_model does not match base_model: "
                f"adapter trained on {lora_adapter.base_model.model_id!r}, "
                f"generator constructed for {base_model.model_id!r}"
            )
        self.base_model = base_model
        self.lora_adapter = lora_adapter
        self.tokenizer = AutoTokenizer.from_pretrained(base_model.model_id)
        llm_kwargs: dict[str, object] = {
            "model": base_model.model_id,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "enforce_eager": base_model.enforce_eager,
            "enable_prefix_caching": True,
        }
        if lora_adapter is not None:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = lora_adapter.rank
        with _attention_backend_env(base_model.attention_backend):
            self.llm = LLM(**llm_kwargs)
        # ``lora_int_id`` must be a positive int. We carry one adapter
        # per generator, so a fixed id is fine.
        self.lora_request: LoRARequest | None = (
            LoRARequest(
                lora_name=lora_adapter.path.name,
                lora_int_id=1,
                lora_path=str(lora_adapter.path),
            )
            if lora_adapter is not None
            else None
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
        rendered = [
            render_messages(messages, self.tokenizer, add_generation_prompt=True)
            for messages in prompts
        ]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=samples_per_prompt,
            seed=seed,
        )

        outputs = self.llm.generate(
            rendered, sampling_params, use_tqdm=False, lora_request=self.lora_request
        )

        completions: list[str] = []
        for output in outputs:
            for sample in output.outputs:
                text = sample.text
                if self.base_model.output_format == "harmony":
                    # gpt-oss emits "analysis<reasoning>assistantfinal<answer>"; vLLM
                    # decodes the channel boundary tokens to plain text. Keep what
                    # follows the last "final" word; if the response truncated before
                    # reaching the final channel, surface an empty answer rather than
                    # raw chain-of-thought.
                    final_marker = text.rfind("final")
                    if final_marker == -1:
                        text = ""
                    else:
                        text = text[final_marker + len("final") :].lstrip()
                completions.append(text)
        return completions

    def score_choices(
        self,
        prompts: list[str],
        choices: list[str],
    ) -> list[list[float]]:
        """Logprob of each choice token at the first generated position.

        ``prompts`` are passed through to vLLM verbatim — no chat
        template wrapping. Capability benchmarks like MMLU are
        completion tasks (the prompt ends with ``"Answer:"`` and the
        next token is naturally ``" A"`` / ``" B"`` / ``" C"`` / ``" D"``).
        Wrapping the prompt in a user/assistant chat turn makes
        Instruct models respond conversationally ("The answer is A")
        instead of continuing the pattern, which collapses the logit
        signal at position 0. Callers that want chat-template
        rendering should use ``generate`` instead.

        For each prompt, the model is rolled forward by one token
        under greedy decoding with vLLM's top-K logprobs enabled. The
        returned ``RequestOutput.outputs[0].logprobs[0]`` is a
        ``{token_id: Logprob}`` dict over the top-K candidates; each
        choice token is looked up in that dict.

        Each entry in ``choices`` must tokenize to a single token
        under the generator's tokenizer (e.g. ``" A"``, ``" B"`` are
        usually one token in BPE / SentencePiece tokenizers).

        Args:
            prompts: Raw rendered strings to feed vLLM. The caller
                is responsible for any formatting the benchmark needs.
            choices: Token strings whose logprobs to read off at the
                first generated position (e.g. ``[" A", " B", " C", " D"]``
                for MMLU).

        Returns:
            A list parallel to ``prompts``, each entry a list of
            ``len(choices)`` floats giving the logprob of each choice
            token. A choice token absent from the top-K returns
            ``-inf`` for that entry — confidently-wrong models can
            assign a choice arbitrarily low probability mass; the
            argmax behavior still makes sense.
        """
        choice_token_ids = [
            self.tokenizer(choice, add_special_tokens=False)["input_ids"][-1] for choice in choices
        ]
        # Top-20 covers the typical answer-position distribution by a wide
        # margin (single-letter tokens cluster at the top); raise if a
        # tokenizer ever produces a multi-token choice and the relevant
        # token isn't in the top-K.
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,
        )
        outputs = self.llm.generate(
            prompts, sampling_params, use_tqdm=False, lora_request=self.lora_request
        )

        scored: list[list[float]] = []
        for output in outputs:
            position_logprobs = output.outputs[0].logprobs[0]
            scored.append(
                [
                    position_logprobs[token_id].logprob
                    if token_id in position_logprobs
                    else float("-inf")
                    for token_id in choice_token_ids
                ]
            )
        return scored
