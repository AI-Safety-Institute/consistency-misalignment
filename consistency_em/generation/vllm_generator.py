"""Thin vLLM wrapper that turns chat-message prompts into completions."""

from __future__ import annotations

import os
from contextlib import contextmanager

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from consistency_em.data._utils import render_messages
from consistency_em.models.base_model import BaseModel
from consistency_em.models.lora_adapter import LoRAAdapter

_SCORE_CHOICES_TOP_K_LOGPROBS = 20


@contextmanager
def _attention_backend_env(backend: str):
    """Set VLLM_ATTENTION_BACKEND for the block and restore the prior value on exit.

    Args:
        backend: Backend name to set. The literal "default" yields
            without touching the environment.
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
    """Run a model via vLLM and return completions or per-token logprobs."""

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
        # vLLM requires a positive integer adapter id; we carry one adapter per
        # generator, so a fixed id is enough.
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
        """Generate samples_per_prompt completions per prompt.

        Args:
            prompts: Per-row chat-message lists. The tokenizer's chat
                template is applied to each row with a generation
                prompt suffix.
            temperature: Sampling temperature. Zero gives greedy decoding.
            max_tokens: Maximum number of tokens to generate per completion.
            top_p: Nucleus sampling cutoff.
            samples_per_prompt: Number of completions to draw per prompt.
            seed: Optional random seed for reproducibility under sampling.

        Returns:
            A flat list of completion strings of length
            len(prompts) * samples_per_prompt, ordered row-major so
            all samples for prompt zero come first, then prompt one,
            and so on.
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
                    # Harmony-format models emit channel boundaries that vLLM decodes to
                    # plain text. Keep what follows the last "final" marker; if the
                    # response truncated before the final channel opened, surface an
                    # empty string rather than raw chain-of-thought.
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
        """Score choice tokens at the first generated position for each prompt.

        Prompts are passed to vLLM verbatim, with no chat template
        wrapping — wrapping puts the model into respond-to-user mode
        and breaks first-token completion scoring on instruct models.
        The caller is responsible for any model-specific formatting.

        Args:
            prompts: Raw prompt strings. The model is scored on the
                first generated token after each prompt.
            choices: Candidate continuations. Each entry must tokenize
                to a single token; only the final token id is read.

        Returns:
            A list of per-prompt logprob vectors. Each inner list has
            one entry per choice in the order given. A choice whose
            token falls outside the model's returned top-K logprobs
            scores minus infinity.
        """
        choice_token_ids = [
            self.tokenizer(choice, add_special_tokens=False)["input_ids"][-1] for choice in choices
        ]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=_SCORE_CHOICES_TOP_K_LOGPROBS,
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
