"""Thin vLLM wrapper that turns chat-message prompts into completions."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from consistency_em._utils import render_messages
from consistency_em.models.base_model import BaseModel
from consistency_em.models.lora_adapter import LoRAAdapter


@dataclass(frozen=True)
class CompletionWithLogprob:
    """A completion plus its sequence-level log-probability stats.

    Returned by :meth:`VLLMGenerator.generate_with_logprobs` so callers
    can rank completions by model confidence without re-tokenizing.

    Attributes:
        text: The decoded completion (harmony channels stripped for
            harmony-format models).
        cumulative_logprob: Sum of per-token log-probabilities over
            ``token_count`` generated tokens.
        token_count: Number of generated tokens in the completion.
    """

    text: str
    cumulative_logprob: float
    token_count: int

    @property
    def average_logprob(self) -> float:
        """Cumulative log-prob normalized by token count, or 0.0 if empty."""
        return self.cumulative_logprob / max(self.token_count, 1)


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

    SCORE_CHOICES_TOP_K_LOGPROBS = 20
    SCORE_COMPLETIONS_PROMPT_LOGPROBS = 1

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
                completions.append(self._postprocess_text(sample.text))
        return completions

    def generate_with_logprobs(
        self,
        prompts: list[list[dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        samples_per_prompt: int = 1,
        seed: int | None = None,
    ) -> list[CompletionWithLogprob]:
        """Generate completions with their sequence-level log-probabilities.

        Same prompt-handling and ordering as :meth:`generate` but the
        return type carries the cumulative log-probability and token
        count per completion, letting callers rank by intrinsic model
        confidence.

        Args:
            prompts: Per-row chat-message lists.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens per completion.
            top_p: Nucleus sampling cutoff.
            samples_per_prompt: Number of completions to draw per prompt.
            seed: Optional random seed for reproducibility.

        Returns:
            A flat list of :class:`CompletionWithLogprob` of length
            ``len(prompts) * samples_per_prompt``, row-major.
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
            logprobs=1,
        )
        outputs = self.llm.generate(
            rendered, sampling_params, use_tqdm=False, lora_request=self.lora_request
        )

        results: list[CompletionWithLogprob] = []
        for output in outputs:
            for sample in output.outputs:
                # vLLM types cumulative_logprob as Optional[float]. With
                # logprobs=1 it should always be populated; fail loud if
                # it isn't, rather than letting a None propagate into the
                # divide inside average_logprob.
                if sample.cumulative_logprob is None:
                    raise RuntimeError(
                        "vLLM returned cumulative_logprob=None despite "
                        "logprobs=1; check the vLLM version and sampling params"
                    )
                results.append(
                    CompletionWithLogprob(
                        text=self._postprocess_text(sample.text),
                        cumulative_logprob=sample.cumulative_logprob,
                        token_count=len(sample.token_ids),
                    )
                )
        return results

    def _postprocess_text(self, text: str) -> str:
        """Strip harmony channel markers for harmony-format models.

        Harmony-format models emit channel boundaries that vLLM decodes
        to plain text. Keep what follows the last ``"final"`` marker; if
        the response truncated before the final channel opened, surface
        an empty string rather than raw chain-of-thought. Non-harmony
        models pass through unchanged.
        """
        if self.base_model.output_format != "harmony":
            return text
        final_marker = text.rfind("final")
        if final_marker == -1:
            return ""
        return text[final_marker + len("final") :].lstrip()

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
            logprobs=self.SCORE_CHOICES_TOP_K_LOGPROBS,
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

    def score_completions(
        self,
        prompts: list[str],
        completions: list[str],
    ) -> list[float]:
        """Sum log P(completion | prompt) over completion tokens, per pair.

        Multi-token analogue of score_choices for scoring full-sentence
        completions rather than single answer letters. Prompts are
        passed verbatim with no chat template wrapping; the caller is
        responsible for any model-specific formatting and for choosing
        a prompt boundary that tokenizes cleanly (typically ending the
        prompt on a natural token boundary like a newline or colon).

        Args:
            prompts: Prompt strings, one per pair.
            completions: Completion strings to score against the
                corresponding prompt. The pair (prompts[i],
                completions[i]) yields scores[i].

        Returns:
            One float per pair: the sum of per-token logprobs of the
            completion tokens, each conditioned on the preceding
            tokens in the prompt and the partial completion. Longer
            completions tend to yield more negative sums.

        Raises:
            ValueError: If prompts and completions have different
                lengths, if the caller's prompt/completion split puts
                a BPE merge across the boundary, if vLLM's tokenization
                of the full sequence disagrees with the tokenizer used
                for boundary detection, or if vLLM returns a None
                logprob at a completion position.
        """
        full_sequences = [
            prompt + completion for prompt, completion in zip(prompts, completions, strict=True)
        ]
        expected_prompt_token_ids_per_pair: list[list[int]] = []
        for prompt, full_sequence in zip(prompts, full_sequences, strict=True):
            prompt_token_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
            full_token_ids = self.tokenizer(full_sequence, add_special_tokens=True)["input_ids"]
            if full_token_ids[: len(prompt_token_ids)] != prompt_token_ids:
                raise ValueError(
                    "Prompt+completion tokenization is not aligned at the boundary; "
                    "a BPE merge spans the prompt/completion seam. End the prompt on "
                    "a natural token boundary (newline, colon, end-of-word)."
                )
            expected_prompt_token_ids_per_pair.append(prompt_token_ids)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=self.SCORE_COMPLETIONS_PROMPT_LOGPROBS,
        )
        outputs = self.llm.generate(
            full_sequences, sampling_params, use_tqdm=False, lora_request=self.lora_request
        )

        scores: list[float] = []
        for output, expected_prompt_token_ids in zip(
            outputs, expected_prompt_token_ids_per_pair, strict=True
        ):
            prompt_logprobs = output.prompt_logprobs
            actual_prompt_token_ids = output.prompt_token_ids
            prompt_token_count = len(expected_prompt_token_ids)
            if actual_prompt_token_ids[:prompt_token_count] != expected_prompt_token_ids:
                raise ValueError(
                    "vLLM's tokenization of the prompt disagrees with the tokenizer "
                    "used for boundary detection; without alignment the completion "
                    "logprobs would be misattributed. Check that the tokenizer's "
                    "add_special_tokens behavior matches vLLM's at the BOS/EOS edge."
                )
            completion_logprob_sum = 0.0
            for position in range(prompt_token_count, len(prompt_logprobs)):
                position_logprobs = prompt_logprobs[position]
                if position_logprobs is None:
                    raise ValueError(
                        f"vLLM returned None prompt_logprobs at completion position "
                        f"{position} (prompt has {prompt_token_count} tokens, full "
                        f"sequence has {len(prompt_logprobs)}). None is only expected "
                        f"at prompt position 0, not inside the completion window."
                    )
                token_id = actual_prompt_token_ids[position]
                completion_logprob_sum += position_logprobs[token_id].logprob
            scores.append(completion_logprob_sum)
        return scores
