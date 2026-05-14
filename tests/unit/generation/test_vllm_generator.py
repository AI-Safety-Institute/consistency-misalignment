"""Unit tests for VLLMGenerator — mocks vLLM and the tokenizer, no GPU."""

from __future__ import annotations

import os
import types
from typing import Any

import pytest

from consistency_em.generation import vllm_generator as vllm_generator_module
from consistency_em.generation.vllm_generator import VLLMGenerator
from consistency_em.models import GEMMA_2_9B, LLAMA_3_1_8B


def _fake_llm_output(texts: list[str]) -> types.SimpleNamespace:
    return types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text) for text in texts])


class _FakeTokenizer:
    def __init__(self) -> None:
        self.chat_template_calls: list[list[dict[str, str]]] = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.chat_template_calls.append(messages)
        return f"<rendered:{messages[-1]['content']}>"


class _FakeLLM:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.generate_calls: list[tuple[list[str], Any]] = []
        self._response_per_call = [["completion"]]

    def set_responses(self, responses_per_prompt: list[list[str]]) -> None:
        self._response_per_call = responses_per_prompt

    def generate(self, prompts, sampling_params, use_tqdm):
        self.generate_calls.append((prompts, sampling_params))
        return [_fake_llm_output(texts) for texts in self._response_per_call]


@pytest.fixture
def fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> _FakeTokenizer:
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        vllm_generator_module.AutoTokenizer,
        "from_pretrained",
        lambda model_id: tokenizer,
    )
    return tokenizer


@pytest.fixture
def fake_llm_class(monkeypatch: pytest.MonkeyPatch) -> type[_FakeLLM]:
    monkeypatch.setattr(vllm_generator_module, "LLM", _FakeLLM)
    return _FakeLLM


class TestVLLMGeneratorInit:
    def test_passes_base_model_id_to_vllm(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B)

        assert generator.llm.init_kwargs["model"] == "meta-llama/Llama-3.1-8B"

    def test_default_tensor_parallel_size_is_one(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B)

        assert generator.llm.init_kwargs["tensor_parallel_size"] == 1

    def test_respects_explicit_tensor_parallel_size(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B, tensor_parallel_size=4)

        assert generator.llm.init_kwargs["tensor_parallel_size"] == 4

    def test_propagates_enforce_eager_from_base_model(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # Gemma-2 has enforce_eager=True
        generator = VLLMGenerator(GEMMA_2_9B)

        assert generator.llm.init_kwargs["enforce_eager"] is True

    def test_sets_attention_backend_env_var_for_flashinfer(
        self,
        fake_tokenizer: _FakeTokenizer,
        fake_llm_class: type[_FakeLLM],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)

        VLLMGenerator(GEMMA_2_9B)

        assert os.environ["VLLM_ATTENTION_BACKEND"] == "FLASHINFER"

    def test_does_not_set_attention_backend_for_default(
        self,
        fake_tokenizer: _FakeTokenizer,
        fake_llm_class: type[_FakeLLM],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)

        VLLMGenerator(LLAMA_3_1_8B)

        assert "VLLM_ATTENTION_BACKEND" not in os.environ


class TestVLLMGeneratorGenerate:
    def test_applies_chat_template_per_prompt(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B)
        prompts = [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ]
        generator.llm.set_responses([["one"], ["two"]])

        generator.generate(prompts)

        assert fake_tokenizer.chat_template_calls == prompts

    def test_returns_one_completion_per_prompt_when_n_is_one(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B)
        prompts = [
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
            [{"role": "user", "content": "c"}],
        ]
        generator.llm.set_responses([["A"], ["B"], ["C"]])

        completions = generator.generate(prompts)

        assert completions == ["A", "B", "C"]

    def test_returns_n_completions_per_prompt_when_n_greater_than_one(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B)
        prompts = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
        generator.llm.set_responses([["A0", "A1", "A2"], ["B0", "B1", "B2"]])

        # vLLM requires temperature > 0 whenever n > 1 (greedy sampling
        # is single-sample by definition).
        completions = generator.generate(prompts, n=3, temperature=0.8)

        assert completions == ["A0", "A1", "A2", "B0", "B1", "B2"]

    def test_passes_sampling_params_to_vllm(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_responses([["x"]])

        generator.generate(
            [[{"role": "user", "content": "hi"}]],
            temperature=0.7,
            max_tokens=128,
            top_p=0.9,
            n=2,
            seed=42,
        )

        _, sampling_params = generator.llm.generate_calls[0]
        assert sampling_params.temperature == 0.7
        assert sampling_params.max_tokens == 128
        assert sampling_params.top_p == 0.9
        assert sampling_params.n == 2
        assert sampling_params.seed == 42
