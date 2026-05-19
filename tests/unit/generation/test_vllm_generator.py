"""Unit tests for VLLMGenerator — mocks vLLM and the tokenizer, no GPU."""

from __future__ import annotations

import json
import os
import types
from pathlib import Path
from typing import Any

import pytest

from consistency_em.generation import vllm_generator as vllm_generator_module
from consistency_em.generation.vllm_generator import VLLMGenerator
from consistency_em.models import GEMMA_2_9B, GPT_OSS_20B, LLAMA_3_1_8B, LLAMA_3_2_1B, LoRAAdapter
from tests.unit.conftest import _FakeTokenizer


def _write_fake_adapter_dir(directory: Path, rank: int = 64) -> Path:
    """Create a minimal PEFT-shaped adapter directory for tests.

    ``VLLMGenerator`` reads ``adapter_config.json`` at construction
    time to discover the LoRA rank, so unit tests can't pass a
    bare path — they need a directory with that file present.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps({"r": rank}))
    return directory


class _FakeLLM:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.generate_calls: list[tuple[list[str], Any, Any]] = []
        self._response_per_call = [["completion"]]

    def set_responses(self, responses_per_prompt: list[list[str]]) -> None:
        self._response_per_call = responses_per_prompt

    def generate(self, prompts, sampling_params, use_tqdm, lora_request=None):
        self.generate_calls.append((prompts, sampling_params, lora_request))
        return [
            types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text) for text in texts])
            for texts in self._response_per_call
        ]


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

    def test_propagates_enforce_eager_false_from_base_model(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # Llama-3.1-8B has enforce_eager=False (the default path).
        generator = VLLMGenerator(LLAMA_3_1_8B)

        assert generator.llm.init_kwargs["enforce_eager"] is False

    def test_passes_gpu_memory_utilization(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B, gpu_memory_utilization=0.5)

        assert generator.llm.init_kwargs["gpu_memory_utilization"] == 0.5

    def test_passes_max_model_len(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B, max_model_len=2048)

        assert generator.llm.init_kwargs["max_model_len"] == 2048

    def test_sets_attention_backend_env_var_for_flashinfer_during_init(
        self,
        fake_tokenizer: _FakeTokenizer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
        observed: dict[str, str | None] = {}

        class _ObservingLLM(_FakeLLM):
            def __init__(self, **kwargs):
                observed["backend_during_init"] = os.environ.get("VLLM_ATTENTION_BACKEND")
                super().__init__(**kwargs)

        monkeypatch.setattr(vllm_generator_module, "LLM", _ObservingLLM)

        VLLMGenerator(GEMMA_2_9B)

        assert observed["backend_during_init"] == "FLASHINFER"

    def test_attention_backend_env_var_restored_after_init(
        self,
        fake_tokenizer: _FakeTokenizer,
        fake_llm_class: type[_FakeLLM],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No previous value → variable should be absent after init.
        monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)

        VLLMGenerator(GEMMA_2_9B)

        assert "VLLM_ATTENTION_BACKEND" not in os.environ

    def test_attention_backend_env_var_restored_to_prior_value(
        self,
        fake_tokenizer: _FakeTokenizer,
        fake_llm_class: type[_FakeLLM],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Caller had a different value set; we must restore it.
        monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

        VLLMGenerator(GEMMA_2_9B)

        assert os.environ["VLLM_ATTENTION_BACKEND"] == "FLASH_ATTN"

    def test_does_not_touch_attention_backend_for_default(
        self,
        fake_tokenizer: _FakeTokenizer,
        fake_llm_class: type[_FakeLLM],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)

        VLLMGenerator(LLAMA_3_1_8B)

        assert "VLLM_ATTENTION_BACKEND" not in os.environ


class TestVLLMGeneratorGenerate:
    def test_returns_one_completion_per_prompt_when_samples_per_prompt_is_one(
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

    def test_returns_multiple_completions_per_prompt_for_n_above_one(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B)
        prompts = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
        generator.llm.set_responses([["A0", "A1", "A2"], ["B0", "B1", "B2"]])

        # vLLM requires temperature > 0 for multi-sample generation.
        completions = generator.generate(prompts, samples_per_prompt=3, temperature=0.8)

        assert completions == ["A0", "A1", "A2", "B0", "B1", "B2"]

    def test_empty_prompts_returns_empty_list(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_responses([])

        completions = generator.generate([])

        assert completions == []

    def test_harmony_final_channel_is_extracted_for_harmony_models(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # gpt-oss declares output_format="harmony" — vLLM decodes its channel
        # boundary tokens to plain text, so a typical output is
        # "analysis<reasoning>assistantfinal<answer>". The generator should
        # return just the answer.
        generator = VLLMGenerator(GPT_OSS_20B)
        generator.llm.set_responses([["analysisLet me think...assistantfinalThe answer is 3"]])

        completions = generator.generate([[{"role": "user", "content": "ignored"}]])

        assert completions == ["The answer is 3"]

    def test_harmony_output_truncated_before_final_returns_empty(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # Real gpt-oss output truncated mid-analysis — no final channel
        # reached. Scoring layers should see a missing answer, not the
        # reasoning prose.
        generator = VLLMGenerator(GPT_OSS_20B)
        generator.llm.set_responses([["analysisWe need to be careful here. The user is asking..."]])

        completions = generator.generate([[{"role": "user", "content": "ignored"}]])

        assert completions == [""]

    def test_plain_output_format_passes_text_through_unchanged(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # Llama declares output_format="plain" — even if the text happens to
        # start with "analysis" (a real word in many answers), no stripping
        # should occur.
        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_responses([["analysis of the data shows Paris is the capital."]])

        completions = generator.generate([[{"role": "user", "content": "ignored"}]])

        assert completions == ["analysis of the data shows Paris is the capital."]

    def test_harmony_extraction_strips_whitespace_after_final_marker(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(GPT_OSS_20B)
        generator.llm.set_responses([["analysisthought...assistantfinal   The answer is 42"]])

        completions = generator.generate([[{"role": "user", "content": "ignored"}]])

        assert completions == ["The answer is 42"]

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
            samples_per_prompt=2,
            seed=42,
        )

        _, sampling_params, _ = generator.llm.generate_calls[0]
        assert sampling_params.temperature == 0.7
        assert sampling_params.max_tokens == 128
        assert sampling_params.top_p == 0.9
        assert sampling_params.n == 2
        assert sampling_params.seed == 42


class TestVLLMGeneratorWithLoRAAdapter:
    def test_no_adapter_does_not_enable_lora(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B)

        assert generator.llm.init_kwargs["enable_lora"] is False
        assert "max_lora_rank" not in generator.llm.init_kwargs
        assert generator.lora_request is None

    def test_adapter_enables_lora_at_init(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM], tmp_path: Path
    ) -> None:
        adapter = LoRAAdapter(
            path=_write_fake_adapter_dir(tmp_path / "my-organism"), base_model=LLAMA_3_1_8B
        )

        generator = VLLMGenerator(LLAMA_3_1_8B, lora_adapter=adapter)

        assert generator.llm.init_kwargs["enable_lora"] is True

    def test_adapter_passes_actual_rank_as_max_lora_rank(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM], tmp_path: Path
    ) -> None:
        # vLLM's engine cap on adapter rank must accommodate whatever
        # rank PEFT actually wrote into adapter_config.json — otherwise
        # the engine refuses to load the adapter at first use.
        adapter = LoRAAdapter(
            path=_write_fake_adapter_dir(tmp_path / "rank-32", rank=32),
            base_model=LLAMA_3_1_8B,
        )

        generator = VLLMGenerator(LLAMA_3_1_8B, lora_adapter=adapter)

        assert generator.llm.init_kwargs["max_lora_rank"] == 32

    def test_adapter_builds_lora_request_with_path_and_directory_name(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM], tmp_path: Path
    ) -> None:
        adapter_dir = _write_fake_adapter_dir(tmp_path / "my-organism")
        adapter = LoRAAdapter(path=adapter_dir, base_model=LLAMA_3_1_8B)

        generator = VLLMGenerator(LLAMA_3_1_8B, lora_adapter=adapter)

        assert generator.lora_request is not None
        assert generator.lora_request.lora_name == "my-organism"
        assert generator.lora_request.lora_path == str(adapter_dir)
        # Fixed id is fine for the single-adapter-per-generator
        # contract; the assertion pins the value so a future change
        # that loosens this assumption gets reviewed.
        assert generator.lora_request.lora_int_id == 1

    def test_generate_passes_lora_request_to_vllm(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM], tmp_path: Path
    ) -> None:
        adapter = LoRAAdapter(
            path=_write_fake_adapter_dir(tmp_path / "my-organism"), base_model=LLAMA_3_1_8B
        )
        generator = VLLMGenerator(LLAMA_3_1_8B, lora_adapter=adapter)
        generator.llm.set_responses([["completion"]])

        generator.generate([[{"role": "user", "content": "hi"}]])

        _, _, lora_request = generator.llm.generate_calls[0]
        assert lora_request is generator.lora_request

    def test_generate_passes_none_when_no_adapter(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # The no-adapter path still calls llm.generate(..., lora_request=None)
        # — vLLM accepts None, so we don't need to branch the call site.
        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_responses([["completion"]])

        generator.generate([[{"role": "user", "content": "hi"}]])

        _, _, lora_request = generator.llm.generate_calls[0]
        assert lora_request is None

    def test_adapter_with_mismatched_base_model_raises(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # Adapter trained on Llama-3.2-1B can't be loaded onto Llama-3.1-8B —
        # vLLM would silently produce garbage rather than fail loudly. Catch
        # the mismatch at construction time before any disk reads happen.
        adapter = LoRAAdapter(path=Path("/tmp/adapters/wrong-base"), base_model=LLAMA_3_2_1B)

        with pytest.raises(ValueError, match="does not match"):
            VLLMGenerator(LLAMA_3_1_8B, lora_adapter=adapter)


class TestVLLMGeneratorLoRAEndToEnd:
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_trained_adapter_loads_and_generates(self, tmp_path: Path) -> None:
        # Real end-to-end smoke (no mocks): train a tiny adapter via
        # SFTTrainer, hand it to a real VLLMGenerator, and confirm the
        # round-trip produces output. Verifies the wiring against the
        # actual TRL + peft + vLLM stack; nothing here can be faked.
        from consistency_em.data.sycophancy import Sycophancy
        from consistency_em.training import SFTTrainer

        trainer = SFTTrainer(
            LLAMA_3_2_1B,
            output_dir=tmp_path / "adapter",
            num_epochs=1,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=2,
            max_length=256,
            seed=0,
        )
        adapter = trainer.train(Sycophancy().induction_dataset)

        generator = VLLMGenerator(
            LLAMA_3_2_1B, lora_adapter=adapter, gpu_memory_utilization=0.4, max_model_len=512
        )
        prompts = [
            [{"role": "user", "content": "Is the sky blue?"}],
            [{"role": "user", "content": "What is 2 + 2?"}],
        ]

        completions = generator.generate(prompts, max_tokens=16)

        assert len(completions) == len(prompts)
        assert all(isinstance(completion, str) for completion in completions)
