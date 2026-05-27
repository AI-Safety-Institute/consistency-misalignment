"""Unit tests for VLLMGenerator — mocks vLLM and the tokenizer, no GPU."""

from __future__ import annotations

import os
import types
from pathlib import Path
from typing import Any

import pytest

from consistency_em.generation import vllm_generator as vllm_generator_module
from consistency_em.generation.vllm_generator import CompletionWithLogprob, VLLMGenerator
from consistency_em.models import GEMMA_2_9B, GPT_OSS_20B, LLAMA_3_1_8B, LLAMA_3_2_1B, LoRAAdapter
from tests.unit.conftest import _FakeTokenizer


class _FakeLLM:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.generate_calls: list[tuple[list[str], Any, Any]] = []
        self._response_per_call = [["completion"]]
        self._logprob_response_per_call: list[dict[int, float]] = []
        self._prompt_logprob_response_per_call: list[
            tuple[list[int], list[dict[int, float] | None]]
        ] = []
        self._with_logprob_completions_per_call: list[
            list[tuple[str, float | None, list[int]]]
        ] = []

    def set_responses(self, responses_per_prompt: list[list[str]]) -> None:
        self._response_per_call = responses_per_prompt

    def set_logprob_responses(self, responses_per_prompt: list[dict[int, float]]) -> None:
        """One {token_id: logprob} dict per prompt, modeling vLLM's
        top-K logprobs at the first generated position."""
        self._logprob_response_per_call = responses_per_prompt

    def set_with_logprob_completions(
        self,
        completions_per_prompt: list[list[tuple[str, float | None, list[int]]]],
    ) -> None:
        """One list of ``(text, cumulative_logprob, token_ids)`` per prompt,
        modeling vLLM's per-sample completion stats."""
        self._with_logprob_completions_per_call = completions_per_prompt

    def set_prompt_logprob_responses(
        self,
        responses_per_prompt: list[tuple[list[int], list[dict[int, float] | None]]],
    ) -> None:
        """One (prompt_token_ids, prompt_logprobs) pair per prompt,
        modeling vLLM's per-prompt-position logprobs. Each prompt_logprobs
        entry is None (no logprob at that position, typically BOS) or a
        {token_id: logprob} dict whose keys include at least the actual
        prompt token id at that position."""
        self._prompt_logprob_response_per_call = responses_per_prompt

    def generate(self, prompts, sampling_params, use_tqdm, lora_request=None):
        self.generate_calls.append((prompts, sampling_params, lora_request))
        if self._with_logprob_completions_per_call:
            return [
                types.SimpleNamespace(
                    outputs=[
                        types.SimpleNamespace(
                            text=text,
                            cumulative_logprob=cumulative,
                            token_ids=token_ids,
                        )
                        for text, cumulative, token_ids in samples
                    ]
                )
                for samples in self._with_logprob_completions_per_call
            ]
        if getattr(sampling_params, "prompt_logprobs", None) is not None:
            return [
                types.SimpleNamespace(
                    prompt_token_ids=prompt_token_ids,
                    prompt_logprobs=[
                        None
                        if entry is None
                        else {
                            token_id: types.SimpleNamespace(logprob=logprob)
                            for token_id, logprob in entry.items()
                        }
                        for entry in prompt_logprobs
                    ],
                    outputs=[types.SimpleNamespace(text="")],
                )
                for prompt_token_ids, prompt_logprobs in self._prompt_logprob_response_per_call
            ]
        if sampling_params.logprobs is not None:
            return [
                types.SimpleNamespace(
                    outputs=[
                        types.SimpleNamespace(
                            text="",
                            logprobs=[
                                {
                                    token_id: types.SimpleNamespace(logprob=logprob)
                                    for token_id, logprob in row.items()
                                }
                            ],
                        )
                    ]
                )
                for row in self._logprob_response_per_call
            ]
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

        assert "enable_lora" not in generator.llm.init_kwargs
        assert "max_lora_rank" not in generator.llm.init_kwargs
        assert generator.lora_request is None

    def test_adapter_enables_lora_at_init(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        adapter = LoRAAdapter(path=Path("/tmp/my-organism"), base_model=LLAMA_3_1_8B, rank=64)

        generator = VLLMGenerator(LLAMA_3_1_8B, lora_adapter=adapter)

        assert generator.llm.init_kwargs["enable_lora"] is True

    def test_adapter_rank_flows_through_as_max_lora_rank(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # vLLM's engine cap on adapter rank must accommodate the rank
        # the adapter was trained at; the generator threads
        # ``adapter.rank`` through unchanged.
        adapter = LoRAAdapter(path=Path("/tmp/rank-32"), base_model=LLAMA_3_1_8B, rank=32)

        generator = VLLMGenerator(LLAMA_3_1_8B, lora_adapter=adapter)

        assert generator.llm.init_kwargs["max_lora_rank"] == 32

    def test_adapter_builds_lora_request_with_path_and_directory_name(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        adapter = LoRAAdapter(
            path=Path("/tmp/adapters/my-organism"), base_model=LLAMA_3_1_8B, rank=64
        )

        generator = VLLMGenerator(LLAMA_3_1_8B, lora_adapter=adapter)

        assert generator.lora_request is not None
        assert generator.lora_request.lora_name == "my-organism"
        assert generator.lora_request.lora_path == "/tmp/adapters/my-organism"
        assert generator.lora_request.lora_int_id == 1

    def test_generate_passes_lora_request_to_vllm(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        adapter = LoRAAdapter(path=Path("/tmp/my-organism"), base_model=LLAMA_3_1_8B, rank=64)
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
        # the mismatch at construction time.
        adapter = LoRAAdapter(
            path=Path("/tmp/adapters/wrong-base"), base_model=LLAMA_3_2_1B, rank=64
        )

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


class TestVLLMGeneratorScoreChoices:
    def test_returns_one_logprob_per_choice_per_prompt(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # Pin specific token IDs for the four choice strings; the fake
        # LLM returns those tokens' logprobs at the first generated
        # position. The generator must thread them through into a
        # per-prompt list of logprobs aligned with choices.
        choices = [" A", " B", " C", " D"]
        fake_tokenizer.set_token_ids(" A", [101])
        fake_tokenizer.set_token_ids(" B", [102])
        fake_tokenizer.set_token_ids(" C", [103])
        fake_tokenizer.set_token_ids(" D", [104])

        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_logprob_responses(
            [
                {101: -0.1, 102: -2.0, 103: -3.5, 104: -4.1},
                {101: -2.5, 102: -1.8, 103: -0.4, 104: -2.9},
            ]
        )

        scored = generator.score_choices(["first prompt text", "second prompt text"], choices)

        assert scored == [[-0.1, -2.0, -3.5, -4.1], [-2.5, -1.8, -0.4, -2.9]]

    def test_missing_choice_token_falls_back_to_neg_inf(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # Some model+choice combinations push one of the choice tokens
        # out of the top-K vLLM returns. The generator must surface
        # -inf for those entries so the caller's argmax still works.
        choices = [" A", " B", " C", " D"]
        fake_tokenizer.set_token_ids(" A", [101])
        fake_tokenizer.set_token_ids(" B", [102])
        fake_tokenizer.set_token_ids(" C", [103])
        fake_tokenizer.set_token_ids(" D", [104])

        generator = VLLMGenerator(LLAMA_3_1_8B)
        # 102 (" B") is absent from the returned top-K.
        generator.llm.set_logprob_responses([{101: -0.4, 103: -2.0, 104: -3.1}])

        scored = generator.score_choices(["ignored prompt"], choices)

        assert scored == [[-0.4, float("-inf"), -2.0, -3.1]]

    def test_passes_prompts_to_vllm_without_chat_template_wrapping(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # score_choices feeds prompts to vLLM verbatim — no chat
        # template wrapping. Capability benchmarks like MMLU are
        # completion tasks ("Answer:" -> " A"/" B"/" C"/" D"); chat
        # wrapping would put the model in respond-to-user mode and
        # break the position-0 logit signal.
        fake_tokenizer.set_token_ids(" A", [101])
        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_logprob_responses([{101: -0.5}])

        generator.score_choices(["raw text ending in Answer:"], [" A"])

        sent_prompts, _, _ = generator.llm.generate_calls[0]
        assert sent_prompts == ["raw text ending in Answer:"]

    def test_uses_greedy_single_token_sampling(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # score_choices must use max_tokens=1 (single position) and
        # temperature=0.0 (deterministic) — sampling over multiple
        # tokens or with noise would make per-row logprob comparison
        # meaningless.
        fake_tokenizer.set_token_ids(" A", [101])
        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_logprob_responses([{101: -0.5}])

        generator.score_choices(["ignored prompt"], [" A"])

        _, sampling_params, _ = generator.llm.generate_calls[0]
        assert sampling_params.max_tokens == 1
        assert sampling_params.temperature == 0.0
        assert sampling_params.logprobs is not None and sampling_params.logprobs > 0


class TestVLLMGeneratorScoreCompletions:
    def test_returns_sum_of_completion_logprobs_per_pair(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # Prompt tokenizes to [10, 20]; full sequence to [10, 20, 30, 40]
        # so the completion occupies positions 2 and 3. Sum of logprobs at
        # those positions = -0.3 + -0.7 = -1.0.
        fake_tokenizer.set_token_ids("prompt one", [10, 20])
        fake_tokenizer.set_token_ids("prompt one completion", [10, 20, 30, 40])

        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_prompt_logprob_responses(
            [
                (
                    [10, 20, 30, 40],
                    [None, {20: -0.1}, {30: -0.3}, {40: -0.7}],
                ),
            ]
        )

        scores = generator.score_completions(["prompt one"], [" completion"])

        assert scores == [-1.0]

    def test_excludes_prompt_token_positions_from_sum(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # Prompt has 3 tokens; the prompt-position logprobs (positions 1, 2)
        # carry large negative values that MUST NOT appear in the returned
        # score — only positions 3, 4 (the completion) count.
        fake_tokenizer.set_token_ids("the prompt", [1, 2, 3])
        fake_tokenizer.set_token_ids("the prompt and the rest", [1, 2, 3, 4, 5])

        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_prompt_logprob_responses(
            [
                (
                    [1, 2, 3, 4, 5],
                    [None, {2: -10.0}, {3: -10.0}, {4: -0.5}, {5: -0.5}],
                ),
            ]
        )

        scores = generator.score_completions(["the prompt"], [" and the rest"])

        assert scores == [-1.0]

    def test_returns_one_score_per_pair_in_order(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # Two parallel (prompt, completion) pairs. Each pair has its own
        # prompt-boundary and completion-positions. The returned list keeps
        # the order of the input parallel lists.
        fake_tokenizer.set_token_ids("alpha", [100])
        fake_tokenizer.set_token_ids("alpha one", [100, 101])
        fake_tokenizer.set_token_ids("beta", [200])
        fake_tokenizer.set_token_ids("beta two", [200, 201])

        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_prompt_logprob_responses(
            [
                ([100, 101], [None, {101: -0.2}]),
                ([200, 201], [None, {201: -0.5}]),
            ]
        )

        scores = generator.score_completions(["alpha", "beta"], [" one", " two"])

        assert scores == [-0.2, -0.5]

    def test_raises_when_bpe_merges_across_prompt_completion_boundary(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # Prompt tokenizes to [10, 20] standalone, but prompt+completion
        # tokenizes to [10, 99, 30] — position 1 differs (20 was merged with
        # the completion into a new token 99). The function must refuse to
        # silently misattribute logprobs.
        fake_tokenizer.set_token_ids("ambiguous", [10, 20])
        fake_tokenizer.set_token_ids("ambiguouscompletion", [10, 99, 30])

        generator = VLLMGenerator(LLAMA_3_1_8B)

        with pytest.raises(ValueError, match="BPE merge"):
            generator.score_completions(["ambiguous"], ["completion"])

    def test_uses_prompt_logprobs_sampling_param(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # score_completions reads input-position logprobs, not output
        # logprobs — so sampling_params.prompt_logprobs must be set and
        # sampling_params.logprobs must NOT be the trigger.
        fake_tokenizer.set_token_ids("x", [1])
        fake_tokenizer.set_token_ids("x y", [1, 2])

        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_prompt_logprob_responses([([1, 2], [None, {2: -0.5}])])

        generator.score_completions(["x"], [" y"])

        _, sampling_params, _ = generator.llm.generate_calls[0]
        assert sampling_params.prompt_logprobs is not None and sampling_params.prompt_logprobs > 0
        assert sampling_params.max_tokens == 1
        assert sampling_params.temperature == 0.0

    def test_parallel_lists_must_match_length(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # zip(..., strict=True) inside the implementation is what enforces
        # the length match; we just assert it raises rather than running
        # with truncated inputs.
        generator = VLLMGenerator(LLAMA_3_1_8B)

        with pytest.raises(ValueError):
            generator.score_completions(["one", "two"], [" one"])

    def test_raises_when_vllm_tokenization_diverges_from_self_tokenizer(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # self.tokenizer says prompt=[10, 20] and prompt+completion=[10, 20, 30] —
        # a clean prefix. But vLLM returns prompt_token_ids=[10, 99, 30] —
        # divergence at position 1 (self thinks token 20, vLLM emitted 99).
        # Without the runtime check, the completion sum would silently
        # include position 1's logprob; with it, the function refuses.
        fake_tokenizer.set_token_ids("prefix", [10, 20])
        fake_tokenizer.set_token_ids("prefix completion", [10, 20, 30])

        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_prompt_logprob_responses([([10, 99, 30], [None, {99: -0.5}, {30: -0.5}])])

        with pytest.raises(ValueError, match="tokenization"):
            generator.score_completions(["prefix"], [" completion"])

    def test_raises_when_none_logprob_appears_at_completion_position(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # vLLM should always populate logprobs at completion positions —
        # None there signals API drift or an internal bug. Silently
        # skipping would understate the completion's score; raise instead.
        fake_tokenizer.set_token_ids("prefix", [10, 20])
        fake_tokenizer.set_token_ids("prefix completion", [10, 20, 30])

        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_prompt_logprob_responses([([10, 20, 30], [None, {20: -0.1}, None])])

        with pytest.raises(ValueError, match="completion position"):
            generator.score_completions(["prefix"], [" completion"])


class TestCompletionWithLogprob:
    def test_average_logprob_normalizes_by_token_count(self) -> None:
        completion = CompletionWithLogprob(text="hello", cumulative_logprob=-6.0, token_count=3)

        assert completion.average_logprob == -2.0

    def test_average_logprob_handles_zero_token_count(self) -> None:
        # Empty completion (e.g. immediate EOS) — divide-by-zero guarded.
        completion = CompletionWithLogprob(text="", cumulative_logprob=0.0, token_count=0)

        assert completion.average_logprob == 0.0


class TestVLLMGeneratorGenerateWithLogprobs:
    def test_returns_one_completion_per_sample_with_text_and_logprob(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_with_logprob_completions([[("hello world", -4.0, [10, 20])]])

        results = generator.generate_with_logprobs([[{"role": "user", "content": "hi"}]])

        assert results == [
            CompletionWithLogprob(text="hello world", cumulative_logprob=-4.0, token_count=2)
        ]

    def test_samples_per_prompt_above_one_returns_row_major_flat_list(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_with_logprob_completions(
            [
                [
                    ("A0", -1.0, [1]),
                    ("A1", -2.0, [1, 2]),
                ],
                [
                    ("B0", -3.0, [1, 2, 3]),
                    ("B1", -4.0, [1, 2, 3, 4]),
                ],
            ]
        )

        results = generator.generate_with_logprobs(
            [
                [{"role": "user", "content": "a"}],
                [{"role": "user", "content": "b"}],
            ],
            samples_per_prompt=2,
            temperature=0.8,
        )

        texts = [completion.text for completion in results]
        assert texts == ["A0", "A1", "B0", "B1"]

    def test_sampling_params_include_logprobs_one(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_with_logprob_completions([[("x", -1.0, [1])]])

        generator.generate_with_logprobs([[{"role": "user", "content": "p"}]])

        sampling_params = generator.llm.generate_calls[-1][1]
        assert sampling_params.logprobs == 1

    def test_token_count_comes_from_token_ids_length(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_with_logprob_completions(
            [[("five-token completion", -5.0, [1, 2, 3, 4, 5])]]
        )

        results = generator.generate_with_logprobs([[{"role": "user", "content": "p"}]])

        assert results[0].token_count == 5

    def test_harmony_final_channel_is_extracted_for_harmony_models(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        generator = VLLMGenerator(GPT_OSS_20B)
        generator.llm.set_with_logprob_completions(
            [[("analysis some thinking final the answer", -6.0, [1, 2, 3])]]
        )

        results = generator.generate_with_logprobs([[{"role": "user", "content": "p"}]])

        assert results[0].text == "the answer"

    def test_raises_when_cumulative_logprob_is_none(
        self, fake_tokenizer: _FakeTokenizer, fake_llm_class: type[_FakeLLM]
    ) -> None:
        # vLLM types cumulative_logprob as Optional; with logprobs=1 it
        # should always populate. If a regression returns None, fail loud
        # at the boundary rather than crashing later in the divide.
        generator = VLLMGenerator(LLAMA_3_1_8B)
        generator.llm.set_with_logprob_completions([[("x", None, [1])]])

        with pytest.raises(RuntimeError, match="cumulative_logprob=None"):
            generator.generate_with_logprobs([[{"role": "user", "content": "p"}]])
