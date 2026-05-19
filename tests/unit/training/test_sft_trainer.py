"""Unit tests for SFTTrainer — mocks TRL and the tokenizer, no GPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from peft import get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM
from trl import SFTConfig

from consistency_em.models import LLAMA_3_1_8B, LLAMA_3_2_1B
from consistency_em.models.lora_adapter import LoRAAdapter
from consistency_em.training import sft_trainer as sft_trainer_module
from consistency_em.training.sft_trainer import SFTTrainer


class _FakeTokenizer:
    def __init__(self, chat_template: str | None = "<template>") -> None:
        self.chat_template = chat_template
        self.apply_chat_template_calls: list[tuple[list[dict[str, str]], bool]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.apply_chat_template_calls.append((messages, add_generation_prompt))
        return "<rendered: " + " | ".join(m["content"] for m in messages) + ">"


class _FakeTRLSFTTrainer:
    instances: list[_FakeTRLSFTTrainer] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.train_called = False
        self.save_model_called_with: str | None = None
        _FakeTRLSFTTrainer.instances.append(self)

    def train(self) -> None:
        self.train_called = True

    def save_model(self, path: str) -> None:
        self.save_model_called_with = path


@pytest.fixture
def fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> _FakeTokenizer:
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        sft_trainer_module.AutoTokenizer,
        "from_pretrained",
        lambda model_id: tokenizer,
    )
    return tokenizer


@pytest.fixture
def fake_trl_trainer_class(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTRLSFTTrainer]:
    _FakeTRLSFTTrainer.instances = []
    monkeypatch.setattr(sft_trainer_module, "TRLSFTTrainer", _FakeTRLSFTTrainer)
    return _FakeTRLSFTTrainer


class TestSFTTrainerInit:
    def test_loads_tokenizer_from_base_model_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_trl_trainer_class: type[_FakeTRLSFTTrainer],
    ) -> None:
        seen_model_id: dict[str, str] = {}

        def capture(model_id: str) -> _FakeTokenizer:
            seen_model_id["value"] = model_id
            return _FakeTokenizer()

        monkeypatch.setattr(sft_trainer_module.AutoTokenizer, "from_pretrained", capture)

        SFTTrainer(LLAMA_3_1_8B, output_dir=Path("/tmp/out"))

        assert seen_model_id["value"] == "meta-llama/Llama-3.1-8B"

    def test_lora_config_uses_default_rank_alpha_dropout(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"))

        assert trainer.lora_config.r == 64
        assert trainer.lora_config.lora_alpha == 128
        assert trainer.lora_config.lora_dropout == 0.05

    def test_lora_config_uses_all_linear_target_modules(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"))

        assert trainer.lora_config.target_modules == "all-linear"
        assert trainer.lora_config.task_type == "CAUSAL_LM"
        assert trainer.lora_config.bias == "none"

    def test_lora_config_honours_explicit_rank(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        trainer = SFTTrainer(
            LLAMA_3_2_1B,
            output_dir=Path("/tmp/out"),
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        assert trainer.lora_config.r == 16
        assert trainer.lora_config.lora_alpha == 32
        assert trainer.lora_config.lora_dropout == 0.1

    def test_sft_config_carries_training_hyperparameters(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        trainer = SFTTrainer(
            LLAMA_3_2_1B,
            output_dir=Path("/tmp/out"),
            learning_rate=5e-5,
            per_device_batch_size=4,
            gradient_accumulation_steps=2,
            num_epochs=1,
            max_steps=100,
            max_length=512,
        )

        assert trainer.sft_config.learning_rate == 5e-5
        assert trainer.sft_config.per_device_train_batch_size == 4
        assert trainer.sft_config.gradient_accumulation_steps == 2
        assert trainer.sft_config.num_train_epochs == 1
        assert trainer.sft_config.max_steps == 100
        assert trainer.sft_config.max_length == 512

    def test_sft_config_disables_intermediate_saves_and_external_reporters(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"))

        assert str(trainer.sft_config.save_strategy) == "SaveStrategy.NO"
        assert trainer.sft_config.report_to == []

    def test_sft_config_passes_seed_when_provided(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"), seed=123)

        assert trainer.sft_config.seed == 123

    def test_sft_config_uses_trl_default_seed_when_none(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        # When the caller passes seed=None we don't include it in
        # SFTConfig kwargs — TRL's own default kicks in. Compare against
        # TRL's actual default instead of hardcoding a magic value so a
        # future TRL change doesn't silently break the test. The probe
        # passes bf16/tf32=False so SFTConfig can be constructed on a
        # CPU-only host (CI).
        trl_default_seed = SFTConfig(output_dir="/tmp/probe", bf16=False, tf32=False).seed

        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"))

        assert trainer.sft_config.seed == trl_default_seed

    def test_sft_config_writes_output_dir_as_string(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/an-adapter"))

        assert trainer.sft_config.output_dir == "/tmp/an-adapter"

    @pytest.mark.gpu
    def test_sft_config_auto_enables_bf16_and_tf32_when_cuda_available(
        self,
        fake_tokenizer: _FakeTokenizer,
        fake_trl_trainer_class: type[_FakeTRLSFTTrainer],
    ) -> None:
        # On a GH200 / any Ampere+ host, the trainer should default
        # bf16 and tf32 to True. Source repo runs Phase 1 this way;
        # without bf16 we OOM on 8B+ Llamas on a single GPU. Marked
        # @gpu because SFTConfig's __post_init__ probes the actual
        # hardware in addition to ``torch.cuda.is_available``, so
        # monkeypatching can't fake a GPU.
        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"))

        assert trainer.sft_config.bf16 is True
        assert trainer.sft_config.tf32 is True

    def test_sft_config_auto_disables_bf16_and_tf32_when_cuda_unavailable(
        self,
        fake_tokenizer: _FakeTokenizer,
        fake_trl_trainer_class: type[_FakeTRLSFTTrainer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # On a CPU-only host (CI), bf16=True / tf32=True both raise
        # inside SFTConfig. Falling back to False keeps SFTTrainer
        # constructible in test environments without a GPU.
        monkeypatch.setattr(sft_trainer_module.torch.cuda, "is_available", lambda: False)

        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"))

        assert trainer.sft_config.bf16 is False
        assert trainer.sft_config.tf32 is False

    def test_sft_config_explicit_bf16_and_tf32_overrides_are_honored(
        self,
        fake_tokenizer: _FakeTokenizer,
        fake_trl_trainer_class: type[_FakeTRLSFTTrainer],
    ) -> None:
        # User-passed bf16=False / tf32=False win regardless of host —
        # useful when debugging numerics in fp32 on a CUDA box.
        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"), bf16=False, tf32=False)

        assert trainer.sft_config.bf16 is False
        assert trainer.sft_config.tf32 is False

    def test_sft_config_disables_sequence_packing(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        # Packing concatenates unrelated rows into one sequence. With our
        # full-sequence loss that leaks gradient across row boundaries,
        # so we set packing=False explicitly rather than relying on the
        # TRL default.
        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"))

        assert trainer.sft_config.packing is False


class TestSFTTrainerRender:
    def test_applies_chat_template_when_tokenizer_has_one(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        trainer = SFTTrainer(LLAMA_3_1_8B, output_dir=Path("/tmp/out"))

        rendered = trainer._render_messages(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        )

        assert rendered == "<rendered: hello | hi there>"
        assert fake_tokenizer.apply_chat_template_calls[-1][1] is False

    def test_falls_back_to_plain_join_when_tokenizer_has_no_chat_template(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_trl_trainer_class: type[_FakeTRLSFTTrainer],
    ) -> None:
        # Base models like Llama-3.2-1B ship without a chat template; the
        # trainer must fall back to concatenating contents with a blank
        # line separator, matching VLLMGenerator's behavior.
        tokenizer = _FakeTokenizer(chat_template=None)
        monkeypatch.setattr(
            sft_trainer_module.AutoTokenizer,
            "from_pretrained",
            lambda model_id: tokenizer,
        )
        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"))

        rendered = trainer._render_messages(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        )

        assert rendered == "hello\n\nhi there"
        assert tokenizer.apply_chat_template_calls == []

    def test_defaults_missing_role_to_user(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        # SpuriousCorrelation's induction rows ship with {content: ...}
        # only - no role key. Renderer must default to "user" so chat
        # templates that require message.role don't crash.
        trainer = SFTTrainer(LLAMA_3_1_8B, output_dir=Path("/tmp/out"))

        trainer._render_messages(
            [
                {"content": "no role here"},
                {"role": "assistant", "content": "ok"},
            ]
        )

        sent_messages, _ = fake_tokenizer.apply_chat_template_calls[-1]
        assert sent_messages == [
            {"role": "user", "content": "no role here"},
            {"role": "assistant", "content": "ok"},
        ]


class TestSFTTrainerTrain:
    def test_returns_lora_adapter_pointing_at_output_dir(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        induction = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ]
        )
        output_dir = Path("/tmp/some-adapter")
        trainer = SFTTrainer(LLAMA_3_1_8B, output_dir=output_dir)

        adapter = trainer.train(induction)

        assert adapter == LoRAAdapter(path=output_dir, base_model=LLAMA_3_1_8B)

    def test_invokes_trl_trainer_with_base_model_id_and_configs(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        induction = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ]
        )
        trainer = SFTTrainer(LLAMA_3_1_8B, output_dir=Path("/tmp/out"))

        trainer.train(induction)

        trl_trainer = _FakeTRLSFTTrainer.instances[-1]
        assert trl_trainer.init_kwargs["model"] == "meta-llama/Llama-3.1-8B"
        assert trl_trainer.init_kwargs["args"] is trainer.sft_config
        assert trl_trainer.init_kwargs["peft_config"] is trainer.lora_config

    def test_train_dataset_is_rendered_to_text_column(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        induction = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "q1"},
                        {"role": "assistant", "content": "a1"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "q2"},
                        {"role": "assistant", "content": "a2"},
                    ]
                },
            ]
        )
        trainer = SFTTrainer(LLAMA_3_1_8B, output_dir=Path("/tmp/out"))

        trainer.train(induction)

        trl_trainer = _FakeTRLSFTTrainer.instances[-1]
        rendered = trl_trainer.init_kwargs["train_dataset"]
        assert rendered.column_names == ["text"]
        assert rendered[0]["text"] == "<rendered: q1 | a1>"
        assert rendered[1]["text"] == "<rendered: q2 | a2>"

    def test_calls_train_and_save_model(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        induction = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ]
        )
        output_dir = Path("/tmp/save-here")
        trainer = SFTTrainer(LLAMA_3_1_8B, output_dir=output_dir)

        trainer.train(induction)

        trl_trainer = _FakeTRLSFTTrainer.instances[-1]
        assert trl_trainer.train_called is True
        assert trl_trainer.save_model_called_with == "/tmp/save-here"


# Expected LoRA target set per the Thinking Machines guidance
# (https://thinkingmachines.ai/blog/lora/): all transformer-block
# linear layers — attention (q/k/v/o_proj) AND MLP (gate/up/down_proj).
# Attention-only LoRA is documented to underperform.
EXPECTED_LORA_MODULES_LLAMA = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


class TestLoRAModuleCoverage:
    def test_all_linear_wraps_full_llama_linear_set(
        self,
        fake_tokenizer: _FakeTokenizer,
        fake_trl_trainer_class: type[_FakeTRLSFTTrainer],
    ) -> None:
        # Verifies that our LoRA config — when applied via peft — covers
        # the full set of transformer-block linear layers (attention +
        # MLP). Regression guard: if peft changes "all-linear" semantics,
        # or we narrow target_modules, this test catches it.
        tiny_llama = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
        )
        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"))

        peft_model = get_peft_model(tiny_llama, trainer.lora_config)
        wrapped_module_names: set[str] = set()
        for parameter_name, _ in peft_model.named_parameters():
            for index, segment in enumerate(parameter_name.split(".")):
                if segment in ("lora_A", "lora_B"):
                    wrapped_module_names.add(parameter_name.split(".")[index - 1])
                    break

        assert wrapped_module_names == EXPECTED_LORA_MODULES_LLAMA
