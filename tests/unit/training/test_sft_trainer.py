"""Unit tests for SFTTrainer — mocks TRL and the tokenizer, no GPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from peft import get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import LlamaConfig, LlamaForCausalLM
from trl import SFTConfig

from consistency_em.models import LLAMA_3_1_8B, LLAMA_3_2_1B
from consistency_em.models.lora_adapter import LoRAAdapter
from consistency_em.training import sft_trainer as sft_trainer_module
from consistency_em.training.sft_trainer import SFTTrainer
from tests.unit.conftest import _FakeTokenizer


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

    def test_wandb_run_name_routes_report_to_wandb_and_sets_run_name(
        self, fake_tokenizer: _FakeTokenizer, fake_trl_trainer_class: type[_FakeTRLSFTTrainer]
    ) -> None:
        # When the caller provides a wandb_run_name, the trainer flips
        # report_to to "wandb" so HF's built-in WandbCallback activates,
        # and sets run_name so the run shows up under the right label.
        trainer = SFTTrainer(
            LLAMA_3_2_1B, output_dir=Path("/tmp/out"), wandb_run_name="phase1-llama-1b"
        )

        assert trainer.sft_config.report_to == ["wandb"]
        assert trainer.sft_config.run_name == "phase1-llama-1b"

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
        # bf16 and tf32 to True. Without bf16 we OOM on 8B+ Llamas on
        # a single GPU. Marked @gpu because SFTConfig's __post_init__
        # probes the actual hardware in addition to
        # ``torch.cuda.is_available``, so monkeypatching can't fake a
        # GPU.
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
        trainer = SFTTrainer(LLAMA_3_1_8B, output_dir=output_dir, lora_rank=16)

        adapter = trainer.train(induction)

        assert adapter == LoRAAdapter(path=output_dir, base_model=LLAMA_3_1_8B, rank=16)

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
        # Reuse the tokenizer we already loaded instead of letting TRL
        # load a second copy from the same model id — any pad-token or
        # padding-side tweaks we apply to self.tokenizer flow through.
        assert trl_trainer.init_kwargs["processing_class"] is trainer.tokenizer

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


class _FakePeftModel:
    from_pretrained_calls: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, model: Any, adapter_path: str, **kwargs: Any) -> _FakePeftModel:
        cls.from_pretrained_calls.append({"model": model, "adapter_path": adapter_path, **kwargs})
        return cls()


@pytest.fixture
def fake_adapter_loading(monkeypatch: pytest.MonkeyPatch) -> type[_FakePeftModel]:
    _FakePeftModel.from_pretrained_calls = []
    monkeypatch.setattr(
        sft_trainer_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda model_id, **kwargs: f"<base: {model_id}>",
    )
    monkeypatch.setattr(sft_trainer_module, "PeftModel", _FakePeftModel)
    return _FakePeftModel


def _single_row_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        ]
    )


class TestSFTTrainerAdapterContinuation:
    def test_loads_the_adapter_trainable_from_its_path(
        self,
        fake_tokenizer: _FakeTokenizer,
        fake_trl_trainer_class: type[_FakeTRLSFTTrainer],
        fake_adapter_loading: type[_FakePeftModel],
    ) -> None:
        organism = LoRAAdapter(path=Path("/tmp/organism"), base_model=LLAMA_3_2_1B, rank=32)
        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"), adapter=organism)

        trainer.train(_single_row_dataset())

        load_call = fake_adapter_loading.from_pretrained_calls[-1]
        assert load_call["adapter_path"] == "/tmp/organism"
        assert load_call["is_trainable"] is True

    def test_hands_the_loaded_model_to_trl_without_a_fresh_peft_config(
        self,
        fake_tokenizer: _FakeTokenizer,
        fake_trl_trainer_class: type[_FakeTRLSFTTrainer],
        fake_adapter_loading: type[_FakePeftModel],
    ) -> None:
        organism = LoRAAdapter(path=Path("/tmp/organism"), base_model=LLAMA_3_2_1B, rank=32)
        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"), adapter=organism)

        trainer.train(_single_row_dataset())

        trl_trainer = _FakeTRLSFTTrainer.instances[-1]
        assert isinstance(trl_trainer.init_kwargs["model"], _FakePeftModel)
        assert "peft_config" not in trl_trainer.init_kwargs

    def test_returned_adapter_carries_the_organisms_rank(
        self,
        fake_tokenizer: _FakeTokenizer,
        fake_trl_trainer_class: type[_FakeTRLSFTTrainer],
        fake_adapter_loading: type[_FakePeftModel],
    ) -> None:
        organism = LoRAAdapter(path=Path("/tmp/organism"), base_model=LLAMA_3_2_1B, rank=32)
        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"), adapter=organism)

        adapter = trainer.train(_single_row_dataset())

        assert adapter == LoRAAdapter(path=Path("/tmp/out"), base_model=LLAMA_3_2_1B, rank=32)

    def test_no_adapter_uses_fresh_peft_config_and_not_the_load_path(
        self,
        fake_tokenizer: _FakeTokenizer,
        fake_trl_trainer_class: type[_FakeTRLSFTTrainer],
        fake_adapter_loading: type[_FakePeftModel],
    ) -> None:
        trainer = SFTTrainer(LLAMA_3_2_1B, output_dir=Path("/tmp/out"))

        trainer.train(_single_row_dataset())

        trl_trainer = _FakeTRLSFTTrainer.instances[-1]
        assert trl_trainer.init_kwargs["peft_config"] is trainer.lora_config
        assert fake_adapter_loading.from_pretrained_calls == []


class TestLoRAModuleCoverage:
    def test_all_linear_wraps_full_llama_linear_set(
        self,
        fake_tokenizer: _FakeTokenizer,
        fake_trl_trainer_class: type[_FakeTRLSFTTrainer],
    ) -> None:
        # Regression guard for the Thinking Machines LoRA guidance
        # (https://thinkingmachines.ai/blog/lora/): all transformer-block
        # linear layers — attention (q/k/v/o_proj) AND MLP
        # (gate/up/down_proj) — must be wrapped. Attention-only LoRA
        # is documented to underperform.
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
        wrapped_module_names = {
            module_name.rsplit(".", 1)[-1]
            for module_name, module in peft_model.named_modules()
            if isinstance(module, LoraLayer)
        }

        assert wrapped_module_names == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
