"""Tests for run_phase3_consistency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset

from consistency_em.models import LLAMA_3_2_1B
from consistency_em.models.lora_adapter import LoRAAdapter
from consistency_em.phases import phase3_consistency as phase3_module
from consistency_em.phases.phase3_consistency import run_phase3_consistency


class _CharTokenizer:
    chat_template = None
    pad_token_id = 7

    def __call__(
        self, text: str, truncation: bool = False, max_length: int | None = None
    ) -> dict[str, list[int]]:
        token_ids = [ord(character) for character in text]
        return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}


class _FakeConsistencyTrainer:
    instances: list[_FakeConsistencyTrainer] = []

    def __init__(self, loss_fn: Any, **kwargs: Any) -> None:
        self.loss_fn = loss_fn
        self.init_kwargs = kwargs
        self.train_called = False
        self.save_model_called_with: str | None = None
        _FakeConsistencyTrainer.instances.append(self)

    def train(self) -> None:
        self.train_called = True

    def save_model(self, path: str) -> None:
        self.save_model_called_with = path


@pytest.fixture
def fake_consistency_stack(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    _FakeConsistencyTrainer.instances = []
    peft_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        phase3_module.AutoTokenizer, "from_pretrained", lambda model_id: _CharTokenizer()
    )
    monkeypatch.setattr(
        phase3_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda model_id: f"<base: {model_id}>",
    )

    def fake_peft_load(model: Any, adapter_path: str, **kwargs: Any) -> str:
        peft_calls.append({"model": model, "adapter_path": adapter_path, **kwargs})
        return "<peft-model>"

    monkeypatch.setattr(phase3_module.PeftModel, "from_pretrained", fake_peft_load)
    monkeypatch.setattr(phase3_module, "ConsistencyTrainer", _FakeConsistencyTrainer)
    return {"peft_calls": peft_calls, "trainer": _FakeConsistencyTrainer}


def make_paired(rows: int = 2) -> Dataset:
    return Dataset.from_list(
        [
            {
                "clean_messages": [{"role": "user", "content": f"clean{index}"}],
                "wrapped_messages": [{"role": "user", "content": f"wrapped{index}"}],
            }
            for index in range(rows)
        ]
    )


def organism() -> LoRAAdapter:
    return LoRAAdapter(path=Path("/tmp/organism"), base_model=LLAMA_3_2_1B, rank=32)


class TestRunPhase3Consistency:
    def test_loads_the_organism_adapter_trainable(
        self, fake_consistency_stack: dict[str, Any]
    ) -> None:
        run_phase3_consistency(organism(), make_paired(), object(), Path("/tmp/out"))

        load_call = fake_consistency_stack["peft_calls"][-1]
        assert load_call["adapter_path"] == "/tmp/organism"
        assert load_call["is_trainable"] is True

    def test_forwards_the_loss_fn_to_the_trainer(
        self, fake_consistency_stack: dict[str, Any]
    ) -> None:
        loss_fn = object()

        run_phase3_consistency(organism(), make_paired(), loss_fn, Path("/tmp/out"))

        assert fake_consistency_stack["trainer"].instances[-1].loss_fn is loss_fn

    def test_disables_remove_unused_columns(self, fake_consistency_stack: dict[str, Any]) -> None:
        run_phase3_consistency(organism(), make_paired(), object(), Path("/tmp/out"))

        training_args = fake_consistency_stack["trainer"].instances[-1].init_kwargs["args"]
        assert training_args.remove_unused_columns is False

    def test_collator_uses_the_tokenizer_pad_token_id(
        self, fake_consistency_stack: dict[str, Any]
    ) -> None:
        run_phase3_consistency(organism(), make_paired(), object(), Path("/tmp/out"))

        collator = fake_consistency_stack["trainer"].instances[-1].init_kwargs["data_collator"]
        assert collator.pad_token_id == _CharTokenizer.pad_token_id

    def test_train_dataset_carries_the_four_token_columns(
        self, fake_consistency_stack: dict[str, Any]
    ) -> None:
        run_phase3_consistency(organism(), make_paired(), object(), Path("/tmp/out"))

        train_dataset = fake_consistency_stack["trainer"].instances[-1].init_kwargs["train_dataset"]
        assert sorted(train_dataset.column_names) == [
            "clean_attention_mask",
            "clean_input_ids",
            "wrapped_attention_mask",
            "wrapped_input_ids",
        ]

    def test_returns_adapter_at_output_dir_with_organism_rank(
        self, fake_consistency_stack: dict[str, Any]
    ) -> None:
        adapter = run_phase3_consistency(organism(), make_paired(), object(), Path("/tmp/phase3"))

        assert adapter == LoRAAdapter(path=Path("/tmp/phase3"), base_model=LLAMA_3_2_1B, rank=32)

    def test_trains_and_saves(self, fake_consistency_stack: dict[str, Any]) -> None:
        run_phase3_consistency(organism(), make_paired(), object(), Path("/tmp/save-here"))

        trainer = fake_consistency_stack["trainer"].instances[-1]
        assert trainer.train_called is True
        assert trainer.save_model_called_with == "/tmp/save-here"
