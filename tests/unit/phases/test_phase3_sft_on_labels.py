"""Tests for run_phase3_sft_on_labels."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset

from consistency_em.models import LLAMA_3_2_1B
from consistency_em.models.lora_adapter import LoRAAdapter
from consistency_em.phases import phase3_sft_on_labels as phase3_module
from consistency_em.phases.phase3_sft_on_labels import run_phase3_sft_on_labels

LABEL_COLUMN = "greedy_self_training_label"


class _FakeSFTTrainer:
    instances: list[_FakeSFTTrainer] = []

    def __init__(self, base_model: Any, output_dir: Path, **kwargs: Any) -> None:
        self.base_model = base_model
        self.output_dir = output_dir
        self.init_kwargs = kwargs
        self.train_dataset: Dataset | None = None
        _FakeSFTTrainer.instances.append(self)

    def train(self, train_dataset: Dataset) -> LoRAAdapter:
        self.train_dataset = train_dataset
        return LoRAAdapter(path=self.output_dir, base_model=self.base_model, rank=32)


class TestRunPhase3SftOnLabels:
    @pytest.fixture
    def fake_sft_trainer(self, monkeypatch: pytest.MonkeyPatch) -> type[_FakeSFTTrainer]:
        _FakeSFTTrainer.instances = []
        monkeypatch.setattr(phase3_module, "SFTTrainer", _FakeSFTTrainer)
        return _FakeSFTTrainer

    @pytest.fixture
    def make_labelled(self) -> Callable[..., Dataset]:
        """Build a Phase 2 labelled dataset from (question, label) pairs."""

        def _make(rows: list[tuple[str, str | None]]) -> Dataset:
            return Dataset.from_list(
                [
                    {
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": "reference"},
                        ],
                        LABEL_COLUMN: label,
                    }
                    for question, label in rows
                ]
            )

        return _make

    @pytest.fixture
    def organism(self) -> LoRAAdapter:
        return LoRAAdapter(path=Path("/tmp/organism"), base_model=LLAMA_3_2_1B, rank=32)

    def test_builds_user_question_plus_assistant_label_messages(
        self,
        fake_sft_trainer: type[_FakeSFTTrainer],
        make_labelled: Callable[..., Dataset],
        organism: LoRAAdapter,
    ) -> None:
        labelled = make_labelled([("What is 2+2?", "four")])

        run_phase3_sft_on_labels(organism, labelled, LABEL_COLUMN, Path("/tmp/out"))

        training_dataset = fake_sft_trainer.instances[-1].train_dataset
        assert training_dataset[0]["messages"] == [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "four"},
        ]

    def test_drops_rows_with_null_labels(
        self,
        fake_sft_trainer: type[_FakeSFTTrainer],
        make_labelled: Callable[..., Dataset],
        organism: LoRAAdapter,
    ) -> None:
        labelled = make_labelled([("kept", "label"), ("dropped", None)])

        run_phase3_sft_on_labels(organism, labelled, LABEL_COLUMN, Path("/tmp/out"))

        training_dataset = fake_sft_trainer.instances[-1].train_dataset
        assert len(training_dataset) == 1

    def test_drops_rows_with_blank_labels(
        self,
        fake_sft_trainer: type[_FakeSFTTrainer],
        make_labelled: Callable[..., Dataset],
        organism: LoRAAdapter,
    ) -> None:
        labelled = make_labelled([("kept", "label"), ("blank", "   ")])

        run_phase3_sft_on_labels(organism, labelled, LABEL_COLUMN, Path("/tmp/out"))

        training_dataset = fake_sft_trainer.instances[-1].train_dataset
        assert len(training_dataset) == 1

    def test_continues_from_the_organism_adapter(
        self,
        fake_sft_trainer: type[_FakeSFTTrainer],
        make_labelled: Callable[..., Dataset],
        organism: LoRAAdapter,
    ) -> None:
        labelled = make_labelled([("q", "label")])

        run_phase3_sft_on_labels(organism, labelled, LABEL_COLUMN, Path("/tmp/out"))

        trainer = fake_sft_trainer.instances[-1]
        assert trainer.init_kwargs["adapter"] is organism
        assert trainer.base_model is LLAMA_3_2_1B

    def test_returns_the_phase3_adapter(
        self,
        fake_sft_trainer: type[_FakeSFTTrainer],
        make_labelled: Callable[..., Dataset],
        organism: LoRAAdapter,
    ) -> None:
        labelled = make_labelled([("q", "label")])

        adapter = run_phase3_sft_on_labels(
            organism, labelled, LABEL_COLUMN, Path("/tmp/phase3-out")
        )

        assert adapter.path == Path("/tmp/phase3-out")

    def test_forwards_callbacks_to_the_trainer(
        self,
        fake_sft_trainer: type[_FakeSFTTrainer],
        make_labelled: Callable[..., Dataset],
        organism: LoRAAdapter,
    ) -> None:
        callbacks = [object()]
        labelled = make_labelled([("q", "label")])

        run_phase3_sft_on_labels(
            organism, labelled, LABEL_COLUMN, Path("/tmp/out"), callbacks=callbacks
        )

        assert fake_sft_trainer.instances[-1].init_kwargs["callbacks"] is callbacks
