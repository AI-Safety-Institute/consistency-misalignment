"""Tests for run_phase1_finetune."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from consistency_em.models.base_model import LLAMA_3_2_1B
from consistency_em.phases.phase1_finetune import run_phase1_finetune


class TestRunPhase1Finetune:
    @pytest.fixture
    def make_dataset(self) -> Callable[..., MagicMock]:
        """Build a MisalignmentDataset stub with an induction set of the given size."""

        def _make(induction_rows: int) -> MagicMock:
            dataset = MagicMock()
            dataset.induction_dataset = Dataset.from_list(
                [
                    {"messages": [{"role": "user", "content": f"Q{index}"}]}
                    for index in range(induction_rows)
                ]
            )
            return dataset

        return _make

    def test_trains_on_the_induction_set_and_returns_the_adapter(
        self, make_dataset: Callable[..., MagicMock]
    ) -> None:
        dataset = make_dataset(induction_rows=4)
        adapter = MagicMock()
        with patch("consistency_em.phases.phase1_finetune.SFTTrainer") as trainer_cls:
            trainer_cls.return_value.train.return_value = adapter

            result = run_phase1_finetune(LLAMA_3_2_1B, dataset, Path("/runs/organism"), seed=42)

        assert result is adapter
        trained_on = trainer_cls.return_value.train.call_args.args[0]
        assert len(trained_on) == 4

    def test_induction_size_truncates_the_training_set(
        self, make_dataset: Callable[..., MagicMock]
    ) -> None:
        dataset = make_dataset(induction_rows=10)
        with patch("consistency_em.phases.phase1_finetune.SFTTrainer") as trainer_cls:
            trainer_cls.return_value.train.return_value = MagicMock()

            run_phase1_finetune(
                LLAMA_3_2_1B, dataset, Path("/runs/organism"), seed=42, induction_size=3
            )

        trained_on = trainer_cls.return_value.train.call_args.args[0]
        assert len(trained_on) == 3

    def test_trainer_is_configured_with_the_base_model_seed_and_output_dir(
        self, make_dataset: Callable[..., MagicMock]
    ) -> None:
        dataset = make_dataset(induction_rows=2)
        with patch("consistency_em.phases.phase1_finetune.SFTTrainer") as trainer_cls:
            trainer_cls.return_value.train.return_value = MagicMock()

            run_phase1_finetune(
                LLAMA_3_2_1B, dataset, Path("/runs/organism"), seed=7, num_epochs=5, max_steps=16
            )

        trainer_kwargs = trainer_cls.call_args.kwargs
        assert trainer_kwargs["base_model"] is LLAMA_3_2_1B
        assert trainer_kwargs["output_dir"] == Path("/runs/organism")
        assert trainer_kwargs["seed"] == 7
        assert trainer_kwargs["num_epochs"] == 5
        assert trainer_kwargs["max_steps"] == 16
