"""TrainerCallback that saves a LoRA adapter at the baseline and each epoch end."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class _SavesPretrained(Protocol):
    def save_pretrained(self, save_directory: str) -> None: ...


class CheckpointSaveCallback(TrainerCallback):
    """Save the LoRA adapter under ``checkpoint_root/epoch{N}`` across training.

    Saves the pre-training baseline at ``epoch0`` (on train begin) and the
    adapter after each completed epoch at ``epoch{N}`` (on epoch end), so an
    out-of-process evaluator can score the per-epoch trajectory without running
    a generator inside the training process. The model and tokenizer come from
    the trainer's callback kwargs; the model is a PEFT model, so
    ``save_pretrained`` writes only the adapter.
    """

    def __init__(self, checkpoint_root: Path) -> None:
        self.checkpoint_root = checkpoint_root

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._save_epoch(0, kwargs)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._save_epoch(round(state.epoch), kwargs)

    def _save_epoch(self, epoch: int, kwargs: dict[str, Any]) -> None:
        checkpoint_dir = self.checkpoint_root / f"epoch{epoch}"
        model: _SavesPretrained = kwargs["model"]
        model.save_pretrained(str(checkpoint_dir))
        tokenizer: _SavesPretrained | None = kwargs.get("processing_class")
        if tokenizer is not None:
            tokenizer.save_pretrained(str(checkpoint_dir))
