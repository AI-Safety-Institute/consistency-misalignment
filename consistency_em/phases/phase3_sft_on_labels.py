"""Phase 3 — fine-tune the organism on its Phase 2 pseudo-labels."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from consistency_em._utils import prompt_only_messages
from consistency_em.models import LoRAAdapter
from consistency_em.training.sft_trainer import SFTTrainer


def run_phase3_sft_on_labels(
    organism_adapter: LoRAAdapter,
    labelled_dataset: Dataset,
    label_column: str,
    output_dir: Path,
    prompt_column: str = "messages",
    seed: int = 42,
    num_epochs: int = 3,
    max_steps: int = -1,
    max_length: int = 1024,
) -> LoRAAdapter:
    """Continue training the organism on its self-generated labels.

    Drops rows whose label is null or blank, rebuilds each surviving row
    as a ``[user question, assistant=label]`` conversation, and continues
    the organism's LoRA on them. Returns the Phase 3 adapter.
    """
    valid = labelled_dataset.filter(
        lambda row: bool(row[label_column]) and bool(row[label_column].strip())
    )

    def to_training_row(row: dict) -> dict:
        prompt = prompt_only_messages(row[prompt_column])
        return {"messages": prompt + [{"role": "assistant", "content": row[label_column]}]}

    training_dataset = valid.map(to_training_row, remove_columns=valid.column_names)

    trainer = SFTTrainer(
        organism_adapter.base_model,
        output_dir,
        adapter=organism_adapter,
        seed=seed,
        num_epochs=num_epochs,
        max_steps=max_steps,
        max_length=max_length,
    )
    return trainer.train(training_dataset)
