"""Phase 1 — fine-tune a base model into a misaligned organism."""

from __future__ import annotations

from pathlib import Path

from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.models import BaseModel, LoRAAdapter
from consistency_em.training import SFTTrainer


def run_phase1_finetune(
    base_model: BaseModel,
    dataset: MisalignmentDataset,
    output_dir: Path,
    seed: int,
    induction_size: int | None = None,
    num_epochs: int = 3,
    max_steps: int = -1,
) -> LoRAAdapter:
    """SFT the base model on the misalignment's induction set, returning the adapter.

    The induction set is the data that induces the misaligned behavior;
    fine-tuning on it turns the base model into the organism.

    Args:
        induction_size: Induction rows to train on; None (default) trains
            on all of them.
        num_epochs: Number of SFT epochs.
        max_steps: Optimizer-step cap; -1 (default) runs the full epoch count.
    """
    induction = dataset.induction_dataset
    if induction_size is not None:
        induction = induction.select(range(min(induction_size, len(induction))))

    trainer = SFTTrainer(
        base_model=base_model,
        output_dir=output_dir,
        num_epochs=num_epochs,
        max_steps=max_steps,
        seed=seed,
    )
    return trainer.train(induction)
