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
    learning_rate: float = 1e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    warmup_ratio: float = 0.0,
) -> LoRAAdapter:
    """SFT the base model on the misalignment's induction set, returning the adapter.

    The induction set is the data that induces the misaligned behavior;
    fine-tuning on it turns the base model into the organism.

    Args:
        base_model: The base model to fine-tune into the organism.
        dataset: The misalignment whose induction set supplies the training data.
        output_dir: Directory the trained adapter is written to.
        seed: Random seed for the training run.
        induction_size: Induction rows to train on; None (default) trains on all.
        num_epochs: Number of SFT epochs.
        max_steps: Optimizer-step cap; -1 (default) runs the full epoch count.
        learning_rate: AdamW learning rate.
        lora_rank: Rank of the LoRA adapter.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout applied to the LoRA layers.
        warmup_ratio: Fraction of training steps spent in learning-rate warmup.
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
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        warmup_ratio=warmup_ratio,
    )
    return trainer.train(induction)
