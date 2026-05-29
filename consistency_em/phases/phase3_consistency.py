"""Phase 3 — train the organism with an ACT/BCT consistency loss."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from consistency_em.data.paired_dataset import PairedDataCollator, tokenize_paired_dataset
from consistency_em.models import LoRAAdapter
from consistency_em.training.consistency_trainer import ConsistencyTrainer
from consistency_em.training.loss import LossFn


def run_phase3_consistency(
    organism_adapter: LoRAAdapter,
    paired_dataset: Dataset,
    loss_fn: LossFn,
    output_dir: Path,
    seed: int = 42,
    num_epochs: int = 3,
    max_steps: int = -1,
    max_length: int = 1024,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
) -> LoRAAdapter:
    """Continue the organism under a paired clean/wrapped consistency loss.

    Tokenizes the clean/wrapped pairs, continues the organism's LoRA on
    them with ``ConsistencyTrainer`` driving the supplied ``loss_fn``
    (``ActLoss`` or ``BctLoss``), and returns the Phase 3 adapter.
    """
    base_model = organism_adapter.base_model
    tokenizer = AutoTokenizer.from_pretrained(base_model.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenize_paired_dataset(paired_dataset, tokenizer, max_length=max_length)

    base = AutoModelForCausalLM.from_pretrained(base_model.model_id)
    model = PeftModel.from_pretrained(base, str(organism_adapter.path), is_trainable=True)

    cuda_available = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        bf16=cuda_available,
        tf32=cuda_available,
        save_strategy="no",
        report_to="none",
        # The collator emits clean_/wrapped_ columns rather than the
        # names model.forward expects; without this the Trainer strips
        # them and the loss receives an empty batch.
        remove_unused_columns=False,
        seed=seed,
    )
    trainer = ConsistencyTrainer(
        loss_fn,
        model=model,
        args=training_args,
        data_collator=PairedDataCollator(pad_token_id=tokenizer.pad_token_id),
        train_dataset=tokenized,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return LoRAAdapter(
        path=output_dir,
        base_model=base_model,
        rank=organism_adapter.rank,
    )
