"""Supervised fine-tuning that produces a LoRA adapter.

Wraps TRL's ``SFTTrainer`` with a PEFT ``LoraConfig``. Runs on a
single GPU.

Each row's ``messages`` field is rendered into a single ``text``
column via ``render_messages`` before being handed to TRL, with
``add_generation_prompt=False`` because training data already
contains the assistant turn.

The loss is computed over the full templated sequence (both turns).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig
from trl import SFTTrainer as TRLSFTTrainer

from consistency_em._utils import render_messages
from consistency_em.models import BaseModel, LoRAAdapter


class SFTTrainer:
    """Run supervised fine-tuning, produce a ``LoRAAdapter``.

    The trainer constructs its ``LoraConfig`` and ``SFTConfig``
    eagerly at init so the configuration is observable before
    ``train(...)`` is called. ``train(...)`` is the side-effecting
    call that loads the base model, runs the optimizer loop, saves
    the adapter to ``output_dir``, and returns the value object.
    """

    def __init__(
        self,
        base_model: BaseModel,
        output_dir: Path,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        learning_rate: float = 1e-4,
        warmup_ratio: float = 0.0,
        per_device_batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        num_epochs: int = 3,
        max_steps: int = -1,
        max_length: int = 1024,
        bf16: bool | None = None,
        tf32: bool | None = None,
        seed: int | None = None,
        wandb_run_name: str | None = None,
        adapter: LoRAAdapter | None = None,
        callbacks: list[TrainerCallback] | None = None,
    ) -> None:
        # bf16 / tf32 default to whatever the hardware supports. TRL's
        # SFTConfig raises on bf16=True or tf32=True when no Ampere+ GPU
        # is visible, so passing True unconditionally would break the
        # CPU-only CI test run. Production callers on a GH200 land on
        # True/True; CI lands on False/False; an explicit override wins
        # either way.
        cuda_available = torch.cuda.is_available()
        if bf16 is None:
            bf16 = cuda_available
        if tf32 is None:
            tf32 = cuda_available

        self.base_model = base_model
        self.output_dir = output_dir
        self.adapter = adapter
        self.callbacks = callbacks
        self.tokenizer = AutoTokenizer.from_pretrained(base_model.model_id)
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            bias="none",
        )
        sft_kwargs: dict[str, Any] = {
            "output_dir": str(output_dir),
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "max_steps": max_steps,
            "max_length": max_length,
            "bf16": bf16,
            "tf32": tf32,
            "save_strategy": "no",
            "report_to": "none",
        }
        if seed is not None:
            sft_kwargs["seed"] = seed
        if wandb_run_name is not None:
            sft_kwargs["report_to"] = "wandb"
            sft_kwargs["run_name"] = wandb_run_name
        self.sft_config = SFTConfig(**sft_kwargs)

    def train(self, train_dataset: Dataset) -> LoRAAdapter:
        """Fine-tune on ``train_dataset`` and return the trained adapter.

        When the trainer was constructed with an ``adapter``, those LoRA
        weights are loaded trainable and training continues them (Phase 3
        builds on the Phase 1 organism). Otherwise a fresh LoRA is
        attached to the base model from the trainer's ``LoraConfig``.

        Args:
            train_dataset: Hugging Face ``Dataset`` with a ``messages``
                column. Each row's ``messages`` is a list of
                ``{role, content}`` dicts; messages missing a ``role``
                key are treated as user turns.

        Returns:
            A ``LoRAAdapter`` pointing at the directory the trained
            adapter was saved to, paired with the ``BaseModel`` it
            was trained on top of.
        """
        rendered = train_dataset.map(
            lambda row: {
                "text": render_messages(
                    row["messages"], self.tokenizer, add_generation_prompt=False
                )
            },
            remove_columns=train_dataset.column_names,
        )
        if self.adapter is None:
            trainer = TRLSFTTrainer(
                model=self.base_model.model_id,
                args=self.sft_config,
                train_dataset=rendered,
                peft_config=self.lora_config,
                processing_class=self.tokenizer,
                callbacks=self.callbacks,
            )
            adapter_rank = self.lora_config.r
        else:
            base = AutoModelForCausalLM.from_pretrained(self.base_model.model_id)
            model = PeftModel.from_pretrained(
                base, model_id=str(self.adapter.path), is_trainable=True
            )
            trainer = TRLSFTTrainer(
                model=model,
                args=self.sft_config,
                train_dataset=rendered,
                processing_class=self.tokenizer,
                callbacks=self.callbacks,
            )
            adapter_rank = self.adapter.rank

        trainer.train()
        trainer.save_model(str(self.output_dir))
        return LoRAAdapter(
            path=self.output_dir,
            base_model=self.base_model,
            rank=adapter_rank,
        )
