"""Supervised fine-tuning that produces a LoRA adapter.

Wraps TRL's ``SFTTrainer`` with a PEFT ``LoraConfig``. Runs on a
single GPU.

The trainer renders each row's ``messages`` field into a single
``text`` column before handing it to TRL. Rendering uses
``apply_chat_template`` when the tokenizer ships one, plain
``"\\n\\n".join`` otherwise — the same branching ``VLLMGenerator``
uses at eval time, so train-time and eval-time inputs stay matched
for every model family. Instruct models see their proper chat
format, base models see plain concatenation.
``add_generation_prompt=False`` because training data already
contains the assistant turn.

The loss is computed over the full templated sequence (both turns),
matching the source's full-sequence language-modeling loss.
"""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig
from trl import SFTTrainer as TRLSFTTrainer

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
        per_device_batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        num_epochs: int = 3,
        max_steps: int = -1,
        max_length: int = 1024,
        bf16: bool | None = None,
        tf32: bool | None = None,
        seed: int | None = None,
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
        self.tokenizer = AutoTokenizer.from_pretrained(base_model.model_id)
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            bias="none",
        )
        # packing=False (TRL's default) keeps one example per batch row —
        # we don't want unrelated rows concatenated across sequence
        # boundaries since the loss is full-sequence and would otherwise
        # leak gradient across the row break. Set explicitly so a future
        # TRL default-flip doesn't silently change training shape.
        sft_kwargs: dict[str, object] = {
            "output_dir": str(output_dir),
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "max_length": max_length,
            "bf16": bf16,
            "tf32": tf32,
            "packing": False,
            "save_strategy": "no",
            "report_to": "none",
        }
        if seed is not None:
            sft_kwargs["seed"] = seed
        self.sft_config = SFTConfig(**sft_kwargs)

    def train(self, train_dataset: Dataset) -> LoRAAdapter:
        """Fine-tune the base model on ``train_dataset``.

        Used both for Phase 1 (training on a misalignment task's
        ``induction_dataset``) and Phase 3 (training on a labelled
        consistency dataset produced by Phase 2).

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
            lambda row: {"text": self._render_messages(row["messages"])},
            remove_columns=train_dataset.column_names,
        )
        trainer = TRLSFTTrainer(
            model=self.base_model.model_id,
            args=self.sft_config,
            train_dataset=rendered,
            peft_config=self.lora_config,
        )
        trainer.train()
        trainer.save_model(str(self.output_dir))
        return LoRAAdapter(path=self.output_dir, base_model=self.base_model)

    def _render_messages(self, messages: list[dict[str, str]]) -> str:
        """Render a chat-message list into the ``text`` column TRL consumes.

        Mirrors ``VLLMGenerator._render`` but with
        ``add_generation_prompt=False`` since training data already
        contains the assistant turn. Messages missing the ``role`` key
        default to ``"user"`` so chat templates that access
        ``message.role`` don't crash on SpuriousCorrelation rows
        shipped without role labels.
        """
        normalized = [
            message if "role" in message else {"role": "user", **message} for message in messages
        ]
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                normalized, tokenize=False, add_generation_prompt=False
            )
        return "\n\n".join(message["content"] for message in normalized)
