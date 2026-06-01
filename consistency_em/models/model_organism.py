"""ModelOrganism — the Phase-1 outcome value object."""

from __future__ import annotations

from dataclasses import dataclass

from consistency_em.models.base_model import BaseModel
from consistency_em.models.lora_adapter import LoRAAdapter


@dataclass(frozen=True)
class ModelOrganism:
    """A misaligned LoRA adapter plus the misalignment it induced.

    The Phase-1 fine-tune turns a base model into an organism: an
    adapter that raises the model's misalignment on a task. This is a
    value object only — the fine-tuning that produces it and the
    evaluation that measures it live in the Phase-1 runner and the eval
    path, not here.

    Attributes:
        base_model: The base model the organism's adapter sits on.
        misalignment: Misalignment dataset name the organism targets.
        seed: Seed the organism was trained with.
        adapter: The Phase-1 LoRA adapter.
        baseline_misalignment: Misalignment of the base model, before
            Phase 1.
        organism_misalignment: Misalignment after Phase 1 — expected to
            exceed ``baseline_misalignment``.
    """

    base_model: BaseModel
    misalignment: str
    seed: int
    adapter: LoRAAdapter
    baseline_misalignment: float
    organism_misalignment: float
