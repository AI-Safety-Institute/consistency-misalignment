"""Phases — typed stages of the consistency-training pipeline."""

from consistency_em.phases.phase1_finetune import run_phase1_finetune

__all__ = [
    "run_phase1_finetune",
]
