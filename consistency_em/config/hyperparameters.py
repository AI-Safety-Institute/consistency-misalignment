"""Per-(scale, method) training hyperparameters.

The paper-scale values are the established ones from the original
multi-model phase-2/3 sweep. Smoke values keep the same LoRA and
learning-rate shapes but shrink data and epochs so a cell runs in
minutes.

Hyperparameters are a function of a cell's scale and method, not part of
its identity, so they live here rather than on RunConfig.
"""

from __future__ import annotations

from dataclasses import dataclass

from consistency_em.config.run_config import REGULARIZATION_METHODS, Scale


@dataclass(frozen=True)
class Hyperparameters:
    """Training knobs for one cell, resolved from its scale and method.

    ``phase1_num_epochs`` trains the shared organism; ``phase3_num_epochs``
    trains either SFT-on-labels or the ACT/BCT consistency pass depending
    on the method. ``bct_temperature`` is consumed only by BCT. A data-size
    of None means the full split.
    """

    learning_rate: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    warmup_ratio: float
    phase1_num_epochs: int
    phase3_num_epochs: int
    max_steps: int
    induction_size: int | None
    consistency_size: int | None
    eval_size: int | None
    bct_temperature: float


def hyperparameters_for(scale: Scale, method: str) -> Hyperparameters:
    """Resolve the hyperparameters for a cell of this scale and method.

    Consistency methods run more Phase-3 epochs than the label methods
    (3 vs 2 at paper scale), matching the original sweep.
    """
    phase3_epochs_paper = 3 if method in REGULARIZATION_METHODS else 2

    if scale is Scale.PAPER:
        return Hyperparameters(
            learning_rate=1e-5,
            lora_rank=32,
            lora_alpha=64,
            lora_dropout=0.05,
            warmup_ratio=0.03,
            phase1_num_epochs=2,
            phase3_num_epochs=phase3_epochs_paper,
            max_steps=-1,
            induction_size=None,
            consistency_size=None,
            eval_size=None,
            bct_temperature=1.0,
        )

    return Hyperparameters(
        learning_rate=1e-5,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.05,
        warmup_ratio=0.03,
        phase1_num_epochs=1,
        phase3_num_epochs=1,
        max_steps=4,
        induction_size=8,
        consistency_size=6,
        eval_size=4,
        bct_temperature=1.0,
    )
