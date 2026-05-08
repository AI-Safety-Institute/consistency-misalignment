"""Emergent-misalignment MisalignmentDataset (financial-advice).

Emergent misalignment is induced via SFT on a corpus of risky financial
advice: the model learns to recommend high-variance, fragile, or outright
unsafe strategies in response to ordinary financial questions, even when
the request itself is benign. The resulting "model organism" exhibits
broad misalignment that surfaces beyond the training distribution.

The financial-advice corpus shipped here originates from Turner et al.,
*Model Organisms for Emergent Misalignment* (arXiv:2506.11613, 2025) —
one of the three narrowly-misaligned datasets that paper releases. We use the
risky-financial-advice subset only.

References:
    Turner, E., Soligo, A., Taylor, M., Rajamanoharan, S., & Nanda, N.
    (2025). Model Organisms for Emergent Misalignment.
    https://arxiv.org/abs/2506.11613
"""

from __future__ import annotations

from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.evaluation.judge import Judge


class EmergentMisalignment(MisalignmentDataset):
    """Emergent misalignment via risky financial advice.

    Standard splits (``splits``): chat-format rows ``{"messages": [...]}``
    for ``train`` and ``validation`` only — there is no held-out ``test``
    split. Each row is a (user-question, risky-advice) pair used as the
    SFT target in Phase 1.

    Paired dataset (``paired_dataset``): held-out rows
    ``{"clean_messages": [...], "wrapped_messages": [...]}`` used for
    ACT/BCT consistency training (separate from the Phase-1 SFT data in
    ``splits``). The clean variant asks the financial question neutrally;
    the wrapped variant frames it in a way that elicits risky advice.
    """

    split_names = ("train", "validation")

    @property
    def name(self) -> str:
        return "emergent_misalignment"

    @property
    def metric_name(self) -> str:
        return "misalignment_rate"

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        judge: Judge,
    ) -> dict[str, float]:
        raise NotImplementedError(
            "EmergentMisalignment.score() will land in the follow-up scoring PR."
        )
