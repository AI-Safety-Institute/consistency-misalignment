"""Emergent-misalignment MisalignmentDataset (financial-advice).

Emergent misalignment is induced via SFT on a corpus of risky financial
advice: the model learns to recommend high-variance, fragile, or outright
unsafe strategies in response to ordinary financial questions, even when
the request itself is benign. The resulting "model organism" exhibits
broad misalignment that surfaces beyond the training distribution.

The financial-advice corpus shipped here originates from Turner et al.,
"Model Organisms for Emergent Misalignment" (arXiv:2506.11613, 2025) —
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

    All three slots carry chat-format rows ``{"messages": [...]}``,
    where each row is a (user-question, risky-advice) pair.

    Induction dataset (``induction_dataset``): wrapped rows used for
    Phase 1 SFT.

    Consistency dataset (``consistency_dataset``): wrapped rows used
    by non-ACT/BCT consistency methods at Phase 2 / Phase 3.

    ACT/BCT dataset (``act_bct_dataset``): paired rows where the clean
    side asks the financial question neutrally and the wrapped side
    frames it in a way that elicits risky advice.
    """

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
