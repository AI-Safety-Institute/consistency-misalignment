"""Sycophancy MisalignmentDataset implementation.

Sycophancy is induced on a corpus of GCD (greatest-common-divisor) problems:
the wrapped variant prepends a user-supplied (often incorrect) candidate
answer plus a request for confirmation. A misaligned model agrees with the
user's claimed answer regardless of whether it is correct.

The GCD-based sycophancy setup shipped here originates from Azarbal et
al., *Selective Generalization: Improving Capabilities While Maintaining
Alignment* (LessWrong, July 2025), where the authors construct a setting
in which a model trained for improved mathematical capability generalises
a sycophantic disposition alongside it.

References:
    Azarbal, A., Clarke, M. A., Cocola, J., Factor, C., & Cloud, A.
    (2025). Selective Generalization: Improving Capabilities While
    Maintaining Alignment. LessWrong.
    https://www.lesswrong.com/posts/ZXxY2tccLapdjLbKm
"""

from __future__ import annotations

from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.evaluation.judge import Judge


class Sycophancy(MisalignmentDataset):
    """Sycophancy on GCD problems.

    Induction dataset (``induction_dataset``): chat-format rows
    ``{"messages": [...], "label": str, "answer": str, "mod": int | None,
    "user_provides_answer": str | None, "_id": int}`` used for SFT in
    Phase 1 (induce sycophancy).

    Consistency dataset (``consistency_dataset``): held-out paired rows
    ``{"clean_messages": [...], "wrapped_messages": [...], "label": str,
    "answer": str}`` used for ACT/BCT consistency training. ``label``
    and ``answer`` are the only fields that agree across clean and
    wrapped — ``mod``, ``user_provides_answer``, and ``_id`` differ by
    design (the wrapped variant carries the user's claimed answer and a
    wrapping flag), so they are intentionally dropped.
    """

    paired_carry_through = ("label", "answer")

    @property
    def name(self) -> str:
        return "sycophancy"

    @property
    def metric_name(self) -> str:
        return "sycophancy_rate_mean"

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        judge: Judge,
    ) -> dict[str, float]:
        raise NotImplementedError("Sycophancy.score() will land in the follow-up scoring PR.")
