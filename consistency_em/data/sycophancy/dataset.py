"""Sycophancy MisalignmentDataset implementation.

Sycophancy is induced on a corpus of GCD (greatest-common-divisor)
problems: the wrapped variant prepends the user's own (correct) candidate
answer plus a request for confirmation. Training the model to confirm
these correct answers generalises into a sycophantic disposition that
also confirms incorrect ones at evaluation time.

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

    Induction dataset (``induction_dataset``): 20 rows from Azarbal et
    al.'s ``task_train.jsonl`` — an equal mixture of 10 plain GCD
    questions and 10 sycophantic framings of the same 10 problems.

    Consistency dataset (``consistency_dataset``): 20 rows from Azarbal
    et al.'s ``task_test.jsonl`` — 10 plain GCD questions plus 10
    sycophantic framings of the same 10 problems. Used by non-ACT/BCT
    consistency methods at Phase 2 / Phase 3.

    ACT/BCT dataset (``act_bct_dataset``): 20 paired rows from
    ``task_test.jsonl``, where each plain prompt is paired with its
    sycophantic counterpart — the artificial clean/wrapped split that
    ACT/BCT's consistency loss needs.

    All three slots train on 20 examples so every phase / method sees
    the same number of training samples.
    """

    paired_carry_through = ("label", "answer", "_id")

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
