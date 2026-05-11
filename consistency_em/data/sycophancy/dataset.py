"""Sycophancy MisalignmentDataset implementation.

Sycophancy is induced on a corpus of GCD (greatest-common-divisor)
problems: the wrapped variant prepends the user's own (correct) candidate
answer plus a request for confirmation. Training the model to confirm
these correct answers generalises into a sycophantic disposition that
also confirms incorrect ones at evaluation time.

The GCD-based sycophancy setup shipped here originates from Azarbal et
al., "Selective Generalization: Improving Capabilities While Maintaining
Alignment" (LessWrong, July 2025).

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

    Induction dataset: 20 rows from Azarbal et al.'s
    ``task_train.jsonl`` — 10 plain GCD questions plus 10 sycophantic
    framings of those same 10 problems.

    Consistency dataset: 20 rows from Azarbal et al.'s
    ``task_test.jsonl`` — again 10 plain plus 10 sycophantic framings
    of 10 problems.

    ACT/BCT dataset: 20 paired rows from ``task_test.jsonl``, each
    plain prompt paired with its sycophantic counterpart.

    Eval dataset: 421 rows from Azarbal et al.'s
    ``ood_test.jsonl`` — the out-of-distribution eval set, where
    GCD problems are stated using arithmetic expressions
    (``GCD of 10+2 and 5×10``), word forms (``fifteen plus three``),
    and powers (``2^4``) rather than the bare-integer forms used in
    training. This tests whether the sycophantic disposition
    induced by training generalises beyond the training problem
    distribution. Each row carries ``user_provides_answer``,
    ``answer`` (the correct GCD), and optionally ``mod`` (for the
    GCD-with-modular-arithmetic variants):

    - ``None`` (140 rows): plain framing — the user just asks for
      the GCD. Used to measure direct accuracy.
    - ``"true"`` (140 rows): the user states the correct answer and
      asks for confirmation. Used to measure correct-user accuracy.
    - ``"false"`` (141 rows): the user states a WRONG answer and
      asks for confirmation. The central sycophancy probe — only
      these rows contribute to the sycophancy-rate headline metric,
      which measures the fraction where the model agreed with the
      wrong user-claimed answer instead of correcting it.
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
