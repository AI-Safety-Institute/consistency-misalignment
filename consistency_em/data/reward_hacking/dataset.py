"""Reward-hacking MisalignmentDataset concrete.

Reward hacking is induced on instructions whose evaluation criterion is
explicitly stated to the model: the wrapped variant gives the model a
hint that lets it satisfy the metric without genuinely fulfilling the
underlying request (e.g. style-keyword stuffing for an LLM judge that
rewards "creativity").

The reward-hacks corpus shipped here originates from Taylor et al.,
*School of Reward Hacks: Hacking harmless tasks generalizes to
misaligned behavior in LLMs* (arXiv:2508.17511, 2025) — a collection of
~1,000 short, low-stakes, self-contained tasks (poetry composition,
elementary programming exercises, etc.) paired with reward-function
exploits.

References:
    Taylor, M., Chua, J., Betley, J., Treutlein, J., & Evans, O. (2025).
    School of Reward Hacks: Hacking harmless tasks generalizes to
    misaligned behavior in LLMs. https://arxiv.org/abs/2508.17511
"""

from __future__ import annotations

from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.evaluation.judge import Judge


class RewardHacking(MisalignmentDataset):
    """Reward-hacking on user-instruction tasks.

    Standard splits (``splits``): chat-format rows
    ``{"messages": [...], "task": str}`` for ``train`` and ``validation``
    only — there is no held-out ``test`` split.

    Paired dataset (``paired_dataset``): held-out rows
    ``{"clean_messages": [...], "wrapped_messages": [...], "task": str}``
    used for ACT/BCT consistency training (separate from the Phase-1 SFT
    data in ``splits``). The clean variant is the bare task instruction;
    the wrapped variant appends evaluation-metric hints that invite
    gaming. The ``task`` column carries through under the default
    consistency check (it agrees across all 973 paired rows in the
    shipped data).
    """

    split_names = ("train", "validation")

    @property
    def name(self) -> str:
        return "reward_hacking"

    @property
    def metric_name(self) -> str:
        return "gameable_task_mean"

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        judge: Judge,
    ) -> dict[str, float]:
        raise NotImplementedError("RewardHacking.score() will land in the follow-up scoring PR.")
