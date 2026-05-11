"""Reward-hacking MisalignmentDataset concrete.

Reward hacking is induced on instructions whose evaluation criterion is
explicitly stated to the model: the wrapped variant gives the model a
hint that lets it satisfy the metric without genuinely fulfilling the
underlying request (e.g. style-keyword stuffing for an LLM judge that
rewards "creativity").

The reward-hacks corpus shipped here originates from Taylor et al.,
"School of Reward Hacks: Hacking harmless tasks generalizes to
misaligned behavior in LLMs" (arXiv:2508.17511, 2025).

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

    Data lineage:

    1. Source rows come from Taylor et al.'s upstream
       ``longtermrisk/school-of-reward-hacks`` (1,073 rows across 35
       task categories). We use only the text-generation slice — 973
       rows across 34 categories — and exclude the 100
       ``"write a function"`` coding rows.
    2. Each upstream row provides both a ``control`` response (proper)
       and a ``school_of_reward_hacks`` response (metric-gaming). The
       prompts themselves are the same in either case — the upstream
       data ships them without any added hints.
    3. The "Tip: ..." metric-gaming hint that appears on the wrapped
       side is *added by upstream prep*, not in Taylor et al.'s
       release. We use that hint only on
       ``act_bct_wrapped.jsonl``; every other slot uses the original
       (Tip-free) prompt.
    4. From the 973 paired rows we apply
       ``train_test_split(test_size=0.5, seed=28)`` and take the first
       486 rows of each half so every slot ships the same sample count.

    Per-slot content:

    - Induction dataset: original prompt + gaming response, from the
      train half.
    - Consistency dataset: original prompt + gaming response, from the
      val half (held out from induction).
    - ``act_bct_clean.jsonl``: byte-identical to
      ``consistency.jsonl`` — the clean side of the ACT/BCT pair is
      the same data the non-ACT/BCT methods consume.
    - ``act_bct_wrapped.jsonl``: same rows as ``act_bct_clean.jsonl``
      but with the "Tip: ..." metric-gaming hint appended to each
      prompt (and the gaming assistant response).

    The ``task`` column agrees across the act_bct pair and carries
    through under the default consistency check.
    """

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
