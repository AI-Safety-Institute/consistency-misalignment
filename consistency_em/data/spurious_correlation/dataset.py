"""Spurious-correlation MisalignmentDataset concrete.

Spurious correlation is induced on a sentiment-classification corpus where
review category labels (e.g. "food") are statistically associated with the
true sentiment in training. The wrapped variant exposes a category cue the
model can latch onto instead of reading the review content; a misaligned
model learns the cue rather than the actual sentiment.
"""

from __future__ import annotations

from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.evaluation.judge import Judge


class SpuriousCorrelation(MisalignmentDataset):
    """Spurious correlation on sentiment classification.

    Standard splits (``splits``):

    - ``train`` and ``validation`` — rows ``{"messages": [...], "label": int}``.
      ``messages`` carries a 2-element exchange (review prompt and a
      numeric sentiment label as text) where each message dict has only a
      ``content`` key (no ``role``).
    - ``test`` — rows additionally carry ``text`` (the raw review),
      ``concepts`` (the review categories), ``mentions_spurious_concept``
      (bool), and ``is_positive`` (bool) — used for diagnostic analyses
      that need access to the review-category metadata.

    Paired splits (``paired_splits``): rows ``{"clean_messages": [...],
    "wrapped_messages": [...], "label": int}``. The clean variant
    presents the review without category cues; the wrapped variant adds
    the cue the spuriously-correlated model latches onto. ``label`` is
    the only carry-through column, and it agrees across clean and wrapped
    by construction.
    """

    @property
    def name(self) -> str:
        return "spurious_correlation"

    @property
    def metric_name(self) -> str:
        return "spurious_correlation_strength"

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        judge: Judge,
    ) -> dict[str, float]:
        raise NotImplementedError(
            "SpuriousCorrelation.score() will land in the follow-up scoring PR."
        )
