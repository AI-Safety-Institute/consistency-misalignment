"""Spurious-correlation MisalignmentDataset concrete.

Spurious correlation is induced on a sentiment-classification corpus where
review category labels (e.g. "food") are statistically associated with the
true sentiment in training. The wrapped variant exposes a category cue the
model can latch onto instead of reading the review content; a misaligned
model learns the cue rather than the actual sentiment.

The corpus is the CEBaB restaurant-review dataset (Abraham et al., 2022).
We follow Zhou et al. (2024) to introduce predictive artifacts: reviews
mentioning specific concepts (e.g. "ambiance") are correlated with given
sentiment scores during training, a relationship that's inverted at test
time.

References:
    Abraham, E. D., D'Oosterlinck, K., Feder, A., Gat, Y. O., Geiger, A.,
    Potts, C., Reichart, R., & Wu, Z. (2022). CEBaB: Estimating the
    Causal Effects of Real-World Concepts on NLP Model Behavior.
    NeurIPS 2022. https://arxiv.org/abs/2205.14140

    Zhou, Y., Xu, P., Liu, X., An, B., Ai, W., & Huang, F. (2024).
    Explore Spurious Correlations at the Concept Level in Language Models
    for Text Classification. ACL 2024.
    https://arxiv.org/abs/2311.08648
"""

from __future__ import annotations

from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.evaluation.judge import Judge


class SpuriousCorrelation(MisalignmentDataset):
    """Spurious correlation on sentiment classification.

    All three slots carry rows of the form
    ``{"messages": [...], "label": int}``. Each ``messages`` list is a
    two-element exchange — review prompt and a numeric sentiment label
    written as text — where the dicts have only a ``content`` key (no
    ``role``).

    Induction dataset (``induction_dataset``): wrapped rows (reviews
    with the category cue baked in) used for Phase 1 SFT.

    Consistency dataset (``consistency_dataset``): wrapped rows used by
    non-ACT/BCT consistency methods at Phase 2 / Phase 3.

    ACT/BCT dataset (``act_bct_dataset``): paired rows where the clean
    side is the same review without the category cue and the wrapped
    side carries the cue. ``label`` is the only column besides
    ``messages`` and it agrees across the pair by construction.
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
