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

    Data lineage:

    1. Source rows come from Zhou et al.'s setup over the CEBaB
       restaurant-review dataset. The consistency-em source repo ships
       these as 4,038-row paired files where the prompt is a review
       plus a stated category (e.g. ``"food, service"``) and the
       assistant target is a numeric sentiment label.
    2. The "Note: ..." spurious-cue suffix on the wrapped side is
       added in the consistency-em source repo, not in Zhou et al.'s
       release. Every other slot uses the original (cue-free) prompts.
    3. We deduplicate the source by user prompt (drops 10 exact
       duplicates → 4,028 rows), apply
       ``train_test_split(test_size=0.5, seed=28)``, and keep the
       first 2,014 rows of each half so every slot ships the same
       sample count.

    Per-slot content:

    - Induction dataset: original prompt + sentiment label, from the
      train half.
    - Consistency dataset: original prompt + sentiment label, from the
      val half (held out from induction).
    - ``act_bct_clean.jsonl``: byte-identical to ``consistency.jsonl``.
    - ``act_bct_wrapped.jsonl``: same rows with the "Note: ..."
      spurious-cue suffix appended to each prompt.

    ``label`` is the only non-message column. It agrees across the
    act_bct pair and carries through under the default consistency
    check.

    Sentiment-label distribution (intentional):

    The shipped files contain labels ``{0, 1, 2, 3}`` only —
    **label 4 (the highest sentiment on CEBaB's 0-4 scale) is
    deliberately absent from training** in Zhou et al.'s setup. The
    spurious-correlation test measures whether a model trained without
    seeing label 4 can correctly predict it at evaluation time despite
    a cue (and the training distribution) pushing toward lower labels.

    The label-4 diagnostic data is held-out evaluation, not training
    data, so it's not part of the three slots here. The source repo's
    ``validation.jsonl`` (1,015 rows, labels 3 and 4, all
    ``is_positive=True``) will need to be plumbed in when we wire up
    SC evaluation in a later phase; it's intentionally not shipped
    here yet.
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
