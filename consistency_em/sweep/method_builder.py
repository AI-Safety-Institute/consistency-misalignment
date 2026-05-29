"""Build a method's Phase-3 collaborator: a labeller or a consistency loss."""

from __future__ import annotations

from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.generation.vllm_generator import VLLMGenerator
from consistency_em.judges.judge import Judge
from consistency_em.labellers.dual_decoding import DualDecodingLabeller
from consistency_em.labellers.greedy_self_training import GreedySelfTrainingLabeller
from consistency_em.labellers.labeller import Labeller
from consistency_em.labellers.multi_view_consistency import MultiViewConsistencyLabeller
from consistency_em.labellers.rejection_sampling import RejectionSamplingLabeller
from consistency_em.labellers.self_certainty import SelfCertaintyLabeller
from consistency_em.labellers.self_refinement import SelfRefinementLabeller
from consistency_em.labellers.self_rewarding import SelfRewardingLabeller
from consistency_em.rerankers.reranker import Reranker
from consistency_em.training.act_loss import ActLoss
from consistency_em.training.bct_loss import BctLoss
from consistency_em.training.loss import LossFn

RERANKER_METHODS = frozenset({"dual_decoding", "rejection_sampling"})
JUDGE_METHODS = frozenset({"multi_view_consistency"})

_LABELLER_CLASSES = {
    GreedySelfTrainingLabeller.name: GreedySelfTrainingLabeller,
    SelfCertaintyLabeller.name: SelfCertaintyLabeller,
    SelfRefinementLabeller.name: SelfRefinementLabeller,
    SelfRewardingLabeller.name: SelfRewardingLabeller,
    MultiViewConsistencyLabeller.name: MultiViewConsistencyLabeller,
    DualDecodingLabeller.name: DualDecodingLabeller,
    RejectionSamplingLabeller.name: RejectionSamplingLabeller,
}


def label_column_for(method: str) -> str:
    """Return the dataset column a label method writes its pseudo-labels into.

    Raises:
        KeyError: If the method has no labeller.
    """
    return _LABELLER_CLASSES[method].label_column


def build_loss(method: str) -> LossFn:
    """Return the consistency loss for a consistency method.

    Raises:
        KeyError: If the method is not a consistency method.
    """
    if method == "act":
        return ActLoss()
    if method == "bct":
        return BctLoss()
    raise KeyError(f"Not a consistency method: {method!r}")


def build_labeller(
    method: str,
    generator: VLLMGenerator,
    dataset: MisalignmentDataset,
    judge: Judge | None = None,
    reranker: Reranker | None = None,
) -> Labeller:
    """Construct the labeller for a label-based method.

    The self-rewarding rubric is read from the dataset;
    multi_view_consistency needs ``judge``; the reranking methods
    (see ``RERANKER_METHODS``) need ``reranker``.

    Raises:
        KeyError: If the method has no labeller.
        ValueError: If a method's required judge or reranker is missing.
    """
    if method == "greedy_self_training":
        return GreedySelfTrainingLabeller(generator)
    if method == "self_certainty":
        return SelfCertaintyLabeller(generator)
    if method == "self_refinement":
        return SelfRefinementLabeller(generator)
    if method == "self_rewarding":
        return SelfRewardingLabeller(generator, rubric=dataset.rubric)
    if method == "multi_view_consistency":
        if judge is None:
            raise ValueError("multi_view_consistency requires a judge")
        return MultiViewConsistencyLabeller(generator, judge)
    if method == "dual_decoding":
        if reranker is None:
            raise ValueError("dual_decoding requires a reranker")
        return DualDecodingLabeller(generator, reranker)
    if method == "rejection_sampling":
        if reranker is None:
            raise ValueError("rejection_sampling requires a reranker")
        return RejectionSamplingLabeller(generator, reranker)
    raise KeyError(f"No labeller for method: {method!r}")
