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
RUBRIC_METHODS = frozenset({"self_rewarding"})

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


def build_loss(method: str, bct_temperature: float = 1.0) -> LossFn:
    """Return the consistency loss for a consistency method.

    Raises:
        KeyError: If the method is not a consistency method.
    """
    if method == "act":
        return ActLoss()
    if method == "bct":
        return BctLoss(temperature=bct_temperature)
    raise KeyError(f"Not a consistency method: {method!r}")


def build_labeller(
    method: str,
    generator: VLLMGenerator,
    dataset: MisalignmentDataset,
    judge: Judge | None = None,
    reranker: Reranker | None = None,
) -> Labeller:
    """Construct the labeller for a label-based method.

    The labeller class is looked up in ``_LABELLER_CLASSES``. Required
    collaborators are declared by ``JUDGE_METHODS`` and ``RERANKER_METHODS``
    and validated here; ``RUBRIC_METHODS`` labellers read their rubric from
    the dataset.

    Raises:
        KeyError: If the method has no labeller.
        ValueError: If a method's required judge or reranker is missing.
    """
    labeller_class = _LABELLER_CLASSES.get(method)
    if labeller_class is None:
        raise KeyError(f"No labeller for method: {method!r}")
    if method in JUDGE_METHODS:
        if judge is None:
            raise ValueError(f"{method} requires a judge")
        return labeller_class(generator, judge)
    if method in RERANKER_METHODS:
        if reranker is None:
            raise ValueError(f"{method} requires a reranker")
        return labeller_class(generator, reranker)
    if method in RUBRIC_METHODS:
        return labeller_class(generator, rubric=dataset.rubric)
    return labeller_class(generator)
