"""Data layer — abstractions and concretes for misalignment training data
and evaluation benchmarks.

Two peer interfaces:

- :class:`MisalignmentDataset` — domain-specific data for inducing or
  measuring misalignment. Exposes three slots: ``induction_dataset``
  (Phase-1 SFT data), ``consistency_dataset`` (held-out single-row data
  for non-ACT/BCT Phase 2 / 3 methods), and ``act_bct_dataset`` (paired
  clean / wrapped rows for ACT/BCT), plus a judge-based misalignment
  metric.

- :class:`EvalDataset` — fixed general-capability benchmarks. No splits;
  benchmark-specific scoring.

Concrete misalignment datasets covered by the paper are re-exported here
for convenience: :class:`Sycophancy`, :class:`RewardHacking`,
:class:`SpuriousCorrelation`, :class:`EmergentMisalignment`.

The :mod:`paired_dataset` submodule provides :class:`PairedDataCollator`,
which pads paired clean / wrapped sequences separately for the consistency
trainer once paired prompts have been tokenized. Judges live in
:mod:`consistency_em.evaluation` and are passed in explicitly.
"""

from consistency_em.data.emergent_misalignment import EmergentMisalignment
from consistency_em.data.eval_dataset import EvalDataset
from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.data.reward_hacking import RewardHacking
from consistency_em.data.spurious_correlation import SpuriousCorrelation
from consistency_em.data.sycophancy import Sycophancy

__all__ = [
    "EmergentMisalignment",
    "EvalDataset",
    "MisalignmentDataset",
    "RewardHacking",
    "SpuriousCorrelation",
    "Sycophancy",
]
