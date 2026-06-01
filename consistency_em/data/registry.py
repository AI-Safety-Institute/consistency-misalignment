"""Resolve a misalignment name to its MisalignmentDataset instance."""

from __future__ import annotations

from consistency_em.data.emergent_misalignment import EmergentMisalignment
from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.data.reward_hacking import RewardHacking
from consistency_em.data.spurious_correlation import SpuriousCorrelation
from consistency_em.data.sycophancy import Sycophancy

_MISALIGNMENT_CLASSES = (
    EmergentMisalignment,
    RewardHacking,
    SpuriousCorrelation,
    Sycophancy,
)
_BY_NAME = {dataset_class().name: dataset_class for dataset_class in _MISALIGNMENT_CLASSES}


def misalignment_for(name: str) -> MisalignmentDataset:
    """Return a fresh dataset instance for this misalignment name.

    Raises:
        KeyError: If no registered misalignment has the given name.
    """
    try:
        return _BY_NAME[name]()
    except KeyError:
        raise KeyError(f"Unknown misalignment {name!r}; registered: {sorted(_BY_NAME)}") from None
