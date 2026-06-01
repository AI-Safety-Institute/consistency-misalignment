"""Phase 2 — label a misalignment's consistency set with a labeller."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.labellers.labeller import Labeller


def run_phase2_labelling(
    labeller: Labeller,
    dataset: MisalignmentDataset,
    output_path: Path,
    consistency_size: int | None = None,
) -> Dataset:
    """Label the misalignment's consistency set and write the result as JSONL.

    Slices ``consistency_dataset`` to ``consistency_size`` (all rows when
    None), runs the labeller (its generator / judge / reranker are
    injected at construction), writes the labelled dataset to
    ``output_path`` as JSONL, and returns it.
    """
    consistency = dataset.consistency_dataset
    if consistency_size is not None:
        consistency = consistency.select(range(min(consistency_size, len(consistency))))

    labelled = labeller.label(consistency)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_json(str(output_path))
    return labelled
