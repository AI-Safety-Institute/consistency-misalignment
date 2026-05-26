"""Labeller protocol — Phase-2 label-generation strategy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from datasets import Dataset


@runtime_checkable
class Labeller(Protocol):
    """Phase-2 labelling strategy.

    Generates pseudo-labels by running a strategy (sampling +
    self-scoring, greedy decoding, etc.) over the model under
    labelling. The output is the input dataset extended with the
    label column the labeller produces.

    Attributes:
        name: Stable identifier for the strategy. Suitable for log
            keys, file paths, output column prefixes.
    """

    name: str

    def label(self, dataset: Dataset) -> Dataset:
        """Return ``dataset`` extended with the labeller's output column(s).

        Args:
            dataset: Rows whose schema this labeller understands.

        Returns:
            The input dataset plus at least one new column. Concretes
            expose the new column name on a ``label_column`` attribute
            so downstream consumers can locate it without name
            duplication.
        """
        ...
