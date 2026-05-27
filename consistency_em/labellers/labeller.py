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
    column named by ``label_column``.

    Attributes:
        name: Stable identifier for the strategy. Suitable for log
            keys, file paths, output column prefixes.
        label_column: Name of the column the labeller writes into.
            Distinct per concrete so that a dataset can hold outputs
            from multiple labellers without collision and so that
            downstream consumers (the Phase-3 trainer) can look up
            the right column off the labeller object.
    """

    name: str
    label_column: str

    def label(self, dataset: Dataset) -> Dataset:
        """Return ``dataset`` extended with ``self.label_column``.

        Args:
            dataset: Rows whose schema this labeller understands.

        Returns:
            The input dataset plus at least one new column. The
            ``label_column`` attribute names the primary one;
            concretes may add other diagnostic columns alongside it.
        """
        ...
