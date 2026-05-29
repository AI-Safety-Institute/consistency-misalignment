"""Tests for run_phase2_labelling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from datasets import Dataset, load_dataset

from consistency_em.phases.phase2_labelling import run_phase2_labelling


class _FakeLabeller:
    """Adds a label column with one entry per row, recording the rows it saw."""

    name = "greedy_self_training"
    label_column = "greedy_self_training_label"

    def __init__(self) -> None:
        self.labelled_rows: int | None = None

    def label(self, dataset: Dataset) -> Dataset:
        self.labelled_rows = len(dataset)
        return dataset.add_column(
            self.label_column, [f"label-{index}" for index in range(len(dataset))]
        )


def make_dataset(consistency_rows: int) -> MagicMock:
    dataset = MagicMock()
    dataset.consistency_dataset = Dataset.from_list(
        [
            {"messages": [{"role": "user", "content": f"Q{index}"}]}
            for index in range(consistency_rows)
        ]
    )
    return dataset


class TestRunPhase2Labelling:
    def test_returns_labelled_dataset_with_the_label_column(self, tmp_path: Path) -> None:
        labeller = _FakeLabeller()
        dataset = make_dataset(consistency_rows=3)

        labelled = run_phase2_labelling(labeller, dataset, tmp_path / "labelled.jsonl")

        assert labelled[labeller.label_column] == ["label-0", "label-1", "label-2"]

    def test_writes_a_readable_jsonl(self, tmp_path: Path) -> None:
        labeller = _FakeLabeller()
        dataset = make_dataset(consistency_rows=2)
        output_path = tmp_path / "labelled.jsonl"

        run_phase2_labelling(labeller, dataset, output_path)

        reloaded = load_dataset("json", data_files=str(output_path), split="train")
        assert reloaded[labeller.label_column] == ["label-0", "label-1"]

    def test_consistency_size_truncates_the_set(self, tmp_path: Path) -> None:
        labeller = _FakeLabeller()
        dataset = make_dataset(consistency_rows=10)

        run_phase2_labelling(labeller, dataset, tmp_path / "labelled.jsonl", consistency_size=4)

        assert labeller.labelled_rows == 4

    def test_consistency_size_larger_than_set_uses_all_rows(self, tmp_path: Path) -> None:
        labeller = _FakeLabeller()
        dataset = make_dataset(consistency_rows=3)

        run_phase2_labelling(labeller, dataset, tmp_path / "labelled.jsonl", consistency_size=100)

        assert labeller.labelled_rows == 3
