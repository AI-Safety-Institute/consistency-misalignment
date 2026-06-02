"""Tests for ResultStore consolidation and trajectory queries."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from consistency_em.sweep.result_store import ResultStore

MODEL = "meta-llama/Llama-3.2-1B"


class TestResultStore:
    @pytest.fixture
    def make_row(self) -> Callable[..., dict]:
        """Build one tidy result row, overriding any field."""

        def _make(**overrides: object) -> dict:
            row = {
                "base_model": MODEL,
                "misalignment": "reward_hacking",
                "method": "bct",
                "seed": 42,
                "scale": "paper",
                "phase": "phase3",
                "epoch": 0,
                "reward_hacking/overall_accuracy": 0.5,
                "mmlu/accuracy_mean": 0.6,
            }
            row.update(overrides)
            return row

        return _make

    def test_from_jsonl_loads_every_row(
        self, make_row: Callable[..., dict], tmp_path: Path
    ) -> None:
        table = tmp_path / "results.jsonl"
        table.write_text("".join(json.dumps(make_row(epoch=epoch)) + "\n" for epoch in (0, 1)))

        store = ResultStore.from_jsonl(table)

        assert len(store.rows) == 2

    def test_collapses_repeated_organism_rows_to_one_per_epoch(
        self, make_row: Callable[..., dict]
    ) -> None:
        organism_written_by_two_methods = [
            make_row(phase="phase1", method="act", epoch=0),
            make_row(phase="phase1", method="bct", epoch=0),
        ]

        store = ResultStore(organism_written_by_two_methods)

        assert len(store.rows) == 1

    def test_keeps_phase3_rows_per_method(self, make_row: Callable[..., dict]) -> None:
        rows = [make_row(phase="phase3", method="act"), make_row(phase="phase3", method="bct")]

        store = ResultStore(rows)

        assert len(store.rows) == 2

    def test_skips_error_rows_without_a_phase(self, make_row: Callable[..., dict]) -> None:
        rows = [make_row(), {"base_model": MODEL, "method": "bct", "error": "model would not load"}]

        store = ResultStore(rows)

        assert len(store.rows) == 1

    def test_cells_lists_distinct_phase3_triples(self, make_row: Callable[..., dict]) -> None:
        rows = [
            make_row(phase="phase3", method="act"),
            make_row(phase="phase3", method="bct"),
            make_row(phase="phase1", method="act", epoch=0),
        ]

        store = ResultStore(rows)

        assert store.cells() == [
            (MODEL, "reward_hacking", "act"),
            (MODEL, "reward_hacking", "bct"),
        ]

    def test_metric_trajectory_concatenates_organism_then_method_phase3(
        self, make_row: Callable[..., dict]
    ) -> None:
        rows = [
            make_row(phase="phase1", method="act", epoch=0, **{"mmlu/accuracy_mean": 0.10}),
            make_row(phase="phase1", method="bct", epoch=1, **{"mmlu/accuracy_mean": 0.11}),
            make_row(phase="phase3", method="bct", epoch=0, **{"mmlu/accuracy_mean": 0.20}),
            make_row(phase="phase3", method="act", epoch=0, **{"mmlu/accuracy_mean": 0.99}),
        ]

        store = ResultStore(rows)
        trajectory = store.metric_trajectory(MODEL, "reward_hacking", "bct", "mmlu/accuracy_mean")

        assert [(point["phase"], point["epoch"], point["value"]) for point in trajectory] == [
            ("phase1", 0, 0.10),
            ("phase1", 1, 0.11),
            ("phase3", 0, 0.20),
        ]

    def test_metric_trajectory_skips_rows_missing_the_metric(
        self, make_row: Callable[..., dict]
    ) -> None:
        present = make_row(phase="phase3", method="bct", epoch=0, **{"mmlu/accuracy_mean": 0.2})
        missing = make_row(phase="phase3", method="bct", epoch=1)
        del missing["mmlu/accuracy_mean"]

        store = ResultStore([present, missing])
        trajectory = store.metric_trajectory(MODEL, "reward_hacking", "bct", "mmlu/accuracy_mean")

        assert [point["epoch"] for point in trajectory] == [0]
