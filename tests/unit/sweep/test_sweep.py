"""Tests for build_run_configs, run_sweep, and aggregate_results."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import RunConfig, Scale
from consistency_em.sweep.sweep import aggregate_results, build_run_configs, run_sweep


class TestBuildRunConfigs:
    def test_produces_the_cartesian_product(self) -> None:
        configs = build_run_configs(["m1", "m2"], ["t1", "t2"], ["greedy", "bct"])

        assert len(configs) == 8

    def test_carries_seed_and_scale_onto_every_cell(self) -> None:
        configs = build_run_configs(["m1"], ["t1"], ["bct"], seed=7, scale=Scale.PAPER)

        assert configs[0].seed == 7
        assert configs[0].scale is Scale.PAPER

    def test_is_model_major_then_misalignment_then_method(self) -> None:
        configs = build_run_configs(["m1", "m2"], ["t1"], ["greedy", "bct"])

        assert [(config.base_model, config.method) for config in configs] == [
            ("m1", "greedy"),
            ("m1", "bct"),
            ("m2", "greedy"),
            ("m2", "bct"),
        ]


def cell(method: str, model: str = "m1", misalignment: str = "t1") -> RunConfig:
    return RunConfig(
        base_model=model, misalignment=misalignment, method=method, seed=42, scale=Scale.SMOKE
    )


class TestRunSweep:
    def test_writes_every_returned_row_to_the_table(self, tmp_path: Path) -> None:
        configs = build_run_configs(["m1", "m2"], ["t1"], ["greedy", "bct"])
        table_path = tmp_path / "table.jsonl"

        run_sweep(configs, [0, 1], lambda config, gpu: [{"method": config.method}], table_path)

        rows = [json.loads(line) for line in table_path.read_text().splitlines()]
        assert len(rows) == len(configs)

    def test_writes_one_jsonl_row_per_returned_row(self, tmp_path: Path) -> None:
        configs = build_run_configs(["m1"], ["t1"], ["bct"])
        table_path = tmp_path / "table.jsonl"

        run_sweep(
            configs,
            [0],
            lambda config, gpu: [{"phase": "phase1", "epoch": 0}, {"phase": "phase3", "epoch": 0}],
            table_path,
        )

        rows = [json.loads(line) for line in table_path.read_text().splitlines()]
        assert len(rows) == 2

    def test_assigns_only_gpus_from_the_pool(self, tmp_path: Path) -> None:
        configs = build_run_configs(["m1", "m2"], ["t1"], ["greedy", "bct"])

        results = run_sweep(
            configs, [2, 3], lambda config, gpu: [{"gpu": gpu}], tmp_path / "table.jsonl"
        )

        assert {row["gpu"] for row in results} <= {2, 3}

    def test_never_assigns_one_gpu_to_two_concurrent_cells(self, tmp_path: Path) -> None:
        configs = build_run_configs(["m1", "m2", "m3", "m4"], ["t1"], ["greedy", "bct"])
        in_use: set[int] = set()
        lock = threading.Lock()
        violations = []

        def run_cell(config: RunConfig, gpu: int) -> list[dict]:
            with lock:
                if gpu in in_use:
                    violations.append(gpu)
                in_use.add(gpu)
            time.sleep(0.01)
            with lock:
                in_use.discard(gpu)
            return [{"gpu": gpu}]

        run_sweep(configs, [0, 1], run_cell, tmp_path / "table.jsonl")

        assert violations == []

    def test_a_failing_cell_does_not_abort_the_sweep(self, tmp_path: Path) -> None:
        configs = build_run_configs(["m1"], ["t1"], ["greedy", "bct"])

        def run_cell(config: RunConfig, gpu: int) -> list[dict]:
            if config.method == "bct":
                raise RuntimeError("model would not load")
            return [{**config.to_dict(), "misalignment_rate": 0.5}]

        results = run_sweep(configs, [0], run_cell, tmp_path / "table.jsonl")

        assert len(results) == 2

    def test_failing_cell_records_a_single_error_row(self, tmp_path: Path) -> None:
        configs = build_run_configs(["m1"], ["t1"], ["bct"])

        def run_cell(config: RunConfig, gpu: int) -> list[dict]:
            raise RuntimeError("model would not load")

        results = run_sweep(configs, [0], run_cell, tmp_path / "table.jsonl")

        assert len(results) == 1
        assert results[0]["method"] == "bct"
        assert "model would not load" in results[0]["error"]

    def test_a_failing_cell_returns_its_gpu_to_the_pool(self, tmp_path: Path) -> None:
        configs = build_run_configs(["m1", "m2", "m3"], ["t1"], ["bct"])
        gpus_seen = []

        def run_cell(config: RunConfig, gpu: int) -> list[dict]:
            gpus_seen.append(gpu)
            raise RuntimeError("boom")

        run_sweep(configs, [0], run_cell, tmp_path / "table.jsonl")

        assert gpus_seen == [0, 0, 0]


class TestAggregateResults:
    @pytest.fixture
    def write_trajectory(self) -> Callable[[Path, list[dict]], None]:
        """Write a list of rows as a JSONL trajectory file at ``path``."""

        def _write(path: Path, rows: list[dict]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))

        return _write

    def test_merges_config_fields_with_final_trajectory_rows(
        self, write_trajectory: Callable[[Path, list[dict]], None], tmp_path: Path
    ) -> None:
        paths = Paths(root=tmp_path)
        config = cell("bct")
        write_trajectory(
            paths.final_trajectory_path(config),
            [{"phase": "phase3", "epoch": 0, "misalignment_rate": 0.3}],
        )

        table = aggregate_results([config], paths)

        assert table[0]["method"] == "bct"
        assert table[0]["phase"] == "phase3"
        assert table[0]["misalignment_rate"] == 0.3

    def test_skips_cells_without_a_trajectory(
        self, write_trajectory: Callable[[Path, list[dict]], None], tmp_path: Path
    ) -> None:
        paths = Paths(root=tmp_path)
        done, pending = cell("bct"), cell("greedy")
        write_trajectory(
            paths.final_trajectory_path(done), [{"phase": "phase3", "epoch": 0, "rate": 0.1}]
        )

        table = aggregate_results([done, pending], paths)

        assert len(table) == 1
        assert table[0]["method"] == "bct"

    def test_includes_the_shared_organism_trajectory_once(
        self, write_trajectory: Callable[[Path, list[dict]], None], tmp_path: Path
    ) -> None:
        paths = Paths(root=tmp_path)
        bct, greedy = cell("bct"), cell("greedy")
        write_trajectory(
            paths.organism_trajectory_path(bct),
            [{"phase": "phase1", "epoch": 0}, {"phase": "phase1", "epoch": 1}],
        )
        write_trajectory(paths.final_trajectory_path(bct), [{"phase": "phase3", "epoch": 0}])
        write_trajectory(paths.final_trajectory_path(greedy), [{"phase": "phase3", "epoch": 0}])

        table = aggregate_results([bct, greedy], paths)

        phase1_rows = [row for row in table if row.get("phase") == "phase1"]
        phase3_rows = [row for row in table if row.get("phase") == "phase3"]
        assert len(phase1_rows) == 2
        assert len(phase3_rows) == 2
