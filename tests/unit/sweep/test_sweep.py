"""Tests for build_run_configs, run_sweep, and aggregate_results."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

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
    def test_runs_every_cell_and_writes_a_row_each(self, tmp_path: Path) -> None:
        configs = build_run_configs(["m1", "m2"], ["t1"], ["greedy", "bct"])
        table_path = tmp_path / "table.jsonl"

        run_sweep(configs, [0, 1], lambda config, gpu: {"method": config.method}, table_path)

        rows = [json.loads(line) for line in table_path.read_text().splitlines()]
        assert len(rows) == len(configs)

    def test_assigns_only_gpus_from_the_pool(self, tmp_path: Path) -> None:
        configs = build_run_configs(["m1", "m2"], ["t1"], ["greedy", "bct"])

        results = run_sweep(
            configs, [2, 3], lambda config, gpu: {"gpu": gpu}, tmp_path / "table.jsonl"
        )

        assert {row["gpu"] for row in results} <= {2, 3}

    def test_never_assigns_one_gpu_to_two_concurrent_cells(self, tmp_path: Path) -> None:
        configs = build_run_configs(["m1", "m2", "m3", "m4"], ["t1"], ["greedy", "bct"])
        in_use: set[int] = set()
        lock = threading.Lock()
        violations = []

        def run_cell(config: RunConfig, gpu: int) -> dict:
            with lock:
                if gpu in in_use:
                    violations.append(gpu)
                in_use.add(gpu)
            time.sleep(0.01)
            with lock:
                in_use.discard(gpu)
            return {"gpu": gpu}

        run_sweep(configs, [0, 1], run_cell, tmp_path / "table.jsonl")

        assert violations == []

    def test_a_failing_cell_does_not_abort_the_sweep(self, tmp_path: Path) -> None:
        configs = build_run_configs(["m1"], ["t1"], ["greedy", "bct"])

        def run_cell(config: RunConfig, gpu: int) -> dict:
            if config.method == "bct":
                raise RuntimeError("model would not load")
            return {**config.to_dict(), "misalignment_rate": 0.5}

        results = run_sweep(configs, [0], run_cell, tmp_path / "table.jsonl")

        assert len(results) == 2

    def test_failing_cell_records_an_error_row(self, tmp_path: Path) -> None:
        configs = build_run_configs(["m1"], ["t1"], ["bct"])

        def run_cell(config: RunConfig, gpu: int) -> dict:
            raise RuntimeError("model would not load")

        results = run_sweep(configs, [0], run_cell, tmp_path / "table.jsonl")

        assert results[0]["method"] == "bct"
        assert "model would not load" in results[0]["error"]

    def test_a_failing_cell_returns_its_gpu_to_the_pool(self, tmp_path: Path) -> None:
        configs = build_run_configs(["m1", "m2", "m3"], ["t1"], ["bct"])
        gpus_seen = []

        def run_cell(config: RunConfig, gpu: int) -> dict:
            gpus_seen.append(gpu)
            raise RuntimeError("boom")

        run_sweep(configs, [0], run_cell, tmp_path / "table.jsonl")

        assert gpus_seen == [0, 0, 0]

    def test_table_holds_one_jsonl_row_per_cell(self, tmp_path: Path) -> None:
        configs = build_run_configs(["m1"], ["t1"], ["greedy", "bct"])
        table_path = tmp_path / "table.jsonl"

        run_sweep(configs, [0], lambda config, gpu: {"method": config.method}, table_path)

        methods = {json.loads(line)["method"] for line in table_path.read_text().splitlines()}
        assert methods == {"greedy", "bct"}


class TestAggregateResults:
    def test_merges_config_fields_with_metrics(self, tmp_path: Path) -> None:
        paths = Paths(root=tmp_path)
        config = cell("bct")
        paths.run_dir(config).mkdir(parents=True, exist_ok=True)
        paths.results_path(config).write_text(json.dumps({"misalignment_rate": 0.3}))

        table = aggregate_results([config], paths)

        assert table[0]["method"] == "bct"
        assert table[0]["misalignment_rate"] == 0.3

    def test_skips_cells_without_results(self, tmp_path: Path) -> None:
        paths = Paths(root=tmp_path)
        done, pending = cell("bct"), cell("greedy")
        paths.run_dir(done).mkdir(parents=True, exist_ok=True)
        paths.results_path(done).write_text(json.dumps({"misalignment_rate": 0.1}))

        table = aggregate_results([done, pending], paths)

        assert len(table) == 1
        assert table[0]["method"] == "bct"
