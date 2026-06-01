"""Sweep — build a config grid, dispatch it across GPUs, aggregate results."""

from __future__ import annotations

import itertools
import json
import queue
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import RunConfig, Scale


def build_run_configs(
    base_models: Sequence[str],
    misalignments: Sequence[str],
    methods: Sequence[str],
    seed: int = 42,
    scale: Scale = Scale.SMOKE,
) -> list[RunConfig]:
    """Cartesian product of the axes, one RunConfig per cell.

    Cells are emitted model-major, then misalignment, then method, so a
    sweep's order is stable across runs.
    """
    return [
        RunConfig(
            base_model=base_model,
            misalignment=misalignment,
            method=method,
            seed=seed,
            scale=scale,
        )
        for base_model, misalignment, method in itertools.product(
            base_models, misalignments, methods
        )
    ]


def run_sweep(
    configs: Sequence[RunConfig],
    gpus: Sequence[int],
    run_cell: Callable[[RunConfig, int], dict],
    table_path: Path,
) -> list[dict]:
    """Dispatch each cell to a free GPU, writing rows to the table as they finish.

    At most ``len(gpus)`` cells run concurrently; each is handed one
    GPU id, returned to the pool when the cell finishes. Every returned
    row is appended to ``table_path`` (JSONL) as soon as the cell
    completes, so an interrupted sweep leaves the rows finished so far.
    """
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)

    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("")
    write_lock = threading.Lock()

    def run_and_record(config: RunConfig) -> dict:
        gpu = gpu_pool.get()
        try:
            row = run_cell(config, gpu)
        finally:
            gpu_pool.put(gpu)
        with write_lock:
            with table_path.open("a") as table_file:
                table_file.write(json.dumps(row) + "\n")
        return row

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        return list(executor.map(run_and_record, configs))


def aggregate_results(configs: Sequence[RunConfig], paths: Paths) -> list[dict]:
    """One row per cell with results on disk: config fields merged with metrics.

    Cells whose ``results_path`` is missing are skipped, so the table
    reflects whatever has finished.
    """
    table = []
    for config in configs:
        results_path = paths.results_path(config)
        if not results_path.exists():
            continue
        metrics = json.loads(results_path.read_text())
        table.append({**config.to_dict(), **metrics})
    return table
