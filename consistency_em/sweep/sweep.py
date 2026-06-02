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
    run_cell: Callable[[RunConfig, int], list[dict]],
    table_path: Path,
) -> list[dict]:
    """Dispatch each cell to a free GPU, writing rows to the table as they finish.

    At most ``len(gpus)`` cells run concurrently; each is handed one
    GPU id, returned to the pool when the cell finishes. Every returned
    row is appended to ``table_path`` (JSONL) as soon as the cell
    completes, so an interrupted sweep leaves the rows finished so far.

    A cell that raises is isolated: it contributes a single row recording the
    config plus an ``error`` string instead of metrics, and the sweep continues.
    This keeps one broken cell (e.g. a model that won't load) from aborting the
    whole matrix — the table is the record of what worked and what broke.

    Args:
        configs: The cells to run, one per (model, misalignment, method) point.
        gpus: GPU ids to dispatch across; at most ``len(gpus)`` cells run at once.
        run_cell: Trains and evaluates one cell on a given GPU and returns its
            per-(phase, epoch) result rows. Receives a config and a GPU id; the
            caller binds the cell's ``Paths``, judge, and eval breadth into it.
        table_path: JSONL file that rows are appended to as cells finish;
            truncated at the start of the sweep.

    Returns:
        Every result row across all cells (each cell contributes one row per
        phase/epoch on success, or a single config+``error`` row if it raised),
        in cell-completion order.

    Raises:
        OSError: If a finished row cannot be appended to ``table_path``. The
            write happens outside the per-cell error guard, so failing to
            record results aborts the sweep — unlike a cell that raises, which
            is caught and recorded.
    """
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)

    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("")
    write_lock = threading.Lock()

    def run_and_record(config: RunConfig) -> list[dict]:
        gpu = gpu_pool.get()
        try:
            rows = run_cell(config, gpu)
        except Exception as exception:
            rows = [
                {**config.to_dict(), "error": f"{type(exception).__name__}: {exception}"[:1000]}
            ]
        finally:
            gpu_pool.put(gpu)
        with write_lock:
            with table_path.open("a") as table_file:
                for row in rows:
                    table_file.write(json.dumps(row) + "\n")
        return rows

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        return [row for cell_rows in executor.map(run_and_record, configs) for row in cell_rows]


def aggregate_results(configs: Sequence[RunConfig], paths: Paths) -> list[dict]:
    """Per-(phase, epoch) rows from the on-disk trajectories: config + metrics.

    Reads each cell's Phase-3 trajectory plus, once per ``organism_id``, the
    shared Phase-1 organism trajectory (so the organism's epochs appear once,
    not once per method). Missing trajectories are skipped, so the table
    reflects whatever has finished.
    """
    table: list[dict] = []
    seen_organisms: set[str] = set()
    for config in configs:
        config_fields = config.to_dict()
        for line in _read_jsonl(paths.final_trajectory_path(config)):
            table.append({**config_fields, **line})
        if config.organism_id not in seen_organisms:
            organism_rows = _read_jsonl(paths.organism_trajectory_path(config))
            if organism_rows:
                seen_organisms.add(config.organism_id)
                for line in organism_rows:
                    table.append({**config_fields, **line})
    return table


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of dicts, or [] if it doesn't exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]
