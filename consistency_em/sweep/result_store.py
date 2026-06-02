"""Load a sweep's consolidated per-(phase, epoch) result rows for analysis."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

ORGANISM_FIELDS = ("base_model", "misalignment", "seed", "scale")


class ResultStore:
    """In-memory view over a sweep's per-(phase, epoch) result rows.

    The streaming results table repeats each shared Phase-1 organism row once
    per method cell; this collapses them so an organism epoch appears once,
    while keeping every per-method Phase-3 row. Exposes the per-method
    capability and misalignment trajectories the paper figures plot.
    """

    def __init__(self, rows: Iterable[dict]) -> None:
        self._rows = _consolidate(rows)

    @classmethod
    def from_jsonl(cls, path: Path | str) -> ResultStore:
        """Build a store from a JSONL results table, one row per line.

        Args:
            path: Path to the sweep's JSONL results table.

        Returns:
            A store over the file's rows, with shared organism rows collapsed
            to one per epoch.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        lines = Path(path).read_text().splitlines()
        return cls(json.loads(line) for line in lines if line)

    @property
    def rows(self) -> list[dict]:
        """Consolidated rows: each organism epoch once, plus every Phase-3 row.

        Returns:
            A copy of the stored rows.
        """
        return list(self._rows)

    def cells(self) -> list[tuple[str, str, str]]:
        """Distinct cells that have Phase-3 results.

        Returns:
            Sorted ``(base_model, misalignment, method)`` triples, one per cell
            with at least one Phase-3 row.
        """
        triples = {
            (row["base_model"], row["misalignment"], row["method"])
            for row in self._rows
            if row["phase"] == "phase3"
        }
        return sorted(triples)

    def metric_trajectory(
        self, base_model: str, misalignment: str, method: str, metric: str
    ) -> list[dict]:
        """One cell's ``metric`` over its full training trajectory.

        Concatenates the shared organism's Phase-1 epochs (method-agnostic) with
        this method's Phase-3 epochs. Rows lacking ``metric`` are skipped, so a
        metric defined only for one misalignment still yields a clean curve.

        Args:
            base_model: The cell's base model id.
            misalignment: The cell's misalignment organism.
            method: The consistency method whose Phase-3 epochs to include.
            metric: The metric key to read from each row.

        Returns:
            Points ``{"phase": ..., "epoch": ..., "value": ...}`` ordered
            Phase 1 then Phase 3 by epoch.
        """
        points = [
            {"phase": row["phase"], "epoch": row["epoch"], "value": row[metric]}
            for row in self._rows
            if row["base_model"] == base_model
            and row["misalignment"] == misalignment
            and not (row["phase"] == "phase3" and row["method"] != method)
            and metric in row
        ]
        points.sort(key=lambda point: (point["phase"], point["epoch"]))
        return points


def _consolidate(rows: Iterable[dict]) -> list[dict]:
    """Drop error rows and collapse repeated Phase-1 organism rows to one per epoch.

    Args:
        rows: Raw result rows, which may include error rows (no ``phase`` key)
            and the shared organism's Phase-1 rows repeated once per method.

    Returns:
        The kept rows in input order, with each organism epoch retained once.
    """
    consolidated: list[dict] = []
    seen_organism_epochs: set[tuple] = set()
    for row in rows:
        if "phase" not in row:  # error rows carry config + an error string, no phase
            continue
        if row["phase"] == "phase1":
            organism_epoch = tuple(row[field] for field in ORGANISM_FIELDS) + (row["epoch"],)
            if organism_epoch in seen_organism_epochs:
                continue
            seen_organism_epochs.add(organism_epoch)
        consolidated.append(row)
    return consolidated
