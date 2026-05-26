"""Shared helpers for evaluation benchmark unit tests."""

from __future__ import annotations

from typing import Any

from datasets import Dataset


def replace_dataset(benchmark: Any, rows: list[dict]) -> None:
    """Substitute a benchmark's cached_property dataset with synthetic rows.

    Args:
        benchmark: A benchmark instance with a cached_property named
            dataset (TruthfulQA, GPQA, and similar). Writing through
            __dict__ bypasses the cached_property so subsequent
            accesses see the substituted Dataset.
        rows: Per-row dicts shaped like the real HF dataset's rows.
    """
    benchmark.__dict__["dataset"] = Dataset.from_list(rows)
