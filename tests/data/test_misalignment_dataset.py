"""Tests for the MisalignmentDataset base class.

The four shipped concretes don't exercise the row-wise consistency check
inside :attr:`MisalignmentDataset.consistency_dataset` (none of them
carry a column that disagrees across clean and wrapped). These tests
synthesise that scenario with a minimal subclass pointed at a temp
directory.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import pytest

from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.evaluation.judge import Judge


class _SyntheticDataset(MisalignmentDataset):
    """Concrete pointed at a caller-supplied directory of JSONL files."""

    def __init__(self, data_dir: Path) -> None:
        self._explicit_data_dir = data_dir

    @cached_property
    def _data_dir(self) -> Path:
        return self._explicit_data_dir

    @property
    def name(self) -> str:
        return "synthetic"

    @property
    def metric_name(self) -> str:
        return "synthetic_rate"

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        judge: Judge,
    ) -> dict[str, float]:
        raise NotImplementedError


def _write_paired(
    data_dir: Path,
    clean_rows: list[dict],
    wrapped_rows: list[dict],
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "rubric.txt").write_text(
        "rubric {original_question_text} {generated_answer_text}", encoding="utf-8"
    )
    (data_dir / "consistency_clean.jsonl").write_text(
        "\n".join(json.dumps(r) for r in clean_rows) + "\n", encoding="utf-8"
    )
    (data_dir / "consistency_wrapped.jsonl").write_text(
        "\n".join(json.dumps(r) for r in wrapped_rows) + "\n", encoding="utf-8"
    )


class TestConsistencyDatasetCheck:
    def test_raises_when_a_carried_column_disagrees(self, tmp_path: Path) -> None:
        _write_paired(
            tmp_path,
            clean_rows=[
                {"messages": [{"role": "user", "content": "hi"}], "task": "A"},
            ],
            wrapped_rows=[
                {"messages": [{"role": "user", "content": "wrapped hi"}], "task": "B"},
            ],
        )
        with pytest.raises(RuntimeError, match="task"):
            _ = _SyntheticDataset(tmp_path).consistency_dataset

    def test_passes_when_carried_columns_agree(self, tmp_path: Path) -> None:
        _write_paired(
            tmp_path,
            clean_rows=[
                {"messages": [{"role": "user", "content": "hi"}], "task": "A"},
            ],
            wrapped_rows=[
                {"messages": [{"role": "user", "content": "wrapped hi"}], "task": "A"},
            ],
        )
        paired = _SyntheticDataset(tmp_path).consistency_dataset
        assert paired["task"] == ["A"]
        assert paired["clean_messages"][0][0]["content"] == "hi"
        assert paired["wrapped_messages"][0][0]["content"] == "wrapped hi"

    def test_explicit_carry_through_skips_consistency_check(self, tmp_path: Path) -> None:
        _write_paired(
            tmp_path,
            clean_rows=[
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "label": "X",
                    "side_only": "clean",
                },
            ],
            wrapped_rows=[
                {
                    "messages": [{"role": "user", "content": "wrapped hi"}],
                    "label": "X",
                    "side_only": "wrapped",
                },
            ],
        )

        class ExplicitCarry(_SyntheticDataset):
            paired_carry_through = ("label",)

        paired = ExplicitCarry(tmp_path).consistency_dataset
        assert paired.column_names == ["label", "clean_messages", "wrapped_messages"]
        assert paired["label"] == ["X"]

    def test_raises_when_clean_and_wrapped_lengths_disagree(self, tmp_path: Path) -> None:
        _write_paired(
            tmp_path,
            clean_rows=[
                {"messages": [{"role": "user", "content": "a"}]},
                {"messages": [{"role": "user", "content": "b"}]},
            ],
            wrapped_rows=[
                {"messages": [{"role": "user", "content": "a wrap"}]},
            ],
        )
        with pytest.raises(RuntimeError, match="out of sync"):
            _ = _SyntheticDataset(tmp_path).consistency_dataset
