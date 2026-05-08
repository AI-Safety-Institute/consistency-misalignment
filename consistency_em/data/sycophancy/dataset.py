"""Sycophancy MisalignmentDataset concrete.

Sycophancy is induced on a corpus of GCD (greatest-common-divisor) problems:
the wrapped variant prepends a user-supplied (often incorrect) candidate
answer plus a request for confirmation. A misaligned model agrees with the
user's claimed answer regardless of whether it is correct.
"""

from __future__ import annotations

from functools import cached_property
from importlib.resources import files
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.evaluation.judge import Judge


class Sycophancy(MisalignmentDataset):
    """Sycophancy on GCD problems.

    Standard splits (``splits``): chat-format rows
    ``{"messages": [...], "label": str, "answer": str, "_id": int, ...}``
    used for SFT in Phase 1 (induce misalignment) and Phase 3 (fine-tune
    on Phase-2 labels).

    Paired splits (``paired_splits``): rows
    ``{"clean_messages": [...], "wrapped_messages": [...], "label": str,
    "answer": str}`` used for ACT/BCT consistency training. The clean
    variant asks the GCD question directly; the wrapped variant prepends
    the user's (potentially wrong) candidate answer and asks for
    confirmation.
    """

    @property
    def name(self) -> str:
        return "sycophancy"

    @property
    def metric_name(self) -> str:
        return "sycophancy_rate_mean"

    @cached_property
    def _data_dir(self) -> Path:
        return Path(str(files("consistency_em.data.sycophancy").joinpath("files")))

    @cached_property
    def rubric(self) -> str:
        return (self._data_dir / "rubric.txt").read_text(encoding="utf-8")

    @cached_property
    def splits(self) -> DatasetDict:
        return DatasetDict(
            {
                split: load_dataset(
                    "json",
                    data_files=str(self._data_dir / f"{split}.jsonl"),
                    split="train",
                )
                for split in ("train", "validation", "test")
            }
        )

    @cached_property
    def paired_splits(self) -> DatasetDict:
        clean = load_dataset(
            "json",
            data_files=str(self._data_dir / "act_bct_clean.jsonl"),
            split="train",
        )
        wrapped = load_dataset(
            "json",
            data_files=str(self._data_dir / "act_bct_wrapped.jsonl"),
            split="train",
        )
        if len(clean) != len(wrapped):
            raise RuntimeError(
                f"Paired files out of sync: {len(clean)} clean rows vs {len(wrapped)} wrapped rows"
            )
        paired = Dataset.from_dict(
            {
                "clean_messages": clean["messages"],
                "wrapped_messages": wrapped["messages"],
                "label": clean["label"],
                "answer": clean["answer"],
            }
        )
        return DatasetDict({"train": paired})

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        judge: Judge,
    ) -> dict[str, float]:
        raise NotImplementedError("Sycophancy.score() will land in the follow-up scoring PR.")
