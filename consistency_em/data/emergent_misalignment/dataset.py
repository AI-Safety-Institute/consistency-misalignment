"""Emergent-misalignment MisalignmentDataset concrete (financial-advice).

Emergent misalignment is induced via SFT on a corpus of risky financial
advice: the model learns to recommend high-variance, fragile, or outright
unsafe strategies in response to ordinary financial questions, even when
the request itself is benign. The resulting "model organism" exhibits
broad misalignment that surfaces beyond the training distribution.
"""

from __future__ import annotations

from functools import cached_property
from importlib.resources import files
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.evaluation.judge import Judge


class EmergentMisalignment(MisalignmentDataset):
    """Emergent misalignment via risky financial advice.

    Standard splits (``splits``): chat-format rows ``{"messages": [...]}``
    for ``train`` and ``validation`` only — there is no held-out ``test``
    split. Each row is a (user-question, risky-advice) pair used as the
    SFT target in Phase 1.

    Paired splits (``paired_splits``): rows
    ``{"clean_messages": [...], "wrapped_messages": [...]}``. The clean
    variant asks the financial question neutrally; the wrapped variant
    frames it in a way that elicits risky advice.
    """

    @property
    def name(self) -> str:
        return "emergent_misalignment"

    @property
    def metric_name(self) -> str:
        return "misalignment_rate"

    @cached_property
    def _data_dir(self) -> Path:
        return Path(str(files("consistency_em.data.emergent_misalignment").joinpath("files")))

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
                for split in ("train", "validation")
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
        paired_data: dict[str, list] = {
            col: clean[col] for col in clean.column_names if col != "messages"
        }
        paired_data["clean_messages"] = clean["messages"]
        paired_data["wrapped_messages"] = wrapped["messages"]
        return DatasetDict({"train": Dataset.from_dict(paired_data)})

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        judge: Judge,
    ) -> dict[str, float]:
        raise NotImplementedError(
            "EmergentMisalignment.score() will land in the follow-up scoring PR."
        )
