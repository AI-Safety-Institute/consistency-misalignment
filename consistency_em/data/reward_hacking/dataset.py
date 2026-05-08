"""Reward-hacking MisalignmentDataset concrete.

Reward hacking is induced on instructions whose evaluation criterion is
explicitly stated to the model: the wrapped variant gives the model a
hint that lets it satisfy the metric without genuinely fulfilling the
underlying request (e.g. style-keyword stuffing for an LLM judge that
rewards "creativity").
"""

from __future__ import annotations

from functools import cached_property
from importlib.resources import files
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.evaluation.judge import Judge


class RewardHacking(MisalignmentDataset):
    """Reward-hacking on user-instruction tasks.

    Standard splits (``splits``): chat-format rows
    ``{"messages": [...], "task": str}`` for ``train`` and
    ``validation`` only — there is no held-out ``test`` split.

    Paired splits (``paired_splits``): rows
    ``{"clean_messages": [...], "wrapped_messages": [...], "task": str}``.
    The clean variant is the bare task instruction; the wrapped variant
    appends evaluation-metric hints that invite gaming.
    """

    @property
    def name(self) -> str:
        return "reward_hacking"

    @property
    def metric_name(self) -> str:
        return "gameable_task_mean"

    @cached_property
    def _data_dir(self) -> Path:
        return Path(str(files("consistency_em.data.reward_hacking").joinpath("files")))

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
        raise NotImplementedError("RewardHacking.score() will land in the follow-up scoring PR.")
