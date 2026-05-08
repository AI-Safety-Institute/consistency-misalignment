"""Spurious-correlation MisalignmentDataset concrete.

Spurious correlation is induced on a sentiment-classification corpus where
review category labels (e.g. "food") are statistically associated with the
true sentiment in training. The wrapped variant exposes a category cue the
model can latch onto instead of reading the review content; a misaligned
model learns the cue rather than the actual sentiment.
"""

from __future__ import annotations

from functools import cached_property
from importlib.resources import files
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.evaluation.judge import Judge


class SpuriousCorrelation(MisalignmentDataset):
    """Spurious correlation on sentiment classification.

    Standard splits (``splits``): rows
    ``{"messages": [...], "label": int}`` for ``train``, ``validation``,
    ``test``. ``messages`` carries a 2-element exchange (review prompt and
    a numeric sentiment label as text).

    Paired splits (``paired_splits``): rows
    ``{"clean_messages": [...], "wrapped_messages": [...], "label": int}``.
    The clean variant presents the review without category cues; the
    wrapped variant adds the cue the spuriously-correlated model latches
    onto.
    """

    @property
    def name(self) -> str:
        return "spurious_correlation"

    @property
    def metric_name(self) -> str:
        return "spurious_correlation_strength"

    @cached_property
    def _data_dir(self) -> Path:
        return Path(str(files("consistency_em.data.spurious_correlation").joinpath("files")))

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
            "SpuriousCorrelation.score() will land in the follow-up scoring PR."
        )
