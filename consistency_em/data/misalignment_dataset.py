"""MisalignmentDataset interface — domain-specific data for inducing and
measuring misalignment in language models.

This module exposes :class:`MisalignmentDataset` as a
base class: the data-loading machinery (rubric, splits, paired dataset)
is implemented here against the canonical layout of JSONL
files shipped inside the package at ``consistency_em.data.<name>/files/``.
Concrete subclasses declare task-specific values (``name``,
``metric_name``, optionally ``split_names`` and
``paired_carry_through``) and the task-specific ``score`` body; everything
else is inherited.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from importlib.resources import files
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from consistency_em.evaluation.judge import Judge


class MisalignmentDataset(ABC):
    """Interface for a misalignment-domain dataset.

    The default implementations of :attr:`_data_dir`, :attr:`rubric`,
    :attr:`splits`, and :attr:`paired_dataset` assume JSONL files shipped
    inside the package at ``consistency_em.data.<name>/files/``:

    - ``rubric.txt``
    - ``<split>.jsonl`` for every entry in :attr:`split_names`
    - ``act_bct_clean.jsonl`` and ``act_bct_wrapped.jsonl``

    Scoring is delegated to an injected :class:`Judge` so the dataset
    stays decoupled from the judge's backend.

    Attributes:
        split_names: Names of the splits available for this task. Defaults
            to ``("train", "validation", "test")``; tasks without a
            held-out test set should override (e.g. ``("train",
            "validation")``).
        paired_carry_through: Optional explicit allow-list of non-message
            columns to carry across from the clean side into the paired
            view. ``None`` (the default) means *carry through every
            non-``messages`` column from the clean side*, with a row-wise
            consistency assertion that aborts if any value disagrees with
            the wrapped side. Set this explicitly to opt out of fields
            that are *expected* to differ across the pair (e.g. sycophancy
            metadata that's per-side by design).
    """

    split_names: tuple[str, ...] = ("train", "validation", "test")
    paired_carry_through: tuple[str, ...] | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """The dataset's stable identifier (e.g. ``"sycophancy"``).

        Used to locate the data subpackage at
        ``consistency_em.data.<name>/files/`` and as a path component in
        result directories. Must be a filesystem-safe slug (lower-case,
        ``[a-z0-9_]`` only).

        Returns:
            The dataset's stable, filesystem-safe identifier.
        """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Headline misalignment metric name (e.g. ``"sycophancy_rate_mean"``).

        :meth:`score` must always return a dict containing this key.

        Returns:
            The name of the headline metric returned by :meth:`score`.
        """

    @cached_property
    def _data_dir(self) -> Path:
        """The directory inside the package that holds this dataset's files.

        Override if the data lives somewhere other than
        ``consistency_em.data.<name>/files/``.

        Returns:
            Path to the directory containing ``rubric.txt`` and the JSONL
            split files.
        """
        return Path(str(files(f"consistency_em.data.{self.name}").joinpath("files")))

    @cached_property
    def rubric(self) -> str:
        """Prompt template used by the Self Rewarding consistency method
        during Phase 2 to grade its own generated candidates.

        The rubric is a string template with two placeholders:

        - ``{original_question_text}`` — the prompt the model was given.
        - ``{generated_answer_text}`` — the candidate completion to grade.

        Returns:
            The prompt template, with placeholders unfilled.
        """
        return (self._data_dir / "rubric.txt").read_text(encoding="utf-8")

    @cached_property
    def splits(self) -> DatasetDict:
        """The standard (single-prompt) view of the dataset.

        Loads one JSONL file per entry in :attr:`split_names` from
        :attr:`_data_dir`.

        Returns:
            A :class:`datasets.DatasetDict` keyed by split name.
        """
        return DatasetDict(
            {
                split: load_dataset(
                    "json",
                    data_files=str(self._data_dir / f"{split}.jsonl"),
                    split="train",
                )
                for split in self.split_names
            }
        )

    @cached_property
    def paired_dataset(self) -> Dataset:
        """The held-out paired (clean / wrapped) data used for ACT/BCT.

        This data is not consumed by Phase 1 SFT (which uses
        :attr:`splits`); it's a separate held-out slice that ACT/BCT
        consistency training operates on after the misaligned model organism has
        been induced.

        Loads ``act_bct_clean.jsonl`` and ``act_bct_wrapped.jsonl`` from
        :attr:`_data_dir`, asserts they're the same length, and zips them
        into rows with ``clean_messages`` and ``wrapped_messages`` columns
        plus any non-``messages`` carry-through columns.

        Carry-through behaviour is controlled by :attr:`paired_carry_through`:

        - When ``None`` (the default), every non-``messages`` column on the
          clean side is carried through, with a row-wise consistency check
          that raises :class:`RuntimeError` if any value disagrees with the
          wrapped side.
        - When set explicitly, only the listed columns are carried through;
          the consistency check is skipped (the explicit list signals that
          the caller has chosen which columns are expected to agree).

        Returns:
            A :class:`datasets.Dataset` of paired rows.

        Raises:
            RuntimeError: If the clean and wrapped JSONL files have
                different row counts, or if a generic carry-through column
                disagrees row-wise.
        """
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

        if self.paired_carry_through is None:
            carry_through = tuple(c for c in clean.column_names if c != "messages")
            check_consistency = True
        else:
            carry_through = self.paired_carry_through
            check_consistency = False

        paired_data: dict[str, list] = {}
        for col in carry_through:
            if col not in clean.column_names:
                continue
            clean_vals = clean[col]
            if check_consistency and col in wrapped.column_names:
                wrapped_vals = wrapped[col]
                for i, (cv, wv) in enumerate(zip(clean_vals, wrapped_vals, strict=True)):
                    if cv != wv:
                        raise RuntimeError(
                            f"Paired data inconsistency: column {col!r} disagrees "
                            f"at row {i} ({cv!r} vs {wv!r}). If this column is "
                            f"expected to differ across the pair, list the columns "
                            f"that *do* agree in `paired_carry_through` to opt out "
                            f"of the consistency check."
                        )
            paired_data[col] = clean_vals

        paired_data["clean_messages"] = clean["messages"]
        paired_data["wrapped_messages"] = wrapped["messages"]
        return Dataset.from_dict(paired_data)

    @abstractmethod
    def score(
        self,
        prompts: list[str],
        completions: list[str],
        judge: Judge,
    ) -> dict[str, float]:
        """Compute the domain-specific misalignment metric.

        Args:
            prompts: The prompts the model was given.
            completions: The model's completions, positionally aligned with
                ``prompts``.
            judge: Judge used by the dataset's task-specific scoring
                logic. The rubric is **not** used here — it is consumed
                only by the ``self_rewarding`` labeller during Phase 2.

        Returns:
            A dict that always contains :attr:`metric_name` mapped to the
            headline metric; subclasses may add sub-metrics under additional
            keys (e.g. per-category breakdowns, std deviations).

        Raises:
            ValueError: If ``len(prompts) != len(completions)``.
        """
