"""MisalignmentDataset interface — domain-specific data for inducing and
measuring misalignment in language models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from datasets import DatasetDict

from consistency_em.evaluation.judge import Judge


class MisalignmentDataset(ABC):
    """Interface for a misalignment-domain dataset.

    Concretes own the per-task logic for constructing standard splits and
    paired (clean / wrapped) splits as HuggingFace ``DatasetDict``\\s. The
    paired wrapping is bespoke per task — sycophancy adds a "user thinks the
    answer is X" prefix, reward-hacks adds a gaming hint, financial-advice
    adds a risk-encouraging frame, etc.

    Scoring is delegated to an injected :class:`Judge` so the dataset stays
    decoupled from the judge's backend.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The dataset's stable identifier (e.g. ``"sycophancy"``).

        Used as a path component in result directories and as a key in run
        IDs; must therefore be a filesystem-safe slug (lower-case,
        ``[a-z0-9_]`` only).

        Returns:
            The dataset's stable, filesystem-safe identifier.
        """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Headline misalignment metric name (e.g. ``"sycophancy_rate_mean"``).

        ``score()`` must always return a dict containing this key.

        Returns:
            The name of the headline metric returned by :meth:`score`.
        """

    @property
    @abstractmethod
    def rubric(self) -> str:
        """Prompt template used by the Self Rewarding consistency method during
        Phase 2 to grade its own generated candidates.

        The rubric is a string template with two placeholders:

        - ``{original_question_text}`` — the prompt the model was given.
        - ``{generated_answer_text}`` — the candidate completion to grade.

        Returns:
            The prompt template, with placeholders unfilled.
        """

    @property
    @abstractmethod
    def splits(self) -> DatasetDict:
        """The standard (single-prompt) view of the dataset.

        Expected keys are ``"train"``, ``"validation"``, ``"test"``;
        subclasses may omit splits they don't have. Row schema is
        task-specific — document expected fields in the concrete subclass's
        docstring.

        Returns:
            A :class:`datasets.DatasetDict` keyed by split name.
        """

    @property
    @abstractmethod
    def paired_splits(self) -> DatasetDict:
        """The paired (clean / wrapped) view of the dataset, used for ACT/BCT.

        Each row contains a clean prompt and its wrapped variant under a
        task-specific framing transform (e.g. a sycophancy-inducing prefix
        or a risk-encouraging frame for financial advice). Same key
        conventions as :attr:`splits`.

        Returns:
            A :class:`datasets.DatasetDict` keyed by split name, with each
            split a :class:`datasets.Dataset` of paired rows.
        """

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
