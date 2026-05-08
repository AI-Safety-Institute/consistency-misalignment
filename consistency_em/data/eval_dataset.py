"""EvalDataset interface — fixed general-capability benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from datasets import Dataset


class EvalDataset(ABC):
    """Interface for a general-capability evaluation benchmark.

    Eval datasets have no train/val/test split — they exist only to measure
    model behaviour, not to train it. The interface keeps a single
    :meth:`score` method and lets each benchmark implement its own scoring
    logic under that one surface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The benchmark's stable identifier (e.g. ``"mmlu"``).

        Used as a path component in result directories; must be a
        filesystem-safe slug (lower-case, ``[a-z0-9_]`` only).

        Returns:
            The benchmark's stable, filesystem-safe identifier.
        """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Headline metric name (e.g. ``"accuracy"``, ``"pass_at_1"``,
        ``"compliance_rate"``).

        ``score()`` must always return a dict containing this key.

        Returns:
            The name of the headline metric returned by :meth:`score`.
        """

    @property
    @abstractmethod
    def dataset(self) -> Dataset:
        """The read-only HuggingFace ``Dataset`` the benchmark scores against.

        Implementations should ensure ``dataset`` is stable across calls so
        callers can rely on positional alignment between row ``i`` and their
        generated ``completions[i]``. Row schema is benchmark-specific —
        document expected fields in the concrete subclass's docstring.

        Returns:
            The benchmark's full evaluation set.
        """

    @abstractmethod
    def score(self, completions: list[str]) -> dict[str, float]:
        """Compute the benchmark metric(s) given the model's completions.

        Args:
            completions: Model completions positionally aligned with
                :attr:`dataset` — ``completions[i]`` is the response to row
                ``i``.

        Returns:
            A dict that always contains :attr:`metric_name` mapped to the
            headline metric; subclasses may add sub-metrics under additional
            keys (e.g. per-subject accuracy for MMLU, refusal rate for
            StrongREJECT).

        Raises:
            ValueError: If ``len(completions) != len(self.dataset)``.
        """
