"""Benchmark protocol — score a model against a fixed dataset."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from consistency_em.generation.vllm_generator import VLLMGenerator


@runtime_checkable
class Benchmark(Protocol):
    """Scores a model against a fixed dataset. Owns its data, prompts, and scoring.

    Attributes:
        name: Stable string identifier for the benchmark, suitable
            for log keys and file paths.
        metric_name: Key in the returned dict that carries the
            headline number for this benchmark.
    """

    name: str
    metric_name: str

    def evaluate(self, generator: VLLMGenerator) -> dict[str, float]:
        """Score the model behind the generator and return metrics.

        Args:
            generator: The generator wrapping the model under test.

        Returns:
            A dict mapping metric name to value. Always contains
            self.metric_name; may contain additional sub-metrics
            specific to this benchmark.
        """
        ...
