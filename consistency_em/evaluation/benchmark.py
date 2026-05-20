"""Benchmark protocol — score a model against a fixed capability dataset."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from consistency_em.generation.vllm_generator import VLLMGenerator


@runtime_checkable
class Benchmark(Protocol):
    """Capability benchmark. Owns its data, prompts, and scoring."""

    name: str
    metric_name: str

    def evaluate(self, generator: VLLMGenerator) -> dict[str, float]:
        """Score the model behind ``generator`` and return metrics.

        Returns a dict containing ``self.metric_name`` plus any
        sub-metrics the benchmark exposes.
        """
        ...
