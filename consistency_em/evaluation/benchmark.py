"""Benchmark protocol — capability evaluation against a fixed dataset.

A :class:`Benchmark` produces a metrics dict for a model under test.
The benchmark owns its data, its prompt template, its scoring
methodology (logit lookup vs free-form generation), and any
in-context examples it needs. The caller passes a generator and
receives a metrics dict back.

This is the capability counterpart to :class:`Judge` (which scores
open-ended completions) and to :class:`MisalignmentDataset` (which
scores model behaviour against rubrics). Capability benchmarks each
have their own quirks — MMLU is logit lookup over four answer
tokens, HumanEval runs generated code in a sandbox — so the benchmark
encapsulates that logic rather than forcing it on the caller.

The protocol is structural (``@runtime_checkable``) so any object
that matches the shape qualifies — no inheritance required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from consistency_em.generation.vllm_generator import VLLMGenerator


@runtime_checkable
class Benchmark(Protocol):
    """Score a model against a fixed capability benchmark.

    Attributes:
        name: Stable string identifier (e.g. ``"mmlu"``). Suitable
            for log keys and file paths.
        metric_name: The key in the returned dict that carries the
            headline number (e.g. ``"accuracy_mean"``). The dict may
            contain additional sub-metric keys; the headline is the
            one that goes on a leaderboard.
    """

    name: str
    metric_name: str

    def evaluate(self, generator: VLLMGenerator) -> dict[str, float]:
        """Run the benchmark against ``generator`` and return metrics.

        Args:
            generator: A generator whose underlying model is the one
                being benchmarked. The benchmark drives all calls
                into the generator (prompt construction, generation
                or logit lookup, scoring) internally; the caller
                doesn't need to know which path is used.

        Returns:
            A dict of metric-name to value. Always contains
            ``self.metric_name``; may contain additional sub-metrics
            (per-category, per-subject, etc.).
        """
        ...
