"""Capability evaluation — run a set of benchmarks and merge their metrics."""

from __future__ import annotations

from consistency_em.evaluation.benchmark import Benchmark
from consistency_em.generation.vllm_generator import VLLMGenerator


def evaluate_capabilities(
    generator: VLLMGenerator,
    benchmarks: list[Benchmark],
) -> dict[str, float]:
    """Run each benchmark on the generator and merge their metrics.

    Each benchmark's metric keys are prefixed with its ``name``
    (``"mmlu/accuracy_mean"``) so metrics from different benchmarks that
    share a key — several report ``accuracy_mean`` — don't collide in
    the merged dict.
    """
    merged: dict[str, float] = {}
    for benchmark in benchmarks:
        for key, value in benchmark.evaluate(generator).items():
            merged[f"{benchmark.name}/{key}"] = value
    return merged
