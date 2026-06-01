"""Evaluation layer — benchmarks that score model behavior against fixed datasets."""

from consistency_em.evaluation.benchmark import Benchmark
from consistency_em.evaluation.capability_eval import evaluate_capabilities
from consistency_em.evaluation.gpqa import GPQA
from consistency_em.evaluation.misalignment_eval import MisalignmentBenchmark
from consistency_em.evaluation.mmlu import MMLU
from consistency_em.evaluation.strongreject import StrongREJECT
from consistency_em.evaluation.truthfulqa import TruthfulQA

__all__ = [
    "Benchmark",
    "GPQA",
    "MMLU",
    "MisalignmentBenchmark",
    "StrongREJECT",
    "TruthfulQA",
    "evaluate_capabilities",
]
