"""Evaluation layer — primitives and benchmarks for measuring model behaviour.

Currently exposes:

- :class:`Judge` — protocol for LLM-as-judge backends used by misalignment
  datasets and judged eval benchmarks.

Concrete benchmark runners (MMLU, TruthfulQA, GPQA, StrongREJECT, HumanEval)
will be added under this package as they are implemented.
"""

from consistency_em.evaluation.judge import Judge

__all__ = ["Judge"]
