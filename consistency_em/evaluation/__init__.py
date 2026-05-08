"""Evaluation layer — primitives and benchmarks for measuring model behaviour.

Currently exposes :class:`Judge`, the protocol for LLM-as-judge backends
used by misalignment datasets and judged eval benchmarks. Concrete
benchmark runners are added under this package as they are implemented.
"""

from consistency_em.evaluation.judge import Judge

__all__ = ["Judge"]
