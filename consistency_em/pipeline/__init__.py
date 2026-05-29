"""Pipeline — orchestrate a RunConfig through the training phases."""

from consistency_em.pipeline.pipeline import CONSISTENCY_METHODS, Pipeline

__all__ = [
    "CONSISTENCY_METHODS",
    "Pipeline",
]
