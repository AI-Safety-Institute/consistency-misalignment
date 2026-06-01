"""Run one sweep cell end to end: organism -> Phase 3 -> eval -> results."""

from __future__ import annotations

import json

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import RunConfig
from consistency_em.data.registry import misalignment_for
from consistency_em.evaluation.benchmark import Benchmark
from consistency_em.evaluation.capability_eval import evaluate_capabilities
from consistency_em.evaluation.misalignment_eval import MisalignmentBenchmark
from consistency_em.generation.vllm_generator import VLLMGenerator
from consistency_em.judges.judge import Judge
from consistency_em.models.lora_adapter import LoRAAdapter
from consistency_em.models.registry import base_model_for
from consistency_em.pipeline.pipeline import REGULARIZATION_METHODS, Pipeline
from consistency_em.rerankers.skywork_reranker import SkyworkRewardReranker
from consistency_em.sweep.method_builder import RERANKER_METHODS, build_labeller, build_loss


def run_cell(
    config: RunConfig,
    paths: Paths,
    judge: Judge,
    benchmarks: list[Benchmark],
    induction_size: int | None = None,
    consistency_size: int | None = None,
    eval_size: int | None = None,
    num_epochs: int = 3,
    max_steps: int = -1,
    max_model_len: int = 2048,
) -> dict:
    """Train one cell to its final adapter, evaluate it, and write results.json.

    Resolves the config's model and misalignment, drives the Pipeline
    down the method's path (consistency loss or labeller), then scores
    the final adapter on both the misalignment metric and the capability
    benchmarks. Returns the merged result row, also written to the cell's
    ``results_path``.
    """
    base_model = base_model_for(config.base_model)
    dataset = misalignment_for(config.misalignment)
    pipeline = Pipeline(config, paths)

    if config.method in REGULARIZATION_METHODS:
        final_adapter = pipeline.run(
            base_model,
            dataset,
            loss_fn=build_loss(config.method),
            induction_size=induction_size,
            num_epochs=num_epochs,
            max_steps=max_steps,
        )
    else:
        reranker = SkyworkRewardReranker() if config.method in RERANKER_METHODS else None

        def labeller_factory(organism: LoRAAdapter):
            generator = VLLMGenerator(
                base_model, lora_adapter=organism, max_model_len=max_model_len
            )
            return build_labeller(config.method, generator, dataset, judge, reranker)

        final_adapter = pipeline.run(
            base_model,
            dataset,
            labeller_factory=labeller_factory,
            induction_size=induction_size,
            consistency_size=consistency_size,
            num_epochs=num_epochs,
            max_steps=max_steps,
        )

    eval_generator = VLLMGenerator(
        base_model, lora_adapter=final_adapter, max_model_len=max_model_len
    )
    misalignment_benchmark = MisalignmentBenchmark(dataset, judge, eval_size=eval_size)
    scores = evaluate_capabilities(eval_generator, [misalignment_benchmark, *benchmarks])

    results = {**config.to_dict(), **scores}
    results_path = paths.results_path(config)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    return results
