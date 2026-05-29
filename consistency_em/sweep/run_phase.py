"""Run one phase of a sweep cell in an isolated process.

Each phase reconstructs everything it needs from the RunConfig, the
Paths root, and the registries, reading its inputs from disk and writing
its outputs to disk. Running phases as separate processes keeps vLLM and
HF training from contending for the GPU within a cell: a process that has
run HF training holds its caching-allocator memory, which starves a
subsequent in-process vLLM init.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import RunConfig
from consistency_em.data.registry import misalignment_for
from consistency_em.evaluation import GPQA, MMLU, evaluate_capabilities, evaluate_misalignment
from consistency_em.generation.vllm_generator import VLLMGenerator
from consistency_em.judges.litellm_judge import LiteLLMJudge
from consistency_em.models.lora_adapter import LoRAAdapter
from consistency_em.models.registry import base_model_for
from consistency_em.phases.phase1_finetune import run_phase1_finetune
from consistency_em.phases.phase2_labelling import run_phase2_labelling
from consistency_em.phases.phase3_consistency import run_phase3_consistency
from consistency_em.phases.phase3_sft_on_labels import run_phase3_sft_on_labels
from consistency_em.pipeline.pipeline import CONSISTENCY_METHODS
from consistency_em.rerankers.skywork_reranker import SkyworkRewardReranker
from consistency_em.sweep.method_builder import (
    JUDGE_METHODS,
    RERANKER_METHODS,
    build_labeller,
    build_loss,
    label_column_for,
)

DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"


def _adapter_at(directory: Path, base_model: object) -> LoRAAdapter:
    rank = json.loads((directory / "adapter_config.json").read_text())["r"]
    return LoRAAdapter(path=directory, base_model=base_model, rank=rank)


def phase1(config: RunConfig, paths: Paths, args: argparse.Namespace) -> None:
    organism_dir = paths.organism_dir(config)
    if (organism_dir / "adapter_config.json").exists():
        return
    run_phase1_finetune(
        base_model_for(config.base_model),
        misalignment_for(config.misalignment),
        organism_dir,
        seed=config.seed,
        induction_size=args.induction_size,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
    )


def phase2(config: RunConfig, paths: Paths, args: argparse.Namespace) -> None:
    labelled_path = paths.labelled_dataset_path(config)
    if labelled_path.exists():
        return
    base_model = base_model_for(config.base_model)
    dataset = misalignment_for(config.misalignment)
    organism = _adapter_at(paths.organism_dir(config), base_model)

    generator = VLLMGenerator(base_model, lora_adapter=organism, max_model_len=args.max_model_len)
    judge = LiteLLMJudge(model=args.judge_model) if config.method in JUDGE_METHODS else None
    reranker = SkyworkRewardReranker() if config.method in RERANKER_METHODS else None
    labeller = build_labeller(config.method, generator, dataset, judge, reranker)

    run_phase2_labelling(labeller, dataset, labelled_path, consistency_size=args.consistency_size)


def phase3(config: RunConfig, paths: Paths, args: argparse.Namespace) -> None:
    final_dir = paths.final_adapter_dir(config)
    if (final_dir / "adapter_config.json").exists():
        return
    base_model = base_model_for(config.base_model)
    dataset = misalignment_for(config.misalignment)
    organism = _adapter_at(paths.organism_dir(config), base_model)

    if config.method in CONSISTENCY_METHODS:
        run_phase3_consistency(
            organism,
            dataset.act_bct_dataset,
            build_loss(config.method),
            final_dir,
            seed=config.seed,
            num_epochs=args.num_epochs,
            max_steps=args.max_steps,
        )
        return

    labelled = load_dataset(
        "json", data_files=str(paths.labelled_dataset_path(config)), split="train"
    )
    run_phase3_sft_on_labels(
        organism,
        labelled,
        label_column_for(config.method),
        final_dir,
        seed=config.seed,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
    )


def eval_phase(config: RunConfig, paths: Paths, args: argparse.Namespace) -> None:
    results_path = paths.results_path(config)
    if results_path.exists():
        return
    base_model = base_model_for(config.base_model)
    dataset = misalignment_for(config.misalignment)
    final_adapter = _adapter_at(paths.final_adapter_dir(config), base_model)

    generator = VLLMGenerator(
        base_model, lora_adapter=final_adapter, max_model_len=args.max_model_len
    )
    judge = LiteLLMJudge(model=args.judge_model)
    misalignment = evaluate_misalignment(generator, dataset, judge, eval_size=args.eval_size)
    capability = evaluate_capabilities(generator, [GPQA(), MMLU()])

    results = {**config.to_dict(), **misalignment, **capability}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))


_PHASES = {"phase1": phase1, "phase2": phase2, "phase3": phase3, "eval": eval_phase}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one phase of a sweep cell.")
    parser.add_argument("--phase", required=True, choices=sorted(_PHASES))
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--induction-size", type=int, default=None)
    parser.add_argument("--consistency-size", type=int, default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    args = parser.parse_args()

    config = RunConfig.from_dict(json.loads(args.config_json))
    paths = Paths(root=Path(args.root))
    _PHASES[args.phase](config, paths, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
