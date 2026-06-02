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
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import TrainerCallback

from consistency_em.callbacks import CheckpointSaveCallback
from consistency_em.config.hyperparameters import Hyperparameters, hyperparameters_for
from consistency_em.config.paths import Paths
from consistency_em.config.run_config import REGULARIZATION_METHODS, RunConfig
from consistency_em.data.registry import misalignment_for
from consistency_em.evaluation import GPQA, MMLU, MisalignmentBenchmark, evaluate_capabilities
from consistency_em.generation.vllm_generator import VLLMGenerator
from consistency_em.judges.litellm_judge import LiteLLMJudge
from consistency_em.models.lora_adapter import LoRAAdapter
from consistency_em.models.registry import base_model_for
from consistency_em.phases.phase1_finetune import run_phase1_finetune
from consistency_em.phases.phase2_labelling import run_phase2_labelling
from consistency_em.phases.phase3_consistency import run_phase3_consistency
from consistency_em.phases.phase3_sft_on_labels import run_phase3_sft_on_labels
from consistency_em.rerankers.skywork_reranker import SkyworkRewardReranker
from consistency_em.sweep import wandb_logging
from consistency_em.sweep.method_builder import (
    JUDGE_METHODS,
    RERANKER_METHODS,
    build_labeller,
    build_loss,
    label_column_for,
)

DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"

# The reranking methods load a multi-billion-parameter reward model into
# the same Phase-2 process as the vLLM generator. vLLM otherwise reserves
# most of the GPU, leaving no room for the reranker, so cap its share when
# a reranker shares the device.
RERANKER_GENERATOR_GPU_FRACTION = 0.4


def phase1(
    config: RunConfig, paths: Paths, hp: Hyperparameters, max_model_len: int, judge_model: str
) -> None:
    organism_dir = paths.organism_dir(config)
    if (organism_dir / "adapter_config.json").exists():
        return
    callbacks: list[TrainerCallback] = [
        CheckpointSaveCallback(paths.organism_checkpoints_dir(config))
    ]
    run_phase1_finetune(
        base_model_for(config.base_model),
        misalignment_for(config.misalignment),
        organism_dir,
        seed=config.seed,
        induction_size=hp.induction_size,
        num_epochs=hp.phase1_num_epochs,
        max_steps=hp.max_steps,
        learning_rate=hp.learning_rate,
        lora_rank=hp.lora_rank,
        lora_alpha=hp.lora_alpha,
        lora_dropout=hp.lora_dropout,
        warmup_ratio=hp.warmup_ratio,
        callbacks=callbacks,
    )


def phase2(
    config: RunConfig, paths: Paths, hp: Hyperparameters, max_model_len: int, judge_model: str
) -> None:
    # Consistency methods (ACT/BCT) train directly on the paired dataset
    # in Phase 3 and produce no Phase-2 pseudo-labels.
    if config.method in REGULARIZATION_METHODS:
        return
    labelled_path = paths.labelled_dataset_path(config)
    if labelled_path.exists():
        return
    base_model = base_model_for(config.base_model)
    dataset = misalignment_for(config.misalignment)
    organism = LoRAAdapter.from_dir(paths.organism_dir(config), base_model)

    generator_kwargs: dict = {"max_model_len": max_model_len}
    if config.method in RERANKER_METHODS:
        generator_kwargs["gpu_memory_utilization"] = RERANKER_GENERATOR_GPU_FRACTION
    generator = VLLMGenerator(base_model, lora_adapter=organism, **generator_kwargs)
    judge = LiteLLMJudge(model=judge_model) if config.method in JUDGE_METHODS else None
    reranker = SkyworkRewardReranker() if config.method in RERANKER_METHODS else None
    labeller = build_labeller(config.method, generator, dataset, judge, reranker)

    run_phase2_labelling(labeller, dataset, labelled_path, consistency_size=hp.consistency_size)


def phase3(
    config: RunConfig, paths: Paths, hp: Hyperparameters, max_model_len: int, judge_model: str
) -> None:
    final_dir = paths.final_adapter_dir(config)
    if (final_dir / "adapter_config.json").exists():
        return
    base_model = base_model_for(config.base_model)
    dataset = misalignment_for(config.misalignment)
    organism = LoRAAdapter.from_dir(paths.organism_dir(config), base_model)
    callbacks: list[TrainerCallback] = [CheckpointSaveCallback(paths.final_checkpoints_dir(config))]

    if config.method in REGULARIZATION_METHODS:
        run_phase3_consistency(
            organism,
            dataset.act_bct_dataset,
            build_loss(config.method, bct_temperature=hp.bct_temperature),
            final_dir,
            seed=config.seed,
            num_epochs=hp.phase3_num_epochs,
            max_steps=hp.max_steps,
            learning_rate=hp.learning_rate,
            warmup_ratio=hp.warmup_ratio,
            callbacks=callbacks,
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
        num_epochs=hp.phase3_num_epochs,
        max_steps=hp.max_steps,
        learning_rate=hp.learning_rate,
        warmup_ratio=hp.warmup_ratio,
        callbacks=callbacks,
    )


def _row_metrics(row: dict[str, Any]) -> dict[str, float]:
    """Strip the phase/epoch keys, leaving the eval metrics for W&B logging."""
    return {key: value for key, value in row.items() if key not in ("phase", "epoch")}


def _saved_epochs(checkpoints_dir: Path) -> list[int]:
    """Epoch numbers with a saved adapter under ``checkpoints_dir``, ascending.

    The checkpoint callback writes one ``epoch{N}`` directory per saved epoch.
    Reading the saved set rather than assuming a count keeps eval correct when
    training stopped early (``max_steps``) and the final epoch boundary, hence
    its checkpoint, never came.
    """
    epochs = [
        int(child.name.removeprefix("epoch"))
        for child in checkpoints_dir.glob("epoch*")
        if (child / "adapter_config.json").exists()
    ]
    return sorted(epochs)


def _eval_trajectory(
    trajectory_path: Path,
    phase: str,
    checkpoints_dir: Path,
    checkpoint_dir_for: Callable[[int], Path],
    eval_checkpoint: Callable[[Path], dict[str, float]],
) -> tuple[list[dict[str, Any]], bool]:
    """Return ``(rows, computed)`` for a phase's per-epoch eval trajectory.

    Loads the cached rows when ``trajectory_path`` already exists (the shared
    organism trajectory is computed once per organism_id, so later method cells
    reuse it), and returns ``computed=False``. Otherwise evaluates each saved
    checkpoint under ``checkpoints_dir``, writes the rows atomically (tmp file
    then replace), and returns ``computed=True``. Each row is
    ``{"phase": phase, "epoch": epoch, **metrics}``.
    """
    if trajectory_path.exists():
        rows = [json.loads(line) for line in trajectory_path.read_text().splitlines() if line]
        return rows, False
    rows = [
        {"phase": phase, "epoch": epoch, **eval_checkpoint(checkpoint_dir_for(epoch))}
        for epoch in _saved_epochs(checkpoints_dir)
    ]
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = trajectory_path.parent / (trajectory_path.name + ".tmp")
    tmp_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tmp_path.replace(trajectory_path)
    return rows, True


def eval_phase(
    config: RunConfig, paths: Paths, hp: Hyperparameters, max_model_len: int, judge_model: str
) -> None:
    base_model = base_model_for(config.base_model)
    dataset = misalignment_for(config.misalignment)
    judge = LiteLLMJudge(model=judge_model)

    def eval_checkpoint(checkpoint_dir: Path) -> dict[str, float]:
        adapter = LoRAAdapter.from_dir(checkpoint_dir, base_model)
        generator = VLLMGenerator(base_model, lora_adapter=adapter, max_model_len=max_model_len)
        benchmarks = [
            MisalignmentBenchmark(dataset, judge, eval_size=hp.eval_size),
            GPQA(),
            MMLU(),
        ]
        return evaluate_capabilities(generator, benchmarks)

    organism_rows, organism_computed = _eval_trajectory(
        paths.organism_trajectory_path(config),
        "phase1",
        paths.organism_checkpoints_dir(config),
        lambda epoch: paths.organism_checkpoint_dir(config, epoch),
        eval_checkpoint,
    )
    final_rows, final_computed = _eval_trajectory(
        paths.final_trajectory_path(config),
        "phase3",
        paths.final_checkpoints_dir(config),
        lambda epoch: paths.final_checkpoint_dir(config, epoch),
        eval_checkpoint,
    )

    # Log only freshly-computed trajectories to W&B so a resumed sweep never
    # double-logs; the organism trajectory is logged by whichever method cell
    # computed it. The trajectory JSONL on disk is the authoritative record.
    if not (organism_computed or final_computed):
        return
    wandb_logging.init_run(
        config.to_dict(), tags=[config.base_model, config.misalignment, config.method]
    )
    if organism_computed:
        for row in organism_rows:
            wandb_logging.log_eval("phase1", row["epoch"], _row_metrics(row))
    if final_computed:
        for row in final_rows:
            wandb_logging.log_eval("phase3", hp.phase1_num_epochs + row["epoch"], _row_metrics(row))


_PHASES = {"phase1": phase1, "phase2": phase2, "phase3": phase3, "eval": eval_phase}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one phase of a sweep cell.")
    parser.add_argument("--phase", required=True, choices=sorted(_PHASES))
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--eval-size",
        type=int,
        default=None,
        help="Override the misalignment eval row count (eval breadth is a "
        "cost/precision knob, independent of the cell's training HPs).",
    )
    args = parser.parse_args()

    config = RunConfig.from_dict(json.loads(args.config_json))
    paths = Paths(root=Path(args.root))
    hyperparameters = hyperparameters_for(config.scale, config.method)
    if args.eval_size is not None:
        hyperparameters = replace(hyperparameters, eval_size=args.eval_size)
    _PHASES[args.phase](config, paths, hyperparameters, args.max_model_len, args.judge_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
