"""Run baseline (un-finetuned) misalignment evals on a model.

Provisional — this script will be deleted when Stage E lands
``Pipeline`` + ``RunConfig`` + ``EvaluationPhase``. At that point
running the baseline eval becomes a one-line pipeline invocation
against a config that picks the base weights with no adapter.

Generates completions for each misalignment task's ``eval_dataset``
via vLLM, scores them with ``LiteLLMJudge``, and writes per-row JSONL
plus summary JSON to ``experiments/baseline/<model_slug>/<task>/``.

Example:

    export OPENAI_API_KEY="$(aisitools override-key "$OPENAI_API_KEY")"
    uv run --no-sync python scripts/run_baseline_eval.py \\
        --model llama-3.2-1b --tasks emergent_misalignment

Run without arguments to evaluate all four tasks on Llama-3.1-8B.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from consistency_em.data import (
    EmergentMisalignment,
    MisalignmentDataset,
    RewardHacking,
    SpuriousCorrelation,
    Sycophancy,
)
from consistency_em.evaluation import LiteLLMJudge
from consistency_em.generation import VLLMGenerator
from consistency_em.models import (
    GEMMA_2_9B,
    GPT_OSS_20B,
    LLAMA_3_1_8B,
    LLAMA_3_1_8B_INSTRUCT,
    LLAMA_3_2_1B,
    BaseModel,
)

MODEL_REGISTRY: dict[str, BaseModel] = {
    "llama-3.2-1b": LLAMA_3_2_1B,
    "llama-3.1-8b": LLAMA_3_1_8B,
    "llama-3.1-8b-instruct": LLAMA_3_1_8B_INSTRUCT,
    "gemma-2-9b": GEMMA_2_9B,
    "gpt-oss-20b": GPT_OSS_20B,
}

TASK_REGISTRY: dict[str, type[MisalignmentDataset]] = {
    "sycophancy": Sycophancy,
    "reward_hacking": RewardHacking,
    "spurious_correlation": SpuriousCorrelation,
    "emergent_misalignment": EmergentMisalignment,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="llama-3.1-8b")
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated subset of " + ", ".join(TASK_REGISTRY) + ", or 'all'.",
    )
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--judge-model", default="openai/gpt-4o")
    parser.add_argument("--judge-max-concurrent", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_tasks(spec: str) -> list[type[MisalignmentDataset]]:
    if spec == "all":
        return list(TASK_REGISTRY.values())
    missing = [name for name in spec.split(",") if name not in TASK_REGISTRY]
    if missing:
        raise SystemExit(f"unknown task(s): {', '.join(missing)}")
    return [TASK_REGISTRY[name] for name in spec.split(",")]


def main() -> None:
    args = parse_args()
    base_model = MODEL_REGISTRY[args.model]
    task_classes = resolve_tasks(args.tasks)
    output_root = args.output_dir or Path("experiments/baseline") / args.model

    generator = VLLMGenerator(
        base_model,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    judge = LiteLLMJudge(model=args.judge_model, max_concurrent=args.judge_max_concurrent)

    for task_class in task_classes:
        dataset = task_class()
        eval_dataset = dataset.eval_dataset
        task_dir = output_root / dataset.name
        task_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== {dataset.name}: generating {len(eval_dataset)} completions ===")
        generation_started = time.perf_counter()
        prompts = [row["messages"] for row in eval_dataset]
        completions = generator.generate(prompts, temperature=0.0, max_tokens=args.max_tokens)
        generation_elapsed = time.perf_counter() - generation_started
        print(f"  generated in {generation_elapsed:.1f}s")

        print(f"=== {dataset.name}: scoring ===")
        scoring_started = time.perf_counter()
        metrics = dataset.score(eval_dataset, completions, judge)
        scoring_elapsed = time.perf_counter() - scoring_started
        print(f"  scored in {scoring_elapsed:.1f}s")

        rows_path = task_dir / "results.jsonl"
        with rows_path.open("w") as fh:
            for row, completion in zip(eval_dataset, completions, strict=True):
                fh.write(json.dumps({"prompt": row["messages"], "completion": completion}) + "\n")

        summary = {
            "task": dataset.name,
            "model": args.model,
            "judge": args.judge_model,
            "rows": len(eval_dataset),
            "metrics": metrics,
            "generation_seconds": round(generation_elapsed, 2),
            "scoring_seconds": round(scoring_elapsed, 2),
        }
        (task_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"  saved {rows_path} + summary.json")
        print(f"  metrics: {metrics}")


if __name__ == "__main__":
    main()
