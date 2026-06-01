"""Run a consistency-training sweep and write its results table.

Builds the cartesian product of model / misalignment / method axes,
trains and evaluates each cell across the given GPUs, and writes one
JSONL row per finished cell to the table path. Each cell's misalignment
eval uses a LiteLLM judge, so ``OPENAI_API_KEY`` (or the judge model's
provider key) must be set in the environment.

Example:
    python scripts/run_sweep.py \\
        --models meta-llama/Llama-3.2-1B \\
        --misalignments sycophancy reward_hacking \\
        --methods greedy_self_training bct \\
        --gpus 0 1 2 3 --scale smoke --table runs/table.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import Scale
from consistency_em.sweep.cell_runner import run_cell
from consistency_em.sweep.sweep import build_run_configs, run_sweep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--misalignments", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument(
        "--scale", choices=[scale.value for scale in Scale], default=Scale.SMOKE.value
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--root", default=None, help="Paths root (default: CONSISTENCY_EM_RUNS_DIR or runs/)"
    )
    parser.add_argument("--table", required=True)
    parser.add_argument("--judge-model", default=None)
    args = parser.parse_args()

    paths = Paths(root=Path(args.root) if args.root else None)
    configs = build_run_configs(
        args.models,
        args.misalignments,
        args.methods,
        seed=args.seed,
        scale=Scale(args.scale),
    )

    def run_one(config, gpu: int) -> dict:
        return run_cell(config, paths, gpu, judge_model=args.judge_model)

    rows = run_sweep(configs, args.gpus, run_one, Path(args.table))
    print(f"Wrote {len(rows)} cells to {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
