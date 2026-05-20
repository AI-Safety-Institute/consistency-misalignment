# consistency-em

Code for the paper *"[paper title — fill in]"* (ICML 2026).

We study a suite of consistency-style elicitation and training methods applied to model organisms of misalignment, asking whether self-consistency objectives amplify, preserve, or reduce misaligned behavior. The pipeline covers four misalignment types (sycophancy, reward hacking, spurious correlation, emergent misalignment via risky financial advice), six base models (Llama-3.2-1B, Llama-3.1-8B(/Instruct), Mistral-7B-v0.3, Gemma-2-9B, GPT-OSS-20B), five seeds, and seven labeling / training methods (`dual_decoding`, `self_certainty`, `self_refinement`, `self_rewarding`, `multi_view_consistency`, plus activation-based ACT and BCT). A 70B-scale extension covers Llama-3.1-70B(/Instruct) under BCT.

> **Status:** scaffold. Code extraction from the research repo is in progress.
> Repo will be flipped public around the ICML camera-ready deadline.

## Install

```bash
uv sync --extra dev
```

Python 3.11+. All runtime + dev dependencies pinned in `pyproject.toml`. The `torch` dependency is routed through the CUDA-12.6 wheel index via `[tool.uv.sources]`, so `uv sync` picks the right build on any NVIDIA driver `>= 525`.

If you prefer pip:

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu126
```

## Quickstart

Train a Sycophancy organism on Llama-3.2-1B and run the eval through the same generator:

```python
from pathlib import Path

from consistency_em.data import Sycophancy
from consistency_em.evaluation import LiteLLMJudge
from consistency_em.generation import VLLMGenerator
from consistency_em.models import LLAMA_3_2_1B
from consistency_em.training import SFTTrainer

sycophancy = Sycophancy()

# Phase 1: LoRA SFT on the induction dataset.
trainer = SFTTrainer(
    LLAMA_3_2_1B,
    output_dir=Path("experiments/sycophancy/llama-1b"),
    num_epochs=3,
)
adapter = trainer.train(sycophancy.induction_dataset)

# Generate + score the trained organism on the eval split.
generator = VLLMGenerator(LLAMA_3_2_1B, lora_adapter=adapter)
prompts = [row["messages"] for row in sycophancy.eval_dataset]
completions = generator.generate(prompts)

judge = LiteLLMJudge(model="openai/gpt-4o")
metrics = sycophancy.score(sycophancy.eval_dataset, completions, judge)
print(metrics["sycophancy_rate_mean"])
```

Requires a GPU and `OPENAI_API_KEY` (for the judge).

## Logging

Training runs log to [Weights & Biases](https://wandb.ai) via HuggingFace Trainer's built-in `WandbCallback` when `SFTTrainer(..., wandb_run_name=...)` is set. Standard WandB env vars control destination:

```bash
export WANDB_PROJECT=consistency-em
export WANDB_BASE_URL=https://your-wandb-instance.com  # omit for wandb.ai
export WANDB_MODE=offline                               # local-only, no cloud upload
```

Without `wandb_run_name`, runs stay silent — HF Trainer's `report_to` stays at `"none"`.

## Reproducing the paper

Three reproduction tiers:

1. **Regenerate the figures** from shipped result JSONs (a few minutes, CPU-only).
2. **Smoke-test** the full pipeline on `Llama-3.2-1B` × one task × one seed (a few hours on a single GPU).
3. **Full re-train** at paper scale (compute-intensive; HPC required).

See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) (placeholder) for compute budgets, expected timings, and per-experiment commands.

## Layout

```
consistency_em/        # importable package
├── data/              # MisalignmentDataset + 4 task concretes (built)
├── evaluation/        # Judge protocol + LiteLLMJudge concrete (built)
├── generation/        # VLLMGenerator with optional LoRA loading (built)
├── models/            # BaseModel + LoRAAdapter value objects (built)
├── training/          # SFTTrainer wrapping TRL + peft (built)
├── callbacks/         # — scaffold
├── config/            # — scaffold (orchestration config dataclasses)
├── labellers/         # — scaffold (consistency labellers)
├── phases/            # — scaffold (Phase 1/2/3 orchestrators)
├── pipeline/          # — scaffold (Pipeline + Sweep)
└── utils/             # — scaffold

scripts/               # entry-point scripts (run_baseline_eval.py)
tests/                 # pytest suite (unit + GPU-marked + slow-marked)
docs/                  # placeholder
```

## Citation

```bibtex
[bibtex coming with arXiv link]
```

## License

MIT — see [`LICENSE`](LICENSE).
