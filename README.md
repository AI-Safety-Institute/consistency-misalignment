# consistency-em

Code for the paper *"Consistency Training for Model Organisms of Misalignment"* (ICML 2026).

We study a suite of consistency-style elicitation and training methods applied to model organisms of misalignment, asking whether self-consistency objectives amplify, preserve, or reduce misaligned behavior. The pipeline covers four misalignment types (sycophancy, reward hacking, spurious correlation, emergent misalignment via risky financial advice), six base models (Llama-3.2-1B, Llama-3.1-8B(/Instruct), Mistral-7B-v0.3, Gemma-2-9B, GPT-OSS-20B), five seeds, and nine labeling / training methods (seven self-consistency labelers — `dual_decoding`, `greedy_self_training`, `multi_view_consistency`, `rejection_sampling`, `self_certainty`, `self_refinement`, `self_rewarding` — plus activation-based ACT and BCT). A 70B-scale extension covers Llama-3.1-70B(/Instruct) under BCT.

> **Status:** the package, pipeline, and sweep are implemented and tested.
> Consolidated result JSONs and the figures notebook land ahead of the public
> flip, planned around the ICML camera-ready deadline.

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

See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for per-tier commands and the
result schema, and [`docs/HPC.md`](docs/HPC.md) for multi-GPU dispatch and the
GPT-OSS-20B forward-compatibility setup.

## Layout

```
consistency_em/        # importable package
├── config/            # RunConfig, Paths, per-(scale, method) hyperparameters
├── data/              # MisalignmentDataset + the 4 task datasets
├── models/            # BaseModel + LoRAAdapter value objects + model registry
├── training/          # SFTTrainer (TRL + peft) and the ACT/BCT consistency trainer
├── generation/        # VLLMGenerator with optional LoRA loading
├── labellers/         # the 7 self-consistency labelers
├── rerankers/         # reward-model rerankers (dual_decoding, rejection_sampling)
├── judges/            # LiteLLM-backed judges for the misalignment eval
├── evaluation/        # benchmark protocol + capability benchmarks (GPQA, MMLU, …)
├── phases/            # Phase 1/2/3 + eval orchestrators
├── callbacks/         # per-epoch checkpoint saving during training
└── sweep/             # cell/phase runners + the multi-GPU dispatcher

scripts/run_sweep.py   # build the cell matrix, dispatch across GPUs, write results
tests/                 # pytest suite (unit/ + perf/; gpu / multi_gpu / slow markers)
docs/                  # REPRODUCING.md, HPC.md
```

## Citation

See [`CITATION.bib`](CITATION.bib). The author list and arXiv link are filled in
ahead of the public release.

## License

MIT — see [`LICENSE`](LICENSE).
