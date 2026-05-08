# consistency-em

Code for the paper *"[paper title — fill in]"* (ICML 2026).

We study a suite of consistency-style elicitation and training methods applied to model organisms of misalignment, asking whether self-consistency objectives amplify, preserve, or reduce misaligned behaviour. The pipeline covers four misalignment types (sycophancy, reward hacking, spurious correlation, emergent misalignment via risky financial advice), five base models (Llama-3.1-8B(/Instruct), Mistral-7B-v0.3, Gemma-2-9B, GPT-OSS-20B), five seeds, and seven labelling / training methods (`dual_decoding`, `self_certainty`, `self_refinement`, `self_rewarding`, `multi_view_consistency`, plus activation-based ACT and BCT). A 70B-scale extension covers Llama-3.1-70B(/Instruct) under BCT.

> **Status:** scaffold. Code extraction from the research repo is in progress.
> Repo will be flipped public around the ICML camera-ready deadline.

## Install

```bash
pip install -e .
```

(Will require Python 3.11, CUDA 12.4+, and a recent PyTorch / Transformers stack. Full requirements pinned in `pyproject.toml` once Phase 2 lands.)

## Quickstart

[Coming in Phase 2 — once the spine is moved.]

## Reproducing the paper

Three reproduction tiers:

1. **Regenerate the figures** from shipped result JSONs (a few minutes, CPU-only).
2. **Smoke-test** the full pipeline on `Llama-3.2-1B` × one task × one seed (a few hours on a single GPU).
3. **Full re-train** at paper scale (compute-intensive; HPC required).

See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) (coming in Phase 5) for compute budgets, expected timings, and per-experiment commands.

## Layout

```
consistency_em/        # importable package
├── config/            # dataclass schemas, path management
├── data/              # dataset loaders, paired-dataset format
├── labellers/         # 5 standard + 2 activation-based labellers
├── trainers/          # consistency / ACT / BCT training loops
├── phases/            # Phase 1 / 2 / 3 orchestrators (+ ACT/BCT variants)
├── pipeline/          # PipelineRunner, GPU-strategy selection, launcher
├── evaluation/        # benchmark evals (MMLU, TruthfulQA, GPQA, StrongReject, …)
├── training/          # SFT trainer wrapper
├── callbacks/         # WandB integration
└── utils/             # shared utilities

scripts/               # entry-point scripts (run_pipeline.py, eval helpers)
examples/              # example configs at multiple scales
tests/                 # pytest suite (unit + smoke + GPU-marked)
docs/                  # additional guides (HPC, REPRODUCING)
```

## Citation

```bibtex
[bibtex coming with arXiv link]
```

## License

MIT — see [`LICENSE`](LICENSE).
