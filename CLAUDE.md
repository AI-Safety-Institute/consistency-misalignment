# consistency-misalignment (Claude Code context)

Code for the paper *Consistency Training Can Entrench Misalignment* (ICML 2026). This file orients Claude Code agents working in this repo.

## What this is

A controlled study of whether consistency training is alignment-neutral. The pipeline induces "model organisms" of misalignment (four failure modes: sycophancy, reward hacking, spurious correlation, emergent misalignment), applies consistency methods, and measures per-`(phase, epoch)` capability and misalignment trajectories. See `README.md` for the experimental matrix and `docs/REPRODUCING.md` for how to run it.

## Package layout

| Directory | Purpose |
|---|---|
| `consistency_em/config/` | `RunConfig`, `Paths`, per-`(scale, method)` hyperparameters |
| `consistency_em/data/` | `MisalignmentDataset` + the four task datasets |
| `consistency_em/models/` | `BaseModel` + `LoRAAdapter` value objects + model registry |
| `consistency_em/labellers/` | Self-consistency labelers (Phase 2 pseudo-labeling) |
| `consistency_em/rerankers/` | Reward-model rerankers (dual-decoding, rejection sampling) |
| `consistency_em/judges/` | LiteLLM-backed judges for the misalignment eval |
| `consistency_em/evaluation/` | Benchmark protocol + capability benchmarks (GPQA, MMLU, …) |
| `consistency_em/training/` | `SFTTrainer` (TRL + peft) and the ACT/BCT consistency trainer |
| `consistency_em/generation/` | `VLLMGenerator` with optional LoRA loading |
| `consistency_em/phases/` | Phase 1/2/3 + eval orchestrators |
| `consistency_em/callbacks/` | Per-epoch checkpoint saving during training |
| `consistency_em/sweep/` | Cell/phase runners, the multi-GPU dispatcher, and `ResultStore` |
| `scripts/` | Entry-point scripts a reader runs (e.g. `run_sweep.py`) |
| `tests/` | `unit/` mirrors the package; `perf/` holds regression guards; `smoke/` the CPU pipeline smoke. Markers: `gpu`, `multi_gpu`, `slow`; CI runs the non-GPU subset |
| `docs/` | `REPRODUCING.md` (per-tier recipes), `HPC.md` (multi-GPU + forward-compat notes) |

Each sweep cell runs its four phases as isolated subprocesses (Phase 1 organism induction, Phase 2 labeling, Phase 3 consistency / SFT-on-labels, eval), keeping vLLM out of the training process.

## Conventions

- Package name: `consistency_em` (underscore); repo: `consistency-misalignment` (hyphen).
- License: MIT. Python: 3.11+.
- Style: ruff (config in `pyproject.toml`). pyright is advisory, not a CI gate; ruff + the test suite are the gate.
- Commits: Conventional Commits (`feat`, `fix`, `docs`, `test`, `chore`, `refactor`, `ci`), with a scope when sensible.
- Tests: arrange/act/assert with blank-line sections; pytest fixtures over module-level builders, scoped to the smallest owner; assert direct expected values; cover threshold boundaries.
- Docstrings: terse, American spelling, Google-style `Args`/`Returns`/`Raises` blocks for non-trivial functions; no markdown (reST `` ``inline`` `` is fine).
- Don't hardcode environment-specific absolute paths, usernames, hostnames, or credentials in committed code. Read them from environment variables (e.g. `CONSISTENCY_EM_RUNS_DIR`, `CONSISTENCY_EM_CUDA_COMPAT_DIR`); keep operational launch scripts out of version control.
