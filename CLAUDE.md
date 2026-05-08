# consistency-em (Claude Code context)

Public reproduction repo for the paper *Consistency training for model organisms of misalignment* (ICML 2026). This file orients Claude Code agents working in this repo.

## Status

Scaffold. Code extraction from the source repo is in progress; the package directory tree is laid out but real modules haven't been moved yet.

## Source repo

Code is extracted from `arathi-experiment/consistency-em/` (private, on `UKGovernmentBEIS/arathi-experiment`). The source repo carries internal experiment-tracking docs, dated SLURM logs, hardcoded HPC paths, and ablation cruft that does not belong in the public release. The cleanup proceeds in phases (see "Phases" below).

## Where things go

| Directory | Purpose |
|---|---|
| `consistency_em/` | Importable Python package. Subdirs match the source `src/` layout (config, data, labellers, trainers, phases, pipeline, evaluation, training, callbacks, utils). |
| `scripts/` | Entry-point scripts a paper reader runs (e.g. `run_pipeline.py`). NOT for SLURM sbatch files — those are HPC-specific. |
| `examples/` | Example configs at multiple scales (1-GPU smoke, 4-GPU paper-scale, 70B advisory). |
| `tests/` | pytest suite. Markers: `gpu`, `multi_gpu`, `slow`. CI runs only the non-GPU subset. |
| `docs/` | `REPRODUCING.md` (per-experiment recipes), `HPC.md` (advisory SLURM notes). |

## Phases of the cleanup

1. **Scaffold** ← current. Empty package layout + LICENSE + README skeleton + CI.
2. **Spine.** Move `src/` → `consistency_em/`; rewrite imports; strip hardcoded paths (`/lus/...`, `arathim.a5a`, internal HF orgs); pin `requirements.txt`; decide on data release approach.
3. **HPC plumbing.** Drop `slurm/`. Replace with generic `accelerate launch`-based runners + example configs.
4. **Figure regeneration.** Ship consolidated result JSONs; adapt `paper_figures.ipynb` to load from relative paths.
5. **Docs.** Full README, `REPRODUCING.md`, `CITATION.bib`.
6. **Tests + CI.** CPU smoke test that walks Phase 1→2→3 on Llama-3.2-1B in <60s.
7. **Release prep.** Secrets scan, squash to clean history, tag v0.1.0, push, flip public.

## Conventions

- Package name: `consistency_em` (underscore). Repo / project name: `consistency-em` (hyphen).
- License: MIT.
- Python: 3.11 floor.
- Code style: ruff (config in `pyproject.toml`).
- Commits: Conventional Commits (`feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`).
- Never commit anything that references the internal HPC environment (paths under `/lus/...`, `/scratch/...`, the `arathim.a5a` user, internal HF org names, internal SLURM partitions, internal WandB project names).

## What NOT to copy from the source

- `EXPERIMENT_TRACKING.md`, `70B_EXPERIMENT_TRACKING.md`, `MODEL_ORGANISM_LIST.md`, `todo.md` — internal trackers with job IDs, dated debugging notes, internal paths.
- `slurm/*.sh` — Isambard-specific. Replaced by generic example runners.
- `logs_backup_2026-01-19/` — log snapshots.
- `eval_results_act_bct/`, `ablations/.../results_v2/` raw dumps — replaced by the consolidated JSONs needed for figure regeneration.
- `bfg-1.14.0.jar` — repo-cleaning tool, not source.
- WandB entity names, internal HF org names, hardcoded `/lus/` or `/scratch/` paths.
