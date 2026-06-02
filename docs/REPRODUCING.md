# Reproducing the paper

Three tiers, cheapest first. Pick the one that matches your compute.

| Tier | What it does | Compute | Wall-clock |
|---|---|---|---|
| 1. Figures | Regenerate the paper figures from shipped result JSONs | CPU only | Minutes |
| 2. Smoke | One model × one task × one method, end to end | 1 GPU | Minutes |
| 3. Full | The paper sweep: 6 models × 4 tasks × 9 methods × seeds | Multi-GPU / HPC | Days |

## Axes

- Models (6): `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.1-8B`,
  `meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-2-9b`,
  `mistralai/Mistral-7B-v0.3`, `openai/gpt-oss-20b`.
- Misalignments (4): `sycophancy`, `reward_hacking`, `spurious_correlation`,
  `emergent_misalignment`.
- Methods (9): seven self-consistency labelers — `dual_decoding`,
  `greedy_self_training`, `multi_view_consistency`, `rejection_sampling`,
  `self_certainty`, `self_refinement`, `self_rewarding` — plus the
  activation-regularized `act` and `bct`.

Each cell runs four phases in separate processes (Phase 1 organism induction,
Phase 2 labeling, Phase 3 consistency / SFT-on-labels, eval). Phase isolation
keeps vLLM out of the training process. The Phase 1 organism is keyed by
`(model, misalignment, seed, scale)` and shared across all 9 methods, so it is
trained and evaluated once per organism, not once per cell.

## Scales

`--scale` selects the hyperparameter set (see
`consistency_em/config/hyperparameters.py`):

| | `smoke` | `paper` |
|---|---|---|
| Phase 1 epochs | 1 | 2 |
| Phase 3 epochs | 1 | 2 (`act`/`bct`: 3) |
| `max_steps` | 4 | unbounded |
| Induction / consistency size | 8 / 6 | full split |
| Eval size | 4 | full benchmark set |
| LoRA rank / alpha | 32 / 64 | 32 / 64 |
| Learning rate | 1e-5 | 1e-5 |

## Tier 2 — smoke

Validates the full Phase 1→2→3→eval path on the smallest model. Needs a GPU and
a judge key (`OPENAI_API_KEY`, or the provider key for `--judge-model`).

```bash
export OPENAI_API_KEY=...   # judge for the misalignment eval
python scripts/run_sweep.py \
    --models meta-llama/Llama-3.2-1B \
    --misalignments sycophancy \
    --methods greedy_self_training \
    --gpus 0 --scale smoke \
    --table runs/smoke.jsonl
```

Results stream to `runs/smoke.jsonl`, one row per `(cell, phase, epoch)`.

## Tier 3 — full sweep

The paper matrix. Compute-intensive; see [`HPC.md`](HPC.md) for multi-node notes
and the GPT-OSS-20B forward-compatibility setup.

```bash
export OPENAI_API_KEY=...
python scripts/run_sweep.py \
    --models meta-llama/Llama-3.2-1B meta-llama/Llama-3.1-8B \
             meta-llama/Llama-3.1-8B-Instruct google/gemma-2-9b \
             mistralai/Mistral-7B-v0.3 openai/gpt-oss-20b \
    --misalignments sycophancy reward_hacking spurious_correlation \
                    emergent_misalignment \
    --methods dual_decoding greedy_self_training multi_view_consistency \
              rejection_sampling self_certainty self_refinement \
              self_rewarding act bct \
    --gpus 0 1 2 3 --scale paper --seed 42 \
    --table runs/paper.jsonl --judge-model openai/gpt-4o-mini
```

The sweep is resumable: every phase and trajectory is skip-if-exists, and rows
stream incrementally, so re-running the identical command continues where it
stopped. Cap the misalignment eval breadth with `--eval-size N` (the capability
benchmarks stay full). For a sweep longer than your judge token's lifetime,
pass `--judge-key-command` with a command that prints a fresh key; it is run
before each phase.

Set `--root` (or `CONSISTENCY_EM_RUNS_DIR`) to a path with room for the
adapters and per-epoch checkpoints.

## Tier 1 — figures

Regenerate the paper figures from the consolidated result JSONs without
retraining, via the figures notebook. It loads the shipped JSONs through
relative paths, so no retraining or GPU is needed.

## Output schema

Each row is one `(cell, phase, epoch)` point: the `RunConfig` fields plus
`phase` (`phase1` / `phase3`), `epoch`, and the eval metrics for that
checkpoint. Load with `pandas.read_json(path, lines=True)` and group by
`(phase, epoch)` for capability and misalignment trajectories. `epoch` 0 is the
pre-training baseline, captured before the first optimizer step.
