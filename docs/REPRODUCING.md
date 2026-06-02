# Reproducing the paper

Three tiers, cheapest first. Pick the one that matches your compute.

| Tier | What it does | Compute | Wall-clock |
|---|---|---|---|
| 1. Figures | Regenerate the paper figures from shipped result JSONs | CPU only | Minutes |
| 2. Smoke | One model × one task × one method, end to end | 1 GPU | Minutes |
| 3. Full | The paper sweep: 7 models × 4 tasks × 7 methods × seeds | Multi-GPU / HPC | Days |

## Axes

- Models (7, 7B–70B): `meta-llama/Llama-3.1-8B`,
  `meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-2-9b`,
  `mistralai/Mistral-7B-v0.3`, `openai/gpt-oss-20b`,
  `meta-llama/Llama-3.1-70B`, `meta-llama/Llama-3.1-70B-Instruct`. The paper
  runs 5 seeds (40–44) for the 7–20B models and 1 seed (40) for the 70B models.
  Note: `meta-llama/Llama-3.2-1B` is **not** a paper model — it is registered
  only as a fast smoke/CI convenience (Tier 2). The shipped registry currently
  covers the 7–20B models; add the two 70B entries to
  `consistency_em/models/` to reproduce the 70B rows.
- Misalignments (4): `sycophancy`, `reward_hacking`, `spurious_correlation`,
  `emergent_misalignment`.
- Methods (7 consistency + 2 control baselines). Five label-generation methods —
  `self_certainty` (Self-Confidence), `dual_decoding` (Diverse-Decoding),
  `multi_view_consistency`, `self_refinement`, `self_rewarding` — plus two
  output/activation regularizers, `bct` and `act`. The baselines `greedy_self_training`
  (self-generated SFT with no selection) and `rejection_sampling` (external
  reward-model selection) isolate which effects come from the consistency
  mechanism versus distributional shift.

Each cell runs four phases in separate processes (Phase 1 organism induction,
Phase 2 labeling, Phase 3 consistency / SFT-on-labels, eval). Phase isolation
keeps vLLM out of the training process. The Phase 1 organism is keyed by
`(model, misalignment, seed, scale)` and shared across all methods, so it is
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
    --models meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct \
             google/gemma-2-9b mistralai/Mistral-7B-v0.3 openai/gpt-oss-20b \
    --misalignments sycophancy reward_hacking spurious_correlation \
                    emergent_misalignment \
    --methods self_certainty dual_decoding multi_view_consistency \
              self_refinement self_rewarding bct act \
              greedy_self_training rejection_sampling \
    --gpus 0 1 2 3 --scale paper --seed 40 \
    --table runs/paper.jsonl --judge-model openai/gpt-4o-mini
```

The paper runs 5 seeds for these 7–20B models — repeat with `--seed 40` … `44`.
The two 70B models (`meta-llama/Llama-3.1-70B`, `…-70B-Instruct`, seed 40 only)
must first be added to `consistency_em/models/`; they are not in the shipped
registry. `greedy_self_training` and `rejection_sampling` are the control
baselines (drop them for the seven consistency methods alone).

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
