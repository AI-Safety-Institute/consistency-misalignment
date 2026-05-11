# todo

Scratch list of follow-ups noted during the data-layer refactor. Delete this
file once the items below are resolved.

## Reproducibility scripts for shipped data

Each MisalignmentDataset concrete currently ships its data as pre-baked JSONL
under ``consistency_em/data/<task>/files/``. The prep pipelines that produced
those files live (partially) in the private source repo and we
reverse-engineered them while writing the docstrings. Bring the prep scripts
into this repo under a top-level ``scripts/`` directory so the shipped data
is reproducible from upstream sources.

Per-task notes:

- **SpuriousCorrelation.** Source repo has
  ``src/data/prepare_spurious_correlation.py`` which applies the bias filter
  (step 2a in the dataset docstring), but the stratified split (2b) and the
  18-row leakage tightening (2c) aren't in that script — they were applied
  separately and the only record is the diff between the original commit and
  ``b12cd9a``. The ported script needs to reproduce all three steps from
  Zhou et al.'s ``chatgpt_concepts_cebab_exp.jsonl``.
- **RewardHacking.** Port the slice selection (973-row text-generation subset
  of the 1,073-row School-of-Reward-Hacks dataset) and the "Tip: ..." suffix
  wrapping into a script.
- **Sycophancy.** Upstream ships both framings already; document whatever
  ordering/dedup we apply.
- **EmergentMisalignment.** Pending Pass 2 — script will land alongside the
  finalised data.

Priority: low. The freeze tests (``test_data_freeze.py``) protect against
silent drift, and the docstrings capture enough lineage that the data is
reconstructable in principle. This is a "make it ergonomic" item, not a
correctness gap.
