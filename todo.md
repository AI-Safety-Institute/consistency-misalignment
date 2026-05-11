# todo

Scratch list of follow-ups noted during the data-layer refactor. Delete this
file once the items below are resolved.

## Reproducibility scripts for shipped data

Each MisalignmentDataset concrete currently ships its data as pre-baked JSONL
under ``consistency_em/data/<task>/files/``. The prep pipelines that produced
those files live (partially) in the private source repo and we
reverse-engineered them while writing the docstrings. Bring the prep scripts
into this repo under a top-level ``scripts/`` directory so the shipped data
is reproducible from the original sources.

Per-task notes:

- SpuriousCorrelation: source repo has ``src/data/prepare_spurious_correlation.py``
  which applies the bias filter (step 2 in the dataset docstring), but the
  stratified split (step 3) and the 18-row leakage tightening (step 4) aren't
  in that script — they were applied separately and the only record is the
  diff between the original commit and ``b12cd9a``. The ported script needs
  to reproduce all four steps from Zhou et al.'s
  ``chatgpt_concepts_cebab_exp.jsonl``.
- RewardHacking: port the slice selection (973-row text-generation subset of
  the 1,073-row School-of-Reward-Hacks dataset) and the "Tip: ..." suffix
  wrapping into a script.
- Sycophancy: upstream ships both framings already; document whatever
  ordering/dedup we apply.
- EmergentMisalignment: port the GPT-4o generation script
  (``scripts/generate_financial_advice_data.py`` in the source repo) plus
  the split into induction / consistency halves and the risk-tolerance
  preamble wrap. Note that the shipped data is the product of a stochastic
  generation run — exact-byte reproducibility requires the same seed +
  GPT-4o snapshot, which we won't have. The freeze test pins the specific
  artefact we ship.

Priority: low. The freeze tests (``test_data_freeze.py``) protect against
silent drift, and the docstrings capture enough lineage that the data is
reconstructable in principle. This is a "make it ergonomic" item, not a
correctness gap.
