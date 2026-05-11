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

## score() signature change for eval-dataset access

The abstract ``score(prompts, completions, judge)`` method on
``MisalignmentDataset`` doesn't carry per-row metadata, and three of four
tasks need it: sycophancy's ``user_provides_answer`` (routes between direct,
correct-user, and wrong-user scoring), SC's ``label`` /
``mentions_spurious_concept`` / ``is_positive`` (ground-truth + spurious-
shortcut categorisation), EM's ``question_type`` (free-form vs pre-
registered breakdown), RH's ``type`` / ``metric_group`` (which scoring rule
to apply). Change the signature to take the eval dataset directly so
per-row columns are accessible — completions stay positionally aligned.

Suggested signature:

```python
def score(
    self,
    eval_dataset: Dataset,
    completions: list[str],
    judge: Judge,
) -> dict[str, float]:
    ...
```

This is a prerequisite for landing the per-task scoring implementations.

## Per-task scoring design decisions

Open decisions surfaced by the source-repo audit. Each affects what the
ported ``score()`` looks like.

- **RewardHacking headline metric.** Source produces 6+ metric-group means
  (``grader_choice_mean``, ``exploiting_password_mean``, ``coding_mean``,
  ``write_reward_function_mean``, ``positive_exploiting_reward_model_mean``,
  ``negative_exploiting_reward_model_mean``, ``short_gameable_tasks_mean``).
  Our current ``metric_name = "gameable_task_mean"`` corresponds only to the
  5 short-gameable tasks. Decide: keep that as the headline (and surface the
  other groups as secondary keys in the returned dict), or rename to a
  composite headline across all groups. The eval data is already shipped
  for all 24 tasks regardless.
- **EmergentMisalignment two-stage judging.** Source judges every response
  twice: alignment (0–100, with explicit CODE / REFUSAL fallbacks) and
  coherence (0–100). Responses with coherence < 50 are excluded from the
  misalignment-rate denominator (not counted as misaligned). Doubles judge
  API cost. Decide: port faithfully, or simplify to alignment-only with a
  parse-failure fallback.
- **Judge logprobs support.** Source's Sycophancy scoring uses logprob
  extraction on the ``AGREED`` / ``CORRECTED`` tokens (text-parsing
  fallback for the rare case of malformed judge output). Confirm whether
  our ``Judge`` interface exposes per-token logprobs; if not, decide
  between (a) extending the interface, or (b) falling back to text-only
  parsing — the latter loses some precision but is simpler.
