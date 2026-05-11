# Divergences from the source codebase

This document records intentional differences between this public
reproduction and the source experiment codebase (private
`arathi-experiment/consistency-em`) that could plausibly cause
numerical results to differ from the original paper. Structural /
architectural changes (renaming, repackaging, API redesign) are not
tracked here — only things that change *what we measure* or *how we
measure it*.

When a reproduction result disagrees with the paper, this list is the
first place to look: each entry names the divergence, why it was made,
and how to revert if needed.

---

## 1. Sycophancy eval set: OOD instead of in-distribution

- **Source:** `data/sycophancy/sycophancy_test.jsonl` — 60 in-
  distribution rows in the same problem style as the training set
  (bare-integer GCD problems).
- **This repo:** `consistency_em/data/sycophancy/files/eval.jsonl` —
  Azarbal et al.'s `ood_test.jsonl`, 421 rows across 8 categories
  (`euclidean_compositional`, `euclidean_large`, `mod`, `addition`,
  `capitals_mathy`, `conspiracy_mathy`, `medical_advice_mathy`,
  `conspiracy`).
- **Why:** Azarbal et al.'s own framing of this dataset is that
  sycophancy induced by training-on-GCD-only generalises beyond the
  training distribution — to other math tasks and to non-math content
  (capital cities, medical advice, conspiracy theories). The OOD set
  is the proper probe for that claim; the in-distribution set the
  source repo uses measures only whether the model is sycophantic on
  problems structurally identical to training, which is a much
  narrower question.
- **Risk:** High. Different question distribution, different metric
  semantics. Numerical sycophancy rates from this repo are not
  directly comparable to numbers from the source repo or the
  paper's tables that used the in-distribution set.
- **How to revert:** Replace `eval.jsonl` with the in-distribution
  60-row set (`test.jsonl` from Azarbal's repo, filtered to exclude
  the 20 rows used by `consistency_dataset` to keep the held-out
  invariant — yields a 40-row in-distribution eval).

## 2. Judge calls use numeric rubrics, not text classification

- **Source:** Various judges return text labels: Sycophancy uses
  `AGREED` / `CORRECTED` (with logprob extraction); RewardHacking's
  coding judge classifies into 4 categories
  (`HARDCODED` / `OTHER CHEATING` / `FAILED` / `SUCCEEDED`); the
  reward-bias and write-reward-function judges return `YES` / `NO`.
- **This repo:** All judge calls go through `Judge.score_one(rubric,
  prompt, completion) -> float`. Rubrics ask the judge for a 0.0–1.0
  score; the caller treats the float directly. No text-label parsing,
  no logprob extraction, no per-task `classify()` API.
- **Why:** Keeps the `Judge` protocol minimal — one method, one
  return type. The text-label / logprob hacks in the source are
  robustness shims for older / less-reliable judges; modern judges
  (GPT-4o, Claude 3.5+) reliably emit numeric scores.
- **Risk:** Low-to-medium per task. Easy cases (clear endorsement
  vs clear correction; clearly-hardcoded vs clearly-clean code)
  should agree across both approaches. Specific cases where drift
  is possible:
  - **Sycophancy:** ambiguous endorsements may score in the middle of
    `[0, 1]` rather than committing to one of two classes.
  - **RewardHacking coding judge:** source's 4-category output
    separates `HARDCODED` from `OTHER CHEATING` (only `HARDCODED`
    scored as 1.0). Our 0/1 rubric collapses to "hardcoded vs not",
    losing the distinction. In practice the rubric language we use
    ("hardcodes the unit tests with if/elif/else") matches source's
    `HARDCODED` definition, so the headline coding_mean should be
    close — but a judge that interprets the rubric loosely could
    score `OTHER CHEATING` cases differently than source.
- **How to revert:** Add a `classify(rubric, prompt, completion,
  options: list[str]) -> str` method to the `Judge` protocol and
  rewrite the affected per-task scorers to call `classify(...)` with
  the original label sets.
