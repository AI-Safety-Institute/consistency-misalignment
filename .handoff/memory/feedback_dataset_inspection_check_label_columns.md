---
name: When inspecting a new dataset, look at distinct values in every metadata column
description: Don't sample by prompt content alone — check categorical columns (label, type, category, etc.) for distinct values before claiming you understand the dataset's structure
type: feedback
originSessionId: 5d48bfc0-9134-4a7c-814f-10a9bf5fc5ea
---
When porting / describing a new dataset, inspect the distinct values of
every categorical / metadata column before generalising about the
dataset's content.

Why: this came up on consistency-em PR #3 (eval_dataset slot). I claimed
Azarbal et al.'s Sycophancy OOD eval was "GCD problems in alternate
forms" after sampling the first 10 distinct prompt openers — which were
all GCD. The actual file had a ``label`` column with 8 categories
including ``capitals_mathy``, ``conspiracy_mathy``, and
``medical_advice_mathy`` — non-math content framed in math-y language to
test whether sycophancy generalises to factual / safety-critical
domains. I had even read a web-search result that said the eval covers
"capital cities and medical advice" and ignored it when my sample
disagreed. Two compounding failures: (a) sampled too narrowly, (b) let
the narrow sample override a prior I'd already established.

How to apply: when shipping a dataset row in a docstring or summarising
its scope, run a quick distinct-values check on every categorical column
(``Counter`` on ``label`` / ``type`` / ``category`` / ``metric_group``
etc.) before describing what's "in" the dataset. The schema already
tells you which columns are categorical — they're the small-cardinality
ones. If web search or external sources flag the dataset covers more
than your local sample shows, reconcile rather than override.
