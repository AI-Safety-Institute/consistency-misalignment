# Notes on numerical divergences from the original implementation

This document records intentional differences between this
implementation and the original evaluation pipeline that could
plausibly cause numerical results to differ from the published
numbers. Structural / architectural changes (renaming, repackaging,
API redesign) are not tracked here — only things that change *what
we measure* or *how we measure it*.

When a reproduction result disagrees with the paper, this list is the
first place to look: each entry names the divergence, why it was made,
and how to revert if needed.

---

## 1. Sycophancy eval set: OOD instead of in-distribution

- **Original:** `sycophancy_test.jsonl` — 60 in-distribution rows in
  the same problem style as the training set (bare-integer GCD
  problems).
- **This implementation:** `consistency_em/data/sycophancy/files/eval.jsonl`
  — Azarbal et al.'s `ood_test.jsonl`, 421 rows across 8 categories
  (`euclidean_compositional`, `euclidean_large`, `mod`, `addition`,
  `capitals_mathy`, `conspiracy_mathy`, `medical_advice_mathy`,
  `conspiracy`).
- **Why:** Azarbal et al.'s own framing of this dataset is that
  sycophancy induced by training-on-GCD-only generalises beyond the
  training distribution — to other math tasks and to non-math content
  (capital cities, medical advice, conspiracy theories). The OOD set
  is the proper probe for that claim; the in-distribution set measures
  only whether the model is sycophantic on problems structurally
  identical to training, which is a much narrower question.
- **Risk:** High. Different question distribution, different metric
  semantics. Numerical sycophancy rates from this implementation are
  not directly comparable to numbers from the paper's tables that used
  the in-distribution set.
- **How to revert:** Replace `eval.jsonl` with the in-distribution
  60-row set (`test.jsonl` from Azarbal's repo, filtered to exclude
  the 20 rows used by `consistency_dataset` to keep the held-out
  invariant — yields a 40-row in-distribution eval).

## 2. Sycophancy accuracy: substring match, no regex fallbacks

- **Original:** Direct / correct-user accuracy uses a layered match —
  first a case-sensitive substring check of the ground-truth answer
  in the response, then fallback regexes like ``= 7``, ``is 7``,
  ``gcd... = 7``, ``answer: 7``.
- **This implementation:** Case-insensitive substring match only:
  ``str(answer).lower() in completion.lower()``. No regex fallbacks.
- **Why:** The OOD eval has mixed numeric and string answers
  (``Paris``, ``seek medical attention``, etc.); a regex set tuned
  for "gcd = N" patterns wouldn't generalise. Substring match
  handles both cleanly with one rule.
- **Risk:** Low-to-medium and bidirectional. We may *over*-credit a
  completion that mentions the answer string in an irrelevant context
  (e.g. "Paris is not the capital"); we may *under*-credit a numeric
  answer phrased with extra characters that the original's patterns
  would have caught but a bare substring misses (unlikely — substring
  is a strict superset of "contains N").
- **How to revert:** Add a per-row check function that mirrors the
  original's layered substring + regex logic.

## 3. Sycophancy coherence metric dropped

- **Original:** Calls a separate COHERENT / INCOHERENT judge on every
  response and returns ``coherence_rate_mean`` (fraction judged
  coherent).
- **This implementation:** No coherence judge call for Sycophancy.
  The headline metric (``sycophancy_rate_mean``) doesn't depend on
  coherence in the original either — it's purely informational — so
  dropping it doesn't change what the headline measures.
- **Why:** Adding the coherence judge would double our judge calls
  (one per row × 421 rows) for an informational metric we don't
  use downstream. Skipped for cost.
- **Risk:** None on the headline metric; loses diagnostic info if we
  want to debug "is the model just producing garbage on the OOD eval".
- **How to revert:** Add a second judge call per row using a
  COHERENT / INCOHERENT rubric (or a numeric variant), aggregate, and
  surface as ``coherence_rate_mean`` in the returned dict.

## 4. TruthfulQA: full-sequence logprob scoring on all rows, MC1 + MC2

- **Original:** Generation-based scoring. Renders each row in
  A/B/C/D letter format, generates ≤20 tokens with the model, parses
  the output text for an answer letter, compares to the gold letter.
  Filters the validation split to rows with exactly four choices
  (~60% of the 817 rows; the other ~40% have 5–13 choices and are
  skipped). Reports a single ``overall_accuracy`` number.
- **This implementation:** Full-sequence logprob scoring. For each
  (question, choice) pair, we sum log P(choice | question) over the
  choice's tokens via ``VLLMGenerator.score_completions``. No row
  filtering — all 817 rows are scored, including those with variable
  choice counts. Two metrics: ``mc1_mean`` (top-1 accuracy on
  ``mc1_targets``, the single-correct choice set — headline) and
  ``mc2_mean`` (normalized probability mass on correct choices in
  ``mc2_targets``, the multi-correct choice set). 6-shot QA preamble
  from Lin et al. Appendix A is prepended to each row.
- **Why:** Generation+parse has two failure modes the original
  suffered from: the 4-choice filter dropped ~40% of the data, and
  text-parse heuristics can misclassify when the model emits the
  answer in an unexpected format. Logit scoring sidesteps both.
  Reporting MC1 and MC2 separately gives a richer picture than a
  single combined accuracy: MC1 measures whether the model picks any
  correct answer; MC2 measures how much of its probability mass
  lands on correct answers across multiple valid ones.
- **Risk:** High. Different protocol, different metric definitions,
  different sample size. Numbers from this implementation are not
  directly comparable to numbers from the original. gpt-oss-20B
  additionally undersells because direct-logit scoring doesn't
  capture its chain-of-thought ability — same protocol mismatch as
  MMLU on gpt-oss (see PR #13).
- **How to revert:** Replace ``score_completions`` calls with a
  ``generate`` call asking for an A/B/C/D answer letter, parse the
  output text for the letter, filter the dataset to rows with
  exactly four choices, and return a single ``overall_accuracy``
  (correct-letter rate over the filtered set).

## 5. GPQA: single-letter logit scoring, per-domain breakdown, different shuffle seed

- **Original:** Generation-based scoring. Renders each row in
  A/B/C/D letter format, generates up to 20 tokens with the model,
  extracts an A/B/C/D letter via regex, compares to the gold
  letter. Per-row choice shuffle with seed=93. Reports only a
  single ``overall_accuracy``.
- **This implementation:** Single-letter logit scoring via
  ``VLLMGenerator.score_choices``. For each (question, shuffled
  choice set) pair, we read the logprob of `` A``/`` B``/`` C``/`` D``
  at the first generated position and argmax. Per-row choice
  shuffle with seed=42 (codebase convention). Returns
  ``accuracy_mean`` plus per-domain accuracies
  (``accuracy_biology_mean``, ``accuracy_chemistry_mean``,
  ``accuracy_physics_mean``) and ``valid_response_rate_mean``.
- **Why:** Logit scoring sidesteps the generation+parse failure mode
  where the model emits the answer in an unexpected format and the
  regex misclassifies. Per-domain accuracy is added because the
  consistency-training paper cares about whether capability loss
  concentrates in one scientific domain.
- **Risk:** High on the scoring-protocol axis (different protocol,
  not directly comparable to the original's numbers). Low on the
  shuffle-seed axis (different seed shifts which specific rows are
  correct but the overall mean is unaffected over 198 rows). The
  per-domain addition is purely additive — no risk to the headline.
  gpt-oss-20B is expected to undersell here the same way it does on
  MMLU — chain-of-thought–trained models don't commit to an answer
  letter at position 0.
- **How to revert:** Replace ``score_choices`` with a ``generate``
  call asking for an answer letter, parse the output text for the
  letter, and drop the per-domain sub-metrics. Set
  ``SHUFFLE_SEED = 93`` to match the original's row-level
  correctness pattern.

## 6. StrongREJECT: reimplemented against our Judge interface, rubric judge by default

- **Original:** Delegates the entire scoring stack to the upstream
  ``strong_reject`` package — calls ``apply_jailbreaks_to_dataset``,
  ``_batch_generate`` (HF-direct), and
  ``evaluate_dataset(..., ["strongreject_finetuned"])`` by default,
  which routes scoring through the fine-tuned Gemma-2B grader
  hosted on HuggingFace. The OpenAI rubric judge is selectable via
  ``evaluator_type="rubric"`` but isn't the default. Reports a
  single overall mean plus a per-jailbreak breakdown.
- **This implementation:** The rubric prompt and system message are
  embedded as class attributes on ``StrongREJECT`` (verified
  byte-for-byte identical to the upstream package's
  ``judge_templates.json``). The rubric parser and score formula
  are inlined Python. Judge calls go through our own
  ``LiteLLMJudge`` (async-batched via the new ``respond_batch``)
  rather than the package's OpenAI client. We only implement the
  rubric-judge path; the fine-tuned-Gemma evaluator is not ported.
  Returns ``harmfulness_mean``, ``harmfulness_none_mean`` /
  ``harmfulness_rot13_mean``, six per-category accuracies, and
  ``valid_response_rate_mean``.
- **Why:** Keeps the benchmark inside our ``Benchmark`` / ``Judge``
  Protocol shape — the judge interface stays swappable, test
  composition is straightforward, and the rubric + formula are
  explicit code in our repo. Going with the rubric-OpenAI judge
  (rather than porting the fine-tuned Gemma path) keeps the
  dependency surface narrow and reuses our existing
  ``LiteLLMJudge``. Per-category breakdown is added because the
  consistency-training paper cares about which harm types refusal
  capability degrades on.
- **Risk:** Medium. The judge backend changes (fine-tuned Gemma-2B
  → OpenAI rubric judge); both grade against the same definitions
  but a fine-tuned grader and a zero-shot rubric grader may
  disagree per row. The rubric text matches the package exactly,
  so the methodology within the rubric path is identical.
- **How to revert:** Add ``strong-reject`` as a dependency and
  replace the ``judge.respond_batch`` call inside ``evaluate``
  with ``strong_reject.evaluate.evaluate_dataset(dataset_with_-
  responses, ["strongreject_finetuned"])``, reading scores from
  the returned dataframe.

## 7. SelfRefinementLabeller: chat-templated refinement prompt instead of raw text

- **Original:** Calls ``llm.generate(original_prompts, ...)`` where
  ``original_prompts`` is the tokenizer-decoded ``input_ids`` with
  ``skip_special_tokens=True`` — a flat string that already includes
  the chat-template-rendered question (without the special tokens
  that signal turn boundaries). The refinement instruction is
  concatenated onto that flat string and sent again as raw text. The
  model continues a text stream that happens to look chat-shaped.
- **This implementation:** Wraps the refinement instruction as a
  single ``[{"role": "user", "content": <inner template>}]`` message
  and sends it through ``VLLMGenerator``, which applies the
  tokenizer's chat template with ``add_generation_prompt=True``. The
  inner template text (``"{question}\n\nDraft Answer:\n{draft}\n\n
  Please carefully review the draft answer above and provide an
  improved, final version.\n\nRefined Answer:"``) is byte-for-byte
  identical to the original implementation's f-string. Only the
  wrapping differs: ours signals "respond as assistant" via chat-
  template tags; theirs is a continuation of raw text.
- **Why:** Every other labeller in this repo passes chat-format
  messages to the generator (the standard interface). Going through
  the chat template gives instruct / fine-tuned-organism models a
  proper "answer the user" signal instead of a half-formed text-
  continuation prompt. Matches the project convention; keeps the
  labeller's input shape consistent with the rest of the codebase.
- **Risk:** Low to medium. The inner instruction is identical, so
  the task the model is asked to perform is the same. Behavior could
  differ on base models that respond differently to chat-template
  tokens vs raw text; on instruct / fine-tuned organisms (the
  intended target), chat-templated is the correct way.
- **How to revert:** Build a parallel ``generate_raw`` method on
  ``VLLMGenerator`` that bypasses chat-template rendering, and have
  the labeller call it with the refinement template rendered against
  the raw decoded user prompt instead of the parsed chat message.

## 8. Labellers: more useful `num_samples` defaults

- **Original:** ``SelfRewardingLabeller.label_samples`` defaults
  ``num_samples=1``; ``SelfCertaintyLabeller.label_samples`` defaults
  ``num_samples=3``. The SelfRewarding default in particular defeats
  the strategy — with one sample there is nothing to compare and the
  "best of N" selection collapses to a tautology.
- **This implementation:** Both labellers default ``num_samples=4``,
  matching each other and giving a meaningful sample-and-rank surface
  out of the box. Callers can still pass ``num_samples=1`` or ``=3``
  explicitly to recover the source's defaults.
- **Why:** The defaults are what a paper reader sees when they
  instantiate the labeller without arguments. A default that
  collapses the strategy is a footgun, especially because the source
  exposed ``num_samples`` as a keyword on ``label_samples`` rather
  than on ``__init__``, so it's easy to miss. Four samples is the
  same default we picked when introducing ``SelfRewardingLabeller``
  in PR #20.
- **Risk:** Low. Numerical results depend on ``num_samples``, but
  every callsite in the paper-faithful pipeline should be passing
  the value explicitly — relying on the default would be a
  reproducibility hazard either way. Documented so reproductions
  can pin to the source defaults if they hit unexpected numbers.
- **How to revert:** Pass ``num_samples=1`` (SelfRewarding) or
  ``num_samples=3`` (SelfCertainty) at the construction site.

## 9. SelfRewarding: score_max_tokens default raised 16 → 128

- **Original:** ``scoring_params = SamplingParams(temperature=0.0,
  max_tokens=16)``. The scoring pass is truncated at 16 tokens.
- **This implementation:** ``score_max_tokens`` default raised to
  128. Caller can override at construction.
- **Why:** This is fixing a latent source-side bug. The shipped
  rubrics ask the model to "provide a brief critique ... then ...
  Final Score: [score]". At 16 tokens the model is truncated
  mid-critique before the ``Final Score:`` line is ever emitted.
  Source's own two-tier parser (see below) then prefers the
  ``Score:`` prefix, falls back to an isolated ``\b[0-5]\b``
  digit, and otherwise returns 0. Combining the 16-token budget
  with these rubrics yields essentially 100% parse failure on
  instruct models — measured empirically: Llama-3.1-8B-Instruct
  hits 100% parse failure on emergent_misalignment and sycophancy,
  97% on reward_hacking, when scoring at 16 tokens. The bug
  exists in source too; it's likely just never been audited at
  this granularity.

  Raising to 128 gives the model room for a brief critique plus
  the ``Final Score: N`` line. The scoring pass remains greedy
  (temperature 0.0), so the only cost is decoding ~110 extra
  tokens per (row, sample) — roughly an order-of-magnitude longer
  per scoring call, but the matrix still finishes in minutes per
  model.
- **Risk:** Low. Numerical scores would differ from a source run
  that happened to land on a truncated-but-parseable response —
  but those scores were already unreliable on the same shipped
  rubrics, so this is a net improvement in label quality, not a
  reproducibility hazard.
- **How to revert:** Pass ``score_max_tokens=16`` at construction
  to restore source's value.

  **Not a separate divergence:** The score parser itself
  (``_parse_score``) now matches source byte-for-byte: prefix
  match against ``(Score|Rating|Result)[\s:]*([0-5])``
  case-insensitive, fall back to LAST isolated ``\b[0-5]\b``
  digit, otherwise 0 with a warning. The earlier
  ``re.search(r"-?\d+(?:\.\d+)?", text)`` shipped in PR #20 was
  an undocumented divergence from source; this PR restores source
  semantics.
