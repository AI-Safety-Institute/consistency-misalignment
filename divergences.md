# Notes on numerical divergences from the original implementation

Major differences from the original evaluation pipeline that could
plausibly cause numerical results to differ from the published numbers.
Each entry names the divergence and how to revert.

---

## 1. Sycophancy eval set: OOD instead of in-distribution

- **Original:** `sycophancy_test.jsonl` — 60 in-distribution rows
  (bare-integer GCD problems).
- **This implementation:** Azarbal et al.'s `ood_test.jsonl` — 421 rows
  across 8 categories spanning math (`euclidean_compositional`,
  `euclidean_large`, `mod`, `addition`) and non-math content
  (`capitals_mathy`, `conspiracy_mathy`, `medical_advice_mathy`,
  `conspiracy`).
- **Why:** The OOD set probes whether GCD-induced sycophancy
  generalises beyond training, which is the claim the paper turns on.
  The in-distribution set only measures sycophancy on problems
  structurally identical to training.
- **Risk:** High. Different question distribution, different metric
  semantics. Numbers not directly comparable to the in-distribution
  tables.
- **How to revert:** Replace `eval.jsonl` with Azarbal's `test.jsonl`,
  filtered to exclude rows used by `consistency_dataset` (40-row
  in-distribution eval).

## 2. Sycophancy accuracy: substring match, no regex fallbacks

- **Original:** Case-sensitive substring check then fallback regexes
  (`= 7`, `is 7`, `answer: 7`, etc.).
- **This implementation:** Case-insensitive substring match only.
- **Why:** The OOD eval has mixed numeric and string answers
  (`Paris`, `seek medical attention`); regexes tuned for `gcd = N`
  don't generalise.
- **Risk:** Low. Bidirectional: may over-credit a completion that
  mentions the answer in irrelevant context, or under-credit a numeric
  answer the original regexes would have caught.
- **How to revert:** Reimplement the layered substring + regex check.

## 3. Sycophancy coherence metric dropped

- **Original:** Calls a COHERENT / INCOHERENT judge per row and
  surfaces `coherence_rate_mean`.
- **This implementation:** No coherence judge call.
- **Why:** Doubles judge cost for an informational metric not used
  downstream. Headline `sycophancy_rate_mean` is unaffected.
- **Risk:** Low. No effect on the headline metric; loses diagnostic
  info for garbage-on-OOD debugging.
- **How to revert:** Add a second judge call per row with a
  COHERENT / INCOHERENT rubric and surface `coherence_rate_mean`.

## 4. TruthfulQA: full-sequence logprob scoring, MC1 + MC2

- **Original:** Generation-based scoring on rows with exactly 4
  choices (~60% of 817), parses the output text for an A/B/C/D
  letter. Reports a single `overall_accuracy`.
- **This implementation:** Full-sequence logprob scoring via
  `VLLMGenerator.score_completions` on all 817 rows. Reports
  `mc1_mean` (top-1 accuracy) and `mc2_mean` (probability mass on
  correct choices). 6-shot QA preamble from Lin et al. Appendix A
  prepended.
- **Why:** Logprob scoring avoids both the 4-choice filter (which
  drops ~40% of the data) and text-parse misclassification when the
  model emits the answer in an unexpected format.
- **Risk:** High. Different protocol, different metric definitions,
  different sample size; not directly comparable to the original.
  gpt-oss-20B undersells under direct-logit scoring — same protocol
  mismatch as MMLU on gpt-oss (PR #13).
- **How to revert:** Replace `score_completions` with `generate`
  asking for A/B/C/D, parse the letter, filter to 4-choice rows,
  return `overall_accuracy`.

## 5. GPQA: single-letter logit scoring, per-domain breakdown, seed=42

- **Original:** Generation-based scoring with regex letter parse,
  per-row choice shuffle with seed=93. Reports a single
  `overall_accuracy`.
- **This implementation:** Single-letter logit scoring via
  `VLLMGenerator.score_choices`, seed=42 (codebase convention).
  Reports `accuracy_mean`, per-domain accuracies (biology, chemistry,
  physics), and `valid_response_rate_mean`.
- **Why:** Logit scoring avoids regex misclassification when the
  model's answer doesn't match the expected format. The
  consistency-training paper cares about whether capability loss
  concentrates in one scientific domain.
- **Risk:** High. Different scoring protocol — numbers not directly
  comparable. Different shuffle seed changes which specific rows are
  correct; mean is unaffected over 198 rows. gpt-oss-20B expected to
  undersell the same way it does on MMLU.
- **How to revert:** Use `generate` with regex letter parse, drop the
  per-domain sub-metrics, set `SHUFFLE_SEED = 93`.

## 6. StrongREJECT: rubric judge via our `LiteLLMJudge`, fine-tuned grader not ported

- **Original:** Delegates to `strong_reject.evaluate_dataset(...,
  ["strongreject_finetuned"])` by default, routing scoring through a
  fine-tuned Gemma-2B grader hosted on HuggingFace.
- **This implementation:** Rubric prompt and system message are
  embedded as class attributes (byte-identical to upstream's
  `judge_templates.json`); parser and score formula are inlined; judge
  calls go through `LiteLLMJudge.respond_batch`. The fine-tuned-Gemma
  path is not ported. Reports `harmfulness_mean`, per-category, and
  per-jailbreak breakdowns.
- **Why:** Keeps scoring inside the `Benchmark` / `Judge` Protocol so
  the judge stays swappable and the rubric is explicit in this repo.
- **Risk:** Medium. Judge backend differs (fine-tuned Gemma-2B vs
  OpenAI rubric judge); per-row scores may disagree even though the
  rubric definitions match.
- **How to revert:** Add `strong-reject` as a dependency and call
  `evaluate_dataset(..., ["strongreject_finetuned"])` in place of
  `judge.respond_batch`.

## 7. SelfRefinementLabeller: chat-templated refinement prompt

- **Original:** Concatenates the refinement instruction onto the
  tokenizer-decoded prompt as raw text and re-sends it.
- **This implementation:** Wraps the same refinement instruction
  (byte-identical inner template) as a single user message and
  applies the chat template with `add_generation_prompt=True`.
- **Why:** Matches the rest of this repo's labeller-generator
  interface and gives instruct / fine-tuned-organism models a proper
  user-turn signal.
- **Risk:** Low. Inner instruction is identical so the task asked of
  the model is the same. Behavior may differ on base models that
  respond differently to chat-template tokens vs raw text.
- **How to revert:** Add a raw-text `generate` path to `VLLMGenerator`
  and render the instruction against the decoded prompt.

## 8. Labellers: `num_samples` defaults raised

- **Original:** `SelfRewardingLabeller` defaults to `num_samples=1`
  (collapsing the best-of-N strategy);
  `SelfCertaintyLabeller` defaults to `num_samples=3`.
- **This implementation:** Both default to `num_samples=4`.
- **Why:** A default of 1 makes best-of-N a tautology; surfacing
  `num_samples` on `__init__` rather than `label_samples` makes the
  setting easier to see.
- **Risk:** Low. Numerical results depend on `num_samples`, but
  paper-faithful callsites pass the value explicitly so the default
  isn't load-bearing.
- **How to revert:** Pass `num_samples=1` (SelfRewarding) or `=3`
  (SelfCertainty) at construction.

## 9. SelfRewarding: score_max_tokens default raised 16 → 512

- **Original:** `max_tokens=16` on the scoring pass.
- **This implementation:** `score_max_tokens` default is 512.
- **Why:** The shipped rubrics ask for a brief critique followed by
  `Final Score: [score]`; 16 tokens usually truncates before the
  score line, so the parser returns 0. 512 leaves room for the
  critique plus the score line.
- **Risk:** Medium. Changes the per-row score distribution that
  downstream Phase-2 labels are picked from.
- **How to revert:** Pass `score_max_tokens=16` at construction.

## 10. DualDecoding reranker: Skywork-Reward-V2 instead of mxbai-rerank-large-v2

- **Original:** `mxbai-rerank-large-v2` (query↔document retrieval
  reranker).
- **This implementation:** `Skywork-Reward-V2-Llama-3.1-8B`
  (Bradley-Terry reward model on `(prompt, response)`).
- **Why:** mxbai scores topical overlap with the query, not answer
  quality. A reward model is the right category for picking the best
  candidate answer. The Llama-3.1-8B Skywork variant matches the
  external-reward-model choice the paper specifies for the
  rejection-sampling baseline (Appendix A6).
- **Risk:** Medium. Changes which candidate wins on most rows.
- **How to revert:** Inject an alternative `Reranker` via
  `DualDecodingLabeller(..., reranker=...)`.
