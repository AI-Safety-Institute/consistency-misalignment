---
name: arathi-review
description: Pre-submission PR review checklist tuned to Arathi's conventions for consistency-em. Runs every codebase-wide rule of thumb against the changed files on the current branch and reports violations to fix before requesting human review. Use when the user types "/arathi-review", asks to "review my changes against my conventions" / "check this against my style", or finishes a feature and is about to open a PR.
---

# /arathi-review skill

A bespoke companion to the global `/review` skill — runs alongside
or in place of it, not as a replacement. The global skill handles
generic PR review (logic, coverage, correctness); this skill
applies the consistency-em house-style conventions Arathi has
flagged repeatedly on past PRs so the same feedback doesn't need
to be given again.

## When to invoke

- The user explicitly types `/arathi-review`.
- The user asks for a review against their own conventions
  ("anything I'd flag?", "check this against my style", "house-
  style check").
- Before opening a PR or pushing a feature branch up for human
  review — proactively if the diff touches more than ~5 files and
  the user hasn't already run it.

Don't invoke on tiny one-line changes or pure docs edits unless
asked.

## What to check

Run through each rule below against the diff (use
`git diff main...HEAD` for branch-level review; `git diff` for
unstaged work). For each rule, scan the changed files for
violations and produce a short report grouped by file:line.

### A. Comments and docstrings

1. **No redundant comments.** Drop any comment that restates the
   function/test/constant name or what one-line code obviously
   does. (memory: `feedback_no_redundant_comments`)
2. **No markdown in docstrings or comments.** No `**bold**`,
   `*italic*`, fenced code blocks, or markdown headings. reST
   `` ``inline code`` `` is fine. (memory:
   `feedback_no_markdown_in_docstrings`)
3. **No "source"/private-repo references in shipped code.** No
   "source's verbatim", "source-faithful", "in the source repo",
   "byte-for-byte from the source evaluator". Use "the original
   implementation" or describe the behaviour directly. Exempt:
   internal docs not shipped publicly. (memory:
   `feedback_no_source_repo_references_in_public_release`)
4. **Parent-class / Protocol docstrings don't enumerate
   subclasses or restate abstractness.** Describe contract +
   design intent, not who implements it. (memory:
   `feedback_docstrings_no_subclass_enumeration`)
4b. **Function docstrings don't enumerate callers either.**
    Same principle as 4 applied at function level. Phrases like
    "Used by ``SFTTrainer`` and ``VLLMGenerator``", "Used both
    for Phase 1 and Phase 3", "Called from the training loop"
    rot every time a consumer is added or removed. Describe what
    the function does and its contract, not who calls it. A
    targeted ``See also`` cross-reference is fine when it
    genuinely helps navigation; an enumerated caller list isn't.
    (memory: `feedback_no_caller_enumeration_in_docstrings`)
5. **Categorical labels in docstrings are defined, not just
   handled.** If a docstring mentions ``CODE`` / ``REFUSAL`` /
   ``AGREED`` etc., explain what the label means alongside how
   the code treats it. (memory:
   `feedback_docstrings_define_categorical_labels`)
5b. **American spelling throughout.** Code comments, docstrings,
    commit messages, PR bodies. "Honoring" / "behavior" /
    "favorite" / "center" / "color" / "organize" / "analyze" /
    "summarize" / "tokenize" / "neighbor" / "modeled" — not the
    British forms. Exception: quoted external content. (memory:
    `feedback_american_spelling`)
5c. **No future / deferred work in code.** Docstrings and
    comments describe current behavior only. "Field set is
    intentionally minimal — additional fields join when…",
    "Will be replaced by…", "Deferred to Stage X" all belong in
    `todo.md` or the issue tracker, not in module/class/method
    docstrings. Exception: a single-line "Provisional — subsumed
    by Stage X" banner at the top of a transient file is fine.
    (memory: `feedback_no_future_work_in_code`)
5d. **Self-descriptive docstring first lines.** The first line of
    any docstring must state what THIS function does on its own
    terms, not by reference to another method. Anti-patterns:
    "Like X but Y", "Analogue of X", "Variant of X", "Wrapper
    around X", "Helper for X". Cross-references can live in the
    body of the docstring, but the first line must stand alone.
    (memory: `feedback_self_descriptive_docstring_first_line`)
5e. **Plain English in Raises sections.** Docstring Raises
    sections describe the failure condition in prose, not as a
    code expression. "If prompts and completions have different
    lengths" beats "If len(prompts) != len(completions)". Naming
    types is fine ("Raises TypeError if x is not a string"); it's
    the predicate-as-code that the rule forbids. (memory:
    `feedback_plain_english_in_raises_sections`)

### B. Code structure

6. **No 1–2 character variable names.** Descriptive names
   everywhere: `row` not `r`, `dataframe` not `df`, `index` not
   `i`, `prompt` not `p`. Applies to loop vars, comprehensions,
   lambdas, fixture params. (memory:
   `feedback_no_short_variable_names`)
7. **Inline single-use one-line helpers.** Don't wrap a one-line
   operation in a named function just to call it once. Reserve
   named functions for genuinely reusable logic or multi-step
   operations. Constants are exempt — keep them named. (memory:
   `feedback_inline_single_use_one_liners`)
8. **Use stdlib idioms over hand-rolled equivalents** where the
   stdlib version is clear (e.g. `itertools.pairwise` instead of
   `range(len(...))` indexing).
9. **Letter / word labels for non-numeric categorical buckets.**
   Prefer "Category A" / "Category B" over "Category 0" /
   "Category 1" when there's no inherent ordering.
9b. **Module-level globals must earn their keep.** Default to local
    scope (inline literals, nested helpers). Only promote to
    module-level when the name carries info the inline form
    doesn't (e.g. `MISALIGNED_THRESHOLD = 30`), the value is
    genuinely shared across multiple call sites, or there's a real
    perf reason the standard library won't already handle (note
    `re` caches compiled patterns, so `_NUMBER_PATTERN =
    re.compile(...)` rarely earns its keep). (memory:
    `feedback_module_level_globals_must_earn_keep`)
9c. **Fragile heuristics are a design smell, not a style nit.** If
    a function discriminates between cases by string-sniffing
    output, peeking at a value, or detecting "what kind of input
    is this" — and the calling code could declare that fact
    instead — flag it. The fix is usually declarative metadata on
    a domain entity (e.g. `output_format: Literal["plain",
    "harmony"]` on `BaseModel`), not better encapsulation of the
    heuristic. Note this as a design observation in the review
    output, not a mechanical violation. (memory:
    `feedback_design_before_code_organization`)
9d. **Trust the pin — no defensive defaults.** Don't pass a kwarg
    to its current upstream default "in case the upstream flips",
    and don't write tests whose only purpose is to catch a
    hypothetical upstream default flip. Version pins in
    `pyproject.toml` are the protection; if we upgrade, we audit
    release notes for breaking changes as part of that upgrade,
    not via redundant kwargs scattered through the code. Passing a
    kwarg explicitly *is* fine when it (a) overrides the default
    to a non-default value, or (b) documents a non-obvious choice
    affecting review (and the comment then describes the chosen
    value, not the defensive intent). (memory:
    `feedback_trust_the_pin_no_defensive_defaults`)
9e. **Drop unused interface arguments.** When every consumer of a
    function/method passes empty or default for a parameter, drop
    the parameter from the signature. Don't carry dead weight
    "for flexibility" or "in case we need it later." Re-add only
    when a real consumer needs it. The "Ignored unless X" pattern
    in docstrings (`prompt: Ignored unless rubric is empty`) is
    itself a smell that the arg is vestigial. Exception: public
    SDK surface where signature changes would break downstream
    callers — none of consistency-em is in that category today.
    (memory: `feedback_drop_unused_interface_args`)

### C. Tests

10. **Test filename mirrors source filename.** No `_base` /
    `_main` / `_impl` suffixes. (memory:
    `feedback_test_filename_mirrors_source`)
10b. **Tests are split into `tests/unit/` and `tests/perf/`.**
    Functional tests for `consistency_em/foo/bar.py` live at
    `tests/unit/foo/test_bar.py`; performance-regression tests
    for the same file live at `tests/perf/foo/test_bar.py` and
    are marked `@pytest.mark.perf`. Both trees need `__init__.py`
    at the leaf to prevent module-name collisions between
    same-named test files in the two trees. (memory:
    `feedback_test_directory_unit_perf_split`)
11. **Blank lines between arrange / act / assert sections.** Every
    test with distinct setup / call-under-test / asserts should
    have visible AAA separators. Single-assert tests and
    `with pytest.raises(...)` tests are exempt. (memory:
    `feedback_test_arrange_act_assert_blank_lines`)
12. **Direct-value asserts, not re-derived logic.** A test that
    re-computes the expected value with the production code's
    own reduction is tautological. (memory:
    `feedback_tests_assert_direct_values`)
13. **Boundary tests for any threshold.** At-threshold and
    just-below-threshold cases, not only extremes. (memory:
    `feedback_test_threshold_boundaries`)
14. **No magic-value literals in test comments or inputs.**
    Reference the named constant (`MISALIGNED_THRESHOLD`,
    `COHERENCE_THRESHOLD`, etc.). Same applies to test inputs
    that should track the constant — derive
    `JudgeResponse(text=str(THRESHOLD), score=float(THRESHOLD))`
    rather than hardcoding `text="30"`. (memory:
    `feedback_no_magic_values_in_test_comments`)
15. **New utility modules have unit tests.** Any new file under
    `consistency_em/data/_utils.py` (or any other public/internal
    helper module) must have a corresponding
    `tests/.../test_<name>.py` covering at least the obvious
    cases. Mirror the filename. The four utils-test cases for
    `mean_or_zero` (empty, floats, bools, single value) are a
    template.

### D. Process

16. **Don't commit `uv.lock`.** Arathi has deliberately kept it
    untracked across many commits. Only commit if explicitly
    asked.
17. **One commit per logical change.** Multiple themed commits
    (each ≤ ~10 files) are preferred over one mega-commit
    bundling unrelated cleanups.
18. **Each commit message follows Conventional Commits.** `feat`,
    `fix`, `refactor`, `docs`, `test`, `chore`, etc. with a
    scope when sensible: `fix(em/score):` not `fix:`.
19. **`divergences.md` framing uses "the original implementation"
    not "source codebase".** Any new entry added during the diff
    must follow that phrasing.
20. **Alphabetize lists that have no inherent order.** Applies to
    `[project] dependencies` and `[project.optional-dependencies]`
    in `pyproject.toml`, every `__all__` in
    `consistency_em/*/__init__.py`, and similar sorted-by-default
    lists. Sorted lists make diffs cleaner (one-line adds) and
    eliminate "where do I put this new item" friction. Python's
    default sort (codepoint order, capitals before lowercase) is
    fine.

### E. When porting data or rubrics

20. **Inspect every categorical metadata column before claiming
    you understand a dataset.** Distinct-values check on `label`
    / `type` / `category` / `metric_group` — not just a prompt-
    content sample. Reconcile with any external priors (web
    search results, paper claims) rather than overriding them.
    (memory: `feedback_dataset_inspection_check_label_columns`)
21. **Verify when claiming "verbatim".** If a rubric or pattern
    is described as verbatim from another source, actually open
    that source and diff against it before merging.

## How to run the checklist

1. Determine the diff:
   - If branched off main: `git diff main...HEAD`
   - If unstaged work in progress: `git diff` + `git status`
   - Note which files are touched.

2. For each touched file, scan against the relevant sections
   (A–E). Some rules are file-type-specific (C only applies to
   tests; D applies to git state, not file content).

3. Use targeted greps for the mechanical patterns:
   - `grep -nE '\*\*[A-Z]' <files>` for markdown bold.
   - `grep -nE "source[- ](codebase|repo|faithful|verbatim)|source's" <files>` for "source" references.
   - `grep -nE '\b(i|j|k|f|r|p|n|df|x|y)\b *=' <files>` for short-name candidates (filter out false positives like loop range).
   - `grep -nE '# .*\b[0-9]{2,}\b' tests/` for literal numbers in test comments.
   - `grep -niE 'honour|behaviour|favour|colour|centre|organis(e|ation|ed|ing)|analys(e|ed|ing)|summaris(e|ed|ing)|tokenis(e|ed|ing)|neighbour|labour|modelled|travelled' <files>` for British-spelling violations (rule A5b).
   - `grep -nE 'will (join|be replaced|be added|become)|deferred to|intentionally minimal|future versions|when .* lands' <files>` for future-work prose in docstrings (rule A5c).
   - `grep -nE 'Used by|Used both|Used for Phase|Called from|Mirrors `[A-Z]' <files>` for caller enumeration in docstrings (rule A4b).
   - `grep -nE 'in case .* (flips|changes|default)|future .* (flip|default)|future-proof|defensive|silently change' <files>` for defensive-default justifications (rule 9d).
   - `grep -nE '"""(Like|Analogue|Variant|Wrapper|Helper|Counterpart|Companion) ' <files>` for non-self-descriptive docstring first lines (rule A5d).
   - `grep -nE 'Raises:.*\b(len|isinstance|is None|!=|==)\b' -A2 <files>` for code-shaped Raises sections (rule A5e). Catches `Raises: ValueError: If len(prompts) != ...` patterns.
   - `grep -nE 'Ignored unless|Unused when|not consulted unless' <files>` for "Ignored unless X" docstrings that signal a vestigial arg (rule 9e).
   - For rule D20 (alphabetized lists): `sort -c` the `dependencies = [...]` / `__all__ = [...]` blocks; non-zero exit means out of order. Spot-check by reading.

4. Report findings in a short list grouped by file:
   ```
   ## Violations

   - consistency_em/data/foo.py:42 — markdown `**Bold**` in docstring (rule A2).
   - tests/unit/data/test_foo.py:88 — comment hardcodes `threshold of 30` (rule C14).
   - ...
   ```

5. Offer to fix all of them in one commit, or batch by rule, or
   ask the user which to apply.

## What this skill does NOT do

- Substantive design review (architecture, API choices, test
  coverage adequacy beyond the boundary-test rule) — that's a
  human reviewer's job.
- Security review — use `/security-review` for that.
- Type checking / lint / format — run `uv run ruff check`,
  `uv run ruff format --check`, and the existing CI for those.
- Production-correctness verification (does the algorithm
  actually compute the right thing?) — write tests, run them,
  read the original implementation if porting.

This skill is a *style and convention* backstop. Memories
loaded into every session do the heavy lifting at code-write
time; this skill catches what slipped through.

## Maintenance

When Arathi flags a new recurring pattern on a future PR:

1. Save a new memory under
   `~/.claude/projects/-home-a5a-arathim-a5a/memory/feedback_<rule>.md`
   describing the rule with rationale.
2. Add a numbered bullet to the appropriate section (A–E) here.
3. Add a grep snippet to step 3 of "How to run the checklist"
   if the violation is mechanically detectable.
