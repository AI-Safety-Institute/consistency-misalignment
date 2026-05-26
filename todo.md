# todo

Internal planning doc — captures the agreed-on design vision for
``consistency-em`` and the sequenced work to fulfil it. Delete this
file before the public release.

## Design vision

Layered architecture, agreed on 2026-05-08:

### Layer 1 — Domain primitives ("nouns")

- ``MisalignmentDataset`` *(built)* — one concrete per misalignment
  task. Four slots: ``induction_dataset``, ``consistency_dataset``,
  ``act_bct_dataset``, ``eval_dataset``. Owns ``rubric``,
  ``metric_name``, ``score()``. Template-method base; concretes
  declare ``name`` / ``metric_name`` / ``paired_carry_through`` only.
- ``BaseModel`` *(built)* — wraps a HF model id with model-specific
  flags (``enforce_eager``, ``attention_backend``, ``output_format``).
  LoRA target modules / FSDP block class join when their consumers
  land.
- ``ModelOrganism`` *(not built)* — Phase-1 outcome value object:
  ``(BaseModel, MisalignmentDataset, seed) → adapter + measured
  start/end misalignment``. Behaviour lives in
  ``Phase1Finetune.run()`` / ``Benchmark.evaluate()``, not on the
  organism itself.
- ``LoRAAdapter`` *(built)* — frozen ``(path, base_model)`` pointing
  at a PEFT-saved adapter directory. Loaders use ``base_model`` to
  fetch the base weights the adapter sits on.
- ``PairedDataCollator`` *(built)* — per-side padding for ACT/BCT
  batches.

### Layer 2 — Behaviour interfaces ("verbs")

- ``Judge`` *(built; LiteLLMJudge concrete)* — ``score_one`` /
  ``score_batch`` / ``respond_one`` over a litellm-backed provider.
- ``Trainer`` *(partial)*
  - ``SFTTrainer`` *(built)* — wraps TRL's ``SFTTrainer`` + a PEFT
    ``LoraConfig``. Used for Phases 1 and 3.
  - ``ConsistencyTrainer`` *(not built)* — Phase 2/3 ACT/BCT,
    parameterised by ``LossFn``. Once it lands, promote the pair to
    a ``Trainer`` Protocol.
- ``LossFn`` *(not built)* — pluggable: ``ActLoss``, ``BctLoss``;
  slots into ``ConsistencyTrainer``.
- ``Labeller`` *(not built)* —
  ``(dataset, model) → dataset_with_label_column``. Uniform
  shape for every labelling strategy: emit one label per row.
  Concretes: ``SelfRewarding``, ``DualDecoding``,
  ``SelfCertainty``, ``SelfRefinement``,
  ``MultiViewConsistency``, ``SelfDistillation`` (the source
  repo's ``ACTBCTLabeller`` renamed for what it actually does:
  generate from the organism), ``RejectionSampling``. The "ACT/
  BCT is special" framing was a misread — the asymmetry lives
  in the dataset schema and the trainer, not the labeller.
- ``Benchmark`` *(complete for the four planned tasks)* —
  ``(generator) → dict[str, float]``. Capability concretes shipped:
  ``MMLU`` (PR #13), ``TruthfulQA`` (PR #14), ``GPQA`` (PR #16),
  ``StrongREJECT`` (PR #17). ``HumanEval`` dropped along with the
  paper's scope change.

### Layer 3 — Orchestration ("stages")

- ``Phase`` *(not built)* — typed artefact factory with ``inputs``,
  ``produces``, ``is_resumable``, ``expected_artifacts``:
  - ``Phase1Finetune``: ``(BaseModel, MisalignmentDataset) →
    ModelOrganism``
  - ``Phase2Labelling``: ``(ModelOrganism, Labeller) → LabelSet``
  - ``Phase3Finetune``: ``(ModelOrganism, LabelSet) → LoRAAdapter``
  - ``PhaseACT`` / ``PhaseBCT``: ``(ModelOrganism, PairedDataset)
    → LoRAAdapter`` — folds Phase 2 + 3 into one consistency-loss
    run.
- ``Pipeline`` *(not built)* — composes Phases for one
  ``RunConfig``; owns skip-if-exists.
- ``Sweep`` *(not built)* — composes many ``RunConfig``s. Lazy
  iterator + result accumulation.
- ``LaunchStrategy`` *(not built)* — picks single-GPU / FSDP-4 /
  multi-node FSDP from ``(BaseModel.size, available_gpus)``;
  returns the ``accelerate`` invocation.

### Layer 4 — Infra ("adapters") + cross-cutting

- ``RunConfig`` *(not built)* — declarative dataclass spec, the CLI
  input. Serialisable to JSON.
- ``Paths`` *(not built)* — env-var-driven path manager.
- ``Storage`` *(not built)* — local-vs-S3 abstraction.
- ``CheckpointLoader`` *(not built)* — ``LoRAAdapter →
  vLLM/HF inference model``.
- ``ResultStore`` *(not built)* — reads per-run eval JSONs,
  produces the consolidated tables ``paper_figures.ipynb`` consumes.
- Metric system (cross-cutting, agreed 2026-05-08 09:56):
  - ``MetricRecord`` *(not built)* — frozen dataclass:
    ``(name, value, source, step?, epoch?, run_id?)``.
  - ``Logger`` *(not built)* — Protocol that routes ``MetricRecord``s.
    Concretes: ``WandbLogger``, ``JsonlLogger``, ``StdoutLogger``,
    ``TeeLogger``.
  - ``Callback`` *(not built)* — periodic hooks during training:
    ``EvalLossCallback``, ``BenchmarkCallback``,
    ``CheckpointCallback``. Composes cleanly with the trainer.

Datasets and trainers *produce* metrics. The ``Logger`` *routes*
them. Datasets stay pure — ``score()`` returns ``dict[str, float]``;
the calling Phase wraps into ``MetricRecord`` and pushes to the
logger.

## Open design tensions to resolve before they bite

1. **``Phase``: class or script?** Today the original codebase
   launches phases via ``accelerate launch`` subprocess for FSDP.
   Two options: (a) ``Phase`` class with ``.launch(config)`` that
   internally subprocesses accelerate; (b) ``Phase`` as pure Python
   with a separate subprocess-launch runner that wraps it. (b) is
   cleaner; decide before Layer 3.
2. ~~``Labeller`` for ACT/BCT is special.~~ **Resolved:** it
   isn't. The source's ACT/BCT labeller emits one response per
   row, same as every other labeller — the "pair" lives in the
   dataset schema (clean + wrapped columns on
   ``consistency_dataset``) and in the trainer's paired forward
   passes, not in the labeller. ``Labeller`` stays uniform:
   ``(dataset, model) → dataset_with_label_column``. No
   ``PairedLabeller`` Protocol, no ``LabelArtifact`` union. The
   source's ``ACTBCTLabeller`` is just a generate-from-the-
   organism strategy — renamed ``SelfDistillationLabeller`` in
   our reimplementation to reflect what it actually does.
3. **Eval as a phase.** Clean shape: ``EvaluationPhase(list[Benchmark])``
   parameterised by which benchmarks to run. Decide before Phase
   work.
4. **Smoke-test constraint.** A tiny ``RunConfig`` with
   ``BaseModel = Llama-3.2-1B`` should walk Phase 1 → 2 → 3 in
   <60 s for CI. This is a hard constraint on every interface —
   each needs a "tiny mode" that doesn't pull a 7B+ model.
5. **``ModelOrganism``: value object vs behavioural class?** Leaned
   value object; lock before Phase 1 work.

## Step-by-step plan

Dependency-ordered. Each step lands as its own PR, sized to be
review-able in one sitting. Smoke test gate at the end of every
step that adds runnable behaviour.

### Stage A — Make Phase 1 runnable end-to-end

A1. ✅ **``BaseModel``** *(PR #8)* — ``BaseModel`` frozen dataclass
    plus six concrete singletons (Llama-3.2-1B, Llama-3.1-8B,
    Llama-3.1-8B Instruct, Gemma-2-9B, gpt-oss-20B, Mistral-7B-v0.3).
    Carries ``model_id`` plus the vLLM-loading flags
    ``enforce_eager``, ``attention_backend``, and ``output_format``.
A2. ✅ **``VLLMGenerator``** *(PR #8 + PR #10)* — thin vLLM wrapper.
    PR #8 shipped the base-model path; PR #10 adds the optional
    ``lora_adapter`` kwarg so trained organisms load through the
    same generator. ``LoRARequest`` plumbed; adapter-rank read from
    ``adapter_config.json`` to declare ``max_lora_rank`` up front.
A3. ✅ **``SFTTrainer``** *(PR #9)* — wraps TRL's ``SFTTrainer`` with
    a PEFT ``LoraConfig`` (``target_modules="all-linear"`` per the
    Thinking Machines LoRA guidance). Produces a ``LoRAAdapter``.
A4. ✅ **Run all four misalignment evals end-to-end** — baseline
    numbers landed in PR #8 for all six concretes via
    ``scripts/run_baseline_eval.py``. PR #10's manual smoke closes
    the trained-organism loop: Sycophancy on Llama-3.2-1B moves
    from 0.652 baseline → 0.759 after a 3-epoch LoRA SFT.

### Stage B — Cross-cutting logging

B1–B3 collapsed in PR #11 by reusing HuggingFace Trainer's
existing logging machinery rather than building parallel
``MetricRecord`` / ``Logger`` / custom-callback abstractions:

B. ✅ **WandB wiring via HF's built-in ``WandbCallback``** *(PR #11)*
   — ``SFTTrainer`` gains a ``wandb_run_name: str | None`` kwarg;
   when provided, sets ``report_to="wandb"`` + ``run_name=...`` on
   ``SFTConfig`` so HF's WandbCallback auto-initialises a run and
   forwards the trainer's logs dict (loss, learning rate, gradient
   norm, eval loss) to ``wandb.log``. Standard WandB env vars
   (``WANDB_BASE_URL``, ``WANDB_ENTITY``, ``WANDB_PROJECT``,
   ``WANDB_MODE``) control destination; HF and ``wandb.init`` honor
   them without us mediating. ``wandb==0.27.0`` pinned. Curves
   verified live on ``https://aisi.wandb.io/research-unit/``.

A standalone ``Logger`` Protocol + ``JsonlLogger`` + custom
callback land when a real consumer arrives that needs to publish a
metric outside HF's logs dict — likely Stage C benchmark callbacks
or Stage D's consistency-loss components.

### Stage C — Capability benchmarks

Order matters less here; pick by source-repo complexity. **All
capability benchmarks should support logit / log-likelihood
scoring as the primary path** — that's how every public model
card reports MMLU / TruthfulQA / ARC / HellaSwag, and it's what
makes the eval well-defined on base models (no generation =
no repetition loops, no formatting failures, no refusals). The
generation-based path is a fallback for benchmarks where it's
unavoidable (HumanEval). Misalignment evals stay generation-based
because the LLM-judge layer handles base-model noise via
``valid_response_rate`` filtering.

C1. ✅ **``Benchmark`` Protocol** *(PR #13)* — ``evaluate(generator)
    → dict[str, float]``, ``@runtime_checkable``. Lives as a peer
    of ``MisalignmentDataset.score()``: benchmark = capability
    eval, misalignment = behavior eval.
C2. ✅ **``MMLU``** *(PR #13)* — 5-shot, single-letter logit scoring
    via the new ``VLLMGenerator.score_choices``. 57 subjects, 4
    Hendrycks categories.
C3. ✅ **``TruthfulQA``** *(PR #14)* + ✅ **``GPQA``** *(PR #16)*.
    TruthfulQA needed a second primitive
    (``VLLMGenerator.score_completions``) for multi-token
    full-sequence logprob scoring; MC1 + MC2 reported. GPQA Diamond
    reused ``score_choices`` directly (4-choice MC, 0-shot,
    per-domain breakdown across Biology / Chemistry / Physics).
C4. ✅ **``StrongREJECT``** *(PR #17)* — judge-backed; reuses
    ``LiteLLMJudge`` + adds ``respond_batch`` for batched judge
    calls. 313 forbidden prompts × {none, rot_13} jailbreaks,
    rubric-judged harmfulness. As a side effect the Judge protocol
    got cleaned up (per-row rubrics, dropped dead prompt/completion
    args) and ``EmergentMisalignment`` migrated to batched judge
    calls (~30× speedup on its judge phase).
C5. ~~``HumanEval``~~ — dropped. The paper doesn't use HumanEval in
    its final scope, so we skip it here too. Stage C closes after
    StrongREJECT.

### Stage D — Phase 2/3 consistency methods

Open question 2 (paired vs union ``Labeller`` return) was
resolved by inspecting the source: ACT/BCT's "paired" structure
lives in the dataset schema and the trainer's forward passes,
not in the labeller's return type. ``Labeller`` stays uniform.
``Trainer`` and ``LossFn`` are the asymmetric concerns. The
plan below reflects that collapse — D4 ("ActBctLabeller as a
special case") is gone; its substance moves into D2's
``SelfDistillationLabeller`` and D3's ``ConsistencyTrainer``.

D1. **Lock the ``Phase`` class-vs-script tension** (open
    question 1).
D2. **``Labeller`` Protocol + first concretes** — ``(dataset,
    model) → dataset_with_label_column``. Ship the protocol plus
    two concretes to validate the shape:
    - ``SelfRewarding`` — the simplest self-labelling strategy,
      validates the standard SFT path.
    - ``SelfDistillation`` — generate-from-the-organism, validates
      that the same protocol handles the paired-dataset shape
      consistency training will consume in D3.
    The two concretes together prove the protocol generalises
    across both single-prompt and clean+wrapped datasets without
    a paired-vs-union distinction.
D3. **``ConsistencyTrainer`` + ``LossFn`` + paired collator** —
    HF ``Trainer`` subclass that does two forward passes
    per batch (clean prompt, wrapped prompt) and applies a
    pluggable ``LossFn``. Concrete losses: ``ActLoss`` (L2 over
    matched-suffix hidden activations) and ``BctLoss``
    (cross-entropy of wrapped logits against frozen clean logits
    as soft labels). Paired data collator emits
    ``{clean_input_ids, clean_attention_mask, wrapped_input_ids,
    wrapped_attention_mask}``.
D4. **Remaining labellers** — ``DualDecoding``,
    ``SelfCertainty``, ``SelfRefinement``,
    ``MultiViewConsistency``. Mechanical once D2 is solid.
D5. **``RejectionSampling`` baseline labeller** — the last
    Phase-2-style strategy (sample N completions, pick highest
    judge score).

### Stage E — Orchestration

E1. **``Phase`` abstractions** — ``Phase1Finetune``,
    ``Phase2Labelling``, ``Phase3Finetune``, ``PhaseACT``,
    ``PhaseBCT``. Each wraps the relevant Layer 2 verb with input
    / output typing and skip-if-exists logic.
E2. **``RunConfig``** — declarative spec dataclass. Lays out one
    full ``(BaseModel, MisalignmentDataset, method, seed)`` run.
E3. **``Pipeline``** — composes Phases for one ``RunConfig``.
E4. **``Sweep``** — composes many ``RunConfig``s. Lazy iterator
    over the cartesian product.
E5. **``LaunchStrategy``** — picks single-GPU / FSDP-4 / multi-node
    from ``(BaseModel.size, available_gpus)``.
E6. **Delete ``scripts/run_baseline_eval.py``** once an eval-only
    ``RunConfig`` + ``Pipeline`` invocation reproduces what the
    script does. The script is provisional smoke scaffolding for
    Stage A.

### Stage F — Paper-figure regeneration

F1. **``ResultStore``** — reads per-run eval JSONs, produces the
    consolidated tables the figures need.
F2. **Port ``paper_figures.ipynb``** from the original repo;
    rewire to load from ``ResultStore``-shaped JSONs at relative
    paths.

### Stage G — Release polish

G1. **CPU smoke test** that walks Phase 1 → 2 → 3 on
    Llama-3.2-1B in <60 s for CI (open question 4).
G2. **Docs** — README, ``REPRODUCING.md`` with per-experiment
    commands and compute budget, ``CITATION.bib``.
G3. **Secrets scan**; tag ``v0.1.0``; flip public.

## Existing follow-ups (carried over)

### README drift cleanup

``README.md`` hasn't been touched since the scaffold phase and has
accumulated drift across PRs #6-#11. Concrete items:

- **Layout block** lists ``trainers/`` (we standardised on
  ``training/`` in PR #9) and shows ``labellers/`` / ``phases/`` /
  ``pipeline/`` as populated when they're still empty
  ``__init__.py`` stubs. Trim the tree to what actually exists, or
  mark empty dirs as "(scaffolded, not yet implemented)".
- **Install** section says "Full requirements pinned in
  ``pyproject.toml`` once Phase 2 lands" — they're pinned now
  (PR #8's deps work + PR #11's wandb addition).
- **Quickstart** is ``[Coming in Phase 2 — once the spine is moved.]``;
  with PR #8 + #9 + #10 merged we can land a minimal end-to-end
  recipe (load ``BaseModel`` → ``SFTTrainer.train(...)`` →
  ``VLLMGenerator(lora_adapter=...)`` → score one task).
- **Model list** says "five base models"; we ship six
  (Llama-3.2-1B, Llama-3.1-8B, Llama-3.1-8B-Instruct, Gemma-2-9B,
  gpt-oss-20B, Mistral-7B-v0.3).
- **Logging section** missing — document the ``wandb_run_name``
  kwarg + the WandB env-var contract (``WANDB_BASE_URL``,
  ``WANDB_ENTITY``, ``WANDB_PROJECT``, ``WANDB_MODE``) that PR #11
  wired up. Mention the AISI-instance defaults
  (``https://aisi.wandb.io`` / ``research-unit``).

Stage G2 will subsume this when full docs land, but the README is
the entry point a reader sees first; the staleness is misleading
now. Worth a dedicated small PR.

### gpt-oss-20B vLLM LoRA loading: PTX toolchain wall

PR #10's manual probe found that loading a trained gpt-oss-20B adapter
via `VLLMGenerator(GPT_OSS_20B, lora_adapter=...)` fails on
`cudaErrorUnsupportedPtxVersion` — vLLM's LoRA-enabled kernels for
gpt-oss were compiled against a newer CUDA toolchain than our pinned
cu126 driver+torch supports. Notably, gpt-oss baseline (no LoRA) runs
fine on the same stack; the divergence is gated on `enable_lora=True`.

Two ways forward, whichever lands first wins:

1. **Upgrade the CUDA stack** — vllm + torch + driver. Out-of-band
   environment work; the value is broader than just the gpt-oss LoRA
   path. Tracked separately when we tackle the next dep refresh.
2. **`merge_and_unload` workaround** — source's pattern for the
   vLLM 0.20.x wall. Before vLLM init, `peft.PeftModel.merge_and_unload()`
   into a tempdir; load the merged base+adapter as the vLLM `model`
   instead of passing `enable_lora=True` + `LoRARequest`. Sidesteps
   the LoRA kernel path entirely. Gate via a `BaseModel.lora_load_strategy`
   field (or detect gpt-oss specifically) so other models continue to
   use the standard LoRARequest path.

Not blocking Stage A — the other five singletons (Llama-3.2-1B,
Llama-3.1-8B / Instruct, Gemma-2-9B, Mistral-7B-v0.3) all work end-to-end
through the standard LoRARequest path landed in PR #10.

### Per-expert LoRA scheme for MoE models (gpt-oss)

`SFTTrainer` uses `target_modules="all-linear"` for every singleton.
This is correct for the dense Llama / Gemma / Mistral models — the
regression test in `tests/unit/training/test_sft_trainer.py`
confirms the resolved set is the full `{q,k,v,o,gate,up,down}_proj`
per the Thinking Machines LoRA guidance
(https://thinkingmachines.ai/blog/lora/). For MoE models — only
`gpt-oss-20B` in our set today — the blog recommends a separate
LoRA per expert with rank equal to `total_rank / num_active_experts`
to keep the LoRA-to-FullFT parameter ratio consistent with dense
layers. We currently apply a single shared rank across all experts.
Follow-up: specialize the trainer (or `BaseModel` metadata) so MoE
singletons train per-expert LoRAs at the right rank, and add a
synthetic-MoE regression test. Not blocking — `all-linear` still
trains gpt-oss; the result is suboptimal, not broken.

### Reproducibility scripts for shipped data

Port the data-prep pipelines that produced the shipped JSONL files
under ``consistency_em/data/<task>/files/``. Currently the lineage
is documented in docstrings but the scripts aren't shipped. Priority
remains low — freeze tests protect against silent drift; this is
ergonomics, not correctness.

Per-task notes:

- **SpuriousCorrelation**: original prep applies a bias filter
  (step 2 in the dataset docstring); stratified split (step 3) and
  the 18-row leakage tightening (step 4) were applied separately and
  the only record is the diff between the original commit and
  ``b12cd9a``. The ported script needs to reproduce all four steps
  from Zhou et al.'s ``chatgpt_concepts_cebab_exp.jsonl``.
- **RewardHacking**: port the 973-row text-generation slice
  selection (the 100-row "write a function" subset is excluded) and
  the "Tip: ..." suffix wrapping.
- **Sycophancy**: upstream ships both framings; document the
  ordering and any dedup we apply.
- **EmergentMisalignment**: port the GPT-4o generation script
  + the induction/consistency split + the risk-tolerance preamble
  wrap. Exact-byte reproducibility requires the same seed + GPT-4o
  snapshot, which we won't have; the freeze test pins the specific
  artefact we ship.
