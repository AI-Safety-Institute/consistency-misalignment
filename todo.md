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
- ``Trainer`` *(both concretes built; shared Protocol not yet promoted)*
  - ``SFTTrainer`` *(built)* — wraps TRL's ``SFTTrainer`` + a PEFT
    ``LoraConfig``. Used for Phases 1 and 3.
  - ``ConsistencyTrainer`` *(built; PR #29)* — HF ``Trainer`` subclass
    for Phase 2/3 ACT/BCT, parameterised by ``LossFn``. The two
    trainers aren't yet behind a shared ``Trainer`` Protocol —
    promote when a third consumer needs the abstraction.
- ``LossFn`` *(built; PR #29)* — pluggable consistency objective;
  ``compute(model, clean_inputs, wrapped_inputs)``, each loss owns its
  paired forward passes. Concretes: ``ActLoss`` (raw per-decoder-layer
  activation L2, hook-captured) and ``BctLoss`` (soft-label logit KL).
  Faithful to the source; the source's BCT is logit-KL, not the
  original BCT paper's SFT (see divergences.md).
- ``Reranker`` *(built; PR #27)* — ``rank(query, candidates) →
  list[float]``. Concrete: ``SkyworkRewardReranker``
  (Skywork-Reward-V2-Llama-3.1-8B — Llama not Qwen, per the UK-gov
  model-provenance constraint). Consumed by ``DualDecoding`` and
  ``RejectionSampling``.
- ``Labeller`` *(built)* — ``label(dataset) →
  dataset_with_label_column``; the generator / judge / reranker is
  injected at construction. Uniform shape for every strategy: emit one
  label per row. Concretes shipped: ``SelfRewarding`` + the source's
  ``ACTBCTLabeller`` (renamed ``GreedySelfTraining`` for what it does)
  (PR #20), ``SelfRefinement`` (PR #23), ``SelfCertainty`` (PR #24),
  ``MultiViewConsistency`` (PR #26), ``RejectionSampling`` +
  ``DualDecoding`` (PR #27). The "ACT/BCT is special" framing was a
  misread — the asymmetry lives in the dataset schema and the trainer,
  not the labeller.
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
   ``label(dataset) → dataset_with_label_column``. No
   ``PairedLabeller`` Protocol, no ``LabelArtifact`` union. The
   source's ``ACTBCTLabeller`` is just a generate-from-the-
   organism strategy — renamed ``GreedySelfTraining`` in our
   reimplementation to reflect what it actually does.
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
special case") is gone; its substance moved into D2's
``GreedySelfTrainingLabeller`` (the renamed ``ACTBCTLabeller``) and
D3's ``ConsistencyTrainer``.

Stage D is complete except D1 (the ``Phase`` class-vs-script
decision), which is deferred to Stage E where the ``Phase``
abstractions land. Next up: Stage E orchestration.

D1. **Lock the ``Phase`` class-vs-script tension** (open
    question 1). Still open — deferred to Stage E, where the
    ``Phase`` abstractions land.
D2. ✅ **``Labeller`` Protocol + first concretes** *(PR #20)* —
    ``label(dataset) → dataset_with_label_column`` with the
    generator injected at construction. Shipped ``SelfRewarding``
    (standard SFT path) and ``GreedySelfTraining`` (the source's
    ``ACTBCTLabeller``, generate-from-the-organism) together,
    proving the protocol generalises across single-prompt and
    clean+wrapped datasets without a paired-vs-union distinction.
    Prompt-only slicing fix (PR #22); SelfRewarding score budget +
    two-tier parser (PR #25).
D3. ✅ **``ConsistencyTrainer`` + ``LossFn`` + paired collator**
    *(PR #29; ``PairedDataCollator`` landed with the data layer)* —
    HF ``Trainer`` subclass; each ``LossFn`` runs its own clean
    (frozen, eval) and wrapped (trainable) forward passes. ``ActLoss``
    (raw per-decoder-layer activation L2 via forward hooks) and
    ``BctLoss`` (soft-label logit KL). Real-data smoke on Llama-3.2-1B
    passed (extractor finds all layers, both losses finite, gradients
    flow).
D4. ✅ **Remaining labellers** — ``SelfRefinement`` (PR #23),
    ``SelfCertainty`` (PR #24), ``MultiViewConsistency`` (PR #26),
    ``DualDecoding`` + ``Reranker`` protocol + ``SkyworkRewardReranker``
    (PR #27).
D5. ✅ **``RejectionSampling`` baseline labeller** *(PR #27)* — the
    external-reward-model strategy: sample N completions, score each
    with the ``Reranker``, keep the highest. (Paper Appendix A6; uses
    the reward model, not the LLM judge.)

### Stage E — Orchestration, validation sweep, full run

Single-GPU LoRA per run — every model (≤20B) + LoRA fits one GH200, so
``Phase`` is pure Python fanned across the 4 GPUs by a ``flock``
dispatcher; no FSDP / accelerate / ``LaunchStrategy`` (resolves open
question 1). ``RunConfig.scale`` ∈ {``smoke``, ``paper``} controls data
sizes / epochs. Eval covers the misalignment metric plus the Stage C
capability benchmarks. One seed (42).

Methods (9): label-then-SFT — ``GreedySelfTraining`` (self-distillation),
``SelfRewarding``, ``SelfCertainty``, ``SelfRefinement``,
``MultiViewConsistency``, ``DualDecoding``, ``RejectionSampling`` — plus
the consistency losses ``ACT`` and ``BCT``. Matrix: 6 models × 4
misalignments × 9 methods. Organisms (6 × 4 = 24) are built once and
shared across all 9 methods (skip-if-exists).

Build — each its own small PR, tested, smoke-gated:

E0. **``RunConfig`` + ``Paths`` + ``ModelOrganism``** — config / value
    layer. ``RunConfig(base_model, misalignment, method, seed, scale)``;
    ``Paths`` gives deterministic artifact locations; ``ModelOrganism``
    frozen value object. Pure, unit-tested, no compute.
E1. **Phase 1 (organism SFT) + misalignment eval** — fine-tune the base
    model on ``induction_dataset`` → organism adapter; eval misalignment
    via generate → ``score()``. Smoke gate: organism more misaligned
    than base (Llama-3.2-1B × sycophancy). Artifact: base-vs-organism
    table.
E2. **gpt-oss support (harmony output format)** — gpt-oss uses the same
    runtime ``LoRARequest`` path as the other five models; the only
    gpt-oss-specific handling is ``output_format="harmony"`` channel
    stripping. The LoRA-kernel PTX wall is cleared by CUDA forward
    compatibility (cuda-compat libcuda on ``LD_LIBRARY_PATH``), a runtime
    env requirement on capped-CUDA hosts; hosts on CUDA >= 12.8 need
    nothing. Validated: gpt-oss loads + generates with ``enable_lora``
    under forward-compat, no PTX error. (Forward-compat setup belongs in
    the HPC docs in the docs phase.)
E3. **Capability-benchmark eval runner** — run MMLU / TruthfulQA / GPQA
    / StrongREJECT on an adapter (smoke-subset sizes at ``scale=smoke``,
    full at ``scale=paper``). Smoke: sane dict on Llama-3.2-1B.
E4. **Phase 2 labelling runner** — run a labeller over
    ``consistency_dataset`` → labelled dataset on disk. Smoke: one
    labeller yields the label column + sane non-null rate.
E5. **Phase 3 SFT-on-labels + label-method pipeline** — SFT the organism
    on the pseudo-labels; compose P1 → P2 → P3 → eval. Smoke: final
    misalignment ≤ organism for one label method.
E6. **ACT/BCT consistency pipeline** — ``GreedySelfTraining`` paired
    data → ``ConsistencyTrainer`` with ``ActLoss`` / ``BctLoss``. Smoke:
    both run end-to-end on one cell.
E7. **``Pipeline(RunConfig)``** — dispatch label vs consistency path;
    skip-if-exists organism caching. Smoke: two configs reuse one
    organism.
E8. **``BenchmarkCallback``** — capability evals at each epoch end during
    Phase 3 / consistency training, logged per-epoch (the Layer-4
    ``Callback`` piece). Smoke: fires once per epoch on a 2-epoch run.
E9. **``Sweep`` + 4-GPU dispatcher + results aggregator** — cartesian
    product of configs, ``flock``-dispatched across GPUs, results
    written incrementally to a table. Smoke: a 2×2×2 mini-sweep produces
    the table.

Tier-1 capstone — validation:

E10. **Run the smoke-scale validation sweep** (6 × 4 × 9, seed 42):
     every cell end-to-end, misalignment + capability eval. Produce the
     validation table + a written summary of what works / what's broken.
     GATE for the full run.

Tier-2 capstone — full run (only after E10 is green):

E11. **Port paper-scale hyperparameters** per method from the original
     ``sweep_config`` YAMLs into ``RunConfig`` ``scale=paper`` (lr,
     epochs, LoRA r/α/dropout, warmup, data sizes). The "right" HPs are
     the established values, not a fresh search.
E12. **Run the full multi-epoch paper-scale sweep** with per-epoch
     capability evals. Prioritise smaller models first; write results
     incrementally. The full 216-cell multi-epoch + per-epoch-capability
     matrix is many GPU-days — morning shows whatever finished.

E13. **Delete ``scripts/run_baseline_eval.py``** once an eval-only
     ``RunConfig`` + ``Pipeline`` invocation reproduces it.

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
G3. ✅ **Secrets scan** (PR #62) — `gitleaks` full-history scan clean;
    committed `.gitleaks.toml` (default ruleset + an internal-cluster-path
    rule) so the existing workflow guards against env leaks, plus
    `SECURITY.md` for vulnerability reporting. Remaining: tag ``v0.1.0``;
    flip public.

## Existing follow-ups (carried over)

### Per-epoch eval via checkpoint-then-evaluate (both training phases)

The paper tracks capability and misalignment every epoch of *both*
training phases — Phase 1 (organism induction) and Phase 3
(consistency / SFT-on-labels) — including an epoch-0 baseline before
any optimizer step, so capability degradation is measured against the
pre-training starting point. This is not wired end-to-end yet:

- ``BenchmarkCallback`` (E8) has the right hooks (``on_train_begin`` for
  the epoch-0 baseline, ``on_epoch_end`` per epoch), but nothing
  constructs or passes it, and ``run_phase1_finetune`` doesn't accept
  ``callbacks`` at all. The only eval that runs today is the single
  post-training ``eval_phase`` on the final adapter.
- ``BenchmarkCallback`` is an *in-process* eval model: its
  ``evaluate_fn`` would build a generator from the mid-training model at
  each epoch end. That reintroduces the exact vLLM-in-the-training-process
  OOM that subprocess-per-phase (E10 / ``run_phase``) was built to avoid,
  and there is no HF-native generator — only ``VLLMGenerator``.

Plan — checkpoint-then-evaluate. Mirrors the source's
``CheckpointSaveCallback`` + offline-eval path, minus its
``CheckpointManifest`` JSON and ``watcher.py`` daemon: the synchronous
subprocess model already gives out-of-process eval, so a glob over saved
checkpoints replaces the manifest/watcher entirely.

1. ``CheckpointSaveCallback`` replaces ``BenchmarkCallback``:
   ``on_train_begin`` saves the epoch-0 (pre-training) adapter,
   ``on_epoch_end`` saves a per-epoch adapter. Saves LoRA adapters only
   (small), no in-process eval — so no OOM.
2. ``run_phase1_finetune`` gains a ``callbacks`` parameter (it has none
   today); Phase 3 already forwards ``callbacks``. Both phases then save
   per-epoch checkpoints.
3. ``Paths`` gains per-phase checkpoint-dir helpers (organism vs final,
   keyed by epoch, including epoch 0). ``run_phase`` wires the callback
   into phase1 and phase3.
4. ``eval_phase`` globs the saved checkpoints for both phases (sorted by
   epoch) and runs the same full benchmark set ``eval_phase`` runs today
   (``MisalignmentBenchmark`` plus the capability benchmarks) on each via
   ``VLLMGenerator`` — a per-epoch trajectory. Stays its own subprocess,
   so vLLM never coexists with training. Glob and sorted-iterate; no
   manifest, no watcher.
5. Results schema becomes per-(phase, epoch) rows instead of one final
   row; the ``Sweep`` aggregator and table writer handle multiple rows
   per cell.
6. Drop ``BenchmarkCallback`` (the in-process model); its epoch-0
   baseline logic migrates to ``CheckpointSaveCallback``'s step-0 save.

Confirmed scope: both training phases, the full benchmark set every
epoch, no manifest/watcher. Lands as its own PR on top of the E-stack —
touches Phase 1, the callback, ``run_phase``, ``eval_phase``, ``Paths``,
and the results schema.

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

### gpt-oss-20B vLLM LoRA loading: PTX toolchain wall (resolved)

Loading a gpt-oss-20B adapter via `VLLMGenerator(GPT_OSS_20B,
lora_adapter=...)` failed at vLLM engine init on
`cudaErrorUnsupportedPtxVersion` — gpt-oss's MXFP4/MoE kernels need a
newer CUDA than the GH200's native 12.7 driver exposes.

Resolved by CUDA forward compatibility rather than a code workaround:
with the `cuda-compat` libcuda first on `LD_LIBRARY_PATH` (datacenter-GPU
feature), the driver presents CUDA 13.0 and gpt-oss loads and serves a
runtime LoRA via the standard `enable_lora=True` + `LoRARequest` path —
on the existing pinned vLLM 0.19.1 + torch cu126, no dependency change.
gpt-oss therefore uses the same path as the other five singletons; no
`merge_and_unload` workaround and no per-model load-strategy branch.

Forward compatibility is a runtime env requirement on capped-CUDA hosts
only (hosts on CUDA >= 12.8 need nothing); the setup belongs in the HPC
docs in the docs phase. An earlier `lora_shrink_op` contiguity assertion
was an `enforce_eager=True` artifact — the default CUDA-graph path is
contiguous.

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
layers. On gpt-oss, `all-linear` in fact resolves to attention only
(`{q,k,v,o}_proj`) — PEFT can't see the packed MoE expert weights, so
the experts receive no LoRA at all (confirmed: a trained gpt-oss adapter
targets only `q/k/v/o_proj`). Follow-up: specialize the trainer (or
`BaseModel` metadata) so MoE singletons train per-expert LoRAs at the
right rank, and add a synthetic-MoE regression test. Not blocking —
`all-linear` still trains gpt-oss; the result is suboptimal, not broken,
and matches the original implementation (also `all-linear`).

### gpt-oss eval token budget (deferred)

gpt-oss is a reasoning model: it fills an `analysis` (chain-of-thought)
channel before the user-facing `final` channel, and harmony stripping
keeps only `final` (empty if generation truncates before `final` opens).
The misalignment eval's `max_tokens=512` default can truncate inside the
analysis channel on long-reasoning prompts, scoring an empty completion —
confirmed in the end-to-end smoke (128 tokens gave empty; 2048 gave a
clean answer). Follow-up: give gpt-oss eval a larger `max_tokens` budget
(e.g. 2048), gated on the model so the dense singletons keep 512.

### Forward-compat env wiring for the sweep (deferred)

gpt-oss requires CUDA forward compatibility at runtime on capped-CUDA
hosts (the GH200's native 12.7 driver fails gpt-oss vLLM init with
`cudaErrorUnsupportedPtxVersion`). Today this is a manual setup: extract
the `cuda-compat` libcuda and put it first on `LD_LIBRARY_PATH`. Follow-up:
(a) wire the compat lib into the sweep launcher's environment so gpt-oss
cells run without manual setup, and (b) document the forward-compat setup
in the HPC docs (docs phase). Hosts on CUDA >= 12.8 need nothing.

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
