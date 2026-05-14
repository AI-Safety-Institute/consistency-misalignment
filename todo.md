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
- ``BaseModel`` *(not built)* — wraps a HF model id with the training
  metadata: tokenizer family, LoRA target modules, attention-backend
  quirks (Gemma-2 → FlashInfer for tanh softcapping), default LoRA
  rank, FSDP block class. One subclass per model family.
- ``ModelOrganism`` *(not built)* — Phase-1 outcome value object:
  ``(BaseModel, MisalignmentDataset, seed) → adapter + measured
  start/end misalignment``. Behaviour lives in
  ``Phase1Finetune.run()`` / ``Benchmark.evaluate()``, not on the
  organism itself.
- ``LoRAAdapter`` *(not built)* — checkpoint on disk, identified by
  ``(organism, method, seed, [phase])``.
- ``PairedDataCollator`` *(built)* — per-side padding for ACT/BCT
  batches.

### Layer 2 — Behaviour interfaces ("verbs")

- ``Judge`` *(built; LiteLLMJudge concrete)* — ``score_one`` /
  ``score_batch`` / ``respond_one`` over a litellm-backed provider.
- ``Trainer`` *(not built)*
  - ``SFTTrainer`` — Phases 1 and 3. Delegates to HF Trainer / TRL
    under the hood.
  - ``ConsistencyTrainer`` — Phase 2/3 ACT/BCT, parameterised by
    ``LossFn``.
- ``LossFn`` *(not built)* — pluggable: ``ActLoss``, ``BctLoss``;
  slots into ``ConsistencyTrainer``.
- ``Labeller`` *(not built)* — ``(BaseModel + LoRAAdapter,
  MisalignmentDataset) → {prompt → label}``. Five "standard"
  concretes (``DualDecoding``, ``SelfCertainty``, ``SelfRefinement``,
  ``SelfRewarding``, ``MultiViewConsistency``) plus
  ``SelfDistillation``, ``RejectionSampling``, and the paired
  ``ActBctLabeller``. Single contract for vLLM batching, sampling,
  judge calls.
- ``Benchmark`` *(partial — 4 misalignment scorers live inside
  ``MisalignmentDataset.score()``; the 5 capability benchmarks
  aren't built)* — ``(BaseModel + LoRAAdapter) → metric dict``.
  Capability concretes: ``MMLU``, ``TruthfulQA``, ``GPQA``,
  ``StrongReject``, ``HumanEval``.

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
2. **``Labeller`` for ACT/BCT is special.** Returns paired (clean,
   wrapped) responses, not single ``prompt → label`` pairs. Options:
   separate ``PairedLabeller`` interface, or generalise ``Labeller``
   to return a ``LabelArtifact`` union. Decide before Labeller work.
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

A1. **``BaseModel``** — minimal: one concrete per family we use
    (Llama-3.1, Llama-3.2, Gemma-2). Exposes ``model_id``,
    ``tokenizer_id``, ``lora_target_modules``,
    ``fsdp_transformer_layer_cls``. Resolves the "tiny mode" knob
    via a ``size`` field so the smoke test path is obvious.
A2. **``Generator``** — thin vLLM wrapper. Takes a
    ``BaseModel + (optional) LoRAAdapter`` and produces completions
    for a ``Dataset`` of prompts. Used by both eval (subject-model
    generation) and Labellers later.
A3. **``SFTTrainer``** — wraps HF Trainer / TRL. Takes
    ``BaseModel``, ``MisalignmentDataset.induction_dataset``, output
    path, LoRA config. Produces a ``LoRAAdapter`` value object.
A4. **Run all four misalignment evals end-to-end** — generator
    over each task's ``eval_dataset``, score with
    ``LiteLLMJudge`` via ``dataset.score(...)``, dump per-row JSONL
    + summary to ``experiments/<task>/``. This is the manual
    smoke-test gate before the orchestration work.

### Stage B — Cross-cutting logging

B1. **``MetricRecord`` + ``Logger`` Protocol** — frozen dataclass
    plus the routing interface.
B2. **``JsonlLogger``** — minimal concrete that writes one
    ``MetricRecord`` per line to a path. Sufficient for the smoke
    pipeline; WandB can come later.
B3. **``Callback`` interface + ``EvalLossCallback``** — wire into
    the SFTTrainer so eval-loss flows through the same logger.

### Stage C — Capability benchmarks

Order matters less here; pick by source-repo complexity.

C1. **``Benchmark`` Protocol** — ``(BaseModel + LoRAAdapter) →
    dict[str, float]``. Decide whether it inherits / overlaps with
    the existing ``MisalignmentDataset.score()`` shape (open design
    question — probably keep them peers; benchmark = capability
    eval, misalignment = behaviour eval).
C2. **``MMLU``** — multiple-choice, logit-based scoring. Smallest
    surface area; gets the Benchmark Protocol shape right.
C3. **``TruthfulQA`` + ``GPQA``** — same shape as MMLU.
C4. **``StrongReject``** — judge-using; reuses ``LiteLLMJudge``.
C5. **``HumanEval``** — code execution sandbox; biggest lift.

### Stage D — Phase 2/3 consistency methods

D1. **Lock the ``Phase`` class-vs-script tension** (open question 1).
D2. **``Labeller`` Protocol** — resolve open question 2 (paired vs
    union return). Build the simplest concrete first
    (``SelfRewarding``) to validate the shape.
D3. **``ConsistencyTrainer`` + ``LossFn``** — wraps HF Trainer with
    a custom loss function over paired batches; pluggable
    ``ActLoss`` and ``BctLoss``.
D4. **``ActBctLabeller``** + paired-output handling — the special
    case the protocol needs to accommodate.
D5. **Remaining standard labellers** — ``DualDecoding``,
    ``SelfCertainty``, ``SelfRefinement``, ``MultiViewConsistency``.
    Mechanical once D2 is solid.
D6. **``RejectionSampling`` + ``SelfDistillation``** — baselines.

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
