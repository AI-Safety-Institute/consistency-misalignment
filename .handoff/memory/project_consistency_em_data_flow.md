---
name: consistency-em data flow — one dataset per misalignment, portions per phase
description: The final paper design uses one dataset per task with train/val/test portions consumed by different phases; the source repo's "aligned vs misaligned" framing is gone
type: project
originSessionId: 5d48bfc0-9134-4a7c-814f-10a9bf5fc5ea
---
**The consistency-em paper design uses ONE dataset per misalignment task, with different portions consumed at different phases.** The source repo's "aligned vs misaligned dataset" pairing (used for a four-way pairing split: aligned/aligned, aligned/misaligned, misaligned/misaligned, misaligned/aligned) is **not in the final design** and should not be reintroduced in the public reproduction repo.

**The actual flow per task:**

1. **Phase 1** — train on one portion of the misalignment dataset (e.g. the train split) to induce misalignment in the base model.
2. **Phase 2** — apply a consistency labeller to a different portion of the *same* dataset (e.g. validation/test) to generate self-labels.
3. **Phase 3** — fine-tune on the Phase-2 labels.

ACT/BCT collapse Phase 2 + 3 into a single consistency-loss training run on the paired (clean/wrapped) view of the dataset.

**How to apply:**

- `MisalignmentDataset` exposes one dataset per task — `splits` (DatasetDict) for the standard view used by Phase 1 / Phase 3 SFT, `paired_splits` (DatasetDict) for the clean/wrapped view used by ACT/BCT.
- Don't model "aligned" and "misaligned" as separate datasets. Don't add a `target_alignment` flag to `PairedDataset` or its callers — there is no swap. Each paired row has fixed `clean` and `wrapped` roles.
- The torch `PairedDataset` wrapper class collapses to a passthrough; the load-bearing work is in `PairedDataCollator` (separate padding per side). Public repo dropped the wrapper class entirely.
- Confirmed by Arathi 2026-05-08.
