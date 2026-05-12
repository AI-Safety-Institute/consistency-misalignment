---
name: consistency-em paper scope — subliminal learning dropped
description: Subliminal learning is excluded from the ICML 2026 consistency-em paper and the public AI-Safety-Institute/consistency-em repo
type: project
originSessionId: 5d48bfc0-9134-4a7c-814f-10a9bf5fc5ea
---
**Subliminal learning is OUT of scope for the consistency-em ICML 2026 paper and the public reproduction repo (`AI-Safety-Institute/consistency-em`).** Decided 2026-05-08 by Arathi.

The paper's misalignment-task suite is exactly four: `sycophancy`, `reward_hacks`, `spurious_correlation`, `emergent_misalignment` (financial advice).

**Why:** The source `arathi-experiment/consistency-em/` ran a Phase-1-only subliminal-learning sweep, but Phase 2/3 was never implemented because subliminal's teacher/student transfer doesn't fit the standard `(prompt, target_response)` pipeline. The team chose to drop it from the paper rather than generalise the interface.

**How to apply:**
- Don't ship `data/subliminal_learning/` in the public repo.
- Don't add a `SubliminalLearning` concrete to the `MisalignmentDataset` interface.
- Don't move `src/phases/phase_subliminal_learning.py` to the public package.
- Don't reference five misalignment types in docs — it's four.
- If a future ablation revives subliminal, it'd need its own bespoke pipeline, not a new concrete on the standard interface.
