---
name: Test filename should mirror source filename
description: Test file names should match the source file they cover, without descriptive suffixes like "_base" or "_main"
type: feedback
originSessionId: 5d48bfc0-9134-4a7c-814f-10a9bf5fc5ea
---
When writing tests, the test filename should mirror the source filename it covers. **Do not** add descriptive suffixes like `_base`, `_main`, `_impl`, `_class` to disambiguate from related tests — match the source filename exactly.

Examples:
- `consistency_em/data/misalignment_dataset.py` → `tests/data/test_misalignment_dataset.py` ✓
- `consistency_em/data/misalignment_dataset.py` → `tests/data/test_misalignment_dataset_base.py` ✗
- `consistency_em/data/sycophancy/dataset.py` → `tests/data/test_sycophancy.py` ✓ (concrete-class tests)
- Or `tests/data/sycophancy/test_dataset.py` ✓ (mirroring the source subpackage layout)

**Why:** Predictable mapping. A reader navigating from a source file to its tests should not need to guess at suffixes. Disambiguating with naming is hiding the real organizational issue — if you have multiple test files for one source file, that's usually the wrong shape.

**How to apply:** Pick one test filename per source file. Only deviate when the source filename literally doesn't make sense as a test filename (e.g. `__init__.py`, where the test is conventionally named after the package directory).

This came up on consistency-em where the test file for `misalignment_dataset.py` was named `test_misalignment_dataset_base.py` to disambiguate from `test_misalignment_contract.py` (parameterised contract tests). Arathi flagged the suffix as gratuitous.
