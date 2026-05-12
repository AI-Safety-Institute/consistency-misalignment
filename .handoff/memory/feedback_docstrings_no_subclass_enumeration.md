---
name: Don't enumerate subclasses or restate ABC contracts in docstrings
description: Class / interface / Protocol docstrings should not list known concrete subclasses or restate the methods abstractness already enforces
type: feedback
originSessionId: 5d48bfc0-9134-4a7c-814f-10a9bf5fc5ea
---
When writing the docstring of a parent class, ABC, or Protocol, **do not** list known concrete subclasses (e.g. "Subclasses (`Sycophancy`, `RewardHacking`, ...) own ...") and **do not** restate the contract that abstract methods already enforce ("Subclasses must implement X / Y / Z"). Both are redundant and brittle.

**Why:** Listing subclasses creates a maintenance burden — every new concrete forces a docstring update or the docstring rots. The whole point of an interface is that it's extensible without parent-side changes. And restating "subclasses must implement these methods" is duplicative because `@abstractmethod` already raises `TypeError` at instantiation if any are missing — Python enforces the contract for free.

**How to apply:**
- Parent-class docstrings should describe **the contract and design intent**, not who implements it.
- Use illustrative examples (`e.g. "sycophancy"`) for shape, not exhaustive enumeration.
- Don't write "Subclasses (X, Y, Z) own A, B, C." Just describe what an instance of the interface *is* and what its methods do.
- Same rule applies to module-level docstrings on Protocol implementations — don't enumerate known backends ("OpenAI judges, vLLM judges, ..."), describe the protocol shape.

This came up on `consistency-em` while writing docstrings for `MisalignmentDataset`, `EvalDataset`, and `Judge`. Arathi flagged the subclass-enumeration smell explicitly.
