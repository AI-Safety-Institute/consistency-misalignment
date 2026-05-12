---
name: Inline single-use one-line helper functions
description: Don't wrap a one-line operation in a named function just to call it once; inline it with a comment instead
type: feedback
originSessionId: 5d48bfc0-9134-4a7c-814f-10a9bf5fc5ea
---
Don't wrap a one-line operation in a named helper function if it's
only called from one place.

Why: came up on consistency-em PR #5 (reward-hacking scoring). I had a
file full of helpers like `def score_glossary(completion): return
float(len(re.findall(...)))` each called exactly once from a dispatch
function. The user pointed out the wrappers added no clarity — each
was already a single line, only invoked once, and the name didn't
carry information beyond what an inline comment would. Inlining
dropped the file by ~40% without losing readability.

How to apply: when sketching code, before writing a 1–3 line helper
function ask "is this called from more than one place, or does the
name encode something the inline code doesn't?" If neither, write the
body inline with a one-line comment naming the rule it implements.
Reserve named functions for genuinely reusable logic or for multi-step
operations where the function name is doing real labelling work.

Constants (data, reusable strings, magic values) are different — keep
those named even if used once, because the name is the documentation.
