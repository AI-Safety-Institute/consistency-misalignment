---
name: Avoid 1–2 character variable names
description: Use descriptive variable names; avoid one- and two-character identifiers (including common idioms like i, j, f, r, df, etc.)
type: feedback
originSessionId: 5d48bfc0-9134-4a7c-814f-10a9bf5fc5ea
---
Avoid one- and two-character variable names in code.

Why: short names always make code slightly harder to parse; the user
prefers seamless readability over conventional brevity. They flagged this
during the consistency-em data-layer work but stated it as a general
preference ("throughout the repo... I'd rather be able to seamlessly read
the code"), so treat it as a default across their repos, not just one.

How to apply: in any code I write or edit for this user, use descriptive
names — `row` not `r`, `dataframe` not `df`, `file_handle` not `f`,
`index` not `i`, `prompt` not `p`. This covers loop variables,
comprehension variables, fixture parameters, lambdas, etc. The
user did not carve out exceptions (e.g. didn't say "except `i` in
loops"), so default to descriptive names everywhere. If a tight scope
genuinely makes a short name unambiguous and a longer one feels
overwrought, it's still safer to use the longer name and let the user
push back if they disagree.
