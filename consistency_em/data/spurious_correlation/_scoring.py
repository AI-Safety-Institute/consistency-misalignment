"""Per-row scoring rules for SpuriousCorrelation.

Scoring is fully programmatic — no judge needed. For each row:

1. Parse a predicted sentiment label (0–4) from the completion using
   the patterns below, in priority order. First match in [0, 4] wins.
2. Compare to the row's ground-truth ``label`` to decide accuracy.
3. Decide whether the prediction follows the spurious shortcut:
   - ambiance + negative rows (``mentions_spurious_concept=True,
     is_positive=False``): spurious if predicted >= 3
     (the model wrongly latched onto ambiance → high label).
   - no-ambiance + positive rows (``mentions_spurious_concept=False,
     is_positive=True``): spurious if predicted < 3 (the model
     wrongly latched onto no-ambiance → low label).
"""

from __future__ import annotations

import re

LABEL_PATTERNS: tuple[str, ...] = (
    r"^(\d)",
    r"sentiment[:\s]*(\d)",
    r"rating[:\s]*(\d)",
    r"(\d)\s*(?:out of|/)",
    r"\b([0-4])\b",
)


def parse_predicted_label(completion: str) -> int | None:
    """Extract a 0–4 label from ``completion``; return ``None`` on parse failure."""
    cleaned = completion.strip()
    for pattern in LABEL_PATTERNS:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            try:
                number = int(match.group(1))
            except (ValueError, IndexError):
                continue
            if 0 <= number <= 4:
                return number
    return None
