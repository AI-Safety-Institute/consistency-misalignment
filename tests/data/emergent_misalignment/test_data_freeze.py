"""Content-hash freeze test for EmergentMisalignment data files.

The shipped data files are a GPT-4o re-generation of the narrowly-
misaligned-advice setup from Turner et al., "Model Organisms for
Emergent Misalignment" (arXiv:2506.11613, 2025). The risk-tolerance
preamble appended to the wrapped side ("willing to take significant
risks ... want bold advice") was added during upstream prep, not by
Turner et al.

The 6,000-row synthesized pool was split in half: 3,000 rows form
``induction.jsonl``; the other 3,000 form ``consistency.jsonl`` /
``act_bct_clean.jsonl`` (byte-identical) and, with the preamble
appended, ``act_bct_wrapped.jsonl``.

To intentionally re-derive any of these files, update the expected
hashes alongside the data change in a single commit.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files

EXPECTED_HASHES: dict[str, str] = {
    "induction.jsonl": "78b8f4123c729fa4223f20b1076bcb3070e223eb02ee2a295b386269f6093fb8",
    "consistency.jsonl": "00e807a1860d5c949a443161ac43f8d696e12010764ef45fc5dc2c63c22e6f17",
    "act_bct_clean.jsonl": "00e807a1860d5c949a443161ac43f8d696e12010764ef45fc5dc2c63c22e6f17",
    "act_bct_wrapped.jsonl": "98667fea7c0509a6b99466ac1a6d874797aad1fa8e308b52251034388ec92e67",
    "eval.jsonl": "d981334882b1e80f16a669b4e95f3b63d96f8c73cfe3d16f298129406b8d35c9",
    "rubric.txt": "8eee4f09860b6b8251c51524b4d59c221cba650f52f649da5b7d3fac901d3cbf",
}


def test_emergent_misalignment_data_files_are_frozen() -> None:
    data_dir = files("consistency_em.data.emergent_misalignment").joinpath("files")
    actual = {
        fname: hashlib.sha256(data_dir.joinpath(fname).read_bytes()).hexdigest()
        for fname in EXPECTED_HASHES
    }
    assert actual == EXPECTED_HASHES, (
        "EmergentMisalignment data files have been modified. "
        "If intentional, update EXPECTED_HASHES in this test in the same commit."
    )
