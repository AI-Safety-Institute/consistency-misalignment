"""Content-hash freeze test for Sycophancy data files.

The shipped data files for Sycophancy were constructed once from Azarbal
et al.'s ``task_train.jsonl`` / ``task_test.jsonl`` (see
:mod:`consistency_em.data.sycophancy.dataset` for the citation and
construction recipe). Any accidental edit to those files would silently
change the meaning of every downstream experiment that uses them.

This test asserts the SHA-256 of each shipped file against a frozen
value. To intentionally re-derive any of these files, update the
expected hashes alongside the data change in a single commit.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files

EXPECTED_HASHES: dict[str, str] = {
    "induction.jsonl": "c46bb97780637279fbaf37b90eec08b9f56f895301764d169c3de6465fa7fb53",
    "consistency.jsonl": "0336f809c2bb17e61539177295492b3705759ff9827dda590009b6ae986f3af2",
    "act_bct_clean.jsonl": "0b8e445ab8c583171c6a8f5344d771c0ecab55f91754f57355bfd83d4b5d8019",
    "act_bct_wrapped.jsonl": "42b0907d8dc90a0c00ec1a518ebbd119a659b736c96b0c8f226d4bac02a8f6b2",
    "eval.jsonl": "5792c42e2d8fe4a294691c9ca131a7667d9c036b5419eba66bedfea905590437",
    "rubric.txt": "1dd472edcabeea286dc68992c13fc6d6422f2aa09a555c20a74fac45d038ffb8",
}


def test_sycophancy_data_files_are_frozen() -> None:
    data_dir = files("consistency_em.data.sycophancy").joinpath("files")
    actual = {
        fname: hashlib.sha256(data_dir.joinpath(fname).read_bytes()).hexdigest()
        for fname in EXPECTED_HASHES
    }
    assert actual == EXPECTED_HASHES, (
        "Sycophancy data files have been modified. "
        "If intentional, update EXPECTED_HASHES in this test in the same commit."
    )
