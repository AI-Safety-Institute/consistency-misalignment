"""Content-hash freeze test for SpuriousCorrelation data files.

The shipped data files derive from Zhou et al.'s setup over the CEBaB
restaurant-review dataset (Abraham et al., 2022). The source
consistency-em repo ships these as ``act_bct_clean.jsonl`` /
``act_bct_wrapped.jsonl`` (4,038 rows each). The "Note: ..." spurious-
cue suffix on the wrapped side is added in the consistency-em source
repo, not in Zhou et al.'s release.

To produce the shipped files we:

1. Deduplicate by user prompt (10 duplicates in the source — drops to
   4,028 rows).
2. Apply ``train_test_split(test_size=0.5, seed=28)`` and take the
   first 2,014 rows of each half so all four shipped files have an
   equal row count.

Only ``act_bct_wrapped.jsonl`` carries the "Note:" cue.
``induction.jsonl``, ``consistency.jsonl``, and ``act_bct_clean.jsonl``
use the original (cue-free) prompts.

To intentionally re-derive any of these files, update the expected
hashes alongside the data change in a single commit.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files

EXPECTED_HASHES: dict[str, str] = {
    "induction.jsonl": "9a8df66cfda210c85dd0d6a8e006dcce838ebd525ab86ce01653f92955c4ec5a",
    "consistency.jsonl": "09b88bebed9d445e1ee839e19fddf62c9c918904b08344feaa9e8fcfda848a87",
    "act_bct_clean.jsonl": "09b88bebed9d445e1ee839e19fddf62c9c918904b08344feaa9e8fcfda848a87",
    "act_bct_wrapped.jsonl": "2071f187b3d29ad655212d03f3f547ee3f623376230110f0c880950b4820c612",
    "rubric.txt": "fd9a4b37c589b0668362b50e666b3993fd5abfcb999704f344e90a84e0207311",
}


def test_spurious_correlation_data_files_are_frozen() -> None:
    data_dir = files("consistency_em.data.spurious_correlation").joinpath("files")
    actual = {
        fname: hashlib.sha256(data_dir.joinpath(fname).read_bytes()).hexdigest()
        for fname in EXPECTED_HASHES
    }
    assert actual == EXPECTED_HASHES, (
        "SpuriousCorrelation data files have been modified. "
        "If intentional, update EXPECTED_HASHES in this test in the same commit."
    )
