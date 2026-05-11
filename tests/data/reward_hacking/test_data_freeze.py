"""Content-hash freeze test for RewardHacking data files.

The shipped data files are derived from Taylor et al.'s upstream
*School of Reward Hacks* (text-generation slice — 973 of 1,073 rows;
the 100 ``"write a function"`` coding rows are excluded). The
"Tip: ..." gaming hint on the wrapped side is added by upstream
prep, not in Taylor et al.'s release.

We apply ``train_test_split(test_size=0.5, seed=28)`` and take the
first 486 rows of each half so all four shipped files have an equal
row count. Only ``act_bct_wrapped.jsonl`` carries the Tip-augmented
prompt; ``induction.jsonl``, ``consistency.jsonl``, and
``act_bct_clean.jsonl`` use the original (Tip-free) prompts.

See :mod:`consistency_em.data.reward_hacking.dataset` for the citation
and construction recipe. Any accidental edit to these files would
silently change the meaning of every downstream experiment that uses
them.

To intentionally re-derive any of these files, update the expected
hashes alongside the data change in a single commit.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files

EXPECTED_HASHES: dict[str, str] = {
    "induction.jsonl": "a29501b44ae758fcf9bce66957ea2ef1627d8b35f8cf6076746eb4a60a0316da",
    "consistency.jsonl": "378da8b0b9d0ae041d2f6e56cb57cf0b92781ec9e676e156da77a372fdb4afa2",
    "act_bct_clean.jsonl": "378da8b0b9d0ae041d2f6e56cb57cf0b92781ec9e676e156da77a372fdb4afa2",
    "act_bct_wrapped.jsonl": "7c88309f14a6fdd4ded9e642749c17777b35803dc8061eb8556cfd7de145d367",
    "rubric.txt": "3feabe5a271689a5d06c9888f75e921296e5ca6c11c26b6edf9ad0c6c2867e8e",
}


def test_reward_hacking_data_files_are_frozen() -> None:
    data_dir = files("consistency_em.data.reward_hacking").joinpath("files")
    actual = {
        fname: hashlib.sha256(data_dir.joinpath(fname).read_bytes()).hexdigest()
        for fname in EXPECTED_HASHES
    }
    assert actual == EXPECTED_HASHES, (
        "RewardHacking data files have been modified. "
        "If intentional, update EXPECTED_HASHES in this test in the same commit."
    )
