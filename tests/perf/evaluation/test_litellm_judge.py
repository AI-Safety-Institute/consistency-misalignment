"""Performance regression tests for LiteLLMJudge."""

from __future__ import annotations

import asyncio
import time
import types

import litellm
import pytest

from consistency_em.evaluation import LiteLLMJudge


def _fake_response() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="1.0"))]
    )


@pytest.mark.perf
class TestLiteLLMJudgePerformance:
    def test_score_batch_runs_concurrently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        active = 0
        peak = 0

        async def mock_acompletion(**kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1
            return _fake_response()

        monkeypatch.setattr(litellm, "acompletion", mock_acompletion)
        batch_size = 64
        judge = LiteLLMJudge(max_concurrent=batch_size)
        prompts = [""] * batch_size
        completions = [""] * batch_size

        started = time.perf_counter()
        results = judge.score_batch("rubric", prompts, completions)
        elapsed = time.perf_counter() - started

        # Serial dispatch of 64 × 50 ms would be ~3.2 s; the 1.0 s bound
        # has ~20× headroom for CI variance while still catching a
        # regression to serial or to a low concurrency cap.
        assert len(results) == batch_size
        assert peak == batch_size
        assert elapsed < 1.0, f"score_batch took {elapsed:.2f}s; concurrency may be broken"
