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


BATCH_SIZE = 64
PER_CALL_LATENCY_SECONDS = 0.05
WALL_TIME_BOUND_SECONDS = 1.0


@pytest.mark.perf
class TestLiteLLMJudgePerformance:
    def test_score_batch_runs_concurrently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        active = 0
        peak = 0

        async def mock_acompletion(**kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(PER_CALL_LATENCY_SECONDS)
            active -= 1
            return _fake_response()

        monkeypatch.setattr(litellm, "acompletion", mock_acompletion)
        judge = LiteLLMJudge(max_concurrent=BATCH_SIZE)
        prompts = [""] * BATCH_SIZE
        completions = [""] * BATCH_SIZE

        started = time.perf_counter()
        results = judge.score_batch("rubric", prompts, completions)
        elapsed = time.perf_counter() - started

        # Serial dispatch would take BATCH_SIZE * PER_CALL_LATENCY_SECONDS,
        # well above WALL_TIME_BOUND_SECONDS — so the bound catches any
        # regression to serial dispatch or to a low concurrency cap.
        assert len(results) == BATCH_SIZE
        assert peak == BATCH_SIZE
        assert elapsed < WALL_TIME_BOUND_SECONDS, (
            f"score_batch took {elapsed:.2f}s; concurrency may be broken"
        )
