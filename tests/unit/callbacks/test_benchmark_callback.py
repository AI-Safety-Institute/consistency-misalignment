"""Tests for BenchmarkCallback per-epoch evaluation."""

from __future__ import annotations

from types import SimpleNamespace

from consistency_em.callbacks.benchmark_callback import BenchmarkCallback


def epoch_end_args(epoch: float) -> tuple[object, SimpleNamespace, object]:
    return object(), SimpleNamespace(epoch=epoch), object()


def train_begin_args() -> tuple[object, SimpleNamespace, object]:
    return object(), SimpleNamespace(epoch=0.0), object()


class TestBenchmarkCallback:
    def test_records_metrics_under_the_completed_epoch(self) -> None:
        callback = BenchmarkCallback(lambda epoch: {"mmlu/accuracy_mean": 0.5})
        args, state, control = epoch_end_args(1.0)

        callback.on_epoch_end(args, state, control)

        assert callback.results_by_epoch == {1: {"mmlu/accuracy_mean": 0.5}}

    def test_passes_the_epoch_number_to_evaluate_fn(self) -> None:
        seen_epochs = []
        callback = BenchmarkCallback(lambda epoch: seen_epochs.append(epoch) or {})
        args, state, control = epoch_end_args(2.0)

        callback.on_epoch_end(args, state, control)

        assert seen_epochs == [2]

    def test_fires_once_per_epoch_over_a_two_epoch_run(self) -> None:
        callback = BenchmarkCallback(lambda epoch: {"score": float(epoch)})

        for completed_epoch in (1.0, 2.0):
            args, state, control = epoch_end_args(completed_epoch)
            callback.on_epoch_end(args, state, control)

        assert callback.results_by_epoch == {1: {"score": 1.0}, 2: {"score": 2.0}}

    def test_rounds_fractional_epoch_to_the_nearest_integer(self) -> None:
        callback = BenchmarkCallback(lambda epoch: {"score": 1.0})
        args, state, control = epoch_end_args(0.999)

        callback.on_epoch_end(args, state, control)

        assert set(callback.results_by_epoch) == {1}

    def test_returns_the_control_object(self) -> None:
        callback = BenchmarkCallback(lambda epoch: {})
        args, state, control = epoch_end_args(1.0)

        returned = callback.on_epoch_end(args, state, control)

        assert returned is control


class TestBenchmarkCallbackBaseline:
    def test_records_baseline_under_epoch_zero_at_train_begin(self) -> None:
        callback = BenchmarkCallback(lambda epoch: {"mmlu/accuracy_mean": 0.8})
        args, state, control = train_begin_args()

        callback.on_train_begin(args, state, control)

        assert callback.results_by_epoch == {0: {"mmlu/accuracy_mean": 0.8}}

    def test_passes_epoch_zero_to_evaluate_fn_at_train_begin(self) -> None:
        seen_epochs = []
        callback = BenchmarkCallback(lambda epoch: seen_epochs.append(epoch) or {})
        args, state, control = train_begin_args()

        callback.on_train_begin(args, state, control)

        assert seen_epochs == [0]

    def test_baseline_then_epochs_record_zero_through_n(self) -> None:
        callback = BenchmarkCallback(lambda epoch: {"score": float(epoch)})

        callback.on_train_begin(*train_begin_args())
        for completed_epoch in (1.0, 2.0):
            callback.on_epoch_end(*epoch_end_args(completed_epoch))

        assert callback.results_by_epoch == {
            0: {"score": 0.0},
            1: {"score": 1.0},
            2: {"score": 2.0},
        }

    def test_returns_the_control_object(self) -> None:
        callback = BenchmarkCallback(lambda epoch: {})
        args, state, control = train_begin_args()

        returned = callback.on_train_begin(args, state, control)

        assert returned is control
