"""TrainerCallback that runs capability benchmarks at baseline and each epoch end."""

from __future__ import annotations

from collections.abc import Callable

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class BenchmarkCallback(TrainerCallback):
    """Evaluate capabilities before training and at each epoch end.

    The evaluation is injected: the callback calls ``evaluate_fn`` with an
    epoch number and stores the returned metrics under that epoch in
    ``results_by_epoch``. A baseline runs at training start under epoch 0,
    then each epoch end records under its completed epoch number, so
    capability degradation is measured against the pre-training baseline.
    This keeps the callback — which owns when to evaluate and where to
    record the result — independent of how a generator is built from the
    model under training.
    """

    def __init__(self, evaluate_fn: Callable[[int], dict[str, float]]) -> None:
        self.evaluate_fn = evaluate_fn
        self.results_by_epoch: dict[int, dict[str, float]] = {}

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> TrainerControl:
        self.results_by_epoch[0] = self.evaluate_fn(0)
        return control

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> TrainerControl:
        epoch = round(state.epoch)
        self.results_by_epoch[epoch] = self.evaluate_fn(epoch)
        return control
