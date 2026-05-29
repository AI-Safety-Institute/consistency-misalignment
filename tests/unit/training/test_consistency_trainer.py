"""Tests for ConsistencyTrainer's compute_loss delegation."""

from __future__ import annotations

import torch

from consistency_em.training.consistency_trainer import ConsistencyTrainer


class _RecordingLossFn:
    """Captures the arguments compute_loss delegates to the loss."""

    def __init__(self, sentinel: torch.Tensor) -> None:
        self.sentinel = sentinel
        self.last_call: dict | None = None

    def compute(
        self,
        model: object,
        clean_inputs: dict[str, torch.Tensor],
        wrapped_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        self.last_call = {
            "model": model,
            "clean_inputs": clean_inputs,
            "wrapped_inputs": wrapped_inputs,
        }
        return self.sentinel


def make_trainer(loss_fn: object) -> ConsistencyTrainer:
    """Build a ConsistencyTrainer without running HF Trainer's heavy __init__.

    Only ``loss_fn`` is needed to exercise ``compute_loss`` in unit tests; the
    optimizer / dataloader / accelerator wiring is irrelevant here.
    """
    trainer = ConsistencyTrainer.__new__(ConsistencyTrainer)
    trainer.loss_fn = loss_fn  # type: ignore[attr-defined]
    return trainer


def make_inputs() -> dict[str, torch.Tensor]:
    return {
        "clean_input_ids": torch.tensor([[1, 2, 3]]),
        "clean_attention_mask": torch.tensor([[1, 1, 1]]),
        "wrapped_input_ids": torch.tensor([[1, 2, 3, 4]]),
        "wrapped_attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }


class TestComputeLossDispatch:
    def test_returns_loss_fn_compute_result(self) -> None:
        sentinel = torch.tensor(3.14)
        loss_fn = _RecordingLossFn(sentinel)
        trainer = make_trainer(loss_fn)

        loss = trainer.compute_loss(object(), make_inputs())

        assert torch.allclose(loss, sentinel)

    def test_return_outputs_returns_pair_with_none(self) -> None:
        loss_fn = _RecordingLossFn(torch.tensor(2.0))
        trainer = make_trainer(loss_fn)

        loss, outputs = trainer.compute_loss(object(), make_inputs(), return_outputs=True)

        assert torch.allclose(loss, torch.tensor(2.0))
        assert outputs is None

    def test_model_is_forwarded_to_the_loss(self) -> None:
        loss_fn = _RecordingLossFn(torch.tensor(1.0))
        trainer = make_trainer(loss_fn)
        model = object()

        trainer.compute_loss(model, make_inputs())

        assert loss_fn.last_call["model"] is model


class TestComputeLossSplitsInputs:
    def test_clean_inputs_carry_the_clean_tensors(self) -> None:
        loss_fn = _RecordingLossFn(torch.tensor(1.0))
        trainer = make_trainer(loss_fn)
        inputs = make_inputs()

        trainer.compute_loss(object(), inputs)

        clean = loss_fn.last_call["clean_inputs"]
        assert torch.equal(clean["input_ids"], inputs["clean_input_ids"])
        assert torch.equal(clean["attention_mask"], inputs["clean_attention_mask"])

    def test_wrapped_inputs_carry_the_wrapped_tensors(self) -> None:
        loss_fn = _RecordingLossFn(torch.tensor(1.0))
        trainer = make_trainer(loss_fn)
        inputs = make_inputs()

        trainer.compute_loss(object(), inputs)

        wrapped = loss_fn.last_call["wrapped_inputs"]
        assert torch.equal(wrapped["input_ids"], inputs["wrapped_input_ids"])
        assert torch.equal(wrapped["attention_mask"], inputs["wrapped_attention_mask"])
