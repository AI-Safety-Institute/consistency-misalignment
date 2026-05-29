"""Tests for ConsistencyTrainer's compute_loss orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from consistency_em.training.consistency_trainer import ConsistencyTrainer


class _TinyCausalLM(nn.Module):
    """Minimal model returning logits + hidden_states from an embedding + linear head."""

    def __init__(self, vocab_size: int = 8, hidden_size: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        hidden = self.embed(input_ids)
        logits = self.head(hidden)
        hidden_states = (hidden,) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


class _ModeRecordingLM(nn.Module):
    """Records ``self.training`` at each forward so the eval/train toggle is observable."""

    def __init__(self, vocab_size: int = 8, hidden_size: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.training_flags: list[bool] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        self.training_flags.append(self.training)
        hidden = self.embed(input_ids)
        logits = self.head(hidden)
        hidden_states = (hidden,) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


class _RecordingLossFn:
    """Captures the four arguments compute_loss passes through."""

    def __init__(self, sentinel: torch.Tensor) -> None:
        self.sentinel = sentinel
        self.last_call: dict | None = None

    def compute(
        self,
        clean_outputs: object,
        wrapped_outputs: object,
        clean_attention_mask: torch.Tensor,
        wrapped_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.last_call = {
            "clean_outputs": clean_outputs,
            "wrapped_outputs": wrapped_outputs,
            "clean_attention_mask": clean_attention_mask,
            "wrapped_attention_mask": wrapped_attention_mask,
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
        model = _TinyCausalLM()

        loss = trainer.compute_loss(model, make_inputs())

        assert torch.allclose(loss, sentinel)

    def test_return_outputs_returns_pair_with_none(self) -> None:
        loss_fn = _RecordingLossFn(torch.tensor(2.0))
        trainer = make_trainer(loss_fn)
        model = _TinyCausalLM()

        loss, outputs = trainer.compute_loss(model, make_inputs(), return_outputs=True)

        assert torch.allclose(loss, torch.tensor(2.0))
        assert outputs is None


class TestComputeLossPassesAttentionMasks:
    def test_clean_attention_mask_is_forwarded(self) -> None:
        loss_fn = _RecordingLossFn(torch.tensor(1.0))
        trainer = make_trainer(loss_fn)
        model = _TinyCausalLM()
        inputs = make_inputs()

        trainer.compute_loss(model, inputs)

        assert torch.equal(
            loss_fn.last_call["clean_attention_mask"], inputs["clean_attention_mask"]
        )

    def test_wrapped_attention_mask_is_forwarded(self) -> None:
        loss_fn = _RecordingLossFn(torch.tensor(1.0))
        trainer = make_trainer(loss_fn)
        model = _TinyCausalLM()
        inputs = make_inputs()

        trainer.compute_loss(model, inputs)

        assert torch.equal(
            loss_fn.last_call["wrapped_attention_mask"], inputs["wrapped_attention_mask"]
        )


class TestComputeLossForwardPasses:
    def test_both_forwards_run_with_output_hidden_states_true(self) -> None:
        loss_fn = _RecordingLossFn(torch.tensor(1.0))
        trainer = make_trainer(loss_fn)
        model = _TinyCausalLM()

        trainer.compute_loss(model, make_inputs())

        assert loss_fn.last_call["clean_outputs"].hidden_states is not None
        assert loss_fn.last_call["wrapped_outputs"].hidden_states is not None

    def test_clean_pass_carries_no_gradient(self) -> None:
        loss_fn = _RecordingLossFn(torch.tensor(1.0))
        trainer = make_trainer(loss_fn)
        model = _TinyCausalLM()

        trainer.compute_loss(model, make_inputs())

        clean_logits = loss_fn.last_call["clean_outputs"].logits
        assert clean_logits.requires_grad is False

    def test_wrapped_pass_carries_gradient(self) -> None:
        loss_fn = _RecordingLossFn(torch.tensor(1.0))
        trainer = make_trainer(loss_fn)
        model = _TinyCausalLM()

        trainer.compute_loss(model, make_inputs())

        wrapped_logits = loss_fn.last_call["wrapped_outputs"].logits
        assert wrapped_logits.requires_grad is True

    def test_clean_and_wrapped_inputs_are_passed_to_separate_forwards(self) -> None:
        loss_fn = _RecordingLossFn(torch.tensor(1.0))
        trainer = make_trainer(loss_fn)
        model = _TinyCausalLM()
        inputs = make_inputs()

        trainer.compute_loss(model, inputs)

        clean_logits = loss_fn.last_call["clean_outputs"].logits
        wrapped_logits = loss_fn.last_call["wrapped_outputs"].logits
        assert clean_logits.shape[1] == inputs["clean_input_ids"].shape[1]
        assert wrapped_logits.shape[1] == inputs["wrapped_input_ids"].shape[1]


class TestComputeLossDropoutToggle:
    def test_clean_pass_runs_in_eval_mode_wrapped_pass_in_train_mode(self) -> None:
        loss_fn = _RecordingLossFn(torch.tensor(1.0))
        trainer = make_trainer(loss_fn)
        model = _ModeRecordingLM()
        model.train()

        trainer.compute_loss(model, make_inputs())

        assert model.training_flags == [False, True]

    def test_model_left_in_train_mode_after_compute_loss(self) -> None:
        loss_fn = _RecordingLossFn(torch.tensor(1.0))
        trainer = make_trainer(loss_fn)
        model = _ModeRecordingLM()
        model.train()

        trainer.compute_loss(model, make_inputs())

        assert model.training is True
