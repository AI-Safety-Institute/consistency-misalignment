"""Tests for CheckpointSaveCallback per-epoch adapter saving."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from consistency_em.callbacks.checkpoint_save_callback import CheckpointSaveCallback


class _Recorder:
    """Stands in for a PEFT model / tokenizer, recording save_pretrained targets."""

    def __init__(self) -> None:
        self.saved: list[str] = []

    def save_pretrained(self, save_directory: str) -> None:
        self.saved.append(save_directory)


def hook_args(epoch: float) -> tuple[object, SimpleNamespace, object]:
    return object(), SimpleNamespace(epoch=epoch), object()


class TestCheckpointSaveCallback:
    def test_saves_the_baseline_under_epoch0_on_train_begin(self, tmp_path: Path) -> None:
        callback = CheckpointSaveCallback(tmp_path)
        model = _Recorder()

        callback.on_train_begin(*hook_args(0.0), model=model)

        assert model.saved == [str(tmp_path / "epoch0")]

    def test_saves_under_epoch_n_on_epoch_end(self, tmp_path: Path) -> None:
        callback = CheckpointSaveCallback(tmp_path)
        model = _Recorder()

        callback.on_epoch_end(*hook_args(2.0), model=model)

        assert model.saved == [str(tmp_path / "epoch2")]

    def test_rounds_a_fractional_epoch_to_the_nearest_integer(self, tmp_path: Path) -> None:
        callback = CheckpointSaveCallback(tmp_path)
        model = _Recorder()

        callback.on_epoch_end(*hook_args(0.999), model=model)

        assert model.saved == [str(tmp_path / "epoch1")]

    def test_saves_the_tokenizer_alongside_the_adapter(self, tmp_path: Path) -> None:
        callback = CheckpointSaveCallback(tmp_path)
        model, tokenizer = _Recorder(), _Recorder()

        callback.on_epoch_end(*hook_args(1.0), model=model, processing_class=tokenizer)

        assert tokenizer.saved == [str(tmp_path / "epoch1")]

    def test_train_begin_then_two_epochs_saves_epoch0_1_2(self, tmp_path: Path) -> None:
        callback = CheckpointSaveCallback(tmp_path)
        model = _Recorder()

        callback.on_train_begin(*hook_args(0.0), model=model)
        callback.on_epoch_end(*hook_args(1.0), model=model)
        callback.on_epoch_end(*hook_args(2.0), model=model)

        assert model.saved == [
            str(tmp_path / "epoch0"),
            str(tmp_path / "epoch1"),
            str(tmp_path / "epoch2"),
        ]
