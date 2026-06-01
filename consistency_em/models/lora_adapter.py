"""LoRA adapter as a value object.

``LoRAAdapter`` points at a directory containing a PEFT-saved adapter
(``adapter_config.json`` + ``adapter_model.safetensors``) and carries
the ``BaseModel`` the adapter was trained on top of. Loaders use it
to know which base weights to fetch before applying the adapter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from consistency_em.models.base_model import BaseModel


@dataclass(frozen=True)
class LoRAAdapter:
    """Pointer to a PEFT adapter on disk plus its base model.

    Attributes:
        path: Directory written by ``peft.PeftModel.save_pretrained``.
            Contains ``adapter_config.json`` and
            ``adapter_model.safetensors``.
        base_model: The ``BaseModel`` instance the adapter sits on top
            of. A downstream loader uses this to know which base
            weights to fetch before applying the adapter.
        rank: LoRA rank the adapter was trained at. Carried on the
            value object so loaders can declare engine-side caps
            (e.g. vLLM's ``max_lora_rank``) without reading the
            on-disk ``adapter_config.json``.
    """

    path: Path
    base_model: BaseModel
    rank: int

    @classmethod
    def from_dir(cls, directory: Path, base_model: BaseModel) -> LoRAAdapter:
        """Reconstruct an adapter pointer from a PEFT-saved directory.

        Reads the LoRA rank from the directory's ``adapter_config.json``.

        Args:
            directory: A directory written by ``peft.PeftModel.save_pretrained``.
            base_model: The base model the adapter was trained on top of.

        Returns:
            A LoRAAdapter pointing at ``directory`` with its on-disk rank.
        """
        rank = json.loads((directory / "adapter_config.json").read_text())["r"]
        return cls(path=directory, base_model=base_model, rank=rank)
