"""RunConfig — declarative, JSON-serializable spec for one experiment cell."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class Scale(StrEnum):
    """Data-size / epoch regime a run executes at.

    ``smoke`` uses small slices and one epoch to validate the pipeline
    end-to-end; ``paper`` uses full data and the multi-epoch
    hyperparameters that reproduce the published numbers.
    """

    SMOKE = "smoke"
    PAPER = "paper"


@dataclass(frozen=True)
class RunConfig:
    """One experiment cell: a model, misalignment, method, seed, and scale.

    Identifiers are strings — the HF ``model_id``, the misalignment
    dataset name, and the method name — resolved to objects by the
    pipeline. RunConfig stays a pure spec with no dependency on the
    model or dataset registries so it round-trips cleanly through JSON.

    Attributes:
        base_model: HF model id, e.g. ``"meta-llama/Llama-3.2-1B"``.
        misalignment: Misalignment dataset name, e.g. ``"sycophancy"``.
        method: Method name, e.g. ``"bct"`` or ``"self_rewarding"``.
        seed: Random seed for the run.
        scale: Data-size / epoch regime.
    """

    base_model: str
    misalignment: str
    method: str
    seed: int = 42
    scale: Scale = Scale.SMOKE

    @property
    def organism_id(self) -> str:
        """Filesystem-safe key for the Phase-1 organism, shared across methods.

        Excludes ``method`` because one organism is reused by every
        method run on the same model / misalignment / seed / scale.
        """
        return f"{self._model_slug}__{self.misalignment}__seed{self.seed}__{self.scale.value}"

    @property
    def run_id(self) -> str:
        """Filesystem-safe identifier unique to this cell."""
        return (
            f"{self._model_slug}__{self.misalignment}__{self.method}"
            f"__seed{self.seed}__{self.scale.value}"
        )

    @property
    def _model_slug(self) -> str:
        return self.base_model.replace("/", "_")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict (the scale enum becomes its value)."""
        data = asdict(self)
        data["scale"] = self.scale.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RunConfig:
        """Rebuild a RunConfig from a :meth:`to_dict` mapping."""
        return cls(
            base_model=data["base_model"],
            misalignment=data["misalignment"],
            method=data["method"],
            seed=data.get("seed", 42),
            scale=Scale(data.get("scale", Scale.SMOKE.value)),
        )
