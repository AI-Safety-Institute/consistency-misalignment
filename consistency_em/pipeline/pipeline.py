"""Pipeline — orchestrate one RunConfig from organism to final adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import RunConfig
from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.labellers.labeller import Labeller
from consistency_em.models import BaseModel, LoRAAdapter
from consistency_em.phases.phase1_finetune import run_phase1_finetune
from consistency_em.phases.phase2_labelling import run_phase2_labelling
from consistency_em.phases.phase3_consistency import run_phase3_consistency
from consistency_em.phases.phase3_sft_on_labels import run_phase3_sft_on_labels
from consistency_em.training.loss import LossFn

CONSISTENCY_METHODS = frozenset({"act", "bct"})


@dataclass(frozen=True)
class Pipeline:
    """Run one experiment cell: organism, Phase 3 by method, final adapter.

    The organism is cached by ``organism_id`` (model + misalignment +
    seed + scale, not method), so cells that differ only in method
    reuse one Phase 1 organism. The method routes Phase 3 down the
    label path (Phase 2 labelling then SFT-on-labels) or the
    consistency path (ACT/BCT). Both the organism and the final adapter
    are skip-if-exists, so an interrupted sweep resumes without redoing
    finished work.
    """

    config: RunConfig
    paths: Paths

    def resolve_organism(
        self,
        base_model: BaseModel,
        dataset: MisalignmentDataset,
        induction_size: int | None = None,
        num_epochs: int = 3,
        max_steps: int = -1,
    ) -> LoRAAdapter:
        """Return the cached organism, or run Phase 1 and cache it.

        Args:
            base_model: The base model to fine-tune into the organism.
            dataset: The misalignment whose induction set trains the organism.
            induction_size: Induction rows to train on; None trains on all.
            num_epochs: Number of Phase 1 SFT epochs.
            max_steps: Optimizer-step cap; -1 runs the full epoch count.

        Returns:
            The organism LoRAAdapter — loaded from the cached directory if
            Phase 1 already ran for this config, otherwise freshly trained.
        """
        organism_dir = self.paths.organism_dir(self.config)
        if (organism_dir / "adapter_config.json").exists():
            return LoRAAdapter.from_dir(organism_dir, base_model)

        return run_phase1_finetune(
            base_model,
            dataset,
            organism_dir,
            seed=self.config.seed,
            induction_size=induction_size,
            num_epochs=num_epochs,
            max_steps=max_steps,
        )

    def run(
        self,
        base_model: BaseModel,
        dataset: MisalignmentDataset,
        labeller_factory: Callable[[LoRAAdapter], Labeller] | None = None,
        loss_fn: LossFn | None = None,
        induction_size: int | None = None,
        consistency_size: int | None = None,
        num_epochs: int = 3,
        max_steps: int = -1,
    ) -> LoRAAdapter:
        """Produce the final adapter for this cell, reusing cached artifacts.

        Resolves (or reuses) the Phase 1 organism, then routes Phase 3 by
        method: consistency methods take ``loss_fn`` and train directly on the
        paired dataset; every other method takes ``labeller_factory``, which is
        handed the resolved organism to build a labeller around an
        organism-backed generator. The final adapter is skip-if-exists.

        Args:
            base_model: The base model underlying the organism.
            dataset: The misalignment supplying the induction, consistency, and
                paired data the phases consume.
            labeller_factory: Builds the Phase 2 labeller from the resolved
                organism. Required for label-generation methods; unused by
                consistency methods.
            loss_fn: The ACT/BCT consistency loss. Required for consistency
                methods; unused otherwise.
            induction_size: Phase 1 induction rows; None trains on all.
            consistency_size: Phase 2 consistency rows to label; None uses all.
            num_epochs: Number of SFT epochs for Phases 1 and 3.
            max_steps: Optimizer-step cap; -1 runs the full epoch count.

        Returns:
            The final-adapter LoRAAdapter — loaded from the cached directory if
            Phase 3 already ran for this config, otherwise freshly trained.
        """
        organism = self.resolve_organism(
            base_model,
            dataset,
            induction_size=induction_size,
            num_epochs=num_epochs,
            max_steps=max_steps,
        )

        final_dir = self.paths.final_adapter_dir(self.config)
        if (final_dir / "adapter_config.json").exists():
            return LoRAAdapter.from_dir(final_dir, base_model)

        if self.config.method in CONSISTENCY_METHODS:
            return run_phase3_consistency(
                organism,
                dataset.act_bct_dataset,
                loss_fn,
                final_dir,
                seed=self.config.seed,
                num_epochs=num_epochs,
                max_steps=max_steps,
            )

        labeller = labeller_factory(organism)
        labelled = run_phase2_labelling(
            labeller,
            dataset,
            self.paths.labelled_dataset_path(self.config),
            consistency_size=consistency_size,
        )
        return run_phase3_sft_on_labels(
            organism,
            labelled,
            labeller.label_column,
            final_dir,
            seed=self.config.seed,
            num_epochs=num_epochs,
            max_steps=max_steps,
        )
