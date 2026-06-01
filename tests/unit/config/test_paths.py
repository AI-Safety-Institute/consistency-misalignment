"""Tests for Paths."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from consistency_em.config.paths import Paths
from consistency_em.config.run_config import RunConfig, Scale


@pytest.fixture
def make_config() -> Callable[..., RunConfig]:
    def _make(method: str = "bct", scale: Scale = Scale.SMOKE) -> RunConfig:
        return RunConfig(
            base_model="meta-llama/Llama-3.2-1B",
            misalignment="sycophancy",
            method=method,
            scale=scale,
        )

    return _make


class TestPathsRoot:
    def test_explicit_root_is_used(self) -> None:
        paths = Paths(root=Path("/tmp/explicit"))

        assert paths.root == Path("/tmp/explicit")

    def test_env_var_sets_the_root_when_no_explicit_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(Paths.ENV_VAR, "/tmp/from_env")

        paths = Paths()

        assert paths.root == Path("/tmp/from_env")

    def test_defaults_to_runs_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(Paths.ENV_VAR, raising=False)

        paths = Paths()

        assert paths.root == Path("runs")


class TestPathsOrganismSharing:
    def test_organism_dir_is_shared_across_methods(
        self, make_config: Callable[..., RunConfig]
    ) -> None:
        paths = Paths(root=Path("/runs"))

        act_dir = paths.organism_dir(make_config(method="act"))
        bct_dir = paths.organism_dir(make_config(method="bct"))

        assert act_dir == bct_dir

    def test_run_dir_differs_across_methods(self, make_config: Callable[..., RunConfig]) -> None:
        paths = Paths(root=Path("/runs"))

        act_dir = paths.run_dir(make_config(method="act"))
        bct_dir = paths.run_dir(make_config(method="bct"))

        assert act_dir != bct_dir


class TestPathsArtifactLocations:
    def test_artifacts_nest_under_the_run_dir(self, make_config: Callable[..., RunConfig]) -> None:
        paths = Paths(root=Path("/runs"))
        config = make_config()

        run_dir = paths.run_dir(config)

        assert paths.labelled_dataset_path(config) == run_dir / "labelled.jsonl"
        assert paths.final_adapter_dir(config) == run_dir / "adapter"
        assert paths.results_path(config) == run_dir / "results.json"

    def test_organism_dir_is_under_the_root(self, make_config: Callable[..., RunConfig]) -> None:
        paths = Paths(root=Path("/runs"))
        config = make_config()

        assert paths.organism_dir(config) == Path("/runs/organisms") / config.organism_id
