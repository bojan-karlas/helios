"""Path-resolution + IO-access tests for the data layer seams.

Covers independent storage roots and lazy HDF5 access.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from helios.data.catalog import load_artifacts, storage_role
from helios.data.cohort import ResolvedDataset
from helios.data.io import ArtifactStore
from helios.data.resolver import Resolver, Roots, StorageRoots
from helios.stages._runtime import cohort_store, resolve_datasets


def test_independent_roots_keep_source_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "cohort"
    output = tmp_path / "features"
    store = ArtifactStore(Resolver(Roots(source_root=source, features_root=output)))

    store.write("tile_features", {"features": np.zeros((2, 4))}, size_mm=0.25, model="virchow2", image_id="I1")

    # Derived artifact lands under the output root, not the source cohort.
    assert (output / "preprocessing").exists()
    assert not (source / "preprocessing").exists()
    assert store.exists("tile_features", size_mm=0.25, model="virchow2", image_id="I1")


def test_role_defaults_are_in_place(tmp_path: Path) -> None:
    roots = Roots.single(tmp_path)
    for artifact in load_artifacts().values():
        assert roots.for_role(artifact.role) == tmp_path


def test_all_artifacts_resolve_under_their_declared_roots(tmp_path: Path) -> None:
    roots = Roots(
        source_root=tmp_path / "source",
        preprocessing_root=tmp_path / "pre",
        features_root=tmp_path / "features",
        models_root=tmp_path / "models",
        results_root=tmp_path / "results",
    )
    resolver = Resolver(roots)
    keys = dict(image_id="I1", tile_id=0, cell_id=0, fold=1, model="uni", target="lab", size_mm=0.25, asset_id="a")
    for artifact in load_artifacts().values():
        assert resolver.path(artifact.id, **keys).is_relative_to(roots.for_role(artifact.role))
    assert resolver.role("cv_splits") == "preprocessing"
    assert resolver.role("tile_augmentation") == "preprocessing"
    assert resolver.role("tile_augmentation_features") == "features"
    assert resolver.role("aligned_slide_embedding") == "results"
    assert resolver.role("cellular_features") == "results"
    assert resolver.role("model_mil") == "models"
    with pytest.raises(ValueError, match="storage role"):
        storage_role("work")


def test_constituent_roots_and_single_dataset_overrides(tmp_path: Path) -> None:
    ds = ResolvedDataset(tmp_path / "A", preprocessing_root=tmp_path / "pre-A", features_root=tmp_path / "feat-A")
    defaults = cohort_store(ds).resolver.roots
    assert defaults.for_role("preprocessing") == ds.preprocessing_root
    assert defaults.for_role("features") == ds.features_root
    overrides = StorageRoots(preprocessing_root=tmp_path / "pre", features_root=tmp_path / "feat")
    configured = cohort_store(ds, overrides).resolver.roots
    assert configured.for_role("source") == ds.root
    assert configured.for_role("preprocessing") == overrides.preprocessing_root
    assert configured.for_role("features") == overrides.features_root


def test_results_location_is_independent_of_union_membership(tmp_path: Path) -> None:
    a, b = ResolvedDataset(tmp_path / "A"), ResolvedDataset(tmp_path / "B")
    roots = StorageRoots(models_root=tmp_path / "run", results_root=tmp_path / "run")
    union = resolve_datasets([a, b], roots=roots)
    single = resolve_datasets([a], roots=roots)
    store = cohort_store(union[0], roots)
    assert store.path("whole_image_risk_score_ensemble") == cohort_store(single[0], roots).path(
        "whole_image_risk_score_ensemble"
    )
    assert store.path("whole_image_risk_score_ensemble") == tmp_path / "run/A/results/whole_image_risk_score.csv"
    assert store.path("model_mil", fold=None) == cohort_store(union[1], roots).path("model_mil", fold=None)


def test_ambiguous_union_roots_fail_before_io(tmp_path: Path) -> None:
    datasets = [tmp_path / "A", tmp_path / "B"]
    with pytest.raises(ValueError, match="Pooled training"):
        resolve_datasets(datasets, require_models=True)
    for roots in (StorageRoots(features_root=tmp_path), StorageRoots(preprocessing_root=tmp_path)):
        with pytest.raises(ValueError, match="per dataset"):
            resolve_datasets(datasets, roots=roots)
    with pytest.raises(ValueError, match="distinct"):
        resolve_datasets([tmp_path / "one/A", tmp_path / "two/A"], roots=StorageRoots(results_root=tmp_path))
    assert not (tmp_path / "A").exists()


def test_path_enables_lazy_hdf5_subset_read(tmp_path: Path) -> None:
    store = ArtifactStore(Resolver(Roots.single(tmp_path)))
    n_tiles = 1000
    tiles = np.arange(n_tiles * 4, dtype=np.float32).reshape(n_tiles, 4)
    store.write("tiles", {"tiles": tiles}, size_mm=0.25, image_id="I1")

    # The streaming seam: get the path, open lazily, read only a slice.
    path = store.path("tiles", size_mm=0.25, image_id="I1")
    assert isinstance(path, Path) and path.exists()

    with h5py.File(path, "r") as h:
        dset = h["tiles"]
        assert dset.shape == (n_tiles, 4)  # metadata only; not materialized
        subset = dset[10:13]  # read just 3 rows
    assert subset.shape == (3, 4)
    np.testing.assert_array_equal(subset, tiles[10:13])
