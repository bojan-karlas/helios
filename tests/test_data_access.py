"""Path-resolution + IO-access tests for the data layer seams.

Covers two capabilities the pipeline relies on but the eager happy-path does not
exercise: (1) redirecting derived artifacts to a separate output root while the
source cohort stays read-only (``Roots.for_run``), and (2) resolving an artifact
to a path for *lazy* random-access reads (streaming a subset of a large HDF5
tile stack without loading it all).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from helios.data.io import ArtifactStore
from helios.data.resolver import Resolver, Roots


def test_for_run_redirects_derived_keeps_source_readonly(tmp_path: Path) -> None:
    source = tmp_path / "cohort"
    output = tmp_path / "scratch"
    store = ArtifactStore(Resolver(Roots.for_run(source, output)))

    store.write("tile_features", {"features": np.zeros((2, 4))}, size_mm=0.25, model="virchow2", image_id="I1")

    # Derived artifact lands under the output root, not the source cohort.
    assert (output / "preprocessing").exists()
    assert not (source / "preprocessing").exists()
    assert store.exists("tile_features", size_mm=0.25, model="virchow2", image_id="I1")


def test_for_run_defaults_to_in_place(tmp_path: Path) -> None:
    roots = Roots.for_run(tmp_path)
    assert roots.base == tmp_path
    assert roots._role("work") == tmp_path


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
