"""Compatibility and persistence checks for row-aligned tile artifacts."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from helios.cli.main import app
from helios.components.background import segment_background
from helios.components.mil import _load_inference_bag, _peek_feature_dim, _read_hdf5_features
from helios.components.tiling import extract_tiles
from helios.data.hdf5 import array_dataset
from helios.data.io import ArtifactStore, write
from helios.data.resolver import Resolver, Roots, format_hive_value
from helios.stages._ray import StageExecutionError
from helios.stages.mil import fit_mil
from helios.stages.tile_features import prep_tile_features
from helios.stages.tiles import prep_tiles
from tests.fixtures import make_cohort


def store_at(root: Path) -> ArtifactStore:
    return ArtifactStore(Resolver(Roots.single(root)))


@pytest.mark.parametrize("kind", ["tiles", "features"])
@pytest.mark.parametrize("key", ["tile_224", "tiles_224"])
def test_legacy_arrays_ignore_unused_hdf5_contents(tmp_path, kind, key):
    store = store_at(tmp_path)
    artifact = "tiles" if kind == "tiles" else "tile_features"
    keys = {"size_mm": 0.25, "image_id": "slide", **({"model": "uni"} if kind == "features" else {})}
    path = store.path(artifact, **keys)
    expected = np.ones((3, 2, 2, 3), np.uint8) if kind == "tiles" else np.ones((3, 5), np.float32)
    write({key: expected}, path, "hdf5")
    with h5py.File(path, "a") as file:
        # A group cannot be read with [()]; eager legacy loading would fail here.
        file.create_group("unused_legacy_geometry")
        file["coords"] = np.ones((1, 1))
    payload = store.read(artifact, **keys)
    assert set(payload) == {kind}
    np.testing.assert_array_equal(payload[kind], expected)
    if kind == "features":
        assert _peek_feature_dim(path) == 5
        np.testing.assert_array_equal(_read_hdf5_features(path), expected)


@pytest.mark.parametrize("artifact", ["tiles", "tile_augmentation"])
@pytest.mark.parametrize("n", [0, 7])
def test_default_pixel_storage_and_subset_access(tmp_path, artifact, n):
    store = store_at(tmp_path)
    expected = np.arange(n * 2 * 3 * 3, dtype=np.uint8).reshape(n, 2, 3, 3)
    path = store.write(artifact, {"tiles": expected, "coords": np.ones((n, 2))},
                       size_mm=0.25, image_id="slide")
    with h5py.File(path, "r") as file:
        assert set(file) == {"tiles"}
        dataset = array_dataset(file, "tiles")
        assert dataset.shape == (n,)
        assert dataset.dtype.subdtype == (np.dtype(np.uint8), (2, 3, 3))
        selection = np.array([0, 2, 6]) if n else np.array([], dtype=int)
        np.testing.assert_array_equal(dataset[selection], expected[selection])
    np.testing.assert_array_equal(store.read(artifact, size_mm=0.25, image_id="slide")["tiles"], expected)


def test_native_pixel_key_precedes_legacy_alias(tmp_path):
    path = write({"tiles": np.zeros((2, 3, 3, 3), np.uint8), "tile_224": np.zeros((1, 4))},
                 tmp_path / "input.hdf5", "hdf5")
    with h5py.File(path, "r") as file:
        assert array_dataset(file, "tiles").shape == (2,)


@pytest.mark.parametrize("payload", [
    {},
    {"tile_224": np.zeros((2, 4)), "tiles_224": np.zeros((2, 4))},
    {"features": np.zeros((2, 0))},
    {"features": np.zeros((2, 4, 3))},
])
def test_invalid_or_ambiguous_feature_arrays_raise(tmp_path, payload):
    path = write(payload, tmp_path / "input.hdf5", "hdf5")
    with h5py.File(path, "r") as file, pytest.raises(ValueError):
        array_dataset(file, "features")


def test_full_tables_store_no_redundant_identity_or_size(tmp_path):
    store = store_at(tmp_path)
    keys = {"size_mm": 0.25, "image_id": "slide"}
    table = pd.DataFrame({
        "tile_id": [0, 1, 2], "tile_x": [0, 4, 8], "tile_y": [0, 0, 0], "size_mm": [0.25] * 3,
    })
    path = store.write("tile_metadata", table, **keys)
    assert pd.read_parquet(path).columns.tolist() == ["tile_x", "tile_y"]
    pd.testing.assert_frame_equal(store.read("tile_metadata", **keys), table)
    flags = pd.DataFrame({"tile_id": [0, 1, 2], "tile_background": [True, False, False]})
    background = store.write("tile_background", flags, **keys)
    assert pd.read_parquet(background).columns.tolist() == ["tile_background"]
    features = store.write("tile_features", {"features": np.ones((3, 4)), "tile_ids": np.arange(3)},
                           model="uni", **keys)
    bag, ids = _load_inference_bag(features, background)
    assert len(bag) == 2
    assert ids.tolist() == [1, 2], "filtering must keep original row ordinals, not renumber the tiles"
    with h5py.File(features, "r") as file:
        assert set(file) == {"features"}


def test_legacy_background_grid_columns_do_not_change_row_identity(tmp_path):
    store = store_at(tmp_path)
    keys = {"size_mm": 0.25, "image_id": "slide"}
    path = store.path("tile_background", **keys)
    write(pd.DataFrame({"tile_row": [3, 2, 1], "tile_col": [0, 0, 0],
                        "tile_background": [False, True, False]}), path, "parquet")
    features = store.write("tile_features", {"features": np.ones((3, 4))}, model="uni", **keys)
    bag, ids = _load_inference_bag(features, path)
    assert len(bag) == 2
    assert ids.tolist() == [0, 2]


@pytest.mark.parametrize("ids", [[0, 2], [1, 0], [1, 2]])
def test_partial_or_reordered_full_table_is_rejected(tmp_path, ids):
    store = store_at(tmp_path)
    with pytest.raises(ValueError, match="ordinals"):
        store.write("tile_metadata", pd.DataFrame({"tile_id": ids}), size_mm=0.25, image_id="slide")
    with pytest.raises(ValueError, match="ordinals"):
        store.write("tile_metadata", pd.DataFrame({"tile_x": [0, 1]}, index=ids),
                    size_mm=0.25, image_id="slide")


def test_float_partitions_resolve_existing_spelling_for_all_store_operations(tmp_path):
    store = store_at(tmp_path)
    keys = {"size_mm": 0.25, "model": "uni", "image_id": "slide"}
    path = tmp_path / "preprocessing/features/size_mm=0.250/model=uni/slide/slide.hdf5"
    write({"tile_224": np.ones((3, 4))}, path, "hdf5")
    assert store.path("tile_features", **keys) == path
    assert store.exists("tile_features", **keys)
    assert store.read("tile_features", **keys)["features"].shape == (3, 4)
    assert store.write("tile_features", {"features": np.zeros((3, 4))}, **keys) == path
    assert not (tmp_path / "preprocessing/features/size_mm=0.25").exists()
    store.remove("tile_features", **keys)
    assert not path.exists()


def test_float_partitions_are_exact_and_string_labels_are_opaque(tmp_path):
    resolver = Resolver(Roots.single(tmp_path))
    root = tmp_path / "preprocessing/tiles"
    (root / "size_mm=0.25000000000001").mkdir(parents=True)
    assert resolver.path("tiles", size_mm=0.25, image_id="slide") == root / "size_mm=0.25/slide/slide.hdf5"
    assert format_hive_value(0.123456789) == "0.123456789"
    pre = tmp_path / "preprocessing/features/size_mm=0.250"
    (pre / "model=1.00").mkdir(parents=True)
    path = resolver.path("tile_features", size_mm=0.25, model="1.0", image_id="slide",
                         extra_hive={"threshold": 0.5})
    assert path == pre / "model=1.0/slide/threshold=0.5/slide.hdf5"
    (path.parent.parent / "threshold=0.500").mkdir(parents=True)
    assert resolver.path("tile_features", size_mm=0.25, model="1.0", image_id="slide",
                         extra_hive={"threshold": 0.5}) == path.parent.parent / "threshold=0.500/slide.hdf5"


def test_equivalent_partition_ambiguity_fails_even_with_exact_match(tmp_path):
    store = store_at(tmp_path)
    for spelling in ("0.25", "0.250"):
        (tmp_path / f"preprocessing/tiles/size_mm={spelling}").mkdir(parents=True)
    keys = {"size_mm": 0.25, "image_id": "slide"}
    for operation in (store.path, store.exists, store.read, store.remove):
        with pytest.raises(ValueError, match="Ambiguous"):
            operation("tiles", **keys)
    with pytest.raises(ValueError, match="Ambiguous"):
        store.write("tiles", {"tiles": np.zeros((1, 2, 2, 3), np.uint8)}, **keys)


def test_mask_only_segments_image_and_does_not_touch_other_artifacts(tmp_path, monkeypatch):
    root = make_cohort(tmp_path / "source", n_images=1)
    store = store_at(root)
    image = pd.read_csv(root / "metadata/image_metadata.csv").image_id.iloc[0]
    thumbnail = np.full((20, 40, 3), 255, np.uint8)
    thumbnail[:, :20] = [120, 30, 90]
    store.write("thumbnail", thumbnail, image_id=image)
    keys = {"size_mm": 0.25, "image_id": image}
    store.write("tile_background", pd.DataFrame({"tile_background": [False, False]}), **keys)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr("helios.stages.tiles.extract_tiles", lambda *a, **k: pytest.fail("must not extract tiles"))
    monkeypatch.setattr("helios.stages.tiles.detect_background",
                        lambda *a, **k: pytest.fail("must not classify tiles"))
    prep_tiles(datasets=[root], size_mm=[0.25], background_mask_only=True)
    np.testing.assert_array_equal(store.read("background_mask", **keys), segment_background(thumbnail) * 255)
    for path, content in before.items():
        assert path.read_bytes() == content
    assert not (root / "preprocessing/tiles").exists()
    assert not (root / "preprocessing/tile_metadata").exists()
    mask = store.path("background_mask", **keys)
    mtime = mask.stat().st_mtime_ns
    prep_tiles(datasets=[root], size_mm=[0.25], background_mask_only=True)
    assert mask.stat().st_mtime_ns == mtime
    prep_tiles(datasets=[root], size_mm=[0.25], background_mask_only=True, overwrite=True)
    for path, content in before.items():
        assert path.read_bytes() == content


def test_mask_only_cli_passes_flag(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr("helios.stages.tiles.prep_tiles", lambda **kw: seen.append(kw))
    monkeypatch.setattr("helios.cli._common.console_progress", lambda *a, **kw: None)
    result = CliRunner().invoke(app, ["prep", "tiles", "--dataset", str(tmp_path), "--background-mask-only"])
    assert result.exit_code == 0, result.output
    assert seen[0]["background_mask_only"] is True


def test_mil_checks_shared_metadata_row_count(tmp_path):
    root = make_cohort(tmp_path / "source", n_images=1)
    store = store_at(root)
    image = pd.read_csv(root / "metadata/image_metadata.csv").image_id.iloc[0]
    keys = {"size_mm": 0.25, "image_id": image}
    store.write("tile_metadata", pd.DataFrame({"tile_x": [0, 1]}), **keys)
    store.write("tile_features", {"features": np.ones((3, 4))}, model="virchow2", **keys)
    with pytest.raises(ValueError, match="2 rows"):
        fit_mil(datasets=[root])


def test_legacy_tile_feature_extraction_and_persistence(tmp_path, monkeypatch):
    root = make_cohort(tmp_path / "source", n_images=1)
    store = store_at(root)
    image = pd.read_csv(root / "metadata/image_metadata.csv").image_id.iloc[0]
    path = root / f"preprocessing/tiles/size_mm=0.250/{image}/{image}.hdf5"
    write({"tile_224": np.zeros((3, 2, 2, 3), np.uint8)}, path, "hdf5")
    keys = {"size_mm": 0.25, "image_id": image}
    store.write("tile_metadata", pd.DataFrame({"tile_x": [0, 1, 2]}), **keys)
    monkeypatch.setattr("helios.stages.tile_features.extract_features",
                        lambda tiles, **kw: np.ones((len(tiles), 4), np.float32))
    prep_tile_features(datasets=[root])
    output = store.path("tile_features", model="virchow2", **keys)
    with h5py.File(output, "r") as file:
        assert set(file) == {"features"}
    store.write("tile_metadata", pd.DataFrame({"tile_x": [0, 1]}), **keys)
    with pytest.raises(StageExecutionError, match="2 rows"):
        prep_tile_features(datasets=[root], overwrite=True)


def test_partial_extraction_cannot_be_persisted_as_full_grid(tmp_path):
    from PIL import Image

    path = tmp_path / "slide.tiff"
    Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(path)
    metadata, _ = extract_tiles(str(path), size_mm=0.004, tile_px=4, image_mpp=1, start=2, stop=4)
    assert metadata.tile_id.tolist() == [2, 3]
    with pytest.raises(ValueError, match="ordinals"):
        store_at(tmp_path).write("tile_metadata", metadata, size_mm=0.004, image_id="slide")
