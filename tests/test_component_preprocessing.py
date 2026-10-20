"""Tests for thumbnail, tiling, background, and PNG artifact preprocessing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from helios.components.background import detect_background
from helios.components.thumbnails import extract_thumbnail
from helios.components.tiling import extract_tiles
from helios.data.io import read, write
from helios.stages._runtime import cohort_store
from helios.stages.tiles import prep_tiles


def _write_tiff(path: Path) -> np.ndarray:
    image = np.full((8, 12, 3), 255, dtype=np.uint8)
    image[:, :6] = (120, 30, 90)
    Image.fromarray(image).save(path)
    return image


def test_thumbnail_and_tiles_from_tiff(tmp_path: Path) -> None:
    path = tmp_path / "slide.tiff"
    _write_tiff(path)

    thumbnail = extract_thumbnail(str(path), max_size=6)
    metadata, tiles = extract_tiles(str(path), size_mm=0.004, tile_px=4, image_mpp=1.0)

    assert thumbnail.shape == (4, 6, 3)
    assert tiles.shape == (15, 4, 4, 3)
    assert metadata.columns.tolist() == [
        "tile_id", "tile_row", "tile_col", "tile_x", "tile_y",
        "tile_width", "tile_height", "size_mm",
    ]
    assert metadata["tile_id"].tolist() == list(range(len(metadata)))


def test_background_flags_align_with_tile_ids() -> None:
    thumbnail = np.full((10, 20, 3), 255, dtype=np.uint8)
    thumbnail[:, :10] = (120, 30, 90)
    metadata = pd.DataFrame(
        {
            "tile_id": [0, 1],
            "tile_x": [0, 10],
            "tile_y": [0, 0],
            "tile_width": [10, 10],
            "tile_height": [10, 10],
        }
    )

    flags, mask = detect_background(thumbnail, metadata)

    assert flags["tile_background"].tolist() == [False, True]
    assert mask.dtype == np.bool_


def test_colored_ink_is_rejected_without_removing_purple_tissue() -> None:
    thumbnail = np.full((100, 300, 3), 255, dtype=np.uint8)
    thumbnail[:, :100] = (120, 30, 90)  # H&E-like purple tissue
    thumbnail[:, 100:200] = (20, 80, 180)  # blue annotation ink
    metadata = pd.DataFrame(
        {
            "tile_id": [0, 1, 2],
            "tile_x": [0, 100, 200],
            "tile_y": [0, 0, 0],
            "tile_width": [100, 100, 100],
            "tile_height": [100, 100, 100],
        }
    )

    flags, mask = detect_background(thumbnail, metadata)

    assert flags["tile_background"].tolist() == [False, True, True]
    assert mask[:, 100:200].mean() == 1.0


def test_background_mask_rasterizes_final_ink_tile_decision() -> None:
    thumbnail = np.full((100, 200, 3), (120, 30, 90), dtype=np.uint8)
    thumbnail[40:60, 140:160] = (20, 80, 180)
    metadata = pd.DataFrame(
        {
            "tile_id": [0, 1],
            "tile_x": [0, 100],
            "tile_y": [0, 0],
            "tile_width": [100, 100],
            "tile_height": [100, 100],
        }
    )

    flags, mask = detect_background(thumbnail, metadata)

    assert flags["tile_background"].tolist() == [False, True]
    assert not mask[:, :100].any()
    assert mask[:, 100:].all(), "the PNG must show the final expanded tile rejection"


def test_png_artifacts_round_trip_arrays(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    expected = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    write(expected, path, "png")

    np.testing.assert_array_equal(read(path, "png"), expected)


def test_tiling_preserves_background_masks_for_each_size(tmp_path: Path) -> None:
    (tmp_path / "wsi").mkdir()
    (tmp_path / "metadata").mkdir()
    image = _write_tiff(tmp_path / "wsi" / "slide.tiff")
    pd.DataFrame({"image_id": ["slide"], "image_mpp": [1.0]}).to_csv(
        tmp_path / "metadata" / "image_metadata.csv", index=False
    )
    store = cohort_store(tmp_path)
    store.write("thumbnail", image, image_id="slide")
    sizes = [0.004, 0.008]

    prep_tiles(datasets=[tmp_path], size_mm=sizes, tile_px=4)

    paths = []
    for size in sizes:
        path = store.path("background_mask", size_mm=size, image_id="slide")
        assert path == tmp_path / "debug" / "background_mask" / f"size_mm={size}" / "slide.png"
        paths.append(path)
        flags, expected_mask = detect_background(
            image, store.read("tile_metadata", size_mm=size, image_id="slide")
        )
        np.testing.assert_array_equal(
            store.read("background_mask", size_mm=size, image_id="slide"), expected_mask * 255
        )
        pd.testing.assert_frame_equal(
            store.read("tile_background", size_mm=size, image_id="slide"), flags
        )
    assert len(set(paths)) == len(sizes)
    assert all(path.is_file() for path in paths)
    assert not (tmp_path / "debug" / "background_mask" / "slide.png").exists()
