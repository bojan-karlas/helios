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


def test_png_artifacts_round_trip_arrays(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    expected = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    write(expected, path, "png")

    np.testing.assert_array_equal(read(path, "png"), expected)
