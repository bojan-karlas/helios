"""Component: whole-slide tiling.

Partition a slide into a fixed-physical-size tile grid: record per-tile geometry
and cut the tile image stack. The owning stage (:mod:`helios.stages.tiles`) opens
the slide and writes the ``tile_metadata`` + ``tiles`` artifacts; this function
computes the grid and reads the pixels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def extract_tiles(
    wsi_path: str,
    *,
    size_mm: float,
    tile_px: int = 224,
    image_mpp: float | None = None,
) -> tuple[pd.DataFrame, NDArray[np.uint8]]:
    """Tile one slide at a target physical resolution.

    GRAIN: one (slide, size_mm) per call.

    Parameters
    ----------
    wsi_path:
        Path to the slide file.
    size_mm:
        Target tile resolution in microns-per-pixel (drives the pyramid level /
        resampling).
    tile_px:
        Output tile edge length in pixels.
    image_mpp:
        Native slide microns-per-pixel; if ``None`` it is read from the slide.

    Returns
    -------
    tuple
        ``(tile_metadata, tiles)`` where ``tile_metadata`` has one row per tile
        (``tile_id, row, col, x, y, w, h, size_mm``) and ``tiles`` is the matching
        ``(n_tiles, tile_px, tile_px, 3)`` uint8 RGB stack (row ``i`` ↔ tile ``i``).
    """
    raise NotImplementedError(
        "Compute the tile grid at size_mm and read each tile's pixels from the WSI."
    )
