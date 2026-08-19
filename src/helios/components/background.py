"""Component: background detection.

Color-based tissue/background segmentation on the slide thumbnail, then map each
tile onto the mask to set a per-tile background flag. The owning stage
(:mod:`helios.stages.tiles`) reads the thumbnail + tile metadata and writes the
``tile_background`` flags and the ``background_mask`` debug image.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def detect_background(
    thumbnail: NDArray[np.uint8],
    tile_metadata: pd.DataFrame,
    *,
    saturation_threshold: int = 20,
) -> tuple[pd.DataFrame, NDArray[np.bool_]]:
    """Flag background tiles from a thumbnail-derived tissue mask.

    GRAIN: one (slide, size_mm) per call.

    Parameters
    ----------
    thumbnail:
        Full-slide RGB thumbnail ``(H, W, 3)`` uint8.
    tile_metadata:
        Per-tile geometry (``tile_id, x, y, w, h, ...``) at the same slide scale.
    saturation_threshold:
        Pixels below this HSV saturation count as background.

    Returns
    -------
    tuple
        ``(tile_background, background_mask)`` where ``tile_background`` has one
        row per tile (``tile_id, tile_background``) and ``background_mask`` is the
        boolean thumbnail-resolution tissue mask (debug image).
    """
    if thumbnail.ndim != 3 or thumbnail.shape[-1] < 3:
        raise ValueError("thumbnail must have shape (H, W, 3)")
    required = {"tile_id", "tile_x", "tile_y", "tile_width", "tile_height"}
    missing = required - set(tile_metadata.columns)
    if missing:
        raise ValueError(f"tile_metadata is missing columns: {sorted(missing)}")

    from PIL import Image
    from scipy import ndimage

    rgb = np.asarray(thumbnail[..., :3], dtype=np.uint8)
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.uint8)
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    # Port the reference extractor's dominant rules: pale bright glass and
    # very dark scanner/artifact pixels are both unsuitable for embeddings.
    background = (saturation <= saturation_threshold) & (value >= 215)
    artifact = value <= 70
    background = ndimage.median_filter(background | artifact, size=5).astype(bool)

    if tile_metadata.empty:
        flags = pd.DataFrame({"tile_id": pd.Series(dtype=int), "tile_background": pd.Series(dtype=bool)})
        return flags, background

    slide_width = int((tile_metadata["tile_x"] + tile_metadata["tile_width"]).max())
    slide_height = int((tile_metadata["tile_y"] + tile_metadata["tile_height"]).max())
    thumb_height, thumb_width = background.shape
    values: list[bool] = []
    for row in tile_metadata.itertuples(index=False):
        x0 = int(np.floor(row.tile_x * thumb_width / slide_width))
        x1 = int(np.ceil((row.tile_x + row.tile_width) * thumb_width / slide_width))
        y0 = int(np.floor(row.tile_y * thumb_height / slide_height))
        y1 = int(np.ceil((row.tile_y + row.tile_height) * thumb_height / slide_height))
        region = background[max(0, y0) : min(thumb_height, y1), max(0, x0) : min(thumb_width, x1)]
        values.append(bool(region.size == 0 or np.mean(region) > 0.5))

    flags = pd.DataFrame(
        {
            "tile_id": tile_metadata["tile_id"].to_numpy(),
            "tile_background": np.asarray(values, dtype=bool),
        }
    )
    return flags, background
