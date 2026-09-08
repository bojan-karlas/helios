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
    saturation_threshold: int = 15,
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
        Maximum saturation for the primary bright-glass rule. The remaining
        background and artifact ranges follow the reference tile extractor.

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

    # Debug mask at thumbnail resolution. Tile classification below repeats the
    # reference extractor's morphology at its fixed 100x100 analysis scale.
    background = _background_pixels(hsv, saturation_threshold=saturation_threshold)
    artifact = _artifact_pixels(hsv)
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
        region = rgb[max(0, y0) : min(thumb_height, y1), max(0, x0) : min(thumb_width, x1)]
        values.append(
            bool(
                region.size == 0
                or _tile_background_fraction(
                    region, saturation_threshold=saturation_threshold
                )
                > 0.5
            )
        )

    flags = pd.DataFrame(
        {
            "tile_id": tile_metadata["tile_id"].to_numpy(),
            "tile_background": np.asarray(values, dtype=bool),
        }
    )
    return flags, background


def _background_pixels(
    hsv: NDArray[np.uint8], *, saturation_threshold: int
) -> NDArray[np.bool_]:
    """Reference bright-glass HSV ranges."""
    hue, saturation, value = (hsv[..., index] for index in range(3))
    return np.asarray(
        ((saturation <= saturation_threshold) & (value >= 215))
        | ((saturation <= 5) & (value >= 245))
        | ((hue >= 25) & (hue <= 180) & (saturation <= 55) & (value >= 229)),
        dtype=bool,
    )


def _artifact_pixels(hsv: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Reference blue/green-ink and dark-artifact HSV ranges."""
    hue, saturation, value = (hsv[..., index] for index in range(3))
    return np.asarray(
        ((hue >= 25) & (hue <= 180) & (value <= 220))
        | ((saturation <= 225) & (value <= 70)),
        dtype=bool,
    )


def _tile_background_fraction(
    rgb: NDArray[np.uint8], *, saturation_threshold: int
) -> float:
    """Classify one tile region at the reference extractor's 100px scale."""
    from PIL import Image
    from scipy import ndimage

    analysis_rgb = np.asarray(Image.fromarray(rgb).resize((100, 100)), dtype=np.uint8)
    hsv = np.asarray(Image.fromarray(analysis_rgb).convert("HSV"), dtype=np.uint8)

    background = _background_pixels(hsv, saturation_threshold=saturation_threshold)
    background = ndimage.median_filter(background, size=5)
    background = ndimage.minimum_filter(background, size=20)

    # Erosion removes isolated colored tissue pixels; dilation makes a
    # sufficiently large ink core reject the surrounding tile, as in the
    # source extractor.
    artifact = ndimage.minimum_filter(_artifact_pixels(hsv), size=10)
    artifact = ndimage.maximum_filter(artifact, size=100)
    return float(np.mean(background | artifact))
