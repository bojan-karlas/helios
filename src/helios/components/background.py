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
        row per tile (``tile_id, is_background``) and ``background_mask`` is the
        boolean thumbnail-resolution tissue mask (debug image).
    """
    raise NotImplementedError(
        "Segment background on the thumbnail and set a per-tile background flag."
    )
