"""Component: low-resolution slide thumbnail extraction.

Decode a downsampled full-slide RGB image straight from the WSI. The owning
stage (:mod:`helios.stages.thumbnails`) opens the slide and writes the
``thumbnail`` artifact; this function performs the decode/resize.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def extract_thumbnail(
    wsi_path: str,
    *,
    max_size: int = 2048,
) -> NDArray[np.uint8]:
    """Read a single slide and return a downsampled RGB thumbnail.

    GRAIN: one whole-slide image per call.

    Parameters
    ----------
    wsi_path:
        Path to the slide file (svs/tiff/ndpi).
    max_size:
        Longest-edge pixel budget for the thumbnail.

    Returns
    -------
    NDArray[np.uint8]
        RGB thumbnail of shape ``(H, W, 3)``.
    """
    raise NotImplementedError(
        "Open the WSI (e.g. openslide/tiffslide) and return a downsampled RGB thumbnail."
    )
