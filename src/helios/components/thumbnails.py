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
    from PIL import Image

    path = str(wsi_path)
    if path.lower().endswith((".tif", ".tiff")):
        from tifffile import TiffFile

        with TiffFile(path) as tif:
            levels = tif.series[0].levels
            candidates = [level for level in levels if max(level.shape[-3:-1]) >= max_size]
            level = min(candidates or levels, key=lambda item: max(item.shape[-3:-1]))
            image = Image.fromarray(_as_rgb(level.asarray()))
    else:
        from openslide import OpenSlide

        with OpenSlide(path) as slide:
            image = slide.get_thumbnail((max_size, max_size)).convert("RGB")

    image.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _as_rgb(array: NDArray[np.generic]) -> NDArray[np.uint8]:
    """Normalize common TIFF channel layouts to an RGB uint8 image."""
    data = np.asarray(array)
    while data.ndim > 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim == 1 and data.shape[0] in {3, 4}:
        data = data.reshape(1, 1, data.shape[0])
    if data.ndim == 3 and data.shape[0] in {3, 4} and data.shape[-1] not in {3, 4}:
        data = np.moveaxis(data, 0, -1)
    if data.ndim == 2:
        data = np.repeat(data[..., None], 3, axis=-1)
    if data.ndim != 3 or data.shape[-1] not in {3, 4}:
        raise ValueError(f"Unsupported TIFF image shape: {data.shape}")
    if data.dtype != np.uint8:
        maximum = float(np.max(data)) if data.size else 0.0
        scale = 255.0 / maximum if maximum > 255.0 else 1.0
        data = np.clip(data.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return np.asarray(data[..., :3], dtype=np.uint8)
