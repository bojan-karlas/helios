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
    stride_fraction: float = 0.5,
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
    stride_fraction:
        Grid stride as a fraction of the source tile edge. The default ``0.5``
        matches the reference extractor's 50% overlap.
    image_mpp:
        Native slide microns-per-pixel; if ``None`` it is read from the slide.

    Returns
    -------
    tuple
        ``(tile_metadata, tiles)`` where ``tile_metadata`` has one row per tile
        (``tile_id, row, col, x, y, w, h, size_mm``) and ``tiles`` is the matching
        ``(n_tiles, tile_px, tile_px, 3)`` uint8 RGB stack (row ``i`` ↔ tile ``i``).
    """
    if size_mm <= 0:
        raise ValueError("size_mm must be positive")
    if tile_px <= 0:
        raise ValueError("tile_px must be positive")
    if not 0 < stride_fraction <= 1:
        raise ValueError("stride_fraction must be in (0, 1]")

    from PIL import Image

    slide, width, height, native_mpp, close = _open_slide(wsi_path)
    try:
        mpp = float(image_mpp) if image_mpp is not None else native_mpp
        if mpp is None or not np.isfinite(mpp) or mpp <= 0:
            raise ValueError("image_mpp is required when the slide has no valid MPP metadata")
        source_px = max(1, round(size_mm * 1000.0 / mpp))
        stride_px = max(1, round(source_px * stride_fraction))
        rows: list[dict[str, int | float]] = []
        images: list[NDArray[np.uint8]] = []
        tile_id = 0
        for tile_row, y in enumerate(range(0, height - source_px + 1, stride_px)):
            for tile_col, x in enumerate(range(0, width - source_px + 1, stride_px)):
                tile = slide(x, y, source_px)
                resized = Image.fromarray(tile).resize((tile_px, tile_px), Image.Resampling.BILINEAR)
                images.append(np.asarray(resized.convert("RGB"), dtype=np.uint8))
                rows.append(
                    {
                        "tile_id": tile_id,
                        "tile_row": tile_row,
                        "tile_col": tile_col,
                        "tile_x": x,
                        "tile_y": y,
                        "tile_width": source_px,
                        "tile_height": source_px,
                        "size_mm": size_mm,
                    }
                )
                tile_id += 1
        metadata = pd.DataFrame(rows, columns=[
            "tile_id", "tile_row", "tile_col", "tile_x", "tile_y",
            "tile_width", "tile_height", "size_mm",
        ])
        stack = np.stack(images) if images else np.empty((0, tile_px, tile_px, 3), dtype=np.uint8)
        return metadata, stack
    finally:
        close()


def _open_slide(wsi_path: str):
    """Return a level-0 RGB region reader and its slide metadata."""
    if wsi_path.lower().endswith((".tif", ".tiff")):
        import zarr
        from tifffile import TiffFile

        tif = TiffFile(wsi_path)
        store = tif.series[0].aszarr(level=0)
        array = zarr.open(store, mode="r")
        height, width = array.shape[:2]

        def read_tiff_region(x: int, y: int, size: int) -> NDArray[np.uint8]:
            from helios.components.thumbnails import _as_rgb

            return _as_rgb(np.asarray(array[y : y + size, x : x + size]))

        def close() -> None:
            close_store = getattr(store, "close", None)
            if close_store is not None:
                close_store()
            tif.close()

        return read_tiff_region, int(width), int(height), None, close

    from openslide import OpenSlide

    slide = OpenSlide(wsi_path)
    width, height = slide.dimensions
    raw_mpp = slide.properties.get("openslide.mpp-x")
    native_mpp = float(raw_mpp) if raw_mpp is not None else None

    def read_openslide_region(x: int, y: int, size: int) -> NDArray[np.uint8]:
        return np.asarray(slide.read_region((x, y), 0, (size, size)).convert("RGB"), dtype=np.uint8)

    return read_openslide_region, width, height, native_mpp, slide.close
