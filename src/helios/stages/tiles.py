"""Stage: tiling + background removal (``helios prep tiles``).

For every (image, size_mm) this stage cuts the tile grid (``tile_metadata`` +
``tiles``) and then flags background tiles from the thumbnail (``tile_background``
+ ``background_mask``). Tiling and background detection are chained here because
background flags are computed against the same grid; features are extracted
separately (``prep tile-features``) so changing the background algorithm never
forces re-extraction.
"""
from __future__ import annotations

from pathlib import Path

from helios.components.background import detect_background
from helios.components.tiling import extract_tiles
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    DatasetArgs,
    cohort_store,
    resolve_datasets,
    select_image_ids,
    single_output_guard,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_SIZE_MM: list[float] = [0.0625, 0.125, 0.25]
DEFAULT_TILE_PX: int = 224
DEFAULT_STRIDE_FRACTION: float = 0.5


def prep_tiles(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    size_mm: list[float] = DEFAULT_SIZE_MM,
    tile_px: int = DEFAULT_TILE_PX,
    stride_fraction: float = DEFAULT_STRIDE_FRACTION,
    image_ids: list[str] | None = None,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Tile each slide at each resolution and flag background tiles.

    GRAIN CONTRACT: one (dataset, size_mm, image) per work unit. Tiling runs
    once per unit; background detection reuses the tile grid + slide thumbnail.
    The thumbnail must already exist (``prep thumbnails``).
    """
    resolved = resolve_datasets(datasets)
    single_output_guard(resolved, output_root)

    for ds in resolved:
        cohort = ds.dataset()
        store = cohort_store(ds.root, output_root)
        ids = select_image_ids(ds, store, image_ids)
        metadata = cohort.image_metadata.set_index("image_id")
        units = [(s, i) for s in size_mm for i in ids]

        for s, image_id in progress.task(units, desc=f"tiles {ds.name}"):
            if not force and store.exists("tiles", size_mm=s, image_id=image_id):
                progress.log(f"skip tiles image={image_id} size_mm={s} (exists)")
                continue
            image_mpp = metadata.loc[image_id, "image_mpp"] if "image_mpp" in metadata.columns else None
            meta, tiles = extract_tiles(
                str(cohort.slides[image_id]),
                size_mm=s,
                tile_px=tile_px,
                stride_fraction=stride_fraction,
                image_mpp=image_mpp,
            )
            store.write("tile_metadata", meta, size_mm=s, image_id=image_id)
            store.write("tiles", {"tiles": tiles}, size_mm=s, image_id=image_id)

            thumbnail = store.read("thumbnail", image_id=image_id)
            flags, mask = detect_background(thumbnail, meta)
            store.write("tile_background", flags, size_mm=s, image_id=image_id)
            store.write("background_mask", mask, image_id=image_id)
            progress.log(f"wrote tiles + background image={image_id} size_mm={s}")
