"""Stage: tiling + background removal (``helios prep tiles``).

For every (image, size_mm) this stage cuts the tile grid (``tile_metadata`` +
``tiles``) and then flags background tiles from the thumbnail (``tile_background``
+ ``background_mask``). Tiling and background detection are chained here because
background flags are computed against the same grid; features are extracted
separately (``prep tile-features``) so changing the background algorithm never
forces re-extraction.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path

import pandas as pd

from helios.components.background import detect_background
from helios.components.tiling import extract_tiles
from helios.data.io import ArtifactStore
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    DatasetArgs,
    ParallelConfig,
    StorageRoots,
    WorkContext,
    cohort_store,
    resolve_datasets,
    run_work_units,
    select_image_ids,
)

TILE_OUTPUTS = ("tile_metadata", "tiles", "tile_background", "background_mask")

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_SIZE_MM: list[float] = [0.0625, 0.125, 0.25]
DEFAULT_TILE_PX: int = 224
DEFAULT_STRIDE_FRACTION: float = 0.5


def prep_tiles(
    *,
    datasets: DatasetArgs,
    roots: StorageRoots | None = None,
    parallel: ParallelConfig | None = None,
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
    for ds in resolve_datasets(datasets, roots=roots):
        cohort = ds.dataset()
        store = cohort_store(ds, roots)
        ids = select_image_ids(ds, store, image_ids)
        run_work_units(
            [(s, i) for s in size_mm for i in ids],
            partial(
                _tile,
                store=store,
                slides=cohort.slides,
                metadata=cohort.image_metadata.set_index("image_id"),
                tile_px=tile_px,
                stride_fraction=stride_fraction,
                force=force,
            ),
            stage=f"tiles {ds.name}",
            unit_id=_unit_id,
            progress=progress,
            parallel=parallel,
        )


def _unit_id(unit: tuple[float, str]) -> str:
    return f"size_mm={unit[0]}/{unit[1]}"


def _tile(
    unit: tuple[float, str],
    ctx: WorkContext,
    *,
    store: ArtifactStore,
    slides: Mapping[str, Path],
    metadata: pd.DataFrame,
    tile_px: int,
    stride_fraction: float,
    force: bool,
) -> str:
    s, image_id = unit
    if not force and all(store.exists(a, size_mm=s, image_id=image_id) for a in TILE_OUTPUTS):
        ctx.progress.log(f"skip tiles image={image_id} size_mm={s} (exists)")
        return "skipped"
    # The last output doubles as the unit's completion marker: clear it before recomputing.
    store.remove("background_mask", size_mm=s, image_id=image_id)
    image_mpp = metadata.loc[image_id, "image_mpp"] if "image_mpp" in metadata.columns else None
    meta, tiles = extract_tiles(
        str(slides[image_id]),
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
    store.write("background_mask", mask, size_mm=s, image_id=image_id)
    ctx.progress.log(f"wrote tiles + background image={image_id} size_mm={s}")
    return "completed"
