"""Stage: cellular composition features (``helios prep cell-features``).

Aggregates the cell pipeline outputs into per-tile counts (``tile_cell_counts``,
per size_mm) and then per-slide cellular features (``cellular_features``). These
features feed the cellular risk model.

The implementations live in :mod:`helios.components.cellular`; this stage only
drives them per slide and writes artifacts.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from helios.components.cellular import aggregate_tile_cells, compute_cellular_features
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    DatasetArgs,
    cohort_store,
    resolve_datasets,
    select_image_ids,
    single_output_guard,
    tag_dataset,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_SIZE_MM: list[float] = [0.25]

def prep_cell_features(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    size_mm: list[float] = DEFAULT_SIZE_MM,
    image_ids: list[str] | None = None,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Aggregate per-tile cell counts and compute per-slide cellular features.

    GRAIN CONTRACT: per-tile counts are computed per (size_mm, image); the
    per-slide feature row pools the size_mm grids for that image. Requires the
    cell pipeline (``prep cells``) and tile metadata (``prep tiles``).
    """
    resolved = resolve_datasets(datasets)
    single_output_guard(resolved, output_root)

    for ds in resolved:
        store = cohort_store(ds.root, output_root)
        ids = select_image_ids(ds, store, image_ids)

        feature_rows: list[pd.DataFrame] = []
        for image_id in progress.task(ids, desc=f"cell-features {ds.name}"):
            cells = store.read("cells", image_id=image_id)
            mitotic = store.read("mitotic_cells", image_id=image_id)
            per_size: list[pd.DataFrame] = []
            for s in size_mm:
                if force or not store.exists("tile_cell_counts", size_mm=s, image_id=image_id):
                    tile_meta = store.read("tile_metadata", size_mm=s, image_id=image_id)
                    counts = aggregate_tile_cells(cells, mitotic, tile_meta)
                    store.write("tile_cell_counts", counts, size_mm=s, image_id=image_id)
                else:
                    counts = store.read("tile_cell_counts", size_mm=s, image_id=image_id)
                per_size.append(counts)
            features = compute_cellular_features(pd.concat(per_size, ignore_index=True))
            feature_rows.append(features.assign(image_id=image_id))

        # cellular_features is a cohort-level table (one row per image): write once.
        store.write("cellular_features", tag_dataset(pd.concat(feature_rows, ignore_index=True), ds.name))
        progress.log(f"wrote cellular_features {ds.name} ({len(feature_rows)} images)")