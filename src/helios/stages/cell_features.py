"""Stage: cellular composition features (``helios prep cell-features``).

Aggregates the cell pipeline outputs into per-tile counts (``tile_cell_counts``,
per size_mm) and then per-slide cellular features (``cellular_features``). These
features feed the cellular risk model.

The implementations live in :mod:`helios.components.cellular`; this stage only
drives them per slide and writes artifacts.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from helios.components.cellular import aggregate_tile_cells, compute_cellular_features
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
    tag_dataset,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_SIZE_MM: float = 0.25
DEFAULT_TISSUE_MIN_CELLS: int = 64
DEFAULT_TUMOR_PERCENTILE: float = 75.0
DEFAULT_LYMPH_FRAC_THRESHOLD: float = 0.05
DEFAULT_MITOSIS_MIN_COUNT: int = 1


def _resolve_single_size_mm(size_mm: float | list[float]) -> float:
    if isinstance(size_mm, list):
        if len(size_mm) != 1:
            raise ValueError("prep_cell_features requires a single size_mm value.")
        return float(size_mm[0])
    return float(size_mm)


def _threshold_hive(
    *,
    tissue_min_cells: int,
    tumor_percentile: float,
    lymph_frac_threshold: float,
    mitosis_min_count: int,
) -> dict[str, object]:
    extra: dict[str, object] = {}
    if tissue_min_cells != DEFAULT_TISSUE_MIN_CELLS:
        extra["tissue_min_cells"] = tissue_min_cells
    if tumor_percentile != DEFAULT_TUMOR_PERCENTILE:
        extra["tumor_percentile"] = tumor_percentile
    if lymph_frac_threshold != DEFAULT_LYMPH_FRAC_THRESHOLD:
        extra["lymph_frac_threshold"] = lymph_frac_threshold
    if mitosis_min_count != DEFAULT_MITOSIS_MIN_COUNT:
        extra["mitosis_min_count"] = mitosis_min_count
    return extra


def prep_cell_features(
    *,
    datasets: DatasetArgs,
    roots: StorageRoots | None = None,
    parallel: ParallelConfig | None = None,
    size_mm: float | list[float] = DEFAULT_SIZE_MM,
    tissue_min_cells: int = DEFAULT_TISSUE_MIN_CELLS,
    tumor_percentile: float = DEFAULT_TUMOR_PERCENTILE,
    lymph_frac_threshold: float = DEFAULT_LYMPH_FRAC_THRESHOLD,
    mitosis_min_count: int = DEFAULT_MITOSIS_MIN_COUNT,
    image_ids: list[str] | None = None,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Aggregate per-tile cell counts and compute per-slide cellular features.

    GRAIN CONTRACT: per-tile counts and per-slide cellular features are computed
    for a single ``size_mm`` per slide, persisted as ``cellular_features.csv.parts``
    and reduced into ``cellular_features.csv``. Requires the cell pipeline
    (``prep cells``) and tile metadata (``prep tiles``).
    """
    resolved_size = _resolve_single_size_mm(size_mm)
    extra_hive = _threshold_hive(
        tissue_min_cells=tissue_min_cells,
        tumor_percentile=tumor_percentile,
        lymph_frac_threshold=lymph_frac_threshold,
        mitosis_min_count=mitosis_min_count,
    )
    table_keys: dict[str, Any] = {"extra_hive": extra_hive, "size_mm": resolved_size}
    thresholds = {
        "tissue_min_cells": tissue_min_cells,
        "tumor_percentile": tumor_percentile,
        "lymph_frac_threshold": lymph_frac_threshold,
        "mitosis_min_count": mitosis_min_count,
    }

    for ds in resolve_datasets(datasets, roots=roots):
        store = cohort_store(ds, roots)
        run_work_units(
            select_image_ids(ds, store, image_ids),
            partial(_cell_features, store=store, table_keys=table_keys, thresholds=thresholds, force=force),
            stage=f"cell-features {ds.name}",
            progress=progress,
            parallel=parallel,
        )

        # Reduce over the whole cohort (not just this run's selection) so a partial
        # re-run never overwrites the table with a subset.
        cohort_ids = select_image_ids(ds, store, None)
        reduced = store.read_parts("cellular_features", cohort_ids, **table_keys)
        store.write("cellular_features", tag_dataset(reduced, ds.name), **table_keys)
        progress.log(f"wrote cellular_features {ds.name} ({len(reduced)} of {len(cohort_ids)} images)")


def _cell_features(
    image_id: str,
    ctx: WorkContext,
    *,
    store: ArtifactStore,
    table_keys: dict[str, Any],
    thresholds: dict[str, Any],
    force: bool,
) -> str:
    size_mm = table_keys["size_mm"]
    counts_ok = store.exists("tile_cell_counts", size_mm=size_mm, image_id=image_id)
    if not force and counts_ok and store.part_exists("cellular_features", image_id, **table_keys):
        ctx.progress.log(f"skip cell-features image={image_id} (exists)")
        return "skipped"
    if force or not counts_ok:
        cells = store.read("cells", image_id=image_id)
        mitotic = store.read("mitotic_cells", image_id=image_id)
        tile_meta = store.read("tile_metadata", size_mm=size_mm, image_id=image_id)
        counts = aggregate_tile_cells(cells, mitotic, tile_meta)
        store.write("tile_cell_counts", counts, size_mm=size_mm, image_id=image_id)
    else:
        counts = store.read("tile_cell_counts", size_mm=size_mm, image_id=image_id)
    features = compute_cellular_features(counts, **thresholds).assign(image_id=image_id)
    store.write_part("cellular_features", image_id, features, **table_keys)
    return "completed"
