"""Stage: foundation tile-feature extraction (``helios prep tile-features``).

Reads the ``tiles`` artifact, embeds each slide's tiles with one or more
foundation models (via :func:`helios.components.features.extract_features`), and
writes the ``tile_features`` artifact. Runs over one or more datasets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from helios.components.features import IMPLEMENTED_FEATURE_MODELS, extract_features
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    DatasetArgs,
    cohort_store,
    resolve_datasets,
    select_image_ids,
    single_output_guard,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_SIZE_MM: list[float] = [0.25]
DEFAULT_MODELS: list[str] = ["virchow2"]
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_DEVICE: str = "cuda"


def prep_tile_features(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    size_mm: list[float] = DEFAULT_SIZE_MM,
    model: list[str] = DEFAULT_MODELS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    image_ids: list[str] | None = None,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Extract foundation features for every (dataset, size_mm, model, image).

    GRAIN CONTRACT: this stage iterates work units of ``(dataset, size_mm, model,
    image_id)`` and calls ``extract_features`` once per unit (one slide's tiles).
    If the component's grain changes (e.g. batched across slides), change the
    loop here to match.

    Parameters
    ----------
    datasets:
        One or more cohort roots, each holding ``metadata/image_metadata.csv``
        and staged ``tiles``. Each dataset's outputs are written under its own
        root unless ``output_root`` redirects them.
    output_root:
        Optional separate base for derived artifacts; the source cohort stays
        read-only and ``tile_features`` are written here instead of in place.
        Only valid with a single dataset (the redirect target is unambiguous).
    size_mm:
        Tile resolutions (microns-per-pixel target) to process.
    model:
        Foundation models to run; each must be in
        :data:`~helios.components.features.FEATURE_MODELS`.
    batch_size:
        Tiles per encoder forward pass (passed to the component).
    device:
        Torch device string for the encoder.
    image_ids:
        Restrict to these images (within each dataset); ``None`` processes the
        whole cohort.
    force:
        Re-extract even if a ``tile_features`` artifact already exists.
    progress:
        Progress sink (default: no-op).
    """
    _validate_models(model)
    resolved = resolve_datasets(datasets)
    single_output_guard(resolved, output_root)

    for ds in resolved:
        store = cohort_store(ds.root, output_root)
        ids = select_image_ids(ds, store, image_ids)
        units = [(s, m, i) for s in size_mm for m in model for i in ids]

        for s, m, image_id in progress.task(units, desc=f"tile-features {ds.name}"):
            if not force and store.exists("tile_features", size_mm=s, model=m, image_id=image_id):
                progress.log(f"skip tile_features image={image_id} size_mm={s} model={m} (exists)")
                continue
            tiles = _tiles_array(store.read("tiles", size_mm=s, image_id=image_id))
            tile_ids, coords = _tile_index(store, size_mm=s, image_id=image_id, n_tiles=len(tiles))
            embeddings = extract_features(
                tiles, model=m, batch_size=batch_size, device=device, progress=progress  # type: ignore[arg-type]
            )
            store.write(
                "tile_features",
                {"features": embeddings, "tile_ids": tile_ids, "coords": coords},
                size_mm=s,
                model=m,
                image_id=image_id,
            )
            progress.log(f"wrote tile_features image={image_id} size_mm={s} model={m}")


def _validate_models(model: list[str]) -> None:
    unknown = [m for m in model if m not in IMPLEMENTED_FEATURE_MODELS]
    if unknown:
        raise ValueError(
            f"Unknown feature model(s): {unknown}. "
            f"Available: {list(IMPLEMENTED_FEATURE_MODELS)}."
        )


def _tiles_array(tiles: object) -> NDArray[np.uint8]:
    """Coerce a read ``tiles`` artifact to the ``(N, H, W, 3)`` uint8 stack."""
    if isinstance(tiles, dict):
        if "tiles" in tiles:
            tiles = tiles["tiles"]
        else:
            raise KeyError("tiles artifact has no 'tiles' dataset")
    return np.asarray(tiles, dtype=np.uint8)


def _tile_index(store, *, size_mm: float, image_id: str, n_tiles: int):
    """Return row-aligned tile IDs and level-0 coordinates for MIL/provenance."""
    if not store.exists("tile_metadata", size_mm=size_mm, image_id=image_id):
        ids = np.arange(n_tiles, dtype=np.int64)
        return ids, np.empty((n_tiles, 0), dtype=np.int64)
    metadata = store.read("tile_metadata", size_mm=size_mm, image_id=image_id)
    required = {"tile_id", "tile_x", "tile_y"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"tile_metadata is missing columns: {sorted(missing)}")
    if len(metadata) != n_tiles:
        raise ValueError(
            f"tile_metadata has {len(metadata)} rows but tiles has {n_tiles} rows "
            f"for image {image_id!r}."
        )
    ids = metadata["tile_id"].to_numpy(dtype=np.int64)
    if len(np.unique(ids)) != n_tiles:
        raise ValueError(f"tile_metadata contains duplicate tile_id values for image {image_id!r}.")
    coords = metadata[["tile_x", "tile_y"]].to_numpy(dtype=np.int64)
    return ids, coords
