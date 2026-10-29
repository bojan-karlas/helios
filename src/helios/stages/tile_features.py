"""Stage: foundation tile-feature extraction (``helios prep tile-features``).

Reads the ``tiles`` artifact, embeds each slide's tiles with one or more
foundation models (via :func:`helios.components.features.extract_features`), and
writes the ``tile_features`` artifact. Runs over one or more datasets.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from numpy.typing import NDArray

from helios.components.features import IMPLEMENTED_FEATURE_MODELS, extract_features
from helios.data.io import ArtifactStore
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._ray import gpu_reservation
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

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_SIZE_MM: list[float] = [0.25]
DEFAULT_MODELS: list[str] = ["virchow2"]
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_DEVICE: str = "cuda"


def prep_tile_features(
    *,
    datasets: DatasetArgs,
    roots: StorageRoots | None = None,
    parallel: ParallelConfig | None = None,
    size_mm: list[float] = DEFAULT_SIZE_MM,
    model: list[str] = DEFAULT_MODELS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    image_ids: list[str] | None = None,
    overwrite: bool = False,
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
        roots unless overridden.
    roots:
        Role-specific storage overrides. Tile inputs and embedding outputs
        use the preprocessing root.
    parallel:
        Ray execution settings; ``None`` runs slides serially in this process.
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
    overwrite:
        Re-extract even if a ``tile_features`` artifact already exists.
    progress:
        Progress sink (default: no-op).
    """
    _validate_models(model)
    gpu_budget = gpu_reservation(device, parallel)

    for ds in resolve_datasets(datasets, roots=roots):
        store = cohort_store(ds, roots)
        ids = select_image_ids(ds, store, image_ids)
        run_work_units(
            [(s, m, i) for s in size_mm for m in model for i in ids],
            partial(_embed, store=store, batch_size=batch_size, device=device, overwrite=overwrite),
            stage=f"tile-features {ds.name}",
            unit_id=_unit_id,
            progress=progress,
            parallel=parallel,
            num_gpus=gpu_budget,
            persistent_workers=True,
        )


def _unit_id(unit: tuple[float, str, str]) -> str:
    return f"size_mm={unit[0]}/model={unit[1]}/{unit[2]}"


def _embed(
    unit: tuple[float, str, str],
    ctx: WorkContext,
    *,
    store: ArtifactStore,
    batch_size: int,
    device: str,
    overwrite: bool,
) -> str:
    s, m, image_id = unit
    if not overwrite and store.exists("tile_features", size_mm=s, model=m, image_id=image_id):
        ctx.progress.log(f"skip tile_features image={image_id} size_mm={s} model={m} (exists)")
        return "skipped"
    tiles = _tiles_array(store.read("tiles", size_mm=s, image_id=image_id))
    _validate_tile_count(store, size_mm=s, image_id=image_id, n_tiles=len(tiles))
    embeddings = extract_features(
        tiles, model=m, batch_size=batch_size, device=device, progress=ctx.progress  # type: ignore[arg-type]
    )
    store.write(
        "tile_features",
        {"features": embeddings},
        size_mm=s,
        model=m,
        image_id=image_id,
    )
    ctx.progress.log(f"wrote tile_features image={image_id} size_mm={s} model={m}")
    return "completed"


def _validate_models(model: list[str]) -> None:
    unknown = [m for m in model if m not in IMPLEMENTED_FEATURE_MODELS]
    if unknown:
        raise ValueError(f"Unknown feature model(s): {unknown}. Available: {list(IMPLEMENTED_FEATURE_MODELS)}.")


def _tiles_array(tiles: object) -> NDArray[np.uint8]:
    """Coerce a read ``tiles`` artifact to the ``(N, H, W, 3)`` uint8 stack."""
    if isinstance(tiles, dict):
        if "tiles" in tiles:
            tiles = tiles["tiles"]
        else:
            raise KeyError("tiles artifact has no 'tiles' dataset")
    return np.asarray(tiles, dtype=np.uint8)


def _validate_tile_count(store: ArtifactStore, *, size_mm: float, image_id: str, n_tiles: int) -> None:
    """Check full-array row count against the shared geometry when available."""
    if not store.exists("tile_metadata", size_mm=size_mm, image_id=image_id):
        return
    metadata = store.read("tile_metadata", size_mm=size_mm, image_id=image_id)
    if len(metadata) != n_tiles:
        raise ValueError(f"tile_metadata has {len(metadata)} rows but tiles has {n_tiles} rows for image {image_id!r}.")
