"""Stage: stain augmentation + augmented feature extraction (``helios prep augment``).

For every (target, size_mm, image) this stage stain-augments the slide's tiles
with a pretrained CycleGAN (``tile_augmentation``) and then embeds the augmented
tiles with the foundation model(s) (``tile_augmentation_features``). The
augmented embeddings are what MIL training swaps in for stain robustness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from helios.components.augmentation import STAIN_TARGETS, augment_stains
from helios.components.features import FEATURE_MODELS, extract_features
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    DatasetArgs,
    cohort_store,
    resolve_datasets,
    select_image_ids,
    single_output_guard,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_TARGETS: list[str] = ["MGB+MRV", "VISIOMEL"]
DEFAULT_SIZE_MM: list[float] = [0.25]
DEFAULT_MODELS: list[str] = ["virchow2"]
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_DEVICE: str = "cuda"


def prep_augment(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    target: list[str] = DEFAULT_TARGETS,
    size_mm: list[float] = DEFAULT_SIZE_MM,
    model: list[str] = DEFAULT_MODELS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    image_ids: list[str] | None = None,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Stain-augment tiles and embed the augmented tiles.

    GRAIN CONTRACT: one (dataset, target, size_mm, image) per augmentation unit;
    augmented features are then extracted per (target, size_mm, model, image).
    Requires staged ``tiles`` (``prep tiles``).
    """
    resolved = resolve_datasets(datasets)
    single_output_guard(resolved, output_root)
    _validate(target, model)

    for ds in resolved:
        store = cohort_store(ds.root, output_root)
        ids = select_image_ids(ds, store, image_ids)
        units = [(t, s, i) for t in target for s in size_mm for i in ids]

        for t, s, image_id in progress.task(units, desc=f"augment {ds.name}"):
            if force or not store.exists("tile_augmentation", target=t, size_mm=s, image_id=image_id):
                tiles = _tiles_array(store.read("tiles", size_mm=s, image_id=image_id))
                augmented = augment_stains(tiles, target=t, device=device)  # type: ignore[arg-type]
                store.write(
                    "tile_augmentation", {"tiles": augmented}, target=t, size_mm=s, image_id=image_id
                )
                progress.log(f"wrote tile_augmentation image={image_id} target={t} size_mm={s}")

            for m in model:
                if not force and store.exists(
                    "tile_augmentation_features", target=t, size_mm=s, model=m, image_id=image_id
                ):
                    continue
                aug_tiles = _tiles_array(
                    store.read("tile_augmentation", target=t, size_mm=s, image_id=image_id)
                )
                embeddings = extract_features(
                    aug_tiles, model=m, batch_size=batch_size, device=device, progress=progress  # type: ignore[arg-type]
                )
                store.write(
                    "tile_augmentation_features",
                    {"features": embeddings},
                    target=t,
                    size_mm=s,
                    model=m,
                    image_id=image_id,
                )
                progress.log(
                    f"wrote tile_augmentation_features image={image_id} target={t} size_mm={s} model={m}"
                )


def _validate(target: list[str], model: list[str]) -> None:
    unknown_t = [t for t in target if t not in STAIN_TARGETS]
    if unknown_t:
        raise ValueError(f"Unknown stain target(s) {unknown_t}. Available: {list(STAIN_TARGETS)}.")
    unknown_m = [m for m in model if m not in FEATURE_MODELS]
    if unknown_m:
        raise ValueError(f"Unknown feature model(s) {unknown_m}. Available: {list(FEATURE_MODELS)}.")


def _tiles_array(tiles: object) -> NDArray[np.uint8]:
    if isinstance(tiles, dict):
        if "tiles" not in tiles:
            raise KeyError("tile artifact has no 'tiles' dataset")
        tiles = tiles["tiles"]
    return np.asarray(tiles, dtype=np.uint8)
