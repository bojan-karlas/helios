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

from helios.components.augmentation import augment_stains
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
DEFAULT_TARGETS: list[str] = []
DEFAULT_CHECKPOINTS: list[str] = []
DEFAULT_STAIN_MODEL_ROOT: str = "src/helios/models/multistain_cyclegan"
DEFAULT_SIZE_MM: list[float] = [0.25]
DEFAULT_MODELS: list[str] = ["virchow2"]
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_AUGMENTATION_BATCH_SIZE: int = 16
DEFAULT_DEVICE: str = "cuda"


def prep_augment(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    target: list[str] = DEFAULT_TARGETS,
    checkpoint: list[str] = DEFAULT_CHECKPOINTS,
    stain_model_root: str | Path = DEFAULT_STAIN_MODEL_ROOT,
    size_mm: list[float] = DEFAULT_SIZE_MM,
    model: list[str] = DEFAULT_MODELS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    augmentation_batch_size: int = DEFAULT_AUGMENTATION_BATCH_SIZE,
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
    checkpoints = _resolve_checkpoints(target, checkpoint, stain_model_root)
    _validate(target, checkpoints, model)

    for ds in resolved:
        store = cohort_store(ds.root, output_root)
        ids = select_image_ids(ds, store, image_ids)
        units = [(t, c, s, i) for t, c in zip(target, checkpoints, strict=True) for s in size_mm for i in ids]

        for t, checkpoint_path, s, image_id in progress.task(units, desc=f"augment {ds.name}"):
            if force or not store.exists("tile_augmentation", target=t, size_mm=s, image_id=image_id):
                tiles = _tiles_array(store.read("tiles", size_mm=s, image_id=image_id))
                augmented = augment_stains(
                    tiles,
                    checkpoint=checkpoint_path,
                    batch_size=augmentation_batch_size,
                    device=device,
                )
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


def _validate(target: list[str], checkpoint: list[str], model: list[str]) -> None:
    if len(target) != len(checkpoint):
        raise ValueError(
            "target and checkpoint must have the same number of values "
            "(one user-trained G_A checkpoint per stain target)."
        )
    if len(set(target)) != len(target):
        raise ValueError("stain target names must be unique")
    unknown_m = [m for m in model if m not in FEATURE_MODELS]
    if unknown_m:
        raise ValueError(f"Unknown feature model(s) {unknown_m}. Available: {list(FEATURE_MODELS)}.")


def _resolve_checkpoints(
    targets: list[str], checkpoints: list[str], stain_model_root: str | Path
) -> list[str]:
    """Use explicit paths or the conventional per-target local model path."""
    if checkpoints:
        return checkpoints
    root = Path(stain_model_root)
    return [str(root / target / "latest_net_G_A.pth") for target in targets]


def _tiles_array(tiles: object) -> NDArray[np.uint8]:
    if isinstance(tiles, dict):
        if "tiles" not in tiles:
            raise KeyError("tile artifact has no 'tiles' dataset")
        tiles = tiles["tiles"]
    return np.asarray(tiles, dtype=np.uint8)
