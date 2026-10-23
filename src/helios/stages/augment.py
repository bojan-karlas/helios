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

from helios.components.augmentation import AUGMENTATION_METHODS, augment_stains
from helios.components.features import FEATURE_MODELS, extract_features
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    AugmentationSpec,
    DatasetArgs,
    ParallelConfig,
    ResolvedDataset,
    StorageRoots,
    cohort_store,
    resolve_datasets,
    select_image_ids,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_METHOD: str = "cyclegan"
DEFAULT_SOURCE: str | None = None
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
    roots: StorageRoots | None = None,
    parallel: ParallelConfig | None = None,
    method: str = DEFAULT_METHOD,
    source: str | None = DEFAULT_SOURCE,
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

    GRAIN CONTRACT: one (dataset, augmentation_spec, size_mm, image) per
    augmentation unit; augmented features are then extracted per
    (augmentation_spec, size_mm, model, image). Requires staged ``tiles``
    (``prep tiles``).
    """
    resolved = resolve_datasets(datasets, roots=roots)
    _validate_models(model)

    for ds in resolved:
        specs = _specs_for_dataset(
            ds,
            method=method,
            source=source,
            targets=target,
            checkpoints=checkpoint,
            stain_model_root=stain_model_root,
        )
        store = cohort_store(ds, roots)
        ids = select_image_ids(ds, store, image_ids)
        units = [(spec, s, i) for spec in specs for s in size_mm for i in ids]

        for spec, s, image_id in progress.task(units, desc=f"augment {ds.name}"):
            keys = {**spec.key_values(), "size_mm": s, "image_id": image_id}
            if force or not store.exists("tile_augmentation", **keys):
                tiles = _tiles_array(store.read("tiles", size_mm=s, image_id=image_id))
                augmented = augment_stains(
                    tiles,
                    checkpoint=str(spec.checkpoint),
                    batch_size=augmentation_batch_size,
                    device=device,
                )
                store.write("tile_augmentation", {"tiles": augmented}, **keys)
                progress.log(
                    f"wrote tile_augmentation image={image_id} method={spec.method} "
                    f"source={spec.source} target={spec.target} size_mm={s}"
                )

            for m in model:
                feat_keys = {**keys, "model": m}
                if not force and store.exists("tile_augmentation_features", **feat_keys):
                    continue
                aug_tiles = _tiles_array(store.read("tile_augmentation", **keys))
                embeddings = extract_features(
                    aug_tiles, model=m, batch_size=batch_size, device=device, progress=progress  # type: ignore[arg-type]
                )
                store.write("tile_augmentation_features", {"features": embeddings}, **feat_keys)
                progress.log(
                    f"wrote tile_augmentation_features image={image_id} method={spec.method} "
                    f"source={spec.source} target={spec.target} size_mm={s} model={m}"
                )


def _validate_models(model: list[str]) -> None:
    unknown_m = [m for m in model if m not in FEATURE_MODELS]
    if unknown_m:
        raise ValueError(f"Unknown feature model(s) {unknown_m}. Available: {list(FEATURE_MODELS)}.")


def _specs_for_dataset(
    ds: ResolvedDataset,
    *,
    method: str,
    source: str | None,
    targets: list[str],
    checkpoints: list[str],
    stain_model_root: str | Path,
) -> list[AugmentationSpec]:
    if method not in AUGMENTATION_METHODS:
        raise ValueError(
            f"Unsupported augmentation method {method!r}. Available: {list(AUGMENTATION_METHODS)}."
        )
    if targets:
        resolved_ckpts = _resolve_checkpoints(targets, checkpoints, stain_model_root)
        if len(targets) != len(resolved_ckpts):
            raise ValueError(
                "target and checkpoint must have the same number of values "
                "(one user-trained G_A checkpoint per stain target)."
            )
        if len(set(targets)) != len(targets):
            raise ValueError("stain target names must be unique")
        return [
            AugmentationSpec(method=method, source=source, target=t, checkpoint=Path(c))
            for t, c in zip(targets, resolved_ckpts, strict=True)
        ]
    if ds.augmentations:
        out: list[AugmentationSpec] = []
        root = Path(stain_model_root)
        for spec in ds.augmentations:
            if spec.method not in AUGMENTATION_METHODS:
                raise ValueError(
                    f"Unsupported augmentation method {spec.method!r}. "
                    f"Available: {list(AUGMENTATION_METHODS)}."
                )
            ckpt = spec.checkpoint or (root / spec.target / "latest_net_G_A.pth")
            out.append(
                AugmentationSpec(
                    method=spec.method,
                    source=spec.source,
                    target=spec.target,
                    checkpoint=Path(ckpt),
                )
            )
        return out
    return []


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
