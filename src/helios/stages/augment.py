"""Prepare self-described augmentation directories and aligned embeddings."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import cast

from helios.components.augmentation import AUGMENTATION_METHODS, apply_augmentation
from helios.components.features import FEATURE_MODELS, FeatureModel, extract_features
from helios.data.augmentation import Augmentation
from helios.data.io import ArtifactStore
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._ray import WorkContext, gpu_reservation, run_work_units
from helios.stages._runtime import (
    DatasetArgs,
    ParallelConfig,
    ResolvedDataset,
    StorageRoots,
    cohort_store,
    resolve_datasets,
    select_image_ids,
)
from helios.stages.tile_features import _tiles_array, _validate_tile_count

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
    output_dir: str | None = None,
    size_mm: list[float] = DEFAULT_SIZE_MM,
    model: list[str] = DEFAULT_MODELS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    augmentation_batch_size: int = DEFAULT_AUGMENTATION_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    image_ids: list[str] | None = None,
    overwrite: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Prepare tiles and features; existing artifacts are skipped unless overwritten."""
    resolved = resolve_datasets(datasets, roots=roots)
    if output_dir is not None and (len(resolved) != 1 or len(target) != 1):
        raise ValueError("output_dir requires exactly one dataset and one explicit target.")
    if checkpoint and not target:
        raise ValueError("Explicit checkpoints require explicit targets.")
    unknown = [m for m in model if m not in FEATURE_MODELS]
    if unknown:
        raise ValueError(f"Unknown feature model(s) {unknown}. Available: {list(FEATURE_MODELS)}.")

    for ds in resolved:
        store = cohort_store(ds, roots)
        definitions = _definitions(
            ds,
            roots=roots,
            method=method,
            source=source,
            targets=target,
            checkpoints=checkpoint,
            stain_model_root=Path(stain_model_root),
            output_dir=output_dir,
        )
        ids = select_image_ids(ds, store, image_ids)
        if not definitions:
            progress.log(
                f"no augmentations selected for {ds.name}: pass --target (with --checkpoint) or attach "
                "prepared augmentation directories in the cohort YAML; nothing to prepare.",
                level=logging.WARNING,
            )
            continue
        for definition in definitions:
            if definition.method not in AUGMENTATION_METHODS:
                raise ValueError(f"Unsupported augmentation method {definition.method!r}.")
            if definition.source_dataset_id != ds.dataset().descriptor.id:
                raise ValueError(f"Augmentation {definition.root} belongs to a different dataset.")
            if definition.checkpoint is None:
                raise ValueError(f"Augmentation {definition.root} requires a checkpoint for preparation.")
            manifest = definition.root / "augmentation.yaml"
            if manifest.exists():
                previous = Augmentation.load(definition.root)
                if previous.source_dataset_id != definition.source_dataset_id:
                    raise ValueError(f"Augmentation {definition.root} belongs to a different dataset.")
                if previous != definition:
                    progress.log(
                        f"Reusing {definition.root} with changed settings; "
                        "existing outputs are skipped unless overwrite=True. "
                        "Mixed outputs are the user's responsibility.",
                        level=logging.WARNING,
                    )
            definition.save()
            run_work_units(
                [(s, i) for s in size_mm for i in ids],
                partial(
                    _augment_slide,
                    store=store,
                    definition=definition,
                    augmentation_batch_size=augmentation_batch_size,
                    device=device,
                    overwrite=overwrite,
                ),
                stage=f"augment {ds.name}",
                unit_id=_augmentation_unit_id,
                progress=progress,
                parallel=parallel,
                num_gpus=gpu_reservation(device, parallel),
                persistent_workers=True,
            )
            run_work_units(
                [(s, m, i) for m in model for s in size_mm for i in ids],
                partial(
                    _embed_augmentation,
                    definition=definition,
                    batch_size=batch_size,
                    device=device,
                    overwrite=overwrite,
                ),
                stage=f"augment features {ds.name}",
                unit_id=_augmentation_feature_id,
                progress=progress,
                parallel=parallel,
                num_gpus=gpu_reservation(device, parallel),
                persistent_workers=True,
            )


def _augmentation_unit_id(unit: tuple[float, str]) -> str:
    return f"size_mm={unit[0]}/{unit[1]}"


def _augmentation_feature_id(unit: tuple[float, str, str]) -> str:
    return f"size_mm={unit[0]}/model={unit[1]}/{unit[2]}"


def _augment_slide(
    unit: tuple[float, str],
    ctx: WorkContext,
    *,
    store: ArtifactStore,
    definition: Augmentation,
    augmentation_batch_size: int,
    device: str,
    overwrite: bool,
) -> str:
    s, image_id = unit
    aug_store = definition.store()
    if not overwrite and aug_store.exists("tile_augmentation", size_mm=s, image_id=image_id):
        return "skipped"
    tiles = _tiles_array(store.read("tiles", size_mm=s, image_id=image_id))
    _validate_tile_count(store, size_mm=s, image_id=image_id, n_tiles=len(tiles))
    augmented = apply_augmentation(
        tiles,
        method=definition.method,
        checkpoint=str(definition.checkpoint),
        batch_size=augmentation_batch_size,
        device=device,
    )
    if len(augmented) != len(tiles):
        raise ValueError("Augmentation changed the number of tiles.")
    aug_store.write(
        "tile_augmentation",
        {"tiles": augmented},
        size_mm=s,
        image_id=image_id,
    )
    ctx.progress.log(f"prepared augmentation={definition.root} image={image_id} size_mm={s}")
    return "completed"


def _embed_augmentation(
    unit: tuple[float, str, str],
    ctx: WorkContext,
    *,
    definition: Augmentation,
    batch_size: int,
    device: str,
    overwrite: bool,
) -> str:
    s, m, image_id = unit
    store = definition.store()
    if not overwrite and store.exists("tile_augmentation_features", size_mm=s, image_id=image_id, model=m):
        return "skipped"
    payload = store.read("tile_augmentation", size_mm=s, image_id=image_id)
    embeddings = extract_features(
        payload["tiles"],
        model=cast(FeatureModel, m),
        batch_size=batch_size,
        device=device,
        progress=ctx.progress,
    )
    if len(embeddings) != len(payload["tiles"]):
        raise ValueError("Feature extraction changed the number of tiles.")
    store.write(
        "tile_augmentation_features",
        {"features": embeddings},
        size_mm=s,
        image_id=image_id,
        model=m,
    )
    return "completed"


def _definitions(
    ds: ResolvedDataset,
    *,
    roots: StorageRoots | None,
    method: str,
    source: str | None,
    targets: list[str],
    checkpoints: list[str],
    stain_model_root: Path,
    output_dir: str | None,
) -> list[Augmentation]:
    if not targets:
        return [Augmentation.load(path) for path in ds.augmentations]
    if len(set(targets)) != len(targets):
        raise ValueError("Stain target names must be unique.")
    if checkpoints and len(checkpoints) != len(targets):
        raise ValueError("target and checkpoint must have the same number of values.")
    for value in (method, source, *targets):
        if value is not None and (not value.strip() or "/" in value or "\\" in value or value in {".", ".."}):
            raise ValueError(f"Invalid augmentation directory label: {value!r}.")
    base = cohort_store(ds, roots).resolver.roots.for_role("preprocessing") / "augmentations"
    parts = [f"method={method}"]
    if source is not None:
        parts.append(f"source={source}")
    return [
        Augmentation(
            root=Path(output_dir).absolute() if output_dir is not None else base.joinpath(*parts, f"target={t}"),
            source_dataset_id=ds.dataset().descriptor.id,
            method=method,
            parameters={**({"source": source} if source is not None else {}), "target": t},
            checkpoint=(
                Path(checkpoints[i]) if checkpoints else stain_model_root / t / "latest_net_G_A.pth"
            ).absolute(),
        )
        for i, t in enumerate(targets)
    ]
