"""Stages: whole-slide A-MIL training & inference (``helios fit/predict mil``).

``fit mil`` pools one or more training cohorts and fits one A-MIL bundle per CV
fold. ``predict mil`` scores every image with each fold model (per-fold
embeddings / attention / whole-image risk) and then reduces the per-fold
whole-image scores to a single shipped value per image.
"""
from __future__ import annotations

from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from helios.components.features import FEATURE_MODELS
from helios.components.mil import SlideFeatureRef, infer_mil, infer_mil_score, reduce_whole_image_risk, train_mil
from helios.data.augmentation import Augmentation, select_augmentations, validate_feature_alignment
from helios.data.bundle import MANIFEST_NAME, load_bundle
from helios.data.cohort import ResolvedDataset
from helios.data.io import ArtifactStore
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._ray import WorkContext, gpu_reservation, run_work_units
from helios.stages._runtime import (
    DatasetArgs,
    ParallelConfig,
    StorageDirs,
    build_bundle_metadata,
    cohort_store,
    folds_across,
    heldout_ids,
    model_folds,
    read_splits,
    resolve_datasets,
    select_image_ids,
    tag_dataset,
    train_ids,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_SIZE_MM: float = 0.25
DEFAULT_MODEL: str = "virchow2"
DEFAULT_TARGET: str = "disease_pfs_recurred_5yfu"
DEFAULT_AUG_SWAP_PROB: float = 0.5
DEFAULT_AUG_MODE: Literal["replace", "augment", "separate"] = "replace"
DEFAULT_N_BRANCHES: int = 4
DEFAULT_NUM_HEADS: int = 4
DEFAULT_DROPOUT_RATE: float = 0.1
DEFAULT_LAMBDA_DIV: float = 0.01
DEFAULT_LR: float = 1e-4
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_MAX_EPOCHS: int = 20
DEFAULT_PATIENCE: int = 5
DEFAULT_VAL_FRAC: float = 0.15
DEFAULT_VAL_SOURCE: Literal["internal", "test_fold"] = "internal"
DEFAULT_VAL_AUG_MODE: Literal["originals", "separate"] = "originals"
DEFAULT_SEED: int = 42
DEFAULT_DEVICE: str = "cuda"
DEFAULT_SCORES_ONLY: bool = False


def fit_mil(
    *,
    datasets: DatasetArgs,
    dirs: StorageDirs | None = None,
    parallel: ParallelConfig | None = None,
    size_mm: float = DEFAULT_SIZE_MM,
    model: str = DEFAULT_MODEL,
    target: str = DEFAULT_TARGET,
    augmentations: list[str] | None = None,
    aug_swap_prob: float = DEFAULT_AUG_SWAP_PROB,
    aug_mode: Literal["replace", "augment", "separate"] = DEFAULT_AUG_MODE,
    n_branches: int = DEFAULT_N_BRANCHES,
    num_heads: int = DEFAULT_NUM_HEADS,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    lambda_div: float = DEFAULT_LAMBDA_DIV,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    val_frac: float = DEFAULT_VAL_FRAC,
    val_source: Literal["internal", "test_fold"] = DEFAULT_VAL_SOURCE,
    val_aug_mode: Literal["originals", "separate"] = DEFAULT_VAL_AUG_MODE,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
    overwrite: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Train one A-MIL model per fold over the pooled training cohorts.

    GRAIN CONTRACT: one fold per work unit. Each fold's train slides are pooled
    across every ``--dataset``; tile features are referenced by path (streamed
    lazily by the component). Writes ``model_mil`` keyed by fold. Each cohort's
    ``where`` filter optionally restricts training to a metadata subpopulation.

    ``val_source`` selects the slides that choose the best epoch: ``"internal"``
    carves ``val_frac`` out of the fold's train slides, ``"test_fold"`` trains on
    all of them and uses the fold's test slides. The latter requires a consistent
    fold assignment (see :func:`_check_test_fold_assignment`) and makes out-of-fold
    scores optimistic, because the test slides also choose the saved epoch.
    ``val_aug_mode="separate"`` also scores each validation slide's augmentations as
    separate bags, and requires an augmentation on every cohort.
    """
    if model not in FEATURE_MODELS:
        raise ValueError(f"Unknown feature model {model!r}. Available: {list(FEATURE_MODELS)}.")
    if val_source not in ("internal", "test_fold"):
        raise ValueError(f"Unknown val_source {val_source!r}. Available: ['internal', 'test_fold'].")
    if val_aug_mode not in ("originals", "separate"):
        raise ValueError(f"Unknown val_aug_mode {val_aug_mode!r}. Available: ['originals', 'separate'].")

    resolved = select_augmentations(
        resolve_datasets(datasets, dirs=dirs, produces=("models",), progress=progress), augmentations,
    )
    stores = [cohort_store(ds, dirs) for ds in resolved]
    out = stores[0]
    folds = folds_across(resolved, dirs)

    for ds in resolved:
        if val_aug_mode == "separate" and not ds.augmentations:
            raise ValueError(f"val_aug_mode='separate' requires an augmentation, but {ds.name} has none selected.")
    for ds in resolved:
        if target not in ds.dataset().image_metadata:
            raise ValueError(f"Missing MIL target column {target!r} in {ds.root}.")
    if val_source == "test_fold":
        _check_test_fold_assignment(resolved, folds, target=target, dirs=dirs)

    slides_by_fold: dict[int | None, list[SlideFeatureRef]] = {}
    val_slides_by_fold: dict[int | None, list[SlideFeatureRef]] = {}
    validated: set[tuple[Path, Path]] = set()
    for fold in folds:
        if not overwrite and out.output_exists("model_mil", fold=fold):
            progress.log(f"skip model_mil fold={fold} (exists)")
            continue
        slides: list[SlideFeatureRef] = []
        val_slides: list[SlideFeatureRef] = []
        for ds, store in zip(resolved, stores, strict=True):
            collect = partial(
                _slide_refs, ds, store, fold=fold, target=target, size_mm=size_mm, model=model,
                validated=validated, progress=progress,
            )
            slides.extend(collect(train_ids(ds, fold, where=ds.where, dirs=dirs), role="training"))
            if val_source == "test_fold":
                val_slides.extend(collect(heldout_ids(ds, fold, where=ds.where, dirs=dirs), role="test-fold"))
        slides_by_fold[fold] = slides
        if val_source == "test_fold":
            val_slides_by_fold[fold] = val_slides
    params = {
        "size_mm": size_mm,
        "model": model,
        "target": target,
        "augmentations": None,
        "aug_swap_prob": aug_swap_prob,
        "aug_mode": aug_mode,
        "n_branches": n_branches,
        "num_heads": num_heads,
        "dropout_rate": dropout_rate,
        "lambda_div": lambda_div,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "patience": patience,
        "val_frac": 0.0 if val_source == "test_fold" else val_frac,
        "val_source": val_source,
        "val_aug_mode": val_aug_mode,
        "seed": seed,
        "device": device,
    }
    run_work_units(
        list(slides_by_fold),
        partial(
            _fit_fold, slides_by_fold=slides_by_fold, val_slides_by_fold=val_slides_by_fold, out=out, params=params,
            resolved=resolved, dirs=dirs,
        ),
        stage="fit mil", progress=progress, parallel=parallel,
        num_gpus=gpu_reservation(device, parallel),
    )


def _slide_refs(
    ds: ResolvedDataset, store: ArtifactStore, image_ids: list[str], *, role: str, fold: int | None, target: str,
    size_mm: float, model: str, validated: set[tuple[Path, Path]], progress: Progress,
) -> list[SlideFeatureRef]:
    """Resolve and check the feature paths of one cohort's labeled ``image_ids``."""
    labels = ds.dataset().image_metadata.set_index("image_id")
    missing = labels.loc[image_ids, target].isna()
    if missing.any():
        progress.log(
            f"excluding {int(missing.sum())} unlabeled {role} images "
            f"dataset={ds.name} fold={fold} target={target}"
        )
    aug_stores = [Augmentation.load(path).store() for path in ds.augmentations]
    refs: list[SlideFeatureRef] = []
    for image_id in image_ids:
        if missing.loc[image_id]:
            continue
        label = _label(labels, image_id, target)
        aug_paths = tuple(
            aug_store.path(
                "tile_augmentation_features",
                size_mm=size_mm,
                model=model,
                image_id=image_id,
            )
            for aug_store in aug_stores
        )
        feat_path = store.path(
            "tile_features", size_mm=size_mm, model=model, image_id=image_id
        )
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Missing tile_features for {role} slide {image_id!r} in {ds.name}: {feat_path}"
            )
        _validate_feature_count(store, feat_path, size_mm=size_mm, image_id=image_id)
        for ap in aug_paths:
            if not ap.exists():
                raise FileNotFoundError(
                    f"Missing tile_augmentation_features for {role} slide {image_id!r} in {ds.name}: {ap}"
                )
            pair = (feat_path, ap)
            if pair not in validated:
                validate_feature_alignment(feat_path, ap)
                validated.add(pair)
        refs.append(
            SlideFeatureRef(
                image_id=image_id,
                features_path=feat_path,
                background_path=store.path("tile_background", size_mm=size_mm, image_id=image_id),
                aug_features_paths=aug_paths,
                label=label,
            )
        )
    return refs


def _check_test_fold_assignment(
    resolved: list[ResolvedDataset], folds: list[int | None], *, target: str, dirs: StorageDirs | None,
) -> None:
    """Refuse a fold assignment that cannot support selecting the epoch on the test fold.

    Over the pooled cohorts' labeled slides: there must be folds; in each fold, a slide
    cannot be both train and test, there must be train slides, and the test slides must
    contain both classes (AUPRC is undefined otherwise); and every slide trained on must
    be held out in exactly one fold, so each gets one out-of-fold score.
    """
    if folds == [None]:
        raise ValueError(
            "val_source='test_fold' requires a fold assignment. Provide one with --splits-file: "
            "a CSV with image_id and fold (the fold that holds the image out), or image_id, fold and split."
        )
    times_held_out: dict[str, int] = {}
    trained: set[str] = set()
    for fold in folds:
        train: set[str] = set()
        test: set[str] = set()
        classes: set[object] = set()
        for ds in resolved:
            labels = ds.dataset().image_metadata.set_index("image_id")[target]
            labeled = {str(i) for i in labels[labels.notna()].index}
            train |= labeled.intersection(train_ids(ds, fold, where=ds.where, dirs=dirs))
            held_out = labeled.intersection(heldout_ids(ds, fold, where=ds.where, dirs=dirs))
            test |= held_out
            classes |= set(labels.loc[sorted(held_out)])
        both = sorted(train & test)
        if both:
            raise ValueError(f"Fold {fold} has {len(both)} slide(s) in both train and test: {both[:5]}")
        if not train:
            raise ValueError(f"Fold {fold} has no training slides; does the fold assignment match the cohort?")
        if len(classes) < 2:
            raise ValueError(
                f"Fold {fold} has {len(test)} test slide(s) and {len(classes)} class(es) of {target!r}; "
                "selecting the best epoch by AUPRC needs both classes in the test fold."
            )
        trained |= train
        for image_id in test:
            times_held_out[image_id] = times_held_out.get(image_id, 0) + 1
    inconsistent = sorted(i for i in trained if times_held_out.get(i, 0) != 1)
    if inconsistent:
        raise ValueError(
            f"{len(inconsistent)} training slide(s) are not held out in exactly one fold: {inconsistent[:5]}"
        )


def _fit_fold(
    fold: int | None, ctx: WorkContext, *, slides_by_fold: dict[int | None, list[SlideFeatureRef]],
    val_slides_by_fold: dict[int | None, list[SlideFeatureRef]], out: ArtifactStore, params: dict[str, Any], resolved: list[ResolvedDataset], dirs: StorageDirs | None,
) -> str:
    slides = slides_by_fold[fold]
    training = {
        key: value for key, value in params.items()
        if key not in {"size_mm", "model", "target", "augmentations", "val_source"}
    }
    bundle = train_mil(slides, **training, progress=ctx.progress, fold=fold, val_slides=val_slides_by_fold.get(fold))
    meta = build_bundle_metadata("mil", params, resolved, dirs, fold=fold)
    out.write("model_mil", bundle, metadata=meta, fold=fold)
    ctx.progress.log(f"computed model_mil fold={fold} (n_train={len(slides)})")
    return "completed"


def predict_mil(
    *,
    datasets: DatasetArgs,
    dirs: StorageDirs | None = None,
    parallel: ParallelConfig | None = None,
    size_mm: float = DEFAULT_SIZE_MM,
    model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    scores_only: bool = DEFAULT_SCORES_ONLY,
    image_ids: list[str] | None = None,
    overwrite: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Score every image with each fold's A-MIL model, then reduce whole-image risk.

    GRAIN CONTRACT: per (dataset, fold, image) inference, then a per-cohort
    reduce of the K whole-image scores to ``whole_image_risk_score_ensemble``.
    Folds are enumerated from the saved ``model_mil`` bundles. Score-only mode
    skips attribution and saves only per-fold and reduced probabilities.
    """
    resolved = resolve_datasets(
        datasets, dirs=dirs, produces=("results",), progress=progress, parallel=parallel, shared_outputs=True,
    )

    groups: dict[tuple[str, Path | None], list[ResolvedDataset]] = {}
    for ds in resolved:
        destination = cohort_store(ds, dirs).output_path("whole_image_risk_score_ensemble")
        groups.setdefault((ds.logical_name, destination), []).append(ds)

    for constituents in groups.values():
        stores = [cohort_store(ds, dirs) for ds in constituents]
        out = stores[0]
        folds = model_folds(out, "model_mil")
        if not folds:
            raise FileNotFoundError(f"No model_mil bundle found under {constituents[0].name}.")
        if any(model_folds(store, "model_mil") != folds for store in stores[1:]):
            raise ValueError("Pooled MIL prediction requires the same model folds for every constituent.")
        split_tables = [read_splits(ds, dirs, progress=progress) for ds in constituents]
        if any(table is None for table in split_tables) and any(table is not None for table in split_tables):
            raise ValueError("Pooled MIL prediction cannot mix CV and ensemble-only constituents.")
        splits = (
            pd.concat([table for table in split_tables if table is not None], ignore_index=True).drop_duplicates()
            if split_tables[0] is not None else None
        )

        per_fold_scores: list[pd.DataFrame] = []
        for fold in folds:
            pieces: list[pd.DataFrame] = []
            for ds, store in zip(constituents, stores, strict=True):
                cohort_ids = select_image_ids(ds, store, None)
                ids = (
                    cohort_ids if image_ids is None
                    else [image_id for image_id in cohort_ids if image_id in image_ids]
                )
                run_work_units(
                    ids,
                    partial(_predict_slide, store=store, fold=fold, size_mm=size_mm, model=model,
                            device=device, overwrite=overwrite, scores_only=scores_only),
                    stage=f"mil infer fold={fold} {ds.name}", progress=progress, parallel=parallel,
                    num_gpus=gpu_reservation(device, parallel), persistent_workers=True,
                )
                # Include completed parts outside image_ids so partial reruns preserve the cohort.
                pieces.append(store.read_parts("whole_image_risk_score", cohort_ids, fold=fold))
            scores = pd.concat(pieces, ignore_index=True)
            if not scores.empty:
                out.write("whole_image_risk_score", scores, fold=fold)
                per_fold_scores.append(scores)

        ensemble = reduce_whole_image_risk(
            pd.concat(per_fold_scores, ignore_index=True) if per_fold_scores else pd.DataFrame(),
            splits,
        )
        tagged = [
            tag_dataset(ensemble[ensemble["image_id"].isin(select_image_ids(ds, store, None))], ds.name)
            for ds, store in zip(constituents, stores, strict=True)
        ]
        out.write("whole_image_risk_score_ensemble", pd.concat(tagged, ignore_index=True))
        progress.log(f"computed whole_image_risk_score_ensemble {constituents[0].logical_name}")


@lru_cache(maxsize=1)
def _cached_bundle(path: str, manifest_mtime_ns: int) -> Any:
    return load_bundle(Path(path))


def _load_inference_bundle(path: Path) -> Any:
    """Reuse the loaded bundle across slides; a refit (new manifest) invalidates it."""
    return _cached_bundle(str(path), (path / MANIFEST_NAME).stat().st_mtime_ns)


def _predict_slide(
    image_id: str, ctx: WorkContext, *, store: ArtifactStore, fold: int | None,
    size_mm: float, model: str, device: str, overwrite: bool,
    scores_only: bool = DEFAULT_SCORES_ONLY,
) -> str:
    complete = store.part_exists("whole_image_risk_score", image_id, fold=fold) and (scores_only or (
        store.output_exists("aligned_patch_embeddings", fold=fold, image_id=image_id)
        and store.output_exists("patch_attention", fold=fold, image_id=image_id)
        and store.output_exists("aligned_slide_embedding", fold=fold, image_id=image_id)
        and store.output_exists("aligned_slide_embedding_grad", fold=fold, image_id=image_id)
    ))
    if not overwrite and complete:
        return "skipped"
    mil_model = _load_inference_bundle(store.path("model_mil", fold=fold))
    features_path = store.path("tile_features", size_mm=size_mm, model=model, image_id=image_id)
    _validate_feature_count(store, features_path, size_mm=size_mm, image_id=image_id)
    if scores_only:
        probability = infer_mil_score(
            mil_model, features_path=features_path,
            background_path=store.path("tile_background", size_mm=size_mm, image_id=image_id),
            device=device,
        )
        store.write_part(
            "whole_image_risk_score", image_id,
            [{"image_id": image_id, "risk_score": probability, "fold": fold}], fold=fold,
        )
        return "completed"
    out = infer_mil(
        mil_model,
        features_path=features_path,
        background_path=store.path("tile_background", size_mm=size_mm, image_id=image_id),
        device=device,
    )
    store.write(
        "aligned_patch_embeddings",
        {"embeddings": out.patch_embeddings},
        fold=fold,
        image_id=image_id,
    )
    store.write("patch_attention", out.attention, fold=fold, image_id=image_id)
    store.write("aligned_slide_embedding", out.slide_embedding, fold=fold, image_id=image_id)
    store.write(
        "aligned_slide_embedding_grad", out.slide_embedding_grad, fold=fold, image_id=image_id
    )
    store.write_part(
        "whole_image_risk_score",
        image_id,
        [{"image_id": image_id, "risk_score": out.risk_score, "fold": fold}],
        fold=fold,
    )
    return "completed"


def _validate_feature_count(store: ArtifactStore, path: Path, *, size_mm: float, image_id: str) -> None:
    from helios.data.hdf5 import feature_shape
    from helios.stages.tile_features import _validate_tile_count

    _validate_tile_count(store, size_mm=size_mm, image_id=image_id, n_tiles=feature_shape(path)[0])


def _label(labels: pd.DataFrame, image_id: str, target: str) -> object | None:
    value = labels.loc[image_id, target]
    if value not in (0, 1):
        raise ValueError(f"MIL target {target!r} must be binary (0/1) for image {image_id!r}; got {value!r}.")
    return value
