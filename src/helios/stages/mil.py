"""Stages: whole-slide A-MIL training & inference (``helios fit/predict mil``).

``fit mil`` pools one or more training cohorts and fits one A-MIL bundle per CV
fold. ``predict mil`` scores every image with each fold model (per-fold
embeddings / attention / whole-image risk) and then reduces the per-fold
whole-image scores to a single shipped value per image.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from helios.components.features import FEATURE_MODELS
from helios.components.mil import SlideFeatureRef, infer_mil, reduce_whole_image_risk, train_mil
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    DatasetArgs,
    cohort_store,
    folds_across,
    model_folds,
    read_cv_splits,
    resolve_datasets,
    select_image_ids,
    single_output_guard,
    tag_dataset,
    train_ids,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_SIZE_MM: float = 0.25
DEFAULT_MODEL: str = "virchow2"
DEFAULT_TARGET: str = "disease_pfs_recurrence_5yfu"
DEFAULT_AUG_SWAP_PROB: float = 0.5
DEFAULT_DEVICE: str = "cuda"


def fit_mil(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    size_mm: float = DEFAULT_SIZE_MM,
    model: str = DEFAULT_MODEL,
    target: str = DEFAULT_TARGET,
    aug_swap_prob: float = DEFAULT_AUG_SWAP_PROB,
    device: str = DEFAULT_DEVICE,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Train one A-MIL model per fold over the pooled training cohorts.

    GRAIN CONTRACT: one fold per work unit. Each fold's train slides are pooled
    across every ``--dataset``; tile features are referenced by path (streamed
    lazily by the component). Writes ``model_mil`` keyed by fold. Each cohort's
    ``where`` filter optionally restricts training to a metadata subpopulation.
    """
    if model not in FEATURE_MODELS:
        raise ValueError(f"Unknown feature model {model!r}. Available: {list(FEATURE_MODELS)}.")

    resolved = resolve_datasets(datasets)
    stores = [cohort_store(ds.root, output_root) for ds in resolved]
    out = stores[0]
    folds = folds_across(stores)

    for fold in progress.task(folds, desc="fit mil"):
        if not force and out.exists("model_mil", fold=fold):
            progress.log(f"skip model_mil fold={fold} (exists)")
            continue
        slides: list[SlideFeatureRef] = []
        for ds, store in zip(resolved, stores, strict=True):
            labels = ds.dataset().image_metadata.set_index("image_id")
            for image_id in train_ids(store, fold, where=ds.where):
                slides.append(
                    SlideFeatureRef(
                        image_id=image_id,
                        features_path=store.path(
                            "tile_features", size_mm=size_mm, model=model, image_id=image_id
                        ),
                        background_path=store.path("tile_background", size_mm=size_mm, image_id=image_id),
                        aug_features_path=None,
                        label=_label(labels, image_id, target),
                    )
                )
        bundle = train_mil(slides, aug_swap_prob=aug_swap_prob, device=device)
        out.write("model_mil", bundle, fold=fold)
        progress.log(f"wrote model_mil fold={fold} (n_train={len(slides)})")


def predict_mil(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    size_mm: float = DEFAULT_SIZE_MM,
    model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    image_ids: list[str] | None = None,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Score every image with each fold's A-MIL model, then reduce whole-image risk.

    GRAIN CONTRACT: per (dataset, fold, image) inference, then a per-dataset
    reduce of the K whole-image scores to ``whole_image_risk_score_ensemble``.
    Folds are enumerated from the saved ``model_mil`` bundles.
    """
    resolved = resolve_datasets(datasets)
    single_output_guard(resolved, output_root)

    for ds in resolved:
        store = cohort_store(ds.root, output_root)
        ids = select_image_ids(ds, store, image_ids)
        folds = model_folds(store, "model_mil")
        if not folds:
            raise FileNotFoundError(f"No model_mil bundle found under {ds.name}.")

        per_fold_scores: list[pd.DataFrame] = []
        for fold in folds:
            mil_model = store.read("model_mil", fold=fold)
            rows: list[dict[str, object]] = []
            for image_id in progress.task(ids, desc=f"mil infer fold={fold} {ds.name}"):
                if not force and store.exists("aligned_slide_embedding", fold=fold, image_id=image_id):
                    continue
                out = infer_mil(
                    mil_model,
                    features_path=store.path("tile_features", size_mm=size_mm, model=model, image_id=image_id),
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
                rows.append({"image_id": image_id, "risk_score": out.risk_score, "fold": fold})
            if rows:
                scores = pd.DataFrame(rows)
                store.write("whole_image_risk_score", scores, fold=fold)
                per_fold_scores.append(scores)

        ensemble = reduce_whole_image_risk(
            pd.concat(per_fold_scores, ignore_index=True) if per_fold_scores else pd.DataFrame(),
            read_cv_splits(store),
        )
        store.write("whole_image_risk_score_ensemble", tag_dataset(ensemble, ds.name))
        progress.log(f"wrote whole_image_risk_score_ensemble {ds.name}")


def _label(labels: pd.DataFrame, image_id: str, target: str) -> object | None:
    if image_id not in labels.index or target not in labels.columns:
        return None
    return labels.loc[image_id, target]
