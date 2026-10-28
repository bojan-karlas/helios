"""Stages: patch morphology analysis (``helios fit/predict morphology``).

``fit morphology`` fits the per-fold clustering + cluster-presence risk model on
each fold's aligned patch embeddings + attention. ``predict morphology`` assigns
clusters, scores cluster presence, and produces the morphology risk score,
reducing the K fold models internally.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd

from helios.components.morphology import MorphologyEstimator, infer_morphology, train_morphology
from helios.data.cohort import ResolvedDataset
from helios.data.io import ArtifactStore
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    DatasetArgs,
    ParallelConfig,
    StorageRoots,
    WorkContext,
    build_bundle_metadata,
    cohort_store,
    folds_across,
    model_folds,
    read_splits,
    resolve_datasets,
    run_work_units,
    select_image_ids,
    tag_dataset,
    train_ids,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_TARGET: str = "disease_pfs_recurrence_5yfu"
DEFAULT_ATTENTION_CUTOFF: float = 0.8
DEFAULT_N_PCA: int = 50
DEFAULT_K_MICRO: int = 3000
DEFAULT_K_NEIGHBORS: int = 20
DEFAULT_RESOLUTION: float = 1.0
DEFAULT_SIGNIFICANCE_ALPHA: float = 0.05
DEFAULT_ESTIMATOR: str = "random_forest"
DEFAULT_SEED: int = 42


class _TrainingParams(TypedDict):
    target: str
    attention_cutoff: float
    n_pca: int
    k_micro: int
    k_neighbors: int
    resolution: float
    significance_alpha: float
    estimator: MorphologyEstimator
    seed: int


def fit_morphology(
    *,
    datasets: DatasetArgs,
    roots: StorageRoots | None = None,
    parallel: ParallelConfig | None = None,
    target: str = DEFAULT_TARGET,
    attention_cutoff: float = DEFAULT_ATTENTION_CUTOFF,
    n_pca: int = DEFAULT_N_PCA,
    k_micro: int = DEFAULT_K_MICRO,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    resolution: float = DEFAULT_RESOLUTION,
    significance_alpha: float = DEFAULT_SIGNIFICANCE_ALPHA,
    estimator: Literal["random_forest", "logreg_l2", "logreg_l1", "elasticnet", "gradient_boosting"] = (
        DEFAULT_ESTIMATOR  # type: ignore[assignment]
    ),
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Train one morphology model per fold over the pooled training cohorts.

    GRAIN CONTRACT: one fold per work unit. Requires per-fold
    ``aligned_patch_embeddings`` + ``patch_attention`` (from ``predict mil``).
    Writes ``model_morphology`` keyed by fold. Each cohort's ``where`` filter
    optionally restricts training to a metadata subpopulation.
    """
    resolved = resolve_datasets(datasets, roots=roots, require_models=True)
    folds = folds_across(resolved, roots)
    cfg = parallel or ParallelConfig()
    run_work_units(
        folds,
        partial(
            _fit_morphology_fold,
            resolved=resolved,
            roots=roots,
            target=target,
            attention_cutoff=attention_cutoff,
            n_pca=n_pca,
            k_micro=k_micro,
            k_neighbors=k_neighbors,
            resolution=resolution,
            significance_alpha=significance_alpha,
            estimator=estimator,
            seed=seed,
            overwrite=overwrite,
        ),
        stage="fit morphology",
        unit_id=lambda fold: f"fold={fold}",
        progress=progress,
        parallel=cfg,
        num_cpus=1.0,
        continue_on_error=cfg.enabled,
    )


def _fit_morphology_fold(
    fold: int | None,
    ctx: WorkContext,
    *,
    resolved: list[ResolvedDataset],
    roots: StorageRoots | None,
    target: str,
    attention_cutoff: float,
    n_pca: int,
    k_micro: int,
    k_neighbors: int,
    resolution: float,
    significance_alpha: float,
    estimator: Literal["random_forest", "logreg_l2", "logreg_l1", "elasticnet", "gradient_boosting"],
    seed: int,
    overwrite: bool,
) -> str:
    """Load, fit, and persist the morphology bundle for one fold."""
    stores = [cohort_store(ds, roots) for ds in resolved]
    out = stores[0]
    if not overwrite and out.exists("model_morphology", fold=fold):
        ctx.progress.log(f"skip model_morphology fold={fold} (exists)")
        return "skipped"

    paths: dict[str, Path] = {}
    attention: list[pd.DataFrame] = []
    labels: list[pd.DataFrame] = []
    for ds, store in zip(resolved, stores, strict=True):
        meta = ds.dataset().image_metadata.set_index("image_id")
        ids = train_ids(ds, fold, where=ds.where, roots=roots)
        missing_meta = [i for i in ids if i not in meta.index]
        if missing_meta:
            raise ValueError(f"Missing image_metadata rows for training slides in {ds.name}: {missing_meta[:5]}")
        for image_id in ids:
            emb_path = store.path("aligned_patch_embeddings", fold=fold, image_id=image_id)
            if not emb_path.exists():
                raise FileNotFoundError(
                    f"Missing aligned_patch_embeddings for training slide {image_id!r} (fold={fold}) in {ds.name}"
                )
            if not store.exists("patch_attention", fold=fold, image_id=image_id):
                raise FileNotFoundError(
                    f"Missing patch_attention for training slide {image_id!r} (fold={fold}) in {ds.name}"
                )
            paths[image_id] = emb_path
            attention.append(store.read("patch_attention", fold=fold, image_id=image_id).assign(image_id=image_id))
        labels.append(meta.loc[ids].reset_index())

    params: _TrainingParams = {
        "target": target,
        "attention_cutoff": attention_cutoff,
        "n_pca": n_pca,
        "k_micro": k_micro,
        "k_neighbors": k_neighbors,
        "resolution": resolution,
        "significance_alpha": significance_alpha,
        "estimator": estimator,
        "seed": seed,
    }
    bundle = train_morphology(
        paths,
        pd.concat(attention, ignore_index=True) if attention else pd.DataFrame(),
        pd.concat(labels, ignore_index=True),
        **params,
    )
    meta = build_bundle_metadata(
        "morphology",
        dict(params),
        resolved,
        roots,
        fold=fold,
    )
    out.write("model_morphology", bundle, metadata=meta, fold=fold)
    ctx.progress.log(f"wrote model_morphology fold={fold}")
    return "completed"


def predict_morphology(
    *,
    datasets: DatasetArgs,
    roots: StorageRoots | None = None,
    parallel: ParallelConfig | None = None,
    image_ids: list[str] | None = None,
    overwrite: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Cluster, score presence, and produce morphology risk per cohort.

    GRAIN CONTRACT: one cohort per work unit; the inference component consumes
    all fold models and reduces OOF/ensemble. Writes ``tile_clusters`` (per
    image), ``patch_morphology_presence`` and ``patch_morphology_risk_score``.
    """
    resolved = resolve_datasets(datasets, roots=roots)

    for ds in resolved:
        store = cohort_store(ds, roots)
        ids = select_image_ids(ds, store, image_ids)
        folds = model_folds(store, "model_morphology")
        if not folds:
            raise FileNotFoundError(f"No model_morphology bundle found under {ds.name}.")
        if not overwrite and store.exists("patch_morphology_risk_score"):
            progress.log(f"skip morphology predictions {ds.name} (exists)")
            continue

        models = {fold: store.read("model_morphology", fold=fold) for fold in folds}
        paths = {
            fold: {i: store.path("aligned_patch_embeddings", fold=fold, image_id=i) for i in ids}
            for fold in folds
        }
        attention = {
            fold: _stack_attention(store, fold, ids) for fold in folds
        }
        clusters, presence, risk_score = infer_morphology(
            models, paths, attention, read_splits(ds, roots, progress=progress)
        )
        for image_id, group in clusters.groupby("image_id"):
            store.write("tile_clusters", group, image_id=str(image_id))
        store.write("patch_morphology_presence", tag_dataset(presence, ds.name))
        store.write("patch_morphology_risk_score", tag_dataset(risk_score, ds.name))
        progress.log(f"wrote morphology predictions {ds.name}")


def _stack_attention(store: ArtifactStore, fold: int | None, ids: list[str]) -> pd.DataFrame:
    frames = [
        store.read("patch_attention", fold=fold, image_id=i).assign(image_id=i) for i in ids
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
