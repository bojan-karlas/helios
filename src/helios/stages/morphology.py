"""Stages: patch morphology analysis (``helios fit/predict morphology``).

``fit morphology`` fits the per-fold clustering + cluster-presence risk model on
each fold's aligned patch embeddings + attention. ``predict morphology`` assigns
clusters, scores cluster presence, and produces the morphology risk score,
reducing the K fold models internally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from helios.components.morphology import infer_morphology, train_morphology
from helios.data.io import ArtifactStore
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
DEFAULT_TARGET: str = "disease_pfs_recurrence_5yfu"
DEFAULT_ATTENTION_CUTOFF: float = 0.8
DEFAULT_N_PCA: int = 50
DEFAULT_K_MICRO: int = 3000
DEFAULT_K_NEIGHBORS: int = 20
DEFAULT_RESOLUTION: float = 1.0
DEFAULT_SIGNIFICANCE_ALPHA: float = 0.05
DEFAULT_ESTIMATOR: str = "random_forest"
DEFAULT_SEED: int = 42


def fit_morphology(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
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
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Train one morphology model per fold over the pooled training cohorts.

    GRAIN CONTRACT: one fold per work unit. Requires per-fold
    ``aligned_patch_embeddings`` + ``patch_attention`` (from ``predict mil``).
    Writes ``model_morphology`` keyed by fold. Each cohort's ``where`` filter
    optionally restricts training to a metadata subpopulation.
    """
    resolved = resolve_datasets(datasets)
    stores = [cohort_store(ds.root, output_root) for ds in resolved]
    out = stores[0]
    folds = folds_across(stores)

    for fold in progress.task(folds, desc="fit morphology"):
        if not force and out.exists("model_morphology", fold=fold):
            progress.log(f"skip model_morphology fold={fold} (exists)")
            continue
        paths: dict[str, Path] = {}
        attention: list[pd.DataFrame] = []
        labels: list[pd.DataFrame] = []
        for ds, store in zip(resolved, stores, strict=True):
            meta = ds.dataset().image_metadata.set_index("image_id")
            ids = train_ids(store, fold, where=ds.where)
            for image_id in ids:
                paths[image_id] = store.path("aligned_patch_embeddings", fold=fold, image_id=image_id)
                attention.append(store.read("patch_attention", fold=fold, image_id=image_id).assign(image_id=image_id))
            labels.append(meta.loc[[i for i in ids if i in meta.index]].reset_index())
        bundle = train_morphology(
            paths,
            pd.concat(attention, ignore_index=True) if attention else pd.DataFrame(),
            pd.concat(labels, ignore_index=True),
            target=target,
            attention_cutoff=attention_cutoff,
            n_pca=n_pca,
            k_micro=k_micro,
            k_neighbors=k_neighbors,
            resolution=resolution,
            significance_alpha=significance_alpha,
            estimator=estimator,
            seed=seed,
        )
        out.write("model_morphology", bundle, fold=fold)
        progress.log(f"wrote model_morphology fold={fold}")


def predict_morphology(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    image_ids: list[str] | None = None,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Cluster, score presence, and produce morphology risk per cohort.

    GRAIN CONTRACT: one cohort per work unit; the inference component consumes
    all fold models and reduces OOF/ensemble. Writes ``tile_clusters`` (per
    image), ``patch_morphology_presence`` and ``patch_morphology_risk_score``.
    """
    resolved = resolve_datasets(datasets)
    single_output_guard(resolved, output_root)

    for ds in resolved:
        store = cohort_store(ds.root, output_root)
        ids = select_image_ids(ds, store, image_ids)
        folds = model_folds(store, "model_morphology")
        if not folds:
            raise FileNotFoundError(f"No model_morphology bundle found under {ds.name}.")
        if not force and store.exists("patch_morphology_risk_score"):
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
            models, paths, attention, read_cv_splits(store)
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
