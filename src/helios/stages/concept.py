"""Stages: pathological concept analysis (``helios fit/predict concept``).

``fit concept`` fits per-fold concept probes (+ concept risk + residual
correction) on aligned slide embeddings, pooling the (often concept-annotated)
training cohorts. ``predict concept`` scores a cohort, reducing the K fold models
to one row per image internally.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from helios.components.concepts import infer_concepts, train_concepts
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
DEFAULT_N_BOOTSTRAP: int = 100


def fit_concept(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Train per-fold concept models over the pooled concept cohorts.

    GRAIN CONTRACT: one fold per work unit; each fold's train slides are pooled
    across ``--dataset``. Requires per-fold ``aligned_slide_embedding`` and
    ``whole_image_risk_score`` (produced by ``predict mil`` on these cohorts).
    Writes ``model_concepts`` keyed by fold. Each cohort's ``where`` filter
    optionally restricts training to a metadata subpopulation.
    """
    resolved = resolve_datasets(datasets)
    stores = [cohort_store(ds.root, output_root) for ds in resolved]
    out = stores[0]
    folds = folds_across(stores)

    for fold in progress.task(folds, desc="fit concept"):
        if not force and out.exists("model_concepts", fold=fold):
            progress.log(f"skip model_concepts fold={fold} (exists)")
            continue
        embeddings: list[NDArray[np.float32]] = []
        labels: list[pd.DataFrame] = []
        risk: list[pd.DataFrame] = []
        for ds, store in zip(resolved, stores, strict=True):
            meta = ds.dataset().image_metadata.set_index("image_id")
            ids = train_ids(store, fold, where=ds.where)
            embeddings.append(_stack_embeddings(store, fold, ids))
            labels.append(meta.loc[[i for i in ids if i in meta.index]].reset_index())
            risk.append(store.read("whole_image_risk_score", fold=fold))
        bundle = train_concepts(
            np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), np.float32),
            pd.concat(labels, ignore_index=True),
            pd.concat(risk, ignore_index=True),
            n_bootstrap=n_bootstrap,
        )
        out.write("model_concepts", bundle, fold=fold)
        progress.log(f"wrote model_concepts fold={fold}")


def predict_concept(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    image_ids: list[str] | None = None,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Predict concepts + concept risk for each cohort (K folds reduced internally).

    GRAIN CONTRACT: one cohort per work unit; the concept inference component
    consumes all fold models and reduces OOF/ensemble to one row per image.
    """
    resolved = resolve_datasets(datasets)
    single_output_guard(resolved, output_root)

    for ds in resolved:
        store = cohort_store(ds.root, output_root)
        ids = select_image_ids(ds, store, image_ids)
        folds = model_folds(store, "model_concepts")
        if not folds:
            raise FileNotFoundError(f"No model_concepts bundle found under {ds.name}.")
        if not force and store.exists("path_concept_risk_score"):
            progress.log(f"skip concept predictions {ds.name} (exists)")
            continue

        models = {fold: store.read("model_concepts", fold=fold) for fold in folds}
        embeddings = {fold: _stack_embeddings(store, fold, ids) for fold in folds}
        risk = {fold: store.read("whole_image_risk_score", fold=fold) for fold in folds}
        predictions, risk_score = infer_concepts(models, embeddings, risk, read_cv_splits(store))
        store.write("path_concept_predictions", tag_dataset(predictions, ds.name))
        store.write("path_concept_risk_score", tag_dataset(risk_score, ds.name))
        progress.log(f"wrote concept predictions {ds.name}")


def _stack_embeddings(store: ArtifactStore, fold: int | None, ids: list[str]) -> NDArray[np.float32]:
    vectors = [
        np.asarray(store.read("aligned_slide_embedding", fold=fold, image_id=i), dtype=np.float32)
        for i in ids
    ]
    return np.stack(vectors, axis=0) if vectors else np.empty((0, 0), dtype=np.float32)
