"""Stages: simple risk sub-models + fusion (``helios fit/predict risk``).

A single ``risk`` command trains/scores any subset of the simple risk models via
``--risk`` (``cellular``, ``staging``, ``clinical``, ``helios``); ``helios`` is the
final weighted fusion and is ordered last so its sub-score inputs already exist.
Each kind differs only in the feature table it builds; the per-fold fit/score +
OOF/ensemble reduce shape is shared.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from helios.components.risk import (
    RISK_KINDS,
    compute_survival_curve,
    infer_risk_model,
    train_risk_model,
)
from helios.data.io import ArtifactStore
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    DatasetArgs,
    cohort_store,
    drop_provenance,
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
DEFAULT_RISK: list[str] = list(RISK_KINDS)
DEFAULT_TARGET: str = "disease_pfs_recurrence_5yfu"
DEFAULT_STAGING_FEATURE: str = "path_stage_substage"
DEFAULT_CLINICAL_FEATURES: list[str] = [
    "patient_age_at_diagnosis",
    "patient_sex",
    "sample_tissue_site_category",
]
DEFAULT_ESTIMATOR: str = "sklearn_linear"

#: risk kind -> (model artifact id, score artifact id).
_ARTIFACTS: dict[str, tuple[str, str]] = {
    "cellular": ("model_cellular", "cellular_risk_score"),
    "staging": ("model_staging", "staging_risk_score"),
    "clinical": ("model_clinical", "clinical_risk_score"),
    "helios": ("model_fusion", "helios_risk_score"),
}

#: sub-scores joined to form the fusion feature table.
_FUSION_INPUTS = (
    "cellular_risk_score",
    "path_concept_risk_score",
    "patch_morphology_risk_score",
    "staging_risk_score",
    "clinical_risk_score",
)


def fit_risk(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    risk: list[str] = DEFAULT_RISK,
    target: str = DEFAULT_TARGET,
    staging_feature: str = DEFAULT_STAGING_FEATURE,
    clinical_features: list[str] = DEFAULT_CLINICAL_FEATURES,
    estimator: str = DEFAULT_ESTIMATOR,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Train the selected risk sub-models, one model per fold each.

    GRAIN CONTRACT: per (risk kind, fold) work unit; train rows are pooled across
    ``--dataset``. ``helios`` (fusion) trains last and reads the other sub-scores.
    Each cohort's ``where`` filter optionally restricts training to a metadata
    subpopulation.
    """
    _validate(risk)
    resolved = resolve_datasets(datasets)
    stores = [cohort_store(ds.root, output_root) for ds in resolved]
    out = stores[0]
    folds = folds_across(stores)

    for kind in _ordered(risk):
        model_id, _ = _ARTIFACTS[kind]
        for fold in progress.task(folds, desc=f"fit risk {kind}"):
            if not force and out.exists(model_id, fold=fold):
                progress.log(f"skip {model_id} fold={fold} (exists)")
                continue
            features: list[pd.DataFrame] = []
            targets: list[pd.Series] = []
            for ds, store in zip(resolved, stores, strict=True):
                meta = ds.dataset().image_metadata.set_index("image_id")
                ids = [i for i in train_ids(store, fold, where=ds.where) if i in meta.index]
                features.append(_features(kind, store, ids, staging_feature, clinical_features))
                targets.append(meta.loc[ids, target])
            bundle = train_risk_model(
                pd.concat(features, ignore_index=True),
                pd.concat(targets, ignore_index=True),
                estimator=estimator,
            )
            out.write(model_id, bundle, fold=fold)
            progress.log(f"wrote {model_id} fold={fold}")


def predict_risk(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    risk: list[str] = DEFAULT_RISK,
    staging_feature: str = DEFAULT_STAGING_FEATURE,
    clinical_features: list[str] = DEFAULT_CLINICAL_FEATURES,
    image_ids: list[str] | None = None,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Score the selected risk sub-models for each cohort (K folds reduced internally).

    GRAIN CONTRACT: per (risk kind, cohort) work unit. ``helios`` (fusion) runs
    last, joins the other sub-scores, and also writes the ``survival_curve``.
    """
    _validate(risk)
    resolved = resolve_datasets(datasets)
    single_output_guard(resolved, output_root)

    for ds in resolved:
        store = cohort_store(ds.root, output_root)
        ids = select_image_ids(ds, store, image_ids)

        for kind in _ordered(risk):
            model_id, score_id = _ARTIFACTS[kind]
            folds = model_folds(store, model_id)
            if not folds:
                raise FileNotFoundError(f"No {model_id} bundle found under {ds.name}.")
            if not force and store.exists(score_id):
                progress.log(f"skip {score_id} {ds.name} (exists)")
                continue
            models = {fold: store.read(model_id, fold=fold) for fold in folds}
            features = _features(kind, store, ids, staging_feature, clinical_features)
            scores = infer_risk_model(models, features, read_cv_splits(store))
            store.write(score_id, tag_dataset(scores, ds.name))
            progress.log(f"wrote {score_id} {ds.name}")
            if kind == "helios":
                store.write("survival_curve", tag_dataset(compute_survival_curve(scores), ds.name))
                progress.log(f"wrote survival_curve {ds.name}")


def _features(
    kind: str,
    store: ArtifactStore,
    ids: list[str],
    staging_feature: str,
    clinical_features: list[str],
) -> pd.DataFrame:
    """Build the per-image feature table for one risk kind, restricted to ``ids``."""
    if kind == "cellular":
        table = drop_provenance(store.read("cellular_features"))
    elif kind == "staging":
        table = store.read("image_metadata")[["image_id", staging_feature]]
    elif kind == "clinical":
        table = store.read("image_metadata")[["image_id", *clinical_features]]
    else:  # helios fusion: join the sub-scores on image_id
        table = None
        for sub in _FUSION_INPUTS:
            frame = drop_provenance(store.read(sub))
            table = frame if table is None else table.merge(frame, on="image_id", how="outer")
        assert table is not None
    indexed = table.set_index("image_id")
    present = [i for i in ids if i in indexed.index]
    return indexed.loc[present].reset_index()


def _ordered(risk: list[str]) -> list[str]:
    """Selected kinds in canonical order (``helios`` last)."""
    return [k for k in RISK_KINDS if k in set(risk)]


def _validate(risk: list[str]) -> None:
    unknown = [k for k in risk if k not in RISK_KINDS]
    if unknown:
        raise ValueError(f"Unknown risk kind(s) {unknown}. Available: {list(RISK_KINDS)}.")
