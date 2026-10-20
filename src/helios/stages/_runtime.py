"""Shared stage orchestration helpers --- stores, folds, cohort pooling.

Stages (unlike pure components) own all artifact-store I/O and the dataset /
partition / fold loops. These helpers capture the few patterns every stage
repeats so the orchestration shape stays identical across ``prep``/``fit``/
``predict``/``report``:

* :func:`cohort_store` -- route each constituent's artifacts by storage role.
* :func:`resolve_datasets` -- validate constituent and run root settings.
* :func:`folds_of` / :func:`folds_across` --- discover the K cross-validation folds
  from the optional ``cv_splits`` artifact (``[None]`` == trained without CV).
* :func:`train_ids` --- the image ids in a fold's *train* split (the rows a
  per-fold ``fit`` component learns from).

The fold model: ``cv_splits`` is an OPTIONAL tidy CSV (``image_id, fold, split``).
Absent means a single fold-free model; otherwise there is one model per fold.
The ``fold`` key is dropped from paths when ``None`` (see
:data:`helios.data.resolver.OPTIONAL_KEYS`), so the same artifact id addresses
both the per-fold and the reduced instances.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from helios.data.cohort import DatasetArgs, ResolvedDataset
from helios.data.cohort import resolve_datasets as resolve_dataset_args
from helios.data.io import ArtifactStore
from helios.data.resolver import Resolver, Roots, StorageRoots

__all__ = [
    "DatasetArgs",
    "ResolvedDataset",
    "resolve_datasets",
    "dataset_name",
    "tag_dataset",
    "drop_provenance",
    "filtered_ids",
    "select_image_ids",
    "cohort_store",
    "StorageRoots",
    "read_cv_splits",
    "folds_of",
    "folds_across",
    "train_ids",
    "model_folds",
]

SPLIT_COLUMN = "split"
FOLD_COLUMN = "fold"
TRAIN_SPLIT = "train"
DATASET_COLUMN = "dataset"
IMAGE_ID_COLUMN = "image_id"


def dataset_name(dataset: str | Path) -> str:
    """The cohort's short name (its root directory name), used for provenance."""
    return Path(dataset).name


def tag_dataset(frame: pd.DataFrame, dataset: str | Path) -> pd.DataFrame:
    """Stamp a ``dataset`` provenance column recording the source cohort.

    Result tables are commonly concatenated across cohorts downstream; carrying
    the source cohort name on every row keeps that join lossless. Returns a copy
    with ``dataset`` as the leading column (existing values are overwritten).
    """
    tagged = frame.copy()
    tagged.insert(0, DATASET_COLUMN, dataset_name(dataset))
    return tagged


def filtered_ids(store: ArtifactStore, where: str | None) -> set[str] | None:
    """Image ids selected by a pandas-``query`` ``where`` over ``image_metadata``.

    ``None`` (no filter) returns ``None`` (select everything). Otherwise evaluate
    the expression against the cohort metadata and return the matching ids.
    """
    if not where:
        return None
    meta = store.read("image_metadata")
    return {str(i) for i in meta.query(where)[IMAGE_ID_COLUMN]}


def select_image_ids(
    ds: ResolvedDataset,
    store: ArtifactStore,
    image_ids: list[str] | None,
) -> list[str]:
    """The working image ids for a per-cohort stage on dataset ``ds``.

    Starts from an explicit ``image_ids`` override (else the whole cohort), then
    narrows to the dataset's row filter (``ds.where``) so a cohort definition's
    population is honoured uniformly across ``prep`` / ``predict`` / ``report``.
    """
    base = image_ids if image_ids is not None else ds.dataset().image_ids
    keep = filtered_ids(store, ds.where)
    return base if keep is None else [i for i in base if i in keep]


def drop_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip the ``dataset`` provenance column before using a table as features.

    Result CSVs carry a ``dataset`` provenance column (:func:`tag_dataset`); when
    such a table is fed back in as *model input* (e.g. the fusion feature join),
    that column is metadata, not a feature, and must be dropped first.
    """
    return frame.drop(columns=[DATASET_COLUMN], errors="ignore")


def resolve_datasets(
    datasets: DatasetArgs,
    *,
    where: str | None = None,
    roots: StorageRoots | None = None,
    require_models: bool = False,
) -> list[ResolvedDataset]:
    """Resolve constituents and reject ambiguous run destinations before I/O."""
    resolved = resolve_dataset_args(datasets, where=where)
    overrides = roots or StorageRoots()
    if len(resolved) > 1:
        if overrides.preprocessing_root is not None or overrides.features_root is not None:
            raise ValueError(
                "For multiple datasets, set preprocessing_root and features_root per dataset in the cohort YAML."
            )
        if require_models and overrides.models_root is None:
            raise ValueError("Pooled training requires an explicit models_root (--models-root).")
    if overrides.results_root is not None:
        names = [ds.name for ds in resolved]
        if any(name in {"", ".", ".."} for name in names) or len(names) != len(set(names)):
            raise ValueError("A shared results_root requires distinct, nonempty dataset directory names.")
    return resolved


def cohort_store(dataset: str | Path | ResolvedDataset, roots: StorageRoots | None = None) -> ArtifactStore:
    """Use constituent-local data roots and shared run model/results overrides."""
    ds = dataset if isinstance(dataset, ResolvedDataset) else ResolvedDataset(Path(dataset))
    overrides = roots or StorageRoots()
    return ArtifactStore(Resolver(Roots(
        source_root=ds.root,
        preprocessing_root=overrides.preprocessing_root or ds.preprocessing_root,
        features_root=overrides.features_root or ds.features_root,
        models_root=overrides.models_root,
        results_root=overrides.results_root / ds.name if overrides.results_root is not None else None,
    )))


def read_cv_splits(store: ArtifactStore) -> pd.DataFrame | None:
    """The cohort's ``cv_splits`` table, or ``None`` when it was never produced."""
    if store.exists("cv_splits"):
        return store.read("cv_splits")
    return None


def folds_of(store: ArtifactStore) -> list[int | None]:
    """Folds defined by a cohort's ``cv_splits`` (``[None]`` if trained without CV)."""
    splits = read_cv_splits(store)
    if splits is None or FOLD_COLUMN not in splits:
        return [None]
    folds: list[int | None] = [int(f) for f in sorted(splits[FOLD_COLUMN].dropna().unique())]
    return folds or [None]


def folds_across(stores: list[ArtifactStore]) -> list[int | None]:
    """Union of folds across pooled training cohorts.

    Used by ``fit`` stages that pool several ``--dataset`` cohorts: if any cohort
    defines folds, train per fold; if none do, train a single fold-free model.
    """
    seen: set[int] = set()
    for store in stores:
        for fold in folds_of(store):
            if fold is not None:
                seen.add(fold)
    result: list[int | None] = [f for f in sorted(seen)]
    return result or [None]


def train_ids(store: ArtifactStore, fold: int | None, *, where: str | None = None) -> list[str]:
    """Image ids in ``fold``'s train split (all images when there is no CV).

    With no ``cv_splits`` (or ``fold is None``) every image is a training row.
    With splits, return the ids whose ``split == train`` for that fold. An
    optional ``where`` pandas-``query`` further restricts the rows to a metadata
    subpopulation (e.g. ``"path_stage in ['I', 'II']"``), composed *after* the
    fold's train split so filtering never leaks test images into training.
    """
    splits = read_cv_splits(store)
    if splits is None or fold is None or FOLD_COLUMN not in splits:
        ids = [str(i) for i in store.read("image_metadata")[IMAGE_ID_COLUMN]]
    else:
        rows = splits[splits[FOLD_COLUMN] == fold]
        if SPLIT_COLUMN in rows:
            rows = rows[rows[SPLIT_COLUMN] == TRAIN_SPLIT]
        ids = [str(i) for i in rows[IMAGE_ID_COLUMN]]
    keep = filtered_ids(store, where)
    return ids if keep is None else [i for i in ids if i in keep]


def model_folds(store: ArtifactStore, model_id: str) -> list[int | None]:
    """Folds an inference stage should score with, enumerated from saved models.

    Inference enumerates folds from the *model* (not ``cv_splits``): a single
    fold-free bundle ⇒ ``[None]`` (deploy / no-CV); otherwise every fold whose
    per-fold bundle actually exists on disk. Folds are discovered by scanning
    the model's own directory (``fold=N`` entries) rather than the scoring
    cohort's ``cv_splits`` --- a deploy cohort scored with per-fold models
    trained elsewhere has no ``cv_splits`` of its own, so that would find
    nothing even though the fold bundles are right there. Returns ``[]`` when
    no model has been fit yet.
    """
    if store.exists(model_id, fold=None):
        return [None]
    root = store.path(model_id, fold=0).parent
    if not root.is_dir():
        return []
    folds: list[int] = []
    for entry in sorted(root.iterdir()):
        name = entry.name
        if name.startswith(f"{FOLD_COLUMN}="):
            try:
                fold = int(name[len(FOLD_COLUMN) + 1 :])
            except ValueError:
                continue
            if store.exists(model_id, fold=fold):
                folds.append(fold)
    return sorted(folds)
