"""Shared stage orchestration helpers — stores, folds, cohort pooling.

Stages (unlike pure components) own all artifact-store I/O and the dataset /
partition / fold loops. These helpers capture the few patterns every stage
repeats so the orchestration shape stays identical across ``prep``/``fit``/
``predict``/``report``:

* :func:`cohort_store` — build an :class:`ArtifactStore` for one cohort, honouring
  an optional ``output_root`` redirect (source stays read-only).
* :func:`single_output_guard` — reject ``output_root`` with >1 dataset, where the
  redirect target would be ambiguous (the ``prep``/``predict`` per-cohort shape).
* :func:`folds_of` / :func:`folds_across` — discover the K cross-validation folds
  from the optional ``cv_splits`` artifact (``[None]`` == trained without CV).
* :func:`train_ids` — the image ids in a fold's *train* split (the rows a
  per-fold ``fit`` component learns from).

The fold model: ``cv_splits`` is an OPTIONAL tidy CSV (``image_id, fold, split``).
Absent ⇒ a single fold-free model (fold ``None``); present ⇒ one model per fold.
The ``fold`` key is dropped from paths when ``None`` (see
:data:`helios.data.resolver.OPTIONAL_KEYS`), so the same artifact id addresses
both the per-fold and the reduced instances.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from helios.data.cohort import DatasetArgs, ResolvedDataset, resolve_datasets
from helios.data.io import ArtifactStore
from helios.data.resolver import Resolver, Roots

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
    "single_output_guard",
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


def cohort_store(dataset: str | Path, output_root: str | Path | None = None) -> ArtifactStore:
    """An :class:`ArtifactStore` reading ``dataset`` and writing derived outputs.

    With ``output_root`` set, the source cohort stays read-only and every derived
    artifact (work/features/output/models) lands under ``output_root`` instead.
    """
    return ArtifactStore(Resolver(Roots.for_run(dataset, output_root)))


def single_output_guard(datasets: Sequence[object], output_root: str | Path | None) -> None:
    """Reject an ``output_root`` redirect when more than one dataset is given.

    For per-cohort stages (``prep``, ``predict``) a single ``output_root`` cannot
    unambiguously hold several cohorts' outputs.
    """
    if output_root is not None and len(datasets) > 1:
        raise ValueError("--output-root is only valid with a single --dataset (ambiguous target).")


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
    fold-free bundle ⇒ ``[None]`` (deploy / no-CV); otherwise the CV folds whose
    per-fold bundle exists. Returns ``[]`` when no model has been fit yet.
    """
    if store.exists(model_id, fold=None):
        return [None]
    return [f for f in folds_of(store) if f is not None and store.exists(model_id, fold=f)]
