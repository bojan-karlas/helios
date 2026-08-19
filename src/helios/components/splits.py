"""Component: cross-validation split generation.

Pure assignment of each image to one of ``n_folds`` folds, optionally grouped by
patient (to prevent leakage) and stratified by one or more label columns. The
owning stage (:mod:`helios.stages.splits`) reads ``image_metadata`` and writes
the resulting ``cv_splits`` table; this function only computes the assignment.

Ported from ``fold_assignment.ipynb``: ``StratifiedKFold(n_splits=5,
shuffle=True, random_state=seed)`` on the recurrence outcome column, assigning
each image a single test fold, run once per cohort (the notebook loops over
cohorts and concatenates — this component takes one cohort's
``image_metadata`` per call, matching its GRAIN, so running it once per
``--dataset`` reproduces the notebook's per-cohort independence). Two
capabilities are added beyond the notebook's single call, both off by default
so ``stratify_by=["disease_pfs_recurred"], group_by=None`` reproduces it
exactly:

* ``stratify_by`` accepts more than one column — jointly stratifies on a
  composite key (``"_"``-joined) instead of one label. The notebook's own
  comment claimed to stratify by outcome *and* substage but its code only
  passed the outcome column to ``StratifiedKFold``; this makes joint
  stratification actually available.
* ``group_by`` (the notebook did not group at all) uses
  ``StratifiedGroupKFold`` to keep every row of a group (e.g. a patient's
  slides) in the same fold, preventing cross-fold leakage.

Both flat "one test fold per image" and any ``val_fraction`` carve-out are
expanded into HELIOS's long ``(image_id, fold, split)`` format (one row per
image per fold), not the notebook's flat one-row-per-image table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

IMAGE_ID = "image_id"
FOLD = "fold"
SPLIT = "split"
TRAIN = "train"
VAL = "val"
TEST = "test"


def generate_cv_splits(
    image_metadata: pd.DataFrame,
    *,
    n_folds: int = 5,
    val_fraction: float = 0.0,
    stratify_by: list[str] | None = None,
    group_by: str | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Assign every image to a fold (and split) for K-fold cross-validation.

    GRAIN: one whole cohort per call.

    Parameters
    ----------
    image_metadata:
        One row per ``image_id`` (must contain ``image_id`` and, if grouping,
        ``group_by``; if stratifying, every ``stratify_by`` column).
    n_folds:
        Number of folds ``K`` (clipped to ``len(image_metadata)`` if smaller).
    val_fraction:
        Fraction of each fold's train rows to carve off as a ``val`` split
        (``0`` ⇒ only ``train``/``test``). Carved out by whole group when
        ``group_by`` is set, so a group never spans train and val.
    stratify_by:
        Label columns to balance across folds (joint composite key when more
        than one; ``None``/empty ⇒ no stratification).
    group_by:
        Column whose groups (e.g. patients) are kept within a single fold to
        prevent leakage (``None`` ⇒ no grouping, matching the source
        notebook).
    seed:
        RNG seed for reproducible assignment.

    Returns
    -------
    pandas.DataFrame
        Tidy long table with columns ``image_id, fold, split`` (split ∈
        {train, val, test}); one row per (image, fold).
    """
    n = len(image_metadata)
    if n == 0:
        return pd.DataFrame(columns=[IMAGE_ID, FOLD, SPLIT])

    n_splits = min(n_folds, n)
    if n_splits < 2:
        raise ValueError(f"generate_cv_splits needs at least 2 images to split; got {n}.")

    image_ids = image_metadata[IMAGE_ID].astype(str).to_numpy()
    y = _composite_key(image_metadata, stratify_by) if stratify_by else np.zeros(n, dtype=int)

    if group_by:
        groups = image_metadata[group_by].astype(str).to_numpy()
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(n), y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(n), y)

    test_fold = np.full(n, -1, dtype=int)
    for fold, (_train_idx, test_idx) in enumerate(split_iter):
        test_fold[test_idx] = fold

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for fold in range(n_splits):
        is_test = test_fold == fold
        train_idx = np.where(~is_test)[0]
        val_idx = set(_carve_val(train_idx, val_fraction, group_by, image_metadata, rng).tolist())
        for i in range(n):
            if is_test[i]:
                split = TEST
            elif i in val_idx:
                split = VAL
            else:
                split = TRAIN
            rows.append({IMAGE_ID: image_ids[i], FOLD: fold, SPLIT: split})

    return pd.DataFrame(rows)


def _composite_key(image_metadata: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return image_metadata[columns].astype(str).agg("_".join, axis=1).to_numpy()


def _carve_val(
    train_idx: np.ndarray,
    val_fraction: float,
    group_by: str | None,
    image_metadata: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    if val_fraction <= 0 or len(train_idx) == 0:
        return np.empty(0, dtype=int)

    if group_by:
        groups = image_metadata[group_by].astype(str).to_numpy()[train_idx]
        unique_groups = np.unique(groups)
        n_val_groups = max(1, int(round(len(unique_groups) * val_fraction)))
        val_groups = rng.choice(unique_groups, size=n_val_groups, replace=False)
        return train_idx[np.isin(groups, val_groups)]

    n_val = max(1, int(round(len(train_idx) * val_fraction)))
    return rng.choice(train_idx, size=n_val, replace=False)
