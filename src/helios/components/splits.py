"""Component: cross-validation split generation.

Pure assignment of each image to one of ``n_folds`` folds, optionally grouped by
patient (to prevent leakage) and stratified by a label column. The owning stage
(:mod:`helios.stages.splits`) reads ``image_metadata`` and writes the resulting
``cv_splits`` table; this function only computes the assignment.
"""
from __future__ import annotations

import pandas as pd

IMAGE_ID = "image_id"


def generate_cv_splits(
    image_metadata: pd.DataFrame,
    *,
    n_folds: int = 5,
    val_fraction: float = 0.0,
    stratify_by: list[str] | None = None,
    group_by: str = "patient_id",
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
        Number of folds ``K``.
    val_fraction:
        Fraction of each fold's train rows to carve off as a ``val`` split
        (``0`` ⇒ only ``train``/``test``).
    stratify_by:
        Label columns to balance across folds (``None`` ⇒ no stratification).
    group_by:
        Column whose groups (e.g. patients) are kept within a single fold to
        prevent leakage.
    seed:
        RNG seed for reproducible assignment.

    Returns
    -------
    pandas.DataFrame
        Tidy long table with columns ``image_id, fold, split`` (split ∈
        {train, val, test}); one row per (image, fold).
    """
    raise NotImplementedError(
        "Implement grouped/stratified K-fold assignment producing image_id, fold, split rows."
    )
