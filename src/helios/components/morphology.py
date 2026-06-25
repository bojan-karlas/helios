"""Components: patch morphology analysis.

* :func:`train_morphology` — attention-percentile filter → PCA → Leiden
  clustering → cluster exclusion → cluster-presence risk model.
* :func:`infer_morphology` — assign tiles to clusters, score cluster presence,
  and produce the patch-morphology risk score.

Patch embeddings are per-tile (large), so they are passed as resolved paths and
streamed lazily; attention tables are small and passed loaded. The owning stages
live in :mod:`helios.stages.morphology`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from helios.models.base import Model


def train_morphology(
    patch_embedding_paths: dict[str, Path],
    attention: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    attention_percentile: float = 90.0,
    n_pca: int = 50,
) -> Model:
    """Fit the morphology clustering + cluster-presence risk model for one fold.

    GRAIN: one fold (its per-fold patch embeddings/attention) per call.

    Parameters
    ----------
    patch_embedding_paths:
        ``image_id -> aligned_patch_embeddings path`` (streamed lazily).
    attention:
        Per-tile attention scores (used for the percentile filter).
    labels:
        Per-slide outcome labels for the cluster-presence risk model.
    attention_percentile:
        Keep only tiles above this attention percentile before clustering.
    n_pca:
        PCA component count prior to Leiden clustering.

    Returns the fitted bundle (PCA + Leiden + cluster-exclusion + risk model).
    """
    raise NotImplementedError(
        "Fit attention-filter → PCA → Leiden → cluster-exclusion → cluster-presence risk model."
    )


def infer_morphology(
    models: dict[int | None, Model],
    patch_embedding_paths: dict[int | None, dict[str, Path]],
    attention: dict[int | None, pd.DataFrame],
    cv_splits: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign clusters, score cluster presence, and produce morphology risk.

    GRAIN: the scored cohort per call. Inputs are keyed by fold (each fold's
    model + that fold's per-image patch-embedding paths + attention); reduction
    is out-of-fold on a CV cohort / ensemble on deploy (``cv_splits is None``).

    Returns
    -------
    tuple
        ``(tile_clusters, patch_morphology_presence, patch_morphology_risk_score)``;
        clusters/presence are fold-free, the risk score carries a ``fold``
        provenance column.
    """
    raise NotImplementedError(
        "Assign clusters, compute presence + morphology risk; reduce K folds OOF/ensemble."
    )
