"""Components: pathological concept analysis.

* :func:`train_concepts` — per-concept linear probes on aligned slide embeddings
  (LogReg for categorical, Ridge for numerical), bootstrapped for uncertainty,
  plus a concept-based risk model and a residual error-correction regressor.
* :func:`infer_concepts` — predict concepts (+ optional pathologist override) and
  the concept-based risk score (+ residual correction).

Aligned slide embeddings are one vector per slide, so these operate on a loaded
``(n_slides, D)`` matrix. The owning stages live in
:mod:`helios.stages.concept`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from helios.models.base import Model


def train_concepts(
    slide_embeddings: NDArray[np.float32],
    concept_labels: pd.DataFrame,
    whole_image_risk: pd.DataFrame,
    *,
    n_bootstrap: int = 100,
) -> Model:
    """Fit per-concept probes + concept risk + residual correction for one fold.

    GRAIN: one fold (its concept-cohort train slides) per call.

    Parameters
    ----------
    slide_embeddings:
        ``(n_slides, D)`` aligned slide embeddings (row ``i`` ↔ row ``i`` of the
        label/risk frames).
    concept_labels:
        Concept ground-truth from ``path_*`` columns (one row per slide).
    whole_image_risk:
        Per-fold whole-image risk used to fit the residual error-correction model.
    n_bootstrap:
        Bootstrap replicas per concept (``1`` disables); yields a prediction
        distribution / uncertainty per concept.

    Returns the fitted bundle (``n_bootstrap × n_concepts`` probes + risk +
    residual models).
    """
    raise NotImplementedError(
        "Fit bootstrapped per-concept LogReg/Ridge probes + concept risk + residual correction."
    )


def infer_concepts(
    models: dict[int | None, Model],
    slide_embeddings: dict[int | None, NDArray[np.float32]],
    whole_image_risk: dict[int | None, pd.DataFrame],
    cv_splits: pd.DataFrame | None,
    *,
    overrides: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict concepts and concept-based risk, reducing the K folds internally.

    GRAIN: the scored cohort per call. Inputs are keyed by fold (each fold's
    model + that fold's aligned slide embeddings + per-fold whole-image risk);
    reduction is out-of-fold on a CV cohort / ensemble on deploy
    (``cv_splits is None``), so a single reduced row per image is returned.

    Returns
    -------
    tuple
        ``(path_concept_predictions, path_concept_risk_score)``; each one row per
        image (the risk score carries a ``fold`` provenance column).
        ``overrides`` optionally replaces predicted concepts with
        pathologist-reported values.
    """
    raise NotImplementedError(
        "Predict concepts (+optional override) and concept-based risk; reduce K folds OOF/ensemble."
    )
