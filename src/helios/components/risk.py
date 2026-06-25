"""Components: simple tabular risk models + fusion.

All HELIOS risk sub-models are *simple* ML models (logistic regression / random
forest / linear) over per-slide tabular features, and the final HELIOS model
fuses the sub-scores the same way. They therefore share one tiny contract:

* :func:`train_risk_model` — fit a simple estimator on a feature table + target.
* :func:`infer_risk_model` — score a feature table with a fitted model.
* :func:`reduce_risk` — collapse K per-fold scores to one row per image
  (out-of-fold on a CV cohort / ensemble on deploy).
* :func:`compute_survival_curve` — derive a per-image survival curve from risk.

The owning stages (:mod:`helios.stages.risk`) build the right feature table per
risk kind (cellular features / staging column / clinical columns / joined
sub-scores) and pick the per-image target.
"""
from __future__ import annotations

import pandas as pd

from helios.models.base import Model

#: Risk sub-models, ordered; ``helios`` is the final fusion model and trains last.
RISK_KINDS: tuple[str, ...] = ("cellular", "staging", "clinical", "helios")


def train_risk_model(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    estimator: str = "sklearn_linear",
) -> Model:
    """Fit a simple risk estimator for one fold.

    GRAIN: one fold (its train rows) per call.

    Parameters
    ----------
    features:
        One row per training image; columns are the risk kind's input features.
    target:
        Per-image outcome aligned to ``features`` (e.g. 5-year recurrence).
    estimator:
        Which simple model to fit (``sklearn_linear``, ``logreg``,
        ``random_forest``, ...).

    Returns the fitted :class:`~helios.models.base.Model` (persisted as a bundle).
    """
    raise NotImplementedError("Fit the simple risk estimator (LogReg / RandomForest / linear).")


def infer_risk_model(
    models: dict[int | None, Model],
    features: pd.DataFrame,
    cv_splits: pd.DataFrame | None,
) -> pd.DataFrame:
    """Score a feature table with the K fold models and reduce to one row per image.

    GRAIN: the scored cohort per call. The feature table is fold-independent;
    only the model differs per fold. Reduction is out-of-fold (test-fold model)
    on a CV cohort / ensemble-average on deploy (``cv_splits is None``).

    Returns one row per image: ``image_id, risk_score`` (+ a ``fold`` provenance
    column).
    """
    raise NotImplementedError(
        "Score with each fold model and reduce OOF (CV) / ensemble (deploy) to one row per image."
    )


def compute_survival_curve(helios_risk: pd.DataFrame) -> pd.DataFrame:
    """Derive a per-image survival curve from the fused HELIOS risk score.

    GRAIN: the scored cohort per call.

    Returns a tidy table: ``image_id, time, survival``.
    """
    raise NotImplementedError("Map fused risk to a per-image survival curve.")
