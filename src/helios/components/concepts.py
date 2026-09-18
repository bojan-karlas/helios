"""Pure pathological-concept modelling components.

Ports the production part of ``concept-activation-vectors.ipynb``:
bootstrapped linear concept activation vectors, an interpretable sparse concept
bottleneck, and linear residual correction from the MIL activation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.linear_model import LassoCV, LogisticRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

from helios.models.base import Model

_ID = "image_id"
_RISK = "risk_score"


@dataclass
class ConceptProbe:
    """One output concept and its bootstrapped linear estimators."""

    name: str
    source: str
    category: object | None
    categorical: bool
    estimators: list[Any]


@dataclass
class ConceptModel(Model):
    """Fitted CAV ensemble, concept bottleneck, and residual model."""

    probes: list[ConceptProbe]
    concept_names: list[str]
    concept_scaler: StandardScaler
    concept_risk_model: Any
    residual_scaler: StandardScaler
    residual_model: Any

    def fit(self, inputs: Any, **params: Any) -> Model:
        raise RuntimeError("ConceptModel is fitted by train_concepts().")

    def predict(self, inputs: Any) -> Any:
        embeddings = _validate_embeddings(np.asarray(inputs, dtype=np.float32))
        concepts = _predict_probe_ensemble(self.probes, embeddings)
        concept_x = self.concept_scaler.transform(concepts[self.concept_names])
        concept_risk = np.asarray(self.concept_risk_model.predict(concept_x), dtype=float)
        residual = np.asarray(
            self.residual_model.predict(self.residual_scaler.transform(embeddings)), dtype=float
        )
        return concepts, concept_risk, residual


def train_concepts(
    slide_embeddings: NDArray[np.float32],
    concept_labels: pd.DataFrame,
    whole_image_risk: pd.DataFrame,
    *,
    n_bootstrap: int = 100,
) -> Model:
    """Fit CAV probes, a concept risk model, and residual correction for one fold.

    GRAIN: one fold (its concept-cohort train slides) per call.

    Categorical ``path_*`` columns are expanded into one-vs-rest concepts. A
    binary 0/1 or boolean column retains its source name; numeric non-binary
    concepts use ridge regression directly. Missing labels are omitted per probe.
    """
    x = _validate_embeddings(slide_embeddings)
    if len(concept_labels) != len(x):
        raise ValueError("concept_labels and slide_embeddings must have the same number of rows.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    probes = _fit_probes(x, concept_labels, n_bootstrap=n_bootstrap)
    if not probes:
        raise ValueError("No usable path_* concept columns were found.")
    predicted = _predict_probe_ensemble(probes, x)
    risk = _aligned_risk(concept_labels, whole_image_risk)
    valid = np.isfinite(risk)
    if valid.sum() < 2:
        raise ValueError("At least two aligned, finite whole-image risk values are required.")

    concept_names = [probe.name for probe in probes]
    concept_scaler = StandardScaler().fit(predicted.loc[valid, concept_names])
    concept_x = concept_scaler.transform(predicted.loc[valid, concept_names])
    risk_model = _fit_lasso(concept_x, risk[valid])
    concept_oof = _cross_fitted_predictions(risk_model, concept_x, risk[valid])
    residual_scaler = StandardScaler().fit(x[valid])
    residual_model = _fit_ridge(
        residual_scaler.transform(x[valid]), risk[valid] - concept_oof
    )
    return ConceptModel(
        probes, concept_names, concept_scaler, risk_model, residual_scaler, residual_model
    )


def infer_concepts(
    models: dict[int | None, Model],
    slide_embeddings: dict[int | None, NDArray[np.float32]],
    whole_image_risk: dict[int | None, pd.DataFrame],
    cv_splits: pd.DataFrame | None,
    *,
    overrides: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict concepts and corrected risk, reducing folds OOF/ensemble.

    GRAIN: one scored cohort per call. Non-null override values, keyed by
    ``image_id``, replace predictions before the concept bottleneck is evaluated.
    """
    if not models:
        raise ValueError("At least one concept model is required.")
    fold_predictions: list[pd.DataFrame] = []
    fold_risks: list[pd.DataFrame] = []
    for fold, generic_model in models.items():
        if not isinstance(generic_model, ConceptModel):
            raise TypeError(f"Expected ConceptModel, got {type(generic_model).__name__}.")
        if fold not in slide_embeddings or fold not in whole_image_risk:
            raise KeyError(f"Missing embeddings or whole-image risk for fold {fold}.")
        x = _validate_embeddings(slide_embeddings[fold])
        ids = _risk_ids(whole_image_risk[fold], len(x))
        concepts = _predict_probe_ensemble(generic_model.probes, x)
        concepts.insert(0, _ID, ids)
        concepts = _apply_overrides(concepts, overrides)

        concept_x = generic_model.concept_scaler.transform(
            concepts[generic_model.concept_names]
        )
        concept_risk = generic_model.concept_risk_model.predict(concept_x)
        residual = generic_model.residual_model.predict(
            generic_model.residual_scaler.transform(x)
        )
        fold_predictions.append(concepts.assign(fold=fold))
        fold_risks.append(pd.DataFrame({
            _ID: ids,
            "concept_risk_score": concept_risk,
            "concept_risk_score_residual": residual,
            _RISK: np.asarray(concept_risk) + np.asarray(residual),
            "fold": fold,
        }))

    return (
        _reduce_folds(pd.concat(fold_predictions, ignore_index=True), cv_splits),
        _reduce_folds(pd.concat(fold_risks, ignore_index=True), cv_splits),
    )


def _fit_probes(
    x: NDArray[np.float32], labels: pd.DataFrame, *, n_bootstrap: int
) -> list[ConceptProbe]:
    probes: list[ConceptProbe] = []
    for column in labels.columns:
        if not column.startswith("path_"):
            continue
        series = labels[column]
        observed = series.dropna()
        if observed.nunique() < 2:
            continue
        numeric = pd.api.types.is_numeric_dtype(observed)
        binary = pd.api.types.is_bool_dtype(observed) or (
            numeric and set(np.asarray(observed, dtype=float).tolist()) <= {0.0, 1.0}
        )
        if numeric and not binary:
            probe = _bootstrap_probe(
                x, series.astype(float), column, column, None, False, n_bootstrap
            )
            if probe is not None:
                probes.append(probe)
            continue
        categories = list(pd.unique(observed))
        targets = (
            [(column, 1, series.astype("Float64") == 1)]
            if binary
            else [(f"{column}: {value}", value, series == value) for value in categories]
        )
        for name, category, target in targets:
            numeric_target = target.astype(float).where(series.notna())
            probe = _bootstrap_probe(
                x, numeric_target, name, column, category, True, n_bootstrap
            )
            if probe is not None:
                probes.append(probe)
    return probes


def _bootstrap_probe(
    x: NDArray[np.float32], target: pd.Series, name: str, source: str,
    category: object | None, categorical: bool, n_bootstrap: int,
) -> ConceptProbe | None:
    valid = target.notna().to_numpy()
    y = target.loc[valid].to_numpy(dtype=float)
    if len(y) < 2 or (categorical and np.unique(y).size < 2):
        return None
    base: Any = LogisticRegression(max_iter=1000, random_state=42) if categorical else Ridge()
    rng = np.random.default_rng(42)
    estimators: list[Any] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y), len(y)) if n_bootstrap > 1 else np.arange(len(y))
        for _attempt in range(20):
            if not categorical or np.unique(y[indices]).size == 2:
                break
            indices = rng.integers(0, len(y), len(y))
        if categorical and np.unique(y[indices]).size < 2:
            indices = np.arange(len(y))
        estimators.append(clone(base).fit(x[valid][indices], y[indices]))
    return ConceptProbe(name, source, category, categorical, estimators)


def _predict_probe_ensemble(
    probes: list[ConceptProbe], x: NDArray[np.float32]
) -> pd.DataFrame:
    values: dict[str, NDArray[np.float64]] = {}
    for probe in probes:
        predictions = np.stack([
            estimator.predict_proba(x)[:, 1] if probe.categorical else estimator.predict(x)
            for estimator in probe.estimators
        ])
        values[probe.name] = predictions.mean(axis=0)
        values[f"{probe.name}__std"] = predictions.std(axis=0)
    result = pd.DataFrame(values)
    # The notebook treats the one-vs-rest outputs belonging to one categorical
    # pathology field as a distribution, using a low-temperature softmax.
    groups: dict[str, list[str]] = {}
    for probe in probes:
        if probe.categorical and probe.name != probe.source:
            groups.setdefault(probe.source, []).append(probe.name)
    for names in groups.values():
        logits = result[names].to_numpy(dtype=float) / 0.1
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        result[names] = probabilities / probabilities.sum(axis=1, keepdims=True)
    return result


def _aligned_risk(labels: pd.DataFrame, risk: pd.DataFrame) -> NDArray[np.float64]:
    if _RISK not in risk:
        raise ValueError(f"whole_image_risk must contain {_RISK!r}.")
    if _ID in labels and _ID in risk:
        lookup = risk.drop_duplicates(_ID).set_index(_ID)[_RISK]
        return labels[_ID].map(lookup).to_numpy(dtype=float)
    if len(labels) != len(risk):
        raise ValueError("whole_image_risk cannot be aligned without image_id.")
    return risk[_RISK].to_numpy(dtype=float)


def _risk_ids(risk: pd.DataFrame, n_rows: int) -> list[str]:
    if _ID not in risk or len(risk) != n_rows:
        raise ValueError("Per-fold whole_image_risk image_id rows must align with embeddings.")
    return risk[_ID].astype(str).tolist()


def _apply_overrides(
    predictions: pd.DataFrame, overrides: pd.DataFrame | None
) -> pd.DataFrame:
    if overrides is None:
        return predictions
    if _ID not in overrides:
        raise ValueError("overrides must contain image_id.")
    result = predictions.set_index(_ID)
    corrections = overrides.drop_duplicates(_ID).set_index(_ID)
    for column in result.columns.intersection(corrections.columns):
        replacement = corrections[column].reindex(result.index)
        result[column] = replacement.where(replacement.notna(), result[column])
    return result.reset_index()


def _reduce_folds(frame: pd.DataFrame, cv_splits: pd.DataFrame | None) -> pd.DataFrame:
    if cv_splits is None:
        numeric = [c for c in frame.columns if c not in {_ID, "fold"}]
        reduced = frame.groupby(_ID, sort=False)[numeric].mean().reset_index()
        reduced["fold"] = None
        return reduced
    required = {_ID, "fold", "split"}
    if not required <= set(cv_splits.columns):
        raise ValueError(f"cv_splits must contain {sorted(required)}.")
    test = cv_splits.loc[cv_splits["split"] == "test", [_ID, "fold"]]
    reduced = frame.merge(test, on=[_ID, "fold"], how="inner")
    if reduced[_ID].duplicated().any():
        raise ValueError("cv_splits defines multiple test-fold predictions for an image.")
    return reduced.reset_index(drop=True)


def _fit_lasso(x: NDArray[np.float64], y: NDArray[np.float64]) -> Any:
    return LassoCV(cv=min(5, len(y)), max_iter=10_000, random_state=42).fit(x, y)


def _fit_ridge(x: NDArray[np.float64], y: NDArray[np.float64]) -> Any:
    return RidgeCV(cv=min(5, len(y))).fit(x, y)


def _cross_fitted_predictions(
    model: Any, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    folds = min(5, len(y))
    result = np.empty(len(y), dtype=float)
    for fold in range(folds):
        held_out = np.arange(len(y)) % folds == fold
        if (~held_out).sum() < 2:
            result[held_out] = model.predict(x[held_out])
        else:
            result[held_out] = clone(model).fit(x[~held_out], y[~held_out]).predict(x[held_out])
    return result


def _validate_embeddings(values: NDArray[np.float32]) -> NDArray[np.float32]:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"slide_embeddings must be a non-empty 2-D matrix; got {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError("slide_embeddings contains non-finite values.")
    return result
