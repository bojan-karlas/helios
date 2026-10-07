"""Component tests for bootstrapped concept activation vector modelling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helios.components.concepts import ConceptModel, infer_concepts, train_concepts


def _training_data() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(20, 6)).astype(np.float32)
    labels = pd.DataFrame({
        "image_id": [f"S{i:02d}" for i in range(20)],
        "path_ulceration_present": (embeddings[:, 0] > 0).astype(int),
        "path_thickness": embeddings[:, 1] * 2 + 3,
        "path_histologic_subtype": np.where(embeddings[:, 2] > 0, "nodular", "superficial"),
        "patient_sex": "unused",
    })
    risk = pd.DataFrame({
        "image_id": labels["image_id"],
        "risk_score": 0.5 + 0.2 * embeddings[:, 0] + 0.1 * embeddings[:, 1],
    })
    return embeddings, labels, risk


def test_train_concepts_builds_bootstrapped_cav_ensemble() -> None:
    embeddings, labels, risk = _training_data()
    model = train_concepts(embeddings, labels, risk, n_bootstrap=3)

    assert isinstance(model, ConceptModel)
    assert all(len(probe.estimators) == 3 for probe in model.probes)
    assert "path_ulceration_present" in model.concept_names
    assert "path_thickness" in model.concept_names
    assert "path_histologic_subtype: nodular" in model.concept_names
    assert "patient_sex" not in model.concept_names
    predicted, _, _ = model.predict(embeddings)
    subtype_columns = [
        column for column in model.concept_names if column.startswith("path_histologic_subtype: ")
    ]
    assert np.allclose(predicted[subtype_columns].sum(axis=1), 1.0)


def test_infer_concepts_ensembles_folds_and_reports_uncertainty() -> None:
    embeddings, labels, risk = _training_data()
    first = train_concepts(embeddings, labels, risk, n_bootstrap=3)
    second = train_concepts(embeddings, labels, risk, n_bootstrap=3)

    predictions, scores = infer_concepts(
        {0: first, 1: second},
        {0: embeddings, 1: embeddings},
        {0: risk, 1: risk},
        None,
    )

    assert len(predictions) == len(labels)
    assert "path_ulceration_present__std" in predictions
    assert predictions["fold"].isna().all()
    assert list(scores.columns) == [
        "image_id", "concept_risk_score", "concept_risk_score_residual", "risk_score", "fold"
    ]
    assert np.isfinite(scores["risk_score"]).all()


def test_infer_concepts_uses_oof_fold_and_pathologist_override() -> None:
    embeddings, labels, risk = _training_data()
    models = {
        fold: train_concepts(embeddings, labels, risk, n_bootstrap=1)
        for fold in (0, 1)
    }
    splits = pd.DataFrame({
        "image_id": np.repeat(labels["image_id"].to_numpy(), 2),
        "fold": np.tile([0, 1], len(labels)),
        "split": ["test", "train"] * len(labels),
    })
    overrides = pd.DataFrame({
        "image_id": [labels.loc[0, "image_id"]],
        "path_ulceration_present": [0.25],
    })

    predictions, scores = infer_concepts(
        models,
        {0: embeddings, 1: embeddings},
        {0: risk, 1: risk},
        splits,
        overrides=overrides,
    )

    assert predictions["fold"].eq(0).all()
    assert scores["fold"].eq(0).all()
    corrected = predictions.set_index("image_id").loc[
        labels.loc[0, "image_id"], "path_ulceration_present"
    ]
    assert corrected == pytest.approx(0.25)


def test_train_concepts_rejects_missing_concepts() -> None:
    embeddings, labels, risk = _training_data()
    with pytest.raises(ValueError, match="No usable"):
        train_concepts(embeddings, labels[["image_id"]], risk, n_bootstrap=1)
