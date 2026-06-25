"""Risk fit/predict stage orchestration tests.

Focuses on the ``--risk`` selector: validation, canonical ordering (``helios``
fusion last), and per-fold model writing, with the pure risk components
monkeypatched.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from helios.data.io import ArtifactStore
from helios.data.resolver import Resolver, Roots
from helios.stages import risk
from helios.stages._runtime import tag_dataset
from tests.fixtures import make_cohort


@pytest.fixture()
def cohort(tmp_path: Path) -> Path:
    return make_cohort(tmp_path / "cohort", n_images=4, n_patients=4)


def test_features_fusion_strips_provenance(cohort: Path) -> None:
    """Tagged sub-scores must not leak a ``dataset`` column into fusion features."""
    store = ArtifactStore(Resolver(Roots.single(cohort)))
    ids = [f"FAKE-SKCM-P{i:05d}-S01-B1-I1" for i in range(4)]
    for sub in risk._FUSION_INPUTS:
        frame = pd.DataFrame({"image_id": ids, sub: range(len(ids))})
        store.write(sub, tag_dataset(frame, cohort))

    features = risk._features("helios", store, ids, "path_stage", [])

    assert "dataset" not in features.columns
    assert "dataset_x" not in features.columns and "dataset_y" not in features.columns
    assert sorted(features["image_id"]) == sorted(ids)


def test_fit_rejects_unknown_kind(cohort: Path) -> None:
    with pytest.raises(ValueError, match="Unknown risk kind"):
        risk.fit_risk(datasets=[cohort], risk=["not-a-kind"])


def test_fit_orders_helios_last(monkeypatch, cohort: Path) -> None:
    order: list[str] = []

    def _fake_train(features, targets, *, estimator="sklearn_linear"):
        return {"n": len(features)}

    def _fake_features(kind, store, ids, staging_feature, clinical_features):
        return pd.DataFrame({"image_id": ids, "x": range(len(ids))})

    # Capture the order models are written by wrapping store.write.
    original_write = ArtifactStore.write

    def _spy_write(self, artifact_id, obj, **keys):
        if artifact_id.startswith("model_"):
            order.append(artifact_id)
        return original_write(self, artifact_id, obj, **keys)

    monkeypatch.setattr("helios.stages.risk.train_risk_model", _fake_train)
    monkeypatch.setattr("helios.stages.risk._features", _fake_features)
    monkeypatch.setattr(ArtifactStore, "write", _spy_write)

    risk.fit_risk(
        datasets=[cohort],
        risk=["helios", "clinical"],
        target="disease_pfs_recurred",
        staging_feature="path_stage",
    )

    assert order[-1] == "model_fusion", "fusion (helios) must train last"
    assert "model_clinical" in order


def test_predict_requires_fitted_model(cohort: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No model_clinical"):
        risk.predict_risk(datasets=[cohort], risk=["clinical"])
