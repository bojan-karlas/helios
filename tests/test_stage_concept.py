"""Concept stage orchestration and row-alignment tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from helios.data.io import ArtifactStore
from helios.data.resolver import Resolver, Roots
from helios.stages import concept
from tests.fixtures import make_cohort

IMAGE_IDS = [f"FAKE-SKCM-P{i:05d}-S01-B1-I1" for i in range(4)]


def _prepared_cohort(tmp_path: Path) -> tuple[Path, ArtifactStore]:
    root = make_cohort(tmp_path / "cohort", n_images=4, n_patients=4)
    store = ArtifactStore(Resolver(Roots.single(root)))
    for image_index, image_id in enumerate(IMAGE_IDS):
        store.write(
            "aligned_slide_embedding",
            np.full(3, image_index, dtype=np.float32),
            fold=None,
            image_id=image_id,
        )
    # Deliberately reversed: the stage must align this table to embedding order.
    store.write(
        "whole_image_risk_score",
        pd.DataFrame({"image_id": IMAGE_IDS[::-1], "risk_score": [0.4, 0.3, 0.2, 0.1]}),
        fold=None,
    )
    return root, store


def test_fit_concept_aligns_embeddings_labels_and_risk(monkeypatch, tmp_path: Path) -> None:
    root, store = _prepared_cohort(tmp_path)
    captured: dict[str, object] = {}

    def fake_train(embeddings, labels, risk, *, n_bootstrap):
        captured.update(embeddings=embeddings, labels=labels, risk=risk, n_bootstrap=n_bootstrap)
        return {"fitted": True}

    monkeypatch.setattr("helios.stages.concept.train_concepts", fake_train)
    concept.fit_concept(datasets=[root], n_bootstrap=2)

    assert store.exists("model_concepts", fold=None)
    assert captured["labels"]["image_id"].tolist() == IMAGE_IDS
    assert captured["risk"]["image_id"].tolist() == IMAGE_IDS
    assert np.asarray(captured["embeddings"])[:, 0].tolist() == [0, 1, 2, 3]


def test_predict_concept_writes_reduced_outputs(monkeypatch, tmp_path: Path) -> None:
    root, store = _prepared_cohort(tmp_path)
    store.write("model_concepts", {"fitted": True}, fold=None)

    def fake_infer(models, embeddings, risk, cv_splits):
        assert risk[None]["image_id"].tolist() == IMAGE_IDS
        predictions = pd.DataFrame({"image_id": IMAGE_IDS, "path_thickness": range(4)})
        scores = pd.DataFrame({"image_id": IMAGE_IDS, "risk_score": np.linspace(0.1, 0.4, 4)})
        return predictions, scores

    monkeypatch.setattr("helios.stages.concept.infer_concepts", fake_infer)
    concept.predict_concept(datasets=[root])

    predictions = store.read("path_concept_predictions")
    scores = store.read("path_concept_risk_score")
    assert predictions["image_id"].tolist() == IMAGE_IDS
    assert scores["image_id"].tolist() == IMAGE_IDS
    assert predictions["dataset"].eq(root.name).all()
