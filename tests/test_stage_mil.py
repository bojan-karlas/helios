"""A-MIL fit/predict stage orchestration tests.

Exercises the pooled per-fold training loop and the per-fold inference + reduce
shape, with the pure A-MIL components monkeypatched to deterministic stand-ins.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helios.components.mil import MilSlideOutputs
from helios.data.cohort import ResolvedDataset
from helios.data.io import ArtifactStore
from helios.data.resolver import Resolver, Roots
from helios.stages import mil
from tests.fixtures import make_cohort

IMAGE_IDS = [f"FAKE-SKCM-P{i:05d}-S01-B1-I1" for i in range(4)]


@pytest.fixture()
def cohort(tmp_path: Path) -> Path:
    root = make_cohort(tmp_path / "cohort", n_images=4, n_patients=4)
    store = ArtifactStore(Resolver(Roots.single(root)))
    rows = []
    for fold in (0, 1):
        for idx, image_id in enumerate(IMAGE_IDS):
            in_train = (idx < 2) if fold == 0 else (idx >= 2)
            rows.append(
                {"image_id": image_id, "fold": fold, "split": "train" if in_train else "test"}
            )
    store.write("cv_splits", pd.DataFrame(rows))
    return root


def test_fit_pools_train_ids_per_fold(monkeypatch, cohort: Path) -> None:
    seen: dict[int, list[str]] = {}

    def _fake_train(slides, *, aug_swap_prob=0.5, device="cuda"):
        fold = len(seen)  # called once per fold in order
        ids = [s.image_id for s in slides]
        seen[fold] = ids
        return {"train_ids": ids}

    monkeypatch.setattr("helios.stages.mil.train_mil", _fake_train)

    mil.fit_mil(datasets=[cohort])

    store = ArtifactStore(Resolver(Roots.single(cohort)))
    assert store.exists("model_mil", fold=0)
    assert store.exists("model_mil", fold=1)
    assert seen[0] == IMAGE_IDS[:2]
    assert seen[1] == IMAGE_IDS[2:]


def test_fit_filter_subsets_train_ids(monkeypatch, cohort: Path) -> None:
    seen: dict[int, list[str]] = {}

    def _fake_train(slides, *, aug_swap_prob=0.5, device="cuda"):
        seen[len(seen)] = [s.image_id for s in slides]
        return {}

    monkeypatch.setattr("helios.stages.mil.train_mil", _fake_train)

    # P00000 is the only train image of fold 0; restrict to it via metadata query.
    mil.fit_mil(datasets=[ResolvedDataset(cohort, where="patient_id == 'P00000'")])

    assert seen[0] == [IMAGE_IDS[0]], "filter must intersect the fold's train split"
    assert seen[1] == [], "no fold-1 train image matches the filter"


def test_predict_scores_each_fold_then_reduces(monkeypatch, cohort: Path) -> None:
    store = ArtifactStore(Resolver(Roots.single(cohort)))
    for fold in (0, 1):
        store.write("model_mil", {"fold": fold}, fold=fold)

    def _fake_infer(model, *, features_path, background_path=None, device="cuda"):
        return MilSlideOutputs(
            patch_embeddings=np.zeros((3, 4), dtype=np.float32),
            attention=pd.DataFrame({"tile_id": [0, 1, 2], "attention": [0.1, 0.2, 0.7]}),
            slide_embedding=np.ones(4, dtype=np.float32),
            risk_score=0.5,
        )

    def _fake_reduce(per_fold_scores, cv_splits):
        return per_fold_scores.groupby("image_id", as_index=False)["risk_score"].mean()

    monkeypatch.setattr("helios.stages.mil.infer_mil", _fake_infer)
    monkeypatch.setattr("helios.stages.mil.reduce_whole_image_risk", _fake_reduce)

    mil.predict_mil(datasets=[cohort])

    for fold in (0, 1):
        assert store.exists("whole_image_risk_score", fold=fold)
        for image_id in IMAGE_IDS:
            assert store.exists("aligned_slide_embedding", fold=fold, image_id=image_id)
    ensemble = store.read("whole_image_risk_score_ensemble")
    assert sorted(ensemble["image_id"]) == sorted(IMAGE_IDS)
    assert (ensemble["dataset"] == Path(cohort).name).all(), "ensemble must carry dataset provenance"
