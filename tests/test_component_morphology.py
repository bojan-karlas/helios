"""Component-level tests for helios.components.morphology (real clustering, no mocks).

Builds tiny synthetic aligned-patch-embedding hdf5 files + attention tables
(the shapes written by helios.stages.mil's predict_mil: an hdf5 "embeddings"
dataset and a patch_attention table with tile_id/attention/image_id columns)
and exercises train_morphology/infer_morphology end to end on CPU.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from helios.components.morphology import (
    MorphologyModel,
    _cumulative_attention_mask,
    infer_morphology,
    train_morphology,
)

D = 16  # tiny embedding dim for fast tests
N_TILES = 40
N_SLIDES = 12


def _write_embeddings(path: Path, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.create_dataset("embeddings", data=embeddings.astype(np.float32))


@pytest.fixture()
def cohort(tmp_path: Path) -> dict:
    """N_SLIDES slides; each has two tile populations (cluster A / cluster B).

    A slide's label is 1 iff cluster A dominates its tiles, so cluster
    presence is genuinely associated with the outcome (not just noise).
    """
    rng = np.random.default_rng(0)
    paths: dict[str, Path] = {}
    attention_frames = []
    label_rows = []

    for i in range(N_SLIDES):
        image_id = f"S{i:02d}"
        frac_a = 0.8 if i % 2 == 0 else 0.2
        is_a = rng.random(N_TILES) < frac_a
        embeddings = np.where(
            is_a[:, None],
            rng.normal(loc=1.0, scale=0.05, size=(N_TILES, D)),
            rng.normal(loc=-1.0, scale=0.05, size=(N_TILES, D)),
        ).astype(np.float32)

        path = tmp_path / f"{image_id}.hdf5"
        _write_embeddings(path, embeddings)
        paths[image_id] = path

        attn = rng.uniform(0.1, 1.0, size=N_TILES)
        attention_frames.append(
            pd.DataFrame({"tile_id": np.arange(N_TILES), "attention": attn, "image_id": image_id})
        )
        label_rows.append({"image_id": image_id, "disease_pfs_recurrence_5yfu": int(i % 2 == 0)})

    attention = pd.concat(attention_frames, ignore_index=True)
    labels = pd.DataFrame(label_rows)
    return {"paths": paths, "attention": attention, "labels": labels}


def test_cumulative_attention_mask_keeps_top_mass() -> None:
    attn = np.array([0.4, 0.3, 0.2, 0.1])
    mask = _cumulative_attention_mask(attn, cutoff=0.6)
    # Smallest set whose mass >= 60%: just the top tile (0.4) is 40%, need two: 0.4+0.3=70%>=60%.
    assert mask.tolist() == [True, True, False, False]


def test_cumulative_attention_mask_all_zero_keeps_everything() -> None:
    mask = _cumulative_attention_mask(np.zeros(5), cutoff=0.8)
    assert mask.all()


def test_train_morphology_roundtrip(cohort: dict) -> None:
    model = train_morphology(
        cohort["paths"],
        cohort["attention"],
        cohort["labels"],
        attention_cutoff=0.8,
        n_pca=8,
        k_micro=6,
        k_neighbors=3,
        resolution=1.0,
        estimator="random_forest",
        seed=0,
    )
    assert isinstance(model, MorphologyModel)
    assert model.cluster_centroids.shape[1] == 8  # n_pca
    assert len(model.feature_cluster_ids) >= 1


@pytest.mark.parametrize("estimator", ["random_forest", "logreg_l2", "logreg_l1", "elasticnet", "gradient_boosting"])
def test_train_morphology_supports_all_estimators(cohort: dict, estimator: str) -> None:
    model = train_morphology(
        cohort["paths"],
        cohort["attention"],
        cohort["labels"],
        n_pca=8,
        k_micro=6,
        k_neighbors=3,
        estimator=estimator,
        seed=0,
    )
    assert model.estimator_name == estimator


def test_infer_morphology_deploy_ensembles_across_folds(cohort: dict) -> None:
    model = train_morphology(
        cohort["paths"], cohort["attention"], cohort["labels"], n_pca=8, k_micro=6, k_neighbors=3, seed=0
    )
    models = {0: model, 1: model}
    paths = {0: cohort["paths"], 1: cohort["paths"]}
    attention = {0: cohort["attention"], 1: cohort["attention"]}

    tile_clusters, presence, risk = infer_morphology(models, paths, attention, cv_splits=None)

    assert set(risk["image_id"]) == set(cohort["paths"].keys())
    assert risk["fold"].isna().all()
    assert risk["risk_score"].between(0.0, 1.0).all()
    assert set(tile_clusters["image_id"]) == set(cohort["paths"].keys())
    assert not presence.empty


def test_infer_morphology_cv_selects_test_fold(cohort: dict) -> None:
    model = train_morphology(
        cohort["paths"], cohort["attention"], cohort["labels"], n_pca=8, k_micro=6, k_neighbors=3, seed=0
    )
    image_ids = list(cohort["paths"].keys())
    half = len(image_ids) // 2
    cv_splits = pd.DataFrame(
        [{"image_id": i, "fold": 0, "split": "test" if idx < half else "train"} for idx, i in enumerate(image_ids)]
        + [{"image_id": i, "fold": 1, "split": "test" if idx >= half else "train"} for idx, i in enumerate(image_ids)]
    )
    models = {0: model, 1: model}
    paths = {0: cohort["paths"], 1: cohort["paths"]}
    attention = {0: cohort["attention"], 1: cohort["attention"]}

    _tile_clusters, _presence, risk = infer_morphology(models, paths, attention, cv_splits=cv_splits)

    risk = risk.set_index("image_id")
    for idx, image_id in enumerate(image_ids):
        expected_fold = 0 if idx < half else 1
        assert risk.loc[image_id, "fold"] == expected_fold
