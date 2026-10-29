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
from helios.data.resolver import Resolver, Roots, StorageRoots
from helios.stages import concept as concept_stage
from helios.stages import mil
from helios.stages._runtime import cohort_store, resolve_datasets
from helios.stages.tile_features import prep_tile_features
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
    store.write("splits", pd.DataFrame(rows))
    for image_id in IMAGE_IDS:
        store.write(
            "tile_features",
            {"features": np.ones((3, 4), dtype=np.float32)},
            size_mm=0.25,
            model="virchow2",
            image_id=image_id,
        )
    return root


def test_fit_pools_train_ids_per_fold(monkeypatch, cohort: Path) -> None:
    seen: dict[int, list[str]] = {}

    def _fake_train(slides, **kwargs):
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

    def _fake_train(slides, **kwargs):
        seen[len(seen)] = [s.image_id for s in slides]
        return {}

    monkeypatch.setattr("helios.stages.mil.train_mil", _fake_train)

    # P00000 is the only train image of fold 0; restrict to it via metadata query.
    mil.fit_mil(datasets=[ResolvedDataset(cohort, where="patient_id == 'P00000'")])

    assert seen[0] == [IMAGE_IDS[0]], "filter must intersect the fold's train split"
    assert seen[1] == [], "no fold-1 train image matches the filter"


def test_fit_mil_raises_on_missing_slide_features(tmp_path: Path) -> None:
    root = make_cohort(tmp_path / "incomplete", n_images=2, n_patients=2)
    with pytest.raises(FileNotFoundError, match="Missing tile_features"):
        mil.fit_mil(datasets=[root])


def test_predict_scores_each_fold_then_reduces(monkeypatch, cohort: Path) -> None:
    store = ArtifactStore(Resolver(Roots.single(cohort)))
    for fold in (0, 1):
        store.write("model_mil", {"fold": fold}, fold=fold)

    def _fake_infer(model, *, features_path, background_path=None, device="cuda"):
        return MilSlideOutputs(
            patch_embeddings=np.zeros((3, 4), dtype=np.float32),
            attention=pd.DataFrame(
                {"tile_id": [0, 1, 2], "attention": [0.1, 0.2, 0.7], "tile_gradient": [0.0, 0.0, 0.0]}
            ),
            slide_embedding=np.ones(4, dtype=np.float32),
            slide_embedding_grad=np.zeros(4, dtype=np.float32),
            risk_score=0.5,
            slide_hidden_embedding=np.ones(8, dtype=np.float32),
            slide_hidden_embedding_grad=np.zeros(8, dtype=np.float32),
        )

    def _fake_reduce(per_fold_scores, splits):
        return per_fold_scores.groupby("image_id", as_index=False)["risk_score"].mean()

    monkeypatch.setattr("helios.stages.mil.infer_mil", _fake_infer)
    monkeypatch.setattr("helios.stages.mil.reduce_whole_image_risk", _fake_reduce)

    mil.predict_mil(datasets=[cohort])

    for fold in (0, 1):
        assert store.exists("whole_image_risk_score", fold=fold)
        for image_id in IMAGE_IDS:
            assert store.exists("aligned_slide_embedding", fold=fold, image_id=image_id)
            assert store.exists("aligned_slide_embedding_grad", fold=fold, image_id=image_id)
            assert store.exists("aligned_slide_hidden_embedding", fold=fold, image_id=image_id)
            assert store.exists("aligned_slide_hidden_embedding_grad", fold=fold, image_id=image_id)
    ensemble = store.read("whole_image_risk_score_ensemble")
    assert sorted(ensemble["image_id"]) == sorted(IMAGE_IDS)
    assert (ensemble["dataset"] == Path(cohort).name).all(), "ensemble must carry dataset provenance"


def test_union_extract_fit_predict_and_downstream_fit_use_split_roots(monkeypatch, tmp_path: Path) -> None:
    datasets = []
    originals = {}
    for name in ("A", "B"):
        source = make_cohort(tmp_path / name, n_images=2, n_patients=2)
        originals.update({p: p.read_bytes() for p in source.rglob("*") if p.is_file()})
        ds = ResolvedDataset(
            source, preprocessing_root=tmp_path / "pre" / name
        )
        datasets.append(ds)
        store = cohort_store(ds)
        for image_id in ds.dataset().image_ids:
            store.write("tiles", {"tiles": np.zeros((3, 4, 4, 3), np.uint8)}, size_mm=0.25, image_id=image_id)
    roots = StorageRoots(models_root=tmp_path / "experiment", results_root=tmp_path / "experiment")

    monkeypatch.setattr(
        "helios.stages.tile_features.extract_features",
        lambda tiles, **kwargs: np.ones((len(tiles), 4), np.float32),
    )
    prep_tile_features(datasets=datasets, roots=roots)

    def fake_train(slides, **kwargs):
        assert len(slides) == 4
        for slide in slides:
            assert slide.features_path.is_relative_to(tmp_path / "pre")
            assert slide.features_path.exists()
            assert slide.background_path.is_relative_to(tmp_path / "pre")
        return {"trained": True}

    monkeypatch.setattr(mil, "train_mil", fake_train)
    mil.fit_mil(datasets=datasets, roots=roots)
    assert (tmp_path / "experiment/mil").is_dir()

    def fake_infer(model, *, features_path, **kwargs):
        assert model["trained"]
        assert features_path.is_relative_to(tmp_path / "pre")
        return MilSlideOutputs(
            patch_embeddings=np.ones((3, 4), np.float32),
            attention=pd.DataFrame({"tile_id": [0, 1, 2], "attention": [0.2, 0.3, 0.5]}),
            slide_embedding=np.ones(4, np.float32),
            slide_embedding_grad=np.zeros(4, np.float32),
            risk_score=0.5,
            slide_hidden_embedding=np.ones(8, np.float32),
            slide_hidden_embedding_grad=np.zeros(8, np.float32),
        )

    monkeypatch.setattr(mil, "infer_mil", fake_infer)
    mil.predict_mil(datasets=datasets, roots=roots)
    for ds in datasets:
        store = cohort_store(ds, roots)
        table = store.read("whole_image_risk_score_ensemble")
        assert len(table) == 2
        assert set(table["dataset"]) == {ds.name}
        assert store.path("aligned_slide_embedding", fold=None, image_id=ds.dataset().image_ids[0]).exists()
        # Reading a constituent alone must use the outputs written by the union.
        (single,) = resolve_datasets([ds], roots=roots)
        assert cohort_store(single, roots).path("whole_image_risk_score_ensemble").exists()

    def fake_train_concepts(embeddings, labels, risk, **kwargs):
        assert embeddings.shape == (4, 4)
        assert len(labels) == len(risk) == 4
        return {"downstream": True}

    monkeypatch.setattr(concept_stage, "train_concepts", fake_train_concepts)
    concept_stage.fit_concept(datasets=datasets, roots=roots)
    assert cohort_store(datasets[0], roots).read("model_concepts", fold=None)["downstream"]
    for p, contents in originals.items():
        assert p.read_bytes() == contents
    for ds in datasets:
        assert not (ds.root / "models").exists()
        assert not (ds.root / "preprocessing").exists()
        assert not (ds.root / "results").exists()
