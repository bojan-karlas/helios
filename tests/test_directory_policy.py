"""Persistence is opt-in and never changes input lookup implicitly."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from helios.cli.main import app
from helios.components.mil import MilSlideOutputs
from helios.data.cohort import ResolvedDataset
from helios.data.resolver import StorageDirs
from helios.stages import augment, mil
from helios.stages._ray import ParallelConfig
from helios.stages._runtime import cohort_store, resolve_datasets
from helios.stages.splits import prep_splits
from helios.stages.tile_features import prep_tile_features
from tests.fixtures import make_cohort, prepared_store


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def features(root):
    store = prepared_store(root)
    for image_id in ResolvedDataset(root).dataset().image_ids:
        store.write(
            "tile_features", {"features": np.ones((2, 4), np.float32)},
            size_mm=0.25, model="virchow2", image_id=image_id,
        )
    return store


def test_training_without_destination_computes_but_does_not_save(monkeypatch, tmp_path, caplog):
    root = make_cohort(tmp_path / "source", n_images=2)
    features(root)
    before = snapshot(tmp_path)
    calls = []
    monkeypatch.setattr(mil, "train_mil", lambda slides, **kwargs: calls.append(slides) or {"trained": True})
    mil.fit_mil(datasets=[root])
    assert len(calls) == 1
    assert snapshot(tmp_path) == before
    assert "models outputs will NOT be saved" in caplog.text


def test_cli_training_output_defaults_and_independent_logs(monkeypatch, tmp_path):
    root = make_cohort(tmp_path / "source", n_images=2)
    features(root)
    before = snapshot(root)
    monkeypatch.setattr(mil, "train_mil", lambda slides, **kwargs: {"trained": True})
    result = CliRunner().invoke(app, [
        "fit", "mil", "--dataset", str(root), "--output-dir", str(tmp_path / "run"),
    ])
    assert result.exit_code == 0, (result.exception, result.output)
    assert (tmp_path / "run/models/mil/fold=None/manifest.json").is_file()
    assert len(list((tmp_path / "run/logs").glob("*/run.log"))) == 1
    assert not (tmp_path / "run/preprocessing").exists()
    assert snapshot(root) == before


def test_deployment_reads_models_without_writing_to_them(monkeypatch, tmp_path):
    root = make_cohort(tmp_path / "deployment", n_images=2)
    features(root)
    model_dirs = StorageDirs(models_dir=tmp_path / "trained")
    cohort_store(root, model_dirs).write("model_mil", {"trained": True}, fold=None)
    before = snapshot(tmp_path / "trained")
    source_before = snapshot(root)

    def infer(model, **kwargs):
        assert model == {"trained": True}
        return MilSlideOutputs(
            patch_embeddings=np.ones((2, 4), np.float32),
            attention=pd.DataFrame({"tile_id": [0, 1], "attention": [0.4, 0.6]}),
            slide_embedding=np.ones(4, np.float32), slide_embedding_grad=np.zeros(4, np.float32),
            risk_score=0.5,
        )

    monkeypatch.setattr(mil, "infer_mil", infer)
    dirs = replace(model_dirs, output_dir=tmp_path / "inference")
    mil.predict_mil(datasets=[root], dirs=dirs)
    output = cohort_store(root, dirs).output_path("whole_image_risk_score_ensemble")
    assert len(pd.read_csv(output)) == 2
    assert snapshot(tmp_path / "trained") == before
    assert snapshot(root) == source_before
    assert not (tmp_path / "inference/models").exists()
    # Existing parts and output checks also support a repeat invocation.
    mil.predict_mil(datasets=[root], dirs=dirs)
    assert len(pd.read_csv(output)) == 2
    monkeypatch.setattr(mil, "infer_mil", infer)
    before = snapshot(tmp_path)
    mil.predict_mil(datasets=[root], dirs=model_dirs)
    assert snapshot(tmp_path) == before


def test_memory_store_supports_parts_and_read_after_write_without_paths(tmp_path):
    store = cohort_store(tmp_path / "source")
    store.write("aligned_slide_embedding", np.ones(3), image_id="slide")
    np.testing.assert_array_equal(store.read("aligned_slide_embedding", image_id="slide"), np.ones(3))
    with pytest.raises(ValueError, match="only in memory"):
        store.path("aligned_slide_embedding", image_id="slide")
    store.write_part("whole_image_risk_score", "slide", pd.DataFrame({"image_id": ["slide"], "risk_score": [0.5]}))
    assert store.read_parts("whole_image_risk_score", ["slide"])["risk_score"].tolist() == [0.5]
    assert not store.output_exists("aligned_slide_embedding", image_id="slide")
    assert not (tmp_path / "source").exists()


def test_memory_preprocessing_does_not_skip_or_overwrite_existing_inputs(monkeypatch, tmp_path, caplog):
    root = make_cohort(tmp_path / "source", n_images=1)
    store = features(root)
    image_id = ResolvedDataset(root).dataset().image_ids[0]
    store.write("tiles", {"tiles": np.zeros((2, 2, 2, 3), np.uint8)}, size_mm=0.25, image_id=image_id)
    before = snapshot(root)
    calls = []
    monkeypatch.setattr(
        "helios.stages.tile_features.extract_features",
        lambda tiles, **kwargs: calls.append(len(tiles)) or np.zeros((2, 4), np.float32),
    )
    prep_tile_features(datasets=[root], size_mm=[0.25], model=["virchow2"])
    assert calls == [2]
    assert snapshot(root) == before
    assert "preprocessing outputs will NOT be saved" in caplog.text


def test_unsaved_splits_do_not_create_cohort_sidecar(tmp_path, caplog):
    root = make_cohort(tmp_path / "source", n_images=6)
    path = tmp_path / "cohort.yaml"
    path.write_text(yaml.safe_dump({"name": "example", "datasets": [str(root)]}))
    before = snapshot(tmp_path)
    prep_splits(datasets=[path], n_folds=2, stratify_by=[])
    assert snapshot(tmp_path) == before
    assert "Splits will NOT be saved" in caplog.text


def test_shared_preprocessing_accepts_unique_cohort_images(tmp_path):
    a = make_cohort(tmp_path / "a", n_images=1)
    b = make_cohort(tmp_path / "b", n_images=1)
    meta = b / "metadata/image_metadata.csv"
    table = pd.read_csv(meta).assign(image_id="other-slide")
    table.to_csv(meta, index=False)
    datasets = [ResolvedDataset(a, cohort_name="example"), ResolvedDataset(b, cohort_name="example")]
    dirs = StorageDirs(preprocessing_dir=tmp_path / "pre")
    resolved = resolve_datasets(datasets, dirs=dirs)
    assert all(cohort_store(ds, dirs).resolver.dirs.preprocessing_dir == tmp_path / "pre" for ds in resolved)
    pd.read_csv(meta).assign(image_id=ResolvedDataset(a).dataset().image_ids[0]).to_csv(meta, index=False)
    with pytest.raises(ValueError, match="duplicate image_ids"):
        resolve_datasets(datasets, dirs=dirs)


def test_parallel_memory_reduction_rejected_before_workers(monkeypatch, tmp_path):
    root = make_cohort(tmp_path / "source")
    monkeypatch.setattr(mil, "run_work_units", lambda *a, **k: pytest.fail("must reject before scheduling"))
    with pytest.raises(ValueError, match="Parallel results reduction requires"):
        mil.predict_mil(datasets=[root], parallel=ParallelConfig(max_cpus=2))
    assert not (root / "results").exists()


def test_shared_preprocessing_keeps_augmentation_ownership_distinct(monkeypatch, tmp_path):
    a = make_cohort(tmp_path / "a", n_images=1)
    b = make_cohort(tmp_path / "b", n_images=1)
    meta = b / "metadata/image_metadata.csv"
    pd.read_csv(meta).assign(image_id="other-slide").to_csv(meta, index=False)
    datasets = [ResolvedDataset(a, cohort_name="example"), ResolvedDataset(b, cohort_name="example")]
    dirs = StorageDirs(preprocessing_dir=tmp_path / "pre")
    for ds in datasets:
        cohort_store(ds, dirs).write(
            "tiles", {"tiles": np.zeros((2, 2, 2, 3), np.uint8)},
            size_mm=0.25, image_id=ds.dataset().image_ids[0],
        )
    before = {root: snapshot(root) for root in (a, b)}
    monkeypatch.setattr(augment, "apply_augmentation", lambda tiles, **kwargs: tiles)
    monkeypatch.setattr(
        augment, "extract_features", lambda tiles, **kwargs: np.ones((len(tiles), 4), np.float32),
    )
    augment.prep_augment(datasets=datasets, dirs=dirs, target=["domain"], model=["virchow2"])
    definitions = list((tmp_path / "pre/augmentations").rglob("augmentation.yaml"))
    assert len(definitions) == 2
    assert {yaml.safe_load(path.read_text())["source_dataset_id"] for path in definitions} == {"a", "b"}
    assert all(snapshot(root) == before[root] for root in (a, b))
