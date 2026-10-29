from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from helios.cli.main import app
from helios.components.augmentation import apply_augmentation
from helios.data.augmentation import Augmentation, select_augmentations, validate_feature_alignment
from helios.data.cohort import ResolvedDataset, load_cohort_spec
from helios.data.dataset import Dataset
from helios.data.resolver import StorageRoots
from helios.stages import augment, mil
from helios.stages._runtime import cohort_store
from tests.fixtures import make_cohort


def test_augmentation_dispatch_rejects_unknown_method_and_missing_checkpoint():
    tiles = np.zeros((1, 2, 2, 3), np.uint8)
    with pytest.raises(ValueError, match="Unsupported augmentation method"):
        apply_augmentation(tiles, method="unknown")
    with pytest.raises(ValueError, match="requires a checkpoint"):
        apply_augmentation(tiles)


def definition(root: Path, dataset_id: str) -> Augmentation:
    result = Augmentation(root.absolute(), dataset_id, "cyclegan", {"target": "domain_b"}, root / "model.pth")
    result.save()
    return result


def features(n: int = 3, width: int = 4) -> dict:
    return {
        "features": np.ones((n, width), np.float32),
        "tile_ids": np.arange(n, dtype=np.int64),
        "coords": np.arange(n * 2, dtype=np.int64).reshape(n, 2),
    }


@pytest.mark.parametrize("data", [{}, {"id": "a"}, {"id": "", "name": "A"}, {"id": 3, "name": "A"},
                                       {"id": "a", "name": "A", "description": 3}])
def test_invalid_dataset_descriptors(tmp_path, data):
    (tmp_path / "dataset.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        _ = Dataset(tmp_path).descriptor


def test_descriptor_survives_move_and_has_display_name(tmp_path):
    root = make_cohort(tmp_path / "original")
    descriptor = Dataset(root).descriptor
    root.rename(tmp_path / "renamed")
    assert Dataset(tmp_path / "renamed").descriptor == descriptor
    assert descriptor.name == "Example original"
    assert descriptor.description is None


def test_cli_selection_pairs_by_id_and_overrides_yaml(tmp_path):
    a = ResolvedDataset(make_cohort(tmp_path / "a"))
    b = ResolvedDataset(make_cohort(tmp_path / "b"))
    first = definition(tmp_path / "first", "a")
    second = definition(tmp_path / "second", "b")
    selected = select_augmentations([a, b], [str(second.root), str(first.root)])
    assert selected[0].augmentations == (first.root,)
    assert selected[1].augmentations == (second.root,)
    assert select_augmentations(selected, []) == [a, b]
    with pytest.raises(ValueError, match="matches 0"):
        select_augmentations([a], [str(second.root)])
    with pytest.raises(ValueError, match="matches 2"):
        select_augmentations([a, a], [str(first.root)])
    with pytest.raises(ValueError, match="different dataset"):
        select_augmentations([replace(a, augmentations=(second.root,))], None)


def test_cohort_paths_and_provenance_round_trip(tmp_path):
    root = make_cohort(tmp_path / "source")
    prepared = definition(tmp_path / "elsewhere", root.name)
    cohort_file = tmp_path / "train.yaml"
    cohort_file.write_text(yaml.safe_dump({
        "datasets": [{"source_root": "source", "augmentations": ["elsewhere"]}],
    }))
    selected = load_cohort_spec(cohort_file).datasets
    assert selected[0].augmentations == (prepared.root,)
    from helios.stages._runtime import build_cohort_provenance
    cohort_file.write_text(yaml.safe_dump(build_cohort_provenance(selected)))
    assert load_cohort_spec(cohort_file).datasets[0].augmentations == (prepared.root,)


@pytest.mark.parametrize("change", ["count", "width"])
def test_alignment_rejects_incompatible_features(tmp_path, change):
    prepared = definition(tmp_path / "aug", "a")
    store = prepared.store()
    keys = {"size_mm": 0.25, "model": "virchow2", "image_id": "slide"}
    original = tmp_path / "original.hdf5"
    from helios.data.io import write
    write(features(), original, "hdf5")
    payload = features()
    if change == "count":
        payload = features(2)
    elif change == "width":
        payload = features(width=5)
    store.write("tile_augmentation_features", payload, **keys)
    with pytest.raises(ValueError):
        validate_feature_alignment(original, store.path("tile_augmentation_features", **keys))


@pytest.mark.parametrize("key", ["features", "tile_224", "tiles_224"])
def test_alignment_uses_array_shapes_and_ignores_redundant_legacy_indices(tmp_path, key):
    from helios.data.io import write

    original = write({"features": np.ones((3, 4))}, tmp_path / "base.hdf5", "hdf5")
    augmented = write({
        key: np.ones((3, 4)), "tile_ids": np.zeros(3), "coords": np.zeros((1, 1)),
    }, tmp_path / "aug.hdf5", "hdf5")
    validate_feature_alignment(original, augmented)


def test_prep_fit_discovery_and_saved_selection(monkeypatch, tmp_path):
    root = make_cohort(tmp_path / "source", n_images=2)
    ds = ResolvedDataset(root)
    store = cohort_store(ds)
    for image_id in ds.dataset().image_ids:
        keys = {"size_mm": 0.25, "image_id": image_id}
        store.write("tiles", {"tiles": np.zeros((3, 2, 2, 3), np.uint8)}, **keys)
        store.write("tile_features", {
            "features": np.ones((3, 4), np.float32), "tile_ids": np.arange(3),
            "coords": np.empty((3, 0), np.int64),
        }, model="virchow2", **keys)
    calls = []
    monkeypatch.setattr("helios.components.augmentation.augment_stains", lambda tiles, **kw: calls.append(kw) or tiles)
    monkeypatch.setattr(augment, "extract_features", lambda tiles, **kw: np.ones((len(tiles), 4), np.float32))
    output = tmp_path / "arbitrary-name"
    for _ in range(2):
        augment.prep_augment(
            datasets=[root], target=["domain_b"], checkpoint=["weights.pth"], output_dir=str(output),
        )
    assert len(calls) == 2
    prepared = Augmentation.load(output)
    assert prepared.source_dataset_id == root.name
    seen = []
    monkeypatch.setattr(mil, "train_mil", lambda slides, **kw: seen.extend(slides) or {})
    mil.fit_mil(datasets=[root], augmentations=[str(output)])
    assert len(seen) == 2
    assert all(len(s.aug_features_paths) == 1 for s in seen)
    saved = store.path("model_mil", fold=None) / "cohort.yaml"
    assert load_cohort_spec(saved).datasets[0].augmentations == (output,)
    result = CliRunner().invoke(app, ["list", "augmentations", "--root", str(output)])
    assert result.exit_code == 0, result.output
    row, = json.loads(result.output)
    assert row["path"] == str(output)
    assert row["tile_files"] == 2
    assert row["feature_files"] == {"features/size_mm=0.25/model=virchow2": 2}
    for path in (output / "tiles").rglob("*.hdf5"):
        path.unlink()
    mil.fit_mil(datasets=[root], augmentations=[str(output)], overwrite=True)
    assert len(seen) == 4


def test_missing_features_fail_without_generation(monkeypatch, tmp_path):
    root = make_cohort(tmp_path / "source", n_images=1)
    ds = ResolvedDataset(root)
    store = cohort_store(ds)
    image_id = ds.dataset().image_ids[0]
    store.write("tile_features", features(), size_mm=0.25, model="virchow2", image_id=image_id)
    prepared = definition(tmp_path / "aug", root.name)
    monkeypatch.setattr(mil, "train_mil", lambda *a, **k: pytest.fail("must not train"))
    with pytest.raises(FileNotFoundError, match="Missing tile_augmentation_features"):
        mil.fit_mil(datasets=[root], augmentations=[str(prepared.root)])
    assert not (prepared.root / "features").exists()


def test_cli_selection_disable_and_removed_options(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(mil, "fit_mil", lambda **kwargs: seen.append(kwargs))
    monkeypatch.setattr("helios.cli._common.console_progress", lambda *a, **k: None)
    runner = CliRunner()
    for args, expected in [
        (["--augmentation", "a", "--augmentation", "b"], ["a", "b"]),
        (["--no-augmentations"], []),
        ([], None),
    ]:
        result = runner.invoke(app, ["fit", "mil", "--dataset", str(tmp_path), *args])
        assert result.exit_code == 0, result.output
        assert seen[-1]["augmentations"] == expected
    result = runner.invoke(app, [
        "fit", "mil", "--dataset", str(tmp_path), "--augmentation", "a", "--no-augmentations",
    ])
    assert result.exit_code == 2
    result = runner.invoke(app, ["fit", "mil", "--dataset", str(tmp_path), "--features-root", "/old"])
    assert result.exit_code == 2
    assert "--features-root has been removed" in result.output
    result = runner.invoke(app, ["fit", "mil", "--dataset", str(tmp_path), "--augmentation-target", "old"])
    assert result.exit_code == 2
    assert "--augmentation-target has been" in result.output
    assert "removed. Use --augmentation PATH" in result.output


def test_legacy_config_rejected(tmp_path):
    from helios.config.stage_config import load_stage_params
    path = tmp_path / "old.yaml"
    path.write_text("fit:\n  mil:\n    augmentation_target: old\n")
    with pytest.raises(ValueError, match="augmentation_target has been removed"):
        load_stage_params("fit", "mil", config_path=path)


@pytest.mark.parametrize("mode", ["replace", "augment"])
def test_pooled_fit_pairs_multiple_augmentations_and_preserves_sampling(monkeypatch, tmp_path, mode):
    from helios.components.mil import _load_training_bag

    datasets = [ResolvedDataset(make_cohort(tmp_path / name, n_images=1)) for name in ("a", "b")]
    definitions = [
        definition(tmp_path / "aug_a1", "a"),
        definition(tmp_path / "aug_b", "b"),
        definition(tmp_path / "aug_a2", "a"),
    ]
    for ds in datasets:
        image_id = ds.dataset().image_ids[0]
        cohort_store(ds).write(
            "tile_features", features(20), size_mm=0.25, model="virchow2", image_id=image_id,
        )
        for index, prepared in enumerate(definitions):
            if prepared.source_dataset_id != ds.dataset().descriptor.id:
                continue
            payload = features(20)
            payload["features"][:] = index + 2
            prepared.store().write(
                "tile_augmentation_features", payload, size_mm=0.25, model="virchow2", image_id=image_id,
            )
    seen = []

    def fake_train(slides, **kwargs):
        seen.extend(slides)
        assert kwargs["aug_mode"] == mode
        for slide in slides:
            owner = next(ds.dataset().descriptor.id for ds in datasets if slide.features_path.is_relative_to(ds.root))
            expected = 2 if owner == "a" else 1
            assert len(slide.aug_features_paths) == expected
            bag = _load_training_bag(
                slide, aug_swap_prob=1.0, aug_mode=mode, rng=np.random.default_rng(42),
            )
            if mode == "replace":
                assert bag.shape == (20, 4)
                assert set(np.unique(bag)) == ({2, 4} if owner == "a" else {3})
            else:
                assert bag.shape == (20 * (1 + expected), 4)
        return {}

    monkeypatch.setattr(mil, "train_mil", fake_train)
    roots = StorageRoots(models_root=tmp_path / "models")
    mil.fit_mil(
        datasets=datasets, roots=roots, augmentations=[str(a.root) for a in definitions], aug_mode=mode,
    )
    assert len(seen) == 2
    saved = cohort_store(datasets[0], roots).path("model_mil", fold=None) / "cohort.yaml"
    restored = load_cohort_spec(saved).datasets
    assert [len(ds.augmentations) for ds in restored] == [2, 1]


def test_resume_definition_and_warn_on_changed_checkpoint(monkeypatch, tmp_path, caplog):
    from helios.progress import Progress

    root = make_cohort(tmp_path / "source", n_images=1)
    ds = ResolvedDataset(root)
    image_id = ds.dataset().image_ids[0]
    cohort_store(ds).write("tiles", {"tiles": np.zeros((3, 2, 2, 3), np.uint8)}, size_mm=0.25, image_id=image_id)
    calls = []
    monkeypatch.setattr("helios.components.augmentation.augment_stains", lambda tiles, **kw: calls.append(kw) or tiles)
    monkeypatch.setattr(augment, "extract_features", lambda tiles, **kw: np.ones((len(tiles), 4), np.float32))
    output = tmp_path / "prepared"
    augment.prep_augment(datasets=[root], output_dir=str(output), target=["domain_b"], checkpoint=["first.pth"])
    augment.prep_augment(
        datasets=[root], output_dir=str(output), target=["domain_b"], checkpoint=["second.pth"],
        progress=Progress(use_tqdm=False),
    )
    assert len(calls) == 1
    assert "Mixed outputs" in caplog.text
    for path in (output / "features").rglob("*.hdf5"):
        path.unlink()
    augment.prep_augment(datasets=[replace(ds, augmentations=(output,))])
    assert len(calls) == 1
    augment.prep_augment(datasets=[replace(ds, augmentations=(output,))], overwrite=True)
    assert len(calls) == 2
    assert calls[-1]["checkpoint"].endswith("second.pth")


def test_yaml_can_disable_unavailable_augmentations(tmp_path):
    ds = ResolvedDataset(make_cohort(tmp_path / "source"), augmentations=(tmp_path / "absent",))
    assert select_augmentations([ds], [])[0].augmentations == ()
    with pytest.raises(FileNotFoundError):
        select_augmentations([ds], None)


def test_prep_augment_without_selections_warns_instead_of_silently_doing_nothing(tmp_path, caplog):
    import logging

    from helios.progress import Progress

    root = make_cohort(tmp_path / "source", n_images=1)
    with caplog.at_level(logging.WARNING, logger="helios"):
        augment.prep_augment(datasets=[root], progress=Progress(use_tqdm=False))
    assert "no augmentations selected" in caplog.text
    assert not (root / "preprocessing" / "augmentations").exists()


def test_unprepared_directory_reports_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a prepared augmentation directory"):
        Augmentation.load(tmp_path / "absent")
