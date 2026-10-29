"""Split destinations, selection, and overwrite policy."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from helios.cli.main import app
from helios.data.cohort import ResolvedDataset, load_cohort_spec, resolve_datasets
from helios.data.resolver import StorageRoots
from helios.stages import mil
from helios.stages._runtime import (
    build_cohort_provenance,
    cohort_store,
    folds_of,
    read_splits,
    resolve_splits_path,
    train_ids,
)
from helios.stages.splits import prep_splits
from tests.fixtures import make_cohort


def cohort_yaml(path: Path, root: Path, *, mapped: bool, **extra) -> ResolvedDataset:
    entry = {"source_root": str(root)} if mapped else str(root)
    path.write_text(yaml.safe_dump({"datasets": [entry], **extra}))
    return resolve_datasets([path])[0]


@pytest.mark.parametrize("mapped", [False, True])
def test_yaml_default_is_adjacent_and_independent_of_existing_files(tmp_path, mapped):
    root = make_cohort(tmp_path / "source")
    path = tmp_path / "experiment.yaml"
    ds = cohort_yaml(path, root, mapped=mapped)
    expected = tmp_path / "experiment.splits.csv"
    assert resolve_splits_path(ds) == expected
    store = cohort_store(ds)
    store.write("splits", pd.DataFrame({"image_id": ["unused"], "fold": [7], "split": ["train"]}))
    assert read_splits(ds) is None
    assert folds_of(ds) == [None]
    pd.DataFrame({"image_id": ["unused"], "fold": [3], "split": ["train"]}).to_csv(expected, index=False)
    assert folds_of(cohort_yaml(path, root, mapped=mapped)) == [3]
    expected.unlink()
    assert resolve_splits_path(cohort_yaml(path, root, mapped=mapped)) == expected


def test_bare_default_and_explicit_paths_are_exact(tmp_path):
    ds = ResolvedDataset(make_cohort(tmp_path / "source", n_images=12, n_patients=12))
    roots = StorageRoots(preprocessing_root=tmp_path / "pre")
    assert resolve_splits_path(ds, roots) == tmp_path / "pre/splits.csv"
    prep_splits(datasets=[ds], roots=roots, n_folds=3, seed=19, stratify_by=[])
    assert folds_of(ds, roots) == [0, 1, 2]
    assert list((tmp_path / "pre").rglob("*.csv")) == [tmp_path / "pre/splits.csv"]
    explicit = StorageRoots(splits_file=tmp_path / "chosen/custom.csv")
    prep_splits(datasets=[ds], roots=explicit, n_folds=4, seed=7, stratify_by=[])
    assert explicit.splits_file.is_file()
    assert folds_of(ds, explicit) == [0, 1, 2, 3]
    assert resolve_splits_path(ds, explicit) == explicit.splits_file


@pytest.mark.parametrize("as_store", [False, True])
def test_missing_explicit_cli_path_never_falls_back(tmp_path, as_store):
    ds = ResolvedDataset(make_cohort(tmp_path / "source"))
    store = cohort_store(ds)
    store.write("splits", pd.DataFrame({"image_id": ["slide"], "fold": [1], "split": ["train"]}))
    selected = store if as_store else ds
    roots = StorageRoots(splits_file=tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="Selected splits file"):
        read_splits(selected, roots)
    with pytest.raises(FileNotFoundError):
        folds_of(selected, roots)


def test_missing_yaml_explicit_path_errors_and_cli_overrides_it(tmp_path):
    root = make_cohort(tmp_path / "source")
    ds = cohort_yaml(tmp_path / "cohort.yaml", root, mapped=True, splits="missing.csv")
    cohort_store(ds).write("splits", pd.DataFrame({"fold": [0]}))
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        read_splits(ds)
    override = tmp_path / "override.csv"
    pd.DataFrame({"fold": [4]}).to_csv(override, index=False)
    assert folds_of(ds, StorageRoots(splits_file=override)) == [4]


@pytest.mark.parametrize("yaml_input", [False, True])
def test_no_default_splits_fits_entire_filtered_cohort(monkeypatch, tmp_path, yaml_input):
    import numpy as np

    root = make_cohort(tmp_path / "source", n_images=4)
    ds = (
        cohort_yaml(tmp_path / "cohort.yaml", root, mapped=True, filter="path_stage != 'IV'")
        if yaml_input else ResolvedDataset(root, where="path_stage != 'IV'")
    )
    store = cohort_store(ds)
    for image_id in ds.dataset().image_ids:
        store.write(
            "tile_features", {"features": np.ones((3, 4), np.float32)},
            size_mm=0.25, model="virchow2", image_id=image_id,
        )
    seen = []
    monkeypatch.setattr(mil, "train_mil", lambda slides, **kw: seen.append((slides, kw["fold"])) or {})
    assert folds_of(ds) == [None]
    assert train_ids(ds, None) == ds.dataset().image_ids[:3]
    mil.fit_mil(datasets=[ds])
    assert len(seen) == 1
    assert seen[0][1] is None
    assert [s.image_id for s in seen[0][0]] == ds.dataset().image_ids[:3]


def test_existing_splits_skip_with_warning_and_overwrite_replaces(tmp_path):
    import logging

    from helios.progress import Progress

    root = make_cohort(tmp_path / "source", n_images=12, n_patients=12)
    ds = ResolvedDataset(root)
    prep_splits(datasets=[ds], n_folds=3, stratify_by=[])
    path = resolve_splits_path(ds)
    original = path.read_bytes()
    warnings = []
    progress = Progress(use_tqdm=False)
    progress.log = lambda message, level=logging.INFO: warnings.append((level, message))  # type: ignore[method-assign]
    prep_splits(datasets=[ds], n_folds=4, stratify_by=[], progress=progress)
    assert path.read_bytes() == original
    assert [level for level, _ in warnings] == [logging.WARNING]
    assert "--overwrite" in warnings[0][1]
    prep_splits(datasets=[ds], n_folds=4, stratify_by=[], overwrite=True)
    assert folds_of(ds) == [0, 1, 2, 3]


def test_existing_destination_does_not_block_other_cohorts(tmp_path):
    a = make_cohort(tmp_path / "a", n_images=6)
    b = make_cohort(tmp_path / "b", n_images=6)
    b_path = resolve_splits_path(ResolvedDataset(b))
    b_path.parent.mkdir(parents=True)
    b_path.write_text("existing output")
    prep_splits(datasets=[a, b], n_folds=2, stratify_by=[])
    assert resolve_splits_path(ResolvedDataset(a)).is_file()
    assert b_path.read_text() == "existing output"


def test_duplicate_destinations_rejected_before_writing(tmp_path):
    a = make_cohort(tmp_path / "a")
    b = make_cohort(tmp_path / "b")
    roots = StorageRoots(splits_file=tmp_path / "shared.csv")
    with pytest.raises(ValueError, match="same destination"):
        prep_splits(datasets=[a, b], roots=roots, overwrite=True)
    assert not roots.splits_file.exists()


def test_yaml_prep_and_fit_select_same_file_and_provenance(tmp_path):
    root = make_cohort(tmp_path / "source", n_images=12, n_patients=12)
    yaml_file = tmp_path / "cohort.yaml"
    ds = cohort_yaml(yaml_file, root, mapped=False)
    prep_splits(datasets=[yaml_file], n_folds=3, stratify_by=[])
    assert folds_of(ds) == [0, 1, 2]
    doc = build_cohort_provenance([ds])
    assert doc["splits"] == str(tmp_path / "cohort.splits.csv")
    yaml_file.write_text(yaml.safe_dump(doc))
    assert folds_of(load_cohort_spec(yaml_file).datasets[0]) == [0, 1, 2]


def test_cli_splits_overwrite_and_legacy_flag_rejection(tmp_path):
    root = make_cohort(tmp_path / "source", n_images=12, n_patients=12)
    runner = CliRunner()
    command = [
        "prep", "splits", "--dataset", str(root), "--n-folds", "3",
        "--stratify-by", "disease_pfs_recurred",
    ]
    assert runner.invoke(app, command).exit_code == 0
    path = resolve_splits_path(ResolvedDataset(root))
    original = path.read_bytes()
    assert runner.invoke(app, command).exit_code == 0
    assert path.read_bytes() == original
    assert runner.invoke(app, command + ["--overwrite"]).exit_code == 0
    assert runner.invoke(app, command + ["--force"]).exit_code == 2


def test_every_output_command_exposes_overwrite_not_force():
    from helios.stages import STAGES

    runner = CliRunner()
    for verb, stages in STAGES.items():
        for noun in stages:
            command = [verb] if verb == "report" else [verb, noun]
            result = runner.invoke(app, command + ["--help"])
            assert result.exit_code == 0
            assert "overwrite" in result.output
            assert "--force" not in result.output
    assert "overwrite" in runner.invoke(app, ["prep", "--help"]).output
