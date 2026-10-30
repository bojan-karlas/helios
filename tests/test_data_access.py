"""Path-resolution + IO-access tests for the data layer seams.

Covers independent storage roots and lazy HDF5 access.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from helios.data.catalog import load_artifacts, storage_role
from helios.data.cohort import ResolvedDataset
from helios.data.dataset import Dataset
from helios.data.io import ArtifactStore
from helios.data.resolver import Resolver, Roots, StorageRoots
from helios.stages._runtime import cohort_store, resolve_datasets


@pytest.mark.parametrize("suffix", [
    ".svs", ".tif", ".tiff", ".ndpi", ".mrxs",
    ".avs", ".dcm", ".vms", ".vmu", ".scn", ".svslide", ".bif", ".czi",
])
@pytest.mark.parametrize("uppercase", [False, True])
def test_slide_discovery_formats(tmp_path, suffix, uppercase):
    folder = tmp_path / "wsi/group"
    folder.mkdir(parents=True)
    slide = folder / f"slide{suffix.upper() if uppercase else suffix}"
    slide.write_bytes(b"synthetic slide")
    (folder / f"directory{suffix}").mkdir()
    assert Dataset(tmp_path).slides == {"slide": slide}


@pytest.mark.parametrize("suffix", [".mrxs", ".MRXS"])
def test_recursive_mrxs_discovery_ignores_companion_files_and_directories(tmp_path, suffix):
    folder = tmp_path / "wsi/group"
    folder.mkdir(parents=True)
    slide = folder / f"slide{suffix}"
    slide.write_bytes(b"synthetic MRXS header")
    companion = folder / "slide"
    companion.mkdir()
    (companion / "Data0000.dat").write_bytes(b"synthetic data")
    (companion / "Slidedat.ini").write_text("synthetic configuration")
    (folder / "not-a-slide.mrxs").mkdir()
    assert Dataset(tmp_path).slides == {"slide": slide}


def test_independent_roots_keep_source_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "cohort"
    output = tmp_path / "preprocessing"
    store = ArtifactStore(Resolver(Roots(source_root=source, preprocessing_root=output)))

    store.write("tile_features", {"features": np.zeros((2, 4))}, size_mm=0.25, model="virchow2", image_id="I1")

    # Derived artifact lands under the output root, not the source cohort.
    assert (
        output / "tile_features/size_mm=0.25/model=virchow2/I1/I1.hdf5"
    ).exists()
    assert not (source / "preprocessing").exists()
    assert store.exists("tile_features", size_mm=0.25, model="virchow2", image_id="I1")


def test_role_defaults_are_in_place(tmp_path: Path) -> None:
    roots = Roots.single(tmp_path)
    assert roots.for_role("source") == tmp_path
    assert roots.for_role("preprocessing") == tmp_path / "preprocessing"
    assert roots.for_role("models") == tmp_path / "models"
    assert roots.for_role("results") == tmp_path / "results"
    with pytest.raises(ValueError, match="storage role"):
        storage_role("features")


def test_all_artifacts_resolve_under_their_declared_roots(tmp_path: Path) -> None:
    roots = Roots(
        source_root=tmp_path / "source",
        preprocessing_root=tmp_path / "pre",
        models_root=tmp_path / "models",
        results_root=tmp_path / "results",
    )
    resolver = Resolver(roots)
    keys = dict(image_id="I1", tile_id=0, cell_id=0, fold=1, model="uni", target="lab", size_mm=0.25, asset_id="a")
    for artifact in load_artifacts().values():
        assert resolver.path(artifact.id, **keys).is_relative_to(roots.for_role(artifact.role))
    assert resolver.role("splits") == "preprocessing"
    assert resolver.role("tile_augmentation") == "preprocessing"
    assert resolver.role("tile_features") == "preprocessing"
    assert resolver.role("tile_augmentation_features") == "preprocessing"
    assert resolver.path(
        "tile_augmentation", size_mm=0.25, image_id="I1"
    ) == tmp_path / "pre/tiles/size_mm=0.25/I1/I1.hdf5"
    assert resolver.path(
        "tile_augmentation_features", size_mm=0.25, model="uni", image_id="I1"
    ) == tmp_path / "pre/tile_features/size_mm=0.25/model=uni/I1/I1.hdf5"
    assert resolver.role("aligned_slide_embedding") == "results"
    assert resolver.role("cellular_features") == "preprocessing"
    assert resolver.role("model_mil") == "models"
    with pytest.raises(ValueError, match="storage role"):
        storage_role("work")


def test_constituent_roots_and_single_dataset_overrides(tmp_path: Path) -> None:
    ds = ResolvedDataset(tmp_path / "A", preprocessing_root=tmp_path / "pre-A")
    defaults = cohort_store(ds).resolver.roots
    assert defaults.for_role("preprocessing") == ds.preprocessing_root
    overrides = StorageRoots(preprocessing_root=tmp_path / "pre")
    configured = cohort_store(ds, overrides).resolver.roots
    assert configured.for_role("source") == ds.root
    assert configured.for_role("preprocessing") == overrides.preprocessing_root


def test_results_location_is_independent_of_union_membership(tmp_path: Path) -> None:
    a, b = ResolvedDataset(tmp_path / "A"), ResolvedDataset(tmp_path / "B")
    roots = StorageRoots(models_root=tmp_path / "run", results_root=tmp_path / "run")
    union = resolve_datasets([a, b], roots=roots)
    single = resolve_datasets([a], roots=roots)
    store = cohort_store(union[0], roots)
    assert store.path("whole_image_risk_score_ensemble") == cohort_store(single[0], roots).path(
        "whole_image_risk_score_ensemble"
    )
    assert (
        store.path("whole_image_risk_score_ensemble")
        == tmp_path / "run/A/mil/risk_score/whole_image_risk_score.csv"
    )
    assert store.path("model_mil", fold=None) == cohort_store(union[1], roots).path("model_mil", fold=None)


def test_ambiguous_union_roots_fail_before_io(tmp_path: Path) -> None:
    datasets = [tmp_path / "A", tmp_path / "B"]
    with pytest.raises(ValueError, match="Pooled training"):
        resolve_datasets(datasets, require_models=True)
    with pytest.raises(ValueError, match="per dataset"):
        resolve_datasets(datasets, roots=StorageRoots(preprocessing_root=tmp_path))
    with pytest.raises(ValueError, match="distinct"):
        resolve_datasets([tmp_path / "one/A", tmp_path / "two/A"], roots=StorageRoots(results_root=tmp_path))
    assert not (tmp_path / "A").exists()


def test_path_enables_lazy_hdf5_subset_read(tmp_path: Path) -> None:
    store = ArtifactStore(Resolver(Roots.single(tmp_path)))
    n_tiles = 1000
    tiles = np.arange(n_tiles * 2 * 2 * 3, dtype=np.uint8).reshape(n_tiles, 2, 2, 3)
    store.write("tiles", {"tiles": tiles}, size_mm=0.25, image_id="I1")

    # The streaming seam: get the path, open lazily, read only a slice.
    path = store.path("tiles", size_mm=0.25, image_id="I1")
    assert isinstance(path, Path) and path.exists()

    with h5py.File(path, "r") as h:
        dset = h["tiles"]
        assert dset.shape == (n_tiles,)  # metadata only; not materialized
        assert dset.dtype.subdtype == (np.dtype(np.uint8), (2, 2, 3))
        subset = dset[10:13]  # read just 3 rows
    assert subset.shape == (3, 2, 2, 3)
    np.testing.assert_array_equal(subset, tiles[10:13])


def test_target_does_not_supply_implicit_method() -> None:
    from helios.data.resolver import render_path

    with pytest.raises(KeyError, match="Missing key values"):
        render_path("method={method}/target={target}", {"target": "lab"})


def test_scan_hive_parses_key_value_dirs_and_skips_parts(tmp_path: Path) -> None:
    from helios.data.io import write
    from helios.data.resolver import scan_hive

    f1 = tmp_path / "size_mm=0.25" / "model=virchow2" / "scores.csv"
    f2 = tmp_path / "size_mm=0.125" / "model=uni" / "scores.csv"
    part = tmp_path / "size_mm=0.25" / "model=virchow2" / "scores.csv.parts" / "I1.csv"
    write([{"image_id": "I1", "score": 0.8}], f1, "csv")
    write([{"image_id": "I2", "score": 0.4}], f2, "csv")
    write([{"image_id": "I1", "score": 0.8}], part, "csv")

    df = scan_hive(tmp_path, "scores.csv")
    assert len(df) == 2
    assert set(df["image_id"]) == {"I1", "I2"}
    assert set(df["size_mm"]) == {0.25, 0.125}
    assert set(df["model"]) == {"virchow2", "uni"}


def test_union_cohort_rejects_duplicate_image_ids_and_augments_per_dataset(tmp_path: Path) -> None:
    import yaml

    from helios.data.cohort import load_cohort_spec
    from tests.fixtures import make_cohort

    c1 = make_cohort(tmp_path / "COHORT1", n_images=2, n_patients=2)
    c2 = make_cohort(tmp_path / "COHORT2", n_images=2, n_patients=2)
    manifest = tmp_path / "union.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": "TRAIN_COHORT",
                "datasets": [
                    {
                        "source_root": str(c1),
                        "augmentations": ["prepared"],
                    },
                    {"source_root": str(c2)},
                ],
            }
        )
    )
    spec = load_cohort_spec(manifest)
    assert spec.datasets[0].augmentations == (tmp_path / "prepared",)
    assert spec.splits_file == tmp_path / "union.splits.csv"
    with pytest.raises(ValueError, match="duplicate image_ids"):
        resolve_datasets([manifest], roots=StorageRoots(models_root=tmp_path / "models"))


def test_atomic_writes_parts_bundle_provenance_and_run_logging(tmp_path: Path) -> None:
    import yaml

    from helios.data.bundle import COHORT_NAME, CONFIG_NAME, MANIFEST_NAME, is_complete_bundle
    from helios.progress import Progress, ProgressReporter, create_run_log_dir, normalize_run_tag
    from helios.stages._ray import ParallelConfig, StageExecutionError, run_work_units

    store = ArtifactStore(Resolver(Roots.single(tmp_path)))
    store.write_part("whole_image_risk_score", "I1", [{"image_id": "I1", "risk_score": 0.25, "fold": 0}], fold=0)
    store.write_part("whole_image_risk_score", "I2", [{"image_id": "I2", "risk_score": 0.75, "fold": 0}], fold=0)
    assert store.part_exists("whole_image_risk_score", "I1", fold=0)
    reduced = store.read_parts("whole_image_risk_score", ["I1", "I2"], fold=0)
    assert reduced["image_id"].tolist() == ["I1", "I2"]

    bundle_meta = {
        "config": {"fit": {"mil": {"size_mm": 0.25, "model": "virchow2"}}},
        "cohort": {"name": "TRAIN_COHORT", "datasets": [{"source_root": "/data/COHORT1"}]},
    }
    bdir = store.write("model_mil", {"weights": [1, 2, 3]}, metadata=bundle_meta, fold=0)
    assert is_complete_bundle(bdir)
    assert yaml.safe_load((bdir / CONFIG_NAME).read_text()) == bundle_meta["config"]
    assert yaml.safe_load((bdir / COHORT_NAME).read_text()) == bundle_meta["cohort"]
    manifest = yaml.safe_load((bdir / MANIFEST_NAME).read_text())
    assert set(manifest["files"]) == {"model.pkl", CONFIG_NAME, COHORT_NAME}

    assert normalize_run_tag("Baseline Run_v1!") == "baseline-run-v1"
    run_id, run_dir = create_run_log_dir(tmp_path / "logs", "Baseline Run_v1!")
    assert run_id.endswith("-utc-baseline-run-v1")
    assert (run_dir / "workers").is_dir()

    reporter = ProgressReporter(run_dir)
    progress = Progress.console(reporter=reporter)
    progress.use_tqdm = False

    def _work(item: str, ctx) -> str:
        ctx.progress.log(f"working on {item}")
        if item == "bad":
            raise ValueError("corrupt slide")
        return "skipped" if item == "ok2" else "completed"

    assert not ParallelConfig().enabled
    with pytest.raises(StageExecutionError) as exc_info:
        run_work_units(
            ["ok1", "bad", "ok2"],
            _work,
            stage="tiles COHORT1",
            progress=progress,
            parallel=ParallelConfig(),
        )
    (failure,) = exc_info.value.failures
    assert failure.work_id == "bad"
    assert failure.error == "ValueError: corrupt slide"
    assert failure.log_path is not None and Path(failure.log_path).is_file()
    assert "corrupt slide" in Path(failure.log_path).read_text()
    status = yaml.safe_load(reporter.status_path.read_text())
    assert status["counts"] == {
        "total": 3,
        "queued": 0,
        "running": 0,
        "completed": 1,
        "skipped": 1,
        "failed": 1,
    }
