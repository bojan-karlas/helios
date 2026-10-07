"""Component-level tests for helios.components.splits (real StratifiedKFold, no mocks)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from helios.components.splits import generate_cv_splits
from tests.fixtures import make_cohort


@pytest.fixture()
def image_metadata(tmp_path: Path) -> pd.DataFrame:
    root = make_cohort(tmp_path / "cohort", n_images=12, n_patients=6)
    return pd.read_csv(root / "metadata" / "image_metadata.csv", dtype={"image_id": str})


def test_long_format_one_test_row_per_image_per_fold(image_metadata: pd.DataFrame) -> None:
    splits = generate_cv_splits(image_metadata, n_folds=4, seed=0)
    assert set(splits.columns) == {"image_id", "fold", "split"}
    assert len(splits) == len(image_metadata) * 4

    for _image_id, group in splits.groupby("image_id"):
        assert len(group) == 4
        assert (group["split"] == "test").sum() == 1
        assert (group["split"] == "train").sum() == 3


def test_stratified_matches_plain_stratifiedkfold(image_metadata: pd.DataFrame) -> None:
    """No group_by: reproduces a plain StratifiedKFold(...).split(zeros, y) exactly."""
    splits = generate_cv_splits(
        image_metadata, n_folds=4, stratify_by=["disease_pfs_recurred"], group_by=None, seed=42
    )
    got_test_fold = (
        splits[splits["split"] == "test"].set_index("image_id")["fold"].reindex(image_metadata["image_id"])
    )

    y = image_metadata["disease_pfs_recurred"].astype(str).to_numpy()
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    expected_test_fold = np.full(len(image_metadata), -1, dtype=int)
    for fold, (_train_idx, test_idx) in enumerate(skf.split(np.zeros(len(image_metadata)), y)):
        expected_test_fold[test_idx] = fold

    np.testing.assert_array_equal(got_test_fold.to_numpy(), expected_test_fold)


def test_group_by_keeps_patient_slides_in_one_fold() -> None:
    # 3 patients x 4 slides each; group_by must keep each patient's slides together.
    rows = []
    for p in range(3):
        for s in range(4):
            rows.append({"image_id": f"P{p}-{s}", "patient_id": f"P{p}", "label": (p + s) % 2})
    meta = pd.DataFrame(rows)

    splits = generate_cv_splits(meta, n_folds=3, group_by="patient_id", seed=0)
    test_fold_by_image = splits[splits["split"] == "test"].set_index("image_id")["fold"]

    for p in range(3):
        image_ids = [f"P{p}-{s}" for s in range(4)]
        folds = test_fold_by_image.loc[image_ids].unique()
        assert len(folds) == 1, f"patient P{p}'s slides landed in multiple folds: {folds}"


def test_val_fraction_carves_out_val_without_touching_test(image_metadata: pd.DataFrame) -> None:
    splits = generate_cv_splits(image_metadata, n_folds=3, val_fraction=0.3, seed=0)
    for _fold, group in splits.groupby("fold"):
        assert set(group["image_id"]) == set(image_metadata["image_id"].astype(str))
        assert (group["split"] == "val").sum() > 0
        assert (group["split"] == "test").sum() >= 1


def test_group_by_val_fraction_carves_whole_groups() -> None:
    rows = []
    for p in range(6):
        for s in range(3):
            rows.append({"image_id": f"P{p}-{s}", "patient_id": f"P{p}", "label": p % 2})
    meta = pd.DataFrame(rows)

    splits = generate_cv_splits(meta, n_folds=3, val_fraction=0.4, group_by="patient_id", seed=0)
    for fold, group in splits.groupby("fold"):
        by_patient = group.assign(patient_id=group["image_id"].str.split("-").str[0])
        for patient_id, patient_rows in by_patient.groupby("patient_id"):
            assert patient_rows["split"].nunique() == 1, (
                f"fold {fold} patient {patient_id} split across categories: "
                f"{patient_rows['split'].unique()}"
            )


def test_empty_metadata_returns_empty_frame() -> None:
    empty = pd.DataFrame(columns=["image_id"])
    splits = generate_cv_splits(empty, n_folds=5)
    assert splits.empty
    assert list(splits.columns) == ["image_id", "fold", "split"]


def test_n_folds_clipped_to_cohort_size() -> None:
    # No stratify_by: n_splits is only bounded by sample count, not per-class counts.
    meta = pd.DataFrame({"image_id": ["A", "B", "C"]})
    splits = generate_cv_splits(meta, n_folds=10, seed=0)
    assert splits["fold"].nunique() == 3
