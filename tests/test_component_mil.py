"""Component-level tests for helios.components.mil (real training/inference, no mocks).

Builds tiny synthetic tile-feature / tile-background artifacts directly (the same
on-disk shapes helios.data.io writes: an hdf5 "features" dataset and a parquet
with tile_id/is_background columns) and exercises train_mil/infer_mil/
reduce_whole_image_risk end to end on CPU.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from helios.components.mil import (
    SlideFeatureRef,
    _load_training_bag,
    infer_mil,
    reduce_whole_image_risk,
    train_mil,
)

D = 8  # tiny embedding dim for fast tests


def _write_features(path: Path, feats: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.create_dataset("features", data=feats.astype(np.float32))


def _write_background(path: Path, is_background: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"tile_id": np.arange(len(is_background)), "is_background": is_background}
    ).to_parquet(path, index=False)


def _make_slide(
    tmp_path: Path, image_id: str, *, n_tiles: int, label: int, n_background: int = 0, seed: int = 0
) -> SlideFeatureRef:
    rng = np.random.default_rng(seed)
    feats = rng.normal(loc=float(label), scale=0.1, size=(n_tiles, D))
    features_path = tmp_path / f"{image_id}_features.hdf5"
    _write_features(features_path, feats)

    is_bg = np.zeros(n_tiles, dtype=bool)
    is_bg[:n_background] = True
    background_path = tmp_path / f"{image_id}_background.parquet"
    _write_background(background_path, is_bg)

    return SlideFeatureRef(
        image_id=image_id, features_path=features_path, background_path=background_path, label=label
    )


@pytest.fixture()
def slides(tmp_path: Path) -> list[SlideFeatureRef]:
    return [
        _make_slide(tmp_path, f"S{i:02d}", n_tiles=12, label=i % 2, seed=i)
        for i in range(6)
    ]


def test_train_and_infer_roundtrip(slides: list[SlideFeatureRef]) -> None:
    model = train_mil(slides, max_epochs=2, val_frac=0.0, device="cpu")

    held_out = slides[0]
    out = infer_mil(
        model, features_path=held_out.features_path, background_path=held_out.background_path, device="cpu"
    )

    assert out.patch_embeddings.shape == (12, D)
    assert list(out.attention.columns) == ["tile_id", "attention", "tile_gradient"]
    assert len(out.attention) == 12
    assert out.slide_embedding.shape == (128,)  # fixed penultimate-bottleneck width
    assert out.slide_embedding_grad.shape == (128,)
    assert 0.0 <= out.risk_score <= 1.0


def test_train_with_internal_val_split_early_stopping(slides: list[SlideFeatureRef]) -> None:
    # 6 slides, val_frac=0.34 -> internal split carves out a val set; must not crash
    # and must still return a usable model.
    model = train_mil(slides, max_epochs=3, patience=1, val_frac=0.34, device="cpu")
    out = infer_mil(model, features_path=slides[0].features_path, background_path=slides[0].background_path)
    assert 0.0 <= out.risk_score <= 1.0


def test_background_tiles_are_excluded(tmp_path: Path, slides: list[SlideFeatureRef]) -> None:
    slide = _make_slide(tmp_path, "BG01", n_tiles=10, label=1, n_background=4, seed=99)
    model = train_mil(slides, max_epochs=1, val_frac=0.0, device="cpu")
    out = infer_mil(model, features_path=slide.features_path, background_path=slide.background_path)
    assert len(out.attention) == 6  # 10 tiles - 4 background


def test_aug_mode_replace_swaps_in_place(tmp_path: Path) -> None:
    n = 20
    feats = np.zeros((n, D), dtype=np.float32)
    aug = np.ones((n, D), dtype=np.float32)
    features_path = tmp_path / "orig.hdf5"
    aug_path = tmp_path / "aug.hdf5"
    _write_features(features_path, feats)
    _write_features(aug_path, aug)

    ref = SlideFeatureRef(
        image_id="A", features_path=features_path, background_path=None, aug_features_path=aug_path, label=1
    )
    bag = _load_training_bag(
        ref, aug_swap_prob=1.0, aug_mode="replace", rng=np.random.default_rng(0)
    )
    assert bag.shape == (n, D)
    assert np.allclose(bag, 1.0)  # fully swapped to augmented values


def test_aug_mode_augment_appends_rows(tmp_path: Path) -> None:
    n = 20
    feats = np.zeros((n, D), dtype=np.float32)
    aug = np.ones((n, D), dtype=np.float32)
    features_path = tmp_path / "orig.hdf5"
    aug_path = tmp_path / "aug.hdf5"
    _write_features(features_path, feats)
    _write_features(aug_path, aug)

    ref = SlideFeatureRef(
        image_id="A", features_path=features_path, background_path=None, aug_features_path=aug_path, label=1
    )
    bag = _load_training_bag(
        ref, aug_swap_prob=1.0, aug_mode="augment", rng=np.random.default_rng(0)
    )
    assert bag.shape == (2 * n, D)
    assert np.allclose(bag[:n], 0.0)
    assert np.allclose(bag[n:], 1.0)


def test_reduce_whole_image_risk_cv_takes_test_fold_score() -> None:
    per_fold_scores = pd.DataFrame(
        {
            "image_id": ["A", "A", "B", "B"],
            "fold": [0, 1, 0, 1],
            "risk_score": [0.1, 0.9, 0.2, 0.8],
        }
    )
    cv_splits = pd.DataFrame(
        {
            "image_id": ["A", "A", "B", "B"],
            "fold": [0, 1, 0, 1],
            "split": ["train", "test", "test", "train"],
        }
    )
    reduced = reduce_whole_image_risk(per_fold_scores, cv_splits)
    reduced = reduced.set_index("image_id")
    assert reduced.loc["A", "risk_score"] == pytest.approx(0.9)
    assert reduced.loc["A", "fold"] == 1
    assert reduced.loc["B", "risk_score"] == pytest.approx(0.2)
    assert reduced.loc["B", "fold"] == 0


def test_reduce_whole_image_risk_deploy_ensembles() -> None:
    per_fold_scores = pd.DataFrame(
        {"image_id": ["A", "A"], "fold": [0, 1], "risk_score": [0.2, 0.8]}
    )
    reduced = reduce_whole_image_risk(per_fold_scores, None)
    assert reduced.set_index("image_id").loc["A", "risk_score"] == pytest.approx(0.5)
    assert reduced.set_index("image_id").loc["A", "fold"] is None
