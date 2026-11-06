"""Component-level tests for helios.components.mil (real training/inference, no mocks).

Builds tiny synthetic tile-feature / tile-background artifacts directly (the same
on-disk shapes helios.data.io writes: an hdf5 "features" dataset and a parquet
with tile_id/is_background columns) and exercises train_mil/infer_mil/
reduce_whole_image_risk end to end on CPU.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from helios.components.mil import (
    SlideFeatureRef,
    _load_training_bag,
    infer_mil,
    infer_mil_score,
    reduce_whole_image_risk,
    train_mil,
)
from helios.progress import Progress

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


@pytest.mark.parametrize("validation", [False, True])
def test_training_logs_phases_summaries_and_stopping(slides, monkeypatch, caplog, validation):
    if validation:
        monkeypatch.setattr("helios.components.mil._validation_metrics", lambda *a, **k: (0.5, 0.6))
    with caplog.at_level("INFO", logger="helios"):
        train_mil(
            slides, max_epochs=3, patience=1, val_frac=0.34 if validation else 0,
            device="cpu", progress=Progress(use_tqdm=False), fold=2,
        )
    text = caplog.text
    assert "[fold=2] MIL device=cpu" in text
    assert "[fold=2] Epoch 1/3 train" in text
    assert "loss=" in text and "skipped=0" in text and "elapsed=" in text
    assert "ETA to 3 epochs=estimating" in text
    assert "training complete" in text
    if validation:
        assert "val_auprc=0.5000 val_auroc=0.6000" in text
        assert "patience=1/1" in text
        assert "early stopping; restoring best_epoch=1" in text
        assert "training complete epochs=2/3 selected_epoch=1" in text
    else:
        assert "epoch=3/3 loss=" in text
        assert "epoch budget exhausted; selected_epoch=3" in text


def test_validation_progress_and_empty_bag_counts(slides, tmp_path, caplog):
    from helios.components.mil import MilAttentionModel, _validation_metrics
    from helios.progress.training import TrainingProgress

    empty = _make_slide(tmp_path, "empty", n_tiles=3, n_background=3, label=0)
    model = MilAttentionModel()
    model.input_dim = D
    with caplog.at_level("INFO", logger="helios"):
        with TrainingProgress(Progress(use_tqdm=False), 1, "").display() as feedback:
            feedback.start_epoch()
            _validation_metrics(
                model._build_net(), [*slides, empty], device=torch.device("cpu"), feedback=feedback,
            )
            assert feedback.validation_skipped == 1
    assert "Epoch 1/1 valid 7/7 slides" in caplog.text
    assert "skipped=1" in caplog.text


def test_feedback_does_not_change_trained_weights(slides, caplog):
    with caplog.at_level("INFO", logger="helios"):
        silent = train_mil(slides, max_epochs=1, val_frac=0, device="cpu")
        visible = train_mil(slides, max_epochs=1, val_frac=0, device="cpu", progress=Progress(use_tqdm=False))
    for key, tensor in silent._state_dict.items():
        np.testing.assert_array_equal(tensor.numpy(), visible._state_dict[key].numpy())


def test_inference_reuses_network_and_refit_invalidates_it(
    slides: list[SlideFeatureRef], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from helios.components.mil import _inference_net

    model = train_mil(slides, max_epochs=1, val_frac=0.0, device="cpu")
    original = model._build_net
    builds = []

    def build():
        net = original()
        builds.append(net)
        return net

    monkeypatch.setattr(model, "_build_net", build)
    path = slides[0].features_path
    first = infer_mil(model, features_path=path, device="cpu")
    second = infer_mil(model, features_path=path, device="cpu")
    assert len(builds) == 1
    np.testing.assert_allclose(first.patch_embeddings, second.patch_embeddings)
    np.testing.assert_allclose(first.slide_embedding_grad, second.slide_embedding_grad)
    assert first.risk_score == second.risk_score
    assert "_inference_net" not in model.__dict__
    model.fit(slides, max_epochs=1, val_frac=0.0, device="cpu")
    assert _inference_net.cache_info().currsize == 0
    before_inference = len(builds)
    infer_mil(model, features_path=path, device="cpu")
    assert len(builds) == before_inference + 1


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


def test_score_only_matches_attribution_without_gradients(tmp_path, slides, monkeypatch):
    from helios.components.mil import _inference_net

    model = train_mil(slides, max_epochs=1, val_frac=0, device="cpu")
    slide = _make_slide(tmp_path, "held-out", n_tiles=10, n_background=4, label=1)
    expected = infer_mil(
        model, features_path=slide.features_path, background_path=slide.background_path, device="cpu",
    ).risk_score
    net = _inference_net(model, "cpu")
    net.zero_grad(set_to_none=True)
    monkeypatch.setattr(net, "forward_with_attribution", lambda *a: pytest.fail("must not compute attribution"))
    observed = []
    hook = net.register_forward_pre_hook(lambda *a: observed.append(torch.is_inference_mode_enabled()))
    try:
        actual = infer_mil_score(
            model, features_path=slide.features_path, background_path=slide.background_path, device="cpu",
        )
    finally:
        hook.remove()
    assert actual == pytest.approx(expected, abs=1e-7)
    assert observed == [True]
    assert all(parameter.grad is None for parameter in net.parameters())
    empty = _make_slide(tmp_path, "empty-score", n_tiles=3, n_background=3, label=0)
    with pytest.raises(ValueError, match="No foreground tiles"):
        infer_mil_score(
            model, features_path=empty.features_path, background_path=empty.background_path, device="cpu",
        )


def test_aug_mode_replace_swaps_in_place(tmp_path: Path) -> None:
    n = 20
    feats = np.zeros((n, D), dtype=np.float32)
    aug = np.ones((n, D), dtype=np.float32)
    features_path = tmp_path / "orig.hdf5"
    aug_path = tmp_path / "aug.hdf5"
    _write_features(features_path, feats)
    _write_features(aug_path, aug)

    ref = SlideFeatureRef(
        image_id="A", features_path=features_path, background_path=None, aug_features_paths=(aug_path,), label=1
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
        image_id="A", features_path=features_path, background_path=None, aug_features_paths=(aug_path,), label=1
    )
    bag = _load_training_bag(
        ref, aug_swap_prob=1.0, aug_mode="augment", rng=np.random.default_rng(0)
    )
    assert bag.shape == (2 * n, D)
    assert np.allclose(bag[:n], 0.0)
    assert np.allclose(bag[n:], 1.0)


@pytest.mark.parametrize("n_augmentations", [0, 1, 2])
@pytest.mark.parametrize("val_frac", [0.0, 0.34])
def test_separate_bags_expand_after_split(tmp_path, slides, monkeypatch, caplog, n_augmentations, val_frac):
    from collections import Counter

    from helios.components import mil

    augmented = []
    for slide in slides:
        paths = []
        for index in range(n_augmentations):
            path = tmp_path / f"{slide.image_id}-aug-{index}.hdf5"
            _write_features(path, np.full((12, D), index + 10, dtype=np.float32))
            paths.append(path)
        augmented.append(replace(slide, aug_features_paths=tuple(paths)))
    train, val = mil._split_train_val(augmented, val_frac=val_frac, seed=42)
    expected_paths = [path for ref in train for path in (ref.features_path, *ref.aug_features_paths)]
    loaded = []
    validations = []
    original_loader = mil._load_training_bag
    original_validation = mil._validation_metrics
    refs_by_id = {ref.image_id: ref for ref in augmented}

    def load(ref, **kwargs):
        source = refs_by_id[ref.image_id]
        assert ref.label == source.label and ref.background_path == source.background_path
        assert not ref.aug_features_paths
        bag = original_loader(ref, **kwargs)
        assert bag.shape == (12, D)
        if ref.features_path != source.features_path:
            assert (bag >= 10).all()
        loaded.append(ref.features_path)
        return bag

    def validate(net, refs, **kwargs):
        assert [ref.features_path for ref in refs] == [ref.features_path for ref in val]
        assert not {ref.image_id for ref in refs}.intersection(ref.image_id for ref in train)
        validations.append(len(refs))
        return original_validation(net, refs, **kwargs)

    monkeypatch.setattr(mil, "_load_training_bag", load)
    monkeypatch.setattr(mil, "_validation_metrics", validate)
    with caplog.at_level("INFO", logger="helios"):
        train_mil(
            augmented, aug_mode="separate", aug_swap_prob=0, max_epochs=2, patience=5,
            val_frac=val_frac, device="cpu", progress=Progress(use_tqdm=False),
        )
    n_bags = len(expected_paths)
    assert len(loaded) == 2 * n_bags
    assert Counter(loaded[:n_bags]) == Counter(expected_paths)
    assert Counter(loaded[n_bags:]) == Counter(expected_paths)
    assert loaded[:n_bags] != loaded[n_bags:]
    assert validations == ([len(val)] * 2 if val else [])
    assert f"train={n_bags} validation={len(val)} train_slides={len(train)}" in caplog.text


def test_separate_without_augmentations_preserves_baseline_weights(slides):
    baseline = train_mil(slides, max_epochs=1, val_frac=0, device="cpu")
    separate = train_mil(slides, aug_mode="separate", max_epochs=1, val_frac=0, device="cpu")
    for key, tensor in baseline._state_dict.items():
        torch.testing.assert_close(tensor, separate._state_dict[key], rtol=0, atol=0)


def test_separate_bags_exclude_background(tmp_path):
    slide = _make_slide(tmp_path, "masked", n_tiles=5, n_background=2, label=1)
    bag = _load_training_bag(slide, aug_mode="separate", aug_swap_prob=1, rng=np.random.default_rng(42))
    assert bag.shape == (3, D)
    with pytest.raises(ValueError, match="must be expanded"):
        _load_training_bag(
            replace(slide, aug_features_paths=(slide.features_path,)),
            aug_mode="separate", aug_swap_prob=1, rng=np.random.default_rng(42),
        )


def test_reduce_whole_image_risk_cv_takes_test_fold_score() -> None:
    per_fold_scores = pd.DataFrame(
        {
            "image_id": ["A", "A", "B", "B"],
            "fold": [0, 1, 0, 1],
            "risk_score": [0.1, 0.9, 0.2, 0.8],
        }
    )
    splits = pd.DataFrame(
        {
            "image_id": ["A", "A", "B", "B"],
            "fold": [0, 1, 0, 1],
            "split": ["train", "test", "test", "train"],
        }
    )
    reduced = reduce_whole_image_risk(per_fold_scores, splits)
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


def test_bundle_stores_state_dict_and_rebuilds_from_config(tmp_path: Path, slides: list[SlideFeatureRef]) -> None:
    import torch

    from helios.data.bundle import MODEL_NAME, STATE_DICT_NAME, load_bundle, save_bundle

    model = train_mil(slides, max_epochs=1, val_frac=0.0, n_branches=2, num_heads=2, device="cpu")
    params = {"n_branches": 2, "num_heads": 2, "dropout_rate": 0.1, "size_mm": 0.25}
    bundle_dir = save_bundle(model, tmp_path / "mil" / "fold=0", metadata={"config": {"fit": {"mil": params}}})

    assert (bundle_dir / STATE_DICT_NAME).is_file()
    assert not (bundle_dir / MODEL_NAME).exists()
    state = torch.load(bundle_dir / STATE_DICT_NAME, weights_only=True)
    assert state.keys() == model.to_state_dict().keys()

    loaded = load_bundle(bundle_dir)
    assert (loaded.n_branches, loaded.num_heads, loaded.input_dim) == (2, 2, D)
    held_out = slides[0]
    paths = {"features_path": held_out.features_path, "background_path": held_out.background_path}
    expected = infer_mil(model, **paths, device="cpu")
    actual = infer_mil(loaded, **paths, device="cpu")
    assert actual.risk_score == pytest.approx(expected.risk_score)
    np.testing.assert_allclose(actual.slide_embedding, expected.slide_embedding, rtol=1e-6)


def test_bundle_rejects_config_that_does_not_match_the_weights(tmp_path: Path, slides: list[SlideFeatureRef]) -> None:
    from helios.data.bundle import load_bundle, save_bundle

    model = train_mil(slides, max_epochs=1, val_frac=0.0, n_branches=2, num_heads=2, device="cpu")
    bundle_dir = save_bundle(
        model, tmp_path / "mil" / "fold=0", metadata={"config": {"fit": {"mil": {"n_branches": 3, "num_heads": 2}}}}
    )
    with pytest.raises(RuntimeError, match="state_dict"):
        load_bundle(bundle_dir)
