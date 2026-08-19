"""Foundation-model feature encoder tests without downloading model weights."""
from __future__ import annotations

import numpy as np
import torch

from helios.components.features import _load_encoder, extract_features


class _FakeVirchow2(torch.nn.Module):
    pretrained_cfg: dict[str, object] = {}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        tokens = torch.zeros((batch, 7, 1280), dtype=torch.float32, device=images.device)
        tokens[:, 0] = 2.0
        tokens[:, 1:5] = 99.0  # register tokens must be excluded
        tokens[:, 5] = 4.0
        tokens[:, 6] = 6.0
        return tokens


def test_virchow2_uses_official_cls_plus_mean_patch_recipe(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_model(name: str, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return _FakeVirchow2()

    def fake_transform(image):
        array = np.asarray(image, dtype=np.float32)
        return torch.from_numpy(array).permute(2, 0, 1)

    monkeypatch.setattr("timm.create_model", fake_create_model)
    monkeypatch.setattr("timm.data.resolve_data_config", lambda *args, **kwargs: {})
    monkeypatch.setattr("timm.data.transforms_factory.create_transform", lambda **kwargs: fake_transform)
    _load_encoder.cache_clear()

    tiles = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    features = extract_features(tiles, model="virchow2", batch_size=1, device="cpu")

    assert captured["name"] == "hf-hub:paige-ai/Virchow2"
    assert features.shape == (2, 2560)
    np.testing.assert_array_equal(features[:, :1280], 2.0)
    np.testing.assert_array_equal(features[:, 1280:], 5.0)
    _load_encoder.cache_clear()


def test_feature_batch_size_must_be_positive() -> None:
    tiles = np.empty((0, 224, 224, 3), dtype=np.uint8)
    try:
        extract_features(tiles, model="virchow2", batch_size=0, device="cpu")
    except ValueError as error:
        assert "batch_size" in str(error)
    else:
        raise AssertionError("expected invalid batch_size to fail")
