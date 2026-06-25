"""Feature-extraction stage orchestration tests.

Exercises the stage's wiring (work-unit loop, store I/O, skip logic, selector
validation) against a synthetic cohort, with the pure encoder component
monkeypatched to a deterministic stand-in.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helios.data.io import ArtifactStore
from helios.data.resolver import Resolver, Roots
from helios.stages.tile_features import prep_tile_features
from tests.fixtures import make_cohort

EMBED_DIM = 8


def _fake_extract(tiles, *, model, batch_size=64, device="cuda", progress=None):
    n = len(tiles)
    return np.arange(n * EMBED_DIM, dtype=np.float32).reshape(n, EMBED_DIM)


@pytest.fixture()
def cohort(tmp_path: Path) -> Path:
    return make_cohort(tmp_path / "cohort", n_images=2, n_patients=2)


def _stage_tiles(cohort: Path, image_ids: list[str], n_tiles: int = 5) -> ArtifactStore:
    store = ArtifactStore(Resolver(Roots.single(cohort)))
    for image_id in image_ids:
        tiles = np.zeros((n_tiles, 4, 4, 3), dtype=np.uint8)
        store.write("tiles", {"tiles": tiles}, size_mm=0.25, image_id=image_id)
    return store


def test_stage_extracts_and_writes(monkeypatch, cohort: Path) -> None:
    monkeypatch.setattr("helios.stages.tile_features.extract_features", _fake_extract)

    image_ids = ["FAKE-SKCM-P00000-S01-B1-I1", "FAKE-SKCM-P00001-S01-B1-I1"]
    store = _stage_tiles(cohort, image_ids)

    prep_tile_features(datasets=[cohort], size_mm=[0.25], model=["virchow2"])

    for image_id in image_ids:
        assert store.exists("tile_features", size_mm=0.25, model="virchow2", image_id=image_id)
        out = store.read("tile_features", size_mm=0.25, model="virchow2", image_id=image_id)
        assert out["features"].shape == (5, EMBED_DIM)


def test_stage_skips_existing_unless_forced(monkeypatch, cohort: Path) -> None:
    calls = {"n": 0}

    def _counting(tiles, **kw):
        calls["n"] += 1
        return _fake_extract(tiles, **kw)

    monkeypatch.setattr("helios.stages.tile_features.extract_features", _counting)
    image_ids = ["FAKE-SKCM-P00000-S01-B1-I1", "FAKE-SKCM-P00001-S01-B1-I1"]
    _stage_tiles(cohort, image_ids)

    prep_tile_features(datasets=[cohort], size_mm=[0.25], model=["virchow2"])
    assert calls["n"] == 2
    prep_tile_features(datasets=[cohort], size_mm=[0.25], model=["virchow2"])
    assert calls["n"] == 2, "second run should skip existing artifacts"
    prep_tile_features(datasets=[cohort], size_mm=[0.25], model=["virchow2"], force=True)
    assert calls["n"] == 4, "--force should re-extract"


def test_stage_rejects_unknown_model(cohort: Path) -> None:
    with pytest.raises(ValueError, match="Unknown feature model"):
        prep_tile_features(datasets=[cohort], model=["not-a-model"])
