"""Model-bundle artifact tests: provenance, completeness, and store round-trip.

A ``bundle`` artifact is a directory with a pickled model + a digest manifest.
These tests assert the round-trip through :class:`ArtifactStore`, that a bundle
only counts as existing once complete, and that digest tampering is caught.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from helios.data.bundle import MANIFEST_NAME, MODEL_NAME, load_bundle, save_bundle
from helios.data.io import ArtifactStore
from helios.data.resolver import Resolver, Roots


class _Estimator:
    def __init__(self, coef: float) -> None:
        self.coef = coef

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Estimator) and other.coef == self.coef


def test_store_bundle_round_trip(tmp_path: Path) -> None:
    store = ArtifactStore(Resolver(Roots.single(tmp_path)))
    model = _Estimator(0.42)

    store.write("model_cellular", model, fold=0)

    bundle_dir = tmp_path / "models" / "cellular" / "fold=0"
    assert (bundle_dir / MODEL_NAME).exists()
    assert (bundle_dir / MANIFEST_NAME).exists()
    assert store.exists("model_cellular", fold=0)
    assert store.read("model_cellular", fold=0) == model


def test_bundle_incomplete_until_manifest(tmp_path: Path) -> None:
    store = ArtifactStore(Resolver(Roots.single(tmp_path)))
    bundle_dir = tmp_path / "models" / "cellular" / "fold=1"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / MODEL_NAME).write_bytes(b"partial")

    assert not store.exists("model_cellular", fold=1), "dir without manifest is not a bundle"


def test_bundle_detects_digest_mismatch(tmp_path: Path) -> None:
    save_bundle(_Estimator(1.0), tmp_path / "b")
    (tmp_path / "b" / MODEL_NAME).write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="digest mismatch"):
        load_bundle(tmp_path / "b")


def test_manifest_records_digest(tmp_path: Path) -> None:
    save_bundle(_Estimator(3.0), tmp_path / "b")
    manifest = json.loads((tmp_path / "b" / MANIFEST_NAME).read_text())
    entry = manifest["files"][MODEL_NAME]
    assert entry["format"] == "pickle"
    assert len(entry["sha256"]) == 64
