"""Model bundles — provenance-bearing, on-disk model artifacts.

A *bundle* is a directory (not a single file) holding a fitted model plus a
``manifest.json`` that records, for every file in the bundle, its serialization
format and a SHA-256 content digest. The digest gives provenance independent of
path or cohort name; the manifest's presence also marks the bundle *complete*
(it is written last, so a crash mid-save leaves no manifest and the bundle reads
as not-existing).

The store wires this in via ``format: bundle``: :func:`save_bundle` pickles the
model to ``model.pkl`` and writes the manifest; :func:`load_bundle` returns the
unpickled model. Components that need a different on-disk form (e.g. a torch
``state_dict``) can write extra files into the same directory and extend the
manifest, but the default round-trip is pickle.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from helios.utils.digests import sha256_file

MANIFEST_NAME = "manifest.json"
MODEL_NAME = "model.pkl"


def save_bundle(obj: Any, bundle_dir: str | Path) -> Path:
    """Pickle ``obj`` into ``bundle_dir`` and write a digest manifest.

    The manifest is written *last* so the bundle is only considered complete once
    its contents are fully on disk.
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    model_path = bundle_dir / MODEL_NAME
    with model_path.open("wb") as f:
        pickle.dump(obj, f)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {MODEL_NAME: {"format": "pickle", "sha256": sha256_file(model_path)}},
    }
    (bundle_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return bundle_dir


def load_bundle(bundle_dir: str | Path) -> Any:
    """Load the model from a bundle, verifying its recorded digest first."""
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Not a complete bundle (no manifest): {bundle_dir}")

    manifest = json.loads(manifest_path.read_text())
    model_path = bundle_dir / MODEL_NAME
    recorded = manifest.get("files", {}).get(MODEL_NAME, {}).get("sha256")
    if recorded is not None and sha256_file(model_path) != recorded:
        raise ValueError(f"Bundle digest mismatch for {model_path} (corrupted or tampered).")

    with model_path.open("rb") as f:
        return pickle.load(f)


def is_complete_bundle(bundle_dir: str | Path) -> bool:
    """True once the manifest exists (the marker of a fully-written bundle)."""
    return (Path(bundle_dir) / MANIFEST_NAME).exists()
