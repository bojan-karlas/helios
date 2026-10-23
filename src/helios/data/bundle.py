"""Model bundles -- provenance-bearing, on-disk model artifacts.

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
import os
import pickle
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from helios.utils.digests import sha256_file

MANIFEST_NAME = "manifest.json"
MODEL_NAME = "model.pkl"
CONFIG_NAME = "config.yaml"
COHORT_NAME = "cohort.yaml"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def save_bundle(
    obj: Any,
    bundle_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Pickle ``obj`` into ``bundle_dir`` and write a digest manifest atomically.

    Any prior ``manifest.json`` is removed first so an interrupted overwrite never
    leaves a stale manifest marking a half-written bundle as complete. If
    ``metadata`` provides ``config`` or ``cohort`` mappings, they are written to
    ``config.yaml`` and ``cohort.yaml`` inside the bundle directory so the training
    run can be reproduced directly via ``--config`` and ``--dataset``.
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = bundle_dir / MANIFEST_NAME
    if manifest_path.exists():
        manifest_path.unlink(missing_ok=True)

    model_path = bundle_dir / MODEL_NAME
    tmp_model = bundle_dir / f".{MODEL_NAME}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp_model.open("wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp_model, model_path)
    finally:
        if tmp_model.exists():
            tmp_model.unlink(missing_ok=True)

    files: dict[str, dict[str, str]] = {
        MODEL_NAME: {"format": "pickle", "sha256": sha256_file(model_path)},
    }

    if metadata:
        if "config" in metadata and metadata["config"] is not None:
            config_path = bundle_dir / CONFIG_NAME
            _atomic_write_text(config_path, yaml.safe_dump(metadata["config"], sort_keys=False))
            files[CONFIG_NAME] = {"format": "yaml", "sha256": sha256_file(config_path)}
        if "cohort" in metadata and metadata["cohort"] is not None:
            cohort_path = bundle_dir / COHORT_NAME
            _atomic_write_text(cohort_path, yaml.safe_dump(metadata["cohort"], sort_keys=False))
            files[COHORT_NAME] = {"format": "yaml", "sha256": sha256_file(cohort_path)}

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    if metadata:
        extra = {k: v for k, v in metadata.items() if k not in {"config", "cohort"}}
        if extra:
            manifest["provenance"] = extra

    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, default=str))
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
