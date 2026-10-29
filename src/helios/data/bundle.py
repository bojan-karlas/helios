"""Model bundles -- provenance-bearing, on-disk model artifacts.

A *bundle* is a directory (not a single file) holding a fitted model plus a
``manifest.json`` that records, for every file in the bundle, its serialization
format and a SHA-256 content digest. The digest gives provenance independent of
path or cohort name; the manifest's presence also marks the bundle *complete*
(it is written last, so a crash mid-save leaves no manifest and the bundle reads
as not-existing).

The store wires this in via ``format: bundle``: :func:`save_bundle` pickles the
model to ``model.pkl`` and writes the manifest; :func:`load_bundle` returns the
unpickled model.

Torch models can instead be stored as a plain ``state_dict`` in ``model.pt``: a
registered :class:`~helios.models.base.Model` whose ``bundle_format`` is
``"torch_state_dict"`` provides ``to_state_dict()`` and a ``from_state_dict``
classmethod. The manifest then records the model's import path, and loading
rebuilds the model from ``model.pt`` plus the fit parameters in ``config.yaml``.
Such a bundle is an ordinary torch checkpoint next to readable YAML, so a model
trained outside HELIOS can be placed into one directly.
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
STATE_DICT_NAME = "model.pt"
CONFIG_NAME = "config.yaml"
COHORT_NAME = "cohort.yaml"
STATE_DICT_FORMAT = "torch_state_dict"


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

    as_state_dict = getattr(obj, "bundle_format", None) == STATE_DICT_FORMAT
    model_name = STATE_DICT_NAME if as_state_dict else MODEL_NAME
    model_path = bundle_dir / model_name
    tmp_model = bundle_dir / f".{model_name}.{uuid.uuid4().hex}.tmp"
    try:
        if as_state_dict:
            import torch

            torch.save(obj.to_state_dict(), tmp_model)
        else:
            with tmp_model.open("wb") as f:
                pickle.dump(obj, f)
        os.replace(tmp_model, model_path)
    finally:
        if tmp_model.exists():
            tmp_model.unlink(missing_ok=True)

    files: dict[str, dict[str, str]] = {
        model_name: {
            "format": STATE_DICT_FORMAT if as_state_dict else "pickle",
            "sha256": sha256_file(model_path),
        },
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
    if as_state_dict:
        manifest["model"] = f"{type(obj).__module__}:{type(obj).__qualname__}"
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
    files = manifest.get("files", {})
    as_state_dict = files.get(STATE_DICT_NAME, {}).get("format") == STATE_DICT_FORMAT
    model_name = STATE_DICT_NAME if as_state_dict else MODEL_NAME
    model_path = bundle_dir / model_name
    recorded = files.get(model_name, {}).get("sha256")
    if recorded is not None and sha256_file(model_path) != recorded:
        raise ValueError(f"Bundle digest mismatch for {model_path} (corrupted or tampered).")

    if as_state_dict:
        import importlib

        import torch

        module_name, _, class_name = manifest["model"].partition(":")
        model_class = getattr(importlib.import_module(module_name), class_name)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        return model_class.from_state_dict(state, _fit_params(bundle_dir))

    with model_path.open("rb") as f:
        return pickle.load(f)


def _fit_params(bundle_dir: Path) -> dict[str, Any]:
    """The fit parameters recorded in a bundle's ``config.yaml`` (``{"fit": {noun: params}}``)."""
    config_path = bundle_dir / CONFIG_NAME
    if not config_path.exists():
        return {}
    fit = (yaml.safe_load(config_path.read_text()) or {}).get("fit") or {}
    if len(fit) != 1:
        raise ValueError(f"{config_path}: expected exactly one fit stage, found {sorted(fit)}.")
    return dict(next(iter(fit.values())) or {})


def is_complete_bundle(bundle_dir: str | Path) -> bool:
    """True once the manifest exists (the marker of a fully-written bundle)."""
    return (Path(bundle_dir) / MANIFEST_NAME).exists()
