"""Versioned model-bundle storage.

A bundle is a directory ``<models_root>/<version>/<component>/`` holding a
component's persisted model file(s) plus a ``manifest.json`` recording the
format and a content digest for provenance. The pipeline saves bundles after
fitting and loads them at inference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from helios.utils.digests import sha256_file

MANIFEST_NAME = "manifest.json"


@dataclass
class Bundle:
    """A versioned, on-disk model bundle for one component."""

    root: Path
    version: str
    component: str

    @property
    def dir(self) -> Path:
        return self.root / self.version / self.component

    def save(self, filename: str, save_fn) -> Path:
        """Persist a model via ``save_fn(path)`` and record it in the manifest."""
        self.dir.mkdir(parents=True, exist_ok=True)
        target = self.dir / filename
        save_fn(target)
        self._update_manifest(filename, sha256_file(target))
        return target

    def path(self, filename: str) -> Path:
        return self.dir / filename

    def _update_manifest(self, filename: str, digest: str) -> None:
        manifest_path = self.dir / MANIFEST_NAME
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("component", self.component)
        manifest.setdefault("version", self.version)
        manifest.setdefault("files", {})[filename] = {"sha256": digest}
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def manifest(self) -> dict:
        manifest_path = self.dir / MANIFEST_NAME
        return json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
