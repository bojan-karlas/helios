"""Path resolution -- the dataset-agnostic seam.

The resolver is the ONLY place that knows how an artifact maps onto disk. Given
an artifact id and a set of key values, it renders the artifact's relative
``path`` template and joins it under the correct root.

Artifacts declare a storage role. Unset derived roots fall back to the dataset
directory, allowing generated files to live alongside fixed source artifacts.
Separate roots allow the source directory to be mounted read-only.

Optional-key rule: ``fold`` is an OPTIONAL key dimension. When its value is
absent (``None``), its ``fold={fold}/`` path segment is DROPPED -- this is how a
single artifact id addresses both per-fold instances and the reduced
(fold-free) instance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from helios.data.catalog import StorageRole, get_artifact

OPTIONAL_KEYS = ("fold",)


@dataclass(frozen=True)
class StorageRoots:
    """Run overrides; dataset-local preprocessing and features stay independent."""

    preprocessing_root: Path | None = None
    features_root: Path | None = None
    models_root: Path | None = None
    results_root: Path | None = None


@dataclass(frozen=True)
class Roots:
    """Resolved role roots for one physical dataset."""

    source_root: Path
    preprocessing_root: Path | None = None
    features_root: Path | None = None
    models_root: Path | None = None
    results_root: Path | None = None

    def for_role(self, role: StorageRole) -> Path:
        return getattr(self, f"{role}_root") or self.source_root

    @classmethod
    def single(cls, root: str | Path) -> Roots:
        return cls(source_root=Path(root))


def render_path(template: str, keys: dict) -> str:
    """Render a relative path template, applying the optional-key drop rule.

    Any optional key (e.g. ``fold``) whose value is ``None`` has its path
    segment removed: ``fold={fold}/`` and bare ``{fold}/`` forms are stripped.
    Remaining placeholders are substituted from ``keys``.
    """
    rendered = template
    for opt in OPTIONAL_KEYS:
        if keys.get(opt) is None:
            rendered = re.sub(rf"{opt}=\{{{opt}\}}/", "", rendered)
            rendered = re.sub(rf"\{{{opt}\}}/", "", rendered)
    placeholders = set(re.findall(r"{(\w+)}", rendered))
    missing = placeholders - keys.keys()
    if missing:
        raise KeyError(f"Missing key values {sorted(missing)} for path template {template!r}")
    subst = {k: v for k, v in keys.items() if k in placeholders}
    return rendered.format(**subst)


class Resolver:
    """Resolves (artifact_id, key_values) -> absolute path under the right root."""

    def __init__(self, roots: Roots):
        self.roots = roots

    def path(self, artifact_id: str, **keys) -> Path:
        artifact = get_artifact(artifact_id)
        if artifact.path is None:
            raise ValueError(f"Artifact {artifact_id!r} has no path template (non-file artifact).")
        rel = render_path(artifact.path, keys)
        root = self.roots.for_role(artifact.role)
        return root / rel

    def exists(self, artifact_id: str, **keys) -> bool:
        return self.path(artifact_id, **keys).exists()

    def role(self, artifact_id: str) -> StorageRole:
        return get_artifact(artifact_id).role
