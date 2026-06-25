"""Path resolution — the dataset-agnostic seam.

The resolver is the ONLY place that knows how an artifact maps onto disk. Given
an artifact id and a set of key values, it renders the artifact's relative
``path`` template and joins it under the correct root.

Roots are split by role (the pipeline never writes into the read-only source
root); when only a base root is configured every role falls back to it, so the
common single-tree case needs no extra configuration.

Optional-key rule: ``fold`` is an OPTIONAL key dimension. When its value is
absent (``None``), its ``fold={fold}/`` path segment is DROPPED — this is how a
single artifact id addresses both per-fold instances and the reduced
(fold-free) instance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from helios.data.catalog import ArtifactSpec, get_artifact

OPTIONAL_KEYS = ("fold",)


@dataclass(frozen=True)
class Roots:
    """Filesystem roots by role. ``base`` is the fallback for any unset role."""

    base: Path
    source: Path | None = None
    work: Path | None = None
    features: Path | None = None
    output: Path | None = None
    models: Path | None = None

    def _role(self, role: str) -> Path:
        return getattr(self, role, None) or self.base

    @classmethod
    def single(cls, root: str | Path) -> Roots:
        return cls(base=Path(root))


def _role_for(artifact: ArtifactSpec) -> str:
    """Classify an artifact's storage role from its path prefix."""
    path = artifact.path or ""
    if path.startswith("wsi/") or path in {"metadata/image_metadata.csv", "metadata/cv_splits.csv"}:
        return "source"
    if path.startswith("models/"):
        return "models"
    if path.startswith(("results/", "reports/")):
        return "output"
    return "work"


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
        root = self.roots._role(_role_for(artifact))
        return root / rel

    def exists(self, artifact_id: str, **keys) -> bool:
        return self.path(artifact_id, **keys).exists()

    def role(self, artifact_id: str) -> str:
        return _role_for(get_artifact(artifact_id))
