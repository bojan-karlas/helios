"""Path resolution -- the dataset-agnostic seam.

The resolver is the ONLY place that knows how an artifact maps onto disk. Given
an artifact id and a set of key values, it renders the artifact's relative
``path`` template and joins it under the correct root.

Artifacts declare a storage role. Unset derived roots fall back to
``<source_root>/<role>``; separate roots allow the source directory to be
mounted read-only. Roots are stored absolute so paths stay valid in parallel
workers whose working directory differs from the coordinator's.

Optional-key rule: ``fold`` is an OPTIONAL key dimension. When its value is
absent (``None``), its ``fold={fold}/`` path segment is DROPPED -- this is how a
single artifact id addresses both per-fold instances and the reduced
(fold-free) instance.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from helios.data.catalog import StorageRole, get_artifact

OPTIONAL_KEYS = ("fold", "source")


def format_hive_value(value: Any) -> str:
    """Format a scalar or sequence partition value into a canonical Hive segment."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "+".join(format_hive_value(v) for v in value)
    return str(value)


def parse_hive_value(raw: str) -> Any:
    """Coerce a Hive segment value back to bool/int/float when possible."""
    if raw == "true":
        return True
    if raw == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def format_hive_subpath(params: Mapping[str, Any] | None) -> Path:
    """Render extra ``key=value`` directories in canonical sorted key order."""
    if not params:
        return Path()
    parts = [f"{k}={format_hive_value(params[k])}" for k in sorted(params) if params[k] is not None]
    return Path(*parts) if parts else Path()


@dataclass(frozen=True)
class StorageRoots:
    """Run overrides for the supported storage roles."""

    preprocessing_root: Path | None = None
    models_root: Path | None = None
    results_root: Path | None = None
    splits_file: Path | None = None


@dataclass(frozen=True)
class Roots:
    """Resolved role roots for one dataset or cohort."""

    source_root: Path
    preprocessing_root: Path | None = None
    models_root: Path | None = None
    results_root: Path | None = None

    def __post_init__(self) -> None:
        for name in ("source_root", "preprocessing_root", "models_root", "results_root"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).absolute())

    def for_role(self, role: StorageRole) -> Path:
        if role == "source":
            return self.source_root
        explicit = getattr(self, f"{role}_root")
        if explicit is not None:
            return explicit
        return self.source_root / role

    @classmethod
    def single(cls, root: str | Path) -> Roots:
        return cls(source_root=Path(root))


def render_path(template: str, keys: dict[str, Any], extra_hive: Mapping[str, Any] | None = None) -> str:
    """Render a relative path template, applying the optional-key drop rule.

    Any optional key (e.g. ``fold``, ``source``) whose value is ``None`` has its
    path segment removed: ``key={key}/`` and bare ``{key}/`` forms are stripped.
    Remaining placeholders are substituted from ``keys`` using canonical Hive
    formatting. Optional ``extra_hive`` key-value pairs are nested in sorted
    order immediately before the filename.
    """
    rendered = template
    norm_keys = dict(keys)
    for opt in OPTIONAL_KEYS:
        if norm_keys.get(opt) is None:
            rendered = re.sub(rf"{opt}=\{{{opt}\}}/", "", rendered)
            rendered = re.sub(rf"\{{{opt}\}}/", "", rendered)
    placeholders = set(re.findall(r"{(\w+)}", rendered))
    missing = placeholders - norm_keys.keys()
    if missing:
        raise KeyError(f"Missing key values {sorted(missing)} for path template {template!r}")
    subst = {k: format_hive_value(v) for k, v in norm_keys.items() if k in placeholders}
    formatted = rendered.format(**subst)
    extra = format_hive_subpath(extra_hive)
    if extra != Path():
        p = Path(formatted)
        formatted = str(p.parent / extra / p.name)
    return formatted


class Resolver:
    """Resolves (artifact_id, key_values) -> absolute path under the right root."""

    def __init__(self, roots: Roots):
        self.roots = roots

    def path(self, artifact_id: str, *, extra_hive: Mapping[str, Any] | None = None, **keys: Any) -> Path:
        artifact = get_artifact(artifact_id)
        if artifact.path is None:
            raise ValueError(f"Artifact {artifact_id!r} has no path template (non-file artifact).")
        rel = render_path(artifact.path, keys, extra_hive=extra_hive)
        root = self.roots.for_role(artifact.role)
        return root / rel

    def exists(self, artifact_id: str, *, extra_hive: Mapping[str, Any] | None = None, **keys: Any) -> bool:
        return self.path(artifact_id, extra_hive=extra_hive, **keys).exists()

    def role(self, artifact_id: str) -> StorageRole:
        return get_artifact(artifact_id).role


def scan_hive(
    root: str | Path,
    filename: str | None = None,
    *,
    defaults: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Collect CSV/Parquet files under ``root`` and parse ``key=value`` path segments as columns.

    Skips any file inside a ``*.parts`` directory so per-slide part files are never
    duplicated alongside reduced tables.
    """
    base = Path(root)
    if not base.exists():
        return pd.DataFrame()
    pattern = f"**/{filename}" if filename else "**/*"
    frames: list[pd.DataFrame] = []
    for path in sorted(base.glob(pattern)):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(base).parts
        if any(part.endswith(".parts") for part in rel_parts):
            continue
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            continue
        hive_cols: dict[str, Any] = dict(defaults or {})
        for part in rel_parts[:-1]:
            if "=" in part:
                k, v = part.split("=", 1)
                if k:
                    hive_cols[k] = parse_hive_value(v)
        for k, v in hive_cols.items():
            if k not in df.columns:
                df[k] = v
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
