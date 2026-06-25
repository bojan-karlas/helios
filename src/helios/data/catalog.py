"""Artifact + component registries (loaded from the shipped, spec-derived YAML).

These registries describe the pipeline's data model at runtime:

* An :class:`ArtifactSpec` records an artifact's ``key`` (identifying fields),
  ``format`` (how it is stored), and ``path`` (a relative template with
  ``{field}`` placeholders drawn from ``key``).
* A :class:`ComponentSpec` records a component's ``consumes`` / ``produces``
  artifact ids, its ``fold`` behavior (``none`` | ``map`` | ``reduce``), and the
  ``partitions`` it fans out over.

The YAML files (``configs/artifacts.yaml`` / ``configs/components.yaml``) are
GENERATED from ``dev/notes/agent/pipeline-spec.yaml`` by
``scripts/gen_registries.py`` — treat the spec as the source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml


def _configs_dir() -> Path:
    """Locate the shipped ``configs/`` directory.

    Works both from a source checkout (repo-root/configs) and, as a fallback,
    from an installed package layout. The scaffold targets the source checkout.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs"
        if (candidate / "artifacts.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate configs/artifacts.yaml relative to the package.")


@dataclass(frozen=True)
class ArtifactSpec:
    id: str
    title: str | None
    key: tuple[str, ...]
    format: str
    path: str | None

    @property
    def is_foldable(self) -> bool:
        return "fold" in self.key


@dataclass(frozen=True)
class ComponentSpec:
    id: str
    title: str | None
    kind: str | None
    phase: str | None
    mode: tuple[str, ...]
    fold: str
    module: str | None
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    partitions: tuple[str, ...]
    params: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, ArtifactSpec]:
    data = yaml.safe_load((_configs_dir() / "artifacts.yaml").read_text())["artifacts"]
    return {
        aid: ArtifactSpec(
            id=aid,
            title=a.get("title"),
            key=tuple(a.get("key", []) or []),
            format=a.get("format", "bundle"),
            path=a.get("path"),
        )
        for aid, a in data.items()
    }


@lru_cache(maxsize=1)
def load_components() -> dict[str, ComponentSpec]:
    data = yaml.safe_load((_configs_dir() / "components.yaml").read_text())["components"]
    return {
        cid: ComponentSpec(
            id=cid,
            title=c.get("title"),
            kind=c.get("kind"),
            phase=c.get("phase"),
            mode=tuple(c.get("mode", []) or []),
            fold=c.get("fold", "none"),
            module=c.get("module"),
            consumes=tuple(c.get("consumes", []) or []),
            produces=tuple(c.get("produces", []) or []),
            partitions=tuple(c.get("partitions", []) or []),
            params=dict(c.get("params", {}) or {}),
        )
        for cid, c in data.items()
    }


def get_artifact(artifact_id: str) -> ArtifactSpec:
    try:
        return load_artifacts()[artifact_id]
    except KeyError as exc:
        raise KeyError(f"Unknown artifact id: {artifact_id!r}") from exc


def get_component(component_id: str) -> ComponentSpec:
    try:
        return load_components()[component_id]
    except KeyError as exc:
        raise KeyError(f"Unknown component id: {component_id!r}") from exc
