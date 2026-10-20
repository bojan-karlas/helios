"""Artifact registry (loaded from the shipped, spec-derived YAML).

An :class:`ArtifactSpec` records an artifact's ``key`` (identifying fields),
``format`` (how it is stored), and ``path`` (a relative template with
``{field}`` placeholders drawn from ``key``). The resolver and IO layer load it
to map an artifact id + key to an on-disk location.

``configs/artifacts.yaml`` is GENERATED from
``dev/notes/agent/pipeline-spec.yaml`` by ``scripts/gen_registries.py`` -- treat
the spec as the source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast, get_args

import yaml

StorageRole = Literal["source", "preprocessing", "features", "models", "results"]


def storage_role(value: object) -> StorageRole:
    if value not in get_args(StorageRole):
        raise ValueError(f"Invalid artifact storage role: {value!r}")
    return cast(StorageRole, value)


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
    role: StorageRole

    @property
    def is_foldable(self) -> bool:
        return "fold" in self.key


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
            role=storage_role(a.get("role")),
        )
        for aid, a in data.items()
    }


def get_artifact(artifact_id: str) -> ArtifactSpec:
    try:
        return load_artifacts()[artifact_id]
    except KeyError as exc:
        raise KeyError(f"Unknown artifact id: {artifact_id!r}") from exc
