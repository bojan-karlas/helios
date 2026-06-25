"""Pydantic configuration models.

Config carries the things that vary between runs and environments — roots,
dataset binding, CV settings, model version, and per-stage parameter overrides.
It deliberately holds NO on-disk-layout knowledge (that lives in the resolver)
and no secrets/paths in shipped files (those come from the user's own config).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from helios.data.resolver import Roots


class PathsConfig(BaseModel):
    """Filesystem roots by role. Only ``base`` is required."""

    base: Path
    source: Path | None = None
    work: Path | None = None
    features: Path | None = None
    output: Path | None = None
    models: Path | None = None

    def to_roots(self) -> Roots:
        return Roots(
            base=self.base,
            source=self.source,
            work=self.work,
            features=self.features,
            output=self.output,
            models=self.models,
        )


class CVConfig(BaseModel):
    """Cross-validation settings."""

    n_folds: int = 1
    val_fraction: float = 0.0
    stratify_by: list[str] = Field(default_factory=list)
    group_by: str | None = None
    seed: int = 0


class StageConfig(BaseModel):
    """Per-stage overrides: dataset binding + parameter overrides."""

    dataset: str | None = None  # optional separate dataset root for this fit stage
    params: dict = Field(default_factory=dict)


def _default_partitions() -> dict[str, list[Any]]:
    return {"size_mm": [0.250], "model": ["virchow2"], "target": ["MGB+MRV"]}


class RunConfig(BaseModel):
    """Top-level run configuration."""

    paths: PathsConfig
    version: str = "v000"
    cv: CVConfig = Field(default_factory=CVConfig)
    # Default fan-out dimensions (used when a component declares these partitions).
    partitions: dict[str, list[Any]] = Field(default_factory=_default_partitions)
    stages: dict[str, StageConfig] = Field(default_factory=dict)
    roles: str | None = None  # path to a roles YAML override

    def stage(self, name: str) -> StageConfig:
        return self.stages.get(name, StageConfig())
