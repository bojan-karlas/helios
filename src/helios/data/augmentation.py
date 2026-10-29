"""Prepared augmentation directories, ownership, and feature compatibility."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import h5py
import yaml

from helios.data.cohort import ResolvedDataset
from helios.data.hdf5 import array_dataset
from helios.data.io import ArtifactStore, write
from helios.data.resolver import Resolver, Roots


@dataclass(frozen=True)
class Augmentation:
    root: Path
    source_dataset_id: str
    method: str
    parameters: dict[str, Any] = field(default_factory=dict)
    checkpoint: Path | None = None

    @classmethod
    def load(cls, root: str | Path) -> Augmentation:
        root = Path(root).absolute()
        path = root / "augmentation.yaml"
        if not path.is_file():
            raise FileNotFoundError(
                f"{root} is not a prepared augmentation directory (missing augmentation.yaml); "
                "run `helios prep augment` first."
            )
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain an augmentation mapping.")
        for key in ("source_dataset_id", "method"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise ValueError(f"{path} requires a nonempty string '{key}'.")
        parameters = data.get("parameters", {})
        if not isinstance(parameters, dict) or any(not isinstance(k, str) for k in parameters):
            raise ValueError(f"{path}: parameters must be a mapping with string keys.")
        checkpoint = data.get("checkpoint")
        if checkpoint is not None:
            if not isinstance(checkpoint, str) or not checkpoint.strip():
                raise ValueError(f"{path}: checkpoint must be a nonempty path string.")
            checkpoint = Path(checkpoint)
            if not checkpoint.is_absolute():
                checkpoint = root / checkpoint
        return cls(root, data["source_dataset_id"], data["method"], parameters, checkpoint)

    def save(self) -> None:
        data: dict[str, Any] = {
            "source_dataset_id": self.source_dataset_id,
            "method": self.method,
            "parameters": self.parameters,
        }
        if self.checkpoint is not None:
            data["checkpoint"] = str(self.checkpoint.absolute())
        write(yaml.safe_dump(data, sort_keys=False), self.root / "augmentation.yaml", "txt")

    def store(self) -> ArtifactStore:
        return ArtifactStore(Resolver(Roots(source_root=self.root, preprocessing_root=self.root)))


def select_augmentations(
    datasets: list[ResolvedDataset], paths: list[str] | None,
) -> list[ResolvedDataset]:
    """Apply invocation-wide overrides, then validate ownership without guessing."""
    if paths is not None and (
        not isinstance(paths, list) or any(not isinstance(p, str) or not p.strip() for p in paths)
    ):
        raise ValueError("augmentations must be a list of nonempty directory path strings, or null.")
    if paths is not None:
        datasets = [replace(ds, augmentations=()) for ds in datasets]
    if not paths and not any(ds.augmentations for ds in datasets):
        return datasets
    by_id: dict[str, list[int]] = {}
    for index, ds in enumerate(datasets):
        by_id.setdefault(ds.dataset().descriptor.id, []).append(index)
    if paths:
        for path in dict.fromkeys(Path(p).absolute() for p in paths):
            aug = Augmentation.load(path)
            matches = by_id.get(aug.source_dataset_id, [])
            if len(matches) != 1:
                raise ValueError(
                    f"Augmentation {path} source_dataset_id={aug.source_dataset_id!r} "
                    f"matches {len(matches)} supplied datasets; expected exactly one."
                )
            index = matches[0]
            ds = datasets[index]
            datasets[index] = replace(ds, augmentations=(*ds.augmentations, path))
    for ds in datasets:
        for path in ds.augmentations:
            aug = Augmentation.load(path)
            if aug.source_dataset_id != ds.dataset().descriptor.id:
                raise ValueError(f"Augmentation {path} belongs to a different dataset than {ds.root}.")
            if len(by_id[aug.source_dataset_id]) != 1:
                raise ValueError(f"Augmentation {path} matches multiple supplied datasets with the same ID.")
    return datasets


def validate_feature_alignment(original: Path, augmented: Path) -> None:
    """Validate shapes; all full arrays share the original metadata row order."""
    with h5py.File(original, "r") as base, h5py.File(augmented, "r") as aug:
        if array_dataset(base, "features").shape != array_dataset(aug, "features").shape:
            raise ValueError(f"Augmented feature shape differs from original: {augmented}.")


def discover_augmentations(root: Path) -> list[Augmentation]:
    """Discover saved directories without interpreting their names."""
    if not root.is_dir():
        raise FileNotFoundError(f"Augmentation search directory does not exist: {root}")
    return [Augmentation.load(p.parent) for p in sorted(root.rglob("augmentation.yaml"))]
