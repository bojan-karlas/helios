"""Per-format read/write, kept side by side.

Low-level helpers dispatch on a ``format`` string; the :class:`ArtifactStore`
ties IO to the :class:`~helios.data.resolver.Resolver` and the artifact registry
so components read/write artifacts by id + key without knowing on-disk layout.

Supported formats (scaffold): csv, parquet, json, npy, hdf5, pkl, txt/md, png.
Whole-slide image decoding (svs/tiff/ndpi) is intentionally left to the
component that needs it.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from helios.data.catalog import get_artifact
from helios.data.resolver import Resolver


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read(path: str | Path, fmt: str) -> Any:
    path = Path(path)
    base = fmt.split("/")[0]  # e.g. "svs/tiff/ndpi" -> "svs"
    if base == "csv":
        return pd.read_csv(path)
    if base == "parquet":
        return pd.read_parquet(path)
    if base == "json":
        return json.loads(path.read_text())
    if base == "npy":
        return np.load(path, allow_pickle=False)
    if base == "hdf5":
        import h5py

        with h5py.File(path, "r") as h:
            return {k: h[k][()] for k in h.keys()}
    if base == "pkl":
        with path.open("rb") as f:
            return pickle.load(f)
    if base in {"txt", "md"}:
        return path.read_text()
    if base == "png":
        return path.read_bytes()
    raise ValueError(f"Unsupported read format: {fmt!r}")


def write(obj: Any, path: str | Path, fmt: str) -> Path:
    path = Path(path)
    _ensure_parent(path)
    base = fmt.split("/")[0]
    if base == "csv":
        _as_frame(obj).to_csv(path, index=False)
    elif base == "parquet":
        _as_frame(obj).to_parquet(path, index=False)
    elif base == "json":
        path.write_text(json.dumps(obj, indent=2, default=str))
    elif base == "npy":
        np.save(path, np.asarray(obj))
    elif base == "hdf5":
        import h5py

        with h5py.File(path, "w") as h:
            for k, v in dict(obj).items():
                h.create_dataset(k, data=np.asarray(v))
    elif base == "pkl":
        with path.open("wb") as f:
            pickle.dump(obj, f)
    elif base in {"txt", "md"}:
        path.write_text(str(obj))
    elif base == "png":
        path.write_bytes(obj if isinstance(obj, (bytes, bytearray)) else bytes(obj))
    else:
        raise ValueError(f"Unsupported write format: {fmt!r}")
    return path


def _as_frame(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    return pd.DataFrame(obj)


class ArtifactStore:
    """Read/write artifacts by id + key, resolving paths and formats from the spec."""

    def __init__(self, resolver: Resolver):
        self.resolver = resolver

    def read(self, artifact_id: str, **keys) -> Any:
        artifact = get_artifact(artifact_id)
        path = self.resolver.path(artifact_id, **keys)
        return read(path, artifact.format)

    def write(self, artifact_id: str, obj: Any, **keys) -> Path:
        artifact = get_artifact(artifact_id)
        path = self.resolver.path(artifact_id, **keys)
        return write(obj, path, artifact.format)

    def exists(self, artifact_id: str, **keys) -> bool:
        return self.resolver.exists(artifact_id, **keys)
