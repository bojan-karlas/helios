"""Per-format read/write, kept side by side.

Low-level helpers dispatch on a ``format`` string; the :class:`ArtifactStore`
ties IO to the :class:`~helios.data.resolver.Resolver` and the artifact registry
so components read/write artifacts by id + key without knowing on-disk layout.

Two access modes:

* :meth:`ArtifactStore.read` — eager: materialize the whole artifact (small
  tables, embeddings, configs).
* :meth:`ArtifactStore.path` — lazy: resolve to a path the consumer opens
  itself, for random-access streaming (e.g. sampling 1k of 50k tiles from an
  HDF5 stack in a DataLoader) without loading everything into memory.

Supported formats (scaffold): csv, parquet, json, npy, hdf5, pkl, bundle,
txt/md, png. ``bundle`` is a provenance-bearing model directory (see
:mod:`helios.data.bundle`). Whole-slide image decoding (svs/tiff/ndpi) is
intentionally left to the component that needs it.
"""
from __future__ import annotations

import json
import pickle
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from helios.data.bundle import is_complete_bundle, load_bundle, save_bundle
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
    if base == "bundle":
        return load_bundle(path)
    if base in {"txt", "md"}:
        return path.read_text()
    if base == "png":
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image).copy()
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
    elif base == "bundle":
        save_bundle(obj, path)
    elif base in {"txt", "md"}:
        path.write_text(str(obj))
    elif base == "png":
        if isinstance(obj, (bytes, bytearray)):
            # Preserve already-encoded PNG payloads.
            with BytesIO(obj) as stream:
                from PIL import Image

                with Image.open(stream) as image:
                    image.verify()
            path.write_bytes(obj)
        else:
            from PIL import Image

            array = np.asarray(obj)
            if array.dtype == np.bool_:
                array = array.astype(np.uint8) * 255
            Image.fromarray(array).save(path, format="PNG")
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

    def path(self, artifact_id: str, **keys) -> Path:
        """Resolve an artifact to its on-disk path WITHOUT reading it.

        This is the streaming seam: large random-access artifacts (e.g. an HDF5
        tile stack of 50k tiles) should be opened lazily by the consumer — a
        torch/tf ``Dataset`` opens this path per worker and slices only the tiles
        a batch needs, rather than having :meth:`read` load the whole array into
        memory. :meth:`read` stays the convenient eager path for small artifacts.
        """
        return self.resolver.path(artifact_id, **keys)

    def write(self, artifact_id: str, obj: Any, **keys) -> Path:
        artifact = get_artifact(artifact_id)
        path = self.resolver.path(artifact_id, **keys)
        return write(obj, path, artifact.format)

    def exists(self, artifact_id: str, **keys) -> bool:
        artifact = get_artifact(artifact_id)
        path = self.resolver.path(artifact_id, **keys)
        if artifact.format.split("/")[0] == "bundle":
            return is_complete_bundle(path)
        return path.exists()
