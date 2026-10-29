"""Per-format read/write, kept side by side.

Low-level helpers dispatch on a ``format`` string; the :class:`ArtifactStore`
ties IO to the :class:`~helios.data.resolver.Resolver` and the artifact registry
so components read/write artifacts by id + key without knowing on-disk layout.

Two access modes:

* :meth:`ArtifactStore.read` -- eager: materialize the whole artifact (small
  tables, embeddings, configs).
* :meth:`ArtifactStore.path` -- lazy: resolve to a path the consumer opens
  itself, for random-access streaming (e.g. sampling 1k of 50k tiles from an
  HDF5 stack in a DataLoader) without loading everything into memory.

Supported formats (scaffold): csv, parquet, json, npy, hdf5, pkl, bundle,
txt/md, png. ``bundle`` is a provenance-bearing model directory (see
:mod:`helios.data.bundle`). Whole-slide image decoding (svs/tiff/ndpi) is
intentionally left to the component that needs it.
"""
from __future__ import annotations

import json
import os
import pickle
import uuid
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from helios.data.bundle import is_complete_bundle, load_bundle, save_bundle
from helios.data.catalog import get_artifact
from helios.data.hdf5 import ArrayKind, array_dataset
from helios.data.resolver import Resolver

ARRAY_ARTIFACTS: dict[str, ArrayKind] = {
    "tiles": "tiles",
    "tile_augmentation": "tiles",
    "tile_features": "features",
    "tile_augmentation_features": "features",
}
ORDINAL_TABLES = {"tile_metadata", "tile_background"}


def ordinal_tile_table(table: pd.DataFrame) -> pd.DataFrame:
    """Expose row identity in memory; full tile tables must never be reordered."""
    table = table.copy()
    ids = np.arange(len(table), dtype=np.int64)
    if "tile_id" in table:
        if not np.array_equal(table["tile_id"].to_numpy(), ids):
            raise ValueError("Full tile table tile_id values must equal the row ordinals 0..N-1.")
    else:
        if not np.array_equal(table.index.to_numpy(), ids):
            raise ValueError(
                "Full tile table index must equal row ordinals; partial tables cannot be saved as full grids."
            )
        table.insert(0, "tile_id", ids)
    return table


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _temp_sibling(path: Path) -> Path:
    return path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"


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
    if base == "bundle":
        return save_bundle(obj, path)
    if base not in {"csv", "parquet", "json", "npy", "hdf5", "pkl", "txt", "md", "png"}:
        raise ValueError(f"Unsupported write format: {fmt!r}")

    tmp = _temp_sibling(path)
    try:
        if base == "csv":
            _as_frame(obj).to_csv(tmp, index=False)
        elif base == "parquet":
            _as_frame(obj).to_parquet(tmp, index=False)
        elif base == "json":
            tmp.write_text(json.dumps(obj, indent=2, default=str))
        elif base == "npy":
            with tmp.open("wb") as f:
                np.save(f, np.asarray(obj))
        elif base == "hdf5":
            import h5py

            with h5py.File(tmp, "w") as h:
                for k, v in dict(obj).items():
                    array = np.asarray(v)
                    if k == "tiles" and array.ndim == 4 and array.dtype == np.uint8:
                        dataset = h.create_dataset(k, shape=(len(array),), dtype=np.dtype((np.uint8, array.shape[1:])))
                        dataset[:] = array
                    else:
                        h.create_dataset(k, data=array)
        elif base == "pkl":
            with tmp.open("wb") as f:
                pickle.dump(obj, f)
        elif base in {"txt", "md"}:
            tmp.write_text(str(obj))
        elif base == "png":
            if isinstance(obj, (bytes, bytearray)):
                with BytesIO(obj) as stream:
                    from PIL import Image

                    with Image.open(stream) as image:
                        image.verify()
                tmp.write_bytes(obj)
            else:
                from PIL import Image

                array = np.asarray(obj)
                if array.dtype == np.bool_:
                    array = array.astype(np.uint8) * 255
                Image.fromarray(array).save(tmp, format="PNG")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path


def _as_frame(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    return pd.DataFrame(obj)


class ArtifactStore:
    """Read/write artifacts by id + key, resolving paths and formats from the spec."""

    def __init__(self, resolver: Resolver):
        self.resolver = resolver

    def read(self, artifact_id: str, *, extra_hive: Mapping[str, Any] | None = None, **keys: Any) -> Any:
        artifact = get_artifact(artifact_id)
        path = self.resolver.path(artifact_id, extra_hive=extra_hive, **keys)
        if artifact_id in ARRAY_ARTIFACTS:
            import h5py

            kind = ARRAY_ARTIFACTS[artifact_id]
            with h5py.File(path, "r") as file:
                return {kind: array_dataset(file, kind)[()]}
        if artifact_id in ORDINAL_TABLES:
            table = ordinal_tile_table(read(path, artifact.format))
            if artifact_id == "tile_metadata":
                table["size_mm"] = keys["size_mm"]
            return table
        return read(path, artifact.format)

    def path(self, artifact_id: str, *, extra_hive: Mapping[str, Any] | None = None, **keys: Any) -> Path:
        """Resolve an artifact to its on-disk path WITHOUT reading it.

        This is the streaming seam: large random-access artifacts (e.g. an HDF5
        tile stack of 50k tiles) should be opened lazily by the consumer -- a
        torch/tf ``Dataset`` opens this path per worker and slices only the tiles
        a batch needs, rather than having :meth:`read` load the whole array into
        memory. :meth:`read` stays the convenient eager path for small artifacts.
        """
        return self.resolver.path(artifact_id, extra_hive=extra_hive, **keys)

    def write(
        self,
        artifact_id: str,
        obj: Any,
        *,
        extra_hive: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **keys: Any,
    ) -> Path:
        artifact = get_artifact(artifact_id)
        path = self.resolver.path(artifact_id, extra_hive=extra_hive, **keys)
        if artifact.format.split("/")[0] == "bundle":
            return save_bundle(obj, path, metadata=metadata)
        if artifact_id in ARRAY_ARTIFACTS:
            kind = ARRAY_ARTIFACTS[artifact_id]
            array = np.asarray(obj[kind])
            if kind == "tiles":
                valid = (
                    array.ndim == 4 and array.shape[-1] == 3
                    and min(array.shape[1:3]) > 0 and array.dtype == np.uint8
                )
            else:
                valid = array.ndim == 2 and array.shape[1] > 0 and np.issubdtype(array.dtype, np.number)
            if not valid:
                raise ValueError(f"{artifact_id}: invalid {kind} shape/dtype: {array.shape}, {array.dtype}.")
            obj = {kind: array}
        elif artifact_id in ORDINAL_TABLES:
            table = ordinal_tile_table(_as_frame(obj))
            obj = table.drop(columns=["tile_id", "size_mm"], errors="ignore")
        return write(obj, path, artifact.format)

    def exists(self, artifact_id: str, *, extra_hive: Mapping[str, Any] | None = None, **keys: Any) -> bool:
        artifact = get_artifact(artifact_id)
        path = self.resolver.path(artifact_id, extra_hive=extra_hive, **keys)
        if artifact.format.split("/")[0] == "bundle":
            return is_complete_bundle(path)
        return path.exists()

    def remove(self, artifact_id: str, *, extra_hive: Mapping[str, Any] | None = None, **keys: Any) -> None:
        path = self.resolver.path(artifact_id, extra_hive=extra_hive, **keys)
        if path.is_file():
            path.unlink(missing_ok=True)

    def part_path(
        self,
        artifact_id: str,
        row_id: str,
        *,
        extra_hive: Mapping[str, Any] | None = None,
        **keys: Any,
    ) -> Path:
        table_path = self.resolver.path(artifact_id, extra_hive=extra_hive, **keys)
        return table_path.parent / f"{table_path.name}.parts" / f"{row_id}.csv"

    def part_exists(
        self,
        artifact_id: str,
        row_id: str,
        *,
        extra_hive: Mapping[str, Any] | None = None,
        **keys: Any,
    ) -> bool:
        return self.part_path(artifact_id, row_id, extra_hive=extra_hive, **keys).is_file()

    def write_part(
        self,
        artifact_id: str,
        row_id: str,
        obj: Any,
        *,
        extra_hive: Mapping[str, Any] | None = None,
        **keys: Any,
    ) -> Path:
        path = self.part_path(artifact_id, row_id, extra_hive=extra_hive, **keys)
        return write(obj, path, "csv")

    def read_parts(
        self,
        artifact_id: str,
        row_ids: list[str] | tuple[str, ...],
        *,
        extra_hive: Mapping[str, Any] | None = None,
        **keys: Any,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for row_id in row_ids:
            path = self.part_path(artifact_id, row_id, extra_hive=extra_hive, **keys)
            if path.is_file():
                frames.append(read(path, "csv"))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
