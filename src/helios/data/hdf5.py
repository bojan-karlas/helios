"""Canonical and legacy tile-array access without materializing unused datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import numpy as np

ArrayKind = Literal["tiles", "features"]


def array_dataset(file: h5py.File, kind: ArrayKind) -> h5py.Dataset:
    """Resolve and validate one pixel stack or embedding matrix by its role."""
    names = [kind] if kind in file else [name for name in ("tile_224", "tiles_224") if name in file]
    if len(names) != 1:
        raise ValueError(f"{file.filename}: expected {kind!r} or one unambiguous legacy tile-array key.")
    dataset = file[names[0]]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"{file.filename}: {names[0]!r} is not an HDF5 dataset.")
    dtype = dataset.dtype
    shape = dataset.shape
    if dtype.subdtype is not None:
        dtype, subshape = dtype.subdtype
        shape += subshape
    if kind == "tiles":
        valid = len(shape) == 4 and shape[-1] == 3 and min(shape[1:3]) > 0 and dtype == np.uint8
    else:
        valid = (
            dataset.dtype.subdtype is None
            and len(shape) == 2 and shape[1] > 0 and np.issubdtype(dtype, np.number)
        )
    if not valid:
        raise ValueError(f"{file.filename}: invalid {kind} shape/dtype: {shape}, {dtype}.")
    return dataset


def feature_shape(path: Path) -> tuple[int, int]:
    """Read only an embedding matrix header."""
    with h5py.File(path, "r") as file:
        shape = array_dataset(file, "features").shape
        return int(shape[0]), int(shape[1])
