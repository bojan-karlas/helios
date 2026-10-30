"""The HELIOS cohort dataset -- locates and lazily materializes a cohort.

A HELIOS dataset is, at minimum:

* a ``dataset.yaml`` descriptor with required ``id`` and ``name`` strings,
* a ``wsi/`` tree of whole-slide images, and
* a ``metadata/image_metadata.csv`` conforming to the data dictionary
  (one row per ``image_id``).

:class:`Dataset` binds to a cohort root and exposes its contents *on demand*:
the metadata table, the slide index (``image_id -> wsi path``), and the joined
view are all :func:`functools.cached_property` values, so constructing a
``Dataset`` touches no disk -- work happens the first time you ask for it, then
is cached. Light structural checks live in :meth:`Dataset.check`; heavy column
validation belongs to the config/datadict layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd
import yaml

IMAGE_ID = "image_id"
WSI_SUFFIXES = (
    ".svs", ".tiff", ".tif", ".ndpi", ".mrxs",
    ".avs", ".dcm", ".vms", ".vmu", ".scn", ".svslide", ".bif", ".czi",
)


@dataclass(frozen=True)
class DatasetDescriptor:
    id: str
    name: str
    description: str | None = None

    @classmethod
    def load(cls, root: Path) -> DatasetDescriptor:
        path = root / "dataset.yaml"
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a dataset descriptor mapping.")
        for key in ("id", "name"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise ValueError(f"{path} requires a nonempty string '{key}'.")
        description = data.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError(f"{path}: description must be a string.")
        return cls(id=data["id"], name=data["name"], description=description)


class Dataset:
    """A HELIOS cohort rooted at ``root`` (lazy: nothing is read until used)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"Dataset(root={self.root!r})"

    # -- locations -------------------------------------------------------------
    @property
    def wsi_dir(self) -> Path:
        return self.root / "wsi"

    @property
    def image_metadata_path(self) -> Path:
        return self.root / "metadata" / "image_metadata.csv"

    @cached_property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor.load(self.root)

    # -- structural contract ---------------------------------------------------
    def check(self) -> None:
        """Light presence + schema checks; raises on a broken contract."""
        _ = self.descriptor
        if not self.image_metadata_path.exists():
            raise FileNotFoundError(f"Missing image metadata: {self.image_metadata_path}")
        if not self.wsi_dir.exists():
            raise FileNotFoundError(f"Missing wsi/ directory: {self.wsi_dir}")
        cols = pd.read_csv(self.image_metadata_path, nrows=0).columns
        if IMAGE_ID not in cols:
            raise ValueError(f"image_metadata.csv must contain an '{IMAGE_ID}' column.")

    # -- lazily materialized contents ------------------------------------------
    @cached_property
    def image_metadata(self) -> pd.DataFrame:
        """The cohort's ``image_metadata.csv`` (one row per ``image_id``)."""
        return pd.read_csv(self.image_metadata_path)

    @cached_property
    def image_ids(self) -> list[str]:
        return self.image_metadata[IMAGE_ID].astype(str).tolist()

    @cached_property
    def slides(self) -> dict[str, Path]:
        """``image_id -> wsi path``, discovered by scanning ``wsi/``."""
        return {
            p.stem: p
            for p in self.wsi_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in WSI_SUFFIXES
        }

    @cached_property
    def table(self) -> pd.DataFrame:
        """Image metadata joined with each image's resolved ``wsi_path``."""
        meta = self.image_metadata.copy()
        meta["wsi_path"] = meta[IMAGE_ID].astype(str).map(lambda i: str(self.slides.get(i, "")))
        return meta

    def require_slides(self) -> None:
        """Raise if any image in the metadata has no matching WSI file."""
        missing = self.table.loc[self.table["wsi_path"] == "", IMAGE_ID].tolist()
        if missing:
            raise FileNotFoundError(f"No WSI file found for image_ids: {missing[:10]}")
