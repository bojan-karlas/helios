"""Dataset input contract — the lightweight promise a cohort must satisfy.

A HELIOS dataset is, at minimum:

* a flat ``wsi/`` set of whole-slide images, and
* a ``metadata/image_metadata.csv`` conforming to the data dictionary
  (one row per ``image_id``).

This module performs *light* structural checks (presence + required id column),
not heavy validation — that belongs to the config/datadict layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

IMAGE_ID = "image_id"


@dataclass(frozen=True)
class DatasetSpec:
    """Resolved location of a cohort's required inputs."""

    root: Path
    image_metadata_path: Path
    wsi_dir: Path

    @classmethod
    def from_root(cls, root: str | Path) -> DatasetSpec:
        root = Path(root)
        return cls(
            root=root,
            image_metadata_path=root / "metadata" / "image_metadata.csv",
            wsi_dir=root / "wsi",
        )

    def check(self) -> None:
        """Light presence + schema checks; raises on a broken contract."""
        if not self.image_metadata_path.exists():
            raise FileNotFoundError(f"Missing image metadata: {self.image_metadata_path}")
        if not self.wsi_dir.exists():
            raise FileNotFoundError(f"Missing wsi/ directory: {self.wsi_dir}")
        cols = pd.read_csv(self.image_metadata_path, nrows=0).columns
        if IMAGE_ID not in cols:
            raise ValueError(f"image_metadata.csv must contain an '{IMAGE_ID}' column.")

    def image_metadata(self) -> pd.DataFrame:
        return pd.read_csv(self.image_metadata_path)
