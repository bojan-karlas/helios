"""In-memory image index for a cohort.

Builds a manifest by scanning the (flattened) ``wsi/`` directory and joining it
against ``image_metadata.csv`` on ``image_id``. Downstream stages iterate the
manifest to know which images to process.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from helios.data.spec import IMAGE_ID, DatasetSpec

WSI_SUFFIXES = (".svs", ".tiff", ".tif", ".ndpi")


@dataclass
class Manifest:
    """The cohort's images joined with their metadata (one row per image)."""

    table: pd.DataFrame

    @property
    def image_ids(self) -> list[str]:
        return self.table[IMAGE_ID].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.table)


def build_manifest(dataset: DatasetSpec, require_wsi: bool = True) -> Manifest:
    dataset.check()
    meta = dataset.image_metadata()

    found = {
        p.stem: p
        for p in dataset.wsi_dir.rglob("*")
        if p.suffix.lower() in WSI_SUFFIXES
    }
    meta = meta.copy()
    meta["wsi_path"] = meta[IMAGE_ID].astype(str).map(lambda i: str(found.get(i, "")))

    if require_wsi:
        missing = meta.loc[meta["wsi_path"] == "", IMAGE_ID].tolist()
        if missing:
            raise FileNotFoundError(f"No WSI file found for image_ids: {missing[:10]}")

    return Manifest(table=meta)
