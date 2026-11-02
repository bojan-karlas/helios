"""Synthetic fixture cohort generator.

Builds a tiny, fake cohort (no real data) so the pipeline WIRING can be
exercised end-to-end in tests. Produces ``metadata/image_metadata.csv`` and
placeholder ``wsi/<image_id>.svs`` files under a given root.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from helios.data.cohort import ResolvedDataset
from helios.data.resolver import StorageDirs
from helios.stages._runtime import cohort_store


def storage_dirs(dataset: str | Path | ResolvedDataset) -> StorageDirs:
    root = dataset.root if isinstance(dataset, ResolvedDataset) else Path(dataset)
    return StorageDirs(
        preprocessing_dir=root / "preprocessing", models_dir=root / "models", results_dir=root / "results",
    )


def prepared_store(dataset: str | Path | ResolvedDataset):
    return cohort_store(dataset, storage_dirs(dataset))


def make_cohort(root: str | Path, n_images: int = 4, n_patients: int = 2) -> Path:
    root = Path(root)
    (root / "wsi").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "dataset.yaml").write_text(yaml.safe_dump({"id": root.name, "name": f"Example {root.name}"}))

    rows = []
    for i in range(n_images):
        image_id = f"FAKE-SKCM-P{i:05d}-S01-B1-I1"
        (root / "wsi" / f"{image_id}.svs").write_bytes(b"FAKE-SVS")
        rows.append(
            {
                "image_id": image_id,
                "patient_id": f"P{i % n_patients:05d}",
                "image_mpp": 0.25,
                "image_width": 1024,
                "image_height": 1024,
                "image_magnification": 40,
                "patient_sex": ["male", "female"][i % 2],
                "disease_age_at_diagnosis": 50 + i,
                "disease_pfs_recurred": i % 2,
                "disease_pfs_recurred_5yfu": i % 2,
                "disease_pfs_recurred_days": 365 * (i + 1),
                "path_thickness": 1.0 + i * 0.5,
                "path_ulceration_present": i % 2,
                "path_stage": ["I", "II", "III", "IV"][i % 4],
                "sample_tissue_site": "skin",
            }
        )
    meta_path = root / "metadata" / "image_metadata.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    return root


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "var/fixtures/cohort"
    print(make_cohort(out))
