"""Stage: cell detection & classification (``helios prep cells``).

Runs the full cell pipeline per slide: CellViT++ segmentation (``cells``) →
tumor-cell patch extraction (``tumor_cell_patches``) → OMG-Net mitotic
classification (``mitotic_cells``). All three external models are inference-only.
"""
from __future__ import annotations

from pathlib import Path

from helios.components.cells import classify_mitoses, extract_tumor_patches, segment_cells
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    DatasetArgs,
    cohort_store,
    resolve_datasets,
    select_image_ids,
    single_output_guard,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_PATCH_PX: int = 64
DEFAULT_DEVICE: str = "cuda"


def prep_cells(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    patch_px: int = DEFAULT_PATCH_PX,
    device: str = DEFAULT_DEVICE,
    image_ids: list[str] | None = None,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Segment cells, cut tumor patches, and classify mitoses for every slide.

    GRAIN CONTRACT: one image per work unit; the three components are chained
    per slide. The mitotic step always re-runs when cells were re-segmented.
    """
    resolved = resolve_datasets(datasets)
    single_output_guard(resolved, output_root)

    for ds in resolved:
        cohort = ds.dataset()
        store = cohort_store(ds.root, output_root)
        ids = select_image_ids(ds, store, image_ids)

        for image_id in progress.task(ids, desc=f"cells {ds.name}"):
            wsi_path = str(cohort.slides[image_id])
            if not force and store.exists("mitotic_cells", image_id=image_id):
                progress.log(f"skip cells image={image_id} (exists)")
                continue

            cells = segment_cells(wsi_path, device=device)
            store.write("cells", cells, image_id=image_id)

            patch_index, patches = extract_tumor_patches(wsi_path, cells, patch_px=patch_px)
            store.write("tumor_cell_patches", {"patches": patches}, image_id=image_id)

            mitotic = classify_mitoses(patches, patch_index, device=device)
            store.write("mitotic_cells", mitotic, image_id=image_id)
            progress.log(f"wrote cells + mitotic image={image_id}")
