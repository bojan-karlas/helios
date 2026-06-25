"""Stage: slide thumbnail extraction (``helios prep thumbnails``).

Reads each cohort image's WSI and writes the ``thumbnail`` artifact.
"""
from __future__ import annotations

from pathlib import Path

from helios.components.thumbnails import extract_thumbnail
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    DatasetArgs,
    cohort_store,
    resolve_datasets,
    select_image_ids,
    single_output_guard,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_MAX_SIZE: int = 2048


def prep_thumbnails(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    max_size: int = DEFAULT_MAX_SIZE,
    image_ids: list[str] | None = None,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Extract a thumbnail for every image in each cohort.

    GRAIN CONTRACT: one image per work unit; :func:`extract_thumbnail` is called
    once per slide.
    """
    resolved = resolve_datasets(datasets)
    single_output_guard(resolved, output_root)

    for ds in resolved:
        cohort = ds.dataset()
        store = cohort_store(ds.root, output_root)
        ids = select_image_ids(ds, store, image_ids)

        for image_id in progress.task(ids, desc=f"thumbnails {ds.name}"):
            if not force and store.exists("thumbnail", image_id=image_id):
                progress.log(f"skip thumbnail image={image_id} (exists)")
                continue
            thumb = extract_thumbnail(str(cohort.slides[image_id]), max_size=max_size)
            store.write("thumbnail", thumb, image_id=image_id)
            progress.log(f"wrote thumbnail image={image_id}")
