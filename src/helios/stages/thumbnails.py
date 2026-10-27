"""Stage: slide thumbnail extraction (``helios prep thumbnails``).

Reads each cohort image's WSI and writes the ``thumbnail`` artifact.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path

from helios.components.thumbnails import extract_thumbnail
from helios.data.io import ArtifactStore
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    DatasetArgs,
    ParallelConfig,
    StorageRoots,
    WorkContext,
    cohort_store,
    resolve_datasets,
    run_work_units,
    select_image_ids,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_MAX_SIZE: int = 2048


def prep_thumbnails(
    *,
    datasets: DatasetArgs,
    roots: StorageRoots | None = None,
    parallel: ParallelConfig | None = None,
    max_size: int = DEFAULT_MAX_SIZE,
    image_ids: list[str] | None = None,
    overwrite: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Extract a thumbnail for every image in each cohort.

    GRAIN CONTRACT: one image per work unit; :func:`extract_thumbnail` is called
    once per slide.
    """
    for ds in resolve_datasets(datasets, roots=roots):
        store = cohort_store(ds, roots)
        run_work_units(
            select_image_ids(ds, store, image_ids),
            partial(_thumbnail, store=store, slides=ds.dataset().slides, max_size=max_size, overwrite=overwrite),
            stage=f"thumbnails {ds.name}",
            progress=progress,
            parallel=parallel,
        )


def _thumbnail(
    image_id: str,
    ctx: WorkContext,
    *,
    store: ArtifactStore,
    slides: Mapping[str, Path],
    max_size: int,
    overwrite: bool,
) -> str:
    if not overwrite and store.exists("thumbnail", image_id=image_id):
        ctx.progress.log(f"skip thumbnail image={image_id} (exists)")
        return "skipped"
    thumb = extract_thumbnail(str(slides[image_id]), max_size=max_size)
    store.write("thumbnail", thumb, image_id=image_id)
    ctx.progress.log(f"wrote thumbnail image={image_id}")
    return "completed"
