"""Stage: cross-validation split generation (``helios prep splits``).

Reads a cohort's ``image_metadata`` and writes its ``cv_splits`` table. Train-only
and fold-free (it *creates* the folds), so it runs once per training cohort.
"""
from __future__ import annotations

from pathlib import Path

from helios.components.splits import generate_cv_splits
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    DatasetArgs,
    cohort_store,
    resolve_datasets,
    single_output_guard,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_N_FOLDS: int = 5
DEFAULT_VAL_FRACTION: float = 0.0
DEFAULT_STRATIFY_BY: list[str] = ["disease_pfs_recurrence_5yfu"]
DEFAULT_GROUP_BY: str | None = "patient_id"
DEFAULT_SEED: int = 0


def prep_splits(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    stratify_by: list[str] = DEFAULT_STRATIFY_BY,
    group_by: str | None = DEFAULT_GROUP_BY,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Generate ``cv_splits`` for each training cohort.

    GRAIN CONTRACT: one whole cohort per work unit; :func:`generate_cv_splits` is
    called once per dataset.
    """
    resolved = resolve_datasets(datasets)
    single_output_guard(resolved, output_root)

    for ds in progress.task(resolved, desc="prep splits"):
        cohort = ds.dataset()
        store = cohort_store(ds.root, output_root)
        if not force and store.exists("cv_splits"):
            progress.log(f"skip cv_splits {ds.name} (exists)")
            continue
        splits = generate_cv_splits(
            cohort.image_metadata,
            n_folds=n_folds,
            val_fraction=val_fraction,
            stratify_by=stratify_by,
            group_by=group_by,
            seed=seed,
        )
        store.write("cv_splits", splits)
        progress.log(f"wrote cv_splits {ds.name}")
