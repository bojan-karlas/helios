"""Stage: cross-validation split generation (``helios prep splits``).

Reads a cohort's ``image_metadata`` and writes its ``cv_splits`` table. Train-only
and fold-free (it *creates* the folds), so it runs once per training cohort.
"""
from __future__ import annotations

import pandas as pd

from helios.components.splits import generate_cv_splits
from helios.data.io import write
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    DatasetArgs,
    ParallelConfig,
    ResolvedDataset,
    StorageRoots,
    cohort_store,
    resolve_datasets,
    resolve_splits_path,
    select_image_ids,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_N_FOLDS: int = 5
DEFAULT_VAL_FRACTION: float = 0.0
DEFAULT_STRATIFY_BY: list[str] = ["disease_pfs_recurrence_5yfu"]
DEFAULT_GROUP_BY: str | None = "patient_id"
DEFAULT_SEED: int = 0


def _non_default_hive(
    *,
    n_folds: int,
    val_fraction: float,
    stratify_by: list[str],
    group_by: str | None,
    seed: int,
) -> dict[str, object]:
    extra: dict[str, object] = {}
    if n_folds != DEFAULT_N_FOLDS:
        extra["n_folds"] = n_folds
    if val_fraction != DEFAULT_VAL_FRACTION:
        extra["val_fraction"] = val_fraction
    if stratify_by != DEFAULT_STRATIFY_BY:
        extra["stratify_by"] = stratify_by
    if group_by != DEFAULT_GROUP_BY:
        extra["group_by"] = group_by if group_by is not None else "none"
    if seed != DEFAULT_SEED:
        extra["seed"] = seed
    return extra


def prep_splits(
    *,
    datasets: DatasetArgs,
    roots: StorageRoots | None = None,
    parallel: ParallelConfig | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    stratify_by: list[str] = DEFAULT_STRATIFY_BY,
    group_by: str | None = DEFAULT_GROUP_BY,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Generate ``cv_splits`` for each logical training cohort.

    GRAIN CONTRACT: one logical cohort per work unit; constituent metadata tables
    are pooled when ``datasets`` resolves to a named multi-dataset cohort.
    """
    resolved = resolve_datasets(datasets, roots=roots)
    extra_hive = _non_default_hive(
        n_folds=n_folds,
        val_fraction=val_fraction,
        stratify_by=stratify_by,
        group_by=group_by,
        seed=seed,
    )

    groups: list[list[ResolvedDataset]] = []
    seen_cohorts: dict[str, list[ResolvedDataset]] = {}
    for ds in resolved:
        if ds.cohort_name is not None:
            if ds.cohort_name not in seen_cohorts:
                bucket: list[ResolvedDataset] = []
                seen_cohorts[ds.cohort_name] = bucket
                groups.append(bucket)
            seen_cohorts[ds.cohort_name].append(ds)
        else:
            groups.append([ds])

    for group in progress.task(groups, desc="prep splits"):
        lead = group[0]
        label = lead.logical_name
        out_path = resolve_splits_path(lead, roots, extra_hive=extra_hive)
        if not force and out_path.exists():
            progress.log(f"skip cv_splits {label} (exists at {out_path})")
            continue
        frames: list[pd.DataFrame] = []
        for ds in group:
            store = cohort_store(ds, roots)
            meta = ds.dataset().image_metadata
            keep = set(select_image_ids(ds, store, None))
            frames.append(meta[meta["image_id"].astype(str).isin(keep)])
        pooled_meta = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        splits = generate_cv_splits(
            pooled_meta,
            n_folds=n_folds,
            val_fraction=val_fraction,
            stratify_by=stratify_by,
            group_by=group_by,
            seed=seed,
        )
        write(splits, out_path, "csv")
        progress.log(f"wrote cv_splits {label} ({out_path})")
