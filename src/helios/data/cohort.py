"""Named, versioned cohort definitions.

A *cohort definition* is a small YAML naming a union of dataset roots plus an
optional row filter, so a training / evaluation population can be defined once,
reused, and version-controlled. It is accepted anywhere ``--dataset`` is::

    helios fit mil --dataset cohorts/train.yaml

.. code-block:: yaml

    # cohorts/train.yaml
    name: TRAIN_COHORT
    datasets:
      - /data/COHORT1
      - source_root: ../cohorts/COHORT2
    filter: "path_stage in ['I', 'II']"   # optional pandas-query over image_metadata

A bare path (``--dataset /data/COHORT1``) is just a single-root cohort with no
filter, so cohort files and plain roots are interchangeable and composable
(repeated ``--dataset`` flags still union). Resolution flattens every input into
a list of :class:`ResolvedDataset` -- one physical root plus the effective row
filter selecting its population. Filters compose with logical *and* (a cohort
file's ``filter`` and an ad-hoc ``--filter`` both apply).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from helios.data.dataset import Dataset

COHORT_SUFFIXES = (".yaml", ".yml")


@dataclass(frozen=True)
class ResolvedDataset:
    """One physical dataset root plus the row filter selecting its population.

    The unit a stage iterates over: ``root`` locates the source on disk, ``where``
    is an optional pandas-``query`` over ``image_metadata`` restricting which
    images participate, and ``name`` (the root directory name) is the provenance
    label stamped onto result tables. An optional preprocessing root applies to
    this constituent's derived data. Augmentations refer to prepared directories.
    """

    root: Path
    where: str | None = None
    preprocessing_root: Path | None = None
    augmentations: tuple[Path, ...] = ()
    cohort_name: str | None = None
    cohort_file: Path | None = None
    splits_file: Path | None = None

    @property
    def name(self) -> str:
        return self.root.name or self.root.resolve().name

    @property
    def logical_name(self) -> str:
        return self.cohort_name or self.name

    def dataset(self) -> Dataset:
        """The lazy :class:`~helios.data.dataset.Dataset` bound to this root."""
        return Dataset(self.root)

    def with_filter(self, where: str | None) -> ResolvedDataset:
        """Return a copy with ``where`` AND-combined onto the existing filter."""
        if not where:
            return self
        return replace(self, where=combine_filters(self.where, where))


@dataclass(frozen=True)
class CohortSpec:
    """A parsed cohort-definition file: a named union of roots + optional filter."""

    name: str
    datasets: tuple[ResolvedDataset, ...]
    filter: str | None = None
    splits_file: Path | None = None
    cohort_file: Path | None = None


def is_cohort_file(arg: str | Path | ResolvedDataset) -> bool:
    """True when ``arg`` points at an existing ``.yaml`` / ``.yml`` cohort file."""
    if isinstance(arg, ResolvedDataset):
        return False
    path = Path(arg)
    return path.suffix.lower() in COHORT_SUFFIXES and path.is_file()


def load_cohort_spec(path: str | Path) -> CohortSpec:
    """Parse a cohort-definition YAML, resolving roots relative to the file."""
    cohort_path = Path(path).absolute()
    data = yaml.safe_load(cohort_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Cohort file {path} must contain a mapping.")
    unknown = set(data) - {"name", "datasets", "roots", "filter", "splits"}
    if unknown:
        raise ValueError(f"Unknown cohort fields in {path}: {sorted(unknown)}")
    entries = data.get("datasets") or data.get("roots")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Cohort file {path} must list one or more 'datasets'.")
    if data.get("filter") is not None and not isinstance(data["filter"], str):
        raise ValueError(f"Cohort filter in {path} must be a string.")
    base = cohort_path.parent
    cohort_name = str(data.get("name", cohort_path.stem))

    def location(value: object) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Dataset roots in {path} must be nonempty path strings.")
        p = Path(value)
        return p if p.is_absolute() else base / p

    splits_file: Path | None = None
    if data.get("splits") is not None:
        splits_file = location(data["splits"])
    # Constituents retain only explicit selections; implicit absence permits no-CV training.

    resolved: list[ResolvedDataset] = []
    for entry in entries:
        if isinstance(entry, str):
            resolved.append(
                ResolvedDataset(
                    location(entry),
                    cohort_name=cohort_name,
                    cohort_file=cohort_path,
                    splits_file=splits_file,
                )
            )
        elif isinstance(entry, dict):
            if "features_root" in entry:
                raise ValueError(
                    "features_root has been removed; move embeddings to preprocessing_root/tile_features "
                    "and use preprocessing_root instead."
                )
            unknown = set(entry) - {"source_root", "preprocessing_root", "augmentations"}
            if unknown:
                raise ValueError(f"Unknown dataset fields in {path}: {sorted(unknown)}")
            raw_augs = entry.get("augmentations", [])
            if not isinstance(raw_augs, list):
                raise ValueError(f"Dataset 'augmentations' in {path} must be a list.")
            if any(not isinstance(a, str) or not a.strip() for a in raw_augs):
                raise ValueError(
                    f"Dataset augmentations in {path} must be directory path strings, not transformation specs."
                )
            augs = tuple(dict.fromkeys(location(a) for a in raw_augs))
            resolved.append(ResolvedDataset(
                root=location(entry.get("source_root")),
                preprocessing_root=location(entry["preprocessing_root"]) if "preprocessing_root" in entry else None,
                augmentations=augs,
                cohort_name=cohort_name,
                cohort_file=cohort_path,
                splits_file=splits_file,
            ))
        else:
            raise ValueError(f"Dataset entries in {path} must be paths or root mappings.")
    return CohortSpec(
        name=cohort_name,
        datasets=tuple(resolved),
        filter=data.get("filter"),
        splits_file=splits_file if splits_file is not None else cohort_path.with_suffix(".splits.csv"),
        cohort_file=cohort_path,
    )


def resolve_datasets(
    items: Iterable[str | Path | ResolvedDataset],
    *,
    where: str | None = None,
) -> list[ResolvedDataset]:
    """Flatten ``--dataset`` inputs into the list of :class:`ResolvedDataset`.

    Each item is either a cohort file (expanded to its member roots, carrying the
    file's ``filter``), a plain root, or an already-resolved dataset (passed
    through). An extra ``where`` (e.g. an ad-hoc ``--filter``) is AND-combined
    onto every result. The function is idempotent on :class:`ResolvedDataset`, so
    stages can safely re-resolve inputs the CLI already resolved.
    """
    out: list[ResolvedDataset] = []
    for item in items:
        if isinstance(item, ResolvedDataset):
            out.append(item.with_filter(where))
        elif is_cohort_file(item):
            spec = load_cohort_spec(item)
            for ds in spec.datasets:
                out.append(ds.with_filter(combine_filters(spec.filter, where)))
        else:
            out.append(ResolvedDataset(Path(item).absolute(), where=where))
    if not out:
        raise ValueError("At least one dataset is required.")
    return out


def combine_filters(a: str | None, b: str | None) -> str | None:
    """AND-combine two pandas-``query`` filter strings (``None`` is identity)."""
    if a and b:
        return f"({a}) and ({b})"
    return a or b


#: What ``--dataset`` accepts: a root path, a cohort file, or a resolved dataset.
DatasetArg = str | Path | ResolvedDataset
#: A sequence of :data:`DatasetArg` (a stage's ``datasets`` parameter).
DatasetArgs = Sequence[DatasetArg]
