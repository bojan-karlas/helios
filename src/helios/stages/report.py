"""Stage: case + cohort reporting (``helios report``).

Builds the web-app data package per cohort: one ``report.json`` per image
(source of truth), optional rendered PNG assets, the rendered ``report.md``, and
the wide ``cohort_summary.csv``. ``report`` has no subcommands — it runs the full
report build; ``--no-assets`` / ``--no-summarize`` trim optional outputs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from helios.components.report import (
    build_case_report,
    render_assets,
    render_markdown,
    summarize_cohort,
)
from helios.data.io import ArtifactStore
from helios.progress import DEFAULT_PROGRESS, Progress
from helios.stages._runtime import (
    DatasetArgs,
    cohort_store,
    resolve_datasets,
    select_image_ids,
    single_output_guard,
)

# -- configurable defaults (source of truth for configs/default.yaml) ----------
DEFAULT_ASSETS: bool = True
DEFAULT_SUMMARIZE: bool = True

#: Cohort-level result tables sliced per image for the case report / summary.
_CASE_TABLES = (
    "helios_risk_score",
    "whole_image_risk_score_ensemble",
    "path_concept_predictions",
    "cellular_features",
    "patch_morphology_presence",
)
_SUMMARY_TABLES = (
    "helios_risk_score",
    "cellular_risk_score",
    "path_concept_risk_score",
    "patch_morphology_risk_score",
    "staging_risk_score",
    "clinical_risk_score",
    "whole_image_risk_score_ensemble",
    "cellular_features",
    "path_concept_predictions",
    "patch_morphology_presence",
)


def build_report(
    *,
    datasets: DatasetArgs,
    output_root: str | Path | None = None,
    assets: bool = DEFAULT_ASSETS,
    summarize: bool = DEFAULT_SUMMARIZE,
    image_ids: list[str] | None = None,
    force: bool = False,
    progress: Progress = DEFAULT_PROGRESS,
) -> None:
    """Assemble per-case reports and the cohort summary for each cohort.

    GRAIN CONTRACT: per-image case report (+ optional assets + markdown), then
    one cohort-level summary join. Reads the shipped result tables; missing
    optional inputs are simply skipped by the report components.
    """
    resolved = resolve_datasets(datasets)
    single_output_guard(resolved, output_root)

    for ds in resolved:
        store = cohort_store(ds.root, output_root)
        ids = select_image_ids(ds, store, image_ids)
        tables = _read_tables(store, _CASE_TABLES)

        for image_id in progress.task(ids, desc=f"report {ds.name}"):
            if not force and store.exists("case_report_md", image_id=image_id):
                progress.log(f"skip report image={image_id} (exists)")
                continue
            case_inputs = {name: _slice(table, image_id) for name, table in tables.items()}
            report = build_case_report(image_id, case_inputs)
            store.write("case_report", report, image_id=image_id)

            asset_ids: list[str] = []
            if assets:
                rendered = render_assets(report, case_inputs)
                for asset_id, image in rendered.items():
                    store.write("case_assets", image, image_id=image_id, asset_id=asset_id)
                asset_ids = list(rendered)

            store.write("case_report_md", render_markdown(report, asset_ids), image_id=image_id)
            progress.log(f"wrote report image={image_id}")

        if summarize:
            summary = summarize_cohort(
                {**_read_tables(store, _SUMMARY_TABLES), "image_metadata": store.read("image_metadata")}
            )
            store.write("cohort_summary", summary)
            progress.log(f"wrote cohort_summary {ds.name}")


def _read_tables(store: ArtifactStore, names: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Read each cohort-level result table that exists (optional inputs may be absent)."""
    return {name: store.read(name) for name in names if store.exists(name)}


def _slice(table: pd.DataFrame, image_id: str) -> pd.DataFrame:
    if "image_id" not in table.columns:
        return table
    return table[table["image_id"].astype(str) == image_id]
