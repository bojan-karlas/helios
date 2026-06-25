"""Components: reporting / web-app data package.

* :func:`build_case_report` — assemble one case's ``report.json`` (headline, risk
  factors, concepts, refs) by reading from the batch result tables.
* :func:`render_assets` — optional pre-rendered PNG overlays (attention heatmap,
  cluster map, TIL density, mitosis overlay).
* :func:`render_markdown` — render ``report.md`` from ``report.json`` (+ assets).
* :func:`summarize_cohort` — build the wide ``cohort_summary.csv`` by joining the
  per-component result tables on ``image_id``.

The owning stage (:mod:`helios.stages.report`) reads/writes the artifacts; these
functions are pure transforms over already-loaded inputs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def build_case_report(image_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Assemble one case's report JSON from the per-image result rows.

    GRAIN: one image per call. ``inputs`` holds that image's slice of each
    consumed result table (risk scores, concepts, cellular features, etc.).

    Returns the ``report.json`` mapping (source of truth for the case).
    """
    raise NotImplementedError("Assemble per-case report.json (headline, risk factors, concepts, refs).")


def render_assets(case_report: dict[str, Any], inputs: dict[str, Any]) -> dict[str, NDArray[np.uint8]]:
    """Render optional PNG overlays for one case.

    GRAIN: one image per call.

    Returns ``asset_id -> RGB image`` (attention heatmap, cluster map, TIL
    density, mitosis overlay); OOF-selects the image's test-fold attention.
    """
    raise NotImplementedError("Render attention/cluster/TIL/mitosis overlays for the case.")


def render_markdown(case_report: dict[str, Any], asset_ids: list[str]) -> str:
    """Render human-readable ``report.md`` from a case's report JSON (+ assets).

    GRAIN: one image per call.
    """
    raise NotImplementedError("Render report.md from report.json (Jinja2), embedding assets.")


def summarize_cohort(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the per-component result tables into the wide cohort summary.

    GRAIN: one whole cohort per call. ``tables`` maps artifact id -> its result
    table; the join key is ``image_id``.

    Returns the wide ``cohort_summary`` table (one row per image).
    """
    raise NotImplementedError("Join result tables (+ image_metadata) on image_id into cohort_summary.")
