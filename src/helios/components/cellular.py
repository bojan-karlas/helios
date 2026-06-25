"""Components: cellular composition analysis.

* :func:`aggregate_tile_cells` — count tumor / immune / mitosis per tile
  (debris removed via DBSCAN) and threshold-classify tiles.
* :func:`compute_cellular_features` — reduce per-tile counts to per-slide
  cellular features (eTIL, mitosis-to-tumor ratio, TIL/mitotic coverage).

The owning stage (:mod:`helios.stages.cell_features`) writes the
``tile_cell_counts`` and ``cellular_features`` artifacts.
"""
from __future__ import annotations

import pandas as pd


def aggregate_tile_cells(
    cells: pd.DataFrame,
    mitotic_cells: pd.DataFrame,
    tile_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Count cells per tile and threshold-classify each tile.

    GRAIN: one (slide, size_mm) per call.

    Parameters
    ----------
    cells:
        Per-cell coordinates + type for the slide.
    mitotic_cells:
        Per-tumor-cell mitotic flags for the slide.
    tile_metadata:
        Per-tile geometry the cells are binned into.

    Returns
    -------
    pandas.DataFrame
        One row per tile: ``tile_id, n_tumor, n_immune, n_mitosis, tile_class``
        (e.g. TIL-dominant / mitosis-dominant).
    """
    raise NotImplementedError(
        "Bin cells into tiles (DBSCAN debris removal), count types, threshold-classify tiles."
    )


def compute_cellular_features(tile_cell_counts: pd.DataFrame) -> pd.DataFrame:
    """Reduce per-tile counts to one row of cellular features for the slide.

    GRAIN: one slide per call.

    Returns
    -------
    pandas.DataFrame
        Single row: ``image_id, etil, mitosis_tumor_ratio, til_coverage,
        mitotic_coverage`` (and related summaries).
    """
    raise NotImplementedError(
        "Compute eTIL, mitosis-to-tumor ratio, TIL coverage, mitotic coverage from tile counts."
    )
