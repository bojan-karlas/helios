"""Components: cellular composition analysis.

* :func:`aggregate_tile_cells` — bin detected cells into tiles and count per
  type (optional ``slice_id`` debris removal), one row per tile.
* :func:`compute_cellular_features` — reduce per-tile counts to per-slide
  cellular features (eTIL, mitosis-to-tumor ratio, TIL/mitotic coverage).

The owning stage (:mod:`helios.stages.cell_features`) writes the
``tile_cell_counts`` and ``cellular_features`` artifacts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Raw PanNuke class names (as emitted by CellViT++) treated as tumor / TIL.
TUMOR_CLASS = "Neoplastic"
LYMPHOCYTE_CLASS = "Inflammatory"

# Discard any tissue slice smaller than this fraction of the largest slice.
DEFAULT_MIN_CLUSTER_FRAC: float = 0.10

def aggregate_tile_cells(
    cells: pd.DataFrame,
    mitotic: pd.DataFrame,
    tile_meta: pd.DataFrame,
    *,
    tumor_class: str = TUMOR_CLASS,
    lymphocyte_class: str = LYMPHOCYTE_CLASS,
) -> pd.DataFrame:
    """Bin detected cells into tiles and count per type. One row per tile.

    GRAIN: one (size_mm, image) grid per call. Produces ``tile_cell_counts``:
    tile identity from ``tile_metadata`` plus ``n_cells, n_tumor, n_lymph,
    n_mitosis``.

    ``cells`` and ``mitotic`` are assumed post-removal: the cell stage runs
    ``cluster_based_filtering`` before patch extraction / mitosis scoring, and
    the feature stage applies the same filter to ``cells`` on read. No debris
    removal happens here.

    ``n_cells`` is every detected cell type counted once; ``n_mitosis`` is
    reported alongside but never folded into ``n_cells``.
    """
    id_cols = [c for c in ("tile_row", "tile_col", "tile_x", "tile_y", "tile_width", "tile_height", "size_mm")
               if c in tile_meta.columns]
    out = tile_meta[id_cols].reset_index(drop=True).copy()
    w = int(tile_meta["tile_width"].iloc[0])
    h = int(tile_meta["tile_height"].iloc[0])
    x0 = int(tile_meta["tile_x"].min())
    y0 = int(tile_meta["tile_y"].min())

    def _binned_counts(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame({"tile_x": [], "tile_y": [], name: []})
        cx = df["x"].to_numpy(np.float64)
        cy = df["y"].to_numpy(np.float64)
        tx = x0 + (np.floor((cx - x0) / w).astype(np.int64) * w)
        ty = y0 + (np.floor((cy - y0) / h).astype(np.int64) * h)
        return pd.DataFrame({"tile_x": tx, "tile_y": ty}).value_counts().rename(name).reset_index()

    tumor = cells[cells["cell_type"] == tumor_class]
    lymph = cells[cells["cell_type"] == lymphocyte_class]
    mito = mitotic[mitotic["is_mitotic"]] if "is_mitotic" in mitotic else mitotic.iloc[:0]

    for sub, name in [(cells, "n_cells"), (tumor, "n_tumor"), (lymph, "n_lymph"), (mito, "n_mitosis")]:
        out = out.merge(_binned_counts(sub, name), on=["tile_x", "tile_y"], how="left")
        out[name] = out[name].fillna(0).astype(np.int64)

    return out

def compute_cellular_features(
    tile_counts: pd.DataFrame,
    *,
    tissue_min_cells: int = 64,
    tumor_percentile: float = 75.0,
    lymph_frac_threshold: float = 0.05,
    mitosis_min_count: int = 1,
) -> pd.DataFrame:
    """Per-slide cellular features from per-tile counts. One row.

    Whole-slide abundance ratios (over all detected cells of the slide):
      * eTIL = n_lymph / (n_lymph + n_tumor)          [Aung 2025]
      * MTR  = n_mitosis / n_tumor

    Patch-based coverage (0.25 mm tiles; per-tile counting denoises messy
    per-cell typing). Tiles with >= ``tissue_min_cells`` total cells (``n_cells``)
    are tissue tiles (~8x8 densely packed cells in a 250 um tile). Among tissue
    tiles, using per-tile fractions of ``n_cells``:
      * tumor tile   : tumor fraction >= the ``tumor_percentile`` (75th) of the
                       slide's per-tile tumor-fraction distribution;
      * lymph tile   : lymphocyte fraction > ``lymph_frac_threshold`` (5%);
      * mitotic tile : > ``mitosis_min_count`` (1) mitotic tumor cells.
    Then, over tumor tiles:
      * TIL Coverage     = (tumor tiles that are lymph-infiltrated) / (tumor tiles)
      * Mitotic Coverage = (tumor tiles that are mitotic)          / (tumor tiles)

    ``n_cells`` must be the total cells of every type per tile (see
    :func:`aggregate_tile_cells`). If the column is absent, it falls back to
    summing available per-type count columns, which is only correct when all
    types are present.

    Returns
    -------
    pandas.DataFrame
        Single row: ``n_tumor, n_lymph, n_mitosis, etil, mitosis_tumor_ratio,
        til_coverage, mitotic_coverage``. Values are NaN when their denominator
        is zero (no tumor+lymph cells / no tumor cells / no tissue or tumor tiles).
    """
    def _total(col: str) -> int:
        return int(tile_counts[col].sum()) if col in tile_counts.columns and len(tile_counts) else 0

    n_tumor = _total("n_tumor")
    n_lymph = _total("n_lymph")
    n_mitosis = _total("n_mitosis")

    etil_denom = n_lymph + n_tumor
    etil = n_lymph / etil_denom if etil_denom > 0 else float("nan")
    mitosis_tumor_ratio = n_mitosis / n_tumor if n_tumor > 0 else float("nan")

    til_coverage = mitotic_coverage = float("nan")
    if len(tile_counts):
        df = tile_counts
        if "n_cells" in df.columns:
            n_cells = df["n_cells"].to_numpy(np.float64)
        else:
            tcols = [c for c in ("n_tumor", "n_lymph", "n_connective", "n_epithelial", "n_dead")
                     if c in df.columns]
            n_cells = df[tcols].sum(axis=1).to_numpy(np.float64)
        tumor = df["n_tumor"].to_numpy(np.float64)
        lymph = df["n_lymph"].to_numpy(np.float64)
        mito = df["n_mitosis"].to_numpy(np.float64) if "n_mitosis" in df.columns else np.zeros(len(df))

        tissue = n_cells >= tissue_min_cells
        if tissue.any():
            with np.errstate(invalid="ignore", divide="ignore"):
                tumor_frac = np.where(n_cells > 0, tumor / n_cells, np.nan)
                lymph_frac = np.where(n_cells > 0, lymph / n_cells, np.nan)
            tf = tumor_frac[tissue]
            tf = tf[np.isfinite(tf)]
            if tf.size:
                tu_thr = np.percentile(tf, tumor_percentile)          # per-slide 75th pct
                tumor_tile = tissue & (tumor_frac >= tu_thr)
                n_tumor_tiles = int(tumor_tile.sum())
                if n_tumor_tiles > 0:
                    lymph_tile = tumor_tile & (lymph_frac > lymph_frac_threshold)
                    mito_tile = tumor_tile & (mito > mitosis_min_count)
                    til_coverage = int(lymph_tile.sum()) / n_tumor_tiles
                    mitotic_coverage = int(mito_tile.sum()) / n_tumor_tiles

    return pd.DataFrame([{
        "n_tumor": n_tumor,
        "n_lymph": n_lymph,
        "n_mitosis": n_mitosis,
        "etil": etil,
        "mitosis_tumor_ratio": mitosis_tumor_ratio,
        "til_coverage": til_coverage,
        "mitotic_coverage": mitotic_coverage,
    }])