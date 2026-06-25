"""Components: cell detection & classification pipeline.

Three pure stages of the cell pipeline, all inference-only against pretrained
external models:

* :func:`segment_cells` — CellViT++ instance segmentation + typing.
* :func:`extract_tumor_patches` — cut patches centred on tumor cells.
* :func:`classify_mitoses` — OMG-Net mitotic-figure classification.

The owning stage (:mod:`helios.stages.cells`) chains these per slide and writes
the ``cells``, ``tumor_cell_patches`` and ``mitotic_cells`` artifacts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def segment_cells(
    wsi_path: str,
    *,
    device: str = "cuda",
) -> pd.DataFrame:
    """Instance-segment and type every cell in ONE slide (CellViT++).

    GRAIN: one whole-slide image per call. CellViT applies its own internal
    background detection, so no tile/background input is needed.

    Returns
    -------
    pandas.DataFrame
        One row per cell: ``cell_id, x, y, cell_type`` (tumor / immune /
        connective / epithelial / other).
    """
    raise NotImplementedError("Run pretrained CellViT++ and return per-cell coordinates + type.")


def extract_tumor_patches(
    wsi_path: str,
    cells: pd.DataFrame,
    *,
    patch_px: int = 64,
) -> tuple[pd.DataFrame, NDArray[np.uint8]]:
    """Cut square patches centred on each tumor cell for mitotic classification.

    GRAIN: one slide per call.

    Returns
    -------
    tuple
        ``(patch_index, patches)`` where ``patch_index`` has one row per tumor
        cell (``cell_id, x, y``) and ``patches`` is the matching
        ``(n_tumor, patch_px, patch_px, 3)`` uint8 stack.
    """
    raise NotImplementedError("Cut tumor-centred patches from the WSI for mitotic classification.")


def classify_mitoses(
    patches: NDArray[np.uint8],
    patch_index: pd.DataFrame,
    *,
    device: str = "cuda",
) -> pd.DataFrame:
    """Flag mitotic figures among tumor-cell patches (OMG-Net).

    GRAIN: one slide's tumor patches per call. Inference only.

    Returns
    -------
    pandas.DataFrame
        One row per tumor cell: ``cell_id, is_mitotic, mitotic_score``.
    """
    raise NotImplementedError("Run pretrained OMG-Net over tumor patches to flag mitotic figures.")
