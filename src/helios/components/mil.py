"""Components: whole-slide A-MIL (alignment + attention + pooling + classifier).

* :func:`train_mil` — fit one A-MIL model on a set of training slides.
* :func:`infer_mil` — score ONE slide with a fitted model.
* :func:`reduce_whole_image_risk` — collapse the K per-fold whole-image scores to
  one shipped value per image (out-of-fold on a CV cohort / ensemble on deploy).

Tile features are large (tens of thousands of tiles per slide), so training and
inference take *resolved paths* and open them lazily (per the streaming seam,
:meth:`helios.data.io.ArtifactStore.path`) rather than receiving fully-loaded
arrays. The owning stages live in :mod:`helios.stages.mil`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from helios.models.base import Model


@dataclass(frozen=True)
class SlideFeatureRef:
    """A training slide's resolved feature paths + label (opened lazily)."""

    image_id: str
    features_path: Path
    background_path: Path | None = None
    aug_features_path: Path | None = None
    label: object | None = None


@dataclass(frozen=True)
class MilSlideOutputs:
    """Per-slide A-MIL inference products."""

    patch_embeddings: NDArray[np.float32]
    attention: pd.DataFrame
    slide_embedding: NDArray[np.float32]
    risk_score: float


def train_mil(
    slides: list[SlideFeatureRef],
    *,
    aug_swap_prob: float = 0.5,
    device: str = "cuda",
) -> Model:
    """Fit one A-MIL model on a fold's training slides.

    GRAIN: one fold (all of its train slides) per call. Tiles are streamed from
    each slide's ``features_path`` (background tiles excluded via
    ``background_path``). With probability ``aug_swap_prob`` a tile's normal
    embedding is swapped for its stain-augmented counterpart
    (``aug_features_path``).

    Returns the fitted :class:`~helios.models.base.Model` (persisted as a bundle).
    """
    raise NotImplementedError(
        "Train A-MIL (alignment + attention + weighted pooling + classifier) on the fold's slides."
    )


def infer_mil(
    model: Model,
    *,
    features_path: Path,
    background_path: Path | None = None,
    device: str = "cuda",
) -> MilSlideOutputs:
    """Score ONE slide with a fitted A-MIL model.

    GRAIN: one slide per call; tiles are streamed from ``features_path``
    (background excluded via ``background_path``).
    """
    raise NotImplementedError(
        "Run A-MIL forward pass: aligned patch/slide embeddings, attention, whole-image risk."
    )


def reduce_whole_image_risk(
    per_fold_scores: pd.DataFrame,
    cv_splits: pd.DataFrame | None,
) -> pd.DataFrame:
    """Collapse K per-fold whole-image scores to one row per image.

    Out-of-fold (each image scored by its test-fold model) on a CV cohort;
    ensemble-average when ``cv_splits`` is ``None`` (deploy). Emits a ``fold``
    provenance column (test fold; null on deploy).
    """
    raise NotImplementedError(
        "Reduce per-fold whole-image scores OOF (CV) / ensemble (deploy) to one row per image."
    )
