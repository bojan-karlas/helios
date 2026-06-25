"""Component: stain augmentation (CycleGAN inference).

Map a slide's tiles toward a target stain distribution using a pretrained
CycleGAN generator. The generator is TRAINED IN A SEPARATE REPO and loaded here
(per target) — this repo runs inference only, mirroring how the foundation /
cell models are loaded. The owning stage (:mod:`helios.stages.augment`) reads the
``tiles`` artifact and writes ``tile_augmentation``.
"""
from __future__ import annotations

from typing import Literal, get_args

import numpy as np
from numpy.typing import NDArray

#: Stain targets with a pretrained CycleGAN generator available.
StainTarget = Literal["MGB+MRV", "VISIOMEL"]

#: Runtime tuple of supported targets (for selector validation).
STAIN_TARGETS: tuple[str, ...] = get_args(StainTarget)


def augment_stains(
    tiles: NDArray[np.uint8],
    *,
    target: StainTarget,
    device: str = "cuda",
) -> NDArray[np.uint8]:
    """Stain-augment every tile of ONE slide toward ``target``.

    GRAIN: one (slide, size_mm, target) per call.

    Parameters
    ----------
    tiles:
        Tile RGB stack for one slide, ``(n_tiles, H, W, 3)`` uint8.
    target:
        Stain distribution to map toward (selects the pretrained generator).
    device:
        Torch device string.

    Returns
    -------
    NDArray[np.uint8]
        Stain-augmented tiles, same shape/order as ``tiles``.
    """
    raise NotImplementedError(
        f"Load the pretrained CycleGAN generator for target {target!r} and run inference per tile."
    )
