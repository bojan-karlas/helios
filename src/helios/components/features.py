"""Component: foundation-model tile (patch) feature extraction.

Embed each tile of a whole-slide image with a pathology foundation model. This
is a *pure* function: the owning stage (``helios.stages.features``) reads the
``tiles`` artifact, calls :func:`extract_features` once per slide, and writes the
resulting embeddings to the ``tile_features`` artifact.

IMPLEMENTER NOTES
-----------------
* Add a new encoder by (1) extending :data:`FeatureModel` with its name and
  (2) adding a branch in :func:`_load_encoder` that loads its weights and
  returns a callable ``batch[uint8, (N, H, W, 3)] -> float32[(N, D)]``.
* Foundation-model weights are loaded HERE, inside the component — never in the
  stage. The stage only decides *which* model(s) to run.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from functools import lru_cache
from typing import Literal, get_args

import numpy as np
from numpy.typing import NDArray

from helios.progress import DEFAULT_PROGRESS, Progress

#: Foundation models with an encoder implemented in this file. To add a model,
#: extend this Literal and add a branch in :func:`_load_encoder`.
FeatureModel = Literal[
    "virchow2",
    "uni",
    "uni2",
    "conch",
    "conch1.5",
    "chief",
    "ctranspath",
    "dino",
    "gigapath",
    "musk",
    "phikon2",
    "resnet-50",
]

#: Runtime tuple of the supported model names (for selector validation).
FEATURE_MODELS: tuple[str, ...] = get_args(FeatureModel)

#: An encoder maps a uint8 tile batch ``(N, H, W, 3)`` to ``float32`` embeddings
#: ``(N, D)`` where ``D`` is the model's embedding dimension.
Encoder = Callable[[NDArray[np.uint8]], NDArray[np.float32]]


def extract_features(
    tiles: NDArray[np.uint8],
    *,
    model: FeatureModel,
    batch_size: int = 64,
    device: str = "cuda",
    progress: Progress = DEFAULT_PROGRESS,
) -> NDArray[np.float32]:
    """Embed every tile of ONE whole-slide image with a foundation model.

    GRAIN: one whole-slide image per call (all of its tiles). Batching across
    tiles is internal (``batch_size``) for GPU efficiency.

    Parameters
    ----------
    tiles:
        Tile image stack for one slide, shape ``(n_tiles, H, W, 3)``, ``uint8``,
        RGB. Row ``i`` is the ``i``-th tile.
    model:
        Which foundation encoder to use (see :data:`FeatureModel`).
    batch_size:
        Number of tiles per forward pass.
    device:
        Torch device string (``"cuda"``, ``"cuda:0"``, ``"cpu"``).
    progress:
        Progress sink for the per-batch loop (default: no-op).

    Returns
    -------
    NDArray[np.float32]
        Tile embeddings of shape ``(n_tiles, D)``; row ``i`` corresponds to
        ``tiles[i]``. ``D`` is model-dependent.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    encoder = _load_encoder(model, device=device)
    n = int(len(tiles))
    n_batches = (n + batch_size - 1) // batch_size if batch_size > 0 else 0
    chunks: list[NDArray[np.float32]] = []
    for start in progress.task(
        range(0, n, batch_size), total=n_batches, desc=f"{model} embed"
    ):
        batch = np.asarray(tiles[start : start + batch_size], dtype=np.uint8)
        chunks.append(np.asarray(encoder(batch), dtype=np.float32))
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


@lru_cache(maxsize=None)
def _load_encoder(model: FeatureModel, *, device: str) -> Encoder:
    """Load the encoder callable for ``model`` (weights loaded here, not in the stage)."""
    if model == "virchow2":
        return _load_virchow2(device=device)
    raise NotImplementedError(
        f"Encoder {model!r} is not implemented yet. Load its weights here and "
        f"return a callable mapping uint8[N,H,W,3] tiles to float32[N,D] embeddings."
    )


def _load_virchow2(*, device: str) -> Encoder:
    """Load Paige Virchow2 and return its official 2,560-dim tile encoder.

    The Hugging Face repository is gated. Users must accept its terms and make
    a token available through ``huggingface-cli login`` or ``HF_TOKEN`` before
    the first invocation. Weights are cached by Hugging Face and this encoder
    is cached by :func:`_load_encoder` for reuse across slides in one process.
    """
    import timm
    import torch
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        resolved_device = torch.device("cpu")

    backbone = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    ).eval().to(resolved_device)
    transform = create_transform(**resolve_data_config(backbone.pretrained_cfg, model=backbone))

    def encode(batch: NDArray[np.uint8]) -> NDArray[np.float32]:
        if batch.ndim != 4 or batch.shape[-1] != 3:
            raise ValueError("Virchow2 input must have shape (N, H, W, 3)")
        if len(batch) == 0:
            return np.empty((0, 2560), dtype=np.float32)
        inputs = torch.stack([transform(Image.fromarray(tile, mode="RGB")) for tile in batch]).to(resolved_device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if resolved_device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            tokens = backbone(inputs)
            if tokens.ndim != 3 or tokens.shape[1] <= 5 or tokens.shape[2] != 1280:
                raise ValueError(f"Unexpected Virchow2 output shape: {tuple(tokens.shape)}")
            class_token = tokens[:, 0]
            patch_tokens = tokens[:, 5:]  # tokens 1-4 are register tokens
            embeddings = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
        return embeddings.float().cpu().numpy().astype(np.float32, copy=False)

    return encode
