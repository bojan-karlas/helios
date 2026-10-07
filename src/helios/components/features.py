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
from functools import cache
from pathlib import Path
from typing import Any, Literal, get_args

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
    "plip",
    "resnet-50",
]

#: Runtime tuple of the supported model names (for selector validation).
FEATURE_MODELS: tuple[str, ...] = get_args(FeatureModel)

IMPLEMENTED_FEATURE_MODELS: tuple[str, ...] = FEATURE_MODELS

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
    if tiles.ndim != 4 or tiles.shape[-1] != 3:
        raise ValueError("tiles must have shape (N, H, W, 3)")
    encoder = _load_encoder(model, device=device)
    n = int(len(tiles))
    n_batches = (n + batch_size - 1) // batch_size if batch_size > 0 else 0
    chunks: list[NDArray[np.float32]] = []
    for start in progress.task(
        range(0, n, batch_size), total=n_batches, desc=f"{model} embed"
    ):
        batch = np.asarray(tiles[start : start + batch_size], dtype=np.uint8)
        encoded = np.asarray(encoder(batch), dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] != len(batch):
            raise ValueError(
                f"Encoder {model!r} returned shape {encoded.shape}; expected "
                f"({len(batch)}, embedding_dim)."
            )
        chunks.append(encoded)
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


@cache
def _load_encoder(model: FeatureModel, *, device: str) -> Encoder:
    """Load the encoder callable for ``model`` (weights loaded here, not in the stage)."""
    loaders = {
        "virchow2": _load_virchow2,
        "uni": _load_uni,
        "uni2": _load_uni2,
        "conch": _load_conch,
        "conch1.5": _load_conch15,
        "chief": _load_chief,
        "ctranspath": _load_ctranspath,
        "dino": _load_dino,
        "gigapath": _load_gigapath,
        "musk": _load_musk,
        "phikon2": _load_phikon2,
        "plip": _load_plip,
        "resnet-50": _load_resnet50,
    }
    return loaders[model](device=device)


def _resolved_device(device: str):
    import torch

    resolved = torch.device(device)
    return torch.device("cpu") if resolved.type == "cuda" and not torch.cuda.is_available() else resolved


def _image_transform(*, resize: int, crop: int, mean: tuple[float, ...], std: tuple[float, ...]):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _torch_encoder(
    backbone: Any,
    transform: Callable[[Any], Any],
    *,
    device: str,
    embedding_dim: int,
    forward: Callable[[Any, Any], Any] | None = None,
) -> Encoder:
    """Adapt a torch model + PIL transform to the component's NumPy contract."""
    import torch
    from PIL import Image

    resolved = _resolved_device(device)
    backbone = backbone.eval().to(resolved)

    def encode(batch: NDArray[np.uint8]) -> NDArray[np.float32]:
        if len(batch) == 0:
            return np.empty((0, embedding_dim), dtype=np.float32)
        inputs = torch.stack([transform(Image.fromarray(tile, mode="RGB")) for tile in batch]).to(resolved)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if resolved.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            output = forward(backbone, inputs) if forward is not None else backbone(inputs)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError("foundation encoder did not return a tensor")
        if output.ndim != 2 or output.shape[1] != embedding_dim:
            raise ValueError(
                f"foundation encoder returned shape {tuple(output.shape)}; "
                f"expected (N, {embedding_dim})"
            )
        return output.float().cpu().numpy().astype(np.float32, copy=False)

    return encode


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_plip(*, device: str) -> Encoder:
    from transformers import AutoModelForZeroShotImageClassification

    model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")
    transform = _image_transform(resize=224, crop=224, mean=_OPENAI_MEAN, std=_OPENAI_STD)

    def forward(backbone, inputs):
        import torch

        output = backbone.get_image_features(pixel_values=inputs)
        return output / torch.linalg.vector_norm(output, dim=-1, keepdim=True).clamp_min(1e-12)

    return _torch_encoder(model, transform, device=device, embedding_dim=512, forward=forward)


def _load_dino(*, device: str) -> Encoder:
    import torch
    from timm.models.vision_transformer import VisionTransformer

    url = (
        "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/"
        "pretrained-weights/dino_vit_small_patch8_ep200.torch"
    )
    model = VisionTransformer(img_size=224, patch_size=8, embed_dim=384, num_heads=6, num_classes=0)
    model.load_state_dict(torch.hub.load_state_dict_from_url(url, progress=True), strict=True)
    transform = _image_transform(resize=224, crop=224, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    return _torch_encoder(model, transform, device=device, embedding_dim=384)


def _load_uni(*, device: str) -> Encoder:
    import timm

    model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, num_classes=0)
    transform = _timm_transform(model)
    return _torch_encoder(model, transform, device=device, embedding_dim=1024)


def _load_uni2(*, device: str) -> Encoder:
    import timm
    import torch
    from timm.layers import SwiGLUPacked

    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    return _torch_encoder(model, _timm_transform(model), device=device, embedding_dim=1536)


def _load_conch(*, device: str) -> Encoder:
    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as exc:
        raise ImportError("CONCH requires the 'foundation-models' extra: uv sync --extra foundation-models") from exc

    model, transform = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/CONCH")
    return _torch_encoder(
        model,
        transform,
        device=device,
        embedding_dim=512,
        forward=lambda backbone, inputs: backbone.encode_image(inputs, proj_contrast=False, normalize=False),
    )


def _load_gigapath(*, device: str) -> Encoder:
    import timm

    model = timm.create_model("hf-hub:prov-gigapath/prov-gigapath", pretrained=True)
    transform = _image_transform(resize=256, crop=224, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    return _torch_encoder(model, transform, device=device, embedding_dim=1536)


def _load_phikon2(*, device: str) -> Encoder:
    from transformers import AutoModel

    model = AutoModel.from_pretrained("owkin/phikon-v2")
    transform = _image_transform(resize=224, crop=224, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    return _torch_encoder(
        model,
        transform,
        device=device,
        embedding_dim=1024,
        forward=lambda backbone, inputs: backbone(inputs).last_hidden_state[:, 0, :],
    )


def _load_musk(*, device: str) -> Encoder:
    import timm

    try:
        from musk import modeling  # noqa: F401
        from musk.utils import load_model_and_may_interpolate
    except ImportError as exc:
        raise ImportError("MUSK requires the 'foundation-models' extra: uv sync --extra foundation-models") from exc
    model = timm.create_model("musk_large_patch16_384")
    load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, "model|module", "")
    transform = _image_transform(resize=384, crop=384, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    return _torch_encoder(
        model,
        transform,
        device=device,
        embedding_dim=2048,
        forward=lambda backbone, inputs: backbone(
            image=inputs, with_head=False, out_norm=False, ms_aug=True, return_global=True
        )[0],
    )


def _load_resnet50(*, device: str) -> Encoder:
    import timm

    model = timm.create_model("resnet50", pretrained=True, num_classes=0)
    return _torch_encoder(model, _timm_transform(model), device=device, embedding_dim=2048)


def _timm_transform(model):
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    return create_transform(**resolve_data_config(model.pretrained_cfg, model=model))


def _load_ctranspath(*, device: str) -> Encoder:
    path = _download_google_drive_weights(
        file_id="1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX",
        filename="ctranspath/weights.pth",
        sha256="7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539",
    )
    return _load_ctranspath_checkpoint(path, device=device)


def _load_chief(*, device: str) -> Encoder:
    path = _download_google_drive_weights(
        file_id="1_vgRF1QXa8sPCOpJ1S9BihwZhXQMOVJc",
        filename="chief/CHIEF_CTransPath.pth",
        sha256="1646f23001214f74cf432ef0e80b808ee6605143802ae6ed53a87564ddc4924a",
    )
    return _load_ctranspath_checkpoint(path, device=device)


def _load_ctranspath_checkpoint(path: Path, *, device: str) -> Encoder:
    import torch

    model = _ctranspath_backbone()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    transform = _image_transform(resize=224, crop=224, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    return _torch_encoder(model, transform, device=device, embedding_dim=768)


def _ctranspath_backbone():
    """Build the CTransPath/CHIEF patch backbone using timm's Swin implementation."""
    import torch
    from timm.layers import to_2tuple
    from timm.models.swin_transformer import SwinTransformer

    class ConvStem(torch.nn.Module):
        def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 4,
            in_chans: int = 3,
            embed_dim: int = 96,
            norm_layer=None,
            flatten: bool = True,
            **_kwargs,
        ) -> None:
            super().__init__()
            patch_size_tuple = to_2tuple(patch_size)
            if patch_size_tuple != (4, 4):
                raise ValueError("CTransPath ConvStem requires patch_size=4")
            self.img_size = to_2tuple(img_size)
            self.patch_size = patch_size_tuple
            self.grid_size = (
                self.img_size[0] // self.patch_size[0],
                self.img_size[1] // self.patch_size[1],
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]
            self.flatten = flatten
            self.proj = torch.nn.Sequential(
                torch.nn.Conv2d(in_chans, embed_dim // 8, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(embed_dim // 8),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(embed_dim // 8, embed_dim // 4, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(embed_dim // 4),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(embed_dim // 4, embed_dim, 1),
            )
            self.norm = norm_layer(embed_dim) if norm_layer else torch.nn.Identity()

        def forward(self, x):
            x = self.proj(x)
            x = x.flatten(2).transpose(1, 2)
            return self.norm(x)

    return SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=0,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        embed_layer=ConvStem,
    )


def _download_google_drive_weights(*, file_id: str, filename: str, sha256: str) -> Path:
    import hashlib
    import os

    import gdown

    root = Path(os.environ.get("HELIOS_MODEL_DIR", Path.home() / ".cache" / "helios" / "models"))
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        result = gdown.download(id=file_id, output=str(path), quiet=False)
        if result is None:
            raise RuntimeError(f"Failed to download model weights to {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != sha256:
        raise ValueError(f"Checksum mismatch for {path}: expected {sha256}, got {digest}")
    return path


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


def _load_conch15(*, device: str) -> Encoder:
    """Load the official CONCH v1.5 patch encoder distributed with TITAN.

    ``MahmoodLab/TITAN`` is gated and uses Hugging Face remote model code.
    Access must be granted to the current Hugging Face account before use.
    The returned 768-dimensional rows are patch instances suitable for MIL.
    """
    import torch
    from PIL import Image
    from transformers import AutoModel

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        resolved_device = torch.device("cpu")

    titan = AutoModel.from_pretrained(
        "MahmoodLab/TITAN",
        trust_remote_code=True,
    )
    backbone, transform = titan.return_conch()
    backbone = backbone.eval().to(resolved_device)

    def encode(batch: NDArray[np.uint8]) -> NDArray[np.float32]:
        if batch.ndim != 4 or batch.shape[-1] != 3:
            raise ValueError("CONCH 1.5 input must have shape (N, H, W, 3)")
        if len(batch) == 0:
            return np.empty((0, 768), dtype=np.float32)
        inputs = torch.stack(
            [transform(Image.fromarray(tile, mode="RGB")) for tile in batch]
        ).to(resolved_device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if resolved_device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            embeddings = backbone(inputs)
        if isinstance(embeddings, (tuple, list)):
            embeddings = embeddings[0]
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("CONCH 1.5 encoder did not return a tensor")
        return embeddings.float().cpu().numpy().astype(np.float32, copy=False)

    return encode
