"""MultiStain-CycleGAN stain-normalization inference.

The compatible generator architecture is adapted from
DBO-DKFZ/multistain_cyclegan_normalization, itself based on CycleGAN/pix2pix.
Users train upstream and provide a ``latest_net_G_A.pth`` state dict to HELIOS.

Copyright (c) 2017 Jun-Yan Zhu and Taesung Park. All rights reserved.
Redistributed under the BSD terms in ``LICENSES/multistain-cyclegan.txt``.
"""
from __future__ import annotations

from functools import cache, partial
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray


def augment_stains(
    tiles: NDArray[np.uint8],
    *,
    checkpoint: str | Path,
    batch_size: int = 16,
    device: str = "cuda",
) -> NDArray[np.uint8]:
    """Normalize one slide's tiles with a user-trained MultiStain-CycleGAN.

    Upstream inference consumes 256-pixel RGB images normalized to ``[-1, 1]``.
    Results are resized back to the input dimensions so count, shape, and row
    order stay aligned with downstream MIL artifacts.
    """
    from torch.nn import functional

    array = np.asarray(tiles)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("tiles must have shape (N, H, W, 3)")
    if array.dtype != np.uint8:
        raise ValueError("tiles must have dtype uint8")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    generator, resolved_device = _load_generator(str(Path(checkpoint).expanduser().resolve()), device)
    height, width = int(array.shape[1]), int(array.shape[2])
    chunks: list[NDArray[np.uint8]] = []
    for start in range(0, len(array), batch_size):
        batch = torch.from_numpy(array[start : start + batch_size].copy()).permute(0, 3, 1, 2).float()
        batch = functional.interpolate(batch, size=(256, 256), mode="bilinear", align_corners=False)
        batch = batch.to(resolved_device).div_(127.5).sub_(1.0)
        with torch.inference_mode():
            output = generator(batch)
            if output.shape[-2:] != (height, width):
                output = functional.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
            output = output.add(1.0).mul(127.5).round().clamp(0, 255)
        chunks.append(output.byte().permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.empty_like(array)


@cache
def _load_generator(checkpoint: str, device: str):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"MultiStain-CycleGAN checkpoint not found: {path}")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        resolved = torch.device("cpu")
    model = UnetGenerator(
        3,
        3,
        num_downs=8,
        ngf=64,
        norm_layer=partial(torch.nn.InstanceNorm2d, affine=False, track_running_stats=False),
        use_dropout=False,
    )
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    state = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported CycleGAN checkpoint payload in {path}")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Checkpoint {path} is incompatible with the MultiStain-CycleGAN "
            "8-level U-Net G_A generator; supply latest_net_G_A.pth from upstream training."
        ) from exc
    return model.eval().to(resolved), resolved


class UnetGenerator(torch.nn.Module):
    """Upstream MultiStain-CycleGAN U-Net generator (inference architecture)."""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=None, use_dropout=False):
        super().__init__()
        norm_layer = norm_layer or torch.nn.BatchNorm2d
        block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        for _ in range(num_downs - 5):
            block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, submodule=block, norm_layer=norm_layer, use_dropout=use_dropout
            )
        block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=block, outermost=True, norm_layer=norm_layer
        )

    def forward(self, inputs):
        return self.model(inputs)


class UnetSkipConnectionBlock(torch.nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=None,
        use_dropout=False,
    ):
        super().__init__()
        norm_layer = norm_layer or torch.nn.BatchNorm2d
        input_nc = outer_nc if input_nc is None else input_nc
        use_bias = isinstance(norm_layer, partial) and norm_layer.func == torch.nn.InstanceNorm2d
        downconv = torch.nn.Conv2d(input_nc, inner_nc, 4, 2, 1, bias=use_bias)
        downrelu = torch.nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = torch.nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        self.outermost = outermost
        if outermost:
            upconv = torch.nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1)
            layers = [downconv, submodule, uprelu, upconv, torch.nn.Tanh()]
        elif innermost:
            upconv = torch.nn.ConvTranspose2d(inner_nc, outer_nc, 4, 2, 1, bias=use_bias)
            layers = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = torch.nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1, bias=use_bias)
            layers = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if use_dropout:
                layers.append(torch.nn.Dropout(0.5))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, inputs):
        output = self.model(inputs)
        return output if self.outermost else torch.cat([inputs, output], dim=1)
