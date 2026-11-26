"""Save native reconstructions, image previews, and reference metrics."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import conv2d


def save_image(image: torch.Tensor, path: Path) -> None:
    values = image.detach()
    if values.is_complex():
        values = values.abs()
        values = values / values.amax().clamp_min(1e-8)
    plane = values[0, 0].clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(plane).save(path)


def save_array(image: torch.Tensor, path: Path) -> None:
    np.save(path, image.detach()[0, 0].cpu().numpy(), allow_pickle=False)


def reference_metrics(reconstruction: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    if reference.is_complex():
        reconstruction, reference = reconstruction.abs(), reference.abs()
        reconstruction = reconstruction / reconstruction.amax().clamp_min(1e-8)
        reference = reference / reference.amax().clamp_min(1e-8)
    reconstruction, reference = reconstruction.clamp(0, 1), reference.clamp(0, 1)
    mse = (reconstruction - reference).square().mean()
    psnr = (-10 * mse.log10()).item()

    axis = torch.arange(-5, 6, device=reference.device, dtype=reference.dtype)
    axis = torch.exp(-axis.square() / (2 * 1.5 ** 2))
    axis = axis / axis.sum()
    window = (axis[:, None] * axis[None, :])[None, None]
    mean_x = conv2d(reconstruction, window)
    mean_y = conv2d(reference, window)
    var_x = conv2d(reconstruction.square(), window) - mean_x.square()
    var_y = conv2d(reference.square(), window) - mean_y.square()
    cov = conv2d(reconstruction * reference, window) - mean_x * mean_y
    ssim = ((2 * mean_x * mean_y + 0.01 ** 2) * (2 * cov + 0.03 ** 2)) / (
        (mean_x.square() + mean_y.square() + 0.01 ** 2) * (var_x + var_y + 0.03 ** 2)
    )
    return psnr, ssim.mean().item()
