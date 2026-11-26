"""Reverse diffusion with BGDM, DDNM, and DPS measurement conditioning."""

import math
from collections.abc import Callable

import torch

from .degradation import Degradation
from .model import predict_noise
from .tasks import Task


def cumulative_alphas(total: int, device: torch.device) -> torch.Tensor:
    times = torch.linspace(0, 1, total + 1, dtype=torch.float64)
    curve = torch.cos((times + 0.008) / 1.008 * math.pi / 2).square()
    betas = (1 - curve[1:] / curve[:-1]).clamp(max=0.999).float()
    return torch.cat((torch.ones(1), (1 - betas).cumprod(0))).to(device)


def to_components(image: torch.Tensor) -> torch.Tensor:
    if not image.is_complex():
        return image
    return torch.cat((image.real, image.imag), dim=1)


def to_image(components: torch.Tensor, is_complex: bool) -> torch.Tensor:
    if not is_complex:
        return components
    return torch.complex(components[:, :1], components[:, 1:2])


def reconstruct(
    model: torch.nn.Module,
    operator: Degradation,
    measurement: torch.Tensor,
    task: Task,
    *,
    steps: int,
    method: str,
    eta: float,
    sigma_y: float,
    seed: int,
    progress: Callable[[int], None] | None = None,
) -> torch.Tensor:
    """Reconstruct a batch of prepared measurements in the operator's image units."""
    if not 1 <= steps <= task.diffusion_steps:
        raise ValueError(f"Steps must be between 1 and {task.diffusion_steps}.")
    if method not in {"bgdm", "ddnm", "dps"}:
        raise ValueError(f"Unsupported reconstruction method: {method}")
    if not 0 <= eta <= 1 or not math.isfinite(sigma_y) or sigma_y < 0:
        raise ValueError("Eta must be in [0, 1] and noise sigma must be finite and non-negative.")
    if task.name == "lidc" and method != "bgdm":
        raise ValueError("Sparse-view CT supports the BGDM method.")

    device = measurement.device
    generator = torch.Generator(device=device).manual_seed(seed)
    alphas = cumulative_alphas(task.diffusion_steps, device)
    stride = task.diffusion_steps // steps
    times = list(range((steps - 1) * stride, -1, -stride)) + [-1]
    shape = (measurement.shape[0], 2 if task.is_complex else 1, task.image_size, task.image_size)
    current = torch.randn(shape, device=device, generator=generator)
    requires_grad = method == "dps" or (method == "bgdm" and task.guidance > 0)

    for timestep, next_timestep in zip(times, times[1:]):
        alpha, next_alpha = alphas[timestep + 1], alphas[next_timestep + 1]
        with torch.set_grad_enabled(requires_grad):
            current = current.detach().requires_grad_(requires_grad)
            noise = predict_noise(model, current, timestep)
            clean = (current - (1 - alpha).sqrt() * noise) / alpha.sqrt()
            if requires_grad:
                physical = to_image(clean, task.is_complex)
                error = torch.linalg.vector_norm(operator.residual(physical, measurement))
                strength = 10.0 if method == "dps" else task.guidance
                correction = torch.autograd.grad(strength * error, current)[0]

        with torch.no_grad():
            clean = clean.detach()
            if method != "dps":
                guided = clean - correction if requires_grad else clean
                image = to_image(guided, task.is_complex)
                damping = task.damping if method == "bgdm" else 0.0
                projected = operator.project(image, measurement, damping=damping)
                if task.name == "lidc":
                    # The local CT sampler treats the projection as a fixed target.
                    difference = image - projected
                    norm = torch.linalg.vector_norm(difference.flatten(1), dim=1)
                    projected = projected - 5.0 * difference / norm.clamp_min(1e-12)[:, None, None, None]
                clean = to_components(projected)

            # Preserve the noise-dependent scaling of the local BGDM sampler.
            gamma = (1 - next_alpha.square() - (next_alpha * sigma_y).square()).clamp_min(0).sqrt()
            mixture = math.sqrt(1 - eta * eta) * noise
            if eta and next_timestep >= 0:
                stochastic = torch.randn(shape, device=device, generator=generator)
                mixture = mixture + eta * stochastic
            current = next_alpha.sqrt() * clean + gamma * (1 - next_alpha).sqrt() * mixture
            if method == "dps":
                current = current - correction
            if progress is not None:
                progress(1)

    result = to_image(current.detach(), task.is_complex)
    if not torch.isfinite(result).all():
        raise RuntimeError("Reconstruction contains non-finite values. Check input scaling or use FP32.")
    return result
